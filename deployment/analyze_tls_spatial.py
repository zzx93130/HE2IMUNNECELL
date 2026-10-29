"""Exploratory spatial screening for TLS-like B/T aggregates; no TLS classifier."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime" / "matplotlib"))
RUN = ROOT / "outputs" / "whole-slide-20261012"
OUT = RUN / "tls_analysis"


def main():
    import numpy as np
    import pandas as pd
    from scipy.spatial import cKDTree
    from scipy.ndimage import gaussian_filter, maximum_filter
    started = time.perf_counter()
    manifest = json.loads((RUN / "run.json").read_text())
    original = pd.read_csv(manifest["slides"][0]["results"])
    mpp = .252
    positions = original[["centroid_x_global", "centroid_y_global"]].to_numpy()
    near = cKDTree(positions).query_pairs(2, output_type="ndarray")
    near = near[original.patch_id.to_numpy()[near[:, 0]] != original.patch_id.to_numpy()[near[:, 1]]]
    boxes = original[["bbox_xmin_global", "bbox_ymin_global", "bbox_xmax_global_exclusive", "bbox_ymax_global_exclusive"]].to_numpy()
    intersection = np.maximum(0, np.minimum(boxes[near[:, 0], 2:], boxes[near[:, 1], 2:]) - np.maximum(boxes[near[:, 0], :2], boxes[near[:, 1], :2])).prod(1)
    area = (boxes[:, 2:] - boxes[:, :2]).prod(1)
    iou = intersection / (area[near[:, 0]] + area[near[:, 1]] - intersection)
    near = near[iou >= .5]
    parent = np.arange(len(original))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in near:
        aa, bb = find(a), find(b)
        if aa != bb:
            parent[bb] = aa
    groups = {}
    for i in np.unique(near):
        groups.setdefault(find(i), []).append(int(i))
    excluded = []
    audit = []
    for members in groups.values():
        keep = max(members, key=lambda i: (original.confidence.iat[i], -original.global_instance_id.iat[i]))
        for i in members:
            if i != keep:
                excluded.append(i)
                audit.append({"excluded_id": int(original.global_instance_id.iat[i]), "kept_id": int(original.global_instance_id.iat[keep]),
                              "excluded_type": original.celltype.iat[i], "kept_type": original.celltype.iat[keep]})
    pd.DataFrame(audit).to_csv(OUT / "possible_overlap_repeat_audit.csv", index=False)
    cells = original.drop(index=excluded).reset_index(drop=True)
    flips = pd.read_csv(OUT / "all_predicted_BT_flip_checks.csv")
    stable_ids = set(flips.loc[flips.stable_both_flips, "global_instance_id"])
    cells["stable_both_flips"] = cells.global_instance_id.isin(stable_ids)
    xy = cells[["centroid_x_global", "centroid_y_global"]].to_numpy() * mpp
    cells["x_um"], cells["y_um"] = xy[:, 0], xy[:, 1]
    cells[["global_instance_id", "celltype", "confidence", "prob_B", "prob_T", "x_um", "y_um", "stable_both_flips"]].to_csv(OUT / "spatial_cells.csv", index=False)
    b = cells.celltype.eq("B").to_numpy()
    t = cells.celltype.eq("T").to_numpy()
    lymph = b | t
    tree = cKDTree(xy)
    all_tree, b_tree, t_tree = tree, cKDTree(xy[b]), cKDTree(xy[t])
    lymph_xy = xy[lymph]
    lymph_b = b[lymph]
    n_b = int(b.sum())
    rng = np.random.default_rng(569)
    observed_bb = b_tree.query(xy[b], k=2)[0][:, 1]
    observed_bt = t_tree.query(xy[b])[0]
    radii = [25, 50, 100, 150]
    neighborhood = []
    lymph_tree = cKDTree(lymph_xy)
    for radius in radii:
        b_neighbors = b_tree.query_ball_point(xy[b], radius, return_length=True) - 1
        l_neighbors = lymph_tree.query_ball_point(xy[b], radius, return_length=True) - 1
        valid = l_neighbors > 0
        neighborhood.append({"radius_um": radius, "mean_B_fraction_among_lymphocyte_neighbors": float((b_neighbors[valid] / l_neighbors[valid]).mean()),
                             "slide_B_fraction_among_predicted_lymphocytes": float(b.sum() / lymph.sum())})
    pd.DataFrame(neighborhood).to_csv(OUT / "B_neighbor_composition.csv", index=False)
    blocks = np.floor(lymph_xy / 250).astype(int)
    _, block_ids = np.unique(blocks, axis=0, return_inverse=True)
    block_groups = [np.flatnonzero(block_ids == i) for i in range(block_ids.max()+1)]
    null_rows = []
    for permutation in range(199):
        global_points = lymph_xy[rng.choice(len(lymph_xy), n_b, replace=False)]
        global_nn = cKDTree(global_points).query(global_points, k=2)[0][:, 1]
        local_labels = lymph_b.copy()
        for group in block_groups:
            local_labels[group] = rng.permutation(local_labels[group])
        local_points = lymph_xy[local_labels]
        local_nn = cKDTree(local_points).query(local_points, k=2)[0][:, 1]
        null_rows.append({"permutation": permutation, "global_random_label_median_BB_um": float(np.median(global_nn)),
                          "local_250um_block_label_median_BB_um": float(np.median(local_nn))})
    null = pd.DataFrame(null_rows)
    null.to_csv(OUT / "random_label_nulls.csv", index=False)
    observed_median = float(np.median(observed_bb))
    null_stats = {}
    for column in null.columns[1:]:
        null_stats[column] = {"mean": float(null[column].mean()), "p025": float(null[column].quantile(.025)),
                              "p975": float(null[column].quantile(.975)),
                              "one_sided_empirical_p_for_shorter_spacing": float((1 + (null[column] <= observed_median).sum()) / (len(null)+1))}
    print("B-neighbor spatial null tests complete", flush=True)

    cell_size = 25
    edges = [np.arange(0, 29736*mpp+cell_size, cell_size), np.arange(0, 28294*mpp+cell_size, cell_size)]
    def histogram(select, weights=None):
        return np.histogram2d(xy[select,0], xy[select,1], bins=edges, weights=weights)[0]
    base_hist = histogram(b)
    all_hist = histogram(np.ones(len(cells), dtype=bool))
    def peaks(hist, sigma_um):
        field = gaussian_filter(hist, sigma_um/cell_size)
        minimum = max(.02, .2 * hist.sum() / base_hist.sum())
        pos = np.argwhere((field == maximum_filter(field, size=17)) & (field > minimum))
        pos = pos[np.argsort(field[tuple(pos.T)])[::-1]]
        return field, (pos+.5)*cell_size
    field, centers = peaks(base_hist, 100)
    _, centers_50 = peaks(base_hist, 50)
    _, centers_150 = peaks(base_hist, 150)
    confidence_centers = {}
    for cutoff in [.5,.7,.9]:
        _, confidence_centers[cutoff] = peaks(histogram(b & (cells.confidence.to_numpy() >= cutoff)), 100)
    stable_field, stable_centers = peaks(histogram(b & cells.stable_both_flips.to_numpy()), 100)
    weighted_field, weighted_centers = peaks(histogram(np.ones(len(cells), dtype=bool), cells.prob_B.to_numpy()), 100)
    candidates = []
    background_b_fraction = float(b.mean())
    background_l_fraction = float(lymph.mean())
    for index,center in enumerate(centers):
        region_indices = tree.query_ball_point(center, 150)
        inner = tree.query_ball_point(center, 75)
        outer = list(set(tree.query_ball_point(center, 250)) - set(inner))
        region, core, annulus = cells.iloc[region_indices], cells.iloc[inner], cells.iloc[outer]
        region_b, region_t = int(region.celltype.eq("B").sum()), int(region.celltype.eq("T").sum())
        core_b, core_t = int(core.celltype.eq("B").sum()), int(core.celltype.eq("T").sum())
        ann_b, ann_t = int(annulus.celltype.eq("B").sum()), int(annulus.celltype.eq("T").sum())
        fraction_l = (region_b+region_t)/len(region)
        core_fraction = core_b/max(1,core_b+core_t)
        ann_fraction = ann_b/max(1,ann_b+ann_t)
        row = {"peak_index": index+1, "x_um": center[0], "y_um": center[1],
               "x_level0_px": round(center[0]/mpp), "y_level0_px": round(center[1]/mpp),
               "B_150um":region_b, "T_150um":region_t, "all_cells_150um":len(region),
               "lymphocyte_fraction_150um":fraction_l, "B_fraction_all_cells_150um":region_b/len(region),
               "B_enrichment_vs_slide":(region_b/len(region))/background_b_fraction,
               "lymphocyte_enrichment_vs_slide":fraction_l/background_l_fraction,
               "B_fraction_among_BT_core_75um":core_fraction, "B_fraction_among_BT_annulus_75_250um":ann_fraction,
               "core_annulus_B_odds_ratio": ((core_b+.5)/(core_t+.5))/((ann_b+.5)/(ann_t+.5)),
               "B_median_confidence":float(region.loc[region.celltype.eq("B"),"confidence"].median()),
               "T_median_confidence":float(region.loc[region.celltype.eq("T"),"confidence"].median()),
               "B_stable_both_flips":int((region.celltype.eq("B") & region.stable_both_flips).sum()),
               "T_stable_both_flips":int((region.celltype.eq("T") & region.stable_both_flips).sum()),
               "mixed_aggregate_screen": bool(region_b>=30 and region_t>=100 and fraction_l>=.30),
               "near_peak_at_sigma50":bool(len(centers_50) and cKDTree(centers_50).query(center)[0]<=150),
               "near_peak_at_sigma150":bool(len(centers_150) and cKDTree(centers_150).query(center)[0]<=150),
               "near_peak_using_flip_stable_B":bool(len(stable_centers) and cKDTree(stable_centers).query(center)[0]<=150),
               "near_peak_using_probability_mass":bool(len(weighted_centers) and cKDTree(weighted_centers).query(center)[0]<=150)}
        for cutoff in [.5,.7,.9]:
            confident = region[region.confidence >= cutoff]
            row[f"B_conf_ge_{cutoff}"] = int(confident.celltype.eq("B").sum())
            row[f"T_conf_ge_{cutoff}"] = int(confident.celltype.eq("T").sum())
            peaks_at_cutoff = confidence_centers[cutoff]
            row[f"near_peak_conf_ge_{cutoff}"] = bool(len(peaks_at_cutoff) and cKDTree(peaks_at_cutoff).query(center)[0]<=150)
        candidates.append(row)
    candidates = pd.DataFrame(candidates)
    qualified = candidates[candidates.mixed_aggregate_screen].sort_values("B_150um", ascending=False).copy()
    qualified["candidate_id"] = [f"A{i+1}" for i in range(len(qualified))]
    ids = qualified.set_index("peak_index").candidate_id.to_dict()
    candidates["candidate_id"] = candidates.peak_index.map(ids).fillna("")
    candidates.to_csv(OUT / "all_B_density_peaks.csv", index=False)
    qualified.to_csv(OUT / "ranked_candidate_regions.csv", index=False)
    threshold_rows = []
    for minimum_b in [20,30,50]:
        for minimum_fraction in [.25,.30,.40]:
            threshold_rows.append({"minimum_B":minimum_b,"minimum_T":100,"minimum_BT_fraction":minimum_fraction,
                                   "regions_passing":int(((candidates.B_150um>=minimum_b)&(candidates.T_150um>=100)&(candidates.lymphocyte_fraction_150um>=minimum_fraction)).sum())})
    pd.DataFrame(threshold_rows).to_csv(OUT / "heuristic_threshold_sensitivity.csv", index=False)
    np.savez_compressed(OUT / "density_fields.npz", B=field, T=gaussian_filter(histogram(t),4),
                        all=gaussian_filter(all_hist,4), B_stable=stable_field,
                        B_probability_mass=weighted_field, x_edges=edges[0],y_edges=edges[1])
    summary = {"scope":"Exploratory single-slide TLS-like aggregate screening; not a trained or validated TLS detector",
               "source_csv":manifest["slides"][0]["results"],"slide":manifest["slides"][0]["slide"], "microns_per_pixel":mpp,
               "original_cells":len(original), "analysis_cells":len(cells), "possible_repeat_detections_excluded":len(excluded),
               "dedup_rule":"Different source patches, centroids <=2 level-0 pixels apart, bbox IoU >=0.5; keep highest-confidence member. Original predictions untouched.",
               "analysis_counts":cells.celltype.value_counts().to_dict(),
               "observed_median_BB_nearest_um":observed_median,"observed_median_BT_nearest_um":float(np.median(observed_bt)),
               "random_label_nulls":null_stats, "permutations":199,"seed":569,
               "null_interpretation":"Permute B versus T marks at fixed predicted lymphocyte coordinates; local null preserves B/T counts within 250um blocks. Tests predicted-label spatial structure, not TLS presence or biological cell accuracy.",
               "B_density_peaks_sigma100":len(candidates),"candidate_regions_passing_default_screen":len(qualified),
               "screen_parameters":{"grid_um":25,"Gaussian_sigma_um":100,"nonmax_filter_grid_size":17,"disk_radius_um":150,
                                    "minimum_B":30,"minimum_T":100,"minimum_predicted_BT_fraction":.30,
                                    "core_radius_um":75,"annulus_outer_radius_um":250,"peak_matching_radius_um":150},
               "BT_fraction_slide":background_l_fraction,"B_fraction_slide":background_b_fraction,
               "orientation_stable_fraction_original":{"B":float(flips.loc[flips.celltype.eq('B'),'stable_both_flips'].mean()),
                                                       "T":float(flips.loc[flips.celltype.eq('T'),'stable_both_flips'].mean())},
               "maturity_assessment":"Not established: no FDC, HEV, germinal-center or B-cell differentiation markers; predicted T cells are not subtyped.",
               "wall_seconds":time.perf_counter()-started}
    (OUT / "spatial_summary.json").write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    print(qualified[["candidate_id","x_um","y_um","B_150um","T_150um","lymphocyte_fraction_150um","B_enrichment_vs_slide","core_annulus_B_odds_ratio","B_stable_both_flips"]].to_string(index=False))


if __name__ == "__main__":
    main()
