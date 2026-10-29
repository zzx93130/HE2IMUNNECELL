"""TLS states (Part B): Visium functional validation of H&E composition and of the frozen ESCA state model.

Objects: every pathologist TLS object of the 8 USZ Visium sections = connected group of ETLS/MTLS/GC spots
(visium_eval.tls_objects, the same 108 objects as plan item 5a). Cells: the v1 segmentation
(cells_0.25um, 0.79 µm/px TIFFs upsampled to 0.25 µm/px) with the adopted classifier, fine-tuned + tau 0.5
prior (visium_eval.sample_inputs 'finetuned_tau05').

Composition vs expression (primary, same footprint as the RNA): cells whose centre lies inside the 55 µm
capture disc (radius 27.5 µm) of each of the object's spots, pooled over the object. Expression: the h5ad X
of h5ad_preprocessed/<sample>.h5ad holds raw integer UMI counts (identical to filtered_feature_bc_matrix;
it is NOT log-normalised), so counts are summed over the object's spots and log1p(CP10k) is computed per
object; marker scores are means of log1p(CP10k), program scores means of per-gene z-scores across objects.

State: the frozen ESCA model (esca_tls_states.assign_states) applied to features measured on the object's
aggregate as in visium_maturity: the frozen stage-1 nuclear-density proposal(s) overlapping the object's
spot tiles (55 µm-radius discs), clipped to the tiles + 50 µm; objects without one fall back to the tiles.
Sensitivity: features measured on the spot tiles themselves. Nothing is fitted on Visium.

Statistics: Spearman per object pooled over sections with section-cluster bootstrap CIs (2000 resamples,
8 sections), the median of within-section Spearman, and the same at spot level for spots inside TLS objects.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
from scipy import ndimage, sparse, stats
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "visium_tls"))
import esca_tls_states as S  # noqa: E402  (sets the ESCA classifier before esca_analysis is imported)
import esca_analysis as A  # noqa: E402
import visium_cells as V  # noqa: E402
import visium_eval as E  # noqa: E402

CLASSIFIER = "finetuned_tau05"
SPOT_RADIUS_UM, TILE_UM, NEIGHBOURHOOD_UM = 27.5, 55.0, 50.0
MARKERS = {"B_genes": ["MS4A1", "CD79A", "CD19"], "T_genes": ["CD3E", "CD3D", "CD2"], "Macrophage_genes": ["CD68", "CD163", "C1QA"]}
COMPARE = [("B_share", "B_genes"), ("T_share", "T_genes"), ("Macrophage_share", "Macrophage_genes"),
           ("B_per_mm2", "B_genes"), ("T_per_mm2", "T_genes"), ("Macrophage_per_mm2", "Macrophage_genes"), ("nuclei_per_mm2", "B_genes")]
PROGRAMS = {**{k: v for k, v in S.PROGRAMS.items() if k not in ("B_cell", "T_cell", "macrophage")}, "chemokine12": S.CHEMOKINE12}


def disc_mask(spots, mpp, shape, radius_um):
    mask = np.zeros(shape, bool)
    gy, gx = np.mgrid[:shape[0], :shape[1]]
    r = radius_um / A.GRID_UM
    for s in spots.itertuples():
        cy, cx = s.y * mpp / A.GRID_UM, s.x * mpp / A.GRID_UM
        y0, y1 = int(max(cy - r - 1, 0)), int(min(cy + r + 2, shape[0]))
        x0, x1 = int(max(cx - r - 1, 0)), int(min(cx + r + 2, shape[1]))
        mask[y0:y1, x0:x1] |= (gy[y0:y1, x0:x1] + .5 - cy) ** 2 + (gx[y0:y1, x0:x1] + .5 - cx) ** 2 <= r * r
    return mask


def counts_matrix(sample):
    f = h5py.File(V.DATA / "h5ad_preprocessed" / f"{sample}.h5ad", "r")
    X = f["X"]
    matrix = X[:] if isinstance(X, h5py.Dataset) else sparse.csr_matrix((X["data"][:], X["indices"][:], X["indptr"][:])).toarray()
    assert np.allclose(matrix, np.round(matrix)), "h5ad X is not raw counts"
    genes = [g.decode() for g in f["var"]["gene_name"][:]]
    index = [b.decode() for b in f["obs"]["_index"][:]]
    return pd.DataFrame(matrix, index=index, columns=genes).T.groupby(level=0).sum().T  # duplicate symbols summed


def hooknet_gc(sample, shape, mpp):
    """HookNet GC-class pixels on the 8 µm grid (descriptive), or None."""
    folder = V.OUT / "hooknet" / sample
    if not (folder / "maps_4um.npz").exists():
        return None
    hk = json.loads((folder / "summary.json").read_text())
    ratio = hk["source_dimensions"][0] / hk["dimensions_at_0p5um"][0]
    pred = np.load(folder / "maps_4um.npz")["prediction_mask"]
    ys = np.clip(((np.arange(shape[0]) + .5) * A.GRID_UM / mpp / (8 * ratio)).astype(int), 0, pred.shape[0] - 1)
    xs = np.clip(((np.arange(shape[1]) + .5) * A.GRID_UM / mpp / (8 * ratio)).astype(int), 0, pred.shape[1] - 1)
    return pred[np.ix_(ys, xs)] == 3


def section(sample):
    x = E.sample_inputs(sample, CLASSIFIER)
    shape, mpp, cells = x["shape"], x["mpp"], x["cells"]
    gc = hooknet_gc(sample, shape, mpp)
    x.update({"gc": gc if gc is not None else np.zeros(shape, bool), "reference": np.zeros(shape, bool),
              "cells_xy_um": cells[["x_native", "y_native"]].to_numpy() * mpp,
              "celltype": cells[f"celltype_{CLASSIFIER}"].to_numpy()})
    proposals, _ = A.proposal_features(x, **S.STAGE1)
    A._PROPOSALS.clear()
    if gc is None:
        x["gc"] = None
    spots = V.spots(sample)
    part = spots[spots.in_tissue == 1].reset_index(drop=True)
    groups = E.tls_objects(part, 100 / mpp)
    counts = counts_matrix(sample)
    tree = cKDTree(cells[["x_native", "y_native"]].to_numpy())
    types = cells[f"celltype_{CLASSIFIER}"].to_numpy()
    spot_rows, object_rows, object_counts = [], [], {}
    for g in sorted(groups.unique()):
        members = part.loc[groups.index[groups == g]]
        labels = members.label.value_counts()
        tiles = disc_mask(members, mpp, shape, TILE_UM)
        near = ndimage.distance_transform_edt(~tiles) * A.GRID_UM <= NEIGHBOURHOOD_UM
        overlapping = [p for p in proposals if (p["mask"] & tiles).sum() >= A.MATCH_FRACTION * min(p["mask"].sum(), tiles.sum())]
        aggregate = np.zeros(shape, bool)
        for p in overlapping:
            aggregate |= p["mask"]
        aggregate &= near
        source = "stage-1 proposal" if aggregate.sum() * A.GRID_UM ** 2 >= 2000 else "spot tiles"
        if source == "stage-1 proposal":  # keep the largest connected piece (state features describe one object)
            lab, n = ndimage.label(aggregate)
            aggregate = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1) if n > 1 else aggregate
        tile_lab, n = ndimage.label(tiles)
        tiles_main = tile_lab == (np.argmax(np.bincount(tile_lab.ravel())[1:]) + 1) if n > 1 else tiles
        tally = {t: 0 for t in S.CELL_TYPES}
        for s in members.itertuples():
            idx = tree.query_ball_point([s.x, s.y], SPOT_RADIUS_UM / mpp)
            spot_types = types[idx]
            c = {t: int((spot_types == t).sum()) for t in S.CELL_TYPES}
            for t in tally:
                tally[t] += c[t]
            n_all = sum(c.values())
            disc_mm2 = np.pi * SPOT_RADIUS_UM ** 2 / 1e6
            spot_rows.append({"sample": sample, "object": int(g), "barcode": s.barcode, "label": s.label, "cells": n_all,
                              "nuclei_per_mm2": n_all / disc_mm2, **{f"{t}_per_mm2": c[t] / disc_mm2 for t in S.CELL_TYPES},
                              **{f"{t}_cells": c[t] for t in S.CELL_TYPES},
                              **{f"{t}_share": c[t] / n_all if n_all else np.nan for t in S.CELL_TYPES}})
        n_all = sum(tally.values())
        base = {"sample": sample, "object": int(g), "spots": len(members),
                "labels": ",".join(f"{k}:{v}" for k, v in labels.items()),
                "mature": bool(labels.reindex(["MTLS", "GC"]).fillna(0).sum() > 0),
                "has_GC_spot": bool(labels.get("GC", 0) > 0), "aggregate_source": source,
                "accepted_two_stage": any(p["B_share"] >= S.FROZEN["b_share"] for p in overlapping),
                "disc_cells": n_all, "nuclei_per_mm2": n_all / (len(members) * np.pi * SPOT_RADIUS_UM ** 2 / 1e6),
                **{f"{t}_per_mm2": tally[t] / (len(members) * np.pi * SPOT_RADIUS_UM ** 2 / 1e6) for t in S.CELL_TYPES},
                **{f"{t}_share": tally[t] / n_all if n_all else np.nan for t in S.CELL_TYPES}}
        feats = S.object_features(aggregate if source == "stage-1 proposal" else tiles_main, x)
        tile_feats = S.object_features(tiles_main, x)
        object_rows.append({**base, **{f"agg_{k}": v for k, v in feats.items()}, **{f"tile_{k}": v for k, v in tile_feats.items()}})
        object_counts[(sample, int(g))] = counts.loc[members.barcode].sum(0)
    spot_counts = counts.loc[[r["barcode"] for r in spot_rows]]
    return object_rows, spot_rows, object_counts, spot_counts


def log_cp10k(frame):
    return np.log1p(frame.div(frame.sum(1).clip(lower=1), axis=0) * 1e4)


def scores(expr, z_ref=None):
    """Marker means of log1p(CP10k) and program means of per-gene z (z over the rows of z_ref, default expr)."""
    z_ref = expr if z_ref is None else z_ref
    out = pd.DataFrame(index=expr.index)
    for name, genes in MARKERS.items():
        out[name] = expr[[g for g in genes if g in expr]].mean(1)
    for name, genes in PROGRAMS.items():
        present = [g for g in genes if g in expr]
        z = (expr[present] - z_ref[present].mean()) / z_ref[present].std(ddof=1).replace(0, np.nan)
        out[name] = z.mean(1)
    return out


def rho_entry(frame, x, y):
    d = frame[["sample", x, y]].dropna()
    pooled = stats.spearmanr(d[x], d[y])
    within = [stats.spearmanr(g[x], g[y]).statistic for _, g in d.groupby("sample") if len(g) >= 5 and g[x].nunique() > 1]
    return {"x": x, "y": y, "n": len(d), "sections": int(d["sample"].nunique()), "rho": float(pooled.statistic),
            "p": float(pooled.pvalue), "CI95": S.cluster_bootstrap_ci(d, "sample", lambda s: stats.spearmanr(s[x], s[y]).statistic),
            "median_within_section_rho": float(np.nanmedian(within)) if within else np.nan,
            "within_section_rho": [round(float(v), 3) for v in within]}


def main():
    model = S.load_model()
    objects, spots, ocounts, scounts = [], [], {}, []
    for sample in V.SAMPLES:
        o, s, oc, sc = section(sample)
        objects += o
        spots += s
        ocounts.update(oc)
        scounts.append(sc)
        print(sample, len(o), "objects", len(s), "spots", flush=True)
    objects = pd.DataFrame(objects)
    spots = pd.DataFrame(spots)
    # Expression per object (summed counts) and per spot.
    oc = pd.DataFrame(ocounts).T
    oc.index = pd.MultiIndex.from_tuples(oc.index, names=["sample", "object"])
    obj_expr = log_cp10k(oc)
    obj_scores = scores(obj_expr).reset_index()
    obj_scores["umi"] = oc.sum(1).to_numpy()
    objects = objects.merge(obj_scores, on=["sample", "object"])
    sc = pd.concat(scounts)
    spot_scores = scores(log_cp10k(sc))
    spot_scores["umi"] = sc.sum(1).to_numpy()
    spots = pd.concat([spots.reset_index(drop=True), spot_scores.reset_index(drop=True)], axis=1)
    # Frozen ESCA state model on the aggregate features (primary) and the tile features (sensitivity).
    for prefix in ["agg", "tile"]:
        feats = objects[[f"{prefix}_{f}" for f in S.STATE_FEATURES]].rename(columns=lambda c: c[len(prefix) + 1:])
        for which in ["primary", "k3"]:
            a = S.assign_states(feats, which, model)
            tag = f"{prefix}_{which}"
            objects[f"state_{tag}"] = a.state_name.to_numpy()
            objects[f"pBfol_{tag}"] = a.p_B_follicular.to_numpy()
    S.TABLES.mkdir(parents=True, exist_ok=True)
    objects.to_csv(S.TABLES / "visium_tls_objects.csv", index=False)
    spots.to_csv(S.TABLES / "visium_tls_spots.csv", index=False)

    result = {"objects": len(objects), "sections": int(objects["sample"].nunique()), "spots_in_objects": len(spots),
              "classifier": "fine-tuned + tau 0.5 prior (finetuned_tau05), cells_0.25um",
              "expression_source": "h5ad_preprocessed X = raw UMI counts (verified integer, equal to filtered_feature_bc_matrix); "
                                   "summed per object, log1p(CP10k)",
              "median_umi_per_spot": float(spots.umi.median()), "median_umi_per_object": float(objects.umi.median()),
              "aggregate_source": objects.aggregate_source.value_counts().to_dict(),
              "mature_objects": int(objects.mature.sum())}
    comp = []
    for level, frame in [("object", objects), ("spot (within TLS objects)", spots)]:
        for share, genes in COMPARE:
            comp.append({"level": level, **rho_entry(frame, share, genes)})
    result["composition_vs_genes"] = comp
    earlier = json.loads((V.OUT / "analysis" / "results.json").read_text())["spot_RNA_check"]
    result["earlier_spot_level"] = [e for e in earlier if e["classifier"] == CLASSIFIER]
    # Programs vs H&E state (primary: aggregate features, frozen primary model).
    prog = []
    for tag in ["agg_primary", "agg_k3", "tile_primary"]:
        for p in [*PROGRAMS, "B_genes", "T_genes", "Macrophage_genes"]:
            entry = rho_entry(objects, f"pBfol_{tag}", p)
            groups = [g[p].dropna().to_numpy() for _, g in objects.groupby(f"state_{tag}") if g[p].notna().sum() >= 3]
            entry.update({"model": tag, "program": p,
                          "median_by_state": objects.groupby(f"state_{tag}")[p].median().to_dict(),
                          "n_by_state": objects[f"state_{tag}"].value_counts().to_dict(),
                          "kruskal_p": float(stats.kruskal(*groups).pvalue) if len(groups) >= 2 else np.nan})
            prog.append(entry)
    result["programs_vs_state"] = prog
    for p in PROGRAMS:  # programs vs continuous composition, for context
        for share in ["B_share", "T_share", "Macrophage_share"]:
            result.setdefault("programs_vs_composition", []).append({"program": p, **rho_entry(objects, share, p)})
    result["state_counts"] = {tag: objects[f"state_{tag}"].value_counts().to_dict() for tag in ["agg_primary", "agg_k3", "tile_primary"]}
    result["state_vs_pathologist_maturity"] = {tag: pd.crosstab(objects[f"state_{tag}"], objects.mature.map({True: "mature (MTLS/GC)", False: "early (ETLS)"})).to_dict()
                                               for tag in ["agg_primary", "agg_k3"]}
    result["AUROC_pBfol_for_mature"] = {tag: {"AUROC": float(A.auc(objects.mature, objects[f"pBfol_{tag}"])),
                                              "CI95": S.cluster_bootstrap_ci(objects, "sample", lambda s: A.auc(s.mature, s[f"pBfol_{tag}"]))}
                                        for tag in ["agg_primary", "agg_k3", "tile_primary"]}
    result["agg_vs_tile_state_agreement"] = float((objects.state_agg_primary == objects.state_tile_primary).mean())
    S.save_json(result, S.PARTS / "visium_states.json")
    print(json.dumps([{k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items() if k in ("level", "x", "y", "rho", "CI95", "median_within_section_rho", "n")}
                      for c in comp], indent=0, default=str))


if __name__ == "__main__":
    main()
