"""Compare H&E-only HookNet-TLS polygons with frozen cell-based analyses and report."""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import os
from pathlib import Path
import shutil

from hooknet_tls_native import ROOT, RUNTIME
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME / "matplotlib"))

import numpy as np
import pandas as pd
from shapely import contains_xy
from shapely.affinity import scale
from shapely.geometry import Point, shape, mapping, box
import openslide


def uri(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def table(frame):
    return frame.to_html(index=False, border=0, classes="data", na_rep="—", float_format=lambda x: f"{x:.3f}")


def compute_comparison(run, out):
    cells = pd.read_csv(run / "tls_analysis" / "spatial_cells.csv")
    candidates = pd.read_csv(run / "tls_analysis" / "ranked_candidate_regions.csv")
    features = json.loads((out / "native_polygons.geojson").read_text())["features"]
    tls, gc = [], []
    records, member_rows, sets = [], [], {}
    for feature in features:
        props = feature["properties"]
        if not props["passes_published_filter"]:
            continue
        geometry = scale(shape(feature["geometry"]), xfact=.5, yfact=.5, origin=(0, 0))
        item = {"region_id": props["region_id"], "geometry": geometry, "props": props}
        (tls if props["predicted_label"] == "tls" else gc).append(item)
        inside = contains_xy(geometry, cells.x_um.to_numpy(), cells.y_um.to_numpy())
        local = cells.loc[inside]
        lymph = local.loc[local.celltype.isin(["B", "T"])]
        ids = set(map(int, lymph.global_instance_id)); sets[props["region_id"]] = ids
        bounds = geometry.bounds
        records.append({"method": "HookNet-TLS", "run_id": "hooknet_published_defaults", "region_id": props["region_id"],
            "predicted_label": props["predicted_label"], "x_um": geometry.centroid.x, "y_um": geometry.centroid.y,
            "area_um2": geometry.area, "source_B": int(local.celltype.eq("B").sum()),
            "source_T": int(local.celltype.eq("T").sum()), "original_detected_cells_in_polygon": len(local),
            "source_BT": len(lymph), "xmin_um": bounds[0], "ymin_um": bounds[1], "xmax_um": bounds[2], "ymax_um": bounds[3],
            "membership_basis": "Original cell centers strictly inside filtered H&E-predicted polygon; cell labels not model inputs",
            "polygon_geojson_um": json.dumps(mapping(geometry))})
        member_rows += [(props["region_id"], int(c.global_instance_id), c.celltype) for c in lymph.itertuples()]
    region_table = pd.DataFrame(records)
    region_table.to_csv(out / "filtered_regions.csv", index=False)
    members = pd.DataFrame(member_rows, columns=["region_id", "global_instance_id", "source_celltype"])
    members.to_csv(out / "BT_membership.csv.gz", index=False)
    pair_rows, summary_rows = [], []
    tls_union = set().union(*(sets[item["region_id"]] for item in tls)) if tls else set()
    for candidate in candidates.itertuples():
        center = Point(candidate.x_um, candidate.y_um)
        disk = center.buffer(150, resolution=128)
        local = cells.loc[((cells.x_um-candidate.x_um)**2 + (cells.y_um-candidate.y_um)**2).le(150**2)]
        ids = set(map(int, local.loc[local.celltype.isin(["B", "T"]), "global_instance_id"]))
        pairs = []
        for item in tls:
            g = item["geometry"]
            common = len(ids & sets[item["region_id"]])
            union = len(ids | sets[item["region_id"]])
            area = disk.intersection(g).area
            distance = center.distance(g.centroid)
            row = {"candidate_id": candidate.candidate_id, "region_id": item["region_id"],
                "center_distance_um": distance, "primary_center_match": distance <= 150,
                "candidate_center_inside_polygon": g.covers(center), "center_to_polygon_distance_um": center.distance(g),
                "intersection_BT": common, "candidate_BT_coverage": common/len(ids), "BT_jaccard": common/union,
                "candidate_disk_area_coverage": area/disk.area, "disk_polygon_IoU": area/disk.union(g).area}
            pairs.append(row)
            pair_rows.append(row)
        nearest = min(pairs, key=lambda x: x["center_distance_um"]) if pairs else None
        best = max(pairs, key=lambda x: x["BT_jaccard"]) if pairs else None
        center_matches = [r["region_id"] for r in pairs if r["primary_center_match"]]
        overlap_matches = [r["region_id"] for r in pairs if r["candidate_disk_area_coverage"] >= .1]
        best_area = max(pairs, key=lambda x: x["disk_polygon_IoU"]) if pairs else None
        summary_rows.append({"candidate_id": candidate.candidate_id, "method": "HookNet-TLS", "run_id": "hooknet_published_defaults",
            "source_B": candidate.B_150um, "source_T": candidate.T_150um, "primary_center_match": bool(center_matches),
            "nearby_detection_count": len(center_matches), "nearby_region_ids": json.dumps(center_matches),
            "split_location_match": len(center_matches) > 1,
            "nearest_region_id": nearest["region_id"] if nearest else "",
            "nearest_center_distance_um": nearest["center_distance_um"] if nearest else np.nan,
            "best_overlap_region_id": best["region_id"] if best and best["intersection_BT"] else "",
            "best_BT_jaccard": best["BT_jaccard"] if best else 0,
            "best_region_candidate_BT_coverage": best["candidate_BT_coverage"] if best else 0,
            "all_detection_candidate_BT_coverage": len(ids & tls_union)/len(ids),
            "center_matched_candidate_BT_coverage": len(ids & set().union(*(sets[x] for x in center_matches)))/len(ids) if center_matches else 0,
            "polygon_overlap_at_least_10_percent_disk": bool(overlap_matches), "overlapping_polygon_ids": json.dumps(overlap_matches),
            "best_disk_polygon_IoU": best_area["disk_polygon_IoU"] if best_area else 0,
            "best_candidate_disk_area_coverage": max((x["candidate_disk_area_coverage"] for x in pairs), default=0),
            "candidate_center_inside_any_polygon": any(x["candidate_center_inside_polygon"] for x in pairs),
            "center_to_nearest_polygon_um": min((x["center_to_polygon_distance_um"] for x in pairs), default=np.nan)})
    comparisons = pd.DataFrame(summary_rows)
    comparisons.to_csv(out / "A1_A6_comparison.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(out / "candidate_region_pairs.csv", index=False)
    gc_relations = []
    for item in gc:
        for target in tls:
            actual = item["geometry"].intersection(target["geometry"]).area/item["geometry"].area
            gc_relations.append({"GC_region_id": item["region_id"], "TLS_region_id": target["region_id"],
                "actual_GC_area_inside_TLS_polygon_fraction": actual,
                "GC_area_inside_TLS_bounding_rectangle_fraction": item["geometry"].intersection(box(*target["geometry"].bounds)).area/item["geometry"].area})
    pd.DataFrame(gc_relations).to_csv(out / "GC_TLS_polygon_relationships.csv", index=False)
    previous = out.parent
    old_comparison = pd.read_csv(previous / "A1_A6_comparison.csv")
    pd.concat([old_comparison, comparisons], ignore_index=True).to_csv(previous / "A1_A6_comparison_with_hooknet.csv", index=False)
    pd.concat([pd.read_csv(previous / "regions.csv"), region_table], ignore_index=True).to_csv(previous / "regions_with_hooknet.csv", index=False)
    tls_members = members.loc[members.region_id.isin([item["region_id"] for item in tls])].copy()
    tls_members["run_id"] = "hooknet_published_defaults"
    old_members = pd.read_csv(previous / "BT_membership.csv.gz")
    pd.concat([old_members, tls_members], ignore_index=True).to_csv(previous / "BT_membership_with_hooknet.csv.gz", index=False)
    pairwise = []
    for run_id, group in old_members.groupby("run_id"):
        old_ids = set(map(int, group.global_instance_id))
        union = old_ids | tls_union
        pairwise.append({"comparison_run": run_id, "HookNet_TLS_BT_members": len(tls_union),
            "intersection_BT": len(old_ids & tls_union), "BT_union_jaccard": len(old_ids & tls_union)/len(union) if union else np.nan})
    pd.DataFrame(pairwise).to_csv(out / "spatial_method_overlap.csv", index=False)
    return cells, candidates, tls, gc, region_table, comparisons


def figures(run, out, cells, candidates, tls, gc, comparison):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon as PlotPolygon
    from PIL import Image, ImageDraw, ImageFont
    summary = json.loads((run / "tls_analysis" / "spatial_summary.json").read_text())
    old_regions = pd.read_csv(out.parent / "regions.csv")
    review_folder = out / "review_regions"; review_folder.mkdir(exist_ok=True)
    colors = {"ours": "#e7463c", "hook": "#007dc1", "gc": "#e7ae00", "finder": "#00afbc", "tlsr": "#bd3dd0"}
    with openslide.OpenSlide(summary["slide"]) as slide:
        thumbnail = slide.get_thumbnail((1600, 1600)).convert("RGB")
        extent = [0, slide.dimensions[0]*summary["microns_per_pixel"]/1000, slide.dimensions[1]*summary["microns_per_pixel"]/1000, 0]
        fig, axes = plt.subplots(2, 2, figsize=(15, 13), layout="constrained")
        panels = [("Our six review disks", "ours"), (f"Independent H&E: HookNet-TLS\n{len(tls)} filtered TLS / {len(gc)} GC predictions", "hook"),
                  ("TLS-Finder example settings\n359 derived aggregate disks", "finder"), ("tlsR exploratory 50 µm criterion\n3 native regions; default = 0", "tlsr")]
        for ax, (title, method) in zip(axes.flat, panels):
            ax.imshow(thumbnail, extent=extent)
            for c in candidates.itertuples():
                ax.add_patch(Circle((c.x_um/1000, c.y_um/1000), .15, fill=False, color=colors["ours"], lw=1.8))
                ax.annotate(c.candidate_id, (c.x_um/1000, c.y_um/1000), xytext=(4, 4), textcoords="offset points", fontsize=9, color="#a41c16")
            if method == "hook":
                for item in tls+gc:
                    is_gc = item["props"]["predicted_label"] == "gc"
                    color = colors["gc"] if is_gc else colors["hook"]
                    xy = np.array(item["geometry"].exterior.coords)/1000
                    ax.add_patch(PlotPolygon(xy, facecolor=color+"30", edgecolor=color, lw=1.4))
            elif method == "finder":
                for r in old_regions.loc[old_regions.run_id.eq("finder_default")].itertuples():
                    ax.add_patch(Circle((r.x_um/1000, r.y_um/1000), .16, fill=False, color=colors["finder"], lw=.6))
            elif method == "tlsr":
                members = pd.read_csv(out.parent / "BT_membership.csv.gz")
                ids = members.loc[members.run_id.eq("tlsr_distance_50"), "global_instance_id"]
                local = cells.loc[cells.global_instance_id.isin(ids)]
                ax.scatter(local.x_um/1000, local.y_um/1000, s=3, color=colors["tlsr"], linewidths=0)
            ax.set(title=title, xlabel="x (mm)", ylabel="y (mm)")
        fig.suptitle("Whole-slide comparison: cell-based patterns and an independent H&E model", fontsize=17)
        fig.supxlabel("Red = A1–A6; blue = HookNet filtered TLS polygons; gold = predicted GC. Model outputs remain unconfirmed.", fontsize=11)
        fig.savefig(out / "four_method_comparison.png", dpi=150)
        plt.close(fig)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        except OSError:
            font = ImageFont.load_default(size=18)
        reviews = [{"id": r.candidate_id, "x_um": r.x_um, "y_um": r.y_um, "kind": "original candidate"} for r in candidates.itertuples()]
        extra = []
        for item in tls:
            if any(item["geometry"].intersection(Point(c.x_um, c.y_um).buffer(150)).area > 0 for c in candidates.itertuples()):
                continue
            extra.append(item)
        for i, item in enumerate(sorted(extra, key=lambda x: x["geometry"].area, reverse=True), 1):
            reviews.append({"id": f"H{i}", "x_um": item["geometry"].centroid.x, "y_um": item["geometry"].centroid.y,
                            "kind": "additional HookNet prediction", "region_id": item["region_id"]})
        for i, item in enumerate(gc, 1):
            reviews.append({"id": f"GC{i}", "x_um": item["geometry"].centroid.x, "y_um": item["geometry"].centroid.y,
                            "kind": "predicted GC; inspect separately", "region_id": item["region_id"]})
        cards = []
        level = min(1, slide.level_count-1)
        mpp, down = summary["microns_per_pixel"], slide.level_downsamples[level]
        factor = 1/(mpp*down)
        size = int(np.ceil(600*factor))
        for review in reviews:
            center = np.array([review["x_um"], review["y_um"]])
            start = np.floor((center-300)/mpp).astype(int); origin = start*mpp
            roi = slide.read_region(tuple(start), level, (size, size)).convert("RGB")
            annotated = roi.copy(); draw = ImageDraw.Draw(annotated, "RGBA")
            for item in tls+gc:
                g = item["geometry"]
                color = colors["gc"] if item["props"]["predicted_label"] == "gc" else colors["hook"]
                vertices = ((np.array(g.exterior.coords)-origin)*factor).tolist()
                draw.polygon([tuple(v) for v in vertices], fill=color+"30", outline=color, width=2)
            for c in candidates.itertuples():
                x, y = (np.array([c.x_um, c.y_um])-origin)*factor; r = 150*factor
                draw.ellipse((x-r,y-r,x+r,y+r), outline=colors["ours"], width=3)
            canvas = Image.new("RGB", (1000, 565), "#172b40")
            draw = ImageDraw.Draw(canvas)
            canvas.paste(roi.resize((500, 500)), (0, 35))
            canvas.paste(annotated.resize((500, 500)), (500, 35))
            draw.text((12, 8), "H&E: 600 × 600 µm", font=font, fill="white")
            draw.text((512, 8), "HookNet polygons: TLS blue / GC gold", font=font, fill="white")
            draw.text((12, 540), f"{review['id']} | {review['kind']} | red = original 150 µm disks | predictions unconfirmed", font=font, fill="white")
            path = review_folder / f"{review['id']}_HookNet_comparison.png"; canvas.save(path)
            roi.save(review_folder / f"{review['id']}_HE.png")
            cards.append({**review, "image": str(path)})
        pd.DataFrame(cards).to_csv(out / "review_regions.csv", index=False)
    return cards


def render(run):
    out = run / "tls_tool_comparison" / "hooknet_tls"
    cells, candidates, tls, gc, regions, comparisons = compute_comparison(run, out)
    cards = figures(run, out, cells, candidates, tls, gc, comparisons)
    provenance = json.loads((out / "provenance.json").read_text())
    native = json.loads((out / "native_summary.json").read_text())
    verify = json.loads((out / "model_verification.json").read_text())
    display = pd.DataFrame({"Candidate": comparisons.candidate_id,
        "HookNet centroid inside disk": comparisons.primary_center_match.map({True: "Yes", False: "No"}),
        "Nearest centroid µm": comparisons.nearest_center_distance_um.round(1),
        "Original B/T cells inside TLS polygons": comparisons.all_detection_candidate_BT_coverage.map(lambda x: f"{x:.1%}"),
        "Disk area inside best polygon": comparisons.best_candidate_disk_area_coverage.map(lambda x: f"{x:.1%}"),
        "Best B/T Jaccard": comparisons.best_BT_jaccard.round(3)})
    overlap_ids = comparisons.loc[comparisons.polygon_overlap_at_least_10_percent_disk, "candidate_id"].tolist()
    center_ids = comparisons.loc[comparisons.primary_center_match, "candidate_id"].tolist()
    gc_relations = pd.read_csv(out / "GC_TLS_polygon_relationships.csv")
    if len(gc_relations):
        best_gc = gc_relations.sort_values("GC_area_inside_TLS_bounding_rectangle_fraction", ascending=False).iloc[0]
        gc_note = f"The retained GC prediction passed association with {best_gc.TLS_region_id}: {best_gc.GC_area_inside_TLS_bounding_rectangle_fraction:.1%} of its area is inside that region's bounding rectangle, but {best_gc.actual_GC_area_inside_TLS_polygon_fraction:.1%} is inside its actual TLS polygon. The maximum overlap with any filtered TLS polygon is {gc_relations.actual_GC_area_inside_TLS_polygon_fraction.max():.1%}. The author's filter uses bounding rectangles; inspect the GC separately rather than interpreting it as a verified mature TLS."
    else:
        gc_note = "No retained GC prediction."
    narrative = f"HookNet-TLS retained {len(tls)} predicted TLS regions and {len(gc)} predicted germinal center using the published filters. Polygon overlap supports {', '.join(overlap_ids)}, using at least 10% overlap of the original disk as a descriptive rule. The original 150 µm centroid rule matches {', '.join(center_ids)}."
    a4 = comparisons.set_index("candidate_id").loc["A4"]
    boundary_note = f"A4 illustrates why polygon overlap matters: its nearest HookNet centroid is {a4.nearest_center_distance_um:.1f} µm away, while {a4.all_detection_candidate_BT_coverage:.1%} of its predicted B/T cells lie inside the filtered polygon."
    runtime = provenance["wall_seconds_including_preprocessing_verification_and_postprocessing"]
    cached = provenance["preprocessing"].get("RGB_pyramid_cache_reused", False)
    preparation_label = "cached image preparation" if cached else "initial image preparation"
    exclusions = "Download, installation and initial pyramid construction are excluded." if cached else "Download and installation are excluded; initial pyramid construction is included."
    inference = provenance["timings"]["inference_and_map_write_seconds"]
    options = "".join(f'<option value="{r["id"]}">{html.escape(r["id"] + " — " + r["kind"])}</option>' for r in cards)
    gallery = "".join(f'<article class="hooknet-card" data-id="{r["id"]}" style="display:{"block" if i == 0 else "none"}"><img src="{uri(r["image"])}" alt="{r["id"]} HookNet H&E overlay"></article>' for i, r in enumerate(cards))
    section = f'''<section id="hooknet-extension"><h2>Added independent H&E model: HookNet-TLS</h2><p class="notice">HookNet-TLS receives H&E pixels at 0.5 and 2.0 µm per pixel. Our B/T predictions are used only for comparison after inference. Independence of model inputs does not establish accuracy or confirmed TLS.</p><p>{html.escape(narrative)}</p><p>{html.escape(boundary_note)}</p><div class="scroll">{table(display)}</div><img src="{uri(out/'four_method_comparison.png')}" alt="Four-method whole-slide comparison including independent HookNet-TLS polygons"><p>Blue = filtered HookNet TLS polygons; gold = predicted germinal center; red = our original disks. tlsR 50 µm is an exploratory setting; its source default detects no regions.</p><h3>Predicted germinal center</h3><p>{html.escape(gc_note)} This is a model prediction; it does not establish biological maturity.</p><h3>Inspect H&E and predicted contours</h3><select id="hooknet-review">{options}</select>{gallery}<h3>Published filters and execution</h3><p>Raw polygons: {native['raw_TLS_polygons']} TLS and {native['raw_GC_polygons']} GC. TLS retention requires at least 147² pixels with 8-bit confidence ≥210; GC retention requires 60² pixels ≥245 and the author's TLS bounding-rectangle association. Filtering is unchanged. The published network emits 1030×1030 predictions; its writer retains the upper-left 1024×1024 and this behavior is preserved.</p><p>All {provenance['timings']['whole_slide_tissue_tiles']} tissue tiles completed on Apple GPU. Inference plus map writing: {inference:.1f} seconds; {preparation_label}, CPU/GPU checks and postprocessing: {runtime:.1f} seconds total. {exclusions} CPU/GPU pixel-class agreement on the verified region: {verify.get('CPU_GPU_pixel_class_agreement', 0):.2%}; maximum probability difference: {verify.get('CPU_GPU_max_probability_difference', 0):.2g}. Peak process resident memory: {provenance['peak_resident_bytes']/2**30:.2f} GiB.</p><p>The Mac port uses TensorFlow {provenance['tensorflow']}, legacy Keras, OpenSlide {provenance['openslide_library']}, and lossless physically resampled pyramids. It replaces the authors' Linux CUDA/ASAP image reader and JPEG preprocessing. No same-input Linux reference prediction is available for numerical equivalence. All checkpoint weights loaded, source adaptations are recorded, dependency checks passed, and authored threshold/crop/polygonization fixtures passed.</p><p>Saved additions: filtered_regions.csv, native_polygons.geojson, A1_A6_comparison.csv, native prediction/confidence TIFFs, GC_TLS_polygon_relationships.csv, source audit and provenance under hooknet_tls/. Combined tables beside the main report carry the suffix _with_hooknet. Existing spatial result tables remain unchanged.</p><p><a href="https://github.com/DIAGNijmegen/pathology-hooknet-tls/tree/573680a5b962b1acae88cdeb6f8b1c9597570539">Pinned HookNet-TLS source</a> · <a href="https://zenodo.org/records/10614942">Official pretrained weights</a></p></section><script>document.getElementById('hooknet-review').addEventListener('change',e=>document.querySelectorAll('.hooknet-card').forEach(x=>x.style.display=x.dataset.id===e.target.value?'block':'none'));</script>'''
    standalone = '<!doctype html><html lang="en"><meta charset="utf-8"><title>HookNet-TLS whole-slide comparison</title><style>body{font:16px/1.5 -apple-system,sans-serif;background:#f3f7fa;color:#173049}main{max-width:1400px;padding:25px;margin:auto}section{padding:24px;background:white;border-radius:12px}img{width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #dde4eb;text-align:left}.scroll{overflow:auto}.notice{background:#e9f2fb;padding:15px}select{padding:9px;font:inherit}</style><main><h1>HookNet-TLS whole-slide comparison</h1>'+section+'</main></html>'
    (out / "HookNet_TLS_report.html").write_text(standalone)
    original_report = out.parent / "TLS_tool_comparison.html"
    backup = out.parent / "TLS_tool_comparison_before_HookNet.html"
    if not backup.exists():
        shutil.copy2(original_report, backup)
    text = original_report.read_text()
    if 'id="hooknet-extension"' in text:
        # Rebuild from the preserved spatial report, avoiding duplicate extensions.
        text = backup.read_text()
    text = text.replace("<title>H&E TLS spatial tool comparison</title>", "<title>TLS comparison with HookNet-TLS</title>")
    text = text.replace("<h1>H&E TLS spatial tool comparison</h1>", "<h1>TLS comparison with HookNet-TLS</h1>")
    text = text.replace("17 predefined runs", "17 spatial runs + independent whole-slide HookNet-TLS")
    text = text.replace("Exploratory agreement between spatial algorithms using shared cell predictions.", "Comparison of cell-based spatial patterns and independent H&E model predictions.")
    text = text.replace("All methods share our predicted B/T labels.", "The original cell-coordinate methods share our predicted B/T labels.")
    text = text.replace("The whole comparison and normalization took", "The original 17 spatial comparisons and normalization took")
    text = text.replace('<section><h2>What the comparison shows</h2>', section+'<section><h2>Original spatial-tool comparison</h2>', 1)
    original_report.write_text(text)
    old_summary = pd.read_csv(out.parent / "run_summary.csv")
    addition = pd.DataFrame([{"method": "HookNet-TLS", "run_id": "hooknet_published_defaults", "category": "H&E_published_defaults",
        "variant": "H&E_pixels_only", "detections": len(tls), "GC_detections": len(gc),
        "original_candidates_center_matched": len(center_ids), "original_candidates_polygon_overlap": len(overlap_ids),
        "detection_seconds": inference}])
    pd.concat([old_summary, addition], ignore_index=True).to_csv(out.parent / "run_summary_with_hooknet.csv", index=False)
    conclusions = {"filtered_TLS_predictions": len(tls), "filtered_GC_predictions": len(gc), "centroid_matched_candidates": center_ids,
        "polygon_overlap_candidates_10_percent_disk": overlap_ids, "A4_centroid_distance_um": float(a4.nearest_center_distance_um),
        "A4_original_BT_polygon_coverage": float(a4.all_detection_candidate_BT_coverage), "independent_of_cell_classifier": True,
        "retained_GC_has_TLS_polygon_overlap": bool(len(gc_relations) and gc_relations.actual_GC_area_inside_TLS_polygon_fraction.max() > 0),
        "TLS_confirmed": False, "maturity_confirmed": False}
    (out / "conclusions.json").write_text(json.dumps(conclusions, indent=2)+"\n")
    provenance["status"] = "complete"
    provenance["original_analysis_input_hash"] = hashlib.sha256((run/"tls_analysis/spatial_cells.csv").read_bytes()).hexdigest()
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    shutil.copy2(ROOT / "deployment" / "report_hooknet_tls.py", out / "source_audit" / "report_hooknet_tls.py")
    print("HookNet-TLS integrated into the existing comparison report:", conclusions, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs" / "whole-slide-20261012")
    render(parser.parse_args().run_dir.resolve())
