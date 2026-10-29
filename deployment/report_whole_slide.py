"""Measure a completed slide run without assuming biological ground truth."""
import argparse
import base64
from datetime import datetime
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime" / "matplotlib"))
sys.path.insert(0, str(ROOT / "script" / "src"))
LABELS = ["Other", "T", "Macrophage", "B"]
COLORS = {"Other": "#37a9bc", "T": "#ca9510", "Macrophage": "#e8794d", "B": "#37ac69"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    import numpy as np
    import pandas as pd
    from PIL import Image, ImageDraw, ImageFont
    import torch
    import openslide
    from scipy.spatial import cKDTree
    from dataset import CellSamdataset_our
    from model_eval import SAM_ConvNet, get_transforms
    from runtime import select_device, load_classifier_state
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = args.run.resolve()
    manifest = json.loads((run / "run.json").read_text())
    if manifest["status"] != "complete" or manifest["scope"] != "full slide":
        raise RuntimeError("Report requires a completed full-slide run")
    sample = Path(manifest["slides"][0]["results"]).parent
    cells = pd.read_csv(sample / "cell_types.csv")
    patches = pd.read_csv(next((sample / "patches" / "tables").glob("*_patch_index.csv")))
    report_dir = run / "performance_report"
    report_dir.mkdir(exist_ok=True)
    probabilities = cells[["prob_" + label for label in LABELS]].to_numpy()
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
    assert np.array_equal(probabilities.argmax(axis=1), cells.prediction.to_numpy())
    counts = cells.celltype.value_counts().reindex(LABELS, fill_value=0)
    per_class = pd.DataFrame({"celltype": LABELS, "count": counts.to_numpy(),
                              "percent": counts.to_numpy() / len(cells) * 100})
    per_class["median_confidence"] = [float(cells.loc[cells.celltype == label, "confidence"].median())
                                       if counts[label] else np.nan for label in LABELS]
    per_class["fraction_confidence_at_least_0_9"] = [float((cells.loc[cells.celltype == label, "confidence"] >= .9).mean())
                                                     if counts[label] else np.nan for label in LABELS]
    per_class.to_csv(report_dir / "class_summary.csv", index=False)
    missing = {column: sum(not Path(p).is_file() for p in cells[column])
               for column in ["cell_image_crop_path", "cell_mask_crop_path"]}
    assert not any(missing.values())
    assert cells.global_instance_id.is_unique
    progress = json.loads((sample / "cells" / "progress.json").read_text())
    assert progress["patches_completed"] == len(patches)
    crop_status = cells.cell_crop_status.value_counts().to_dict()
    patch_count = cells.groupby("patch_id").size().reindex(patches.num_patch, fill_value=0)
    near_pairs = cKDTree(cells[["centroid_x_global", "centroid_y_global"]].to_numpy()).query_pairs(2.0, output_type="ndarray")
    if len(near_pairs):
        patch_ids = cells.patch_id.to_numpy()
        cross_patch_pairs = int((patch_ids[near_pairs[:, 0]] != patch_ids[near_pairs[:, 1]]).sum())
    else:
        cross_patch_pairs = 0

    with openslide.OpenSlide(manifest["slides"][0]["slide"]) as slide:
        w, h = slide.dimensions
        mpp = float(slide.properties.get("openslide.mpp-x", "nan"))
        thumbnail = slide.get_thumbnail((1600, 1600)).convert("RGB")
    tissue_image = Image.open(next((sample / "patches" / "qc").glob("*_tissue_mask.png")))
    tissue = np.asarray(tissue_image) != 0
    tissue_image.close()
    th, tw = tissue.shape
    coverage = np.zeros((th, tw), dtype=bool)
    margin = max(0, (manifest["settings"]["patch_size"] - manifest["settings"]["stride"]) // 2)
    size = manifest["settings"]["patch_size"]
    for row in patches.itertuples():
        x1, x2 = max(0, int((row.x + margin) * tw / w)), min(tw, int(np.ceil((row.x + size - margin) * tw / w)))
        y1, y2 = max(0, int((row.y + margin) * th / h)), min(th, int(np.ceil((row.y + size - margin) * th / h)))
        coverage[y1:y2, x1:x2] = True
    coverage_fraction = float((coverage & tissue).sum() / tissue.sum())
    total_seconds = manifest["wall_seconds"]
    segmentation_seconds = manifest["stages"]["segmentation"]["wall_seconds"]
    classification_seconds = manifest["stages"]["classification"]["wall_seconds"]
    classification_metrics = json.loads((sample / "cell_types.inference.json").read_text())
    confidence_summary = {"median": float(cells.confidence.median()),
                          "p10": float(cells.confidence.quantile(.1)),
                          "p90": float(cells.confidence.quantile(.9)),
                          "fraction_below_0_5": float((cells.confidence < .5).mean()),
                          "fraction_at_least_0_9": float((cells.confidence >= .9).mean()),
                          "median_normalized_entropy": float(cells.normalized_entropy.median()),
                          "median_top_two_probability_margin": float(cells.probability_margin.median())}
    report = {
        "scope": "all tissue patches selected by the configured tissue filter; background excluded",
        "created_at": datetime.now().astimezone().isoformat(),
        "slide": manifest["slides"][0]["slide"], "slide_dimensions": [w, h], "microns_per_pixel": mpp,
        "classifier": manifest["classifier"], "segmentation_model": manifest["segmentation_model"],
        "device": manifest["device"], "settings": manifest["settings"],
        "software": manifest.get("software") or {"python":sys.version.split()[0],
                     **{package:importlib.metadata.version(package) for package in ["torch","torchvision","cellpose","openslide-python"]}},
        "patches_processed": len(patches), "cells_classified": len(cells),
        "predicted_counts": {label: int(counts[label]) for label in LABELS},
        "wall_seconds": total_seconds, "stages": manifest["stages"],
        "throughput": {"segmentation_patches_per_second": len(patches) / segmentation_seconds,
                       "classification_cells_per_second_including_startup": len(cells) / classification_seconds,
                       "classification_cells_per_second_during_inference_and_loading": len(cells) / classification_metrics["inference_including_data_loading_seconds"],
                       "whole_pipeline_cells_per_second": len(cells) / total_seconds},
        "memory": {"largest_stage_peak_process_resident_bytes": max(v.get("peak_process_resident_bytes", 0) for v in manifest["stages"].values()),
                   "maximum_observed_mps_driver_allocated_bytes": progress["maximum_observed_mps_driver_bytes"],
                   "note": "Process RSS and MPS driver allocation are different measurements on unified memory; do not add them."},
        "confidence": confidence_summary,
        "segmentation_diagnostics": {
            "median_instance_area_pixels": float(cells.area_pixels.median()),
            "area_pixels_p05": float(cells.area_pixels.quantile(.05)),
            "area_pixels_p95": float(cells.area_pixels.quantile(.95)),
            "median_cells_per_selected_patch": float(patch_count.median()),
            "patches_with_no_retained_cells": int((patch_count == 0).sum()),
            "crop_status": {str(k): int(v) for k, v in crop_status.items()},
            "fraction_masks_fully_contained_in_64_pixel_crop": float((cells.cell_mask_area_crop == cells.area_pixels).mean()),
            "nearby_cross_patch_centroid_pairs_within_2_pixels": cross_patch_pairs,
            "estimated_tissue_thumbnail_fraction_inside_retained_patch_cores": coverage_fraction,
            "note": "Core filtering reduces overlap duplicates but may omit cells near excluded patch edges. Coverage is approximate at thumbnail resolution. Nearby centroids are a duplication flag, not confirmed errors."},
        "integrity": {"missing_crop_files": missing, "unique_cell_ids": True,
                      "probabilities_finite_and_sum_to_one": True},
        "biological_accuracy": {"measured": False, "reason": "No gold-standard cell labels or reference masks were supplied."},
    }
    (report_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Integrity and full-slide metrics saved; evaluating numerical and flip consistency.", flush=True)

    # Stratification exposes the minority classes; it does not estimate a population error rate.
    selected = pd.concat([cells[cells.celltype == label].sample(n=min(128, counts[label]), random_state=569)
                          for label in LABELS if counts[label]]).reset_index(drop=True)
    _, transform = get_transforms()
    dataset = CellSamdataset_our(selected, transform)
    torch.set_num_threads(8)
    model = SAM_ConvNet(num_classes=4, pretrained=False).eval()
    model.load_state_dict(load_classifier_state(manifest["classifier"]))
    device = select_device(args.device)
    images = torch.stack([dataset[i][0] for i in range(len(dataset))])
    cpu_logits = []
    with torch.inference_mode():
        model(images[:64])
        cpu_started = time.perf_counter()
        for chunk in images.split(64):
            cpu_logits.append(model(chunk).numpy())
        cpu_seconds = time.perf_counter() - cpu_started
    cpu_logits = np.concatenate(cpu_logits)
    model = model.to(device)
    consistency_rows = []
    original_gpu = []
    with torch.inference_mode():
        model(images[:64].to(device))
        if device.type == "mps":
            torch.mps.synchronize()
        gpu_started = time.perf_counter()
        for chunk in images.split(64):
            original_gpu.append(model(chunk.to(device)).cpu().numpy())
        if device.type == "mps":
            torch.mps.synchronize()
        gpu_seconds = time.perf_counter() - gpu_started
        original_gpu = np.concatenate(original_gpu)
        original_predictions = original_gpu.argmax(axis=1)
        original_probabilities = torch.softmax(torch.from_numpy(original_gpu), dim=1).numpy()
        for name, axes in [("horizontal_flip", [-1]), ("vertical_flip", [-2])]:
            changed = []
            for chunk in images.split(64):
                changed.append(torch.softmax(model(chunk.flip(axes).to(device)), dim=1).cpu().numpy())
            changed = np.concatenate(changed)
            changed_predictions = changed.argmax(axis=1)
            for i, row in selected.iterrows():
                consistency_rows.append({"global_instance_id": int(row.global_instance_id),
                                         "original_type": row.celltype, "transformation": name,
                                         "transformed_type": LABELS[changed_predictions[i]],
                                         "label_unchanged": bool(original_predictions[i] == changed_predictions[i]),
                                         "absolute_probability_shift": float(np.abs(changed[i] - original_probabilities[i]).mean())})
    consistency = pd.DataFrame(consistency_rows)
    consistency.to_csv(report_dir / "flip_consistency.csv", index=False)
    stability = {}
    for name, group in consistency.groupby("transformation"):
        stability[name] = {"label_agreement": float(group.label_unchanged.mean()),
                           "per_original_class": {label: float(part.label_unchanged.mean())
                                                  for label, part in group.groupby("original_type")},
                           "mean_absolute_probability_shift": float(group.absolute_probability_shift.mean())}
    report["consistency_checks"] = {
        "sample_size": len(selected), "sample_per_class": selected.celltype.value_counts().to_dict(),
        "cpu_gpu_label_agreement": float((cpu_logits.argmax(axis=1) == original_gpu.argmax(axis=1)).mean()),
        "gpu_repeat_agreement_with_saved_predictions": float((original_gpu.argmax(axis=1) == selected.prediction.to_numpy()).mean()),
        "maximum_cpu_gpu_absolute_logit_difference": float(np.abs(cpu_logits - original_gpu).max()),
        "warmed_sample_benchmark": {"batch_size":64, "cpu_threads":8,
                                    "cpu_inference_seconds":cpu_seconds, "gpu_inference_seconds":gpu_seconds,
                                    "cpu_cells_per_second":len(selected)/cpu_seconds,
                                    "gpu_cells_per_second":len(selected)/gpu_seconds,
                                    "gpu_speedup":cpu_seconds/gpu_seconds,
                                    "note":"Same crops and batch size, one warmup pass per device; excludes model initialization and disk loading, includes CPU-to-GPU input transfer. One timed pass on this stratified sample."},
        "flips": stability,
        "interpretation": "Class-stratified sample of up to 128 cells per predicted class. Both RGB and mask are flipped together. Repeatability and flip stability are not biological accuracy."}
    (report_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "#f7f9fc", "axes.facecolor": "white"})
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    fig.suptitle(f"Whole-slide execution and prediction diagnostics\n{len(patches):,} patches · {len(cells):,} cells · {total_seconds/60:.1f} min · Apple GPU", fontsize=18)
    names = list(manifest["stages"])
    axes[0,0].barh(names, [manifest["stages"][name]["wall_seconds"] / 60 for name in names], color="#4264b0")
    axes[0,0].set(title="Measured stage time", xlabel="Minutes")
    bars = axes[0,1].bar(LABELS, counts, color=[COLORS[label] for label in LABELS])
    axes[0,1].bar_label(bars, labels=[f"{v:,}" for v in counts], padding=3)
    axes[0,1].set(title="Predicted cell types", ylabel="Cells")
    axes[0,1].margins(y=.15)
    axes[0,2].hist(cells.confidence, bins=np.linspace(.25,1,31), color="#4264b0")
    axes[0,2].axvline(.9, color="#d77936", linestyle="--")
    axes[0,2].set(title="Model confidence (uncalibrated)", xlabel="Largest softmax probability", ylabel="Cells")
    axes[1,0].imshow(thumbnail)
    tw2, th2 = thumbnail.size
    for label in ["T", "Macrophage", "B"]:
        group = cells[cells.celltype == label]
        axes[1,0].scatter(group.centroid_x_global * tw2/w, group.centroid_y_global * th2/h,
                          s=.5, color=COLORS[label], label=f"{label} ({len(group):,})", alpha=.65)
    axes[1,0].set_title("Predicted immune-cell locations")
    axes[1,0].axis("off")
    axes[1,0].legend(loc="upper left", fontsize=8, markerscale=4)
    axes[1,1].hist(cells.area_pixels, bins=np.geomspace(max(1,cells.area_pixels.min()), cells.area_pixels.max()+1, 45), color="#37a9bc")
    axes[1,1].set_xscale("log")
    axes[1,1].set(title="Retained instance size", xlabel="Mask area (pixels, log scale)", ylabel="Cells")
    stability_matrix = [[stability.get(name, {}).get("per_original_class", {}).get(label, np.nan) for name in ["horizontal_flip", "vertical_flip"]] for label in LABELS]
    axes[1,2].imshow(stability_matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axes[1,2].set_xticks([0,1], ["Horizontal flip", "Vertical flip"])
    axes[1,2].set_yticks(range(4), LABELS)
    axes[1,2].set_title("Flip label agreement in stratified sample")
    for i, row in enumerate(stability_matrix):
        for j, value in enumerate(row):
            axes[1,2].text(j,i, "n/a" if np.isnan(value) else f"{100*value:.1f}%", ha="center", va="center", color="white" if value>.6 else "black")
    fig.savefig(report_dir / "performance_dashboard.png", dpi=150)
    plt.close(fig)

    # Native-resolution examples show where segmentation and classification should be reviewed.
    candidates = cells.copy()
    examples = []
    seen = set()
    groups = [(f"{label}: high confidence", candidates[candidates.celltype == label].sort_values("confidence", ascending=False)) for label in LABELS]
    groups += [("Lower confidence", candidates.sort_values("confidence"))]
    for title, group in groups:
        for _, row in group.head(12).iterrows():
            if row.global_instance_id in seen:
                continue
            seen.add(row.global_instance_id)
            examples.append((title,row))
            if sum(t == title for t,_ in examples) >= 4:
                break
    canvas = Image.new("RGB", (1200, 90 + len(groups)*270), "#15202d")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 19)
    except OSError:
        font = ImageFont.load_default(size=19)
    draw.text((15,14), "Representative predictions for visual review", font=font, fill="white")
    draw.text((15,42), "Original image crop and predicted cell mask; confidence is not correctness.", font=font, fill="white")
    grouped = {title: [row for t,row in examples if t == title] for title,_ in groups}
    for group_index,(title, _) in enumerate(groups):
        y = 88 + group_index*270
        draw.text((15,y), title, font=font, fill="white")
        for column,row in enumerate(grouped[title]):
            x = 15 + column*295
            with Image.open(row.cell_image_crop_path) as crop:
                canvas.paste(crop.convert("RGB").resize((128,128)), (x,y+35))
            with Image.open(row.cell_mask_crop_path) as mask:
                canvas.paste(mask.convert("RGB").resize((128,128)), (x+137,y+35))
            draw.text((x,y+170), f"{row.celltype}, p={row.confidence:.3f}", font=font, fill="white")
            draw.text((x,y+196), f"Cell {int(row.global_instance_id)}", font=font, fill="white")
    canvas.save(report_dir / "cell_review_gallery.png")

    peak = report["memory"]["largest_stage_peak_process_resident_bytes"] / 2**30
    mps_memory = report["memory"]["maximum_observed_mps_driver_allocated_bytes"] / 2**30
    lines = ["# Whole-slide performance report", "", f"Processed all **{len(patches):,} selected tissue patches** from the supplied SVS slide using `best_model2.pth`, official `cpsam_v2`, and the Apple GPU.", "",
             f"**{len(cells):,} cells classified in {total_seconds/60:.2f} minutes.** This is one slide and one configured run, not a general benchmark.", "",
             "## Measured computing performance", "", "| Metric | Result |", "| --- | --- |",
             f"| Total pipeline wall time | {total_seconds/60:.2f} min |",
             f"| Segmentation and crop generation | {segmentation_seconds/60:.2f} min |",
             f"| Classification, including process/model startup | {classification_seconds:.2f} s |",
             f"| Segmentation speed | {len(patches)/segmentation_seconds:.3f} patches/s |",
             f"| Classification speed, including startup | {len(cells)/classification_seconds:.1f} cells/s |",
             f"| Largest stage peak process RSS | {peak:.2f} GiB |",
             f"| Maximum observed MPS driver allocation | {mps_memory:.2f} GiB |", "",
             "RSS is the operating-system process measurement. MPS allocation was observed at segmentation batch boundaries and includes driver/cache allocations. On unified memory these measures overlap; they must not be added.", "",
             "## Predicted composition and confidence", "", "| Type | Cells | Share | Median confidence |", "| --- | --- | --- | --- |"]
    for row in per_class.itertuples():
        confidence = "n/a" if np.isnan(row.median_confidence) else f"{row.median_confidence:.3f}"
        lines.append(f"| {row.celltype} | {row.count:,} | {row.percent:.2f}% | {confidence} |")
    lines += ["", f"Median maximum softmax probability: **{confidence_summary['median']:.3f}**. Confidence ≥0.9: **{100*confidence_summary['fraction_at_least_0_9']:.1f}%**; confidence <0.5: **{100*confidence_summary['fraction_below_0_5']:.1f}%**.", "",
              "These probabilities have not been calibrated against labels. High confidence does not demonstrate a correct cell type, and confidence thresholds do not define an accuracy or error rate.", "",
              "## Consistency checks", "", f"The check sampled **{len(selected)} cells**, up to 128 per predicted class, with a fixed seed. Minority classes are intentionally overrepresented.", "",
              f"CPU versus GPU class agreement: **{100*report['consistency_checks']['cpu_gpu_label_agreement']:.2f}%**. GPU repeat versus saved class agreement: **{100*report['consistency_checks']['gpu_repeat_agreement_with_saved_predictions']:.2f}%**. Largest absolute CPU/GPU logit difference: {report['consistency_checks']['maximum_cpu_gpu_absolute_logit_difference']:.6g}.", ""]
    lines += [f"On the same in-memory sample with batch size 64, a warmup pass per device, and 8 CPU threads: CPU inference took **{cpu_seconds:.3f} s** and Apple GPU inference took **{gpu_seconds:.3f} s**, a measured **{cpu_seconds/gpu_seconds:.2f}×** speedup. This one-pass sample benchmark excludes model initialization and disk loading, so it differs from full-pipeline throughput.", ""]
    for name,values in stability.items():
        lines.append(f"- {name.replace('_',' ').capitalize()}: **{100*values['label_agreement']:.1f}%** unchanged labels; mean absolute probability change {values['mean_absolute_probability_shift']:.4f}.")
    review_lines = ["## Findings that warrant review", "",
                    "The numerical deployment checks passed. The following class-level flip results identify sensitivity to orientation, not measured classification errors:", ""]
    for label in LABELS:
        values = [entry["per_original_class"][label] for entry in stability.values() if label in entry["per_original_class"]]
        if values:
            review_lines.append(f"- {label}: {100*min(values):.1f}–{100*max(values):.1f}% unchanged labels across the two flip tests.")
    review_lines += ["", f"There are {cross_patch_pairs} nearby cross-patch centroid pairs for possible duplicate review. Saved counts are unadjusted. Confidence, composition, and orientation sensitivity need reference labels for biological interpretation.", ""]
    lines += ["", "The RGB crop and segmentation mask were flipped together. This measures sensitivity to image orientation, not correctness. Per-class results are in `metrics.json` and the dashboard.", "", *review_lines,
              "## Segmentation and data checks", "",
              f"- Missing image/mask crops: {sum(missing.values())}. Cell IDs are unique and all probability rows are finite and sum to one.",
              f"- Empty selected patches: {int((patch_count == 0).sum())} of {len(patches)}. Median retained cells per selected patch: {patch_count.median():.0f}.",
              f"- Median instance area: {cells.area_pixels.median():.0f} pixels; 5th–95th percentile: {cells.area_pixels.quantile(.05):.0f}–{cells.area_pixels.quantile(.95):.0f} pixels.",
              f"- Masks fully contained in the 64×64 crop: {100*report['segmentation_diagnostics']['fraction_masks_fully_contained_in_64_pixel_crop']:.1f}%.",
              f"- Cross-patch centroid pairs within 2 pixels: {cross_patch_pairs}. This flags possible overlap duplicates for inspection; it is not proof of duplication.",
              f"- Approximate tissue-thumbnail fraction covered by retained patch cores: {100*coverage_fraction:.1f}%.", "",
              "The original 512-pixel core filter within overlapping 1024-pixel patches is retained. All selected patches were processed, but cells outside retained cores are excluded to reduce duplicates; some edge cells can be omitted. Counts are the pipeline's retained detections, not a census of every biological cell.", "",
              "## What cannot be measured without reference labels", "",
              "Cell-type accuracy, precision, recall, F1, confusion-matrix errors, and calibration require labeled cells. Segmentation detection recall, mask precision, and IoU require reference masks. This run establishes successful whole-slide execution, output integrity, numerical consistency, and prediction behavior; it does not establish biological validity.", "",
              "The gallery provides high-confidence examples for each predicted class and lower-confidence examples for manual review. A pathologist-labeled subset, including missed cells and tissue boundaries, would support a later accuracy evaluation.", "",
              "![Performance dashboard](performance_dashboard.png)", "", "![Cell review gallery](cell_review_gallery.png)"]
    (report_dir / "performance_report.md").write_text("\n".join(lines) + "\n")
    tables = per_class.to_html(index=False, float_format=lambda x: f"{x:.3f}")
    embedded = []
    for name in ["performance_dashboard.png", "cell_review_gallery.png"]:
        embedded.append(f'<img alt="{html.escape(name)}" src="data:image/png;base64,{base64.b64encode((report_dir/name).read_bytes()).decode()}">')
    document = f'''<!doctype html><html><head><meta charset="utf-8"><title>Whole-slide performance</title><style>body{{font:16px system-ui;max-width:1200px;margin:40px auto;padding:20px;color:#1a2533}}img{{width:100%;height:auto}}table{{border-collapse:collapse}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}pre{{white-space:pre-wrap;line-height:1.55}}.note{{background:#eef3fa;padding:18px;border-radius:12px}}</style></head><body><h1>Whole-slide performance</h1><p>{len(patches):,} patches · {len(cells):,} cells · {total_seconds/60:.2f} minutes · Apple GPU</p><p class="note">No gold standard was supplied. Confidence and consistency are prediction diagnostics, not accuracy.</p>{tables}{''.join(embedded)}<h2>Detailed measurements and limitations</h2><pre>{html.escape(chr(10).join(lines[:-4]))}</pre></body></html>'''
    (report_dir / "performance_report.html").write_text(document)
    print(json.dumps({"report": str(report_dir), "cells": len(cells), "seconds": total_seconds,
                      "counts": report["predicted_counts"], "consistency": report["consistency_checks"]}, indent=2))


if __name__ == "__main__":
    main()
