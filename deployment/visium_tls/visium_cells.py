"""HE2CELL cell calls on the USZ Visium TLS H&E images (Zenodo 14620362), for external validation.

The TIFFs are ~0.79 µm/px. Tissue under in-tissue spots is upsampled (bicubic) to the pipeline's
0.25 µm/px on the same 1024-px / 768-stride patch grid used for TCGA-ESCA, segmented once with
Cellpose cpsam_v2, and scored by both the original and the ESCA-fine-tuned classifier (D4 TTA).
No parameter is tuned on these images or their labels.

Cell coordinates are written back in native TIFF pixels so they share the spot coordinate frame.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import h5py
import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "script" / "src"
DATA = ROOT / "data" / "TLS_VISIUM_USZ"
OUT = ROOT / "outputs" / "visium_tls_validation"
SAMPLES = ["LC1", "LC2", "LC3", "LC4", "LC5", "KC1", "KC2", "KC3"]
CLASSIFIERS = {"original": ROOT / "models" / "best_model2.pth",
               "finetuned": ROOT / "outputs" / "classifier_finetune_20261019" / "best_finetuned.pth",
               "finetuned40": ROOT / "outputs" / "classifier_finetune_40pt_20261022" / "best_finetuned.pth",
               # Augmentation ablation: same 10 patients and schedule, original flips + HED ±0.012 only.
               "finetuned_weakaug": ROOT / "outputs" / "classifier_finetune_weakaug_20261025" / "best_finetuned.pth"}
DEFAULT_CLASSIFIERS = ["original", "finetuned", "finetuned40"]
# Output folder for cell tables; v2 = one fresh segmentation scored by all three checkpoints.
RUN = os.environ.get("VISIUM_RUN", "")
MIN_FREE_GB = 6
PATCH, STRIDE, MARGIN = 1024, 768, 128
KEEP = ["global_instance_id", "patch_id", "area_pixels", "centroid_x_global", "centroid_y_global",
        "celltype", "prob_Other", "prob_T", "prob_Macrophage", "prob_B", "confidence"]


def native_mpp(sample):
    with tifffile.TiffFile(DATA / "tif_slides" / f"{sample}.tif") as tif:
        numerator, denominator = tif.pages[0].tags["XResolution"].value
    return denominator / numerator  # ImageJ unit is µm


def spots(sample):
    """Spot labels with centres in THIS TIFF's pixel frame.

    The h5ad x_pixel/y_pixel are not in the TIFF frame. Barcode -> (array_row, array_col) comes from
    Space Ranger's tissue_positions.csv; (row, col) -> TIFF pixels from the Loupe manual alignment
    made for these exact TIFFs (its spot diameter equals the h5ad spot_diameter_fullres).
    """
    obs = h5py.File(DATA / "h5ad_preprocessed" / f"{sample}.h5ad", "r")["obs"]
    def categorical(key):
        categories = [c.decode() if isinstance(c, bytes) else c for c in obs[key]["categories"][:]]
        return [categories[c] if c >= 0 else "NA" for c in obs[key]["codes"][:]]
    index = [b.decode() if isinstance(b, bytes) else b for b in obs["_index"][:]]
    labels = pd.DataFrame({"barcode": index, "in_tissue": obs["in_tissue"][:], "label": categorical("manual_anno_tls")})
    positions = pd.read_csv(DATA / "10x_Visium" / sample / "spatial" / "tissue_positions.csv")
    oligo = pd.DataFrame(json.loads((DATA / "tif_slides" / f"{sample}_manual_loupe_alignment.json").read_text())["oligo"])
    frame = labels.merge(positions[["barcode", "array_row", "array_col"]], on="barcode", how="left")
    frame = frame.merge(oligo[["row", "col", "imageX", "imageY"]], left_on=["array_row", "array_col"],
                        right_on=["row", "col"], how="left")
    if frame.imageX.isna().any() or len(frame) != len(labels):
        raise ValueError(f"{sample}: spots without a TIFF position")
    return frame.rename(columns={"imageX": "x", "imageY": "y"})[["barcode", "x", "y", "in_tissue", "label", "array_row", "array_col"]]


def stage(script, *arguments, log):
    env = {**os.environ, "CELLPOSE_LOCAL_MODELS_PATH": str(ROOT / "models" / "cellpose_cache"),
           "MPLCONFIGDIR": str(ROOT / ".runtime" / "matplotlib"), "OMP_NUM_THREADS": "8"}
    started = time.perf_counter()
    with open(log, "a") as handle:
        subprocess.run([sys.executable, str(SRC / script), *map(str, arguments)], check=True,
                       stdout=handle, stderr=subprocess.STDOUT, env=env)
    return time.perf_counter() - started


def patch_grid(sample, target_mpp):
    """Upsampled patches over in-tissue spots, in order: yields (x, y, RGB patch, tissue ratio).

    x, y are the patch origin in the target-mpp frame (the cells' centroid_*_global frame). Shared with
    deployment/baselines/hovernet_cells.py so both pipelines see identical pixels.
    """
    mpp = native_mpp(sample)
    scale = mpp / target_mpp  # target px per native px
    image = tifffile.imread(DATA / "tif_slides" / f"{sample}.tif")[..., :3]
    table = spots(sample)
    spacing = 100 / mpp  # native px between neighbouring spot centres
    tissue = np.zeros(image.shape[:2], np.uint8)
    for s in table[table.in_tissue == 1].itertuples():
        cv2.circle(tissue, (int(round(s.x)), int(round(s.y))), int(round(.62 * spacing)), 1, -1)
    height, width = (int(d * scale) for d in image.shape[:2])
    for y in range(-MARGIN, height - MARGIN, STRIDE):
        for x in range(-MARGIN, width - MARGIN, STRIDE):
            core = tissue[int((y + MARGIN) / scale):int((y + MARGIN + STRIDE) / scale),
                          int((x + MARGIN) / scale):int((x + MARGIN + STRIDE) / scale)]
            if core.size == 0 or core.mean() < .05:
                continue
            # Native region covering the patch (1 px pad for interpolation), then bicubic upsampling.
            x0, y0 = x / scale - 1, y / scale - 1
            sx, sy = int(np.floor(x0)), int(np.floor(y0))
            size = int(np.ceil(PATCH / scale)) + 3
            region = np.full((size, size, 3), 255, np.uint8)
            ax, ay = max(sx, 0), max(sy, 0)
            bx, by = min(sx + size, image.shape[1]), min(sy + size, image.shape[0])
            if bx > ax and by > ay:
                region[ay - sy:by - sy, ax - sx:bx - sx] = image[ay:by, ax:bx]
            matrix = np.float32([[scale, 0, -(x0 - sx + 1) * scale], [0, scale, -(y0 - sy + 1) * scale]])
            patch = cv2.warpAffine(region, matrix, (PATCH, PATCH), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
            yield x, y, patch, float(core.mean())


def process(sample, target_mpp, classifiers=DEFAULT_CLASSIFIERS, identity=False):
    """identity: also keep each classifier's un-augmented (identity view) probabilities from the same
    D4 pass as classifier '<name>_ttanone', i.e. the --tta none prediction without a second pass."""
    out = OUT / (f"cells_{target_mpp:g}um" + (f"_{RUN}" if RUN else "")) / sample
    if (out / "cells.csv.gz").exists():
        return "done"
    started = time.perf_counter()
    work = OUT / f"_tmp_{target_mpp:g}{RUN}" / sample
    shutil.rmtree(work, ignore_errors=True)
    (work / "patches").mkdir(parents=True)
    out.mkdir(parents=True, exist_ok=True)
    mpp = native_mpp(sample)
    scale = mpp / target_mpp  # target px per native px
    rows = []
    for x, y, patch, tissue_ratio in patch_grid(sample, target_mpp):
        path = work / "patches" / f"patch{len(rows)}_x{x}_y{y}.png"
        cv2.imwrite(str(path), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
        rows.append({"path": str(path), "num_patch": len(rows), "sample": sample, "source_tif": sample,
                     "x": x, "y": y, "patch_size": PATCH, "stride": STRIDE, "level": 0, "tissue_ratio": tissue_ratio})
    patches = pd.DataFrame(rows)
    patch_csv = work / f"{sample}_patch_index.csv"
    patches.to_csv(patch_csv, index=False)
    log = out / "pipeline.log"
    seg = stage("run_cellcrops.py", "--patch_csv", patch_csv, "--out_dir", work / "cells", "--model_name", "cpsam_v2",
                "--device", "mps", "--num_gpus", 1, "--batch_size", 16, "--crop_size", 64, log=log)
    cells_csv = work / "cells" / "mask_tables" / f"{patch_csv.stem}_cellpose_mask_instances.csv"
    tables, timings = {}, {"segmentation_seconds": seg}
    for name in classifiers:
        checkpoint = CLASSIFIERS[name]
        result = work / f"cell_types_{name}.csv"
        timings[f"classification_{name}_seconds"] = stage(
            "model_eval.py", "--meta_path", cells_csv, "--save_path", checkpoint, "--output_path", result,
            "--device", "mps", "--batch_size", 256, "--num_workers", 0, "--tta", "d4", log=log)
        table = pd.read_csv(result)
        tables[name] = table[KEEP]
        if identity:
            tables[f"{name}_ttanone"] = table[["global_instance_id", "raw_celltype"]].rename(columns={"raw_celltype": "celltype"}).assign(
                **{f"prob_{k}": table[f"raw_prob_{k}"] for k in ["Other", "T", "Macrophage", "B"]})
    cells = tables[classifiers[0]][["global_instance_id", "patch_id", "area_pixels", "centroid_x_global", "centroid_y_global"]].copy()
    cells["x_native"], cells["y_native"] = cells.centroid_x_global / scale, cells.centroid_y_global / scale
    cells["area_um2"] = cells.area_pixels * target_mpp ** 2
    for name, table in tables.items():
        assert (table.global_instance_id.to_numpy() == cells.global_instance_id.to_numpy()).all()
        cells[f"celltype_{name}"] = table.celltype.to_numpy()
        for column in ["prob_Other", "prob_T", "prob_Macrophage", "prob_B"]:
            cells[f"{column}_{name}"] = table[column].to_numpy().astype(np.float32)
    cells.to_csv(out / "cells.csv.gz", index=False)
    summary = {"sample": sample, "native_mpp": mpp, "target_mpp": target_mpp, "upsampling": scale, "patches": len(rows),
               "cells": len(cells), **timings, "wall_seconds": time.perf_counter() - started,
               "counts": {name: cells[f"celltype_{name}"].value_counts().to_dict() for name in tables},
               "classifiers": {k: str(CLASSIFIERS[k]) for k in classifiers}, "segmentation_model": "cpsam_v2", "tta": "d4",
               "identity_view_kept": identity,
               "interpolation": "bicubic", "patch_size": PATCH, "stride": STRIDE}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    # Keep a few patches for visual QC; drop the rest (cells.csv.gz plus the TIFF regenerate everything).
    qc = out / "qc_patches"
    qc.mkdir(exist_ok=True)
    for path in patches.sort_values("tissue_ratio", ascending=False).path.head(4):
        shutil.copy2(path, qc / Path(path).name)
    shutil.rmtree(work, ignore_errors=True)
    return f"{len(rows)} patches, {len(cells)} cells, {time.perf_counter() - started:.0f}s"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", default=SAMPLES)
    parser.add_argument("--target-mpp", type=float, default=0.25)
    parser.add_argument("--classifiers", nargs="+", choices=list(CLASSIFIERS), default=DEFAULT_CLASSIFIERS)
    parser.add_argument("--identity", action="store_true", help="Also store identity-view (no TTA) predictions as <name>_ttanone")
    args = parser.parse_args()
    for sample in args.samples:
        free_gb = shutil.disk_usage(OUT).free / 1e9
        if free_gb < MIN_FREE_GB:
            print(f"STOP: only {free_gb:.1f} GB free before {sample}", flush=True)
            break
        lock = OUT / "_locks" / f"{sample}_{args.target_mpp:g}{RUN}"
        lock.parent.mkdir(parents=True, exist_ok=True)
        try:
            lock.mkdir()
        except FileExistsError:
            continue
        try:
            print(f"{sample}: {process(sample, args.target_mpp, args.classifiers, args.identity)}", flush=True)
        finally:
            lock.rmdir()


if __name__ == "__main__":
    main()
