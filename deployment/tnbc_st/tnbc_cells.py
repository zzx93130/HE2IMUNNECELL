"""HE2CELL cell calls on the TNBC_ST H&E arrays (Zenodo 14204217, imagesHD), modelled on visium_cells.process().

Per array: the imagesHD jpg (31,744 px square, HD_MPP µm/px, scale.json) is decoded from the tar, resampled
onto the pipeline's 0.25 µm/px grid by visium_cells.patch_grid_image (1024-px patches, stride 768, 128-px core
margin, bicubic), segmented with Cellpose cpsam_v2 (script/src/run_cellcrops.py) and classified by the
deployed checkpoint models/he2cell_esca_finetuned.pth (its tau-0.5 prior sidecar is applied automatically by
model_eval.py) with D4 TTA, on raw H&E (no stain normalisation). Nothing is tuned on TNBC.

Tissue mask (imagesLarge/4 grid, ~3.6 µm/px): smoothed saturation x OD-sum above max(0.1, 0.3 x Otsu), cleaned, minus the pathologist
"Artefacts" and "Hole" classes on annotated arrays, restricted to the ST array frame (all-spot bounding box
+ 200 µm) when the array's spot table is known.

Outputs (outputs/tnbc_st_20261101/cells/<array>/): cells.csv.gz with visium KEEP columns (centroid_*_global in
the 0.25 µm frame) plus x_hd/y_hd (imagesHD px), x_large/y_large (imagesLarge px = annotation frame) and
area_um2; summary.json (timings, counts, checkpoint sha256, prior sidecar); tissue_mask.png; 4 QC patches.
Work units are claimed with mkdir locks and skipped when done; temporary patches are deleted per array.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "visium_tls"))
import tnbc_common as C  # noqa: E402
import visium_cells as V  # noqa: E402

CHECKPOINT = C.ROOT / "models" / "he2cell_esca_finetuned.pth"
CHECKPOINT_SHA256 = "1778b5dd336db6e58f59b3a2533c2b15631d8f9b136d539072fd0038cb2a5f1d"
OUT = C.OUT / "cells"
TARGET_MPP = 0.25
HD, LARGE = 31744, 9523
ANNOTATION_COLOURS = {"Nothing": "#FFFFFF", "Tumor": "#017801", "Necrosis": "#000000", "Fat tissue": "#000080",
                      "Low TIL stroma": "#ff904f", "Vessels": "#dc0000", "Artefacts": "#6e2400",
                      "Lactiferous duct": "#9980e6", "High TIL stroma": "#e9e900", "in situ": "#ccffcc",
                      "Lymphoid nodule": "#80801a", "Hole (whitespace)": "#40e5f6", "Lymphocyte": "#c4417f",
                      "Stroma cell": "#ff9980", "Nerve": "#4d8080", "Heterologous elements": "#808080",
                      "Acellular stroma": "#e9d1bb", "Tumor region": "#258a15"}


def scale():
    return json.loads((C.OUT / "scale.json").read_text())


def inventory():
    return json.loads((C.OUT / "inventory.json").read_text())


def annotation_labels(entry, shape=None, tolerance=20):
    """Annotation PNG -> int label map (index into ANNOTATION_COLOURS, -1 = unmatched colour) in its own frame.

    Colours are matched to the nearest README colour when every channel is within `tolerance` (the PNGs use
    near-duplicates such as #e8d1bb for Acellular stroma, #9080e6 for Lactiferous duct, #50e5f6 for Hole).
    """
    from PIL import Image
    import io
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(io.BytesIO(C.member_bytes("HE", entry["annotation"])))
    reference = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in ANNOTATION_COLOURS.values()])
    if image.mode == "P":
        palette = np.array(image.getpalette()).reshape(-1, 3)
        index = np.asarray(image)
    else:
        rgb = np.asarray(image.convert("RGB")).reshape(-1, 3)
        palette, inverse = np.unique(rgb, axis=0, return_inverse=True)
        index = inverse.reshape(image.size[1], image.size[0])
    diff = np.abs(palette[:, None, :].astype(int) - reference[None]).max(-1)
    lut = np.where(diff.min(1) <= tolerance, diff.argmin(1), -1).astype(np.int16)
    return lut[index]


def tissue_mask(entry, large4):
    """0/1 mask on the imagesLarge/4 grid (large4 = RGB at that grid)."""
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_holes, remove_small_objects
    hsv = cv2.cvtColor(large4, cv2.COLOR_RGB2HSV)
    od = -np.log((large4.astype(np.float32) + 1) / 256).sum(-1)
    raw = (hsv[..., 1].astype(np.float32) / 255) * od
    threshold = float(threshold_otsu(cv2.GaussianBlur(raw, (0, 0), 2)))
    # Heavily smoothed (sigma ~18 µm) saturation x OD-sum above max(0.1, 0.3 x Otsu): a clean outline that keeps
    # pale stroma and fat inside the tissue (pixel-wise Otsu punched holes in loose stroma).
    mask = cv2.GaussianBlur(raw, (0, 0), 5) > max(0.1, 0.3 * threshold)
    px_um = scale()["large_mpp"] * LARGE / large4.shape[0]
    mask = remove_small_objects(mask, max_size=int(20000 / px_um ** 2))
    mask = remove_small_holes(mask, max_size=int(100000 / px_um ** 2))
    info = {"otsu": threshold, "grid_um": px_um}
    if entry.get("annotation"):
        labels = annotation_labels(entry)
        names = list(ANNOTATION_COLOURS)
        small = cv2.resize(labels, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        drop = np.isin(small, [names.index("Artefacts"), names.index("Hole (whitespace)")])
        info["annotation_artefact_or_hole_fraction_of_mask"] = float((mask & drop).sum() / max(mask.sum(), 1))
        mask &= ~drop
    if entry.get("spots_frame_large"):
        x0, y0, x1, y1 = entry["spots_frame_large"]
        f = large4.shape[0] / LARGE
        frame = np.zeros_like(mask)
        frame[max(int(y0 * f), 0):int(np.ceil(y1 * f)), max(int(x0 * f), 0):int(np.ceil(x1 * f))] = True
        info["outside_ST_frame_fraction"] = float((mask & ~frame).sum() / max(mask.sum(), 1))
        mask &= frame
    info["tissue_mm2"] = float(mask.sum() * px_um ** 2 / 1e6)
    return mask.astype(np.uint8), info


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def process(array, entry, dry=False):
    out = OUT / array
    if (out / "cells.csv.gz").exists():
        return "done"
    started = time.perf_counter()
    work = C.OUT / "_tmp_cells" / array
    shutil.rmtree(work, ignore_errors=True)
    (work / "patches").mkdir(parents=True)
    out.mkdir(parents=True, exist_ok=True)
    s = scale()
    hd_mpp = s["hd_mpp"]
    image = C.read_image(entry["hd"])
    if image.shape[:2] != (HD, HD):
        raise ValueError(f"{array}: unexpected HD size {image.shape}")
    large4 = cv2.resize(image, (LARGE // 4, LARGE // 4), interpolation=cv2.INTER_AREA)
    mask, mask_info = tissue_mask(entry, large4)
    cv2.imwrite(str(out / "tissue_mask.png"), mask * 255)
    # tissue mask px per native (HD) px
    tissue_scale = mask.shape[0] / HD
    factor = hd_mpp / TARGET_MPP  # target px per HD px
    rows = []
    for x, y, patch, ratio in V.patch_grid_image(image, hd_mpp, mask, TARGET_MPP, tissue_scale=tissue_scale):
        path = work / "patches" / f"patch{len(rows)}_x{x}_y{y}.png"
        cv2.imwrite(str(path), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
        rows.append({"path": str(path), "num_patch": len(rows), "sample": array, "source_tif": array, "x": x, "y": y,
                     "patch_size": V.PATCH, "stride": V.STRIDE, "level": 0, "tissue_ratio": ratio})
        if C.free_gb() < C.MIN_FREE_GB:
            raise SystemExit(f"STOP: {C.free_gb():.1f} GB free while writing patches of {array}")
    del image
    patches = pd.DataFrame(rows)
    patch_csv = work / f"{array}_patch_index.csv"
    patches.to_csv(patch_csv, index=False)
    patches[["num_patch", "x", "y", "tissue_ratio"]].to_csv(out / "patch_index.csv", index=False)  # HoVer-Net uses the same patches
    if dry:
        shutil.rmtree(work, ignore_errors=True)
        return f"dry: {len(rows)} patches, tissue {mask_info['tissue_mm2']:.1f} mm2"
    log = out / "pipeline.log"
    seg = V.stage("run_cellcrops.py", "--patch_csv", patch_csv, "--out_dir", work / "cells", "--model_name", "cpsam_v2",
                  "--device", "mps", "--num_gpus", 1, "--batch_size", 16, "--crop_size", 64, log=log)
    cells_csv = work / "cells" / "mask_tables" / f"{patch_csv.stem}_cellpose_mask_instances.csv"
    result = work / "cell_types.csv"
    cls = V.stage("model_eval.py", "--meta_path", cells_csv, "--save_path", CHECKPOINT, "--output_path", result,
                  "--device", "mps", "--batch_size", 256, "--num_workers", 0, "--tta", "d4", log=log)
    table = pd.read_csv(result)
    inference = json.loads(result.with_suffix(".inference.json").read_text())
    cells = table[V.KEEP].copy()
    cells["x_hd"] = (cells.centroid_x_global / factor).astype(np.float32)
    cells["y_hd"] = (cells.centroid_y_global / factor).astype(np.float32)
    cells["x_large"] = (cells.x_hd * LARGE / HD).astype(np.float32)
    cells["y_large"] = (cells.y_hd * LARGE / HD).astype(np.float32)
    cells["area_um2"] = (cells.area_pixels * TARGET_MPP ** 2).astype(np.float32)
    for column in ["prob_Other", "prob_T", "prob_Macrophage", "prob_B", "confidence"]:
        cells[column] = cells[column].astype(np.float32)
    cells.to_csv(out / "cells.csv.gz", index=False)
    summary = {"array": array, "sample": entry["sample"], "hd_member": entry["hd"], "hd_mpp": hd_mpp, "target_mpp": TARGET_MPP,
               "resampling_factor": factor, "patches": len(rows), "cells": len(cells), "tissue": mask_info,
               "segmentation_seconds": seg, "classification_seconds": cls, "wall_seconds": time.perf_counter() - started,
               "counts": cells.celltype.value_counts().to_dict(), "median_nucleus_area_um2": float(cells.area_um2.median()),
               "classifier": str(CHECKPOINT.relative_to(C.ROOT)), "classifier_sha256": CHECKPOINT_SHA256,
               "prior_correction": inference.get("prior_correction"), "tta": inference.get("test_time_augmentation"),
               "segmentation_model": "cpsam_v2", "interpolation": "bicubic", "patch_size": V.PATCH, "stride": V.STRIDE,
               "core_margin": V.MARGIN, "stain_normalisation": "none"}
    C.save_json(summary, out / "summary.json")
    qc = out / "qc_patches"
    qc.mkdir(exist_ok=True)
    for path in patches.sort_values("tissue_ratio", ascending=False).path.head(4):
        shutil.copy2(path, qc / Path(path).name)
    shutil.rmtree(work, ignore_errors=True)
    return f"{len(rows)} patches, {len(cells)} cells, {time.perf_counter() - started:.0f}s"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", nargs="+", required=True)
    parser.add_argument("--dry", action="store_true", help="tissue mask + patch count only")
    args = parser.parse_args()
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise SystemExit("checkpoint sha256 mismatch")
    entries = inventory()["arrays"]
    for array in args.arrays:
        if C.free_gb() < C.MIN_FREE_GB:
            print(f"STOP: only {C.free_gb():.1f} GB free before {array}", flush=True)
            break
        lock = OUT / "_locks" / array
        lock.parent.mkdir(parents=True, exist_ok=True)
        try:
            lock.mkdir()
        except FileExistsError:
            print(f"{array}: locked, skipped", flush=True)
            continue
        try:
            print(f"{array}: {process(array, entries[array], args.dry)}", flush=True)
        except Exception as error:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"{array}: FAILED {error!r}", flush=True)
        finally:
            lock.rmdir()


if __name__ == "__main__":
    main()
