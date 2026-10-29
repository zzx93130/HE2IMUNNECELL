"""Run the HE2CELL segmentation + four-class classifier on pre-specified windows of each ESCA slide.

Windows per slide (selected before any HE2CELL prediction is seen, seed = slide barcode):
  tls     up to 3 randomly chosen HookNet-TLS filtered TLS polygons, window centred on each
  random  3 windows centred on random tissue pixels (>=50% tissue), unconditional
  dense   2 hard negatives: highest mean haematoxylin density with no HookNet TLS/GC pixel
          within 100 µm of the window
Windows never overlap. Each window is a 4x4 grid of 1024-px level-0 patches at stride 768,
so the cores (the only place cells are kept) tile a 3072-px (~0.76 mm) square exactly.

Run with .venv/bin/python. Slide-level image crops are deleted after classification; the
source slide plus cell table regenerate them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from esca_qc import artifact_masks, slide_qc
from esca_common import CLASSIFIER, EXCLUDED_DIAGNOSES, OUT, ROOT, RAW, save_json, slide_barcode

SRC = ROOT / "script" / "src"
sys.path.insert(0, str(SRC))
from slide_io import open_slide  # noqa: E402

PATCH, STRIDE, CORE_MARGIN, GRID = 1024, 768, 128, 4
# Run configuration. Windows are always defined in each slide's he2cell/ folder (fixed across runs);
# predictions go to ESCA_RESULT. ESCA_STAIN: reinhard (to the LUAD slide) or none.
WINDOW_DIR = "he2cell"
RESULT = os.environ.get("ESCA_RESULT", "he2cell")
STAIN = os.environ.get("ESCA_STAIN", "reinhard")
RUN_CLASSIFIER = Path(os.environ.get("ESCA_CLASSIFIER", str(CLASSIFIER)))
KEEP_CROPS = os.environ.get("ESCA_KEEP_CROPS", "0") == "1"
CORE = STRIDE * GRID  # 3072 level-0 px
N_TLS, N_RANDOM, N_DENSE = 3, 3, 2
MIN_VALID, MAX_ARTIFACT = 0.5, 0.05
MAX_MPP = 0.30  # classifier operates on native 40x crops; 20x slides are excluded
KEEP = ["global_instance_id", "patch_id", "area_pixels", "centroid_x_global", "centroid_y_global",
        "celltype", "prob_Other", "prob_T", "prob_Macrophage", "prob_B", "confidence",
        "raw_celltype", "raw_confidence"]


REFERENCE_QC = OUT / "hooknet_equivalence_LUAD"  # LUAD example slide, the classifier's validated domain
REFERENCE_SLIDE = ROOT / "data" / "TCGA-55-7725-01Z-00-DX1.4d678777-63b1-4f4a-932a-7fccabf504c7.svs"


def lab_stats(slide_path, qc_folder):
    """LAB mean/std over artifact-free valid tissue at ~4 µm (QC mask from the 8 µm thumbnail)."""
    import cv2
    qc_folder = Path(qc_folder)
    thumb = np.asarray(Image.open(qc_folder / "thumbnail_8um.jpg").convert("RGB"))
    tissue = np.load(qc_folder / "maps_4um.npz")["tissue_mask_2um"][::4, ::4]
    valid = artifact_masks(thumb, tissue)["valid"].astype(np.uint8)
    with open_slide(slide_path) as s:
        mpp = float(s.properties["openslide.mpp-x"])
        level = min(range(s.level_count), key=lambda l: abs(s.level_downsamples[l] * mpp - 4))
        image = np.asarray(s.read_region((0, 0), level, s.level_dimensions[level]).convert("RGB"))
        um_per_px = s.level_downsamples[level] * mpp
    # The thumbnail grid is 8 µm per pixel from the slide origin.
    rows = np.minimum((np.arange(image.shape[0]) * um_per_px / 8).astype(int), valid.shape[0] - 1)
    cols = np.minimum((np.arange(image.shape[1]) * um_per_px / 8).astype(int), valid.shape[1] - 1)
    mask = valid[np.ix_(rows, cols)].astype(bool)
    pixels = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)[mask]
    return pixels.mean(0), pixels.std(0)


def reinhard(rgb, source, target):
    """Reinhard LAB transfer; glass (L >= 215) is left untouched."""
    import cv2
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    background = lab[..., 0] >= 215
    out = (lab - source[0]) / source[1] * target[1] + target[0]
    out[background] = lab[background]
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def seed_for(barcode):
    return int(hashlib.sha256(barcode.encode()).hexdigest()[:8], 16)


def load_hooknet(barcode):
    folder = OUT / "slides" / barcode / "hooknet"
    summary = json.loads((folder / "summary.json").read_text())
    maps = np.load(folder / "maps_4um.npz")
    features = json.loads((folder / "native_polygons.geojson").read_text())["features"]
    return summary, maps, features


def select_windows(barcode, summary, maps, features):
    import cv2
    from scipy import ndimage
    from skimage.color import rgb2hed

    rng = np.random.default_rng(seed_for(barcode))
    width0, height0 = summary["source_dimensions"]
    ratio = width0 / summary["dimensions_at_0p5um"][0]  # level-0 px per 0.5 µm px
    qc = slide_qc(OUT / "slides" / barcode)
    valid8, artifact8 = qc["valid"], qc["artifact"]
    to8 = 1 / (16 * ratio)  # level-0 px -> 8 µm px
    size8 = int(round(CORE * to8))

    def window_box(cx, cy):
        return int(round(cx - CORE / 2)), int(round(cy - CORE / 2))

    def inside(x0, y0):
        return (x0 - CORE_MARGIN >= 0 and y0 - CORE_MARGIN >= 0
                and x0 + CORE + CORE_MARGIN <= width0 and y0 + CORE + CORE_MARGIN <= height0)

    def fraction(mask, x0, y0):
        a = mask[int(y0 * to8):int((y0 + CORE) * to8), int(x0 * to8):int((x0 + CORE) * to8)]
        return float(a.mean()) if a.size else 0.0

    def acceptable(x0, y0):
        return inside(x0, y0) and free(x0, y0) and fraction(valid8, x0, y0) >= MIN_VALID and fraction(artifact8, x0, y0) <= MAX_ARTIFACT

    chosen, skipped = [], []

    def free(x0, y0):
        return all(abs(x0 - w["x0"]) >= CORE or abs(y0 - w["y0"]) >= CORE for w in chosen)

    def add(category, x0, y0, **extra):
        chosen.append({"window_id": f"{category}_{sum(w['category'] == category for w in chosen) + 1}",
                       "category": category, "x0": x0, "y0": y0, "valid_tissue_fraction": fraction(valid8, x0, y0),
                       "artifact_fraction": fraction(artifact8, x0, y0), **extra})

    retained = [f for f in features if f["properties"]["predicted_label"] == "tls" and f["properties"]["passes_published_filter"]]
    for index in rng.permutation(len(retained)):
        if sum(w["category"] == "tls" for w in chosen) == N_TLS:
            break
        # Window centre = the polygon's most interior point (4 µm raster), robust to concave shapes.
        ring = np.asarray(retained[index]["geometry"]["coordinates"][0], dtype=float) / 8
        origin = np.floor(ring.min(axis=0)) - 2
        canvas = np.zeros((int(np.ptp(ring[:, 1])) + 6, int(np.ptp(ring[:, 0])) + 6), np.uint8)
        cv2.fillPoly(canvas, [np.round(ring - origin).astype(np.int32)], 1)
        py_all, px_all = np.nonzero(canvas)
        qy = np.clip(((py_all + origin[1]) // 2).astype(int), 0, artifact8.shape[0] - 1)
        qx = np.clip(((px_all + origin[0]) // 2).astype(int), 0, artifact8.shape[1] - 1)
        region_id = retained[index]["properties"]["region_id"]
        if (artifact8[qy, qx] | ~valid8[qy, qx]).mean() > .25:
            skipped.append(region_id)  # HookNet "TLS" on ink, folds or non-tissue
            continue
        distance = cv2.distanceTransform(canvas, cv2.DIST_L2, 5)
        py, px = np.unravel_index(np.argmax(distance), distance.shape)
        x0, y0 = window_box((px + origin[0] + .5) * 8 * ratio, (py + origin[1] + .5) * 8 * ratio)
        if inside(x0, y0) and free(x0, y0) and fraction(artifact8, x0, y0) <= MAX_ARTIFACT:
            add("tls", x0, y0, anchor=region_id)

    ys, xs = np.nonzero(valid8)
    for _ in range(3000):
        if sum(w["category"] == "random" for w in chosen) == N_RANDOM or not len(xs):
            break
        i = rng.integers(len(xs))
        x0, y0 = window_box((xs[i] + .5) / to8, (ys[i] + .5) / to8)
        if acceptable(x0, y0):
            add("random", x0, y0)

    # Hard negatives: densest mean haematoxylin over valid tissue, no HookNet TLS/GC pixel within 100 µm.
    thumb = np.asarray(Image.open(OUT / "slides" / barcode / "hooknet" / "thumbnail_8um.jpg").convert("RGB"))
    hema = np.clip(rgb2hed(thumb)[..., 0], 0, None) * valid8
    density = ndimage.uniform_filter(hema, size8)
    tls_pixels = np.isin(maps["prediction_mask"], [2, 3])  # 4 µm grid
    near8 = ndimage.binary_dilation(tls_pixels, iterations=25)[::2, ::2]
    blocked = np.zeros(density.shape, bool)
    h, w = min(blocked.shape[0], near8.shape[0]), min(blocked.shape[1], near8.shape[1])
    blocked[:h, :w] = ndimage.maximum_filter(near8[:h, :w], size8)
    density = np.where(blocked, -np.inf, density)
    for flat in np.argsort(density, axis=None)[::-1][:50000]:
        if sum(w["category"] == "dense" for w in chosen) == N_DENSE:
            break
        y8, x8 = np.unravel_index(flat, density.shape)
        if not np.isfinite(density[y8, x8]):
            break
        x0, y0 = window_box((x8 + .5) / to8, (y8 + .5) / to8)
        if acceptable(x0, y0):
            add("dense", x0, y0, hematoxylin_window_mean=float(density[y8, x8]))
    save_json({"artifact_flagged_hooknet_tls": skipped, "retained_hooknet_tls": len(retained)},
              OUT / "slides" / barcode / WINDOW_DIR / "window_selection.json")
    windows = pd.DataFrame(chosen, columns=["window_id", "category", "x0", "y0", "valid_tissue_fraction",
                                             "artifact_fraction", "anchor", "hematoxylin_window_mean"])
    windows["x1"], windows["y1"] = windows.x0 + CORE, windows.y0 + CORE
    windows["level0_per_0p5um"] = ratio
    return windows


def stage(script, *arguments, log):
    env = {**os.environ, "CELLPOSE_LOCAL_MODELS_PATH": str(ROOT / "models" / "cellpose_cache"),
           "MPLCONFIGDIR": str(ROOT / ".runtime" / "matplotlib"), "OMP_NUM_THREADS": "8"}
    started = time.perf_counter()
    with open(log, "a") as handle:
        subprocess.run([sys.executable, str(SRC / script), *map(str, arguments)], check=True,
                       stdout=handle, stderr=subprocess.STDOUT, env=env)
    return time.perf_counter() - started


def process(slide, barcode):
    out = OUT / "slides" / barcode / RESULT
    window_dir = OUT / "slides" / barcode / WINDOW_DIR
    if (out / "cells.csv.gz").exists():
        return "done"
    summary, maps, features = load_hooknet(barcode)
    if summary["source_mpp"] > MAX_MPP:
        save_json({"status": "excluded", "reason": f"MPP {summary['source_mpp']} > {MAX_MPP} (not 40x)"}, out / "excluded.json")
        return "excluded"
    started = time.perf_counter()
    work = OUT / f"_tmp_{RESULT}" / barcode
    shutil.rmtree(work, ignore_errors=True)
    (work / "patches").mkdir(parents=True)
    out.mkdir(parents=True, exist_ok=True)
    window_dir.mkdir(parents=True, exist_ok=True)
    if (window_dir / "windows.csv").exists():
        windows = pd.read_csv(window_dir / "windows.csv")
    else:
        windows = select_windows(barcode, summary, maps, features)
        if windows.empty:
            save_json({"status": "excluded", "reason": "no window passes tissue/artifact QC"}, out / "excluded.json")
            return "excluded (no QC-passing window)"
        windows.to_csv(window_dir / "windows.csv", index=False)
    if STAIN == "reinhard":
        source = lab_stats(slide, OUT / "slides" / barcode / "hooknet")
        target = lab_stats(REFERENCE_SLIDE, REFERENCE_QC)
    rows = []
    with open_slide(slide) as handle:
        level = min(range(handle.level_count), key=lambda l: abs(handle.level_downsamples[l] - 4))
        for w in windows.itertuples():
            # Context image for review (~1 µm/px), not used by any model.
            down = handle.level_downsamples[level]
            if not (window_dir / f"{w.window_id}.jpg").exists():
                preview = handle.read_region((w.x0, w.y0), level, (int(CORE / down), int(CORE / down))).convert("RGB")
                preview.save(window_dir / f"{w.window_id}.jpg", quality=90)
            for gy in range(GRID):
                for gx in range(GRID):
                    x, y = w.x0 - CORE_MARGIN + gx * STRIDE, w.y0 - CORE_MARGIN + gy * STRIDE
                    patch_id = len(rows)
                    path = work / "patches" / f"{w.window_id}_patch{patch_id}_x{x}_y{y}.png"
                    patch = np.asarray(handle.read_region((x, y), 0, (PATCH, PATCH)).convert("RGB"))
                    Image.fromarray(reinhard(patch, source, target) if STAIN == "reinhard" else patch).save(path)
                    rows.append({"path": str(path), "num_patch": patch_id, "sample": slide.stem, "source_tif": str(slide),
                                 "x": x, "y": y, "patch_size": PATCH, "stride": STRIDE, "level": 0,
                                 "tissue_ratio": w.valid_tissue_fraction, "window_id": w.window_id})
    patch_table = pd.DataFrame(rows)
    patch_csv = work / f"{slide.stem}_patch_index.csv"
    patch_table.to_csv(patch_csv, index=False)
    log = out / "pipeline.log"
    seg = stage("run_cellcrops.py", "--patch_csv", patch_csv, "--out_dir", work / "cells", "--model_name", "cpsam_v2",
                "--device", "mps", "--num_gpus", 1, "--batch_size", 16, "--crop_size", 64, log=log)
    cells_csv = work / "cells" / "mask_tables" / f"{patch_csv.stem}_cellpose_mask_instances.csv"
    result_csv = work / "cell_types.csv"
    cls = stage("model_eval.py", "--meta_path", cells_csv, "--save_path", RUN_CLASSIFIER, "--output_path", result_csv,
                "--device", "mps", "--batch_size", 256, "--num_workers", 0, "--tta", "d4", log=log)
    cells = pd.read_csv(result_csv)
    keep = KEEP + (["cell_image_crop_path", "cell_mask_crop_path"] if KEEP_CROPS else [])
    cells = cells[keep].merge(patch_table[["num_patch", "window_id"]].rename(columns={"num_patch": "patch_id"}), on="patch_id")
    cells.to_csv(out / "cells.csv.gz", index=False)
    save_json({"slide": str(slide), "barcode": barcode, "windows": len(windows), "patches": len(rows), "cells": len(cells),
               "counts": cells.celltype.value_counts().to_dict(), "segmentation_seconds": seg, "classification_seconds": cls,
               "wall_seconds": time.perf_counter() - started, "patch_size": PATCH, "stride": STRIDE,
               "classifier": str(RUN_CLASSIFIER), "segmentation_model": "cpsam_v2", "tta": "d4",
               "stain_normalisation": {"method": "Reinhard LAB, per slide, glass untouched", "reference": "TCGA-55-7725 LUAD example slide",
                                       "source_LAB_mean_std": [source[0].tolist(), source[1].tolist()],
                                       "target_LAB_mean_std": [target[0].tolist(), target[1].tolist()]} if STAIN == "reinhard" else "none"},
              out / "summary.json")
    if KEEP_CROPS:
        # Keep 64-px crops (re-scoring needs no segmentation); drop patches and patch label images.
        crops = out / "crops"
        shutil.rmtree(crops, ignore_errors=True)
        shutil.move(str(work / "cells" / "single_cells"), crops)
        cells["cell_image_crop_path"] = cells.cell_image_crop_path.str.replace(str(work / "cells" / "single_cells"), str(crops), regex=False)
        cells["cell_mask_crop_path"] = cells.cell_mask_crop_path.str.replace(str(work / "cells" / "single_cells"), str(crops), regex=False)
        cells.to_csv(out / "cells.csv.gz", index=False)
    shutil.rmtree(work, ignore_errors=True)
    return f"{len(windows)} windows, {len(cells)} cells, {time.perf_counter() - started:.0f}s"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--limit", type=int, help="Stop after this many slides (smoke test)")
    args = parser.parse_args()
    clinical = pd.read_csv(OUT / "rna" / "clinical.csv").set_index("case_id")
    expected = sum(1 for line in (OUT / "tar_listing.txt").read_text().splitlines() if line.endswith(".svs"))
    processed = 0
    while True:
        finished_hooknet = 0
        for summary_path in sorted((OUT / "slides").glob("*/hooknet/summary.json"), key=lambda p: p.stat().st_mtime):
            finished_hooknet += 1
            barcode = summary_path.parents[1].name
            slide = Path(json.loads(summary_path.read_text())["slide"])
            out = OUT / "slides" / barcode / RESULT
            if (out / "cells.csv.gz").exists() or (out / "excluded.json").exists() or (out / "failed.json").exists():
                continue
            diagnosis = clinical.primary_diagnosis.get(barcode[:12], "Not Reported")
            if diagnosis in EXCLUDED_DIAGNOSES:
                save_json({"status": "excluded", "reason": diagnosis}, out / "excluded.json")
                continue
            lock = OUT / "_locks" / barcode
            lock.parent.mkdir(exist_ok=True)
            try:
                lock.mkdir()  # atomic: lets several workers share the GPU without duplicating slides
            except FileExistsError:
                continue
            try:
                print(f"{barcode}: {process(slide, barcode)}", flush=True)
            except Exception as error:
                save_json({"status": "failed", "error": repr(error)}, out / "failed.json")
                print(f"{barcode}: FAILED {error!r}", flush=True)
            finally:
                lock.rmdir()
            processed += 1
            if args.limit and processed >= args.limit:
                return
        failed = len(list((OUT / "slides").glob("*/hooknet/failed.json")))
        if not args.wait or finished_hooknet + failed >= expected:
            break
        time.sleep(120)


if __name__ == "__main__":
    main()
