"""Stream published HookNet-TLS over every extracted TCGA-ESCA slide (independent H&E reference).

Same physical resampling (0.5 µm input, 2.0 µm context), Otsu tissue mask, 1244/1024 tile grid
and authored postprocessing as deployment/run_hooknet_tls.py. Differences are I/O only: the
0.5 µm image and full-resolution maps live in a temporary folder that is deleted after each
slide, and only 4 µm maps, polygons and summaries are kept.

Run with .runtime/hooknet_tls/venv/bin/python. Waits for slides still being extracted.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import openslide
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes, remove_small_objects

from esca_common import OUT, RAW, save_json, slide_barcode
from hooknet_tls_native import load_model
from run_hooknet_tls import infer_and_verify, padded_patch, physical_remap_grid, postprocess

TMP_ROOT = OUT / "_tmp_hooknet"


def expected_sizes():
    """Member sizes from the tar listing, to recognise fully extracted slides."""
    sizes = {}
    for line in (OUT / "tar_listing.txt").read_text().splitlines():
        parts = line.split()
        if parts and parts[-1].endswith(".svs"):
            sizes[Path(parts[-1]).name] = int(parts[4])
    return sizes


def read_strip(slide_path, row, height, width, ratio_x, ratio_y):
    """One 1024-row strip of the lossless physical 0.5 µm image (as prepare_slide)."""
    with openslide.OpenSlide(str(slide_path)) as slide:
        h = min(1024, height - row)
        strip = np.empty((h, width, 3), np.uint8)
        for col in range(0, width, 1024):
            w = min(1024, width - col)
            sx, sy = max(0, int(col * ratio_x) - 1), max(0, int(row * ratio_y) - 1)
            sw = int(np.ceil((col + w) * ratio_x)) - sx + 1
            sh = int(np.ceil((row + h) * ratio_y)) - sy + 1
            source = np.asarray(slide.read_region((sx, sy), 0, (sw, sh)))[..., :3]
            map_x, map_y = physical_remap_grid(w, h, col, row, ratio_x, ratio_y, sx, sy)
            strip[:, col:col + w] = cv2.remap(source, map_x, map_y, cv2.INTER_LINEAR)
    return row, strip


def prepare(slide_path, tmp):
    started = time.perf_counter()
    with openslide.OpenSlide(str(slide_path)) as slide:
        mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
        if not np.isclose(mpp, float(slide.properties[openslide.PROPERTY_NAME_MPP_Y])):
            raise ValueError("Anisotropic pixels")
        dims = slide.dimensions
        magnification = slide.properties.get("aperio.AppMag")
    width, height = (int(round(d * mpp / .5)) for d in dims)
    ratio_x, ratio_y = dims[0] / width, dims[1] / height
    base = np.lib.format.open_memmap(tmp / "RGB_0p5um.npy", mode="w+", dtype=np.uint8, shape=(height, width, 3))
    with ThreadPoolExecutor(max_workers=12) as pool:
        for row, strip in pool.map(lambda r: read_strip(slide_path, r, height, width, ratio_x, ratio_y),
                                   range(0, height, 1024)):
            base[row:row + strip.shape[0]] = strip
    base.flush()
    # Successive 2x bilinear reductions, computed on 1024-row (even) strips: identical to whole-image resize.
    pyramid = [base]
    previous = base
    for level in range(1, 5):
        out_shape = ((previous.shape[0] + 1) // 2, (previous.shape[1] + 1) // 2, 3)
        if level == 1:
            reduced = np.empty(out_shape, np.uint8)
            for row in range(0, previous.shape[0], 2048):
                chunk = np.asarray(previous[row:row + 2048])
                reduced[row // 2:row // 2 + (chunk.shape[0] + 1) // 2] = cv2.resize(
                    chunk, ((chunk.shape[1] + 1) // 2, (chunk.shape[0] + 1) // 2), interpolation=cv2.INTER_LINEAR)
        else:
            reduced = cv2.resize(previous, (out_shape[1], out_shape[0]), interpolation=cv2.INTER_LINEAR)
        pyramid.append(reduced)
        previous = reduced
    gray = rgb2gray(pyramid[4])
    threshold = float(threshold_otsu(gray))
    mask8 = remove_small_holes(remove_small_objects(gray < threshold, min_size=64), area_threshold=64)
    mask2 = np.repeat(np.repeat(mask8, 4, axis=0), 4, axis=1)[:pyramid[2].shape[0], :pyramid[2].shape[1]].astype(np.uint8)
    info = {"source_mpp": mpp, "source_dimensions": list(dims), "aperio_AppMag": magnification,
            "dimensions_at_0p5um": [width, height], "Otsu_gray_threshold": threshold,
            "tissue_area_mm2": float(mask8.sum() * 64 / 1e6), "preparation_seconds": time.perf_counter() - started}
    return pyramid, mask2, info


def mask_patch(mask2, x, y, shape, size=1244):
    """Padded 0.5 µm tissue-mask patch; the published mask is mask2 upsampled 4x and cropped to shape."""
    ys, xs = np.arange(y - size // 2, y - size // 2 + size), np.arange(x - size // 2, x - size // 2 + size)
    valid = ((ys >= 0) & (ys < shape[0]))[:, None] & ((xs >= 0) & (xs < shape[1]))[None, :]
    rows = np.clip(ys // 4, 0, mask2.shape[0] - 1)
    cols = np.clip(xs // 4, 0, mask2.shape[1] - 1)
    return mask2[rows][:, cols] * valid


def infer(model, function, device, pyramid, mask2, shape, tmp):
    import tensorflow as tf
    maps = {name: np.lib.format.open_memmap(tmp / f"{name}.npy", mode="w+", dtype=np.uint8, shape=shape)
            for name in ["prediction_mask", "TLS_confidence", "GC_confidence"]}
    jobs = []
    for y in range(-512, shape[0], 1024):
        for x in range(-512, shape[1], 1024):
            if x - 512 < 0 or y - 512 < 0:
                continue
            if mask_patch(mask2, x, y, shape).any():
                jobs.append((x, y))
    started = time.perf_counter()
    for x, y in jobs:
        high = padded_patch(pyramid[0], x, y, 1244)[None].astype(np.float32)
        low = padded_patch(pyramid[2], x // 4, y // 4, 1244)[None].astype(np.float32)
        with tf.device(f"/{device}:0"):
            probabilities = function(high, low).numpy().reshape(model.output_shape)
        if not np.isfinite(probabilities).all():
            raise ValueError("Non-finite model output")
        local_mask = mask_patch(mask2, x, y, shape)[107:-107, 107:-107][:1024, :1024]
        values = {"prediction_mask": ((probabilities.argmax(-1) + 1)[:1024, :1024] * local_mask).astype(np.uint8),
                  "TLS_confidence": (probabilities[:1024, :1024, 1] * 255 * local_mask).astype(np.uint8),
                  "GC_confidence": (probabilities[:1024, :1024, 2] * 255 * local_mask).astype(np.uint8)}
        col, row = x - 512, y - 512
        h, w = min(1024, shape[0] - row), min(1024, shape[1] - col)
        for name, value in values.items():
            maps[name][row:row + h, col:col + w] = value[:h, :w]
    for array in maps.values():
        array.flush()
    return maps, {"tissue_tiles": len(jobs), "inference_seconds": time.perf_counter() - started}


def process(slide_path, model, config, state, out=None):
    # Non-TCGA slides (e.g. Visium validation) are named by their file stem and need an explicit out.
    barcode = slide_barcode(slide_path.name) if out is None else slide_path.stem
    out = out or OUT / "slides" / barcode / "hooknet"
    if (out / "summary.json").exists():
        return
    (out / "source_audit").mkdir(parents=True, exist_ok=True)
    tmp = TMP_ROOT / barcode
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    started = time.perf_counter()
    try:
        pyramid, mask2, info = prepare(slide_path, tmp)
        shape = pyramid[0].shape[:2]
        if "function" not in state:
            # One CPU/GPU agreement check per process on a real tissue tile (authored tolerance).
            ys, xs = np.nonzero(mask2)
            center = [int(np.median(xs)) * 4, int(np.median(ys)) * 4]
            state["function"], state["device"] = infer_and_verify(model, pyramid[0], pyramid[2], center, OUT / "hooknet_meta", "auto")
        maps, timings = infer(model, state["function"], state["device"], pyramid, mask2, shape, tmp)
        native = postprocess(maps, config, out)
        np.savez_compressed(out / "maps_4um.npz", **{name: np.asarray(array[::8, ::8]) for name, array in maps.items()},
                            tissue_mask_2um=mask2)
        cv2.imwrite(str(out / "thumbnail_8um.jpg"), cv2.cvtColor(pyramid[4], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        save_json({"slide": str(slide_path), "barcode": barcode, **info, **timings, "native": native,
                   "device": state["device"], "wall_seconds": time.perf_counter() - started,
                   "coordinate_units": "polygons in pixels at 0.5 µm; maps_4um are [::8] of 0.5 µm maps"},
                  out / "summary.json")
        print(f"{barcode}: {native['filtered_TLS_polygons']} TLS / {native['filtered_GC_polygons']} GC, "
              f"{info['tissue_area_mm2']:.0f} mm2, {time.perf_counter() - started:.0f}s", flush=True)
    except Exception as error:
        save_json({"slide": str(slide_path), "status": "failed", "error": repr(error)}, out / "failed.json")
        import traceback
        traceback.print_exc()
        print(f"{barcode}: FAILED {error!r}", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true", help="Keep polling until every listed slide is processed")
    parser.add_argument("--slide", type=Path, help="Process one slide (equivalence check) into --out")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(16)
    with tf.device("/CPU:0"):
        model, config, _ = load_model(OUT / "hooknet_meta" / "source_audit")
    state = {}
    if args.slide:
        process(args.slide.resolve(), model, config, state, args.out.resolve())
        return
    sizes = expected_sizes()
    while True:
        pending = 0
        for name, size in sorted(sizes.items()):
            matches = list(RAW.glob(f"*/{name}"))
            if not matches or matches[0].stat().st_size != size:
                pending += 1
                continue
            process(matches[0], model, config, state)
        if not args.wait or pending == 0:
            break
        time.sleep(60)
    print("HookNet-TLS cohort pass complete", flush=True)


if __name__ == "__main__":
    main()
