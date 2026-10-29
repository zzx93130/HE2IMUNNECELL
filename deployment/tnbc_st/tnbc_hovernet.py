"""HoVer-Net (PanNuke, TIAToolbox 1.6.0 hovernet_fast-pannuke) on the TNBC_ST subset arrays, same pixels as HE2CELL.

The imagesHD array is resampled once onto the 0.25 µm grid with the same bicubic affine mapping that
visium_cells.patch_grid_image applies per patch (so every HE2CELL patch is a crop of it, up to interpolation
within 1-2 px of a patch's native-region border, i.e. outside its kept core). HoVer-Net then runs on 3328-px
blocks of 4x4 HE2CELL patches (as for ESCA windows: one image per block, single post-processing tile) and a
nucleus is kept only if its centroid lies in the core of a patch HE2CELL processed. Work units are blocks,
claimed with mkdir locks (several CPU workers share an array); the last worker assembles the array table.
Outputs: outputs/tnbc_st_20261101/baselines/hovernet/<array>.csv.gz (centroid_*_global in the 0.25 µm frame,
x_hd/y_hd, type, type_name, type_prob, area_um2). Run with .runtime/tiatoolbox/venv/bin/python.
"""
from __future__ import annotations

import argparse
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
sys.path.insert(0, str(HERE.parent / "baselines"))
import tnbc_common as C  # noqa: E402
import hovernet_cells as HN  # noqa: E402

OUT = C.OUT / "baselines" / "hovernet"
BLOCK = 4
TARGET_MPP = 0.25


def resampled(entry, hd_mpp):
    image = C.read_image(entry["hd"])
    scale = hd_mpp / TARGET_MPP
    size = int(image.shape[0] * scale)
    # Same mapping as patch_grid_image: target px t <-> native (t / scale), bicubic, white border.
    matrix = np.float32([[scale, 0, 0], [0, scale, 0]])
    return cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


def run(array, entry, hd_mpp, segment):
    final = OUT / f"{array}.csv.gz"
    if final.exists():
        return
    patches = pd.read_csv(C.OUT / "cells" / array / "patch_index.csv")  # x, y of every HE2CELL patch
    V = HN.V
    parts = OUT / "_parts" / array
    parts.mkdir(parents=True, exist_ok=True)
    span = V.STRIDE * BLOCK
    patches["bx"], patches["by"] = (patches.x + V.MARGIN) // span, (patches.y + V.MARGIN) // span
    blocks = sorted(set(zip(patches.bx, patches.by)))
    image = None
    for bx, by in blocks:
        name = f"{array}_{bx:03d}_{by:03d}"
        if (parts / f"{name}.csv.gz").exists() or not HN.claim(name):
            continue
        try:
            if image is None:
                image = resampled(entry, hd_mpp)
            x0, y0 = bx * span - V.MARGIN, by * span - V.MARGIN  # block origin = origin of its top-left patch
            size = V.MARGIN + span + (V.PATCH - V.STRIDE - V.MARGIN)
            block = np.full((size, size, 3), 255, np.uint8)
            ax, ay = max(x0, 0), max(y0, 0)
            bx1, by1 = min(x0 + size, image.shape[1]), min(y0 + size, image.shape[0])
            block[ay - y0:by1 - y0, ax - x0:bx1 - x0] = image[ay:by1, ax:bx1]
            started = time.perf_counter()
            (frame,) = segment([block], (0, size))
            frame = frame.assign(centroid_x_global=frame.cx + x0, centroid_y_global=frame.cy + y0)
            # keep nuclei whose centroid lies in the 768-px core of a processed patch of this block
            mine = patches[(patches.bx == bx) & (patches.by == by)]
            keep = np.zeros(len(frame), bool)
            for p in mine.itertuples():
                keep |= ((frame.centroid_x_global >= p.x + V.MARGIN) & (frame.centroid_x_global < p.x + V.MARGIN + V.STRIDE)
                         & (frame.centroid_y_global >= p.y + V.MARGIN) & (frame.centroid_y_global < p.y + V.MARGIN + V.STRIDE)).to_numpy()
            frame = frame[keep]
            frame["type_name"] = frame.type.map(HN.TYPES)
            HN.save_atomic(frame[["centroid_x_global", "centroid_y_global", "type", "type_name", "type_prob", "area_um2"]],
                           parts / f"{name}.csv.gz")
            print(f"{time.strftime('%F %T')} {name}: {len(frame)} nuclei, {time.perf_counter() - started:.0f}s", flush=True)
        except Exception as error:  # noqa: BLE001
            print(f"{time.strftime('%F %T')} {name}: FAILED {error!r}", flush=True)
        finally:
            HN.release(name)
    if all((parts / f"{array}_{bx:03d}_{by:03d}.csv.gz").exists() for bx, by in blocks) and HN.claim(f"{array}_final"):
        try:
            if not final.exists():
                cells = pd.concat([pd.read_csv(parts / f"{array}_{bx:03d}_{by:03d}.csv.gz") for bx, by in blocks], ignore_index=True)
                factor = hd_mpp / TARGET_MPP
                cells["x_hd"], cells["y_hd"] = cells.centroid_x_global / factor, cells.centroid_y_global / factor
                HN.save_atomic(cells, final)
                shutil.rmtree(parts, ignore_errors=True)
                print(f"{array}: {len(cells)} nuclei", flush=True)
        finally:
            HN.release(f"{array}_final")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", nargs="+", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--wait", action="store_true", help="keep polling until every listed array is done")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    HN.OUT = OUT            # Segmentor temp folder -> OUT/_tmp (not the ESCA/Visium baseline folder)
    HN.CELLS = OUT          # locks and provenance.json here
    (OUT / "_tmp").mkdir(parents=True, exist_ok=True)
    segment = HN.Segmentor(args.device, 8, 2)
    inventory = json.loads((C.OUT / "inventory.json").read_text())["arrays"]
    hd_mpp = json.loads((C.OUT / "scale.json").read_text())["hd_mpp"]
    import time as _t
    while True:
        pending = [a for a in args.arrays if not (OUT / f"{a}.csv.gz").exists()]
        if not pending:
            break
        for array in pending:
            if (C.OUT / "cells" / array / "cells.csv.gz").exists():  # wait for HE2CELL's patch list
                run(array, inventory[array], hd_mpp, segment)
        if not args.wait:
            break
        _t.sleep(60)
    print("HoVer-Net worker finished", flush=True)


if __name__ == "__main__":
    main()
