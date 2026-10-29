"""Gate-A diagnostic (not used for any reported metric): are HE2CELL's TNBC calls scale-sensitive?

Takes a 3x3-patch block centred on given H&E µm coordinates of one array and runs the unchanged pipeline
(cpsam_v2 + he2cell_esca_finetuned.pth, D4, prior) with the imagesHD pixel size assumed to be each value in
--mpps (so the 0.25 µm/px input is rescaled accordingly). Reports class shares per assumption.
Writes outputs/tnbc_st_20261101/qc/stage0/scale_probe_<array>.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "visium_tls"))
import tnbc_common as C  # noqa: E402
import tnbc_cells as TC  # noqa: E402
import visium_cells as V  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--array", required=True)
    parser.add_argument("--x-um", type=float, required=True)
    parser.add_argument("--y-um", type=float, required=True)
    parser.add_argument("--mpps", type=float, nargs="+", default=[0.235, 0.2673, 0.30])
    parser.add_argument("--ref-mpp", type=float, default=0.2673, help="mpp used to convert the µm centre to HD px")
    args = parser.parse_args()
    entry = json.loads((C.OUT / "inventory.json").read_text())["arrays"][args.array]
    image = C.read_image(entry["hd"])
    cx, cy = args.x_um / args.ref_mpp, args.y_um / args.ref_mpp
    result = {}
    for mpp in args.mpps:
        work = C.OUT / "_tmp_probe" / f"{args.array}_{mpp}"
        shutil.rmtree(work, ignore_errors=True)
        (work / "patches").mkdir(parents=True)
        scale = mpp / 0.25
        # 3x3 patches around the centre on the target grid
        tx, ty = cx * scale, cy * scale
        x0 = int(tx - 1.5 * V.STRIDE - V.MARGIN)
        y0 = int(ty - 1.5 * V.STRIDE - V.MARGIN)
        rows = []
        for j in range(3):
            for i in range(3):
                x, y = x0 + i * V.STRIDE, y0 + j * V.STRIDE
                # same affine mapping as visium_cells.patch_grid_image
                xs, ys = x / scale - 1, y / scale - 1
                sx, sy = int(np.floor(xs)), int(np.floor(ys))
                size = int(np.ceil(V.PATCH / scale)) + 3
                region = image[sy:sy + size, sx:sx + size]
                matrix = np.float32([[scale, 0, -(xs - sx + 1) * scale], [0, scale, -(ys - sy + 1) * scale]])
                patch = cv2.warpAffine(region, matrix, (V.PATCH, V.PATCH), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
                path = work / "patches" / f"p{len(rows)}.png"
                cv2.imwrite(str(path), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
                rows.append({"path": str(path), "num_patch": len(rows), "sample": args.array, "source_tif": args.array,
                             "x": x, "y": y, "patch_size": V.PATCH, "stride": V.STRIDE, "level": 0, "tissue_ratio": 1.0})
        csv = work / "index.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)
        log = work / "log.txt"
        V.stage("run_cellcrops.py", "--patch_csv", csv, "--out_dir", work / "cells", "--model_name", "cpsam_v2",
                "--device", "mps", "--num_gpus", 1, "--batch_size", 16, "--crop_size", 64, log=log)
        cells_csv = work / "cells" / "mask_tables" / f"{csv.stem}_cellpose_mask_instances.csv"
        out = work / "types.csv"
        V.stage("model_eval.py", "--meta_path", cells_csv, "--save_path", TC.CHECKPOINT, "--output_path", out,
                "--device", "mps", "--batch_size", 256, "--num_workers", 0, "--tta", "d4", log=log)
        t = pd.read_csv(out)
        result[str(mpp)] = {"cells": len(t), "shares": t.celltype.value_counts(normalize=True).round(4).to_dict(),
                            "median_area_um2": float(np.median(t.area_pixels) * 0.25 ** 2)}
        print(mpp, result[str(mpp)], flush=True)
        shutil.rmtree(work, ignore_errors=True)
    C.save_json(result, C.OUT / "qc" / "stage0" / f"scale_probe_{args.array}.json")


if __name__ == "__main__":
    main()
