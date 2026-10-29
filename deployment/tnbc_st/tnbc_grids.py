"""Per-array 8 µm grids on the H&E frame (shared by all Stage-4 analyses).

Frame: µm from the imagesHD origin (x_um = x_hd * hd_mpp); grid cell (i, j) covers [8j, 8j+8) x [8i, 8i+8) µm.
Holds HE2CELL class counts, HoVer-Net inflammatory counts, HookNet-TLS masks/confidence, the tissue mask,
pathologist annotation labels and the valid region (preregistration.json A.valid_region).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "esca_tls"))
import tnbc_common as C  # noqa: E402
import tnbc_cells as TC  # noqa: E402

GRID = 8.0
HD, LARGE = 31744, 9523
NAMES = list(TC.ANNOTATION_COLOURS)
LN, LYMPHOCYTE, HIGH_TIL = NAMES.index("Lymphoid nodule"), NAMES.index("Lymphocyte"), NAMES.index("High TIL stroma")
EXCLUDE = [NAMES.index("Nothing"), NAMES.index("Artefacts"), NAMES.index("Hole (whitespace)"), -1]


@lru_cache(1)
def scale():
    return json.loads((C.OUT / "scale.json").read_text())


@lru_cache(1)
def inventory():
    return json.loads((C.OUT / "inventory.json").read_text())["arrays"]


def size():
    return int(np.ceil(HD * scale()["hd_mpp"] / GRID))


def resize_to_grid(image, n):
    """Nearest-neighbour sample of a full-FOV raster (any resolution, spanning the imagesHD field) at the n x n cell centres."""
    fov_um = HD * scale()["hd_mpp"]
    h, w = image.shape[:2]
    centres_um = (np.arange(n) + .5) * GRID
    rows = np.clip((centres_um / (fov_um / h)).astype(int), 0, h - 1)
    cols = np.clip((centres_um / (fov_um / w)).astype(int), 0, w - 1)
    return image[np.ix_(rows, cols)]


def counts_grid(x_um, y_um, n, weights=None):
    gx = np.clip((np.asarray(x_um) / GRID).astype(int), 0, n - 1)
    gy = np.clip((np.asarray(y_um) / GRID).astype(int), 0, n - 1)
    return np.bincount(gy * n + gx, weights=weights, minlength=n * n).reshape(n, n).astype(float)


def load(array, hovernet=True, hooknet=True):
    entry = inventory()[array]
    n = size()
    mpp = scale()["hd_mpp"]
    out = {"array": array, "sample": entry["sample"], "case_id": f"TNBC{entry['sample']}", "n": n}
    cells = pd.read_csv(C.OUT / "cells" / array / "cells.csv.gz",
                        usecols=["x_hd", "y_hd", "celltype", "prob_B", "prob_T", "area_um2"])
    x, y = cells.x_hd.to_numpy() * mpp, cells.y_hd.to_numpy() * mpp
    out["cells_xy_um"], out["celltype"] = np.stack([x, y], 1), cells.celltype.to_numpy()
    out["counts"] = {k: counts_grid(x[cells.celltype.to_numpy() == k], y[cells.celltype.to_numpy() == k], n)
                     for k in ["B", "T", "Other", "Macrophage"]}
    tissue = cv2.imread(str(C.OUT / "cells" / array / "tissue_mask.png"), 0) > 0
    # tissue_mask.png is the imagesLarge/4 grid (2380 px for 9520 of the 9523 Large px)
    pad = np.zeros((LARGE // 4 + 1, LARGE // 4 + 1), bool)
    pad[:tissue.shape[0], :tissue.shape[1]] = tissue
    out["tissue"] = resize_to_grid(pad, n)
    if entry.get("annotation"):
        labels = TC.annotation_labels(entry)
        out["labels"] = resize_to_grid(labels, n)
        out["valid"] = out["tissue"] & ~np.isin(out["labels"], EXCLUDE)
    else:
        out["labels"] = None
        out["valid"] = out["tissue"].copy()
    hn = C.OUT / "baselines" / "hovernet" / f"{array}.csv.gz"
    if hovernet and hn.exists():
        h = pd.read_csv(hn)
        inflammatory = (h.type == 2).to_numpy()
        out["hovernet"] = {"inflammatory": counts_grid(h.x_hd.to_numpy()[inflammatory] * mpp, h.y_hd.to_numpy()[inflammatory] * mpp, n),
                           "all": counts_grid(h.x_hd.to_numpy() * mpp, h.y_hd.to_numpy() * mpp, n)}
    hk = C.OUT / "hooknet" / array
    if hooknet and (hk / "summary.json").exists():
        maps = np.load(hk / "maps_4um.npz")
        conf = maps["TLS_confidence"].astype(np.float32) / 255
        pred = maps["prediction_mask"]
        # maps_4um pixel k covers [4k, 4k+4) µm (0.5 µm frame [::8]); grid cell j centre (j+.5)*8 µm -> pixel 2j+1
        # max-pool the 2x2 block of 4 µm pixels inside each 8 µm cell
        c = np.zeros((n, n), np.float32)
        raw = np.zeros((n, n), bool)
        for dy in (0, 1):
            for dx in (0, 1):
                yy = np.minimum(2 * np.arange(n) + dy, conf.shape[0] - 1)
                xx = np.minimum(2 * np.arange(n) + dx, conf.shape[1] - 1)
                c = np.maximum(c, conf[np.ix_(yy, xx)])
                raw |= np.isin(pred[np.ix_(yy, xx)], [2, 3])
        polygons = json.loads((hk / "native_polygons.geojson").read_text())["features"]
        mask = np.zeros((n, n), np.uint8)
        for f in polygons:
            p = f["properties"]
            if p["predicted_label"] != "tls" or not p["passes_published_filter"]:
                continue
            for ring_index, ring in enumerate(f["geometry"]["coordinates"]):
                pts = np.asarray(ring, float) * 0.5 / GRID  # 0.5 µm px -> grid cells
                cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 0 if ring_index else 1)
        out["hooknet"] = {"confidence": c, "raw_tls": raw, "tls": mask.astype(bool)}
    return out
