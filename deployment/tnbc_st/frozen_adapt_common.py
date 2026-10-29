"""Shared helpers for the TNBC_ST frozen-section adaptation add-on (deployment/tnbc_st/frozen_adaptation_plan.md).

Read-only on outputs/tnbc_st_20261101 (cells, inventory, registration, IHC) and on outputs/esca_tls_20261019; writes only
to outputs/tnbc_frozen_adapt_20261101.

Crops: the stage-1 classifier input (64-px RGB + Cellpose mask at 0.25 um/px, raw mode, white image padding) is rebuilt
exactly: each patch is resampled from imagesHD with the visium_cells.patch_grid_image arithmetic (bicubic warpAffine),
the captured stage-1 Cellpose label tif of that patch gives the instance (matched to cells.csv by centroid), and the
crop is cut with run_cellcrops.crop_with_padding. No new segmentation.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "visium_tls"))
sys.path.insert(0, str(HERE.parent / "cell_validation"))
sys.path.insert(0, str(ROOT / "script" / "src"))
import tnbc_common as C  # noqa: E402

MAIN = C.OUT
OUT = ROOT / "outputs" / "tnbc_frozen_adapt_20261101"
CAPTURE = OUT / "seg_capture"
PREREG = json.loads((OUT / "preregistration.json").read_text()) if (OUT / "preregistration.json").exists() else None
CHECKPOINT = ROOT / "models" / "he2cell_esca_finetuned.pth"
PATCH, STRIDE, MARGIN = 1024, 768, 128
TARGET_MPP = 0.25
TYPES = ["Other", "T", "Macrophage", "B"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SEED = 569


def inventory():
    return json.loads((MAIN / "inventory.json").read_text())["arrays"]


def hd_mpp():
    return json.loads((MAIN / "scale.json").read_text())["hd_mpp"]


def folds():
    return PREREG["folds"]["assignment"]


def fold_of(case_id):
    for k, v in folds().items():
        if case_id in v:
            return k
    return None


def priority():
    return (MAIN / "stage1_priority.txt").read_text().split()


def done_arrays():
    """Priority arrays with final stage-1 cells AND captured masks for every patch."""
    out = []
    for a in priority():
        if not (MAIN / "cells" / a / "cells.csv.gz").exists():
            continue
        pi = CAPTURE / a / f"{a}_patch_index.csv"
        md = CAPTURE / a / "patch_masks" / a
        if pi.exists() and md.exists() and len(list(md.glob("*_cp_mask.tif"))) == len(pd.read_csv(pi)):
            out.append(a)
    return out


# ---------------------------------------------------------------------------------------------------------- crops
def patch_rgb(image, x, y, scale):
    """Exactly the patch visium_cells.patch_grid_image yields at (x, y) (target px)."""
    x0, y0 = x / scale - 1, y / scale - 1
    sx, sy = int(np.floor(x0)), int(np.floor(y0))
    size = int(np.ceil(PATCH / scale)) + 3
    region = np.full((size, size, 3), 255, np.uint8)
    ax, ay = max(sx, 0), max(sy, 0)
    bx, by = min(sx + size, image.shape[1]), min(sy + size, image.shape[0])
    if bx > ax and by > ay:
        region[ay - sy:by - sy, ax - sx:bx - sx] = image[ay:by, ax:bx]
    matrix = np.float32([[scale, 0, -(x0 - sx + 1) * scale], [0, scale, -(y0 - sy + 1) * scale]])
    return cv2.warpAffine(region, matrix, (PATCH, PATCH), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


def crop_with_padding(arr, cx, cy, size=64, pad=0):
    half = size // 2
    cx, cy = int(round(cx)), int(round(cy))
    x1, y1 = cx - half, cy - half
    h, w = arr.shape[:2]
    out = np.full((size, size) + arr.shape[2:], pad, arr.dtype)
    sx1, sy1, sx2, sy2 = max(0, x1), max(0, y1), min(w, x1 + size), min(h, y1 + size)
    if sx2 > sx1 and sy2 > sy1:
        out[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1] = arr[sy1:sy2, sx1:sx2]
    return out


def scaled_crop(arr, cx, cy, s, size=64, pad=0, interp=cv2.INTER_CUBIC):
    """A5: content scaled by s around the rounded centroid (window size/s resized to size)."""
    cx, cy = int(round(cx)), int(round(cy))
    # output pixel u -> input cx + (u - 32) / s  (pixel 32 of the 64-px crop is the centroid in crop_with_padding)
    M = np.float32([[1 / s, 0, cx - 32 / s], [0, 1 / s, cy - 32 / s]])
    border = (pad, pad, pad) if arr.ndim == 3 else pad
    return cv2.warpAffine(arr, M, (size, size), flags=interp | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def load_cells(array):
    return pd.read_csv(MAIN / "cells" / array / "cells.csv.gz")


def build_crops(array, cells, scales=(1.0,), image=None, with_props=False):
    """cells: subset of cells.csv rows. Returns dict scale -> (rgb uint8 N,64,64,3, mask uint8 N,64,64 {0,255}) in the order
    of `cells`, plus a found flag (and optional whole-instance props: area_px, solidity, from the full patch mask)."""
    import tifffile
    s = hd_mpp() / TARGET_MPP
    pidx = pd.read_csv(CAPTURE / array / f"{array}_patch_index.csv")
    pidx["stem"] = pidx.path.map(lambda p: Path(p).stem)
    pidx = pidx.set_index("num_patch")
    if image is None:
        image = C.read_image(inventory()[array]["hd"])
    n = len(cells)
    out = {sc: (np.full((n, 64, 64, 3), 255, np.uint8), np.zeros((n, 64, 64), np.uint8)) for sc in scales}
    found = np.zeros(n, bool)
    props = np.full((n, 2), np.nan)
    pos = np.arange(n)
    for pid, sub in cells.assign(_pos=pos).groupby("patch_id"):
        rec = pidx.loc[pid]
        mask = tifffile.imread(CAPTURE / array / "patch_masks" / array / f"{rec.stem}_cp_mask.tif").astype(np.int32)
        rgb = patch_rgb(image, rec.x, rec.y, s)
        nl = mask.max()
        if nl == 0:
            continue
        ys, xs = np.nonzero(mask)
        lab = mask[ys, xs]
        cnt = np.bincount(lab, minlength=nl + 1)
        cx = np.bincount(lab, xs, minlength=nl + 1) / np.maximum(cnt, 1)
        cy = np.bincount(lab, ys, minlength=nl + 1) / np.maximum(cnt, 1)
        lx = sub.centroid_x_global.to_numpy() - rec.x
        ly = sub.centroid_y_global.to_numpy() - rec.y
        # match each stored cell to its instance by centroid (regionprops centroid = pixel mean)
        d = (cx[None, 1:] - lx[:, None]) ** 2 + (cy[None, 1:] - ly[:, None]) ** 2
        best = d.argmin(1) + 1
        ok = d[np.arange(len(sub)), best - 1] < 1e-4
        for j, (p, label, good, x_, y_) in enumerate(zip(sub._pos, best, ok, lx, ly)):
            if not good:
                continue
            found[p] = True
            lc = crop_with_padding(mask, x_, y_, 64, 0)
            for sc in scales:
                if sc == 1.0:
                    out[sc][0][p] = crop_with_padding(rgb, x_, y_, 64, 255)
                    out[sc][1][p] = (lc == label) * 255
                else:
                    out[sc][0][p] = scaled_crop(rgb, x_, y_, sc, pad=255)
                    out[sc][1][p] = (scaled_crop(mask, x_, y_, sc, pad=0, interp=cv2.INTER_NEAREST) == label) * 255
            if with_props:
                from skimage.measure import regionprops
                ys0, xs0 = int(max(y_ - 48, 0)), int(max(x_ - 48, 0))
                win = (mask[ys0:int(y_) + 48, xs0:int(x_) + 48] == label).astype(np.uint8)
                rp = regionprops(win)
                if rp:
                    props[p] = [rp[0].area, rp[0].solidity]
    if with_props:
        return out, found, props
    return out, found


# ---------------------------------------------------------------------------------------------------------- Macenko
IO, ALPHA, BETA = 240.0, 1.0, 0.15


def _od(rgb):
    return -np.log((rgb.reshape(-1, 3).astype(np.float64) + 1) / IO)


def macenko_fit(rgb_pixels, seed=SEED, max_pixels=2_000_000):
    od = _od(rgb_pixels)
    od = od[(od > BETA).all(1)]
    if len(od) > max_pixels:
        od = od[np.random.default_rng(seed).choice(len(od), max_pixels, replace=False)]
    _, vecs = np.linalg.eigh(np.cov(od.T))
    v = vecs[:, [1, 2]]
    v = v * np.sign(v.sum(0, keepdims=True))
    proj = od @ v
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    lo, hi = np.percentile(phi, [ALPHA, 100 - ALPHA])
    a = v @ np.array([np.cos(lo), np.sin(lo)])
    b = v @ np.array([np.cos(hi), np.sin(hi)])
    he = np.stack([a, b], 1) if a[0] > b[0] else np.stack([b, a], 1)  # haematoxylin first (more red absorbance)
    he = he / np.linalg.norm(he, axis=0, keepdims=True)
    conc = np.linalg.pinv(he) @ od.T
    max_c = np.percentile(conc, 99, axis=1)
    return {"HE": he.tolist(), "maxC": max_c.tolist(), "pixels": int(len(od))}


def macenko_apply(rgb, src, tgt):
    """rgb uint8 (..., 3) -> normalised uint8 of same shape."""
    shape = rgb.shape
    od = _od(rgb)
    conc = np.linalg.pinv(np.asarray(src["HE"])) @ od.T
    conc *= (np.asarray(tgt["maxC"]) / np.asarray(src["maxC"]))[:, None]
    out = IO * np.exp(-(np.asarray(tgt["HE"]) @ conc))
    return np.clip(out.T, 0, 255).reshape(shape).astype(np.uint8)


def esca_target():
    path = OUT / "macenko_esca_target.json"
    if path.exists():
        return json.loads(path.read_text())
    from PIL import Image
    bank = ROOT / "outputs" / "esca_tls_20261019" / "cropbank_dev"
    rng = np.random.default_rng(SEED)
    pix = []
    for d in sorted(bank.glob("TCGA-*")):
        c = pd.read_csv(d / "cells.csv", usecols=["cell_image_crop_path"])
        for p in c.cell_image_crop_path.to_numpy()[rng.choice(len(c), min(250, len(c)), replace=False)]:
            pix.append(np.asarray(Image.open(p).convert("RGB")).reshape(-1, 3))
    t = macenko_fit(np.concatenate(pix))
    t["source"] = "250 random crops per TCGA-ESCA dev cropbank slide (12 slides), seed 569"
    C.save_json(t, path)
    return t


# ---------------------------------------------------------------------------------------------------------- model
def load_model(device="cpu"):
    from model_eval import SAM_ConvNet
    from runtime import load_classifier_state
    m = SAM_ConvNet(num_classes=4, loss_function="CrossEntropyLoss", pretrained=False)
    m.load_state_dict(load_classifier_state(CHECKPOINT))
    return m.to(device).eval()


def to_input(rgb, mask):
    import torch
    x = (torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255.0 - torch.from_numpy(MEAN)[:, None, None]) / torch.from_numpy(STD)[:, None, None]
    m = mask.astype(np.float32)
    mx = m.reshape(len(m), -1).max(1)
    m = torch.from_numpy(np.where(mx[:, None, None] > 0, m / np.maximum(mx, 1e-12)[:, None, None], m))[:, None]  # mask/max(mask)
    return torch.cat([x, m], 1).contiguous()


def predict_d4(model, rgb, mask, device, batch=512):
    import torch
    from prediction_tta import average_views
    out = []
    with torch.inference_mode():
        for i in range(0, len(rgb), batch):
            p, _ = average_views(model, to_input(rgb[i:i + batch], mask[i:i + batch]).to(device))
            out.append(p.float().cpu().numpy())
    return np.concatenate(out).astype(np.float64) if out else np.zeros((0, 4))


def adabn(model, rgb, mask, device, batch=256, seed=SEED):
    """Reset every BatchNorm's running stats and re-estimate them (cumulative average) on the given crops."""
    import copy
    import torch
    m = copy.deepcopy(model)
    bns = [x for x in m.modules() if isinstance(x, torch.nn.BatchNorm2d)]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None
    m.eval()
    for bn in bns:
        bn.train()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rgb))
    with torch.no_grad():
        for i in range(0, len(order), batch):
            idx = np.sort(order[i:i + batch])
            if len(idx) < 16:
                continue
            x = to_input(rgb[idx], mask[idx])
            k = int(rng.integers(8))  # one random D4 view per batch
            x = torch.rot90(x.flip(-1) if k >= 4 else x, k % 4, (-2, -1))
            m(x.to(device))
    m.eval()
    return m


def prior_tau():
    from model_eval import load_prior_correction
    return load_prior_correction(CHECKPOINT, "auto")


def apply_tau(p):
    from model_eval import apply_prior
    return apply_prior(p, prior_tau())


def saerens_em(p, train_prior=None, tol=1e-6, max_iter=1000):
    """Saerens et al. 2002: returns new prior pi (and iterations) for posteriors p (N, K) under train_prior."""
    k = p.shape[1]
    t = np.full(k, 1 / k) if train_prior is None else np.asarray(train_prior)
    pi = p.mean(0)
    for it in range(max_iter):
        w = p * (pi / t)
        w /= w.sum(1, keepdims=True)
        new = w.mean(0)
        if np.abs(new - pi).max() < tol:
            return new, it + 1
        pi = new
    return pi, max_iter


def apply_em(p, pi, train_prior=None):
    t = np.full(p.shape[1], 1 / p.shape[1]) if train_prior is None else np.asarray(train_prior)
    w = p * (np.asarray(pi) / t)
    return w / w.sum(1, keepdims=True)
