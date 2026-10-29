"""CD3/CD20 dual IHC (serial section) quantification for the TNBC_ST validation (CPU only).

Per NDPI (copied out of `IHC/Images IHC.tar` into a temp folder, deleted afterwards):
1. Tissue mask at ~3.65 µm/px (level 4): Otsu on log OD-sum, small objects removed.
2. Three stain vectors per slide (haematoxylin + two chromogens) by OD clustering: unit OD
   directions of stained pixels (OD-sum >= 0.25) from up to 80 random 512-px level-1 tiles inside
   tissue, k-means on the sphere initialised at the reference vectors (skimage rgb_from_hdx / rgb_from_rbd:
   haematoxylin, DAB, Fast Red), 10 iterations. Chromogen identity: brown/DAB = CD3, red = CD20
   (Gate A, morphology: red fills follicle cores, brown surrounds them).
3. Colour deconvolution at level 1 (0.457 µm/px) in 4096-px strips. Positive cells by single-scale LoG
   blob detection on each chromogen concentration map: sigma = 2.5 µm (lymphocyte radius ~3.5 µm / sqrt 2),
   local maxima within a 5 µm window, LoG response >= LOG_MIN and smoothed concentration >= CONC_MIN at the
   peak. No Cellpose: the comparison with H&E is region- and TLS-level.
4. Saved (outputs/tnbc_st_20261101/ihc/<sample>/): points.npz (x, y in level-0 px, concentration, LoG
   response per marker), maps_2um.npz (uint8 concentration x 100 for H, CD3, CD20 at 2 µm/px; tissue mask),
   qc_4um.jpg (thumbnail with CD3 brown / CD20 red dots), summary.json (stain vectors, thresholds, counts).

Thresholds are fixed constants (pre-registered in preregistration.json before any H&E output was read).
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
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tnbc_common as C  # noqa: E402

OUT = C.OUT / "ihc"
TMP = C.OUT / "_tmp_ndpi"
REFERENCE = {"H": [0.650, 0.704, 0.286], "DAB": [0.268, 0.570, 0.776], "Red": [0.214, 0.851, 0.478]}  # skimage hdx/rbd
# Endmember search directions in OD chromaticity (x = r - b, y = g) and the cohort default vectors (median
# of the per-slide fits on Gate-A slides 1, 33, 45, 85, 9; the counterstain is bluer than skimage's H).
DIRECTIONS = {"H": (1.0, 0.0), "DAB": (-1.0, -0.15), "Red": (-0.55, 0.83)}
COHORT = {"H": [0.750, 0.552, 0.356], "DAB": [0.297, 0.518, 0.802], "Red": [0.215, 0.800, 0.566]}
FALLBACK_DEG, MAX_COND = 8.0, 20.0
MARKER = {"DAB": "CD3", "Red": "CD20"}   # decided at Gate A from morphology (QC figure)
OD_MIN = 0.25          # OD-sum of pixels used for stain fitting
SIGMA_UM = 2.5         # LoG scale
PEAK_UM = 5.0          # local-maximum window
LOG_MIN = 0.10         # minimum scale-normalised LoG response (concentration units)
CONC_MIN = {"DAB": 0.8, "Red": 0.5}   # chromogen concentration at a blob centre (above the Gate-A background modes)
MAP_UM = 2.0


def od(rgb):
    return -np.log((rgb.astype(np.float32) + 1.0) / 256.0)


def tissue_mask(slide, level=4):
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects, remove_small_holes
    rgb = np.asarray(slide.read_region((0, 0), level, slide.level_dimensions[level]).convert("RGB"))
    s = od(rgb).sum(-1)
    s = cv2.GaussianBlur(s, (0, 0), 2)
    t = max(0.08, float(np.exp(threshold_otsu(np.log(s + 1e-3)))))
    mpp = float(slide.properties["openslide.mpp-x"]) * slide.level_downsamples[level]
    mask = remove_small_objects(s > t, max_size=int(50000 / mpp ** 2))      # >= 0.05 mm2
    mask = remove_small_holes(mask, max_size=int(20000 / mpp ** 2))
    return mask, rgb, mpp, t


def fit_stains(slide, mask, mask_mpp, n_tiles=80, size=512, seed=0):
    """3x3 stain matrix (rows H, DAB, Red; unit OD vectors) by OD endmember clustering.

    In OD chromaticity (OD / OD-sum; x = r - b, y = g) every pixel is a convex mixture of the three stains, so
    each stain is a vertex of the point cloud. For each stain, the pixels in the top 0.5 % along a fixed
    direction from the cloud median (H +x, DAB -x, Red up-left; set from the Gate-A chromaticity plot) form
    that stain's cluster, and its mean unit OD vector is the stain vector. A per-slide vector is replaced by
    the cohort default (COHORT) when it lies > FALLBACK_DEG from it or the matrix is ill-conditioned
    (cond > MAX_COND), e.g. on slides with almost no red chromogen.
    """
    rng = np.random.default_rng(seed)
    l0 = float(slide.properties["openslide.mpp-x"])
    l1 = l0 * slide.level_downsamples[1]
    ys, xs = np.nonzero(mask)
    pick = rng.choice(len(xs), min(n_tiles, len(xs)), replace=False)
    pixels = []
    for i in pick:
        cx, cy = (xs[i] + .5) * mask_mpp / l0, (ys[i] + .5) * mask_mpp / l0
        x, y = int(cx - size * l1 / l0 / 2), int(cy - size * l1 / l0 / 2)
        tile = np.asarray(slide.read_region((max(x, 0), max(y, 0)), 1, (size, size)).convert("RGB"))
        o = od(tile).reshape(-1, 3)
        pixels.append(o[o.sum(1) >= OD_MIN])
    o = np.concatenate(pixels).astype(np.float64)
    if len(o) > 2_000_000:
        o = o[rng.choice(len(o), 2_000_000, replace=False)]
    u = o / o.sum(1, keepdims=True)
    cx, cy = u[:, 0] - u[:, 2], u[:, 1]
    c0 = np.median(cx), np.median(cy)
    fitted, chosen, info = {}, {}, {}
    for k, (dx, dy) in DIRECTIONS.items():
        score = (cx - c0[0]) * dx + (cy - c0[1]) * dy
        sel = score >= np.percentile(score, 99.5)
        v = (o[sel] / np.linalg.norm(o[sel], axis=1, keepdims=True)).mean(0)
        fitted[k] = v / np.linalg.norm(v)
    default = {k: np.asarray(v) / np.linalg.norm(v) for k, v in COHORT.items()}
    angle = {k: float(np.degrees(np.arccos(np.clip(fitted[k] @ default[k], -1, 1)))) for k in fitted}
    for k in fitted:
        chosen[k] = fitted[k] if angle[k] <= FALLBACK_DEG else default[k]
    matrix = np.array([chosen[k] for k in ["H", "DAB", "Red"]])
    if np.linalg.cond(matrix) > MAX_COND:
        matrix = np.array([default[k] for k in ["H", "DAB", "Red"]])
        chosen = {k: default[k] for k in fitted}
    source = {k: ("slide" if np.allclose(chosen[k], fitted[k]) else "cohort default") for k in fitted}
    info = {"fitted_H_DAB_Red": [fitted[k].tolist() for k in ["H", "DAB", "Red"]], "angle_from_cohort_default_deg": angle,
            "source": source, "condition_number": float(np.linalg.cond(matrix)), "pixels_used": int(len(o))}
    return matrix, info


def deconvolve(rgb, stains):
    """Concentrations (H, DAB, Red) per pixel: OD = C @ stains."""
    o = od(rgb).reshape(-1, 3)
    conc = o @ np.linalg.inv(stains)
    return conc.reshape(rgb.shape[:2] + (3,)).clip(0, None)


def blobs(conc, px_um, offset_xy, l0_per_px):
    """LoG blob centres in one concentration tile -> (x0, y0, concentration, response) in level-0 px."""
    sigma = SIGMA_UM / px_um
    response = -sigma ** 2 * ndimage.gaussian_laplace(conc, sigma)
    smooth = ndimage.gaussian_filter(conc, 1.0 / px_um)
    window = int(round(PEAK_UM / px_um)) | 1
    peaks = (response == ndimage.maximum_filter(response, window)) & (response >= LOG_MIN)
    return peaks, response, smooth


def process(sample, force=False):
    import openslide
    out = OUT / f"TNBC{sample}"
    if (out / "summary.json").exists() and not force:
        return "done"
    if C.free_gb() < C.MIN_FREE_GB:
        raise SystemExit(f"STOP: {C.free_gb():.1f} GB free")
    started = time.time()
    slides = C.ihc_slides()
    info = slides[sample]
    path = TMP / f"TNBC{sample}.ndpi"
    if not path.exists():
        C.copy_member("IHC", info["member"], path)
    out.mkdir(parents=True, exist_ok=True)
    try:
        slide = openslide.OpenSlide(str(path))
        l0 = float(slide.properties["openslide.mpp-x"])
        mask, thumb, mask_mpp, otsu = tissue_mask(slide)
        stains, fit = fit_stains(slide, mask, mask_mpp)
        level = 1
        down = slide.level_downsamples[level]
        px = l0 * down
        W, H = slide.level_dimensions[level]
        # Output maps at MAP_UM
        mw, mh = int(np.ceil(W * px / MAP_UM)), int(np.ceil(H * px / MAP_UM))
        maps = np.zeros((3, mh, mw), np.uint8)
        points = {k: [] for k in ["DAB", "Red"]}
        strip, pad = 4096, 32
        mask_l1 = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        for y in range(0, H, strip):
            for x in range(0, W, strip):
                if not mask_l1[y:y + strip, x:x + strip].any():
                    continue
                x0, y0 = max(x - pad, 0), max(y - pad, 0)
                x1, y1 = min(x + strip + pad, W), min(y + strip + pad, H)
                rgb = np.asarray(slide.read_region((int(x0 * down), int(y0 * down)), level, (x1 - x0, y1 - y0)).convert("RGB"))
                conc = deconvolve(rgb, stains)
                inner = (slice(y - y0, y - y0 + min(strip, H - y)), slice(x - x0, x - x0 + min(strip, W - x)))
                tissue = mask_l1[y0:y1, x0:x1].astype(bool)
                for k, ch in [("DAB", 1), ("Red", 2)]:
                    peaks, response, smooth = blobs(conc[..., ch], px, (x0, y0), down)
                    keep = peaks & (smooth >= CONC_MIN[k]) & tissue
                    keep_inner = np.zeros_like(keep)
                    keep_inner[inner] = keep[inner]
                    yy, xx = np.nonzero(keep_inner)
                    points[k].append(np.stack([(xx + x0 + .5) * down, (yy + y0 + .5) * down,
                                               smooth[yy, xx], response[yy, xx]], 1).astype(np.float32))
                # Maps at MAP_UM from the inner region
                sub = conc[inner]
                f = px / MAP_UM
                small = cv2.resize(sub, (max(1, int(round(sub.shape[1] * f))), max(1, int(round(sub.shape[0] * f)))),
                                   interpolation=cv2.INTER_AREA)
                my, mx = int(round(y * f)), int(round(x * f))
                hh, ww = min(small.shape[0], mh - my), min(small.shape[1], mw - mx)
                maps[:, my:my + hh, mx:mx + ww] = np.clip(small[:hh, :ww] * 100, 0, 255).astype(np.uint8).transpose(2, 0, 1)
        pts = {k: (np.concatenate(v) if v else np.zeros((0, 4), np.float32)) for k, v in points.items()}
        tissue_map = cv2.resize(mask.astype(np.uint8), (mw, mh), interpolation=cv2.INTER_NEAREST)
        np.savez_compressed(out / "points.npz", CD3=pts["DAB"], CD20=pts["Red"],
                            columns=np.array(["x_level0", "y_level0", "concentration", "log_response"]))
        np.savez_compressed(out / "maps_2um.npz", H=maps[0], CD3=maps[1], CD20=maps[2], tissue=tissue_map,
                            um_per_px=np.float32(MAP_UM), scale="concentration x 100, uint8")
        # QC overlay at 4 µm
        qc_um = 4.0
        f = mask_mpp / qc_um
        qc = cv2.resize(thumb, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        overlay = qc.copy()
        for k, colour in [("DAB", (120, 60, 0)), ("Red", (230, 0, 60))]:
            p = pts[k]
            xs = (p[:, 0] * l0 / qc_um).astype(int).clip(0, qc.shape[1] - 1)
            ys = (p[:, 1] * l0 / qc_um).astype(int).clip(0, qc.shape[0] - 1)
            overlay[ys, xs] = colour
        cv2.imwrite(str(out / "qc_4um.jpg"), cv2.cvtColor(np.concatenate([qc, overlay], 1), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        tissue_mm2 = float(mask.sum() * mask_mpp ** 2 / 1e6)
        summary = {"sample": sample, "member": info["member"], "label_CD2_instead_of_CD3": info["label_CD2"],
                   "mpp_level0": l0, "objective": slide.properties.get("openslide.objective-power"),
                   "dimensions_level0": list(slide.dimensions), "tissue_mm2": tissue_mm2, "tissue_otsu_odsum": otsu,
                   "stains_H_DAB_Red": stains.tolist(), "fit": fit, "marker_of": MARKER,
                   "parameters": {"sigma_um": SIGMA_UM, "peak_um": PEAK_UM, "log_min": LOG_MIN, "conc_min": CONC_MIN,
                                  "detection_level_mpp": px, "map_um": MAP_UM},
                   "counts": {MARKER[k]: int(len(v)) for k, v in pts.items()},
                   "density_per_mm2": {MARKER[k]: len(v) / max(tissue_mm2, 1e-9) for k, v in pts.items()},
                   "seconds": time.time() - started}
        C.save_json(summary, out / "summary.json")
        return f"{summary['counts']} in {tissue_mm2:.1f} mm2, {summary['seconds']:.0f}s"
    finally:
        if not os.environ.get("TNBC_KEEP_NDPI"):
            path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="*", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    samples = args.samples or list(C.ihc_slides())
    for sample in samples:
        lock = OUT / "_locks" / f"TNBC{sample}"
        lock.parent.mkdir(parents=True, exist_ok=True)
        try:
            lock.mkdir()
        except FileExistsError:
            continue
        try:
            print(f"TNBC{sample}: {process(sample, args.force)}", flush=True)
        except Exception as error:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"TNBC{sample}: FAILED {error!r}", flush=True)
        finally:
            lock.rmdir()


if __name__ == "__main__":
    main()
