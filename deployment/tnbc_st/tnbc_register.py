"""Register each H&E array (imagesLarge frame) to its serial-section CD3/CD20 IHC slide (NDPI level-0 frame).

Images: H&E = OD-sum of imagesLarge/4 (RGB), IHC = OD-sum of the left (RGB) half of ihc/<sample>/qc_4um.jpg;
both resampled to COARSE_UM / FINE_UM, clipped to [0, 3] and blurred (sigma 1 px).
1. Coarse: rotation 0-359 deg in 1 deg steps x flip (mirror x) x scale (H&E µm/px x {0.95 ... 1.05}) by masked
   normalised cross-correlation (cv2.matchTemplate TM_CCOEFF_NORMED, rotated H&E as template) at COARSE_UM.
2. Refine: cv2.findTransformECC, MOTION_AFFINE, at FINE_UM.
3. Fine QC: 1 mm tiles of H&E tissue (>= 50 % tissue in both after warping): ECC translation of the warped H&E
   tile against the IHC tile, residual = |shift| in µm. Accept the pair when >= 3 tiles converge, >= 50 % of
   the tiles converge and the median residual <= 100 µm (preregistration.json).
Outputs registration/<array>.json (affine H&E Large px -> IHC level-0 px, metrics, accept flag) and
registration/<array>_overlay.jpg (IHC in magenta / green H&E overlay and checkerboard). CPU only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tnbc_common as C  # noqa: E402

OUT = C.OUT / "registration"
COARSE_UM, FINE_UM, TILE_UM = 20.0, 10.0, 1000.0
SCALES = [0.95, 0.975, 1.0, 1.025, 1.05]
MAX_MEDIAN_RESIDUAL_UM, MIN_TILES = 100.0, 3
LARGE = 9523


def od_sum(rgb):
    return -np.log((rgb.astype(np.float32) + 1) / 256).sum(-1)


def prep(image):
    return np.clip(cv2.GaussianBlur(image, (0, 0), 1.0), 0, 3).astype(np.float32)


def he_image(entry, large_mpp):
    """H&E OD-sum at FINE_UM and its tissue mask."""
    large4 = C.read_image(entry["large"], 4)
    px = large_mpp * LARGE / large4.shape[0]
    f = px / FINE_UM
    image = cv2.resize(od_sum(large4), None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    # Same tissue rule as tnbc_cells (smoothed saturation x OD-sum, minus annotated artefacts/holes), without the
    # ST-frame crop so the whole tissue piece takes part in the match.
    import tnbc_cells as TC
    mask, _ = TC.tissue_mask({k: v for k, v in entry.items() if k != "spots_frame_large"}, large4)
    mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    image = image * cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8))  # background (stitching tiles) -> 0
    return image, mask, FINE_UM / large_mpp  # Large px per fine px


def ihc_image(sample):
    qc = cv2.cvtColor(cv2.imread(str(C.OUT / "ihc" / f"TNBC{sample}" / "qc_4um.jpg")), cv2.COLOR_BGR2RGB)
    rgb = qc[:, :qc.shape[1] // 2]
    summary = json.loads((C.OUT / "ihc" / f"TNBC{sample}" / "summary.json").read_text())
    maps = np.load(C.OUT / "ihc" / f"TNBC{sample}" / "maps_2um.npz")
    f = 4.0 / FINE_UM
    image = cv2.resize(od_sum(rgb), None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    tissue = cv2.resize(maps["tissue"], (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    image = image * cv2.dilate(tissue.astype(np.uint8), np.ones((3, 3), np.uint8))  # glass, dust and pen -> 0
    return image, tissue, FINE_UM / summary["mpp_level0"]  # level-0 px per fine px


def rotate(image, angle, flip):
    """Rotate (deg, about the centre) after an optional x-mirror; returns the image and its 3x3 matrix."""
    h, w = image.shape
    F = np.array([[-1, 0, w - 1], [0, 1, 0], [0, 0, 1]], float) if flip else np.eye(3)
    d = int(np.ceil(np.hypot(h, w)))
    R = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, 1.0)
    R[0, 2] += (d - w) / 2
    R[1, 2] += (d - h) / 2
    M = np.vstack([R, [0, 0, 1]]) @ F
    return cv2.warpAffine(image, M[:2], (d, d), flags=cv2.INTER_LINEAR, borderValue=0), M


def coarse(he, ihc, he_scale_nominal):
    """he, ihc at FINE_UM; returns 3x3 matrix H&E fine px -> IHC fine px and the NCC."""
    k = FINE_UM / COARSE_UM
    I = prep(cv2.resize(ihc, None, fx=k, fy=k, interpolation=cv2.INTER_AREA))
    best = (-2, None)
    for s in SCALES:
        H = prep(cv2.resize(he, None, fx=k * s, fy=k * s, interpolation=cv2.INTER_AREA))
        S = np.diag([k * s, k * s, 1.0])
        for flip in (0, 1):
            for angle in range(360):
                T, M = rotate(H, angle, flip)
                pad = (T.shape[0] // 2, T.shape[1] // 2)
                P = np.pad(I, ((pad[0], pad[0]), (pad[1], pad[1])))
                r = cv2.matchTemplate(P, T, cv2.TM_CCOEFF_NORMED)
                _, value, _, loc = cv2.minMaxLoc(r)
                if value > best[0]:
                    shift = np.array([[1, 0, loc[0] - pad[1]], [0, 1, loc[1] - pad[0]], [0, 0, 1]], float)
                    # H&E fine px -> coarse H&E px (S) -> rotated template (M) -> coarse IHC (shift) -> fine IHC
                    best = (value, np.diag([1 / k, 1 / k, 1.0]) @ shift @ M @ S, {"scale": s, "flip": flip, "angle": angle})
    return best


def refine(he, ihc, matrix):
    """ECC affine; returns 3x3 H&E fine -> IHC fine and the ECC correlation (None on failure)."""
    warp = np.linalg.inv(matrix)[:2].astype(np.float32)  # IHC fine px -> H&E fine px (ECC warps the input)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6)
    try:
        cc, warp = cv2.findTransformECC(prep(ihc), prep(he), warp, cv2.MOTION_AFFINE, criteria, None, 5)
    except cv2.error:
        return matrix, None
    return np.linalg.inv(np.vstack([warp, [0, 0, 1]])), float(cc)


def tile_residuals(he, he_mask, ihc, ihc_mask, matrix):
    """Per 1 mm tile: ECC translation of warped H&E vs IHC; residual µm (nan if not converged)."""
    shape = (ihc.shape[1], ihc.shape[0])
    warped = cv2.warpAffine(prep(he), matrix[:2], shape, flags=cv2.INTER_LINEAR)
    wmask = cv2.warpAffine(he_mask.astype(np.uint8), matrix[:2], shape, flags=cv2.INTER_NEAREST) > 0
    I = prep(ihc)
    n = int(TILE_UM / FINE_UM)
    margin = int(150 / FINE_UM)
    rows = []
    for y in range(0, ihc.shape[0] - n + 1, n):
        for x in range(0, ihc.shape[1] - n + 1, n):
            a, b = wmask[y:y + n, x:x + n], ihc_mask[y:y + n, x:x + n]
            if a.mean() < .5 or b.mean() < .5:
                continue
            y0, y1, x0, x1 = max(y - margin, 0), min(y + n + margin, ihc.shape[0]), max(x - margin, 0), min(x + n + margin, ihc.shape[1])
            template = I[y0:y1, x0:x1]
            moving = warped[y0:y1, x0:x1]
            warp = np.eye(2, 3, dtype=np.float32)
            try:
                cc, warp = cv2.findTransformECC(template, moving, warp, cv2.MOTION_TRANSLATION,
                                                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), None, 3)
                shift = float(np.hypot(*warp[:, 2])) * FINE_UM
                ok = shift <= 400
            except cv2.error:
                cc, shift, ok = np.nan, np.nan, False
            rows.append({"x_um": x * FINE_UM, "y_um": y * FINE_UM, "residual_um": shift if ok else np.nan, "ecc": float(cc)})
    return rows


def overlay(he, ihc, matrix, path):
    shape = (ihc.shape[1], ihc.shape[0])
    warped = cv2.warpAffine(prep(he), matrix[:2], shape)
    a = np.clip(prep(ihc) / 1.5, 0, 1)
    b = np.clip(warped / 1.5, 0, 1)
    colour = np.stack([a, b, a], -1)
    n = int(TILE_UM / FINE_UM)
    yy, xx = np.indices(a.shape)
    checker = np.where(((yy // n + xx // n) % 2 == 0)[..., None], np.repeat(a[..., None], 3, -1), np.repeat(b[..., None], 3, -1))
    image = (255 - 255 * np.concatenate([colour, checker], 1)).astype(np.uint8)
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 85])


def register(array, entry, sample, large_mpp):
    started = time.time()
    he, he_mask, he_px = he_image(entry, large_mpp)
    ihc, ihc_mask, ihc_px = ihc_image(sample)
    ncc, matrix, params = coarse(he, ihc, 1.0)
    matrix, ecc = refine(he, ihc, matrix)
    rows = tile_residuals(he, he_mask, ihc, ihc_mask, matrix)
    residuals = np.array([r["residual_um"] for r in rows], float)
    converged = np.isfinite(residuals)
    median = float(np.nanmedian(residuals)) if converged.any() else float("nan")
    accept = bool(converged.sum() >= MIN_TILES and converged.mean() >= .5 and median <= MAX_MEDIAN_RESIDUAL_UM)
    # Full transform: H&E Large px -> H&E fine px -> IHC fine px -> IHC level-0 px
    full = np.diag([ihc_px, ihc_px, 1.0]) @ matrix @ np.diag([1 / he_px, 1 / he_px, 1.0])
    linear = full[:2, :2] * C_MPP(sample) / large_mpp
    singular = np.linalg.svd(linear, compute_uv=False)
    result = {"array": array, "sample": sample, "coarse": {"ncc": float(ncc), **params}, "ecc_affine": ecc,
              "he_large_to_ihc_level0": full.tolist(), "relative_scale_singular_values": singular.tolist(),
              "implied_large_mpp": float(large_mpp * np.sqrt(abs(np.linalg.det(linear)))),
              "implied_large_mpp_x_axis": float(np.linalg.norm(full[:2, 0]) * C_MPP(sample)),
              "implied_large_mpp_y_axis": float(np.linalg.norm(full[:2, 1]) * C_MPP(sample)),
              "tiles": rows, "tiles_total": len(rows), "tiles_converged": int(converged.sum()),
              "median_residual_um": median, "accept": accept, "seconds": time.time() - started}
    OUT.mkdir(parents=True, exist_ok=True)
    C.save_json(result, OUT / f"{array}.json")
    overlay(he, ihc, matrix, OUT / f"{array}_overlay.jpg")
    return result


def C_MPP(sample):
    return json.loads((C.OUT / "ihc" / f"TNBC{sample}" / "summary.json").read_text())["mpp_level0"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", nargs="*")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--large-mpp", type=float, default=None, help="override scale.json (for the scale estimate)")
    parser.add_argument("--shard", default="0/1", help="i/n: process every n-th array starting at i")
    args = parser.parse_args()
    inventory = json.loads((C.OUT / "inventory.json").read_text())
    large_mpp = args.large_mpp or json.loads((C.OUT / "scale.json").read_text())["large_mpp"]
    arrays = args.arrays or [a for a, e in inventory["arrays"].items() if e.get("register")]
    i, n = map(int, args.shard.split("/"))
    for array in arrays[i::n]:
        entry = inventory["arrays"][array]
        if (OUT / f"{array}.json").exists() and not args.force:
            continue
        if not (C.OUT / "ihc" / f"TNBC{entry['sample']}" / "summary.json").exists():
            print(f"{array}: IHC not processed yet", flush=True)
            continue
        r = register(array, entry, entry["sample"], large_mpp)
        print(f"{array}: ncc {r['coarse']['ncc']:.2f} ecc {r['ecc_affine']} median {r['median_residual_um']:.0f} µm "
              f"({r['tiles_converged']}/{r['tiles_total']}) accept {r['accept']} scale {r['implied_large_mpp']:.4f} "
              f"{r['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
