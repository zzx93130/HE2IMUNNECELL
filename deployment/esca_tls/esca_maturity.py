"""Germinal-centre (GC) morphology score for accepted two-stage aggregates (plan item 5a).

Composition (B share, B-core enrichment) does not grade TLS maturity, so this tries morphology.
Inside each accepted aggregate (hole-filled, so a pale centre that fell below the nuclear-density
threshold is kept) four GC cues are measured around a GC candidate, the palest 60 µm disk at least
30 µm inside the aggregate edge:
- pale_contrast: haematoxylin OD (colour deconvolution, Gaussian σ = 8 µm) median in the mantle ring
  (30–90 µm from the candidate centre) minus the mean OD of the 60 µm candidate disk, divided by the
  mantle median (stain-intensity invariant; the absolute difference is kept as a descriptive feature);
- size_gradient: log2(median Cellpose nuclear area in the disk / in the mantle ring); centroblasts are larger;
- B_dip: 1 − B density in the centre ring (0–30 µm) / the peak B density of rings 30–60, 60–90, 90–120 µm
  (a mantle peak with an interior dip);
- density_dip: 1 − nuclei/mm² in the disk / in the mantle ring.
Geometric-core variants (inner half of the aggregate by distance to its edge vs the outer half) are
reported as descriptive single features only.

Protocol: a logistic combination of the four primary cues is fitted on ESCA dev aggregates with
HookNet GC pixels (≥ 1% of the aggregate) as a weak proxy label, frozen, then evaluated on ESCA test
aggregates (HookNet GC) and on Visium TLS objects (pathologist mature vs early, the real test) by
`visium_maturity.py` and `report` below.

Stages: `features` (ESCA aggregates -> CSV), `fit` (dev logistic -> frozen JSON), `report` (needs
the Visium CSV from visium_maturity.py; evaluation, gallery, HTML, summary.json).
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ProcessPoolExecutor
import html
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("ESCA_RESULT", "he2cell_ft_tau05")  # default classifier: fine-tuned + tau 0.5 prior

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage, optimize, signal, stats
from skimage.color import rgb2hed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import esca_analysis as A  # noqa: E402
from esca_common import OUT, ROOT, save_json  # noqa: E402
from esca_qc import slide_qc  # noqa: E402

DATE = os.environ.get("MATURITY_DATE", "20261025")
MATURITY = ROOT / "outputs" / f"maturity_{DATE}"
ANALYSIS = OUT / f"analysis_{A.RESULT}"
FROZEN = json.loads((ANALYSIS / "report_summary.json").read_text())["chosen_parameters"][A.TWO_STAGE]
STAGE1 = {k: FROZEN[k] for k in ["sigma_um", "threshold", "min_area_um2", "min_cells"]}

OD_SIGMA_UM = 8.0
GC_RADIUS_UM = 30.0           # candidate GC region = 60 µm disk
CROP_MARGIN_UM = 48.0
TARGET_MPP = 1.0              # ESCA image read at the pyramid level closest to (not finer than needed) 1 µm/px
SEG_MPP = 0.25                # Cellpose runs at 0.25 µm/px in both cohorts; area_pixels * SEG_MPP² = µm²
GC_LABEL_FRACTION = 0.01      # HookNet GC pixels in the aggregate counted as "has GC" (as in esca_analysis)
PRIMARY = ["pale_contrast", "size_gradient", "B_dip", "density_dip"]
# Post-hoc sensitivity (added after the pre-specified 30 µm score failed on Visium; never used for pass/fail):
# candidate radius R with mantle ring R–3R and B rings at multiples of R.
SENSITIVITY_RADII = [50.0, 75.0]
DESCRIPTIVE = ["pale_contrast_abs", "pale_contrast_core", "size_gradient_core", "density_dip_core"]
COMPOSITION = ["B_share", "B_core_enrichment", "T_rim_enrichment", "area_um2"]
NAMES = {"pale_contrast": "Pale-centre contrast (relative OD)", "size_gradient": "Nuclear-size gradient (log2 interior/mantle)",
         "B_dip": "Radial B profile: interior dip", "density_dip": "Nuclear-density dip", "GC_score": "GC morphology score (frozen logistic)",
         "pale_contrast_abs": "Pale-centre contrast (absolute OD)", "pale_contrast_core": "Pale core, geometric (relative OD)",
         "size_gradient_core": "Nuclear-size gradient, geometric core", "density_dip_core": "Nuclear-density dip, geometric core",
         "B_share": "B share (composition)", "B_core_enrichment": "B-core enrichment (composition)",
         "T_rim_enrichment": "T-rim enrichment (composition)", "area_um2": "Aggregate area", "B_fraction_spots": "B fraction of spots (Visium baseline)",
         "B_density_spots": "B density at spots (Visium baseline)", "spots": "Spots in object (Visium baseline)"}


# ---------------------------------------------------------------- shared feature code (also used by Visium)

def composition(obj, counts, valid):
    """The composition features of esca_analysis.proposal_features, for an arbitrary 8 µm mask."""
    total = sum(counts.values())
    distance = ndimage.distance_transform_edt(obj)
    core = obj & (distance >= .5 * distance.max())
    ring = int(round(A.RIM_UM / A.GRID_UM))
    periphery = (obj & ~core) | (ndimage.binary_dilation(obj, iterations=ring) & ~obj & valid)

    def share(kind, mask, among=None):
        denominator = sum(counts[k][mask].sum() for k in among) if among else total[mask].sum()
        return counts[kind][mask].sum() / denominator if denominator else np.nan
    b_core, b_periphery = share("B", core), share("B", periphery)
    t_core, t_periphery = share("T", core, ["B", "T"]), share("T", periphery, ["B", "T"])
    return {"area_um2": obj.sum() * A.GRID_UM ** 2, "cells": int(total[obj].sum()), "B_share": share("B", obj),
            "T_share": share("T", obj), "B_core_enrichment": b_core - b_periphery, "T_rim_enrichment": t_periphery - t_core}


def haematoxylin(rgb, mpp):
    """Tissue mask and haematoxylin OD smoothed at OD_SIGMA_UM with background pixels left out."""
    tissue = (rgb.min(-1) < 215) & (rgb.max(-1) > 30)
    h = rgb2hed(rgb)[..., 0]
    s = OD_SIGMA_UM / mpp
    weight = ndimage.gaussian_filter(tissue.astype(float), s)
    smooth = ndimage.gaussian_filter(np.where(tissue, h, 0.0), s) / np.maximum(weight, 1e-6)
    smooth[weight < .2] = np.nan
    return tissue, smooth


def gc_features(rgb, mpp, filled, cells_xy, cells_area, cells_type, radius_um=GC_RADIUS_UM):
    """GC cues for one aggregate. All arrays in the crop's pixel frame (mpp µm/px).

    filled: hole-filled aggregate mask (bool, crop shape); cells_xy: (n, 2) pixel coordinates.
    Returns the features plus the geometry used (centre, OD map) for the gallery.
    """
    tissue, od = haematoxylin(rgb, mpp)
    edge = ndimage.distance_transform_edt(filled) * mpp
    out = {"aggregate_depth_um": float(edge.max())}
    mantle_um = (radius_um, 3 * radius_um)
    b_rings = [k * radius_um for k in range(5)]
    radius = radius_um / mpp
    r = int(np.ceil(radius))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = (yy ** 2 + xx ** 2 <= radius ** 2).astype(float)
    usable = tissue & np.isfinite(od)
    numerator = signal.fftconvolve(np.where(usable, od, 0.0), disk, mode="same")
    denominator = signal.fftconvolve(usable.astype(float), disk, mode="same")
    disk_mean = np.where(denominator >= .5 * disk.sum(), numerator / np.maximum(denominator, 1e-6), np.nan)
    candidates = filled & (edge >= min(radius_um, edge.max())) & np.isfinite(disk_mean)
    nan = {k: np.nan for k in PRIMARY + DESCRIPTIVE}
    if not candidates.any():
        return {**out, **nan}, None
    cy, cx = np.unravel_index(np.argmin(np.where(candidates, disk_mean, np.inf)), disk_mean.shape)
    gy, gx = np.mgrid[:filled.shape[0], :filled.shape[1]]
    rr = np.hypot(gy - cy, gx - cx) * mpp
    interior = filled & (rr < radius_um)
    mantle = filled & (rr >= mantle_um[0]) & (rr < mantle_um[1])
    mantle_od = np.nanmedian(od[mantle & usable]) if (mantle & usable).sum() else np.nan
    out.update({"centre_x_px": int(cx), "centre_y_px": int(cy), "centre_depth_um": float(edge[cy, cx]),
                "mantle_OD": float(mantle_od), "interior_OD": float(disk_mean[cy, cx]),
                "pale_contrast_abs": float(mantle_od - disk_mean[cy, cx]),
                "pale_contrast": float((mantle_od - disk_mean[cy, cx]) / mantle_od) if mantle_od > 0 else np.nan})
    # Cells: position relative to the candidate centre, kept only inside the aggregate.
    px = np.clip(np.round(cells_xy).astype(int), 0, [filled.shape[1] - 1, filled.shape[0] - 1])
    inside = filled[px[:, 1], px[:, 0]] & (cells_xy[:, 0] >= 0) & (cells_xy[:, 1] >= 0) & \
        (cells_xy[:, 0] < filled.shape[1]) & (cells_xy[:, 1] < filled.shape[0])
    rc = np.hypot(cells_xy[:, 0] - cx, cells_xy[:, 1] - cy) * mpp
    in_i, in_m = inside & (rc < radius_um), inside & (rc >= mantle_um[0]) & (rc < mantle_um[1])
    area_i, area_m = (interior & tissue).sum() * mpp ** 2 / 1e6, (mantle & tissue).sum() * mpp ** 2 / 1e6
    out.update({"cells_interior": int(in_i.sum()), "cells_mantle": int(in_m.sum()),
                "nuclei_per_mm2_interior": in_i.sum() / area_i if area_i else np.nan,
                "nuclei_per_mm2_mantle": in_m.sum() / area_m if area_m else np.nan,
                "median_area_interior_um2": float(np.median(cells_area[in_i])) if in_i.sum() else np.nan,
                "median_area_mantle_um2": float(np.median(cells_area[in_m])) if in_m.sum() else np.nan})
    out["size_gradient"] = (float(np.log2(out["median_area_interior_um2"] / out["median_area_mantle_um2"]))
                            if in_i.sum() >= 5 and in_m.sum() >= 10 else np.nan)
    out["density_dip"] = (1 - out["nuclei_per_mm2_interior"] / out["nuclei_per_mm2_mantle"]
                          if out["nuclei_per_mm2_mantle"] and np.isfinite(out["nuclei_per_mm2_mantle"]) else np.nan)
    rings = []
    is_b = cells_type == "B"
    for lo, hi in zip(b_rings[:-1], b_rings[1:]):
        region = filled & tissue & (rr >= lo) & (rr < hi)
        full = np.pi * (hi ** 2 - lo ** 2) / mpp ** 2
        rings.append(((inside & is_b & (rc >= lo) & (rc < hi)).sum() / (region.sum() * mpp ** 2 / 1e6))
                     if region.sum() >= .1 * full else np.nan)
    for (lo, hi), value in zip(zip(b_rings[:-1], b_rings[1:]), rings):
        out[f"B_per_mm2_{lo:.0f}_{hi:.0f}um"] = value
    peak = np.nanmax(rings[1:]) if np.isfinite(rings[1:]).any() else np.nan
    out["B_dip"] = 1 - rings[0] / peak if np.isfinite(rings[0]) and peak > 0 else np.nan
    # Geometric-core variants (no candidate selection).
    core = filled & (edge >= .5 * edge.max())
    outer = filled & ~core
    od_core, od_outer = np.nanmedian(od[core & usable]), np.nanmedian(od[outer & usable])
    out["pale_contrast_core"] = float((od_outer - od_core) / od_outer) if od_outer > 0 else np.nan
    cell_depth = edge[px[:, 1], px[:, 0]]
    in_core, in_outer = inside & (cell_depth >= .5 * edge.max()), inside & (cell_depth < .5 * edge.max())
    out["size_gradient_core"] = (float(np.log2(np.median(cells_area[in_core]) / np.median(cells_area[in_outer])))
                                 if in_core.sum() >= 5 and in_outer.sum() >= 10 else np.nan)
    a_core, a_outer = (core & tissue).sum(), (outer & tissue).sum()
    out["density_dip_core"] = (1 - (in_core.sum() / a_core) / (in_outer.sum() / a_outer)
                               if a_core and a_outer and in_outer.sum() else np.nan)
    return out, {"od": od, "centre": (cx, cy), "mpp": mpp}


def gc_features_all(rgb, mpp, filled, cells_xy, cells_area, cells_type):
    """Pre-specified features (R = 30 µm) plus the primary cues at the post-hoc sensitivity radii."""
    values, geo = gc_features(rgb, mpp, filled, cells_xy, cells_area, cells_type)
    for radius in SENSITIVITY_RADII:
        extra, _ = gc_features(rgb, mpp, filled, cells_xy, cells_area, cells_type, radius)
        values.update({f"{k}_r{radius:.0f}": extra[k] for k in PRIMARY})
    return values, geo


def grid_to_crop(mask8, x_um0, y_um0, shape, mpp):
    """Sample an 8 µm-grid mask at the centres of a crop's pixels; crop origin (x_um0, y_um0) in grid µm."""
    ys = np.floor((y_um0 + (np.arange(shape[0]) + .5) * mpp) / A.GRID_UM).astype(int)
    xs = np.floor((x_um0 + (np.arange(shape[1]) + .5) * mpp) / A.GRID_UM).astype(int)
    ok_y, ok_x = (ys >= 0) & (ys < mask8.shape[0]), (xs >= 0) & (xs < mask8.shape[1])
    out = mask8[np.ix_(ys.clip(0, mask8.shape[0] - 1), xs.clip(0, mask8.shape[1] - 1))]
    return out & ok_y[:, None] & ok_x[None, :]


def crop_box(filled8):
    """µm box (x0, y0, x1, y1) on the 8 µm grid around a mask, with CROP_MARGIN_UM."""
    ys, xs = np.nonzero(filled8)
    return (xs.min() * A.GRID_UM - CROP_MARGIN_UM, ys.min() * A.GRID_UM - CROP_MARGIN_UM,
            (xs.max() + 1) * A.GRID_UM + CROP_MARGIN_UM, (ys.max() + 1) * A.GRID_UM + CROP_MARGIN_UM)


# ---------------------------------------------------------------- ESCA

def esca_crop(slide, window, mpp0, box):
    """RGB crop of a window-relative µm box at the pyramid level nearest TARGET_MPP."""
    import openslide  # noqa: F401  (slide is an OpenSlide)
    level = slide.get_best_level_for_downsample(1.05 * TARGET_MPP / mpp0)  # coarsest level not much coarser than the target
    mpp = mpp0 * slide.level_downsamples[level]
    x0, y0 = window.x0 + box[0] / mpp0, window.y0 + box[1] / mpp0
    size = (int(np.ceil((box[2] - box[0]) / mpp)), int(np.ceil((box[3] - box[1]) / mpp)))
    rgba = np.asarray(slide.read_region((int(round(x0)), int(round(y0))), level, size))
    rgb = rgba[..., :3].copy()
    rgb[rgba[..., 3] == 0] = 255
    return rgb, mpp


def esca_slide(slide_folder, keep_geometry=None):
    """Features of every accepted aggregate on one slide (window loop mirrors esca_analysis.load_cohort)."""
    import openslide
    slide_folder = Path(slide_folder)
    barcode = slide_folder.name
    summary = json.loads((slide_folder / "hooknet" / "summary.json").read_text())
    maps = {"prediction_mask": np.load(slide_folder / "hooknet" / "maps_4um.npz")["prediction_mask"]}
    selection = json.loads((slide_folder / "he2cell" / "window_selection.json").read_text())
    flagged = set(selection["artifact_flagged_hooknet_tls"])
    features = [f for f in json.loads((slide_folder / "hooknet" / "native_polygons.geojson").read_text())["features"]
                if f["properties"]["predicted_label"] == "tls" and f["properties"]["passes_published_filter"]
                and f["properties"]["region_id"] not in flagged]
    qc = slide_qc(slide_folder)
    cells = pd.read_csv(slide_folder / A.RESULT / "cells.csv.gz")
    slide = openslide.OpenSlide(summary["slide"])
    mpp0 = summary["source_mpp"]
    rows, geometry = [], {}
    for w in pd.read_csv(slide_folder / "he2cell" / "windows.csv").itertuples():
        inputs = A.window_inputs(slide_folder, w, summary, maps, features, qc, cells)
        proposals, _ = A.proposal_features(inputs, **STAGE1)
        A._PROPOSALS.clear()  # cache is keyed by id(inputs); ids are reused once a window is freed
        c = cells[cells.window_id == w.window_id]
        for index, proposal in enumerate(proposals, 1):
            accepted, _ = A.accept(proposal, FROZEN["b_share"], FROZEN.get("require_b_core", False))
            if not accepted:
                continue
            check = composition(proposal["mask"], inputs["counts"], inputs["valid"])
            assert all(np.isclose(check[k], proposal[k], equal_nan=True) for k in check), (barcode, w.window_id, index)
            filled8 = ndimage.binary_fill_holes(proposal["mask"])
            box = crop_box(filled8)
            rgb, mpp = esca_crop(slide, w, mpp0, box)
            filled = grid_to_crop(filled8, box[0], box[1], rgb.shape[:2], mpp)
            xy = np.stack([((c.centroid_x_global.to_numpy() - w.x0) * mpp0 - box[0]) / mpp,
                           ((c.centroid_y_global.to_numpy() - w.y0) * mpp0 - box[1]) / mpp], 1)
            values, geo = gc_features_all(rgb, mpp, filled, xy, c.area_pixels.to_numpy() * SEG_MPP ** 2, c.celltype.to_numpy())
            key = (barcode, w.window_id, index)
            rows.append({"barcode": barcode, "case_id": barcode[:12], "window_id": w.window_id, "proposal": index,
                         "split": "dev" if A.is_dev(barcode[:12]) else "test", "category": w.category,
                         "matches_HookNet": proposal["matches_HookNet"],
                         "HookNet_GC_pixel_fraction": proposal["HookNet_GC_pixel_fraction"],
                         "filled_hole_um2": float((filled8 & ~proposal["mask"]).sum() * A.GRID_UM ** 2),
                         "image_mpp": mpp, **{k: proposal[k] for k in ["B_share", "T_share", "B_core", "B_periphery"]},
                         **check, **values})
            if keep_geometry is not None and key in keep_geometry:
                geometry[key] = {"rgb": rgb, "filled": filled, **(geo or {})}
    return (rows, geometry) if keep_geometry is not None else rows


def esca_features(workers):
    folders = sorted(p.parent for p in (OUT / "slides").glob(f"*/{A.RESULT}") if (p / "cells.csv.gz").exists())
    started = time.time()
    rows = []
    with ProcessPoolExecutor(workers) as pool:
        for i, part in enumerate(pool.map(esca_slide, folders), 1):
            rows += part
            if i % 20 == 0:
                print(f"{i}/{len(folders)} slides, {len(rows)} aggregates, {time.time() - started:.0f} s", flush=True)
    frame = pd.DataFrame(rows)
    frame["has_HookNet_GC"] = frame.HookNet_GC_pixel_fraction >= GC_LABEL_FRACTION
    # Cross-check against the frozen analysis table: same accepted aggregates, same composition.
    table = pd.read_csv(ANALYSIS / "two_stage_proposals.csv")
    table = table[table.accepted]
    merged = frame.merge(table, on=["barcode", "window_id", "proposal"], how="outer", suffixes=("", "_table"), indicator=True)
    assert (merged._merge == "both").all(), merged._merge.value_counts()
    assert np.allclose(merged.B_share, merged.B_share_table, equal_nan=True)
    MATURITY.mkdir(parents=True, exist_ok=True)
    frame.to_csv(MATURITY / "esca_aggregate_features.csv", index=False)
    print(f"{len(frame)} accepted aggregates written ({time.time() - started:.0f} s); matches two_stage_proposals.csv")
    return frame


# ---------------------------------------------------------------- frozen logistic combination

def fit_logistic(X, y, l2=1.0):
    """L2-penalised logistic regression (intercept unpenalised) by BFGS."""
    X1 = np.c_[np.ones(len(X)), X]

    def loss(beta):
        z = X1 @ beta
        return np.sum(np.logaddexp(0, z) - y * z) + .5 * l2 * np.sum(beta[1:] ** 2)

    def grad(beta):
        g = X1.T @ (1 / (1 + np.exp(-(X1 @ beta))) - y)
        g[1:] += l2 * beta[1:]
        return g
    return optimize.minimize(loss, np.zeros(X1.shape[1]), jac=grad, method="BFGS").x


def fit(frame=None):
    frame = pd.read_csv(MATURITY / "esca_aggregate_features.csv") if frame is None else frame
    dev = frame[frame.split == "dev"]
    X = dev[PRIMARY].to_numpy(float)
    median = np.nanmedian(X, 0)
    X = np.where(np.isfinite(X), X, median)
    mean, sd = X.mean(0), X.std(0)
    beta = fit_logistic((X - mean) / sd, dev.has_HookNet_GC.to_numpy(float))
    model = {"features": PRIMARY, "impute_median": median.tolist(), "mean": mean.tolist(), "sd": sd.tolist(),
             "intercept": float(beta[0]), "coefficients": dict(zip(PRIMARY, beta[1:].tolist())), "l2": 1.0,
             "label": f"HookNet GC pixels >= {GC_LABEL_FRACTION:.0%} of the aggregate (weak proxy)",
             "dev_aggregates": len(dev), "dev_patients": int(dev.case_id.nunique()), "dev_positive": int(dev.has_HookNet_GC.sum())}
    save_json(model, MATURITY / "frozen_gc_score.json")
    print(json.dumps(model, indent=1))
    return model


def score(frame, model):
    X = frame[model["features"]].to_numpy(float)
    X = np.where(np.isfinite(X), X, np.asarray(model["impute_median"]))
    z = (X - np.asarray(model["mean"])) / np.asarray(model["sd"])
    return model["intercept"] + z @ np.asarray([model["coefficients"][f] for f in model["features"]])


# ---------------------------------------------------------------- evaluation and report

def cluster_bootstrap(frame, cluster, function, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    ids = frame[cluster].unique()
    groups = {c: frame[frame[cluster] == c] for c in ids}
    values = [function(pd.concat([groups[c] for c in rng.choice(ids, len(ids))])) for _ in range(n)]
    return np.nanpercentile(np.asarray(values, float), [2.5, 97.5]).tolist()


def auroc_table(frame, label, cluster, features):
    rows = []
    for f in features:
        sub = frame[frame[f].notna()]
        rows.append({"feature": f, "name": NAMES.get(f, f), "n": len(sub), "positive": int(sub[label].sum()),
                     "AUROC": float(A.auc(sub[label], sub[f])),
                     "CI95": cluster_bootstrap(sub, cluster, lambda s: A.auc(s[label], s[f])),
                     "median_positive": float(sub.loc[sub[label], f].median()), "median_negative": float(sub.loc[~sub[label], f].median())})
    return rows


def png(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def gallery_tile(rgb, filled, geo, caption):
    """RGB with aggregate outline, candidate disk and mantle ring | haematoxylin OD map with the same marks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.7))
    od = geo["od"] if geo else np.full(rgb.shape[:2], np.nan)
    finite = od[np.isfinite(od) & filled]
    lo, hi = (np.percentile(finite, [2, 98]) if finite.size else (0, 1))
    axes[0].imshow(rgb)
    axes[1].imshow(od, cmap="magma", vmin=lo, vmax=hi)
    for ax in axes:
        ax.contour(filled, levels=[.5], colors=["#1baf7a"], linewidths=1)
        if geo:
            cx, cy = geo["centre"]
            for radius, style in [(GC_RADIUS_UM, "-"), (3 * GC_RADIUS_UM, "--")]:
                ax.add_patch(plt.Circle((cx, cy), radius / geo["mpp"], fill=False, color="#2a78d6", lw=1.1, ls=style))
        ax.set_axis_off()
    fig.suptitle(caption, fontsize=7, y=.02, va="bottom")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=.1, wspace=.02)
    path = MATURITY / "gallery" / (caption.replace(" ", "_").replace("·", "").replace("/", "-")[:90] + ".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def esca_gallery(frame, n=6):
    """Top / bottom GC-score accepted test aggregates; re-reads their crops."""
    test = frame[frame.split == "test"].sort_values("GC_score", ascending=False)
    picks = {"Highest GC score": test.head(n), "Lowest GC score": test.tail(n)}
    out = {}
    for group, part in picks.items():
        tiles = []
        for barcode, rows in part.groupby("barcode", sort=False):
            keys = {(barcode, r.window_id, r.proposal) for r in rows.itertuples()}
            _, geometry = esca_slide(OUT / "slides" / barcode, keep_geometry=keys)
            for r in rows.itertuples():
                g = geometry[(barcode, r.window_id, r.proposal)]
                caption = (f"{barcode[:12]} {r.window_id}#{r.proposal} · score {r.GC_score:.2f} · "
                           f"HookNet GC {r.HookNet_GC_pixel_fraction:.0%} · contrast {r.pale_contrast:.2f}")
                tiles.append((r.GC_score, gallery_tile(g["rgb"], g["filled"], g if "od" in g else None, caption), caption))
        out[group] = [t[1:] for t in sorted(tiles, key=lambda t: -t[0])]
    return out


def roc_figure(esca_test, visium, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colours = {"GC_score": "#0b0b0b", "pale_contrast": "#2a78d6", "size_gradient": "#eb6834", "B_dip": "#1baf7a",
               "density_dip": "#6250d6", "B_share": "#9a9893", "B_core_enrichment": "#c3c2b7"}
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9))
    for ax, (title, frame, label) in zip(axes, [("ESCA test aggregates vs HookNet GC (proxy)", esca_test, "has_HookNet_GC"),
                                                ("Visium TLS objects, pathologist mature vs early", visium, "mature")]):
        for f, colour in colours.items():
            sub = frame[frame[f].notna()]
            y, s = sub[label].to_numpy(bool), sub[f].to_numpy(float)
            order = np.argsort(-s, kind="stable")
            ax.plot(np.r_[0, np.cumsum(~y[order]) / max((~y).sum(), 1)], np.r_[0, np.cumsum(y[order]) / max(y.sum(), 1)],
                    color=colour, lw=2.2 if f == "GC_score" else 1.3, ls="-" if f in PRIMARY + ["GC_score"] else ":",
                    label=f"{NAMES[f].split(' (')[0]}  {A.auc(y, s):.2f}")
        ax.plot([0, 1], [0, 1], color="#52514e", lw=.8, ls=":")
        ax.set(title=title, xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1.01))
        ax.title.set_fontsize(8.5)
        ax.legend(frameon=False, fontsize=6.5, loc="lower right", title="AUROC", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def visium_gallery(frame, n=6):
    """Top / bottom GC-score Visium objects (aggregate region), with the pathologist label in the caption."""
    sys.path.insert(0, str(HERE.parent / "visium_tls"))
    import visium_maturity as VM
    part = frame.sort_values("GC_score", ascending=False)
    picks = {"Highest GC score": part.head(n), "Lowest GC score": part.tail(n)}
    out = {}
    for group, rows in picks.items():
        tiles = []
        for sample, sub in rows.groupby("sample", sort=False):
            _, geometry = VM.sample_objects(sample, keep={(sample, int(o)) for o in sub.object})
            for r in sub.itertuples():
                g = geometry[(sample, int(r.object))]
                caption = (f"{sample} object {r.object} · {'MATURE' if r.mature else 'early'} ({r.labels}) · "
                           f"score {r.GC_score:.2f} · contrast {r.pale_contrast:.2f}")
                tiles.append((r.GC_score, gallery_tile(g["rgb"], g["filled"], g if "od" in g else None, caption), caption))
        out[group] = [t[1:] for t in sorted(tiles, key=lambda t: -t[0])]
    return out


def fmt(v, d=2):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def auc_finite(labels, values):
    """AUROC over the rows where the feature is defined."""
    labels, values = np.asarray(labels, bool), np.asarray(values, float)
    ok = np.isfinite(values)
    return float(A.auc(labels[ok], values[ok]))


def stratified_auroc(frame, label, feature, by):
    """AUROC within halves of a size variable (median split), to see whether a feature only tracks size."""
    cut = frame[by].median()
    return {half: auc_finite(sub[label], sub[feature]) for half, sub in
            [("smaller half", frame[frame[by] <= cut]), ("larger half", frame[frame[by] > cut])]}


def report():
    model = json.loads((MATURITY / "frozen_gc_score.json").read_text())
    esca = pd.read_csv(MATURITY / "esca_aggregate_features.csv")
    visium_all = pd.read_csv(MATURITY / "visium_object_features.csv")
    esca["GC_score"] = score(esca, model)
    visium_all["GC_score"] = score(visium_all, model)
    esca.to_csv(MATURITY / "esca_aggregate_features.csv", index=False)
    visium_all.to_csv(MATURITY / "visium_object_features.csv", index=False)
    visium = visium_all[visium_all.region == "aggregate"].copy()
    tiles = visium_all[visium_all.region == "spot tiles"].copy()
    test, dev = esca[esca.split == "test"], esca[esca.split == "dev"]
    features = ["GC_score"] + PRIMARY + DESCRIPTIVE + COMPOSITION
    vis_features = features + ["B_fraction_spots", "B_density_spots", "spots"]
    results = {"esca_test": auroc_table(test, "has_HookNet_GC", "case_id", features),
               "esca_dev_in_sample": [{"feature": f, "AUROC": float(A.auc(dev[dev[f].notna()].has_HookNet_GC, dev[f].dropna()))} for f in features],
               "visium": auroc_table(visium, "mature", "sample", vis_features),
               "visium_spot_tiles": auroc_table(tiles, "mature", "sample", ["GC_score"] + PRIMARY + COMPOSITION)}
    get = {k: {r["feature"]: r for r in v} for k, v in results.items() if k != "esca_dev_in_sample"}
    gc_esca, gc_vis = get["esca_test"]["GC_score"], get["visium"]["GC_score"]
    passed = gc_vis["AUROC"] >= .75 and gc_vis["CI95"][0] > .6 and gc_esca["AUROC"] >= .70
    single_pass = [f for f in PRIMARY if get["visium"][f]["AUROC"] >= .75 and get["visium"][f]["CI95"][0] > .6
                   and get["esca_test"][f]["AUROC"] >= .70]
    # Sensitivity: other label definitions, subsets, size confounding.
    sensitivity = {
        "visium_majority_rule_label": {f: auc_finite(visium.mature_majority_rule, visium[f]) for f in ["GC_score"] + PRIMARY + ["B_share", "spots"]},
        "visium_GC_spot_vs_rest": {"n_with_GC_spot": int(visium.has_GC_spot.sum()),
                                   **{f: auc_finite(visium.has_GC_spot, visium[f]) for f in ["GC_score"] + PRIMARY + ["area_um2", "spots"]}},
        "visium_proposed_objects_only": {"n": int(visium.proposed.sum()), "mature": int(visium[visium.proposed].mature.sum()),
                                         **{f: auc_finite(visium[visium.proposed].mature, visium[visium.proposed][f])
                                            for f in ["GC_score"] + PRIMARY}},
        "esca_label_5pct": {f: auc_finite(test.HookNet_GC_pixel_fraction >= .05, test[f]) for f in ["GC_score"] + PRIMARY + ["area_um2"]},
        "esca_spearman_GC_fraction": {f: float(stats.spearmanr(test[f], test.HookNet_GC_pixel_fraction, nan_policy="omit").statistic)
                                      for f in ["GC_score"] + PRIMARY + ["area_um2"]},
        "size_stratified": {"esca_test_GC_score_by_area": stratified_auroc(test, "has_HookNet_GC", "GC_score", "area_um2"),
                            "esca_test_pale_contrast_by_area": stratified_auroc(test, "has_HookNet_GC", "pale_contrast", "area_um2"),
                            "visium_GC_score_by_spots": stratified_auroc(visium, "mature", "GC_score", "spots"),
                            "spearman_GC_score_vs_area_esca_test": float(stats.spearmanr(test.GC_score, test.area_um2).statistic),
                            "spearman_GC_score_vs_area_visium": float(stats.spearmanr(visium.GC_score, visium.area_um2).statistic)}}
    # Post-hoc radii: same protocol (dev fit, frozen, scored on test and Visium); exploratory only.
    posthoc = []
    for radius in SENSITIVITY_RADII:
        names = [f"{k}_r{radius:.0f}" for k in PRIMARY]
        X = dev[names].to_numpy(float)
        median = np.nanmedian(X, 0)
        X = np.where(np.isfinite(X), X, median)
        beta = fit_logistic((X - X.mean(0)) / X.std(0), dev.has_HookNet_GC.to_numpy(float))
        m = {"features": names, "impute_median": median.tolist(), "mean": X.mean(0).tolist(), "sd": X.std(0).tolist(),
             "intercept": float(beta[0]), "coefficients": dict(zip(names, beta[1:].tolist()))}
        esca[f"GC_score_r{radius:.0f}"], visium[f"GC_score_r{radius:.0f}"] = score(esca, m), score(visium, m)
        test = esca[esca.split == "test"]
        for f in [f"GC_score_r{radius:.0f}"] + names:
            posthoc.append({"radius_um": radius, "feature": f, "esca_test_AUROC": float(A.auc(test[test[f].notna()].has_HookNet_GC, test[f].dropna())),
                            "visium_AUROC": float(A.auc(visium[visium[f].notna()].mature, visium[f].dropna())),
                            "visium_CI95": cluster_bootstrap(visium[visium[f].notna()], "sample", lambda s: A.auc(s.mature, s[f]))})
    # Figures and gallery.
    (MATURITY / "figures").mkdir(parents=True, exist_ok=True)
    roc_figure(test, visium, MATURITY / "figures" / "roc.png")
    galleries = {"ESCA test aggregates": esca_gallery(esca), "Visium TLS objects": visium_gallery(visium)}
    summary = {"item": "5a GC morphology score", "date": DATE, "classifier": A.RESULT, "stage1_and_acceptance_frozen": FROZEN,
               "feature_definitions": __doc__, "od_sigma_um": OD_SIGMA_UM, "gc_disk_diameter_um": 2 * GC_RADIUS_UM,
               "mantle_ring_um": [GC_RADIUS_UM, 3 * GC_RADIUS_UM], "frozen_model": model,
               "cohorts": {"esca_dev_aggregates": len(dev), "esca_test_aggregates": len(test), "esca_test_patients": int(test.case_id.nunique()),
                           "esca_test_with_HookNet_GC": int(test.has_HookNet_GC.sum()), "visium_objects": len(visium),
                           "visium_mature": int(visium.mature.sum()), "visium_early": int((~visium.mature).sum()),
                           "visium_sections": int(visium["sample"].nunique()),
                           "visium_aggregate_source": visium.aggregate_source.value_counts().to_dict()},
               "acceptance": {"rule": "Visium AUROC >= 0.75 with CI lower bound > 0.6 and ESCA test AUROC >= 0.70",
                              "GC_score_visium_AUROC": gc_vis["AUROC"], "GC_score_visium_CI95": gc_vis["CI95"],
                              "GC_score_esca_test_AUROC": gc_esca["AUROC"], "GC_score_esca_test_CI95": gc_esca["CI95"],
                              "passed": bool(passed), "single_features_passing": single_pass},
               "results": results, "sensitivity": sensitivity, "posthoc_radii_exploratory": posthoc}
    save_json(summary, MATURITY / "summary.json")
    if passed:
        table = pd.read_csv(ANALYSIS / "two_stage_proposals.csv")
        table = table.merge(esca[["barcode", "window_id", "proposal", "GC_score"]], on=["barcode", "window_id", "proposal"], how="left")
        table.to_csv(MATURITY / "two_stage_proposals_with_GC_score.csv", index=False)
    write_html(summary, galleries)
    print(json.dumps(summary["acceptance"], indent=1))


def write_html(summary, galleries):
    r = summary["results"]
    acc = summary["acceptance"]
    c = summary["cohorts"]
    dev = {x["feature"]: x["AUROC"] for x in r["esca_dev_in_sample"]}
    vis = {x["feature"]: x for x in r["visium"]}
    tiles = {x["feature"]: x for x in r["visium_spot_tiles"]}

    def group(f):
        return "combination" if f == "GC_score" else "GC cue (in score)" if f in PRIMARY else \
            "GC cue (descriptive)" if f in DESCRIPTIVE else "size baseline" if f in ("area_um2", "spots") else "composition baseline"
    main_rows = "".join(
        f"<tr><td>{html.escape(NAMES[x['feature']])}</td><td>{group(x['feature'])}</td><td>{fmt(dev.get(x['feature']))}</td>"
        f"<td><b>{fmt(x['AUROC'])}</b> [{fmt(x['CI95'][0])}–{fmt(x['CI95'][1])}]</td>"
        f"<td><b>{fmt(vis[x['feature']]['AUROC'])}</b> [{fmt(vis[x['feature']]['CI95'][0])}–{fmt(vis[x['feature']]['CI95'][1])}]</td>"
        f"<td>{fmt(tiles[x['feature']]['AUROC']) if x['feature'] in tiles else '–'}</td></tr>" for x in r["esca_test"])
    spot_rows = "".join(f"<tr><td>{html.escape(NAMES[f])}</td><td>{group(f)} (spots)</td><td>–</td><td>–</td>"
                        f"<td><b>{fmt(vis[f]['AUROC'])}</b> [{fmt(vis[f]['CI95'][0])}–{fmt(vis[f]['CI95'][1])}]</td><td>–</td></tr>"
                        for f in ["B_fraction_spots", "B_density_spots", "spots"])
    median_rows = "".join(
        f"<tr><td>{html.escape(NAMES[x['feature']])}</td><td>{fmt(x['median_negative'], 3)}</td><td>{fmt(x['median_positive'], 3)}</td>"
        f"<td>{fmt(vis[x['feature']]['median_negative'], 3)}</td><td>{fmt(vis[x['feature']]['median_positive'], 3)}</td></tr>"
        for x in r["esca_test"] if x["feature"] in ["GC_score"] + PRIMARY)
    coef_rows = "".join(f"<tr><td>{html.escape(NAMES[f])}</td><td>{fmt(v, 3)}</td></tr>" for f, v in summary["frozen_model"]["coefficients"].items())
    s = summary["sensitivity"]

    def kv(d, skip=()):
        return "; ".join(f"{html.escape(NAMES.get(k, k))} {fmt(v) if isinstance(v, float) else v}" for k, v in d.items() if k not in skip)
    posthoc_rows = "".join(
        f"<tr><td>{p['radius_um']:.0f} µm</td><td>{html.escape(p['feature'])}</td><td>{fmt(p['esca_test_AUROC'])}</td>"
        f"<td>{fmt(p['visium_AUROC'])} [{fmt(p['visium_CI95'][0])}–{fmt(p['visium_CI95'][1])}]</td></tr>" for p in summary["posthoc_radii_exploratory"])
    gallery_html = ""
    for cohort, groups in galleries.items():
        for name, items in groups.items():
            figs = "".join(f'<figure><img src="{png(path)}" alt="{html.escape(cap)}"><figcaption>{html.escape(cap)}</figcaption></figure>'
                           for path, cap in items)
            gallery_html += f"<h3>{cohort}: {name.lower()}</h3><div class='gallery'>{figs}</div>"
    verdict = "PASSED" if acc["passed"] else "FAILED"
    limitation = ("" if acc["passed"] else
                  f"<h2>Limitation wording (plan 5b)</h2><blockquote>Composition and density grade TLS presence, not maturity. "
                  f"GC-aware morphology features (pale-centre haematoxylin contrast, nuclear-size gradient, radial B-cell profile and "
                  f"nuclear-density dip, combined by a logistic model frozen on TCGA-ESCA development patients) were tested and did not reach "
                  f"the pre-specified acceptance: AUROC {fmt(acc['GC_score_visium_AUROC'])} (95% CI {fmt(acc['GC_score_visium_CI95'][0])}–"
                  f"{fmt(acc['GC_score_visium_CI95'][1])}) for pathologist mature vs early TLS on 8 Visium sections "
                  f"({c['visium_mature']} mature, {c['visium_early']} early) and {fmt(acc['GC_score_esca_test_AUROC'])} "
                  f"({fmt(acc['GC_score_esca_test_CI95'][0])}–{fmt(acc['GC_score_esca_test_CI95'][1])}) against HookNet germinal-centre pixels on "
                  f"{c['esca_test_aggregates']} TCGA-ESCA test aggregates (required ≥ 0.75 with lower bound &gt; 0.6, and ≥ 0.70). "
                  f"Maturity grading would need a germinal-centre B-cell class trained on Ki67/BCL6/CD21 or pathologist-marked germinal centres.</blockquote>"
                  f"<p>A GC-B class (plan 5b) needs labels that do not exist yet: item 2 immunostains or item 1 SFL reads.</p>")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TLS Maturity Score</title><style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--line:#e4e3df;--card:#fff;--bad:#b3261e;--good:#1b7a4b}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--line:#3a3936;--card:#232321;--bad:#f2867e;--good:#6fd3a0}}}}
:root[data-theme="dark"]{{--bg:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--line:#3a3936;--card:#232321;--bad:#f2867e;--good:#6fd3a0}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1080px;margin:0 auto;padding:28px 16px 64px}} h1{{font-size:26px;margin:0 0 4px}} h2{{margin-top:36px;border-top:1px solid var(--line);padding-top:18px}}
.muted{{color:var(--muted)}} table{{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow-x:auto}}
th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;white-space:nowrap}} img.fig{{max-width:100%;background:#fff;border-radius:8px;border:1px solid var(--line)}}
.verdict{{border:2px solid {'var(--good)' if acc['passed'] else 'var(--bad)'};border-radius:8px;padding:12px 16px;background:var(--card)}}
.verdict b.v{{color:{'var(--good)' if acc['passed'] else 'var(--bad)'}}} blockquote{{margin:0;padding:10px 16px;border-left:3px solid var(--line);background:var(--card)}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:10px}} figure{{margin:0}} figure img{{width:100%;border-radius:6px;background:#fff}} figcaption{{font-size:11px;color:var(--muted)}}
</style></head><body><main>
<h1>Germinal-centre morphology score for TLS maturity (plan item 5a)</h1>
<p class="muted">HE2CELL two-stage aggregates (fine-tuned classifier, τ 0.5 prior; frozen ESCA parameters). Score fitted on TCGA-ESCA dev patients with HookNet germinal-centre pixels as a weak label, frozen, then tested on ESCA test patients and on pathologist-labelled Visium TLS. Generated {summary['date']}.</p>
<div class="verdict"><b class="v">{verdict}</b> — acceptance rule: {html.escape(acc['rule'])}.<br>
GC score: Visium mature vs early AUROC <b>{fmt(acc['GC_score_visium_AUROC'])}</b> [{fmt(acc['GC_score_visium_CI95'][0])}–{fmt(acc['GC_score_visium_CI95'][1])}];
ESCA test vs HookNet GC <b>{fmt(acc['GC_score_esca_test_AUROC'])}</b> [{fmt(acc['GC_score_esca_test_CI95'][0])}–{fmt(acc['GC_score_esca_test_CI95'][1])}].
Single GC cues meeting the same rule: {', '.join(NAMES[f] for f in acc['single_features_passing']) or 'none'}.</div>

<h2>Design</h2>
<p>Each accepted aggregate is hole-filled (a pale GC can fall below the nuclear-density threshold). The GC candidate is the palest 60 µm disk of haematoxylin OD (colour deconvolution, Gaussian σ = 8 µm, background excluded) whose centre lies ≥ 30 µm inside the aggregate; the mantle ring is 30–90 µm from that centre, inside the aggregate.</p>
<ul><li><b>Pale-centre contrast</b>: (mantle-ring median OD − candidate-disk OD) / mantle median.</li>
<li><b>Nuclear-size gradient</b>: log2 of median Cellpose nuclear area, disk / mantle ring (Visium areas come from the 0.25 µm/px upsampled segmentation; OD maps use native 0.79 µm/px).</li>
<li><b>Radial B profile</b>: 1 − predicted-B density at 0–30 µm / peak of the 30–60, 60–90, 90–120 µm rings.</li>
<li><b>Nuclear-density dip</b>: 1 − nuclei/mm² in the disk / in the mantle ring.</li></ul>
<p>Combination: L2 logistic regression (standardised, λ = 1) on {summary['frozen_model']['dev_aggregates']} ESCA dev aggregates ({summary['frozen_model']['dev_patients']} patients, {summary['frozen_model']['dev_positive']} with HookNet GC pixels ≥ 1%). ESCA test: {c['esca_test_aggregates']} accepted aggregates, {c['esca_test_patients']} patients, {c['esca_test_with_HookNet_GC']} with HookNet GC. Visium: {c['visium_objects']} pathologist TLS objects (connected TLS spots) on {c['visium_sections']} sections, mature = any MTLS or GC spot ({c['visium_mature']}), early = ETLS only ({c['visium_early']}). Visium objects are measured on the frozen stage-1 proposal(s) overlapping their spots, clipped to the spots + 50 µm ({kv(c['visium_aggregate_source'])}); the spot-tile column measures every object on its 55 µm spot discs. CIs: 2000 cluster-bootstrap resamples (patients / sections).</p>
<table><tr><th>Frozen coefficient (standardised)</th><th>Value</th></tr>{coef_rows}<tr><td>Intercept</td><td>{fmt(summary['frozen_model']['intercept'], 3)}</td></tr></table>
<p class="muted">The coefficients are all near zero: on dev patients the four cues barely separate aggregates with and without HookNet GC pixels.</p>

<h2>Discrimination</h2>
<img class="fig" src="{png(MATURITY / 'figures' / 'roc.png')}" alt="ROC curves for ESCA and Visium">
<table><tr><th>Feature</th><th>Role</th><th>ESCA dev (in-sample)</th><th>ESCA test vs HookNet GC [95% CI]</th><th>Visium mature vs early [95% CI]</th><th>Visium, spot-tile region</th></tr>{main_rows}{spot_rows}</table>
<p class="muted">AUROC &gt; 0.5 means higher values in aggregates with GC (ESCA) or mature TLS (Visium); no feature was flipped after looking. Aggregate area and spot count are size, not morphology: larger aggregates are more likely to contain any HookNet GC pixel and more likely to contain an MTLS spot.</p>
<table><tr><th>Feature</th><th>ESCA median, no GC</th><th>ESCA median, GC</th><th>Visium median, early</th><th>Visium median, mature</th></tr>{median_rows}</table>

<h2>Sensitivity analyses</h2>
<ul><li>Visium, majority-rule label (MTLS+GC ≥ ETLS, as in the Visium report): {kv(s['visium_majority_rule_label'])}.</li>
<li>Visium, objects containing a pathologist GC spot vs all others ({s['visium_GC_spot_vs_rest']['n_with_GC_spot']} objects): {kv(s['visium_GC_spot_vs_rest'], ['n_with_GC_spot'])}.</li>
<li>Visium, objects with a stage-1 proposal only (n = {s['visium_proposed_objects_only']['n']}, {s['visium_proposed_objects_only']['mature']} mature): {kv(s['visium_proposed_objects_only'], ['n', 'mature'])}.</li>
<li>ESCA test, stricter label (HookNet GC ≥ 5% of aggregate): {kv(s['esca_label_5pct'])}.</li>
<li>ESCA test, Spearman ρ with HookNet GC pixel fraction: {kv(s['esca_spearman_GC_fraction'])}.</li>
<li>Size: GC score AUROC within area halves, ESCA test {kv(s['size_stratified']['esca_test_GC_score_by_area'])}; pale contrast {kv(s['size_stratified']['esca_test_pale_contrast_by_area'])}; Visium by spot count {kv(s['size_stratified']['visium_GC_score_by_spots'])}. Spearman GC score vs area: ESCA {fmt(s['size_stratified']['spearman_GC_score_vs_area_esca_test'])}, Visium {fmt(s['size_stratified']['spearman_GC_score_vs_area_visium'])}.</li></ul>

<h2>Post-hoc: larger GC candidate (exploratory)</h2>
<p>After the pre-specified 30 µm-radius score failed, the gallery showed that the palest 60 µm disk often lands on a vessel or stroma island rather than on a GC. The same protocol (dev fit, freeze, test) was repeated with candidate radius R = 50 and 75 µm (mantle R–3R, B rings at multiples of R). These were chosen after seeing the Visium result and cannot change the verdict.</p>
<table><tr><th>Radius</th><th>Feature</th><th>ESCA test AUROC</th><th>Visium AUROC [95% CI]</th></tr>{posthoc_rows}</table>
{limitation}

<h2>Gallery: does the score find pale centres?</h2>
<p class="muted">Left: H&amp;E; right: smoothed haematoxylin OD (bright = dark-staining, each tile scaled to its own 2–98th percentile; white = background). Green: hole-filled aggregate. Blue solid circle: the 60 µm GC candidate (palest disk); dashed: outer edge of the mantle ring (90 µm).</p>
<p>What the tiles show: the palest 60 µm disk frequently lands on a vessel lumen, a stroma island or the gap between follicles rather than on a germinal centre, including in aggregates where GCs are visible by eye as pale holes ringed by a dark mantle in the OD map (e.g. LC3 object 8). Pale-centre contrast therefore measures hypocellular interruptions of the aggregate in general, which is why it tracks aggregate size and does not grade maturity.</p>
{gallery_html}

<h2>Files</h2>
<p><code>esca_aggregate_features.csv</code> (one row per accepted ESCA aggregate, dev + test), <code>visium_object_features.csv</code> (two rows per Visium TLS object: region = aggregate | spot tiles), <code>frozen_gc_score.json</code>, <code>summary.json</code>, <code>gallery/</code>, <code>figures/</code>. Code: <code>deployment/esca_tls/esca_maturity.py</code>, <code>deployment/visium_tls/visium_maturity.py</code>.</p>
</main></body></html>"""
    (MATURITY / "Maturity_report.html").write_text(page)
    print("Report written", MATURITY / "Maturity_report.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["features", "fit", "report"])
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.stage == "features":
        fit(esca_features(args.workers))
    elif args.stage == "fit":
        fit()
    else:
        report()
