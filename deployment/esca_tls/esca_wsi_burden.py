"""Whole-slide HE2CELL TLS burden for TCGA-ESCA: cheap haematoxylin screen, then cells only where it fires.

1. Screen (no model, CPU, ~45 s per slide): a cheap surrogate of the frozen stage 1 (all-nuclei density).
   Nuclear blobs = LoG maxima of haematoxylin OD (colour deconvolution) on the ~1 µm pyramid level, counted
   on the 8 µm QC grid of esca_qc, restricted to its artefact-free tissue mask, smoothed at the stage-1 sigma
   (25 µm). Components of blob density >= d_extent that contain >= a minimum area of density >= d are grown
   by a 100 µm (square) margin. Tuned on dev patients only: `--calibrate` scores a grid by processed cores and
   by the share of dev accepted TLS with >= 90% of their area processed; `--calibrate-e2e` re-runs the frozen
   two-stage rule on dev windows with unprocessed cells removed for the 10 cheapest settings and keeps the
   cheapest with >= 95% dev recovery (98% was not reachable at < 2x the cost). Frozen in
   wsi_screen_calibration/screen_params.json (d = d_extent = 9000 blobs/mm², >= 5000 µm²). Haematoxylin-OD screens on the 4 µm level were tried first
   and failed on dev (stain intensity varies between slides; local-contrast filters suppress the interior
   of large aggregates): the best absolute-OD and per-slide-quantile DoG settings covered 95% and 89% of
   dev TLS while processing a median 136 and 56 mm² per slide (*_absoluteOD.csv, *_DoG.csv).
2. Cells: every 768-px core of a global level-0 grid that touches the grown screen mask is segmented
   (cpsam_v2) as a 1024-px patch with a 128-px margin, exactly the esca_windows convention, and classified
   with models/he2cell_esca_finetuned.pth (+ its tau 0.5 prior sidecar, D4 TTA). No stain normalisation.
3. Detection: the frozen two-stage rule of esca_analysis (stage 1 all-nuclei density, stage 2 B share),
   run per connected processed region on the 8 µm grid. Burden = accepted TLS per cm² of artefact-free
   tissue and accepted TLS area / tissue area.

Outputs per slide: slides/<barcode>/wsi_burden/{cells.csv.gz, tls.csv, summary.json}.
`--validate` checks the frozen screen on held-out test windows (end-to-end: cells outside the processed
region are removed and the two-stage rule is re-run). `--run` processes slides sequentially under the
shared GPU lock (outputs/_gpu.lock), resumable, one slide lock per slide in OUT/_locks.
"""
from __future__ import annotations

import argparse
from collections import namedtuple
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ["ESCA_RESULT"] = "he2cell_ft_tau05"  # window predictions of the adopted classifier (esca_analysis)

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import esca_analysis as A  # noqa: E402
from esca_common import OUT, ROOT, save_json  # noqa: E402
from esca_qc import slide_qc  # noqa: E402
from esca_windows import CORE_MARGIN, KEEP, PATCH, STRIDE, open_slide, stage  # noqa: E402

CLASSIFIER = ROOT / "models" / "he2cell_esca_finetuned.pth"
ANALYSIS = OUT / "analysis_he2cell_ft_tau05"
RESULT = "wsi_burden"
GPU_LOCK = ROOT / "outputs" / "_gpu.lock"
PARAMS = json.loads((ANALYSIS / "report_summary.json").read_text())["chosen_parameters"][A.TWO_STAGE]
STAGE1 = {k: PARAMS[k] for k in ["sigma_um", "threshold", "min_area_um2", "min_cells"]}
SCREEN_SIGMA_UM, SCREEN_MARGIN_UM = 25.0, 100.0
# Nuclear blobs: LoG (sigma 2.5 µm ~ lymphocyte nucleus radius / sqrt 2) maxima >= 5 µm apart on ~1 µm haematoxylin.
BLOB_UM, BLOB_SIGMA_UM, BLOB_SPACING_UM, BLOB_MIN_LOG, BLOB_MIN_OD = 1.0, 2.5, 5.0, 0.002, 0.02
# (seed density, extent density, minimum seed area); extent = seed is a plain threshold.
SCREEN_GRID = [(d, e, a) for d in [7000.0, 8000.0, 9000.0, 10000.0, 11000.0, 12000.0]
               for e in [5000.0, 6000.0, 7000.0, d] if e <= d
               for a in [5000.0, 10000.0, 20000.0]]
SCREEN_GRID = sorted(set(SCREEN_GRID))
CALIBRATION = OUT / "wsi_screen_calibration"
SCREEN_PARAMS_FILE = CALIBRATION / "screen_params.json"
DEV_TARGET, COVERED = 0.98, 0.9  # dev aggregates with >=90% of their area inside the processed region
# --calibrate-e2e first applied DEV_TARGET (not reached: best 97.5% at 2x cost); the frozen choice used 95%, see screen_params.json.
Window = namedtuple("Window", "x0 y0 x1 y1 window_id")


def slide_path(barcode):
    return Path(json.loads((OUT / "slides" / barcode / "hooknet" / "summary.json").read_text())["slide"])


def blob_counts(barcode, tile=2048, pad=16):
    """Nuclear blobs per 8 µm QC cell: LoG maxima of haematoxylin OD on the ~1 µm level, tissue tiles only."""
    from skimage.color import rgb2hed
    folder = OUT / "slides" / barcode
    summary = json.loads((folder / "hooknet" / "summary.json").read_text())
    ratio = summary["source_dimensions"][0] / summary["dimensions_at_0p5um"][0]
    valid8 = slide_qc(folder)["valid"]
    to8 = 1 / (16 * ratio)  # level-0 px -> 8 µm QC px
    counts = np.zeros(valid8.shape, np.float32)
    with open_slide(slide_path(barcode)) as handle:
        mpp0 = float(handle.properties["openslide.mpp-x"])
        level = min(range(handle.level_count), key=lambda l: abs(handle.level_downsamples[l] * mpp0 - BLOB_UM))
        down = handle.level_downsamples[level]
        mpp = down * mpp0
        width, height = handle.level_dimensions[level]
        for ty in range(0, height, tile):
            for tx in range(0, width, tile):
                if not valid8[int(ty * down * to8):int((ty + tile) * down * to8) + 1,
                              int(tx * down * to8):int((tx + tile) * down * to8) + 1].any():
                    continue
                x0, y0 = max(tx - pad, 0), max(ty - pad, 0)
                w, h = min(tile + 2 * pad, width - x0), min(tile + 2 * pad, height - y0)
                rgb = np.asarray(handle.read_region((int(x0 * down), int(y0 * down)), level, (w, h)).convert("RGB"))
                hema = np.clip(rgb2hed(rgb)[..., 0], 0, None).astype(np.float32)
                log = -ndimage.gaussian_laplace(hema, BLOB_SIGMA_UM / mpp)
                peaks = ((log == ndimage.maximum_filter(log, size=int(round(BLOB_SPACING_UM / mpp))))
                         & (log > BLOB_MIN_LOG) & (hema > BLOB_MIN_OD))
                py, px = np.nonzero(peaks)
                inside = (px + x0 >= tx) & (px + x0 < tx + tile) & (py + y0 >= ty) & (py + y0 < ty + tile)
                gx = ((px[inside] + x0) * down * to8).astype(int)
                gy = ((py[inside] + y0) * down * to8).astype(int)
                ok = (gx < counts.shape[1]) & (gy < counts.shape[0])
                np.add.at(counts, (gy[ok], gx[ok]), 1)
    return counts, valid8, ratio, mpp0, summary["source_dimensions"]


def screen_maps(barcode):
    """Screen inputs on the 8 µm QC grid: blob density (per mm², Gaussian sigma = stage-1 sigma) and valid tissue.

    Blob counts are cached in slides/<barcode>/wsi_burden/blob_counts_8um.npz (CPU only, ~45 s per slide).
    """
    cache = OUT / "slides" / barcode / RESULT / "blob_counts_8um.npz"
    if cache.exists():
        data = np.load(cache)
        counts, valid, ratio, mpp, level0 = data["counts"], data["valid"], float(data["ratio"]), float(data["mpp"]), data["level0"].tolist()
    else:
        counts, valid, ratio, mpp, level0 = blob_counts(barcode)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, counts=counts, valid=valid, ratio=ratio, mpp=mpp, level0=np.asarray(level0))
    down = 16 * ratio  # level-0 px per grid px
    um = down * mpp
    density = ndimage.gaussian_filter(counts, SCREEN_SIGMA_UM / um, mode="constant") / (um ** 2 / 1e6)
    return {"density": density, "valid": valid.astype(bool), "down": down, "um": um, "mpp": mpp, "ratio": ratio,
            "level0": level0}


def candidates(maps, d, min_um2, d_extent=None):
    """Screen-positive grid cells (hysteresis): components of blob density >= d_extent on valid tissue that
    contain >= min_um2 of density >= d (the seed). d_extent = d is a plain threshold."""
    d_extent = d if d_extent is None else min(d_extent, d)
    extent = (maps["density"] >= d_extent) & maps["valid"]
    labels, count = ndimage.label(extent)
    if count:
        seed = ndimage.sum(maps["density"] >= d, labels, np.arange(1, count + 1)) * maps["um"] ** 2
        extent = np.isin(labels, np.flatnonzero(seed >= min_um2) + 1)
    return extent


def processed_cores(maps, mask):
    """Boolean grid of global 768-px level-0 cores that touch the screen mask grown by the margin."""
    margin = int(np.ceil(SCREEN_MARGIN_UM / maps["um"]))
    grown = ndimage.maximum_filter(mask, size=2 * margin + 1) if mask.any() else mask
    width0, height0 = maps["level0"]
    nx, ny = int(np.ceil(width0 / STRIDE)), int(np.ceil(height0 / STRIDE))
    ys, xs = np.nonzero(grown)
    cores = np.zeros((ny, nx), bool)
    # A screen pixel covers level-0 [x*down, (x+1)*down); mark every core it overlaps.
    for dx in (0, 1):
        for dy in (0, 1):
            cx = np.minimum(((xs + dx) * maps["down"] - dx * 1e-6) // STRIDE, nx - 1).astype(int)
            cy = np.minimum(((ys + dy) * maps["down"] - dy * 1e-6) // STRIDE, ny - 1).astype(int)
            cores[cy, cx] = True
    return cores


def in_cores(cores, x0, y0):
    """Level-0 points -> inside a processed core."""
    cx = np.clip((np.asarray(x0) // STRIDE).astype(int), 0, cores.shape[1] - 1)
    cy = np.clip((np.asarray(y0) // STRIDE).astype(int), 0, cores.shape[0] - 1)
    return cores[cy, cx]


def load_screen_params():
    return json.loads(SCREEN_PARAMS_FILE.read_text())


# ---------------------------------------------------------------- calibration and validation on windows

def accepted_aggregates():
    """Two-stage accepted aggregates in every window (frozen parameters), as level-0 pixel centres."""
    windows, inputs = A.load_cohort()
    rows, kept, tables = [], {}, {}
    for w in windows.itertuples():
        x = inputs[(w.barcode, w.window_id)]
        proposals, _ = A.proposal_features(x, **STAGE1)
        if w.barcode not in tables:
            tables[w.barcode] = pd.read_csv(OUT / "slides" / w.barcode / "he2cell" / "windows.csv").set_index("window_id")
        window = tables[w.barcode].loc[w.window_id]
        for index, proposal in enumerate(proposals, 1):
            if not A.accept(proposal, PARAMS["b_share"], PARAMS["require_b_core"])[0]:
                continue
            gy, gx = np.nonzero(proposal["mask"])
            step = A.GRID_UM / x["mpp"]
            key = (w.barcode, w.window_id, index)
            kept[key] = proposal["mask"]
            rows.append({"barcode": w.barcode, "case_id": w.case_id, "window_id": w.window_id, "proposal": index,
                         "split": w.split, "category": w.category, "area_um2": proposal["area_um2"],
                         "x0_level0": window.x0, "y0_level0": window.y0,
                         "px_x": (window.x0 + (gx + .5) * step), "px_y": (window.y0 + (gy + .5) * step)})
    A._PROPOSALS.clear()
    return pd.DataFrame(rows), windows, inputs, kept


def _calibrate_slide(job):
    barcode, aggregates = job
    maps = screen_maps(barcode)
    tissue_mm2 = float(maps["valid"].sum() * maps["um"] ** 2 / 1e6)
    out = []
    for d, d_extent, min_um2 in SCREEN_GRID:
        mask = candidates(maps, d, min_um2, d_extent)
        cores = processed_cores(maps, mask)
        entry = {"barcode": barcode, "d": d, "d_extent": d_extent, "min_um2": min_um2, "tissue_mm2": tissue_mm2, "cores": int(cores.sum()),
                 "processed_mm2": float(cores.sum() * (STRIDE * maps["mpp"]) ** 2 / 1e6),
                 "candidate_mm2": float(mask.sum() * maps["um"] ** 2 / 1e6)}
        coverage = []
        for a in aggregates:
            inside = in_cores(cores, a["px_x"], a["px_y"])
            sx = np.minimum((a["px_x"] / maps["down"]).astype(int), mask.shape[1] - 1)
            sy = np.minimum((a["px_y"] / maps["down"]).astype(int), mask.shape[0] - 1)
            coverage.append([float(inside.mean()), float(mask[sy, sx].mean())])
        entry["coverage"] = coverage
        out.append(entry)
    return barcode, out


def calibrate(workers):
    from multiprocessing import Pool
    CALIBRATION.mkdir(parents=True, exist_ok=True)
    aggregates, windows, _, _ = accepted_aggregates()
    reference = pd.read_csv(ANALYSIS / "two_stage_proposals.csv")
    check = {"accepted_reconstructed": len(aggregates), "accepted_in_two_stage_proposals_csv": int(reference.accepted.sum()),
             "test_reconstructed": int((aggregates.split == "test").sum()),
             "test_in_csv": int(reference[reference.split == "test"].accepted.sum())}
    print(check, flush=True)
    if check["accepted_reconstructed"] != check["accepted_in_two_stage_proposals_csv"]:
        raise RuntimeError(f"Reconstructed accepted aggregates do not match two_stage_proposals.csv: {check}")
    aggregates.drop(columns=["px_x", "px_y"]).to_csv(CALIBRATION / "accepted_aggregates.csv", index=False)
    jobs = []
    for barcode in sorted(windows.barcode.unique()):
        part = aggregates[aggregates.barcode == barcode]
        jobs.append((barcode, [{"px_x": r.px_x, "px_y": r.px_y} for r in part.itertuples()]))
    rows, coverage_rows = [], []
    with Pool(workers) as pool:
        for barcode, result in pool.imap_unordered(_calibrate_slide, jobs):
            part = aggregates[aggregates.barcode == barcode]
            for entry in result:
                for (processed, candidate), a in zip(entry.pop("coverage"), part.itertuples()):
                    coverage_rows.append({"barcode": barcode, "window_id": a.window_id, "proposal": a.proposal,
                                          "split": a.split, "d": entry["d"], "d_extent": entry["d_extent"], "min_um2": entry["min_um2"],
                                          "processed_coverage": processed, "candidate_coverage": candidate})
                rows.append(entry)
            print(f"screened {barcode}", flush=True)
    grid = pd.DataFrame(rows)
    coverage = pd.DataFrame(coverage_rows)
    grid.to_csv(CALIBRATION / "screen_grid_per_slide.csv", index=False)
    coverage.to_csv(CALIBRATION / "screen_grid_aggregate_coverage.csv", index=False)
    per_setting = (coverage.assign(ok=coverage.processed_coverage >= COVERED)
                   .groupby(["d", "d_extent", "min_um2", "split"]).ok.mean().unstack("split"))
    area = grid.groupby(["d", "d_extent", "min_um2"]).agg(median_cores=("cores", "median"), p90_cores=("cores", lambda c: c.quantile(.9)),
                                        median_processed_mm2=("processed_mm2", "median"),
                                        total_cores=("cores", "sum"))
    table = per_setting.join(area).reset_index()
    table.to_csv(CALIBRATION / "screen_grid_summary.csv", index=False)
    # Dev only: among settings reaching the dev target, the one with the fewest cores in total; if none
    # reaches it, the highest dev coverage (then fewest cores), flagged.
    eligible = table[table.dev >= DEV_TARGET]
    reached = not eligible.empty
    best = (eligible.sort_values("total_cores").iloc[0] if reached
            else table.sort_values(["dev", "total_cores"], ascending=[False, True]).iloc[0])
    params = {"feature": "LoG nuclear-blob density per mm2 (8 um grid, Gaussian sigma 25 um) on artefact-free tissue",
              "density_per_mm2": float(best.d), "extent_density_per_mm2": float(best.d_extent), "min_component_um2": float(best.min_um2), "sigma_um": SCREEN_SIGMA_UM,
              "blob": {"level_um": BLOB_UM, "log_sigma_um": BLOB_SIGMA_UM, "min_spacing_um": BLOB_SPACING_UM,
                       "min_log": BLOB_MIN_LOG, "min_od": BLOB_MIN_OD},
              "margin_um": SCREEN_MARGIN_UM,
              "rule": f"choose (density, min area) with the fewest processed cores over the cohort among those covering >= {COVERED:.0%} "
                      f"of the area of >= {DEV_TARGET:.0%} of dev accepted aggregates",
              "dev_target_reached": bool(reached),
              "dev_fraction_covered": float(best.dev), "median_cores_per_slide": float(best.median_cores),
              "p90_cores_per_slide": float(best.p90_cores), "reconstruction_check": check,
              "tuned_on": "dev split only; test coverage is reported by --validate"}
    save_json(params, SCREEN_PARAMS_FILE)
    print(table.to_string(), flush=True)
    print(json.dumps(params, indent=1))


def end_to_end(settings, barcodes, aggregates, kept):
    """Per accepted window aggregate and screen setting: is it still accepted when only cells inside the
    cores that setting would process are kept and the frozen two-stage rule is re-run?"""
    rows = []
    for barcode in barcodes:
        maps = screen_maps(barcode)
        folder = OUT / "slides" / barcode
        summary = json.loads((folder / "hooknet" / "summary.json").read_text())
        maps_h = np.load(folder / "hooknet" / "maps_4um.npz")
        maps_h = {"prediction_mask": maps_h["prediction_mask"]}
        selection = json.loads((folder / "he2cell" / "window_selection.json").read_text())
        flagged = set(selection["artifact_flagged_hooknet_tls"])
        features = [f for f in json.loads((folder / "hooknet" / "native_polygons.geojson").read_text())["features"]
                    if f["properties"]["predicted_label"] == "tls" and f["properties"]["passes_published_filter"]
                    and f["properties"]["region_id"] not in flagged]
        all_cells = pd.read_csv(folder / "he2cell_ft_tau05" / "cells.csv.gz")
        qc = slide_qc(folder)
        windows = pd.read_csv(folder / "he2cell" / "windows.csv")
        for d, d_extent, min_um2 in settings:
            cores = processed_cores(maps, candidates(maps, d, min_um2, d_extent))
            cells = all_cells[in_cores(cores, all_cells.centroid_x_global.to_numpy(), all_cells.centroid_y_global.to_numpy())]
            for w in windows.itertuples():
                original = aggregates[(aggregates.barcode == barcode) & (aggregates.window_id == w.window_id)]
                if original.empty:
                    continue
                x = A.window_inputs(folder, w, summary, maps_h, features, qc, cells)
                proposals, _ = A.proposal_features(x, **STAGE1)
                accepted = [p["mask"] for p in proposals if A.accept(p, PARAMS["b_share"], PARAMS["require_b_core"])[0]]
                for a in original.itertuples():
                    mask = kept[(barcode, w.window_id, a.proposal)]
                    hit = any((mask & m).sum() >= A.MATCH_FRACTION * min(mask.sum(), m.sum()) for m in accepted)
                    rows.append({"barcode": barcode, "case_id": a.case_id, "window_id": w.window_id, "proposal": a.proposal,
                                 "split": a.split, "category": a.category, "area_um2": a.area_um2,
                                 "d": d, "d_extent": d_extent, "min_um2": min_um2, "recovered_end_to_end": hit})
            A._PROPOSALS.clear()
        print(f"end-to-end {barcode}", flush=True)
    return pd.DataFrame(rows)


def calibrate_end_to_end(n_settings=10):
    """Dev only: among the cheapest area-grid settings, the cheapest whose dev end-to-end recovery is >= 98%."""
    table = pd.read_csv(CALIBRATION / "screen_grid_summary.csv")
    shortlist = table[table.dev >= 0.85].sort_values("total_cores").head(n_settings)
    settings = list(shortlist[["d", "d_extent", "min_um2"]].itertuples(index=False, name=None))
    aggregates, _, _, kept = accepted_aggregates()
    dev = aggregates[aggregates.split == "dev"]
    frame = end_to_end(settings, sorted(dev.barcode.unique()), dev, kept)
    frame.to_csv(CALIBRATION / "screen_end_to_end_dev.csv", index=False)
    recovery = frame.groupby(["d", "d_extent", "min_um2"]).recovered_end_to_end.mean().rename("dev_end_to_end")
    shortlist = shortlist.merge(recovery.reset_index(), on=["d", "d_extent", "min_um2"])
    shortlist.drop(columns="test").to_csv(CALIBRATION / "screen_end_to_end_dev_summary.csv", index=False)
    eligible = shortlist[shortlist.dev_end_to_end >= DEV_TARGET]
    reached = not eligible.empty
    best = (eligible.sort_values("total_cores").iloc[0] if reached
            else shortlist.sort_values(["dev_end_to_end", "total_cores"], ascending=[False, True]).iloc[0])
    params = load_screen_params()
    shutil.copy(SCREEN_PARAMS_FILE, CALIBRATION / "screen_params_area_proxy.json")
    params.update({"density_per_mm2": float(best.d), "extent_density_per_mm2": float(best.d_extent),
                   "min_component_um2": float(best.min_um2), "dev_target_reached": bool(reached),
                   "dev_fraction_covered": float(best.dev), "dev_end_to_end_recovery": float(best.dev_end_to_end),
                   "median_cores_per_slide": float(best.median_cores), "p90_cores_per_slide": float(best.p90_cores),
                   "rule": f"among the {n_settings} cheapest settings of the area grid (dev area coverage >= 85%), the one with "
                           f"the fewest processed cores whose dev end-to-end recovery of accepted TLS is >= {DEV_TARGET:.0%}"})
    save_json(params, SCREEN_PARAMS_FILE)
    print(shortlist.drop(columns="test").round(3).to_string())
    print(json.dumps(params, indent=1))


def validate():
    """Frozen screen on all windows; the test split is the held-out check (>= 95% end-to-end recovery)."""
    params = load_screen_params()
    aggregates, windows, inputs, kept = accepted_aggregates()
    coverage = pd.read_csv(CALIBRATION / "screen_grid_aggregate_coverage.csv")
    coverage = coverage[np.isclose(coverage.d, params["density_per_mm2"]) & np.isclose(coverage.d_extent, params["extent_density_per_mm2"])
                        & np.isclose(coverage.min_um2, params["min_component_um2"])]
    setting = (params["density_per_mm2"], params["extent_density_per_mm2"], params["min_component_um2"])
    rows = end_to_end([setting], sorted(aggregates.barcode.unique()), aggregates, kept).to_dict("records")
    frame = pd.DataFrame(rows).merge(coverage[["barcode", "window_id", "proposal", "processed_coverage", "candidate_coverage"]],
                                     on=["barcode", "window_id", "proposal"], how="left")
    frame.to_csv(CALIBRATION / "screen_validation_aggregates.csv", index=False)
    result = {"screen_params": params}
    for split in ["dev", "test"]:
        part = frame[frame.split == split]
        hits = part.recovered_end_to_end.astype(bool)
        result[split] = {"accepted_aggregates": len(part), "patients": int(part.case_id.nunique()),
                         "recovered_end_to_end": int(hits.sum()), "recovered_fraction": float(hits.mean()),
                         "area_covered_ge_90pct": float((part.processed_coverage >= COVERED).mean()),
                         "missed": part[~hits][["barcode", "window_id", "proposal", "area_um2", "processed_coverage"]].to_dict("records")}
        if split == "test":
            rng = np.random.default_rng(0)
            cases = part.case_id.unique()
            groups = {c: part[part.case_id == c].recovered_end_to_end.astype(bool).to_numpy() for c in cases}
            boots = [np.concatenate([groups[c] for c in rng.choice(cases, len(cases))]).mean() for _ in range(2000)]
            result[split]["recovered_fraction_CI95"] = np.percentile(boots, [2.5, 97.5]).tolist()
    result["passes_95pct_on_test"] = result["test"]["recovered_fraction"] >= 0.95
    save_json(result, CALIBRATION / "screen_validation.json")
    print(json.dumps({s: {k: v for k, v in result[s].items() if k != "missed"} for s in ["dev", "test"]}, indent=1))
    print("passes:", result["passes_95pct_on_test"])


# ---------------------------------------------------------------- whole-slide run

def take_gpu_lock(log):
    while True:
        try:
            GPU_LOCK.mkdir()
            (GPU_LOCK / "owner").write_text(f"item4a-wsi-burden pid {os.getpid()}\n")
            return
        except FileExistsError:
            owner = (GPU_LOCK / "owner").read_text().strip() if (GPU_LOCK / "owner").exists() else "?"
            log(f"waiting for {GPU_LOCK} ({owner})")
            time.sleep(60)


def release_gpu_lock():
    owner = GPU_LOCK / "owner"
    if owner.exists() and owner.read_text().startswith(f"item4a-wsi-burden pid {os.getpid()}"):
        shutil.rmtree(GPU_LOCK, ignore_errors=True)


def tls_features(barcode):
    """HookNet published-filter TLS polygons, minus those mostly on artefact (as slide_hooknet_burden)."""
    folder = OUT / "slides" / barcode
    qc = slide_qc(folder)
    kept = []
    for f in json.loads((folder / "hooknet" / "native_polygons.geojson").read_text())["features"]:
        if f["properties"]["predicted_label"] != "tls" or not f["properties"]["passes_published_filter"]:
            continue
        ring = np.asarray(f["geometry"]["coordinates"][0], float) / 16
        canvas = np.zeros(qc["valid"].shape, np.uint8)
        cv2.fillPoly(canvas, [np.round(ring).astype(np.int32)], 1)
        inside = canvas.astype(bool)
        if inside.sum() and (qc["artifact"][inside] | ~qc["valid"][inside]).mean() <= .25:
            kept.append(f)
    return kept


def detect_regions(barcode, cores, cells, mpp):
    """Frozen two-stage detection per connected processed region (8 µm grid, region-local origin)."""
    folder = OUT / "slides" / barcode
    summary = json.loads((folder / "hooknet" / "summary.json").read_text())
    maps_h = np.load(folder / "hooknet" / "maps_4um.npz")
    maps_h = {"prediction_mask": maps_h["prediction_mask"]}
    features = tls_features(barcode)
    qc = slide_qc(folder)
    labels, count = ndimage.label(cores, structure=np.ones((3, 3)))
    rows, unproposed = [], 0
    cells = cells.copy()
    cells["window_id"] = "none"
    core_label = labels[np.clip((cells.centroid_y_global // STRIDE).astype(int), 0, labels.shape[0] - 1),
                        np.clip((cells.centroid_x_global // STRIDE).astype(int), 0, labels.shape[1] - 1)]
    cells["window_id"] = [f"region_{r}" for r in core_label]
    for region, slc in enumerate(ndimage.find_objects(labels), 1):
        x0, y0 = slc[1].start * STRIDE, slc[0].start * STRIDE
        side = max(slc[1].stop - slc[1].start, slc[0].stop - slc[0].start) * STRIDE
        window = Window(x0, y0, x0 + side, y0 + side, f"region_{region}")
        x = A.window_inputs(folder, window, summary, maps_h, features, qc, cells)
        inside = np.zeros((x["n"], x["n"]), bool)  # this region's cores only (squares can reach other regions)
        step = STRIDE * mpp / A.GRID_UM
        for cy, cx in zip(*np.nonzero(labels[slc] == region)):
            inside[int(cy * step):int(np.ceil((cy + 1) * step)), int(cx * step):int(np.ceil((cx + 1) * step))] = True
        x["valid"] = x["valid"] & inside
        proposals, missed = A.proposal_features(x, **STAGE1)
        unproposed += missed
        for index, proposal in enumerate(proposals, 1):
            accepted, reason = A.accept(proposal, PARAMS["b_share"], PARAMS["require_b_core"])
            gy, gx = np.nonzero(proposal["mask"])
            rows.append({"barcode": barcode, "region": region, "proposal": index, "accepted": accepted,
                         "rejection_reason": reason,
                         "centroid_x_level0": x0 + (gx.mean() + .5) * A.GRID_UM / mpp,
                         "centroid_y_level0": y0 + (gy.mean() + .5) * A.GRID_UM / mpp,
                         **{k: v for k, v in proposal.items() if k != "mask"}})
        A._PROPOSALS.clear()
    columns = ["barcode", "region", "proposal", "accepted", "rejection_reason", "centroid_x_level0", "centroid_y_level0",
               "area_um2", "cells", "B_share", "T_share", "B_core", "B_periphery", "B_core_enrichment",
               "T_of_lymphocytes_core", "T_of_lymphocytes_periphery", "T_rim_enrichment", "HookNet_GC_pixel_fraction",
               "matches_HookNet"]
    return pd.DataFrame(rows, columns=columns), len(features), unproposed


def process(barcode, log):
    out = OUT / "slides" / barcode / RESULT
    if (out / "summary.json").exists():
        return "done"
    started = time.perf_counter()
    params = load_screen_params()
    out.mkdir(parents=True, exist_ok=True)
    maps = screen_maps(barcode)
    mask = candidates(maps, params["density_per_mm2"], params["min_component_um2"], params["extent_density_per_mm2"])
    cores = processed_cores(maps, mask)
    tissue_mm2 = float(slide_qc(OUT / "slides" / barcode)["valid"].sum() * 64 / 1e6)
    np.savez_compressed(out / "screen.npz", candidate=mask, cores=cores, down=maps["down"])
    screen_seconds = time.perf_counter() - started
    work = OUT / f"_tmp_{RESULT}" / barcode
    shutil.rmtree(work, ignore_errors=True)
    (work / "patches").mkdir(parents=True)
    rows = []
    slide = slide_path(barcode)
    with open_slide(slide) as handle:
        for cy, cx in zip(*np.nonzero(cores)):
            x, y = int(cx * STRIDE - CORE_MARGIN), int(cy * STRIDE - CORE_MARGIN)
            patch_id = len(rows)
            path = work / "patches" / f"wsi_patch{patch_id}_x{x}_y{y}.png"
            handle.read_region((x, y), 0, (PATCH, PATCH)).convert("RGB").save(path, compress_level=1)
            rows.append({"path": str(path), "num_patch": patch_id, "sample": slide.stem, "source_tif": str(slide),
                         "x": x, "y": y, "patch_size": PATCH, "stride": STRIDE, "level": 0,
                         "tissue_ratio": 1.0, "window_id": "wsi"})
    seg = cls = 0.0
    if rows:
        patch_table = pd.DataFrame(rows)
        patch_csv = work / f"{slide.stem}_patch_index.csv"
        patch_table.to_csv(patch_csv, index=False)
        pipeline_log = out / "pipeline.log"
        take_gpu_lock(log)
        try:
            seg = stage("run_cellcrops.py", "--patch_csv", patch_csv, "--out_dir", work / "cells", "--model_name", "cpsam_v2",
                        "--device", "mps", "--num_gpus", 1, "--batch_size", 16, "--crop_size", 64, log=pipeline_log)
            cells_csv = work / "cells" / "mask_tables" / f"{patch_csv.stem}_cellpose_mask_instances.csv"
            result_csv = work / "cell_types.csv"
            cls = stage("model_eval.py", "--meta_path", cells_csv, "--save_path", CLASSIFIER, "--output_path", result_csv,
                        "--device", "mps", "--batch_size", 256, "--num_workers", 0, "--tta", "d4", log=pipeline_log)
        finally:
            release_gpu_lock()
        cells = pd.read_csv(result_csv)[KEEP]
    else:
        cells = pd.DataFrame(columns=KEEP)
    tls, hooknet_tls, hooknet_unproposed = detect_regions(barcode, cores, cells, maps["mpp"])
    accepted = tls[tls.accepted]
    cells.to_csv(out / "cells.csv.gz", index=False)
    tls.to_csv(out / "tls.csv", index=False)
    core_mm2 = (STRIDE * maps["mpp"]) ** 2 / 1e6
    save_json({"barcode": barcode, "slide": str(slide), "status": "complete",
               "valid_tissue_mm2": tissue_mm2, "candidate_mm2": float(mask.sum() * maps["um"] ** 2 / 1e6),
               "processed_cores": int(cores.sum()), "processed_mm2": float(cores.sum() * core_mm2),
               "processed_fraction_of_tissue": float(cores.sum() * core_mm2 / max(tissue_mm2, 1e-9)),
               "cells": len(cells), "counts": cells.celltype.value_counts().to_dict(),
               "stage1_aggregates": len(tls), "accepted_TLS": len(accepted),
               "accepted_TLS_area_mm2": float(accepted.area_um2.sum() / 1e6),
               "TLS_per_cm2": float(len(accepted) / max(tissue_mm2 / 100, 1e-9)),
               "TLS_per_100mm2": float(100 * len(accepted) / max(tissue_mm2, 1e-9)),
               "B_TLS_area_fraction": float(accepted.area_um2.sum() / 1e6 / max(tissue_mm2, 1e-9)),
               "accepted_matching_HookNet": int(accepted.matches_HookNet.sum()),
               "hooknet_TLS_artifact_free": hooknet_tls, "hooknet_TLS_inside_processed_not_proposed": int(hooknet_unproposed),
               "screen_seconds": screen_seconds, "segmentation_seconds": seg, "classification_seconds": cls,
               "wall_seconds": time.perf_counter() - started, "screen_params": params, "two_stage_params": PARAMS,
               "patch_size": PATCH, "stride": STRIDE, "core_margin": CORE_MARGIN, "classifier": str(CLASSIFIER),
               "classifier_sha256": hashlib.sha256(CLASSIFIER.read_bytes()).hexdigest(),
               "segmentation_model": "cpsam_v2", "tta": "d4", "stain_normalisation": "none"},
              out / "summary.json")
    shutil.rmtree(work, ignore_errors=True)
    return (f"{int(cores.sum())} cores ({cores.sum() * core_mm2:.1f} of {tissue_mm2:.0f} mm2), {len(cells)} cells, "
            f"{len(accepted)}/{len(tls)} accepted, {time.perf_counter() - started:.0f}s")


def usable_barcodes():
    """Slides of the analysed cohort: those with adopted-classifier window predictions."""
    return sorted(p.parents[1].name for p in (OUT / "slides").glob("*/he2cell_ft_tau05/cells.csv.gz"))


ORDER_SEED = 20261025
SURVIVAL_EVERY = 10  # completed slides between interim survival rewrites


def patient_order(barcodes):
    """Patients in a seeded random order (a patient's slides together), so any completed prefix is an
    unbiased subset of the cohort."""
    cases = sorted({b[:12] for b in barcodes})
    rank = {case: i for i, case in enumerate(np.random.default_rng(ORDER_SEED).permutation(cases))}
    return sorted(barcodes, key=lambda b: (rank[b[:12]], b))


def interim_survival(log):
    """Rewrite the survival report with the HE2CELL burden of the patients completed so far."""
    import subprocess
    out = ROOT / "outputs" / "esca_survival_20261025"
    with open(out / "survival_rerun.log", "a") as handle:
        status = subprocess.run([sys.executable, str(HERE / "esca_survival.py"), "--out", str(out)],
                                stdout=handle, stderr=subprocess.STDOUT).returncode
    log(f"survival report rewritten (exit {status})")


def run(limit=None, only=None, order="random-patient"):
    log_path = OUT / "wsi_burden.log"

    def log(message):
        with open(log_path, "a") as handle:
            handle.write(f"{time.strftime('%F %T')} {message}\n")
        print(message, flush=True)

    barcodes = patient_order(only or usable_barcodes()) if order == "random-patient" else (only or usable_barcodes())
    log(f"run start: {len(barcodes)} slides in seeded random patient order (seed {ORDER_SEED})")
    done = completed = 0
    for barcode in barcodes:
        out = OUT / "slides" / barcode / RESULT
        if (out / "summary.json").exists():
            continue
        lock = OUT / "_locks" / f"{barcode}.wsi_burden"
        lock.parent.mkdir(exist_ok=True)
        if lock.exists() and not _alive(lock):
            shutil.rmtree(lock, ignore_errors=True)  # stale lock of a killed run
        try:
            lock.mkdir()
            (lock / "pid").write_text(str(os.getpid()))
        except FileExistsError:
            continue
        try:
            log(f"{barcode}: start")
            log(f"{barcode}: {process(barcode, log)}")
            (out / "failed.json").unlink(missing_ok=True)
            completed += 1
            if completed % SURVIVAL_EVERY == 0:
                interim_survival(log)
        except Exception as error:
            save_json({"status": "failed", "error": repr(error)}, out / "failed.json")
            log(f"{barcode}: FAILED {error!r}")
        finally:
            shutil.rmtree(lock, ignore_errors=True)
        done += 1
        if limit and done >= limit:
            break
    finished = sum((OUT / "slides" / b / RESULT / "summary.json").exists() for b in usable_barcodes())
    log(f"run ended: {finished}/{len(usable_barcodes())} slides complete")
    interim_survival(log)


def _alive(lock):
    try:
        os.kill(int((lock / "pid").read_text()), 0)
        return True
    except (OSError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibrate", action="store_true", help="tune the screen on dev windows (CPU)")
    parser.add_argument("--calibrate-e2e", action="store_true", help="dev end-to-end recovery for the cheapest settings (CPU)")
    parser.add_argument("--validate", action="store_true", help="end-to-end screen check on test windows (CPU)")
    parser.add_argument("--run", action="store_true", help="whole-slide burden for all usable slides (GPU)")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--barcodes", nargs="*")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.calibrate:
        calibrate(args.workers)
    if args.calibrate_e2e:
        calibrate_end_to_end()
    if args.validate:
        validate()
    if args.run:
        run(args.limit, args.barcodes)


if __name__ == "__main__":
    main()
