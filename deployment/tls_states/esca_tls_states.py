"""TLS states (Part A): composition / shape features of every accepted two-stage aggregate in the
TCGA-ESCA windows, a frozen Gaussian-mixture state model, and patient-level associations with histology
and bulk-RNA immune programs.

This module also holds the shared code that visium_tls_states.py and esca_state_survival.py import:
object_features(), the numpy EM mixture, the frozen-model JSON and assign_states().

Everything works on the 8 µm grid of esca_analysis.

Features of one object (a boolean 8 µm mask plus the cell-count grids of esca_analysis.window_inputs):
- area_um2, cells, nuclei_per_mm2 (all segmented nuclei inside the object / object area);
- B / T / Macrophage / Other shares (cell counts of the classifier's labels inside the object);
- B_core_enrichment and T_rim_enrichment, identical to esca_analysis.proposal_features;
- radial profile: B, T and macrophage density (cells per mm² of artefact-free tissue) in 30 µm rings
  (0-30, ..., 120-150 µm) around the object centroid, inside and outside the object; a ring with < 30% of
  its area on valid tissue is NaN. Gradients = log2((density 0-60 µm + 50) / (density 90-150 µm + 50));
- shape of the hole-filled object: compactness 4πA/P² and solidity A/convex-hull area (largest outer contour);
- HookNet_GC_pixel_fraction (HookNet germinal-centre class pixels inside the object; descriptive only).

The state model is a Gaussian mixture fitted by EM in numpy (scikit-learn is not installed in .venv and
the shared environment was not changed while the whole-slide job runs). It is stored as plain JSON
(feature list, imputation medians, scaler, mixture parameters, state names) so assign_states() needs
nothing but numpy.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "esca_tls"))
import os  # noqa: E402
os.environ.setdefault("ESCA_RESULT", "he2cell_ft_tau05")  # adopted classifier (fine-tuned + tau 0.5 prior)
import esca_analysis as A  # noqa: E402

ROOT = HERE.parents[1]
OUT = ROOT / "outputs" / "tls_states_20261026"
TABLES = OUT / "tables"
MODEL_JSON = OUT / "tls_state_model.json"
RINGS_UM = [0, 30, 60, 90, 120, 150]
RING_MIN_VALID = 0.3
PSEUDO_PER_MM2 = 50.0
CELL_TYPES = ["B", "T", "Macrophage", "Other"]
# Pre-specified clustering features (written before the first fit). Other share is implied by the three shares.
STATE_FEATURES = ["log10_area_um2", "nuclei_per_mm2", "B_share", "T_share", "Macrophage_share",
                  "B_core_enrichment", "T_rim_enrichment", "B_radial_gradient", "T_radial_gradient",
                  "solidity", "compactness"]
FEATURE_NAMES = {"log10_area_um2": "Area (log10 µm²)", "nuclei_per_mm2": "Nuclear density", "B_share": "B share",
                 "T_share": "T share", "Macrophage_share": "Macrophage share", "Other_share": "Other share",
                 "B_core_enrichment": "B-core enrichment", "T_rim_enrichment": "T-rim enrichment",
                 "B_radial_gradient": "B radial gradient", "T_radial_gradient": "T radial gradient",
                 "Macrophage_radial_gradient": "Macrophage radial gradient",
                 "solidity": "Solidity", "compactness": "Compactness", "HookNet_GC_pixel_fraction": "HookNet GC pixels"}


# ------------------------------------------------------------------ features

def ring_columns():
    return [f"{t}_per_mm2_{lo}_{hi}um" for t in ["B", "T", "Macrophage"] for lo, hi in zip(RINGS_UM[:-1], RINGS_UM[1:])]


def object_features(obj, x):
    """All features of one object. x: dict with 'counts' (8 µm grids), 'valid', 'cells_xy_um' (µm in the
    grid frame), 'celltype' and optionally 'gc' (HookNet GC class grid)."""
    counts, valid = x["counts"], x["valid"]
    total = sum(counts[k] for k in CELL_TYPES)
    n_cells = total[obj].sum()
    area = obj.sum() * A.GRID_UM ** 2
    distance = ndimage.distance_transform_edt(obj)
    core = obj & (distance >= .5 * distance.max())
    ring = int(round(A.RIM_UM / A.GRID_UM))
    periphery = (obj & ~core) | (ndimage.binary_dilation(obj, iterations=ring) & ~obj & valid)

    def share(kind, mask, among=None):
        denominator = sum(counts[k][mask].sum() for k in among) if among else total[mask].sum()
        return counts[kind][mask].sum() / denominator if denominator else np.nan
    out = {"area_um2": float(area), "log10_area_um2": float(np.log10(max(area, 1))), "cells": int(n_cells),
           "nuclei_per_mm2": float(n_cells / (area / 1e6)) if area else np.nan,
           **{f"{k}_share": share(k, obj) for k in CELL_TYPES}}
    out["B_core_enrichment"] = share("B", core) - share("B", periphery)
    out["T_rim_enrichment"] = share("T", periphery, ["B", "T"]) - share("T", core, ["B", "T"])
    out["HookNet_GC_pixel_fraction"] = float(x["gc"][obj].mean()) if x.get("gc") is not None else np.nan
    # Radial profile around the centroid (µm, grid frame).
    gy, gx = np.nonzero(obj)
    cy, cx = (gy.mean() + .5) * A.GRID_UM, (gx.mean() + .5) * A.GRID_UM
    out["centroid_x_um"], out["centroid_y_um"] = float(cx), float(cy)
    rmax = RINGS_UM[-1]
    step = 2.0
    offs = np.arange(-rmax + step / 2, rmax, step)
    py, px = np.meshgrid(cy + offs, cx + offs, indexing="ij")
    iy, ix = np.floor(py / A.GRID_UM).astype(int), np.floor(px / A.GRID_UM).astype(int)
    inside = (iy >= 0) & (ix >= 0) & (iy < valid.shape[0]) & (ix < valid.shape[1])
    ok = np.zeros_like(inside)
    ok[inside] = valid[iy[inside], ix[inside]]
    rr = np.hypot(py - cy, px - cx)
    xy = x["cells_xy_um"]
    rc = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
    near = rc < rmax
    cxi, cyi = np.floor(xy[near, 0] / A.GRID_UM).astype(int), np.floor(xy[near, 1] / A.GRID_UM).astype(int)
    keep = (cyi >= 0) & (cxi >= 0) & (cyi < valid.shape[0]) & (cxi < valid.shape[1])
    cell_ok = np.zeros(near.sum(), bool)
    cell_ok[keep] = valid[cyi[keep], cxi[keep]]
    rc_near, types_near = rc[near][cell_ok], np.asarray(x["celltype"])[near][cell_ok]
    ring_area, ring_counts = [], {t: [] for t in ["B", "T", "Macrophage"]}
    for lo, hi in zip(RINGS_UM[:-1], RINGS_UM[1:]):
        a_valid = ((rr >= lo) & (rr < hi) & ok).sum() * step ** 2
        a_full = np.pi * (hi ** 2 - lo ** 2)
        ring_area.append(a_valid if a_valid >= RING_MIN_VALID * a_full else np.nan)
        sel = (rc_near >= lo) & (rc_near < hi)
        for t in ring_counts:
            ring_counts[t].append(float(((types_near == t) & sel).sum()))
    ring_area = np.asarray(ring_area, float)
    for t, c in ring_counts.items():
        c = np.asarray(c)
        for (lo, hi), v, a in zip(zip(RINGS_UM[:-1], RINGS_UM[1:]), c, ring_area):
            out[f"{t}_per_mm2_{lo}_{hi}um"] = float(v / (a / 1e6)) if np.isfinite(a) else np.nan
        inner_a, outer_a = np.nansum(ring_area[:2]) if np.isfinite(ring_area[:2]).all() else np.nan, \
            np.nansum(ring_area[3:]) if np.isfinite(ring_area[3:]).all() else np.nan
        inner = c[:2].sum() / (inner_a / 1e6) if np.isfinite(inner_a) else np.nan
        outer = c[3:].sum() / (outer_a / 1e6) if np.isfinite(outer_a) else np.nan
        out[f"{t}_radial_gradient"] = float(np.log2((inner + PSEUDO_PER_MM2) / (outer + PSEUDO_PER_MM2))) \
            if np.isfinite(inner) and np.isfinite(outer) else np.nan
    # Shape of the hole-filled object.
    filled = ndimage.binary_fill_holes(obj).astype(np.uint8)
    contours, _ = cv2.findContours(np.pad(filled, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest = max(contours, key=cv2.contourArea)
    poly_area, perimeter = cv2.contourArea(largest), cv2.arcLength(largest, True)
    hull = cv2.contourArea(cv2.convexHull(largest))
    if poly_area > 0 and perimeter > 0:
        out["compactness"] = float(4 * np.pi * poly_area / perimeter ** 2)
        out["solidity"] = float(poly_area / hull) if hull else np.nan
    else:  # one-pixel-wide objects
        out["compactness"], out["solidity"] = np.nan, np.nan
    out["touches_grid_edge"] = bool(obj[0].any() or obj[-1].any() or obj[:, 0].any() or obj[:, -1].any())
    return out


# ------------------------------------------------------------------ Gaussian mixture (numpy EM)

def _log_gauss(X, mean, cov, kind):
    d = X.shape[1]
    if kind == "diag":
        var = cov
        return -.5 * (np.sum((X - mean) ** 2 / var, 1) + np.sum(np.log(var)) + d * np.log(2 * np.pi))
    L = np.linalg.cholesky(cov)
    z = np.linalg.solve(L, (X - mean).T)
    return -.5 * (np.sum(z ** 2, 0) + 2 * np.sum(np.log(np.diag(L))) + d * np.log(2 * np.pi))


def log_resp(X, model):
    kind = model["covariance_type"]
    logp = np.stack([np.log(w) + _log_gauss(X, np.asarray(m), np.asarray(c), kind)
                     for w, m, c in zip(model["weights"], model["means"], model["covariances"])], 1)
    norm = np.logaddexp.reduce(logp, 1)
    return logp - norm[:, None], norm


def _kmeanspp(X, k, rng):
    centres = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d2 = np.min([np.sum((X - c) ** 2, 1) for c in centres], 0)
        centres.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    return np.asarray(centres)


def fit_gmm(X, k, kind="full", n_init=20, seed=20261026, reg=1e-3, max_iter=1000, tol=1e-7):
    """EM for a k-component Gaussian mixture; best of n_init k-means++ starts by log-likelihood."""
    n, d = X.shape
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        centres = _kmeanspp(X, k, rng)
        labels = np.argmin(((X[:, None] - centres[None]) ** 2).sum(-1), 1)
        resp = np.eye(k)[labels] + 1e-6
        resp /= resp.sum(1, keepdims=True)
        previous = -np.inf
        for _ in range(max_iter):
            nk = resp.sum(0) + 1e-10
            means = resp.T @ X / nk[:, None]
            covs = []
            for j in range(k):
                diff = X - means[j]
                if kind == "diag":
                    covs.append((resp[:, j] @ diff ** 2) / nk[j] + reg)
                else:
                    covs.append((resp[:, j, None] * diff).T @ diff / nk[j] + reg * np.eye(d))
            model = {"covariance_type": kind, "weights": nk / n, "means": means, "covariances": covs}
            lr, norm = log_resp(X, model)
            ll = norm.sum()
            resp = np.exp(lr)
            if ll - previous < tol * abs(ll):
                break
            previous = ll
        if best is None or ll > best[0]:
            best = (ll, model, resp)
    ll, model, resp = best
    params = (k - 1) + k * d + (k * d if kind == "diag" else k * d * (d + 1) // 2)
    hard = resp.argmax(1)
    return {"model": model, "log_likelihood": float(ll), "n_parameters": params,
            "BIC": float(-2 * ll + params * np.log(n)), "labels": hard,
            "min_state_fraction": float(np.bincount(hard, minlength=k).min() / n), "resp": resp}


def adjusted_rand(a, b):
    a, b = np.asarray(a), np.asarray(b)
    table = pd.crosstab(a, b).to_numpy()
    comb = lambda v: v * (v - 1) / 2  # noqa: E731
    s = comb(table).sum()
    sa, sb, n = comb(table.sum(1)).sum(), comb(table.sum(0)).sum(), comb(len(a))
    expected = sa * sb / n
    return float((s - expected) / (.5 * (sa + sb) - expected)) if .5 * (sa + sb) != expected else 1.0


# ------------------------------------------------------------------ frozen model I/O and assignment

def prepare(frame, features, medians=None, mean=None, sd=None):
    X = frame[features].to_numpy(float)
    medians = np.nanmedian(X, 0) if medians is None else np.asarray(medians)
    X = np.where(np.isfinite(X), X, medians)
    mean = X.mean(0) if mean is None else np.asarray(mean)
    sd = X.std(0) if sd is None else np.asarray(sd)
    return (X - mean) / sd, medians, mean, sd


def load_model(path=MODEL_JSON):
    return json.loads(Path(path).read_text())


def assign_states(frame, which="primary", model=None):
    """Apply the frozen state model to a table that has the STATE_FEATURES columns.

    Returns a copy with 'state' (integer, ordered by descending mean B share), 'state_name' and
    posterior columns 'p_state<k>'. which: 'primary' (BIC-chosen) or 'k3' (fixed k = 3).
    """
    model = model or load_model()
    m = model[which]
    X, *_ = prepare(frame, model["features"], model["impute_median"], model["mean"], model["sd"])
    lr, _ = log_resp(X, m)
    post = np.exp(lr)
    out = frame.copy()
    out["state"] = post.argmax(1)
    out["state_name"] = [m["state_names"][s] for s in out.state]
    for j in range(post.shape[1]):
        out[f"p_state{j}"] = post[:, j]
    out["p_B_follicular"] = post[:, m["state_names"].index("B-follicular")]
    return out


def cluster_bootstrap_ci(frame, cluster, function, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    ids = frame[cluster].unique()
    groups = {c: frame[frame[cluster] == c] for c in ids}
    values = [function(pd.concat([groups[c] for c in rng.choice(ids, len(ids))])) for _ in range(n)]
    return np.nanpercentile(np.asarray(values, float), [2.5, 97.5]).tolist()


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    tmp.replace(path)


# ================================================================== Part A: ESCA window aggregates

from esca_common import OUT as ESCA_OUT  # noqa: E402
from esca_qc import slide_qc  # noqa: E402

ANALYSIS = ESCA_OUT / "analysis_he2cell_ft_tau05"
FROZEN = json.loads((ANALYSIS / "report_summary.json").read_text())["chosen_parameters"][A.TWO_STAGE]
STAGE1 = {k: FROZEN[k] for k in ["sigma_um", "threshold", "min_area_um2", "min_cells"]}
FIG = OUT / "figures"
PARTS = OUT / "parts"
SEED = 20261026
K_RANGE = range(2, 7)
MIN_STATE_FRACTION = 0.10
# Patient RNA programs (log2 TPM+1, patient median of primary-tumour aliquots, z within ESCA, mean z).
PROGRAMS = {"GC": ["AICDA", "BCL6", "RGS13", "MKI67"], "Tfh": ["CXCL13", "PDCD1", "ICOS", "CXCR5"],
            "plasma": ["MZB1", "JCHAIN", "IGHG1"], "mature_DC": ["LAMP3", "CCL19", "CCL21"],
            "cytotoxic": ["CD8A", "GZMB", "PRF1"], "checkpoint": ["PDCD1", "CTLA4", "LAG3", "HAVCR2", "CD274"],
            "B_cell": ["MS4A1", "CD19", "CD79A", "CR2"], "T_cell": ["CD3E", "CD3D", "CD2"],
            "macrophage": ["CD68", "CD163", "C1QA"]}
CHEMOKINE12 = ["CCL2", "CCL3", "CCL4", "CCL5", "CCL8", "CCL18", "CCL19", "CCL21", "CXCL9", "CXCL10", "CXCL11", "CXCL13"]
PROGRAM_NAMES = {"chemokine12": "12-chemokine (PC1)", "GC": "Germinal centre", "Tfh": "Tfh", "plasma": "Plasma cell",
                 "mature_DC": "Mature DC", "cytotoxic": "Cytotoxic", "checkpoint": "Exhaustion / checkpoint",
                 "B_cell": "B cell", "T_cell": "T cell", "macrophage": "Macrophage"}


def slide_inputs(slide_folder):
    """Per-slide inputs of esca_analysis.load_cohort (HookNet maps, filtered polygons, QC, cells)."""
    summary = json.loads((slide_folder / "hooknet" / "summary.json").read_text())
    maps = {"prediction_mask": np.load(slide_folder / "hooknet" / "maps_4um.npz")["prediction_mask"]}
    selection = json.loads((slide_folder / "he2cell" / "window_selection.json").read_text())
    flagged = set(selection["artifact_flagged_hooknet_tls"])
    features = [f for f in json.loads((slide_folder / "hooknet" / "native_polygons.geojson").read_text())["features"]
                if f["properties"]["predicted_label"] == "tls" and f["properties"]["passes_published_filter"]
                and f["properties"]["region_id"] not in flagged]
    return summary, maps, features, slide_qc(slide_folder)


def esca_slide(job):
    """Features of the accepted aggregates of one slide. job = (barcode, {(window_id, proposal)}, keep_inputs)."""
    barcode, wanted, keep = job
    folder = ESCA_OUT / "slides" / barcode
    summary, maps, features, qc = slide_inputs(folder)
    cells = pd.read_csv(folder / A.RESULT / "cells.csv.gz")
    windows = pd.read_csv(folder / "he2cell" / "windows.csv")
    rows, kept = [], {}
    for w in windows[windows.window_id.isin({k[0] for k in wanted})].itertuples():
        x = A.window_inputs(folder, w, summary, maps, features, qc, cells)
        proposals, _ = A.proposal_features(x, **STAGE1)
        A._PROPOSALS.clear()
        for index, p in enumerate(proposals, 1):
            if (w.window_id, index) not in wanted:
                continue
            f = object_features(p["mask"], x)
            assert np.isclose(f["B_share"], p["B_share"], equal_nan=True) and f["cells"] == p["cells"], (barcode, w.window_id, index)
            rows.append({"barcode": barcode, "case_id": barcode[:12], "window_id": w.window_id, "proposal": index,
                         "category": w.category, "split": "dev" if A.is_dev(barcode[:12]) else "test",
                         "matches_HookNet": p["matches_HookNet"], **f})
            if keep:
                kept[(w.window_id, index)] = (p["mask"], x)
    return (rows, kept) if keep else rows


def esca_features(workers=8, recompute=False):
    path = TABLES / "esca_window_tls_features.csv"
    if path.exists() and not recompute:
        return pd.read_csv(path)
    from concurrent.futures import ProcessPoolExecutor
    table = pd.read_csv(ANALYSIS / "two_stage_proposals.csv")
    table = table[table.accepted]
    jobs = [(b, set(zip(g.window_id, g.proposal)), False) for b, g in table.groupby("barcode")]
    rows = []
    with ProcessPoolExecutor(workers) as pool:
        for part in pool.map(esca_slide, jobs):
            rows += part
    frame = pd.DataFrame(rows)
    merged = frame.merge(table, on=["barcode", "window_id", "proposal"], suffixes=("", "_table"))
    assert len(merged) == len(table) == len(frame), (len(frame), len(table))
    for k in ["B_core_enrichment", "T_rim_enrichment", "area_um2", "HookNet_GC_pixel_fraction"]:
        assert np.allclose(merged[k], merged[f"{k}_table"], equal_nan=True), k
    clinical = pd.read_csv(ESCA_OUT / "rna" / "clinical.csv").set_index("case_id")
    frame["histology"] = frame.case_id.map(clinical.histology_primary)
    TABLES.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def order_and_name(fit, frame, labels_key="labels"):
    """Relabel states by descending mean B share and name them by rule (state means vs cohort means):
    highest B share -> B-follicular; of the rest, the highest T share among states with T share >= the cohort
    mean and >= 2x their B share -> T-dominant; then the highest macrophage share if >= 2x the cohort mean ->
    Macrophage-rich/mixed; then states with Other share >= cohort mean + 0.10 -> Lymphoid-poor (Other-rich);
    others -> Intermediate (+ the feature that most distinguishes them).
    (Amended after the first fit: the first rule named any second-highest-T state "T-dominant" and any state
    with >= 1.25x mean macrophage share "Macrophage-rich", which mislabelled an Other-rich k = 2 state and a
    T-rich k = 3 state with 1.4% macrophages. Only names changed; the B-follicular state is unaffected.)"""
    model, labels = fit["model"], fit[labels_key]
    k = len(model["weights"])
    means = pd.DataFrame({f: [frame[f].to_numpy()[labels == j].mean() for j in range(k)] for f in frame.columns})
    order = list(np.argsort(-means.B_share.to_numpy(), kind="stable"))
    remap = {old: new for new, old in enumerate(order)}
    model = {"covariance_type": model["covariance_type"],
             "weights": [float(model["weights"][j]) for j in order],
             "means": [np.asarray(model["means"][j]).tolist() for j in order],
             "covariances": [np.asarray(model["covariances"][j]).tolist() for j in order]}
    labels = np.asarray([remap[v] for v in labels])
    means = means.iloc[order].reset_index(drop=True)
    cohort = frame.mean()
    names = [None] * k
    names[0] = "B-follicular"
    rest = list(range(1, k))
    t_ok = [j for j in rest if means.T_share[j] >= cohort.T_share and means.T_share[j] >= 2 * means.B_share[j]]
    if t_ok:
        t = max(t_ok, key=lambda j: means.T_share[j])
        names[t] = "T-dominant"
        rest.remove(t)
    if rest:
        m = max(rest, key=lambda j: means.Macrophage_share[j])
        if means.Macrophage_share[m] >= 2 * cohort.Macrophage_share:
            names[m] = "Macrophage-rich/mixed"
            rest.remove(m)
    for j in list(rest):
        if means.Other_share[j] >= cohort.Other_share + .10:
            names[j] = "Lymphoid-poor (Other-rich)" if "Lymphoid-poor (Other-rich)" not in names else f"Lymphoid-poor {j}"
            rest.remove(j)
    z = (means - cohort) / frame.std()
    for j in rest:
        f = z.loc[j, STATE_FEATURES].abs().idxmax()
        names[j] = f"Intermediate ({'high' if z.loc[j, f] > 0 else 'low'} {FEATURE_NAMES[f].lower()})"
    return model, labels, names


def fit_state_model(frame):
    X, medians, mean, sd = prepare(frame, STATE_FEATURES)
    grid, fits = [], {}
    for kind in ["full", "diag"]:
        for k in K_RANGE:
            f = fit_gmm(X, k, kind, seed=SEED)
            fits[(k, kind)] = f
            grid.append({"k": k, "covariance_type": kind, "BIC": f["BIC"], "log_likelihood": f["log_likelihood"],
                         "n_parameters": f["n_parameters"], "min_state_fraction": f["min_state_fraction"],
                         "eligible": f["min_state_fraction"] >= MIN_STATE_FRACTION})
    grid = pd.DataFrame(grid)
    eligible = grid[grid.eligible]
    primary_key = tuple(eligible.sort_values("BIC").iloc[0][["k", "covariance_type"]])
    k3 = grid[(grid.k == 3) & grid.eligible]
    k3_key = tuple((k3 if len(k3) else grid[grid.k == 3]).sort_values("BIC").iloc[0][["k", "covariance_type"]])
    raw = frame[STATE_FEATURES].copy()
    raw[:] = np.where(np.isfinite(raw.to_numpy(float)), raw.to_numpy(float), medians)
    model = {"features": STATE_FEATURES, "impute_median": medians.tolist(), "mean": mean.tolist(), "sd": sd.tolist(),
             "fitted_on": f"{len(frame)} accepted two-stage aggregates in TCGA-ESCA windows (dev + test), "
                          f"{frame.case_id.nunique()} patients; adopted classifier he2cell_esca_finetuned.pth + tau 0.5 prior",
             "selection_rule": f"min BIC over k = {K_RANGE.start}..{K_RANGE.stop - 1} x covariance (full, diag) "
                               f"with every state >= {MIN_STATE_FRACTION:.0%} of objects (hard assignment)",
             "naming_rule": " ".join(order_and_name.__doc__.split()), "seed": SEED, "fitter": "numpy EM, 20 k-means++ starts, reg_covar 1e-3"}
    for name, key in [("primary", primary_key), ("k3", k3_key)]:
        params, labels, names = order_and_name(fits[(int(key[0]), key[1])], raw.assign(**{
            c: frame[c] for c in frame.columns if c not in STATE_FEATURES}).select_dtypes("number"))
        model[name] = {**params, "k": int(key[0]), "state_names": names, "BIC": fits[(int(key[0]), key[1])]["BIC"]}
    save_json(model, MODEL_JSON)
    import pickle
    MODEL_JSON.with_suffix(".pkl").write_bytes(pickle.dumps(model))
    return model, grid


def stability(frame, model, n_boot=200):
    """Patient-cluster bootstrap: refit the chosen (k, covariance) and compare its assignment of the full
    cohort with the frozen model (adjusted Rand index)."""
    rng = np.random.default_rng(SEED)
    X, *_ = prepare(frame, model["features"], model["impute_median"], model["mean"], model["sd"])
    out = {}
    for which in ["primary", "k3"]:
        m = model[which]
        reference = log_resp(X, m)[0].argmax(1)
        cases = frame.case_id.unique()
        aris = []
        for b in range(n_boot):
            pick = np.concatenate([np.flatnonzero(frame.case_id.to_numpy() == c) for c in rng.choice(cases, len(cases))])
            f = fit_gmm(X[pick], m["k"], m["covariance_type"], n_init=5, seed=SEED + b)
            aris.append(adjusted_rand(reference, log_resp(X, f["model"])[0].argmax(1)))
        out[which] = {"median_ARI": float(np.median(aris)), "IQR_ARI": np.percentile(aris, [25, 75]).tolist(), "resamples": n_boot}
    return out


def stability_grid(frame, model, grid, n_boot=100):
    """Bootstrap ARI (as stability()) for every eligible (k, covariance) of the BIC grid, refitted on the full
    cohort with the same seed; descriptive (it does not change the frozen choice)."""
    rng = np.random.default_rng(SEED + 1)
    X, *_ = prepare(frame, model["features"], model["impute_median"], model["mean"], model["sd"])
    cases = frame.case_id.unique()
    picks = [np.concatenate([np.flatnonzero(frame.case_id.to_numpy() == c) for c in rng.choice(cases, len(cases))])
             for _ in range(n_boot)]
    rows = []
    for g in grid[grid.eligible].itertuples():
        reference = fit_gmm(X, g.k, g.covariance_type, seed=SEED)["labels"]
        aris = [adjusted_rand(reference, log_resp(X, fit_gmm(X[p], g.k, g.covariance_type, n_init=5, seed=SEED + b)["model"])[0].argmax(1))
                for b, p in enumerate(picks)]
        rows.append({"k": g.k, "covariance_type": g.covariance_type, "BIC": g.BIC, "median_ARI": float(np.median(aris)),
                     "IQR_low": float(np.percentile(aris, 25)), "IQR_high": float(np.percentile(aris, 75))})
    return pd.DataFrame(rows)


def gc_sensitivity(frame, model):
    """Refit the primary (k, covariance) with HookNet GC pixel fraction (arcsine sqrt) as a 12th feature."""
    m = model["primary"]
    extra = frame.assign(gc_asin=np.arcsin(np.sqrt(frame.HookNet_GC_pixel_fraction.clip(0, 1))))
    X, *_ = prepare(extra, STATE_FEATURES + ["gc_asin"])
    f = fit_gmm(X, m["k"], m["covariance_type"], seed=SEED)
    reference = assign_states(frame, "primary", model).state.to_numpy()
    return {"ARI_vs_primary": adjusted_rand(reference, f["labels"]), "k": m["k"], "covariance_type": m["covariance_type"]}


# ------------------------------------------------------------------ RNA programs

def rna_genes():
    """log2(TPM+1) per primary-tumour aliquot for every program gene; missing genes are pulled from Xena
    with esca_rna's helpers (public TCGA ids and gene symbols only) and cached in tables/."""
    cache = TABLES / "rna_program_genes_log2tpm.csv"
    if cache.exists():
        return pd.read_csv(cache)
    base = pd.read_csv(ESCA_OUT / "rna" / "log2_tpm_signature_genes.csv")
    needed = sorted({g for genes in PROGRAMS.values() for g in genes} | set(CHEMOKINE12))
    missing = [g for g in needed if g not in base.columns]
    if missing:
        import esca_rna as R
        samples = base.sample_id.tolist()
        values = R.xena("datasetGeneProbeAvg", [R.DATASET, samples, missing])
        for row in values:
            scores = np.asarray(row["scores"], float).reshape(-1)
            if len(scores) != len(samples) or not np.isfinite(scores).all():
                raise ValueError(f"Missing values for {row.get('gene')}")
            base[row["gene"]] = scores
        absent = set(missing) - set(base.columns)
        if absent:
            raise ValueError(f"Genes not in {R.DATASET}: {sorted(absent)}")
    base = base[["case_id", "sample_id", *needed]]
    base.to_csv(cache, index=False)
    return base


def patient_programs():
    genes = rna_genes()
    patients = genes.groupby("case_id")[[c for c in genes.columns if c not in ("case_id", "sample_id")]].median()
    z = (patients - patients.mean()) / patients.std(ddof=1)
    out = pd.DataFrame({name: z[g].mean(1) for name, g in PROGRAMS.items()})
    sig = pd.read_csv(ESCA_OUT / "rna" / "patient_signatures.csv").set_index("case_id")
    out.insert(0, "chemokine12", sig.chemokine12_PC1.reindex(out.index))
    return out


def spearman_ci(frame, x, y, cluster="case_id", n=2000):
    from scipy import stats
    d = frame[[x, y, cluster]].dropna()
    rho = stats.spearmanr(d[x], d[y]).statistic if len(d) > 3 else np.nan
    p = stats.spearmanr(d[x], d[y]).pvalue if len(d) > 3 else np.nan
    ci = cluster_bootstrap_ci(d, cluster, lambda s: stats.spearmanr(s[x], s[y]).statistic, n=n)
    return {"rho": float(rho), "CI95": ci, "p": float(p), "n": len(d), "clusters": int(d[cluster].nunique())}


# ------------------------------------------------------------------ figures

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
    return plt


def heatmap(frame, labels, names, path, title):
    plt = _plt()
    cols = STATE_FEATURES + ["Other_share", "Macrophage_radial_gradient", "HookNet_GC_pixel_fraction"]
    z = (frame[cols] - frame[cols].mean()) / frame[cols].std()
    means = np.asarray([z[labels == j].mean().to_numpy() for j in range(len(names))])
    fig, ax = plt.subplots(figsize=(7.6, .55 * len(names) + 2.2))
    im = ax.imshow(means, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{n} (n={int((labels == j).sum())})" for j, n in enumerate(names)])
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([FEATURE_NAMES[c] + (" *" if c not in STATE_FEATURES else "") for c in cols], rotation=50, ha="right")
    for (i, j), v in np.ndenumerate(means):
        ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7, color="white" if abs(v) > .9 else "black")
    fig.colorbar(im, ax=ax, fraction=.025, label="mean z-score")
    ax.set_title(title + "  (* descriptive, not a clustering feature)", fontsize=8, loc="left")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def radial_figure(frame, labels, names, path):
    plt = _plt()
    colours = {"B": "#eb6834", "T": "#2a78d6", "Macrophage": "#1baf7a"}
    mids = [(lo + hi) / 2 for lo, hi in zip(RINGS_UM[:-1], RINGS_UM[1:])]
    fig, axes = plt.subplots(1, len(names), figsize=(2.6 * len(names), 2.6), sharey=True)
    for j, (ax, name) in enumerate(zip(np.atleast_1d(axes), names)):
        part = frame[labels == j]
        for t, c in colours.items():
            v = part[[f"{t}_per_mm2_{lo}_{hi}um" for lo, hi in zip(RINGS_UM[:-1], RINGS_UM[1:])]]
            ax.plot(mids, v.median(), "-o", color=c, ms=3, label=t)
            ax.fill_between(mids, v.quantile(.25), v.quantile(.75), color=c, alpha=.12, lw=0)
        ax.set_title(f"{name}\n(n={len(part)})", fontsize=8)
        ax.set_xlabel("µm from centroid")
    np.atleast_1d(axes)[0].set_ylabel("cells / mm² (median, IQR)")
    np.atleast_1d(axes)[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def gallery(frame, model, per_state=4):
    """Example tiles per state (primary and k = 3; highest posterior, one per patient) with esca_report.gallery_tile."""
    import esca_report as ER
    from PIL import Image
    out = {}
    picks = []
    for which, state_col, prefix in [("primary", "state", ""), ("k3", "state_k3", "k3_")]:
        for j, name in enumerate(model[which]["state_names"]):
            label = f"{name} ({'primary' if which == 'primary' else 'k = 3'})"
            part = frame[frame[state_col] == j].sort_values(f"{prefix}p_state{j}", ascending=False).drop_duplicates("case_id").head(per_state)
            picks += [(label, r._replace(state=j) if which == "k3" else r, f"{prefix}p_state{j}") for r in part.itertuples()]
    by_slide = {}
    for name, r, pcol in picks:
        by_slide.setdefault(r.barcode, []).append((name, r, pcol))
    for barcode, items in by_slide.items():
        wanted = {(r.window_id, r.proposal) for _, r, _ in items}
        _, kept = esca_slide((barcode, wanted, True))
        for name, r, pcol in items:
            mask, x = kept[(r.window_id, r.proposal)]
            tile = ER.gallery_tile(barcode, r.window_id, x, mask)
            path = FIG / "gallery" / f"{''.join(ch for ch in name if ch.isalnum())[:30]}_{barcode[:12]}_{r.window_id}_{r.proposal}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(tile).save(path)
            caption = (f"{r.case_id} {r.window_id}#{r.proposal} · {r.histology} · B {r.B_share:.0%} T {r.T_share:.0%} "
                       f"Mac {r.Macrophage_share:.1%} · {r.area_um2 / 1e3:.0f}k µm² · HookNet GC {r.HookNet_GC_pixel_fraction:.0%} · p={getattr(r, pcol):.2f}")
            out.setdefault(name, []).append({"path": str(path.relative_to(OUT)), "caption": caption})
    return out


# ------------------------------------------------------------------ main

def describe(frame, labels, names):
    rows = []
    cols = STATE_FEATURES + ["area_um2", "cells", "Other_share", "Macrophage_radial_gradient", "HookNet_GC_pixel_fraction"] + ring_columns()
    for j, name in enumerate(names):
        part = frame[labels == j]
        rows.append({"state": j, "state_name": name, "objects": len(part), "fraction": len(part) / len(frame),
                     "patients": part.case_id.nunique(), "with_HookNet_GC_1pct": float((part.HookNet_GC_pixel_fraction >= .01).mean()),
                     "matches_HookNet": float(part.matches_HookNet.mean()), "touches_window_edge": float(part.touches_grid_edge.mean()),
                     **{f"median_{c}": float(part[c].median()) for c in cols}})
    return pd.DataFrame(rows)


def main():
    import argparse
    from scipy import stats
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true", help="recompute aggregate features")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-gallery", action="store_true")
    parser.add_argument("--gallery-only", action="store_true", help="redraw the example tiles from the saved state table")
    args = parser.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    if args.gallery_only:
        summary = json.loads((PARTS / "esca_states.json").read_text())
        summary["gallery"] = gallery(pd.read_csv(TABLES / "esca_window_tls_states.csv"), load_model())
        save_json(summary, PARTS / "esca_states.json")
        return
    frame = esca_features(args.workers, args.recompute)
    print(f"{len(frame)} aggregates, {frame.case_id.nunique()} patients", flush=True)
    model, grid = fit_state_model(frame)
    grid.to_csv(TABLES / "state_model_selection.csv", index=False)
    summary = {"objects": len(frame), "patients": int(frame.case_id.nunique()), "slides": int(frame.barcode.nunique()),
               "by_split": frame.split.value_counts().to_dict(), "by_window_category": frame.category.value_counts().to_dict(),
               "model_selection": grid.to_dict("records"), "stage1": STAGE1, "b_share_cut": FROZEN["b_share"],
               "features": STATE_FEATURES}
    assigned = {}
    for which in ["primary", "k3"]:
        a = assign_states(frame, which, model)
        assigned[which] = a
        m = model[which]
        prof = describe(frame, a.state.to_numpy(), m["state_names"])
        prof.to_csv(TABLES / f"state_profiles_{which}.csv", index=False)
        heatmap(frame, a.state.to_numpy(), m["state_names"], FIG / f"heatmap_{which}.png",
                f"ESCA window TLS states, {which} (k = {m['k']}, {m['covariance_type']} covariance)")
        radial_figure(frame, a.state.to_numpy(), m["state_names"], FIG / f"radial_{which}.png")
        summary[which] = {"k": m["k"], "covariance_type": m["covariance_type"], "state_names": m["state_names"],
                          "profiles": prof.to_dict("records")}
    summary["primary_vs_k3_ARI"] = adjusted_rand(assigned["primary"].state, assigned["k3"].state)
    summary["primary_vs_k3_crosstab"] = pd.crosstab(assigned["primary"].state_name, assigned["k3"].state_name).to_dict()
    print(json.dumps({w: {"k": summary[w]["k"], "cov": summary[w]["covariance_type"], "names": summary[w]["state_names"]}
                      for w in ["primary", "k3"]}), flush=True)
    summary["stability"] = stability(frame, model)
    summary["gc_feature_sensitivity"] = gc_sensitivity(frame, model)
    sg = stability_grid(frame, model, grid)
    sg.to_csv(TABLES / "state_model_stability_grid.csv", index=False)
    summary["stability_grid"] = sg.to_dict("records")
    out = assigned["primary"].copy()
    k3 = assigned["k3"]
    out["state_k3"], out["state_name_k3"] = k3.state, k3.state_name
    for c in [c for c in k3.columns if c.startswith("p_state")]:
        out[f"k3_{c}"] = k3[c]
    out.to_csv(TABLES / "esca_window_tls_states.csv", index=False)

    # ---------- patient level: histology and RNA programs
    names = model["primary"]["state_names"]
    patients = out.groupby("case_id").agg(aggregates=("state", "size"), histology=("histology", "first"),
                                          mean_p_B_follicular=("p_B_follicular", "mean"))
    for j, name in enumerate(names):
        patients[f"frac_{name}"] = out.groupby("case_id").state.apply(lambda s: (s == j).mean())
    for j, name in enumerate(model["k3"]["state_names"]):
        patients[f"k3frac_{name}"] = out.groupby("case_id").state_k3.apply(lambda s: (s == j).mean())
    programs = patient_programs()
    programs.to_csv(TABLES / "esca_patient_rna_programs.csv")
    patients = patients.join(programs, how="left")
    patients.reset_index().to_csv(TABLES / "esca_patient_state_fractions.csv", index=False)
    assoc = []
    exposures = [f"frac_{n}" for n in names] + [f"k3frac_{n}" for n in model["k3"]["state_names"]] + ["mean_p_B_follicular"]
    d = patients.reset_index()
    for e in exposures:
        for p in ["chemokine12", *PROGRAMS]:
            for subset, part in [("all", d), ("ESCC", d[d.histology == "ESCC"]), ("EAC", d[d.histology == "EAC"])]:
                assoc.append({"exposure": e, "program": p, "subset": subset, **spearman_ci(part, e, p)})
    assoc = pd.DataFrame(assoc)
    assoc.to_csv(TABLES / "esca_state_rna_associations.csv", index=False)
    hist = []
    for e in exposures:
        a, b = d.loc[d.histology == "ESCC", e].dropna(), d.loc[d.histology == "EAC", e].dropna()
        diff_ci = cluster_bootstrap_ci(d[d.histology.isin(["ESCC", "EAC"])], "case_id",
                                       lambda s: s.loc[s.histology == "EAC", e].mean() - s.loc[s.histology == "ESCC", e].mean())
        hist.append({"exposure": e, "ESCC_n": len(a), "EAC_n": len(b), "ESCC_mean": float(a.mean()), "EAC_mean": float(b.mean()),
                     "EAC_minus_ESCC": float(b.mean() - a.mean()), "CI95": diff_ci,
                     "mann_whitney_p": float(stats.mannwhitneyu(a, b).pvalue) if len(a) and len(b) else np.nan})
    hist = pd.DataFrame(hist)
    hist.to_csv(TABLES / "esca_state_histology.csv", index=False)
    summary["patients_with_aggregates"] = len(d)
    summary["histology"] = hist.to_dict("records")
    summary["aggregate_state_by_histology"] = pd.crosstab(out.histology, out.state_name).to_dict()
    summary["rna_associations"] = assoc.to_dict("records")
    if not args.no_gallery:
        summary["gallery"] = gallery(out, model)
    save_json(summary, PARTS / "esca_states.json")
    print("Part A written", flush=True)


if __name__ == "__main__":
    main()
