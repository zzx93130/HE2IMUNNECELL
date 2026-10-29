"""HoVer-Net (PanNuke) inflammatory-cell baseline for TLS detection: tuning, frozen evaluation, paired stats.

Cells come from deployment/baselines/hovernet_cells.py (TIAToolbox hovernet_fast-pannuke). The PanNuke
"inflammatory" class is the standard lymphocyte proxy. Two detector arms, both tuned on ESCA dev
patients (split == "dev" in the adopted windows.csv), frozen, then scored once on ESCA test patients and
on the 8 USZ Visium sections:

(a) HoVer-Net inflammatory density: the esca_analysis machinery unchanged (Gaussian-smoothed density on
    the 8 µm grid, threshold, minimum area, minimum cells; esca_analysis.detect) with the inflammatory
    count grid in place of the predicted-B grid. Grid: sigma {15, 25} µm, threshold (cells/mm²),
    minimum area {5000, 10000, 20000} µm², minimum cells {20, 50}.
(b) HoVer-Net inflammatory DBSCAN (Barmpoutis-style lymphocyte clustering): DBSCAN on inflammatory-cell
    centroids in valid tissue, eps 30 µm, min_samples tuned on dev from {20, 50, 100}; a cluster is a
    TLS candidate when its convex hull is >= 20 000 µm². The hull, rasterised on the 8 µm grid and
    clipped to valid tissue, is the detected object. Window score = cells in the largest cluster.

Evaluation is identical to the existing arms: ESCA object precision/recall/F1 against HookNet-TLS
(esca_analysis.match_counts), dense-window specificity, window AUROC; Visium spot AUROC (TLS vs all,
TLS vs INFL), object recall, INFL-called-TLS rate (visium_paired_stats.visium_metric). Paired differences
against the adopted HE2CELL B-density arm reuse esca_paired_stats / visium_paired_stats (same 2000
cluster resamples for both arms, sign-flip permutation p). When inference is incomplete every arm is
restricted to the windows / sections HoVer-Net has finished, and the output is marked INTERIM.

Cell-level check on Visium: per in-tissue spot, HoVer-Net inflammatory count vs HE2CELL lymphocyte (T+B)
count, and each vs spot RNA (PTPRC, CD3E, MS4A1; log1p CP10k): Spearman per section, averaged over
sections, section-cluster bootstrap CI (paired for HoVer-Net vs HE2CELL).

DBSCAN is implemented with scipy (cKDTree core points + connected components; border points join their
nearest core point's cluster), because the main .venv has no scikit-learn; `--check-dbscan` (run in the
TIAToolbox venv) verifies it against sklearn.cluster.DBSCAN.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deployment" / "esca_tls"))
sys.path.insert(0, str(ROOT / "deployment" / "visium_tls"))
import esca_analysis as A  # noqa: E402
import esca_paired_stats as P  # noqa: E402
import visium_eval as VE  # noqa: E402
import visium_paired_stats as VP  # noqa: E402
from esca_common import save_json  # noqa: E402
from esca_qc import slide_qc  # noqa: E402

V = VE.V
OUT = ROOT / "outputs" / "baselines_hovernet_20261026"
CELLS = OUT / "cells"
ADOPTED_DIR = A.OUT / "analysis_he2cell_ft_tau05"
GRID_UM = A.GRID_UM
TYPES = {0: "no_label", 1: "neoplastic", 2: "inflammatory", 3: "connective", 4: "dead", 5: "epithelial"}
INFL = "inflammatory"
DENSITY_ARM, DBSCAN_ARM = "HoVer-Net inflammatory density", "HoVer-Net inflammatory DBSCAN"
ADOPTED, NUCLEI = "HE2CELL B density (adopted)", "Nuclei only (no classifier)"
PARAM_GRID = {
    DENSITY_ARM: {"sigma_um": [15, 25],
                  "threshold": [1000, 1500, 2000, 3000, 4000, 5000, 6500, 8000, 10000, 12500, 15000],
                  "min_area_um2": [5000, 10000, 20000], "min_cells": [20, 50]},
    DBSCAN_ARM: {"eps_um": [30], "min_samples": [20, 50, 100], "min_area_um2": [20000]},
}
GENES = ["PTPRC", "CD3E", "MS4A1"]


# ---------------------------------------------------------------- DBSCAN (scipy)
def dbscan(xy, eps, min_samples):
    """Labels like sklearn DBSCAN(eps, min_samples) with Euclidean metric (-1 = noise).

    Neighbourhoods include the point itself (sklearn convention). Border points join the cluster of their
    nearest core neighbour (sklearn takes the first core point that reaches them; same clusters' cores).
    """
    n = len(xy)
    labels = np.full(n, -1)
    if n == 0:
        return labels
    tree = cKDTree(xy)
    pairs = tree.query_pairs(eps, output_type="ndarray")
    degree = np.bincount(pairs.ravel(), minlength=n) + 1
    core = degree >= min_samples
    if not core.any():
        return labels
    both = pairs[core[pairs[:, 0]] & core[pairs[:, 1]]]
    index = np.flatnonzero(core)
    position = np.full(n, -1)
    position[index] = np.arange(len(index))
    graph = sparse.coo_matrix((np.ones(len(both)), (position[both[:, 0]], position[both[:, 1]])), shape=(len(index), len(index)))
    _, components = sparse.csgraph.connected_components(graph, directed=False)
    labels[index] = components
    border = np.flatnonzero(~core)
    if len(border):
        core_tree = cKDTree(xy[index])
        distance, nearest = core_tree.query(xy[border], distance_upper_bound=eps * (1 + 1e-9))
        hit = np.isfinite(distance)
        labels[border[hit]] = components[nearest[hit]]
    return labels


def cluster_mask(xy_um, labels, valid, min_area_um2):
    """(mask on the 8 µm grid of clusters whose convex hull reaches min_area_um2, largest cluster size)."""
    keep = np.zeros(valid.shape, np.uint8)
    largest = 0
    for label in np.unique(labels[labels >= 0]):
        points = xy_um[labels == label].astype(np.float32)
        largest = max(largest, len(points))
        if len(points) < 3:
            continue
        hull = cv2.convexHull(points)
        if cv2.contourArea(hull) >= min_area_um2:
            cv2.fillPoly(keep, [np.round(hull[:, 0, :] / GRID_UM - .5).astype(np.int32)], 1)
    return keep.astype(bool) & valid, largest


def detect_dbscan(x, eps_um, min_samples, min_area_um2):
    """x: inputs dict with valid grid and inflammatory centroids (µm, grid frame)."""
    xy = x["infl_xy_um"]
    labels = dbscan(xy, eps_um, min_samples)
    return cluster_mask(xy, labels, x["valid"], min_area_um2)


# ---------------------------------------------------------------- ESCA
def esca_cells(barcode, window_id):
    path = CELLS / "esca" / barcode / f"{window_id}.csv.gz"
    return pd.read_csv(path) if path.exists() else None


def count_grids(gx, gy, kinds, shape):
    ny, nx = shape
    return {name: np.bincount(gy[kinds == name] * nx + gx[kinds == name], minlength=ny * nx).reshape(ny, nx).astype(float)
            for name in TYPES.values()}


def infl_points(gx, gy, xy_um, kinds, valid):
    """Inflammatory centroids (µm, grid frame) lying on valid tissue."""
    m = (kinds == INFL) & valid[gy, gx]
    return xy_um[m]


def load_esca():
    """Adopted windows table (split, labels) restricted to windows with HoVer-Net cells, and their inputs."""
    windows = pd.read_csv(ADOPTED_DIR / "windows.csv")
    have = np.array([(CELLS / "esca" / b / f"{w}.csv.gz").exists() for b, w in zip(windows.barcode, windows.window_id)])
    windows = windows[have].reset_index(drop=True)
    inputs = {}
    for barcode, part in windows.groupby("barcode"):
        slide_folder = A.OUT / "slides" / barcode
        summary = json.loads((slide_folder / "hooknet" / "summary.json").read_text())
        maps = np.load(slide_folder / "hooknet" / "maps_4um.npz")
        maps = {k: maps[k] for k in ["prediction_mask"]}
        selection = json.loads((slide_folder / "he2cell" / "window_selection.json").read_text())
        flagged = set(selection["artifact_flagged_hooknet_tls"])
        features = [f for f in json.loads((slide_folder / "hooknet" / "native_polygons.geojson").read_text())["features"]
                    if f["properties"]["predicted_label"] == "tls" and f["properties"]["passes_published_filter"]
                    and f["properties"]["region_id"] not in flagged]
        qc = slide_qc(slide_folder)
        geometry = pd.read_csv(slide_folder / "he2cell" / "windows.csv").set_index("window_id")
        for window_id in part.window_id:
            cells = esca_cells(barcode, window_id)
            cells = cells.assign(window_id=window_id, celltype=cells.type_name)
            w = geometry.loc[window_id]
            w = pd.Series({**w.to_dict(), "window_id": window_id})
            x = A.window_inputs(slide_folder, w, summary, maps, features, qc, cells)
            n, mpp = x["n"], x["mpp"]
            gx = ((cells.centroid_x_global - w.x0) * mpp / GRID_UM).astype(int).clip(0, n - 1).to_numpy()
            gy = ((cells.centroid_y_global - w.y0) * mpp / GRID_UM).astype(int).clip(0, n - 1).to_numpy()
            kinds = cells.type_name.to_numpy()
            x["counts"] = count_grids(gx, gy, kinds, (n, n))
            x["infl_xy_um"] = infl_points(gx, gy, x["cells_xy_um"], kinds, x["valid"])
            inputs[(barcode, window_id)] = x
    return windows, inputs


def evaluate(windows, inputs, arm, params):
    rows = []
    for w in windows.itertuples():
        x = inputs[(w.barcode, w.window_id)]
        if arm == DENSITY_ARM:
            predicted, d = A.detect(x, [INFL], **params)
            score = float((d * x["valid"]).max())
        else:
            predicted, score = detect_dbscan(x, **params)
        rows.append({"barcode": w.barcode, "case_id": w.case_id, "window_id": w.window_id, "max_density": float(score),
                     "proposals_rejected": 0, "predicted_um2": predicted.sum() * GRID_UM ** 2,
                     "dice": 2 * (predicted & x["reference"]).sum() / max(predicted.sum() + x["reference"].sum(), 1),
                     **A.match_counts(predicted, x["reference"], x["raw_tls"])})
    return pd.DataFrame(rows).merge(windows, on=["barcode", "case_id", "window_id"]).assign(method=arm)


def tune(windows, inputs):
    """Dev-only grid search per arm (best dev object F1; first of ties), like esca_analysis.tune."""
    dev = windows[windows.split == "dev"]
    chosen, grid_rows, edges = {}, [], {}
    for arm, grid in PARAM_GRID.items():
        best = None
        for values in itertools.product(*grid.values()):
            params = dict(zip(grid, values))
            p, r, f1 = A.prf(evaluate(dev, inputs, arm, params))
            grid_rows.append({"method": arm, **params, "dev_precision": p, "dev_recall": r, "dev_F1": f1})
            if best is None or f1 > best[0]:
                best = (f1, params)
        chosen[arm] = best[1]
        edges[arm] = {k: bool(v in (grid[k][0], grid[k][-1])) for k, v in best[1].items() if len(grid[k]) > 2}
    return chosen, pd.DataFrame(grid_rows), edges


# ---------------------------------------------------------------- Visium
def visium_cells(sample):
    path = CELLS / "visium" / f"{sample}.csv.gz"
    return pd.read_csv(path) if path.exists() else None


def visium_inputs(sample, cells):
    """Grids on the 8 µm native-frame grid as in visium_eval.sample_inputs, with HoVer-Net classes."""
    mpp = V.native_mpp(sample)
    spots = V.spots(sample)
    with __import__("tifffile").TiffFile(V.DATA / "tif_slides" / f"{sample}.tif") as tif:
        height, width = tif.pages[0].shape[:2]
    ny, nx = int(np.ceil(height * mpp / GRID_UM)), int(np.ceil(width * mpp / GRID_UM))
    gx = (cells.x_native * mpp / GRID_UM).astype(int).clip(0, nx - 1).to_numpy()
    gy = (cells.y_native * mpp / GRID_UM).astype(int).clip(0, ny - 1).to_numpy()
    valid = np.zeros((ny, nx), np.uint8)
    for s in spots[spots.in_tissue == 1].itertuples():
        cv2.circle(valid, (int(s.x * mpp / GRID_UM), int(s.y * mpp / GRID_UM)), int(round(62 / GRID_UM)), 1, -1)
    valid = valid.astype(bool)
    kinds = cells.type_name.to_numpy()
    xy_um = np.stack([cells.x_native.to_numpy() * mpp, cells.y_native.to_numpy() * mpp], 1)
    return {"counts": count_grids(gx, gy, kinds, (ny, nx)), "valid": valid, "n": max(nx, ny), "shape": (ny, nx),
            "mpp": mpp, "spots": spots, "infl_xy_um": infl_points(gx, gy, xy_um, kinds, valid)}


def visium_spot_table(samples, chosen):
    rows = []
    for sample in samples:
        x = visium_inputs(sample, visium_cells(sample))
        spots, mpp, shape = x["spots"], x["mpp"], x["shape"]
        disks = VE.spot_disks(spots, mpp, shape)
        p = chosen[DENSITY_ARM]
        density_mask, _ = A.detect(x, [INFL], **p)
        infl_density = A.density(x["counts"], [INFL], p["sigma_um"])
        dbscan_mask, _ = detect_dbscan(x, **chosen[DBSCAN_ARM])
        total = sum(x["counts"].values())
        for i, s in enumerate(spots.itertuples()):
            ys, xs = disks[i]
            rows.append({"sample": sample, "barcode": s.barcode, "label": s.label, "in_tissue": s.in_tissue,
                         "hovernet_nuclei": float(total[ys, xs].sum()), "hovernet_inflammatory": float(x["counts"][INFL][ys, xs].sum()),
                         "infl_density_at_spot": float(infl_density[ys, xs].mean()),
                         "cover_infl_density": float(density_mask[ys, xs].mean()),
                         "cover_infl_dbscan": float(dbscan_mask[ys, xs].mean())})
    return pd.DataFrame(rows)


def genes(sample, barcodes):
    """log1p CP10k of GENES per spot (same normalisation as visium_eval.expression)."""
    import h5py
    f = h5py.File(V.DATA / "h5ad_preprocessed" / f"{sample}.h5ad", "r")
    X = f["X"]
    matrix = X[:] if isinstance(X, h5py.Dataset) else sparse.csr_matrix((X["data"][:], X["indices"][:], X["indptr"][:])).toarray()
    names = [g.decode() for g in f["var"]["gene_name"][:]]
    index = [b.decode() for b in f["obs"]["_index"][:]]
    library = matrix.sum(1, keepdims=True)
    frame = pd.DataFrame(np.log1p(matrix[:, [names.index(g) for g in GENES]] / np.maximum(library, 1) * 1e4), index=index, columns=GENES)
    return frame.loc[barcodes]


def cell_level(hover, adopted):
    """Spearman per section between per-spot counts and RNA; mean over sections with section bootstrap."""
    tissue = hover[hover.in_tissue == 1].merge(
        adopted[["sample", "barcode", "B", "T", "cells"]].rename(columns={"cells": "he2cell_nuclei"}), on=["sample", "barcode"])
    tissue["he2cell_lymphocytes"] = tissue["B"] + tissue["T"]
    tissue = pd.concat([part.join(genes(sample, part.barcode).reset_index(drop=True).set_index(part.index))
                        for sample, part in tissue.groupby("sample")])
    samples = sorted(tissue["sample"].unique())
    boot = P.draws(len(samples))
    scores = {"HoVer-Net inflammatory": "hovernet_inflammatory", "HE2CELL lymphocytes (T+B)": "he2cell_lymphocytes",
              "HoVer-Net all nuclei": "hovernet_nuclei", "HE2CELL all nuclei": "he2cell_nuclei"}
    def per_section(a, b):
        return np.array([stats.spearmanr(tissue.loc[tissue["sample"] == s, a], tissue.loc[tissue["sample"] == s, b]).statistic
                         for s in samples])
    def summarise(rho):
        draws = np.nanmean(rho[boot], axis=1)
        return {"mean_spearman": float(np.nanmean(rho)), "CI95": np.nanpercentile(draws, [2.5, 97.5]).tolist(),
                "per_section": {s: round(float(r), 3) for s, r in zip(samples, rho)}}
    out = {"spots": int(len(tissue)), "sections": samples,
           "counts_agreement": summarise(per_section("hovernet_inflammatory", "he2cell_lymphocytes")),
           "vs_RNA": [], "paired_HoVerNet_minus_HE2CELL": []}
    rho = {}
    for (name, column), gene in itertools.product(scores.items(), GENES):
        rho[(name, gene)] = per_section(column, gene)
        out["vs_RNA"].append({"score": name, "gene": gene, **summarise(rho[(name, gene)])})
    for gene in GENES:
        for a, b in [("HoVer-Net inflammatory", "HE2CELL lymphocytes (T+B)"), ("HoVer-Net all nuclei", "HE2CELL all nuclei")]:
            d = rho[(a, gene)] - rho[(b, gene)]
            draws = np.nanmean(d[boot], axis=1)
            out["paired_HoVerNet_minus_HE2CELL"].append({"a": a, "b": b, "gene": gene, "difference": float(np.nanmean(d)),
                                                        "CI95_difference": np.nanpercentile(draws, [2.5, 97.5]).tolist(),
                                                        "sections_a_higher": int((d > 0).sum())})
    tissue.to_csv(OUT / "visium_spot_cell_counts.csv.gz", index=False)
    return out


# ---------------------------------------------------------------- main
def progress():
    windows = pd.read_csv(ADOPTED_DIR / "windows.csv")
    done = sum((CELLS / "esca" / b / f"{w}.csv.gz").exists() for b, w in zip(windows.barcode, windows.window_id))
    sections = [s for s in V.SAMPLES if (CELLS / "visium" / f"{s}.csv.gz").exists()]
    return {"esca_windows_done": int(done), "esca_windows_total": len(windows),
            "visium_sections_done": sections, "visium_sections_total": len(V.SAMPLES),
            "complete": bool(done == len(windows) and len(sections) == len(V.SAMPLES))}


def main():
    global CELLS, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--cells-dir", default=None, help="HoVer-Net cell tables (default <out>/cells)")
    parser.add_argument("--check-dbscan", action="store_true", help="Compare the scipy DBSCAN with sklearn (TIAToolbox venv)")
    args = parser.parse_args()
    if args.check_dbscan:
        return check_dbscan()
    OUT = Path(args.out)
    CELLS = Path(args.cells_dir) if args.cells_dir else OUT / "cells"
    started = time.time()
    out = OUT
    analysis = out / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    state = progress()
    print("progress", json.dumps(state), flush=True)
    summary = {"design": __doc__, "progress": state, "interim": not state["complete"],
               "weights": json.loads((CELLS / "provenance.json").read_text()) if (CELLS / "provenance.json").exists() else None,
               "param_grid": PARAM_GRID}

    # ESCA
    windows, inputs = load_esca()
    summary["esca_windows_evaluated"] = {"all": len(windows), **windows.split.value_counts().to_dict(),
                                         "test_patients": int(windows[windows.split == "test"].case_id.nunique())}
    esca_ok = (windows.split == "dev").any() and (windows.split == "test").any()
    if esca_ok:
        chosen, grid, edges = tune(windows, inputs)
        grid.to_csv(analysis / "dev_parameter_grid.csv", index=False)
        per_window = pd.concat([evaluate(windows, inputs, arm, params) for arm, params in chosen.items()], ignore_index=True)
        per_window.to_csv(analysis / "per_window_results.csv", index=False)
        summary["chosen_parameters"] = chosen
        summary["chosen_value_at_grid_edge"] = edges
        summary["dev_object_F1"] = {arm: float(A.prf(per_window[(per_window.method == arm) & (per_window.split == "dev")])[2])
                                    for arm in chosen}
        keys = windows[windows.split == "test"][["barcode", "window_id"]]
        arms = {ADOPTED: P.load_arm(ADOPTED_DIR, "B (HE2CELL)"), NUCLEI: P.load_arm(ADOPTED_DIR, "All nuclei (no classifier)"),
                **{arm: P.load_arm(analysis, arm) for arm in chosen}}
        arms = {name: frame.merge(keys, on=["barcode", "window_id"]) for name, frame in arms.items()}
        for frame in arms.values():  # rank within the evaluated windows (load_arm ranked over its own rows)
            frame["rank_score"] = frame.max_density.rank(pct=True)
        pairs = [(ADOPTED, DENSITY_ARM), (ADOPTED, DBSCAN_ARM), (DENSITY_ARM, NUCLEI), (DENSITY_ARM, DBSCAN_ARM)]
        summary["ESCA"] = P.compare_arms(arms, pairs)
        summary["ESCA"]["dev_F1_adopted_reference"] = float(A.prf(P.load_arm(ADOPTED_DIR, "B (HE2CELL)", "dev").merge(
            windows[windows.split == "dev"][["barcode", "window_id"]], on=["barcode", "window_id"]))[2])
    else:
        print("ESCA: need HoVer-Net cells on both dev and test windows; skipped", flush=True)
        prior = out / "summary.json"
        if prior.exists():  # keep earlier frozen parameters if any (none on a first partial run)
            summary["chosen_parameters"] = json.loads(prior.read_text()).get("chosen_parameters")

    # Visium (frozen ESCA-dev parameters only)
    samples = state["visium_sections_done"]
    if samples and summary.get("chosen_parameters"):
        chosen = summary["chosen_parameters"]
        hover = visium_spot_table(samples, chosen)
        hover.to_csv(analysis / "visium_spot_table.csv.gz", index=False)
        table = pd.read_csv(VP.SPOT_TABLE)
        table = table[table["sample"].isin(samples)]
        adopted = table[table.classifier == "finetuned_tau05"]
        members = VP.tls_object_members(table[table.classifier == "original"])
        h = hover.assign(**{"cover_HoVer-Net inflammatory density": hover.cover_infl_density})
        arms = {ADOPTED: VP.arm_frames(adopted, "B_density_at_spot", "cover_B density", members),
                NUCLEI: VP.arm_frames(adopted, "nuclei_density_at_spot", "cover_All nuclei (no classifier)", members),
                "HookNet-TLS": VP.arm_frames(adopted, "cover_HookNet-TLS", "cover_HookNet-TLS", members),
                DENSITY_ARM: VP.arm_frames(h, "infl_density_at_spot", "cover_infl_density", members),
                DBSCAN_ARM: VP.arm_frames(h, "cover_infl_dbscan", "cover_infl_dbscan", members)}
        pairs = [(ADOPTED, DENSITY_ARM), (ADOPTED, DBSCAN_ARM), (DENSITY_ARM, NUCLEI), (DENSITY_ARM, DBSCAN_ARM)]
        summary["Visium"] = VP.compare_arms(arms, pairs)
        summary["Visium"]["sections"] = samples
        summary["Visium"]["hovernet_type_shares"] = {
            s: visium_cells(s).type_name.value_counts(normalize=True).round(4).to_dict() for s in samples}
        summary["Visium_cell_level"] = cell_level(hover, adopted)
    if esca_ok:
        summary["ESCA_hovernet_type_shares_by_category"] = {
            c: pd.Series({k: sum(inputs[(b, w)]["counts"][k].sum() for b, w in zip(part.barcode, part.window_id)) for k in TYPES.values()}
                         ).pipe(lambda s: (s / s.sum()).round(4)).to_dict()
            for c, part in windows.groupby("category")}
    summary["seconds"] = time.time() - started
    save_json(summary, out / "summary.json")
    print("wrote", out / "summary.json", f"({summary['seconds']:.0f}s)", flush=True)


def check_dbscan():
    from sklearn.cluster import DBSCAN
    rng = np.random.default_rng(0)
    worst = 0.0
    for trial in range(20):
        centres = rng.uniform(0, 2000, (rng.integers(1, 8), 2))
        xy = np.concatenate([rng.normal(c, rng.uniform(20, 80), (rng.integers(50, 600), 2)) for c in centres]
                            + [rng.uniform(0, 2000, (400, 2))])
        for min_samples in [20, 50, 100]:
            ours = dbscan(xy, 30.0, min_samples)
            ref = DBSCAN(eps=30.0, min_samples=min_samples).fit_predict(xy)
            assert ((ours >= 0) == (ref >= 0)).all(), "noise sets differ"
            core = DBSCAN(eps=30.0, min_samples=min_samples).fit(xy).core_sample_indices_
            # same partition of core points
            a, b = ours[core], ref[core]
            assert len(set(zip(a, b))) == len(set(a)) == len(set(b)), "core partitions differ"
            worst = max(worst, float((pd.crosstab(ours, ref).max(axis=1).sum()) / len(xy)))
    print("scipy DBSCAN matches sklearn: identical noise sets and core partitions over 60 random cases")


if __name__ == "__main__":
    main()
