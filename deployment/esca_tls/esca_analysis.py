"""Evaluate HE2CELL cell-based TLS detection on TCGA-ESCA windows.

Reference: HookNet-TLS published-filter TLS polygons (an independent H&E pixel model), minus
polygons on ink/folds. Detector: connected regions of smoothed predicted-B density above a
threshold, with minimum area and B-cell count. Baselines use identical machinery on
(i) all segmented nuclei (no classifier) and (ii) B+T lymphocytes. All detector parameters are
chosen on dev patients only; test patients are scored once with frozen parameters.
"""
from __future__ import annotations

import hashlib
import os
import itertools
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage, stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from esca_common import OUT, save_json  # noqa: E402
from esca_qc import slide_qc  # noqa: E402

GRID_UM = 8.0
MIN_REF_UM2 = 5000.0      # HookNet objects smaller than this inside a window are edge slivers
MATCH_FRACTION = 0.10     # overlap / smaller-object area needed for a match
TWO_STAGE = "Two-stage (nuclei propose, composition accepts)"
SCORES = {"B (HE2CELL)": ["B"], "B+T (HE2CELL)": ["B", "T"], "All nuclei (no classifier)": None, TWO_STAGE: "two-stage"}
PARAM_GRID = {
    "B (HE2CELL)": {"sigma_um": [15, 25], "threshold": [300, 500, 800, 1200, 1800, 2500, 3500],
                    "min_area_um2": [5000, 10000, 20000], "min_cells": [20, 50]},
    "B+T (HE2CELL)": {"sigma_um": [15, 25], "threshold": [1000, 1500, 2000, 3000, 4000, 5000, 6500],
                      "min_area_um2": [5000, 10000, 20000], "min_cells": [20, 50]},
    "All nuclei (no classifier)": {"sigma_um": [15, 25], "threshold": [4000, 5000, 6000, 7000, 8000, 9500, 11000],
                                   "min_area_um2": [5000, 10000, 20000], "min_cells": [20, 50]},
    # Stage 1 proposes aggregates from all nuclei (no classifier); stage 2 keeps a proposal only if the
    # classifier's B share inside it reaches b_share and, when require_b_core is on, predicted B is
    # more concentrated in the core than in the periphery. Tuned jointly on dev patients like the others.
    # Stage 1 is fixed to the all-nuclei detector's own dev-tuned parameters (tuning stage 1 for the
    # two-stage F1 favours low thresholds whose proposals fill whole windows and match any TLS trivially).
    TWO_STAGE: {"b_share": [0.03, 0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30], "require_b_core": [False, True]},
}
DEV_FRACTION = 0.4
RESULT = os.environ.get("ESCA_RESULT", "he2cell")  # prediction folder; windows always live in he2cell/


def is_dev(case_id):
    return int(hashlib.sha256(("esca-split-" + case_id).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < DEV_FRACTION


def rasterize_polygons(features, x0, y0, n, ratio, mpp):
    """Polygons (0.5 µm px) -> window grid of GRID_UM cells starting at level-0 (x0, y0)."""
    canvas = np.zeros((n, n), np.uint8)
    scale = ratio * mpp / GRID_UM
    for feature in features:
        for ring_index, ring in enumerate(feature["geometry"]["coordinates"]):
            pts = np.asarray(ring, float)
            pts = np.stack([pts[:, 0] * scale - x0 * mpp / GRID_UM, pts[:, 1] * scale - y0 * mpp / GRID_UM], 1)
            cv2.fillPoly(canvas, [np.round(pts).astype(np.int32)], 0 if ring_index else 1)
    return canvas.astype(bool)


def window_inputs(slide_folder, window, summary, maps, features, qc, cells):
    mpp = summary["source_mpp"]
    ratio = summary["source_dimensions"][0] / summary["dimensions_at_0p5um"][0]
    n = int(np.ceil((window.x1 - window.x0) * mpp / GRID_UM))
    to8 = 1 / (16 * ratio)
    yy = np.clip(((window.y0 + (np.arange(n) + .5) * GRID_UM / mpp) * to8).astype(int), 0, qc["valid"].shape[0] - 1)
    xx = np.clip(((window.x0 + (np.arange(n) + .5) * GRID_UM / mpp) * to8).astype(int), 0, qc["valid"].shape[1] - 1)
    valid = qc["valid"][np.ix_(yy, xx)]
    to4 = 1 / (8 * ratio)
    y4 = np.clip(((window.y0 + (np.arange(n) + .5) * GRID_UM / mpp) * to4).astype(int), 0, maps["prediction_mask"].shape[0] - 1)
    x4 = np.clip(((window.x0 + (np.arange(n) + .5) * GRID_UM / mpp) * to4).astype(int), 0, maps["prediction_mask"].shape[1] - 1)
    hooknet_classes = maps["prediction_mask"][np.ix_(y4, x4)]
    raw_tls = np.isin(hooknet_classes, [2, 3])
    gc = hooknet_classes == 3  # HookNet germinal-centre class pixels (unfiltered); maturity reference only
    reference = rasterize_polygons(features, window.x0, window.y0, n, ratio, mpp)
    c = cells[cells.window_id == window.window_id]
    gx = ((c.centroid_x_global - window.x0) * mpp / GRID_UM).astype(int).clip(0, n - 1).to_numpy()
    gy = ((c.centroid_y_global - window.y0) * mpp / GRID_UM).astype(int).clip(0, n - 1).to_numpy()
    counts = {}
    for name in ["B", "T", "Other", "Macrophage"]:
        m = (c.celltype == name).to_numpy()
        counts[name] = np.bincount(gy[m] * n + gx[m], minlength=n * n).reshape(n, n).astype(float)
    return {"n": n, "mpp": mpp, "valid": valid, "reference": reference, "raw_tls": raw_tls, "gc": gc, "counts": counts,
            "cells_xy_um": np.stack([(c.centroid_x_global - window.x0).to_numpy() * mpp,
                                     (c.centroid_y_global - window.y0).to_numpy() * mpp], 1),
            "celltype": c.celltype.to_numpy()}


def density(counts, types, sigma_um):
    keys = types or list(counts)
    total = sum(counts[k] for k in keys)
    return ndimage.gaussian_filter(total, sigma_um / GRID_UM, mode="constant") / (GRID_UM ** 2 / 1e6)


def detect(inputs, types, sigma_um, threshold, min_area_um2, min_cells):
    d = density(inputs["counts"], types, sigma_um)
    labels, count = ndimage.label((d >= threshold) & inputs["valid"])
    keys = types or list(inputs["counts"])
    raw = sum(inputs["counts"][k] for k in keys)
    keep = np.zeros_like(labels, bool)
    for index, slc in enumerate(ndimage.find_objects(labels), 1):
        region = labels[slc] == index
        if region.sum() * GRID_UM ** 2 >= min_area_um2 and raw[slc][region].sum() >= min_cells:
            keep[slc] |= region
    return keep, d


RIM_UM = 32.0  # periphery = outer half of the aggregate plus this ring of surrounding tissue
_PROPOSALS = {}


def proposal_features(inputs, sigma_um, threshold, min_area_um2, min_cells):
    """Stage 1: nuclear-density aggregates (no classifier) with the composition features stage 2 uses.

    Core = the inner half of each aggregate by distance to its edge; periphery = the outer half plus a
    RIM_UM ring of valid tissue around it. Shares are cell counts from the classifier's labels.
    Cached per window and stage-1 parameters, because stage-2 tuning reuses the same proposals.
    """
    key = (id(inputs), sigma_um, threshold, min_area_um2, min_cells)
    if key in _PROPOSALS:
        return _PROPOSALS[key]
    proposals, _ = detect(inputs, None, sigma_um, threshold, min_area_um2, min_cells)
    counts, total = inputs["counts"], sum(inputs["counts"].values())
    references = objects(inputs["reference"], MIN_REF_UM2)
    ring = int(round(RIM_UM / GRID_UM))
    out = []
    for obj in objects(proposals):
        distance = ndimage.distance_transform_edt(obj)
        core = obj & (distance >= .5 * distance.max())
        periphery = (obj & ~core) | (ndimage.binary_dilation(obj, iterations=ring) & ~obj & inputs["valid"])

        def share(kind, mask, among=None):
            denominator = sum(counts[k][mask].sum() for k in among) if among else total[mask].sum()
            return counts[kind][mask].sum() / denominator if denominator else np.nan
        b_core, b_periphery = share("B", core), share("B", periphery)
        t_core, t_periphery = share("T", core, ["B", "T"]), share("T", periphery, ["B", "T"])
        out.append({"mask": obj, "area_um2": obj.sum() * GRID_UM ** 2, "cells": int(total[obj].sum()),
                    "B_share": share("B", obj), "T_share": share("T", obj),
                    "B_core": b_core, "B_periphery": b_periphery, "B_core_enrichment": b_core - b_periphery,
                    "T_of_lymphocytes_core": t_core, "T_of_lymphocytes_periphery": t_periphery,
                    "T_rim_enrichment": t_periphery - t_core,
                    "HookNet_GC_pixel_fraction": float(inputs["gc"][obj].mean()),
                    "matches_HookNet": any((obj & r).sum() >= MATCH_FRACTION * min(obj.sum(), r.sum()) for r in references)})
    unproposed = sum(not any((r & p["mask"]).sum() >= MATCH_FRACTION * min(r.sum(), p["mask"].sum()) for p in out)
                     for r in references)
    _PROPOSALS[key] = (out, unproposed)
    return _PROPOSALS[key]


def accept(proposal, b_share, require_b_core):
    """Stage 2 decision and, when rejected, the reason."""
    if not proposal["B_share"] >= b_share:
        return False, "B share below cut"
    if require_b_core and not proposal["B_core_enrichment"] > 0:
        return False, "B not concentrated in core"
    return True, ""


def detect_two_stage(inputs, sigma_um, threshold, min_area_um2, min_cells, b_share, require_b_core=False):
    """Nuclear-density proposals accepted only when the classifier's composition supports TLS.

    Returns the accepted mask, the window score (highest B share among proposals; 0 with none),
    and the number of proposals rejected by composition.
    """
    proposals, _ = proposal_features(inputs, sigma_um, threshold, min_area_um2, min_cells)
    keep = np.zeros((inputs["n"], inputs["n"]), bool)
    best, rejected = 0.0, 0
    for proposal in proposals:
        best = max(best, np.nan_to_num(proposal["B_share"]))
        if accept(proposal, b_share, require_b_core)[0]:
            keep |= proposal["mask"]
        else:
            rejected += 1
    return keep, best, rejected


def two_stage_table(windows, inputs, params):
    """One row per stage-1 aggregate, with its stage-2 outcome and its relation to HookNet-TLS."""
    stage1 = {k: params[k] for k in ["sigma_um", "threshold", "min_area_um2", "min_cells"]}
    rows, misses = [], []
    for w in windows.itertuples():
        x = inputs[(w.barcode, w.window_id)]
        proposals, unproposed = proposal_features(x, **stage1)
        misses.append({"barcode": w.barcode, "case_id": w.case_id, "window_id": w.window_id, "split": w.split,
                       "HookNet_TLS_not_proposed": unproposed})
        for index, proposal in enumerate(proposals, 1):
            accepted, reason = accept(proposal, params["b_share"], params.get("require_b_core", False))
            outcome = ("TLS, HookNet agrees" if accepted and proposal["matches_HookNet"] else
                       "TLS, HE2CELL only" if accepted else
                       "aggregate found, composition rejected; HookNet calls TLS" if proposal["matches_HookNet"] else
                       "aggregate found, composition rejected; HookNet agrees not TLS")
            rows.append({"barcode": w.barcode, "case_id": w.case_id, "window_id": w.window_id, "split": w.split,
                         "category": w.category, "proposal": index, "accepted": accepted, "rejection_reason": reason,
                         "outcome": outcome, **{k: v for k, v in proposal.items() if k != "mask"}})
    return pd.DataFrame(rows), pd.DataFrame(misses)


def objects(mask, min_um2=0):
    labels, count = ndimage.label(mask)
    return [labels == i for i in range(1, count + 1) if (labels == i).sum() * GRID_UM ** 2 >= min_um2]


def match_counts(predicted, reference, lenient):
    """Object-level TP/FP/FN within one window."""
    pred = objects(predicted)
    ref = objects(reference, MIN_REF_UM2)
    def overlaps(a, b):
        return (a & b).sum() >= MATCH_FRACTION * min(a.sum(), b.sum())
    tp_ref = sum(any(overlaps(r, p) for p in pred) for r in ref)
    tp_pred = sum(any(overlaps(p, r) for r in ref) for p in pred)
    tp_pred_lenient = sum((p & lenient).sum() >= MATCH_FRACTION * p.sum() for p in pred)
    return {"n_ref": len(ref), "n_pred": len(pred), "tp_ref": tp_ref, "tp_pred": tp_pred, "tp_pred_lenient": tp_pred_lenient}


def prf(frame):
    recall = frame.tp_ref.sum() / max(frame.n_ref.sum(), 1)
    precision = frame.tp_pred.sum() / max(frame.n_pred.sum(), 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def auc(labels, scores):
    labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
    if labels.all() or (~labels).all():
        return np.nan
    return stats.mannwhitneyu(scores[labels], scores[~labels]).statistic / (labels.sum() * (~labels).sum())


def bootstrap(frame, function, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    cases = frame.case_id.unique()
    groups = {c: frame[frame.case_id == c] for c in cases}
    values = []
    for _ in range(n):
        sample = pd.concat([groups[c] for c in rng.choice(cases, len(cases))])
        values.append(function(sample))
    values = np.asarray(values, float)
    return np.nanpercentile(values, [2.5, 97.5]).tolist()


def load_cohort():
    clinical = pd.read_csv(OUT / "rna" / "clinical.csv").set_index("case_id")
    windows, inputs = [], {}
    for folder in sorted((OUT / "slides").glob(f"*/{RESULT}")):
        if not (folder / "cells.csv.gz").exists():
            continue
        slide_folder = folder.parent
        barcode = slide_folder.name
        summary = json.loads((slide_folder / "hooknet" / "summary.json").read_text())
        maps = np.load(slide_folder / "hooknet" / "maps_4um.npz")
        maps = {k: maps[k] for k in ["prediction_mask"]}
        selection = json.loads((slide_folder / "he2cell" / "window_selection.json").read_text())
        flagged = set(selection["artifact_flagged_hooknet_tls"])
        features = [f for f in json.loads((slide_folder / "hooknet" / "native_polygons.geojson").read_text())["features"]
                    if f["properties"]["predicted_label"] == "tls" and f["properties"]["passes_published_filter"]
                    and f["properties"]["region_id"] not in flagged]
        qc = slide_qc(slide_folder)
        cells = pd.read_csv(folder / "cells.csv.gz")
        for w in pd.read_csv(slide_folder / "he2cell" / "windows.csv").itertuples():
            key = (barcode, w.window_id)
            inputs[key] = window_inputs(slide_folder, w, summary, maps, features, qc, cells)
            x = inputs[key]
            ref_area = x["reference"].sum() * GRID_UM ** 2
            windows.append({"barcode": barcode, "case_id": barcode[:12], "window_id": w.window_id, "category": w.category,
                            "histology": clinical.histology.get(barcode[:12], "Other"),
                            "split": "dev" if is_dev(barcode[:12]) else "test",
                            "valid_mm2": x["valid"].sum() * GRID_UM ** 2 / 1e6, "reference_um2": ref_area,
                            "hooknet_label": "TLS" if ref_area >= MIN_REF_UM2 else ("ambiguous" if ref_area > 0 else "none"),
                            **{f"n_{k}": int(v.sum()) for k, v in x["counts"].items()}})
    return pd.DataFrame(windows), inputs


def evaluate(windows, inputs, method, params):
    types = SCORES[method]
    rows = []
    for w in windows.itertuples():
        x = inputs[(w.barcode, w.window_id)]
        rejected = 0
        if types == "two-stage":
            predicted, score, rejected = detect_two_stage(x, **params)
        else:
            predicted, d = detect(x, types, **params)
            score = float((d * x["valid"]).max())
        # max_density is the window score used for AUROC: peak density, or for the two-stage method
        # the highest B share among nuclear-density proposals.
        rows.append({"barcode": w.barcode, "case_id": w.case_id, "window_id": w.window_id,
                     "max_density": score, "proposals_rejected": rejected, "predicted_um2": predicted.sum() * GRID_UM ** 2,
                     "dice": 2 * (predicted & x["reference"]).sum() / max(predicted.sum() + x["reference"].sum(), 1),
                     **match_counts(predicted, x["reference"], x["raw_tls"])})
    return pd.DataFrame(rows)


def tune(windows, inputs):
    dev = windows[windows.split == "dev"]
    chosen, grid_rows = {}, []
    for method, grid in PARAM_GRID.items():
        best = None
        for values in itertools.product(*grid.values()):
            params = dict(zip(grid, values))
            if method == TWO_STAGE:
                params = {**chosen["All nuclei (no classifier)"], **params}
            p, r, f1 = prf(evaluate(dev, inputs, method, params))
            grid_rows.append({"method": method, **params, "dev_precision": p, "dev_recall": r, "dev_F1": f1})
            if best is None or f1 > best[0]:
                best = (f1, params)
        chosen[method] = best[1]
    return chosen, pd.DataFrame(grid_rows)


def two_stage_breakdown(windows, inputs, params, analysis):
    """Outcome categories of every stage-1 aggregate on test patients, plus a germinal-centre proxy check."""
    table, misses = two_stage_table(windows, inputs, params)
    table.to_csv(analysis / "two_stage_proposals.csv", index=False)
    misses.to_csv(analysis / "two_stage_unproposed_HookNet_TLS.csv", index=False)
    test, test_misses = table[table.split == "test"], misses[misses.split == "test"]
    result = {"parameters": params, "test_aggregates": len(test),
              "outcomes": test.outcome.value_counts().to_dict(),
              "rejection_reasons": test[~test.accepted].rejection_reason.value_counts().to_dict(),
              "HookNet_TLS_not_proposed_by_stage1": int(test_misses.HookNet_TLS_not_proposed.sum())}
    features = ["B_share", "B_core_enrichment", "T_rim_enrichment", "area_um2"]
    # Accepted aggregates that HookNet agrees with versus HookNet TLS the composition rule rejected.
    agreed, rejected = test[test.outcome == "TLS, HookNet agrees"], test[test.outcome.str.contains("HookNet calls TLS")]
    result["HookNet_TLS_accepted_vs_rejected_median"] = {f: [float(agreed[f].median()), float(rejected[f].median())]
                                                         for f in features if len(agreed) and len(rejected)}
    accepted = test[test.accepted].copy()
    accepted["has_GC"] = accepted.HookNet_GC_pixel_fraction >= .01
    maturity = {"accepted_aggregates": len(accepted), "with_HookNet_GC_pixels": int(accepted.has_GC.sum())}
    if accepted.has_GC.sum() >= 5 and (~accepted.has_GC).sum() >= 5:
        for f in features:
            a, b = accepted.loc[accepted.has_GC, f].dropna(), accepted.loc[~accepted.has_GC, f].dropna()
            maturity[f] = {"median_with_GC": float(a.median()), "median_without_GC": float(b.median()),
                           "mann_whitney_p": float(stats.mannwhitneyu(a, b).pvalue),
                           "AUROC_for_GC": auc(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b])}
    result["maturity_proxy"] = maturity
    return result


def main():
    analysis = OUT / ("analysis" if RESULT == "he2cell" else f"analysis_{RESULT}")
    analysis.mkdir(parents=True, exist_ok=True)
    windows, inputs = load_cohort()
    windows.to_csv(analysis / "windows.csv", index=False)
    chosen, grid = tune(windows, inputs)
    grid.to_csv(analysis / "dev_parameter_grid.csv", index=False)
    results, per_window = {}, []
    for method, params in chosen.items():
        frame = evaluate(windows, inputs, method, params).merge(windows, on=["barcode", "case_id", "window_id"])
        frame["method"] = method
        per_window.append(frame)
        for split in ["dev", "test"]:
            part = frame[frame.split == split]
            labelled = part[part.hooknet_label != "ambiguous"]
            y = labelled.hooknet_label == "TLS"
            hard = labelled[labelled.category.isin(["tls", "dense"])]
            p, r, f1 = prf(part)
            entry = {"parameters": params, "windows": len(part), "patients": part.case_id.nunique(),
                     "object_precision": p, "object_recall": r, "object_F1": f1,
                     "precision_lenient_any_HookNet_TLS_pixel": part.tp_pred_lenient.sum() / max(part.n_pred.sum(), 1),
                     "window_AUROC_max_density": auc(y, labelled.max_density),
                     "window_AUROC_vs_dense_hard_negatives": auc(hard.hooknet_label == "TLS", hard.max_density),
                     "median_dice_in_HookNet_TLS_windows": float(part[part.hooknet_label == "TLS"].dice.median()),
                     "proposals_rejected_by_composition": int(part.proposals_rejected.sum()),
                     "rejected_proposals_in_HookNet_negative_windows": int(part[part.hooknet_label == "none"].proposals_rejected.sum())}
            if split == "test":
                entry["CI95_object_F1"] = bootstrap(part, lambda s: prf(s)[2])
                entry["CI95_object_recall"] = bootstrap(part, lambda s: prf(s)[1])
                entry["CI95_object_precision"] = bootstrap(part, lambda s: prf(s)[0])
                entry["CI95_window_AUROC"] = bootstrap(labelled, lambda s: auc(s.hooknet_label == "TLS", s.max_density))
                entry["CI95_AUROC_vs_dense"] = bootstrap(hard, lambda s: auc(s.hooknet_label == "TLS", s.max_density))
            results.setdefault(method, {})[split] = entry
    pd.concat(per_window).to_csv(analysis / "per_window_results.csv", index=False)
    breakdown = two_stage_breakdown(windows, inputs, chosen[TWO_STAGE], analysis)
    save_json({"design": __doc__, "two_stage_breakdown": breakdown, "grid_um": GRID_UM, "min_reference_um2": MIN_REF_UM2, "match_fraction": MATCH_FRACTION,
               "dev_fraction": DEV_FRACTION, "results": results,
               "cohort": {"slides": int(windows.barcode.nunique()), "windows": len(windows),
                          "windows_by_category": windows.category.value_counts().to_dict(),
                          "hooknet_label_by_category": pd.crosstab(windows.category, windows.hooknet_label).to_dict()}},
              analysis / "detection_results.json")
    print(json.dumps({m: {s: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items() if k != "parameters"}
                          for s, e in r.items()} for m, r in results.items()}, indent=1))
    return windows, inputs, chosen


if __name__ == "__main__":
    main()
