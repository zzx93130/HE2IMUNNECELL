"""Sigma and threshold ablations on TCGA-ESCA (adopted cells: fine-tuned, D4, prior tau 0.5).

Sigma: for each sigma in 10/15/25/40 µm the existing esca_analysis grid is re-run on dev patients with
sigma fixed (threshold, minimum area, minimum cells and the two-stage B-share cut re-tuned), then the
frozen parameters score every window. Output per sigma: outputs/ablations_<date>/esca_sigma<s>/.
Baseline fairness: the adopted nuclei-only threshold (11000) is the top of its grid, so nuclei-only is
also re-tuned on dev with the threshold grid extended to 20000 (outputs/ablations_<date>/esca_nuclei_extended/).
Threshold: with every other adopted parameter frozen, object F1 / precision / recall and dense-window
specificity as a function of the density threshold (B density and nuclei-only), on dev and test.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys

os.environ["ESCA_RESULT"] = "he2cell_ft_tau05"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deployment" / "esca_tls"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import esca_analysis as A  # noqa: E402
from esca_common import save_json  # noqa: E402

OUT = ROOT / "outputs" / "ablations_20261025"
SIGMAS = [10, 15, 25, 40]
NUCLEI_EXTENDED = [4000, 5000, 6000, 7000, 8000, 9500, 11000, 12500, 14000, 16000, 18000, 20000]
METHODS = ["B (HE2CELL)", "All nuclei (no classifier)", A.TWO_STAGE]
THRESHOLDS = {"B (HE2CELL)": np.unique(np.round(np.geomspace(150, 5000, 22), -1)),
              "All nuclei (no classifier)": np.unique(np.round(np.geomspace(3000, 18000, 18), -2))}
FROZEN = json.loads((A.OUT / "analysis_he2cell_ft_tau05" / "report_summary.json").read_text())["chosen_parameters"]


def per_window(windows, inputs, chosen):
    frames = []
    for method, params in chosen.items():
        frame = A.evaluate(windows, inputs, method, params).merge(windows, on=["barcode", "case_id", "window_id"])
        frames.append(frame.assign(method=method))
    return pd.concat(frames, ignore_index=True)


def summary(frame):
    precision, recall, f1 = A.prf(frame)
    dense = frame[(frame.category == "dense") & (frame.hooknet_label == "none")]
    return {"object_precision": precision, "object_recall": recall, "object_F1": f1,
            "dense_specificity": float((dense.n_pred == 0).mean()), "predicted_objects": int(frame.n_pred.sum())}


def main():
    windows, inputs = A.load_cohort()
    grid0 = copy.deepcopy(A.PARAM_GRID)
    for sigma in SIGMAS:
        folder = OUT / f"esca_sigma{sigma}"
        if (folder / "per_window_results.csv").exists():
            continue
        folder.mkdir(parents=True, exist_ok=True)
        A.PARAM_GRID = {m: ({**grid0[m], "sigma_um": [sigma]} if "sigma_um" in grid0[m] else grid0[m]) for m in METHODS}
        chosen, grid = A.tune(windows, inputs)
        grid.to_csv(folder / "dev_parameter_grid.csv", index=False)
        per_window(windows, inputs, chosen).to_csv(folder / "per_window_results.csv", index=False)
        edges = {m: {k: p[k] in (grid0[m][k][0], grid0[m][k][-1]) for k in p if k in grid0[m] and len(grid0[m][k]) > 2}
                 for m, p in chosen.items()}
        save_json({"sigma_um": sigma, "chosen_parameters": chosen, "chosen_value_at_grid_edge": edges,
                   "source_cells": "he2cell_ft_tau05", "tuned_on": "dev patients only"}, folder / "chosen_parameters.json")
        print(sigma, json.dumps(chosen), flush=True)
    folder = OUT / "esca_nuclei_extended"
    if not (folder / "per_window_results.csv").exists():
        folder.mkdir(parents=True, exist_ok=True)
        method = "All nuclei (no classifier)"
        A.PARAM_GRID = {method: {**grid0[method], "threshold": NUCLEI_EXTENDED}}
        chosen, grid = A.tune(windows, inputs)
        grid.to_csv(folder / "dev_parameter_grid.csv", index=False)
        per_window(windows, inputs, chosen).to_csv(folder / "per_window_results.csv", index=False)
        save_json({"chosen_parameters": chosen, "threshold_grid": NUCLEI_EXTENDED, "tuned_on": "dev patients only"},
                  folder / "chosen_parameters.json")
        print("nuclei extended", json.dumps(chosen), flush=True)
    A.PARAM_GRID = grid0
    if (OUT / "esca_threshold_curve.csv").exists():
        return
    rows = []
    for method, thresholds in THRESHOLDS.items():
        for threshold in thresholds:
            params = {**FROZEN[method], "threshold": float(threshold)}
            frame = A.evaluate(windows, inputs, method, params).merge(windows, on=["barcode", "case_id", "window_id"])
            for split in ["dev", "test"]:
                rows.append({"method": method, "threshold": float(threshold), "split": split,
                             "frozen_threshold": FROZEN[method]["threshold"], **summary(frame[frame.split == split])})
        print(method, "threshold curve done", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "esca_threshold_curve.csv", index=False)


if __name__ == "__main__":
    main()
