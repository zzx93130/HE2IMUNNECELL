"""Re-run the HE2CELL window path with Macenko normalisation on dev-patient windows (and LUAD A1/A2),
then compare against the existing Reinhard results. Test patients are never touched here."""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import esca_analysis as A  # noqa: E402
import esca_diagnose as D  # noqa: E402
import esca_stain as M  # noqa: E402
import esca_windows as W  # noqa: E402
from esca_common import OUT, save_json  # noqa: E402
from esca_qc import slide_qc  # noqa: E402

DEST = OUT / "stain_comparison"


def dev_slides():
    return [p.parents[1].name for p in sorted((OUT / "slides").glob("*/he2cell/cells.csv.gz")) if A.is_dev(p.parents[1].name[:12])]


def run_macenko(reference):
    for barcode in dev_slides():
        out = DEST / "macenko" / barcode / "cells.csv.gz"
        if out.exists():
            continue
        folder = OUT / "slides" / barcode
        slide = Path(json.loads((folder / "hooknet" / "summary.json").read_text())["slide"])
        source = M.slide_stains(slide, folder / "hooknet")
        save_json({"H_E_stain_matrix": source[0].tolist(), "max_concentrations": source[1].tolist()},
                  DEST / "macenko" / barcode / "stains.json")
        windows = pd.read_csv(folder / "he2cell" / "windows.csv")
        D.DIAG = DEST
        D.run(slide, windows, f"macenko/{barcode}", lambda p, s=source: M.normalise(p, s, reference))
        print(f"{barcode}: Macenko windows done", flush=True)


def window_frame(method):
    rows, inputs = [], {}
    for barcode in dev_slides():
        cells_path = (OUT / "slides" / barcode / "he2cell" / "cells.csv.gz" if method == "Reinhard"
                      else DEST / "macenko" / barcode / "cells.csv.gz")
        if not cells_path.exists():
            continue
        slide_folder = OUT / "slides" / barcode
        summary = json.loads((slide_folder / "hooknet" / "summary.json").read_text())
        maps = {"prediction_mask": np.load(slide_folder / "hooknet" / "maps_4um.npz")["prediction_mask"]}
        flagged = set(json.loads((slide_folder / "he2cell" / "window_selection.json").read_text())["artifact_flagged_hooknet_tls"])
        features = [f for f in json.loads((slide_folder / "hooknet" / "native_polygons.geojson").read_text())["features"]
                    if f["properties"]["predicted_label"] == "tls" and f["properties"]["passes_published_filter"]
                    and f["properties"]["region_id"] not in flagged]
        cells = pd.read_csv(cells_path)
        qc = slide_qc(slide_folder)
        for w in pd.read_csv(slide_folder / "he2cell" / "windows.csv").itertuples():
            x = A.window_inputs(slide_folder, w, summary, maps, features, qc, cells)
            inputs[(barcode, w.window_id)] = x
            gi = (x["cells_xy_um"] / A.GRID_UM).astype(int).clip(0, x["n"] - 1)
            in_ref = x["reference"][gi[:, 1], gi[:, 0]]
            ref_area = x["reference"].sum() * A.GRID_UM ** 2
            rows.append({"barcode": barcode, "case_id": barcode[:12], "window_id": w.window_id, "category": w.category,
                         "hooknet_label": "TLS" if ref_area >= A.MIN_REF_UM2 else ("ambiguous" if ref_area > 0 else "none"),
                         "valid_mm2": x["valid"].sum() * A.GRID_UM ** 2 / 1e6,
                         "cells_in_HookNet_TLS": int(in_ref.sum()),
                         "B_in_HookNet_TLS": int(((x["celltype"] == "B") & in_ref).sum()),
                         "T_in_HookNet_TLS": int(((x["celltype"] == "T") & in_ref).sum()),
                         **{f"n_{k}": int(v.sum()) for k, v in x["counts"].items()}})
    return pd.DataFrame(rows), inputs


def summarise(method):
    windows, inputs = window_frame(method)
    grid = A.PARAM_GRID["B (HE2CELL)"]
    best = None
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid, values))
        result = A.evaluate(windows, inputs, "B (HE2CELL)", params)
        p, r, f1 = A.prf(result)
        if best is None or f1 > best[0]:
            best = (f1, p, r, params, result)
    f1, p, r, params, result = best
    merged = result.merge(windows, on=["barcode", "case_id", "window_id"])
    labelled = merged[merged.hooknet_label != "ambiguous"]
    hard = labelled[labelled.category.isin(["tls", "dense"])]
    tls_cells = windows.cells_in_HookNet_TLS.sum()
    dense = windows[windows.category == "dense"]
    return windows, {
        "windows": len(windows), "slides": int(windows.barcode.nunique()),
        "B_share_of_cells_inside_HookNet_TLS": float(windows.B_in_HookNet_TLS.sum() / max(tls_cells, 1)),
        "T_share_of_cells_inside_HookNet_TLS": float(windows.T_in_HookNet_TLS.sum() / max(tls_cells, 1)),
        "B_per_mm2_dense_hard_negatives": float(dense.n_B.sum() / dense.valid_mm2.sum()),
        "B_per_mm2_random_no_TLS": float(windows[(windows.category == "random") & (windows.hooknet_label == "none")].n_B.sum()
                                         / windows[(windows.category == "random") & (windows.hooknet_label == "none")].valid_mm2.sum()),
        "best_dev_params": params, "object_precision": p, "object_recall": r, "object_F1": f1,
        "detections_in_dense_windows": int(merged[merged.category == "dense"].n_pred.sum()),
        "window_AUROC_peak_B": A.auc(labelled.hooknet_label == "TLS", labelled.max_density),
        "AUROC_vs_dense": A.auc(hard.hooknet_label == "TLS", hard.max_density),
    }


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    reference = M.slide_stains(W.REFERENCE_SLIDE, W.REFERENCE_QC)
    save_json({"H_E_stain_matrix": reference[0].tolist(), "max_concentrations": reference[1].tolist(),
               "reference_slide": str(W.REFERENCE_SLIDE)}, DEST / "macenko_reference.json")
    # LUAD A1/A2 invariance: Macenko to its own reference should leave predictions essentially unchanged.
    if not (DEST / "LUAD_macenko" / "cells.csv.gz").exists():
        candidates = pd.read_csv(W.ROOT / "outputs" / "whole-slide-20261012" / "tls_analysis" / "ranked_candidate_regions.csv")
        luad = pd.DataFrame([{"window_id": r.candidate_id, "x0": int(r.x_level0_px - W.CORE / 2), "y0": int(r.y_level0_px - W.CORE / 2)}
                             for r in candidates.head(2).itertuples()])
        D.DIAG = DEST
        D.run(W.REFERENCE_SLIDE, luad, "LUAD_macenko", lambda p: M.normalise(p, reference, reference))
    run_macenko(reference)
    report = {}
    luad = pd.read_csv(DEST / "LUAD_macenko" / "cells.csv.gz")
    diag = json.loads((OUT / "diagnostics" / "diagnostics.json").read_text())
    for window_id in ["A1", "A2"]:
        report[f"LUAD {window_id}"] = {"window_path_raw": diag[f"LUAD {window_id}"]["window_path"],
                                       "macenko_self": luad[luad.window_id == window_id].celltype.value_counts().to_dict()}
    tables = []
    for method in ["Reinhard", "Macenko"]:
        windows, report[method] = summarise(method)
        windows["method"] = method
        tables.append(windows)
    pd.concat(tables).to_csv(DEST / "dev_window_counts.csv", index=False)
    save_json(report, DEST / "comparison.json")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
