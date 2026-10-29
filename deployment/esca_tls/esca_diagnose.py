"""Diagnose missing B calls: (1) window path on known LUAD aggregates vs the original whole-slide
pipeline; (2) ESCA TLS windows with and without Reinhard stain normalisation to the LUAD slide."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import esca_windows as W  # noqa: E402
from esca_common import CLASSIFIER, OUT, ROOT, save_json  # noqa: E402
from slide_io import open_slide  # noqa: E402

LUAD = ROOT / "data" / "TCGA-55-7725-01Z-00-DX1.4d678777-63b1-4f4a-932a-7fccabf504c7.svs"
LUAD_RUN = ROOT / "outputs" / "whole-slide-20261012" / "TCGA-55-7725-01Z-00-DX1.4d678777-63b1-4f4a-932a-7fccabf504c7" / "cell_types.csv"
DIAG = OUT / "diagnostics"


def lab_stats(slide_path, mask=None):
    """Mean/std in LAB over tissue pixels of a ~4 µm overview."""
    with open_slide(slide_path) as s:
        level = min(range(s.level_count), key=lambda l: abs(s.level_downsamples[l] * float(s.properties["openslide.mpp-x"]) - 4))
        image = np.asarray(s.read_region((0, 0), level, s.level_dimensions[level]).convert("RGB"))
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    tissue = (lab[..., 0] < 215) & (lab[..., 0] > 40) if mask is None else mask
    pixels = lab[tissue]
    return pixels.mean(0), pixels.std(0)


def reinhard(rgb, source, target):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    background = lab[..., 0] >= 215  # keep glass white
    out = (lab - source[0]) / source[1] * target[1] + target[0]
    out[background] = lab[background]
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def run(slide, windows, name, transform=None):
    work = DIAG / name / "_work"
    shutil.rmtree(work, ignore_errors=True)
    (work / "patches").mkdir(parents=True)
    rows = []
    with open_slide(slide) as handle:
        for w in windows.itertuples():
            for gy in range(W.GRID):
                for gx in range(W.GRID):
                    x, y = w.x0 - W.CORE_MARGIN + gx * W.STRIDE, w.y0 - W.CORE_MARGIN + gy * W.STRIDE
                    patch = np.asarray(handle.read_region((x, y), 0, (W.PATCH, W.PATCH)).convert("RGB"))
                    if transform is not None:
                        patch = transform(patch)
                    path = work / "patches" / f"{w.window_id}_patch{len(rows)}_x{x}_y{y}.png"
                    cv2.imwrite(str(path), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
                    rows.append({"path": str(path), "num_patch": len(rows), "sample": slide.stem, "source_tif": str(slide),
                                 "x": x, "y": y, "patch_size": W.PATCH, "stride": W.STRIDE, "level": 0, "tissue_ratio": 1,
                                 "window_id": w.window_id})
    table = pd.DataFrame(rows)
    patch_csv = work / f"{slide.stem}_patch_index.csv"
    table.to_csv(patch_csv, index=False)
    log = DIAG / name / "pipeline.log"
    W.stage("run_cellcrops.py", "--patch_csv", patch_csv, "--out_dir", work / "cells", "--model_name", "cpsam_v2",
            "--device", "mps", "--num_gpus", 1, "--batch_size", 16, "--crop_size", 64, log=log)
    result = work / "cell_types.csv"
    W.stage("model_eval.py", "--meta_path", work / "cells" / "mask_tables" / f"{patch_csv.stem}_cellpose_mask_instances.csv",
            "--save_path", CLASSIFIER, "--output_path", result, "--device", "mps", "--batch_size", 256,
            "--num_workers", 0, "--tta", "d4", log=log)
    cells = pd.read_csv(result)[W.KEEP].merge(table[["num_patch", "window_id"]].rename(columns={"num_patch": "patch_id"}), on="patch_id")
    cells.to_csv(DIAG / name / "cells.csv.gz", index=False)
    shutil.rmtree(work, ignore_errors=True)
    return cells


def main():
    DIAG.mkdir(parents=True, exist_ok=True)
    report = {}
    # (1) LUAD A1/A2 with the window path; compare to original predictions in the same cores.
    candidates = pd.read_csv(ROOT / "outputs" / "whole-slide-20261012" / "tls_analysis" / "ranked_candidate_regions.csv")
    luad_windows = pd.DataFrame([{"window_id": r.candidate_id, "x0": int(r.x_level0_px - W.CORE / 2), "y0": int(r.y_level0_px - W.CORE / 2)}
                                 for r in candidates.head(2).itertuples()])
    luad_windows["x1"], luad_windows["y1"] = luad_windows.x0 + W.CORE, luad_windows.y0 + W.CORE
    new = run(LUAD, luad_windows, "LUAD_window_path")
    original = pd.read_csv(LUAD_RUN, usecols=["centroid_x_global", "centroid_y_global", "celltype"])
    for w in luad_windows.itertuples():
        inside = original[(original.centroid_x_global >= w.x0) & (original.centroid_x_global < w.x1)
                          & (original.centroid_y_global >= w.y0) & (original.centroid_y_global < w.y1)]
        report[f"LUAD {w.window_id}"] = {"original_pipeline": inside.celltype.value_counts().to_dict(),
                                         "window_path": new[new.window_id == w.window_id].celltype.value_counts().to_dict()}
    # (2) ESCA HookNet-TLS windows: raw vs Reinhard-normalised to the LUAD slide.
    target = lab_stats(LUAD)
    report["LUAD_LAB_mean_std"] = [target[0].round(2).tolist(), target[1].round(2).tolist()]
    for barcode in ["TCGA-L5-A4OE-01Z-00-DX1", "TCGA-L5-A4OF-01Z-00-DX1"]:
        folder = OUT / "slides" / barcode
        slide = Path(json.loads((folder / "hooknet" / "summary.json").read_text())["slide"])
        windows = pd.read_csv(folder / "he2cell" / "windows.csv")
        windows = windows[windows.category == "tls"].head(2)
        source = lab_stats(slide)
        report[f"{barcode} LAB_mean_std"] = [source[0].round(2).tolist(), source[1].round(2).tolist()]
        normalised = run(slide, windows, f"{barcode}_reinhard", lambda p: reinhard(p, source, target))
        raw = pd.read_csv(folder / "he2cell" / "cells.csv.gz")
        for w in windows.itertuples():
            report[f"{barcode} {w.window_id}"] = {"raw": raw[raw.window_id == w.window_id].celltype.value_counts().to_dict(),
                                                  "reinhard_to_LUAD": normalised[normalised.window_id == w.window_id].celltype.value_counts().to_dict()}
    save_json(report, DIAG / "diagnostics.json")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
