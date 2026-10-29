"""Raw (un-normalised) ESCA cell crops from DEV patients only, for checkpoint selection.

For each dev slide with HE2CELL windows, re-segment its tls and dense windows from the original
slide and keep the 64-px RGB/mask crops. Each cell is labelled by whether its centroid lies inside
an artifact-free HookNet-TLS polygon. Test patients are never touched.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import esca_analysis as A  # noqa: E402
import esca_windows as W  # noqa: E402
from esca_common import OUT  # noqa: E402
from slide_io import open_slide  # noqa: E402

BANK = OUT / "cropbank_dev"


def main(max_slides=12):
    BANK.mkdir(parents=True, exist_ok=True)
    folders = [f for f in sorted((OUT / "slides").glob("*")) if (f / "he2cell" / "cells.csv.gz").exists() and A.is_dev(f.name[:12])]
    tables = []
    for folder in folders[:max_slides]:
        barcode = folder.name
        if (BANK / barcode / "cells.csv").exists():
            tables.append(pd.read_csv(BANK / barcode / "cells.csv"))
            continue
        summary = json.loads((folder / "hooknet" / "summary.json").read_text())
        slide = Path(summary["slide"])
        windows = pd.read_csv(folder / "he2cell" / "windows.csv")
        windows = windows[windows.category.isin(["tls", "dense"])]
        work = BANK / barcode
        shutil.rmtree(work, ignore_errors=True)
        (work / "patches").mkdir(parents=True)
        rows = []
        with open_slide(slide) as handle:
            for w in windows.itertuples():
                for gy in range(W.GRID):
                    for gx in range(W.GRID):
                        x, y = w.x0 - W.CORE_MARGIN + gx * W.STRIDE, w.y0 - W.CORE_MARGIN + gy * W.STRIDE
                        path = work / "patches" / f"{w.window_id}_patch{len(rows)}_x{x}_y{y}.png"
                        handle.read_region((x, y), 0, (W.PATCH, W.PATCH)).convert("RGB").save(path)
                        rows.append({"path": str(path), "num_patch": len(rows), "sample": slide.stem, "source_tif": str(slide),
                                     "x": x, "y": y, "patch_size": W.PATCH, "stride": W.STRIDE, "level": 0, "tissue_ratio": 1,
                                     "window_id": w.window_id})
        patches = pd.DataFrame(rows)
        patch_csv = work / f"{slide.stem}_patch_index.csv"
        patches.to_csv(patch_csv, index=False)
        W.stage("run_cellcrops.py", "--patch_csv", patch_csv, "--out_dir", work / "cells", "--model_name", "cpsam_v2",
                "--device", "mps", "--num_gpus", 1, "--batch_size", 16, "--crop_size", 64, log=work / "pipeline.log")
        cells = pd.read_csv(work / "cells" / "mask_tables" / f"{patch_csv.stem}_cellpose_mask_instances.csv")
        cells = cells.merge(patches[["num_patch", "window_id"]].rename(columns={"num_patch": "patch_id"}), on="patch_id")
        # Label each cell by HookNet-TLS membership on the analysis grid.
        selection = json.loads((folder / "he2cell" / "window_selection.json").read_text())
        flagged = set(selection["artifact_flagged_hooknet_tls"])
        features = [f for f in json.loads((folder / "hooknet" / "native_polygons.geojson").read_text())["features"]
                    if f["properties"]["predicted_label"] == "tls" and f["properties"]["passes_published_filter"]
                    and f["properties"]["region_id"] not in flagged]
        mpp = summary["source_mpp"]
        ratio = summary["source_dimensions"][0] / summary["dimensions_at_0p5um"][0]
        inside = np.zeros(len(cells), bool)
        for w in windows.itertuples():
            n = int(np.ceil((w.x1 - w.x0) * mpp / A.GRID_UM))
            ref = A.rasterize_polygons(features, w.x0, w.y0, n, ratio, mpp)
            m = (cells.window_id == w.window_id).to_numpy()
            gx = ((cells.centroid_x_global[m] - w.x0) * mpp / A.GRID_UM).astype(int).clip(0, n - 1)
            gy = ((cells.centroid_y_global[m] - w.y0) * mpp / A.GRID_UM).astype(int).clip(0, n - 1)
            inside[m] = ref[gy.to_numpy(), gx.to_numpy()]
        cells["in_hooknet_tls"] = inside
        cells["category"] = cells.window_id.str.split("_").str[0]
        cells["barcode"] = barcode
        keep = ["barcode", "window_id", "category", "in_hooknet_tls", "global_instance_id", "area_pixels",
                "centroid_x_global", "centroid_y_global", "cell_image_crop_path", "cell_mask_crop_path"]
        cells[keep].to_csv(work / "cells.csv", index=False)
        shutil.rmtree(work / "patches", ignore_errors=True)
        shutil.rmtree(work / "cells" / "patch_masks", ignore_errors=True)
        tables.append(cells[keep])
        print(f"{barcode}: {len(cells)} cells, {inside.sum()} in HookNet TLS", flush=True)
    bank = pd.concat(tables, ignore_index=True)
    bank.to_csv(BANK / "bank.csv", index=False)
    print(f"bank: {len(bank)} cells from {bank.barcode.nunique()} dev slides")


if __name__ == "__main__":
    main()
