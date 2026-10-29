"""HookNet-TLS failure mode on frozen-section ST H&E: fraction of tissue called TLS per array and a QC figure
(imagesLarge thumbnails with published-filter TLS in blue, pathologist Lymphoid nodules in yellow) for the arrays
at the 25th/50th/75th/90th percentile of that fraction. Writes analysis/hooknet_qc.{csv,json} and
qc/hooknet_failure_mode.jpg.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tnbc_common as C  # noqa: E402
import tnbc_grids as G  # noqa: E402


def main():
    rows = []
    for a in (C.OUT / "stage1_priority.txt").read_text().split():
        x = G.load(a, hovernet=False, hooknet=True)
        s = json.loads((C.OUT / "hooknet" / a / "summary.json").read_text())
        ln = (x["labels"] == G.LN) & x["tissue"]
        rows.append({"array": a, "case_id": x["case_id"], "filtered_TLS_polygons": s["native"]["filtered_TLS_polygons"],
                     "tissue_fraction_called_TLS": float((x["hooknet"]["tls"] & x["tissue"]).sum() / x["tissue"].sum()),
                     "raw_TLS_class_fraction": float((x["hooknet"]["raw_tls"] & x["tissue"]).sum() / x["tissue"].sum()),
                     "LN_fraction_of_tissue": float(ln.sum() / x["tissue"].sum()),
                     "LN_covered_by_HookNet": float((ln & x["hooknet"]["tls"]).sum() / max(ln.sum(), 1)) if ln.any() else np.nan})
    d = pd.DataFrame(rows)
    (C.OUT / "analysis").mkdir(exist_ok=True)
    d.to_csv(C.OUT / "analysis" / "hooknet_qc.csv", index=False)
    summary = {"arrays": len(d), "median_tissue_fraction_called_TLS": float(d.tissue_fraction_called_TLS.median()),
               "IQR": d.tissue_fraction_called_TLS.quantile([.25, .75]).tolist(),
               "median_LN_fraction_of_tissue": float(d.LN_fraction_of_tissue.median()),
               "filtered_TLS_polygons_total": int(d.filtered_TLS_polygons.sum())}
    C.save_json(summary, C.OUT / "analysis" / "hooknet_qc.json")
    inv = G.inventory()
    picks = [d["array"].iloc[(d.tissue_fraction_called_TLS - d.tissue_fraction_called_TLS.quantile(q)).abs().argmin()] for q in [.25, .5, .75, .9]]
    panels = []
    for a in picks:
        x = G.load(a, hovernet=False, hooknet=True)
        img = C.read_image(inv[a]["large"], 4)
        n = x["n"]
        size = 700
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        for mask, colour in [(x["hooknet"]["tls"], (40, 90, 255)), ((x["labels"] == G.LN), (255, 220, 0))]:
            m = cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, contours, -1, colour, 2)
        f = d[d["array"] == a].iloc[0]
        cv2.putText(img, f"{a} {f.case_id}: HookNet TLS {f.tissue_fraction_called_TLS:.0%} of tissue", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 2)
        panels.append(img)
    grid = np.concatenate([np.concatenate(panels[:2], 1), np.concatenate(panels[2:], 1)], 0)
    cv2.imwrite(str(C.OUT / "qc" / "hooknet_failure_mode.jpg"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(json.dumps(summary, indent=1), picks)


if __name__ == "__main__":
    main()
