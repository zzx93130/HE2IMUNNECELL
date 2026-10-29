"""Interim checkpoint (coordinator request after Gate A) on the arrays processed so far. Labelled INTERIM.

1. Per-class HE2CELL label composition (tnbc_diagnostic) vs FFPE references.
2. 200 µm-tile Spearman (stage-4b metric, early, on accepted registrations): T vs CD3, B vs CD20, nuclei vs CD3+CD20
   (CD2 slides excluded from CD3 terms), per-sample rho, mean with patient bootstrap CI.
3. Two-stage / B-density / nuclei-only / HookNet object counts vs Lymphoid-nodule objects (reference arrays), counts only.
4. GPU hours (cells + HookNet) from summaries.
Writes analysis/interim/interim.json.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tnbc_common as C  # noqa: E402
import tnbc_grids as G  # noqa: E402
import tnbc_eval_ihc as EI  # noqa: E402
import tnbc_eval_path as EP  # noqa: E402

OUT = C.OUT / "analysis" / "interim"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inv = G.inventory()
    done = [a for a, e in inv.items() if e["annotated"] and (C.OUT / "cells" / a / "cells.csv.gz").exists()]
    result = {"status": "INTERIM", "arrays_processed": done}
    # 2. IHC tiles at 200 µm
    rows = []
    reg_status = []
    for a in done:
        rp = C.OUT / "registration" / f"{a}.json"
        if not rp.exists():
            continue
        r = json.loads(rp.read_text())
        reg_status.append({"array": a, "accept": r["accept"], "median_residual_um": r["median_residual_um"]})
        if not r["accept"]:
            continue
        x = G.load(a, hovernet=False, hooknet=False)
        ihc = EI.mapped_ihc(a, x)
        overlap = x["tissue"] & ihc["tissue"]
        t = EI.tile_table(x, ihc, overlap, 200, None)
        t.insert(0, "case_id", x["case_id"])
        t.insert(0, "array", a)
        rows.append(t)
    tiles = pd.concat(rows) if rows else pd.DataFrame()
    tiles.to_csv(OUT / "tiles_200.csv.gz", index=False)
    ps = EI.per_sample(tiles) if len(tiles) else pd.DataFrame()
    ps.to_csv(OUT / "per_sample_200.csv", index=False)
    if len(ps):
        no_cd2 = ps[~ps.CD2_slide]
        result["ihc_200um_interim"] = {
            "arrays_registered_accepted": int(len(ps)), "registration": reg_status,
            "rho_T_CD3": EI.mean_ci(no_cd2, "rho_T_CD3"), "rho_T_CD3_incl_CD2_as_CD3": EI.mean_ci(ps, "rho_T_CD3"),
            "rho_B_CD20": EI.mean_ci(ps, "rho_B_CD20"),
            "rho_nuclei_CD3plusCD20": EI.mean_ci(no_cd2, "rho_nuclei_total"),
            "rho_TB_CD3plusCD20": EI.mean_ci(no_cd2, "rho_TB_total"),
            "rho_B_CD3": EI.mean_ci(no_cd2, "rho_B_CD3"), "rho_T_CD20": EI.mean_ci(no_cd2, "rho_T_CD20"),
            "disc_B": EI.mean_ci(no_cd2, "disc_B"), "disc_T": EI.mean_ci(no_cd2, "disc_T")}
    # 3. object counts vs Lymphoid nodules
    counts = []
    for a in done:
        x = G.load(a, hovernet=False, hooknet=True)
        refs, ln, inflamed = EP.reference(x)
        if not refs:
            continue
        for arm, (mask, _) in EP.detector_masks(x).items():
            m = EP.match(mask & x["valid"], refs, ln)
            counts.append({"array": a, "case_id": x["case_id"], "arm": arm, **m})
    c = pd.DataFrame(counts)
    c.to_csv(OUT / "object_counts.csv", index=False)
    if len(c):
        result["object_counts_vs_lymphoid_nodule"] = c.groupby("arm")[["n_ref", "tp_ref", "n_pred", "tp_pred", "tp_pred_lenient"]].sum().astype(int).to_dict("index")
        result["object_counts_arrays"] = int(c.array.nunique())
    # 4. GPU hours
    seconds = 0.0
    for a in done:
        s = json.loads((C.OUT / "cells" / a / "summary.json").read_text())
        seconds += s["segmentation_seconds"] + s["classification_seconds"]
        h = C.OUT / "hooknet" / a / "summary.json"
        if h.exists():
            seconds += json.loads(h.read_text()).get("wall_seconds", 0)
    result["gpu_hours_stage1_and_hooknet"] = seconds / 3600
    C.save_json(result, OUT / "interim.json")
    print(json.dumps({k: v for k, v in result.items() if k != "arrays_processed"}, indent=1, default=str)[:5000])


if __name__ == "__main__":
    main()
