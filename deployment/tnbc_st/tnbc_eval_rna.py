"""Stage 4d: HE2CELL cell counts vs ST spot RNA on the same H&E section (preregistration.json D, secondary).

Spots: byArray spot_data-all (pixel_x/pixel_y in the imagesLarge frame), selected == 1, centre inside the valid
tissue, disk of 50 µm radius. Expression: raw counts (rawCountsMatrices, spot ids "<x>x<y>") -> log1p(CP10k);
panel = mean over genes. Gene symbols -> Ensembl ids from rest.ensembl.org (gene_ids_ensembl_rest.txt), matched
on the unversioned id. Spearman per array, mean over arrays per patient, patient bootstrap. IHC ceiling: CD3+/CD20+
blob counts in the same disks on registered arrays.

Writes analysis/rna/{spots.csv.gz, per_array.csv, results.json}.
"""
from __future__ import annotations

import argparse
import bz2
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tnbc_common as C  # noqa: E402
import tnbc_grids as G  # noqa: E402
import tnbc_eval_path as EP  # noqa: E402
import tnbc_eval_ihc as EI  # noqa: E402

OUT = C.OUT / "analysis" / "rna"
PANELS = json.loads((C.OUT / "preregistration.json").read_text())["D_ST_RNA"]["panels"]
RADIUS_UM = 50.0


def gene_ids():
    out = {}
    for line in (C.OUT / "gene_ids_ensembl_rest.txt").read_text().split("\n"):
        if line.strip():
            g, i = line.split()
            out[g] = i
    return out


def expression(entry):
    ids = gene_ids()
    path = next((C.EXTRAS / "rawCountsMatrices").rglob(entry["count_matrix"]))
    with bz2.open(path, "rt") as handle:
        header = handle.readline().rstrip("\n").split("\t")
    base = {h.split(".")[0]: i for i, h in enumerate(header)}
    wanted = {g: base[i] for g, i in ids.items() if i in base}
    # library size needs every column: read in chunks
    library, panel_rows, index = [], [], []
    for chunk in pd.read_csv(path, sep="\t", index_col=0, chunksize=500, compression="bz2"):
        library.append(chunk.sum(1))
        panel_rows.append(chunk.iloc[:, [c - 1 for c in wanted.values()]].set_axis(list(wanted), axis=1))
    lib = pd.concat(library)
    genes = pd.concat(panel_rows)
    logcp = np.log1p(genes.div(lib.clip(lower=1), axis=0) * 1e4)
    frame = pd.DataFrame(index=genes.index)
    for name, panel in PANELS.items():
        present = [g for g in panel if g in logcp]
        frame[name] = logcp[present].mean(1)
    frame["MS4A1_raw"], frame["CD3E_raw"] = genes.get("MS4A1", 0), genes.get("CD3E", 0)
    frame["umi"] = lib
    frame["genes_missing"] = ",".join(g for p in PANELS.values() for g in p if g not in logcp)
    return frame


def per_array(array):
    inv = G.inventory()
    entry = inv[array]
    x = G.load(array, hovernet=False, hooknet=False)
    spots = pd.read_csv(C.ROOT / entry["spot_table_all"], sep="\t")
    spots = spots[spots.selected == 1].copy()
    s = G.scale()
    spots["x_um"], spots["y_um"] = spots.pixel_x * s["large_mpp"], spots.pixel_y * s["large_mpp"]
    gi = np.clip((spots.y_um / G.GRID).astype(int), 0, x["n"] - 1)
    gj = np.clip((spots.x_um / G.GRID).astype(int), 0, x["n"] - 1)
    spots = spots[x["valid"][gi, gj]].copy()
    spots["spot"] = spots.x.astype(str) + "x" + spots.y.astype(str)
    tree = cKDTree(x["cells_xy_um"])
    hits = tree.query_ball_point(spots[["x_um", "y_um"]].to_numpy(), RADIUS_UM)
    ct = x["celltype"]
    for k in ["B", "T", "Other", "Macrophage"]:
        spots[f"HE_{k}"] = [int((ct[h] == k).sum()) for h in hits]
    spots["HE_nuclei"] = [len(h) for h in hits]
    # Two-stage TLS cover of the disk
    masks = EP.detector_masks(x)
    ts = masks["Two-stage (HE2CELL)"][0]
    spots["two_stage_cover"] = [float(ts[max(int((yy - RADIUS_UM) / G.GRID), 0):int((yy + RADIUS_UM) / G.GRID) + 1,
                                        max(int((xx - RADIUS_UM) / G.GRID), 0):int((xx + RADIUS_UM) / G.GRID) + 1].mean())
                                for xx, yy in zip(spots.x_um, spots.y_um)]
    reg = C.OUT / "registration" / f"{array}.json"
    if reg.exists() and json.loads(reg.read_text())["accept"]:
        ihc = EI.mapped_ihc(array, x)
        for k in ["CD3", "CD20"]:
            t = cKDTree(ihc[k])
            spots[f"IHC_{k}"] = [len(h) for h in t.query_ball_point(spots[["x_um", "y_um"]].to_numpy(), RADIUS_UM)]
        gi = np.clip((spots.y_um / G.GRID).astype(int), 0, x["n"] - 1)
        gj = np.clip((spots.x_um / G.GRID).astype(int), 0, x["n"] - 1)
        spots["IHC_tissue"] = ihc["tissue"][gi, gj]
    expr = expression(entry)
    spots = spots.merge(expr, left_on="spot", right_index=True, how="inner")
    spots.insert(0, "case_id", x["case_id"])
    spots.insert(0, "array", array)
    return spots


TESTS = {"T_vs_Tgenes": ("HE_T", "T"), "B_vs_Bgenes": ("HE_B", "B"), "nuclei_vs_Tgenes": ("HE_nuclei", "T"),
         "nuclei_vs_Bgenes": ("HE_nuclei", "B"), "B_vs_Tgenes": ("HE_B", "T"), "T_vs_Bgenes": ("HE_T", "B"),
         "B_vs_TLS12": ("HE_B", "TLS12"), "two_stage_cover_vs_TLS12": ("two_stage_cover", "TLS12"),
         "nuclei_vs_TLS12": ("HE_nuclei", "TLS12"),
         "IHC_CD3_vs_Tgenes": ("IHC_CD3", "T"), "IHC_CD20_vs_Bgenes": ("IHC_CD20", "B")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", nargs="*")
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    inv = G.inventory()
    if not args.stats_only:
        arrays = args.arrays or [a for a, e in inv.items() if "spot_table_all" in e and e["count_matrix_present"]
                                 and (C.OUT / "cells" / a / "cells.csv.gz").exists()]
        frames = []
        for array in arrays:
            f = per_array(array)
            frames.append(f)
            print(array, len(f), "spots", flush=True)
        pd.concat(frames).to_csv(OUT / "spots.csv.gz", index=False)
    spots = pd.read_csv(OUT / "spots.csv.gz")
    rows = []
    for (array, case), f in spots.groupby(["array", "case_id"]):
        r = {"array": array, "case_id": case, "spots": len(f)}
        for name, (a, b) in TESTS.items():
            sub = f[f.IHC_tissue == True] if a.startswith("IHC") and "IHC_tissue" in f else f  # noqa: E712
            r[name] = EI.spearman(sub[a], sub[b]) if a in sub and len(sub) >= 20 else np.nan
        r["B_minus_nuclei_Bgenes"] = r["B_vs_Bgenes"] - r["nuclei_vs_Bgenes"]
        r["T_minus_nuclei_Tgenes"] = r["T_vs_Tgenes"] - r["nuclei_vs_Tgenes"]
        r["B_disc"] = r["B_vs_Bgenes"] - r["B_vs_Tgenes"]
        r["T_disc"] = r["T_vs_Tgenes"] - r["T_vs_Bgenes"]
        rows.append(r)
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "per_array.csv", index=False)
    results = {"arrays": int(len(per)), "patients": int(per.case_id.nunique()), "spots": int(len(spots))}
    for col in [c for c in per if c not in ("array", "case_id", "spots")]:
        results[col] = EI.mean_ci(per, col)
    C.save_json(results, OUT / "results.json")
    print(json.dumps(results, indent=1)[:3000])


if __name__ == "__main__":
    main()
