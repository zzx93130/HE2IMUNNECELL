"""Stage 4b/c/e: HE2CELL vs serial-section CD3/CD20 IHC (claims 1-3), per preregistration.json B, C, E.

For each annotated array whose registration to its IHC slide passed QC, IHC CD3+/CD20+ blobs and the IHC tissue
mask are mapped onto the H&E 8 µm grid with the inverse of the accepted affine. Overlap = H&E tissue (valid)
AND IHC tissue at the mapped position.

b. Region level: per 500 µm (primary) and 200 µm tiles with >= 50 % overlap, densities per mm² of overlap;
   per-sample Spearman, mean over samples with a patient bootstrap; discriminant validity and partial correlations.
c. TLS level: CD20 aggregates (2,000 CD20+/mm² after sigma 25 µm, >= 20,000 µm², >= 20 CD20+, >= 20 CD3+ within
   50 µm; sensitivity 1,000 / 4,000) vs each H&E arm, 100 µm matching tolerance; ceiling: pathologist Lymphoid nodules.
e. Composition: B/(B+T) of two-stage TLS objects vs CD20/(CD3+CD20) of the matched aggregate (both dilated 50 µm).

Writes analysis/ihc/{tiles_<size>.csv.gz, per_sample_<size>.csv, aggregates.csv, tls_match.csv, composition.csv, results.json}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import ndimage, stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "esca_tls"))
import tnbc_common as C  # noqa: E402
import tnbc_grids as G  # noqa: E402
import tnbc_eval_path as EP  # noqa: E402
import esca_analysis as A  # noqa: E402
import esca_paired_stats as P  # noqa: E402

OUT = C.OUT / "analysis" / "ihc"
PREREG = json.loads((C.OUT / "preregistration.json").read_text())
CD2 = {9, 10, 78, 82, 83, 84}
TOLERANCE_UM = 100.0
DILATE_UM = 50.0


def mapped_ihc(array, x):
    """IHC points (H&E µm) and IHC tissue on the H&E grid, via the accepted registration."""
    reg = json.loads((C.OUT / "registration" / f"{array}.json").read_text())
    s = G.scale()
    M = np.asarray(reg["he_large_to_ihc_level0"])  # Large px -> IHC level-0 px
    inv = np.linalg.inv(M)
    sample = x["sample"]
    pts = np.load(C.OUT / "ihc" / f"TNBC{sample}" / "points.npz")
    out = {}
    for key in ["CD3", "CD20"]:
        p = pts[key]
        h = np.c_[p[:, 0], p[:, 1], np.ones(len(p))] @ inv.T
        out[key] = h[:, :2] * s["large_mpp"]  # Large px -> µm
    maps = np.load(C.OUT / "ihc" / f"TNBC{sample}" / "maps_2um.npz")
    tissue = maps["tissue"] > 0
    l0 = json.loads((C.OUT / "ihc" / f"TNBC{sample}" / "summary.json").read_text())["mpp_level0"]
    n = x["n"]
    cy, cx = np.meshgrid((np.arange(n) + .5) * G.GRID, (np.arange(n) + .5) * G.GRID, indexing="ij")
    large = np.stack([cx.ravel() / s["large_mpp"], cy.ravel() / s["large_mpp"], np.ones(n * n)], 1) @ M.T
    ti = (large[:, 1] * l0 / 2.0).astype(int)
    tj = (large[:, 0] * l0 / 2.0).astype(int)
    inside = (ti >= 0) & (ti < tissue.shape[0]) & (tj >= 0) & (tj < tissue.shape[1])
    mapped = np.zeros(n * n, bool)
    mapped[inside] = tissue[ti[inside], tj[inside]]
    out["tissue"] = mapped.reshape(n, n)
    out["registration"] = {k: reg[k] for k in ["median_residual_um", "tiles_converged", "tiles_total", "accept"]}
    return out


def tile_table(x, ihc, overlap, tile_um, hovernet):
    n = x["n"]
    centres = (np.arange(n) + .5) * G.GRID
    ti = (centres // tile_um).astype(int)
    tid = ti[:, None] * 10000 + ti[None, :]
    area = pd.Series(overlap.ravel()).groupby(tid.ravel()).sum() * G.GRID ** 2 / 1e6
    total = pd.Series(np.ones(n * n)).groupby(tid.ravel()).sum() * G.GRID ** 2 / 1e6
    frame = pd.DataFrame({"overlap_mm2": area, "tile_mm2": total})

    def count(xy, name):
        gx = np.clip((xy[:, 0] / G.GRID).astype(int), 0, n - 1)
        gy = np.clip((xy[:, 1] / G.GRID).astype(int), 0, n - 1)
        ok = overlap[gy, gx] & (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (xy[:, 0] < n * G.GRID) & (xy[:, 1] < n * G.GRID)
        t = (xy[ok, 1] // tile_um).astype(int) * 10000 + (xy[ok, 0] // tile_um).astype(int)
        frame[name] = pd.Series(t).value_counts().reindex(frame.index).fillna(0)
    xy, ct = x["cells_xy_um"], x["celltype"]
    for k in ["B", "T", "Other", "Macrophage"]:
        count(xy[ct == k], f"HE_{k}")
    count(xy, "HE_nuclei")
    count(ihc["CD3"], "IHC_CD3")
    count(ihc["CD20"], "IHC_CD20")
    if hovernet is not None:
        count(hovernet, "HN_inflammatory")
    frame = frame[frame.overlap_mm2 >= .5 * (tile_um / 1000) ** 2].copy()
    for c in [c for c in frame if c.startswith(("HE_", "IHC_", "HN_"))]:
        frame[c] = frame[c] / frame.overlap_mm2
    frame["HE_TB"] = frame.HE_T + frame.HE_B
    frame["IHC_total"] = frame.IHC_CD3 + frame.IHC_CD20
    return frame.reset_index(names="tile")


def spearman(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return stats.spearmanr(a, b).statistic


def partial_spearman(a, b, z):
    ra, rb, rz = (stats.rankdata(v) for v in (a, b, z))
    res = lambda v: v - np.polyval(np.polyfit(rz, v, 1), rz)  # noqa: E731
    if np.std(rz) == 0:
        return np.nan
    return float(np.corrcoef(res(ra), res(rb))[0, 1])


PAIRS = {"rho_B_CD20": ("HE_B", "IHC_CD20"), "rho_B_CD3": ("HE_B", "IHC_CD3"), "rho_T_CD3": ("HE_T", "IHC_CD3"),
         "rho_T_CD20": ("HE_T", "IHC_CD20"), "rho_TB_total": ("HE_TB", "IHC_total"), "rho_nuclei_total": ("HE_nuclei", "IHC_total"),
         "rho_HN_total": ("HN_inflammatory", "IHC_total"), "rho_B_share_CD20_share": None}


def per_sample(tiles):
    rows = []
    for (array, case), t in tiles.groupby(["array", "case_id"]):
        if len(t) < 10:
            continue
        r = {"array": array, "case_id": case, "tiles": len(t), "CD2_slide": int(case[4:]) in CD2}
        for name, pair in PAIRS.items():
            if pair is None:
                continue
            a, b = pair
            r[name] = spearman(t[a], t[b]) if a in t and t[a].notna().all() else np.nan
        r["partial_B_CD20_given_TB"] = partial_spearman(t.HE_B, t.IHC_CD20, t.HE_TB)
        r["partial_T_CD3_given_TB"] = partial_spearman(t.HE_T, t.IHC_CD3, t.HE_TB)
        r["partial_B_CD20_given_nuclei"] = partial_spearman(t.HE_B, t.IHC_CD20, t.HE_nuclei)
        r["partial_T_CD3_given_nuclei"] = partial_spearman(t.HE_T, t.IHC_CD3, t.HE_nuclei)
        r["partial_B_CD20_given_IHCtotal"] = partial_spearman(t.HE_B, t.IHC_CD20, t.IHC_total)
        r["partial_T_CD3_given_IHCtotal"] = partial_spearman(t.HE_T, t.IHC_CD3, t.IHC_total)
        r["disc_B"] = r["rho_B_CD20"] - r["rho_B_CD3"]
        r["disc_T"] = r["rho_T_CD3"] - r["rho_T_CD20"]
        r["TB_minus_nuclei"] = r["rho_TB_total"] - r["rho_nuclei_total"]
        r["TB_minus_HN"] = r["rho_TB_total"] - r.get("rho_HN_total", np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


def mean_ci(frame, column, n=2000, seed=20261025):
    v = frame[["case_id", column]].dropna()
    if v.empty:
        return {"mean": np.nan, "CI95": [np.nan, np.nan], "samples": 0}
    by = v.groupby("case_id")[column].mean()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(by), (n, len(by)))
    boots = by.to_numpy()[idx].mean(1)
    return {"mean": float(by.mean()), "median": float(by.median()), "CI95": np.percentile(boots, [2.5, 97.5]).tolist(),
            "samples": int(len(by)), "positive": int((by > 0).sum())}


def ihc_aggregates(x, ihc, overlap, threshold, use_cd3):
    n = x["n"]
    cd20 = G.counts_grid(ihc["CD20"][:, 0], ihc["CD20"][:, 1], n) * overlap
    cd3 = G.counts_grid(ihc["CD3"][:, 0], ihc["CD3"][:, 1], n) * overlap
    d = ndimage.gaussian_filter(cd20, 25 / G.GRID, mode="constant") / (G.GRID ** 2 / 1e6)
    out = []
    ring = int(round(DILATE_UM / G.GRID))
    for obj in EP.objects((d >= threshold) & overlap):
        if obj.sum() * G.GRID ** 2 < 20000 or cd20[obj].sum() < 20:
            continue
        grown = ndimage.binary_dilation(obj, iterations=ring)
        if use_cd3 and cd3[grown].sum() < 20:
            continue
        out.append(obj)
    return out, cd3, cd20


def match_tolerance(pred, refs):
    """pred, refs: lists of masks. Returns matched flags (pred, refs) at TOLERANCE_UM."""
    k = int(round(TOLERANCE_UM / G.GRID))
    grown_refs = [ndimage.binary_dilation(r, iterations=k) for r in refs]
    pred_ok = [any((p & g).any() for g in grown_refs) for p in pred]
    ref_ok = [any((p & g).any() for p in pred) for g in grown_refs]
    return pred_ok, ref_ok


def per_array(array, tiles_um=(500, 200)):
    x = G.load(array, hovernet=True, hooknet=True)
    ihc = mapped_ihc(array, x)
    overlap = x["tissue"] & ~np.isin(x["labels"], [G.NAMES.index("Artefacts"), G.NAMES.index("Hole (whitespace)")]) & ihc["tissue"] \
        if x["labels"] is not None else x["tissue"] & ihc["tissue"]
    hn = None
    hp = C.OUT / "baselines" / "hovernet" / f"{array}.csv.gz"
    if hp.exists():
        h = pd.read_csv(hp)
        h = h[h.type == 2]
        hn = np.c_[h.x_hd, h.y_hd] * G.scale()["hd_mpp"]
    tiles = {}
    for size in tiles_um:
        t = tile_table(x, ihc, overlap, size, hn)
        t.insert(0, "case_id", x["case_id"])
        t.insert(0, "array", array)
        tiles[size] = t
    # TLS level
    use_cd3 = x["sample"] not in CD2
    xv = dict(x)
    xv["valid"] = overlap & x["valid"]
    masks = EP.detector_masks(xv)
    refs_ln, ln, _ = EP.reference(xv) if x["labels"] is not None else ([], None, None)
    match_rows, agg_rows, comp_rows = [], [], []
    for threshold in [2000, 1000, 4000]:
        aggs, cd3, cd20 = ihc_aggregates(x, ihc, overlap, threshold, use_cd3)
        for i, a in enumerate(aggs):
            if threshold == 2000:
                agg_rows.append({"array": array, "case_id": x["case_id"], "aggregate": i, "area_um2": a.sum() * G.GRID ** 2,
                                 "CD20": cd20[a].sum(), "CD3_in_50um": cd3[ndimage.binary_dilation(a, iterations=6)].sum()})
        arms = {arm: [o for o in EP.objects(m & xv["valid"])] for arm, (m, _) in masks.items()}
        arms["Pathologist Lymphoid nodule (ceiling)"] = refs_ln
        for arm, pred in arms.items():
            pred_ok, ref_ok = match_tolerance(pred, aggs)
            match_rows.append({"array": array, "case_id": x["case_id"], "threshold": threshold, "arm": arm, "CD2_slide": not use_cd3,
                               "n_ref": len(aggs), "n_pred": len(pred), "tp_ref": int(sum(ref_ok)), "tp_pred": int(sum(pred_ok))})
        if threshold == 2000:
            k = int(round(DILATE_UM / G.GRID))
            for p in arms["Two-stage (HE2CELL)"]:
                for j, a in enumerate(aggs):
                    if (ndimage.binary_dilation(a, iterations=int(round(TOLERANCE_UM / G.GRID))) & p).any():
                        pg, ag = ndimage.binary_dilation(p, iterations=k), ndimage.binary_dilation(a, iterations=k)
                        b, t = x["counts"]["B"][pg].sum(), x["counts"]["T"][pg].sum()
                        comp_rows.append({"array": array, "case_id": x["case_id"], "CD2_slide": not use_cd3, "aggregate": j,
                                          "HE_area_um2": p.sum() * G.GRID ** 2, "HE_B": b, "HE_T": t, "HE_B_share_BT": b / max(b + t, 1),
                                          "HE_B_share_all": b / max(sum(c[pg].sum() for c in x["counts"].values()), 1),
                                          "IHC_CD20": cd20[ag].sum(), "IHC_CD3": cd3[ag].sum(),
                                          "IHC_CD20_share": cd20[ag].sum() / max(cd20[ag].sum() + cd3[ag].sum(), 1)})
                        break
    return tiles, match_rows, agg_rows, comp_rows, ihc["registration"]


def tls_metric(parts):
    (f,) = parts
    recall = f.tp_ref.sum() / max(f.n_ref.sum(), 1)
    precision = f.tp_pred.sum() / max(f.n_pred.sum(), 1)
    return {"recall": recall, "precision": precision, "F1": 2 * recall * precision / max(recall + precision, 1e-12)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", nargs="*")
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    inv = G.inventory()
    if not args.stats_only:
        arrays = args.arrays or [a for a, e in inv.items() if e.get("register") and (C.OUT / "cells" / a / "cells.csv.gz").exists()]
        tiles = {500: [], 200: []}
        matches, aggs, comps, regs = [], [], [], []
        for array in arrays:
            reg_path = C.OUT / "registration" / f"{array}.json"
            if not reg_path.exists():
                regs.append({"array": array, "status": "not registered"})
                continue
            reg = json.loads(reg_path.read_text())
            regs.append({"array": array, "case_id": f"TNBC{inv[array]['sample']}", "accept": reg["accept"],
                         "median_residual_um": reg["median_residual_um"], "tiles_converged": reg["tiles_converged"],
                         "tiles_total": reg["tiles_total"]})
            if not reg["accept"]:
                continue
            t, m, a, c, _ = per_array(array)
            for k in tiles:
                tiles[k].append(t[k])
            matches += m
            aggs += a
            comps += c
            print(array, len(t[500]), "tiles500", flush=True)
        for k, v in tiles.items():
            pd.concat(v).to_csv(OUT / f"tiles_{k}.csv.gz", index=False)
        pd.DataFrame(matches).to_csv(OUT / "tls_match.csv", index=False)
        pd.DataFrame(aggs).to_csv(OUT / "aggregates.csv", index=False)
        pd.DataFrame(comps).to_csv(OUT / "composition.csv", index=False)
        pd.DataFrame(regs).to_csv(OUT / "registration_status.csv", index=False)
    results = {}
    for size in [500, 200]:
        t = pd.read_csv(OUT / f"tiles_{size}.csv.gz")
        ps = per_sample(t)
        ps.to_csv(OUT / f"per_sample_{size}.csv", index=False)
        cd3_ok = ps[~ps.CD2_slide]
        entry = {}
        for col in ["rho_B_CD20", "rho_B_CD3", "rho_T_CD3", "rho_T_CD20", "rho_TB_total", "rho_nuclei_total", "rho_HN_total",
                    "partial_B_CD20_given_TB", "partial_T_CD3_given_TB", "partial_B_CD20_given_nuclei", "partial_T_CD3_given_nuclei",
                    "partial_B_CD20_given_IHCtotal", "partial_T_CD3_given_IHCtotal",
                    "disc_B", "disc_T", "TB_minus_nuclei", "TB_minus_HN"]:
            frame = ps if col in ("rho_B_CD20", "partial_B_CD20_given_TB", "partial_B_CD20_given_nuclei") else cd3_ok
            entry[col] = mean_ci(frame, col)
            entry[col + "_incl_CD2_as_CD3"] = mean_ci(ps, col)
        pooled = {}
        for name, pair in PAIRS.items():
            if pair and pair[0] in t:
                sub = t[~t.case_id.str[4:].astype(int).isin(CD2)] if "CD3" in pair[1] or "total" in pair[1] else t
                pooled[name] = float(spearman(sub[pair[0]], sub[pair[1]]))
        entry["pooled_tile_rho"] = pooled
        entry["tiles"] = int(len(t))
        results[f"tiles_{size}um"] = entry
    m = pd.read_csv(OUT / "tls_match.csv")
    tls = {}
    clusters = sorted(m.case_id.unique())
    boot = P.draws(len(clusters))
    for threshold, part in m.groupby("threshold"):
        tls[int(threshold)] = {}
        for arm, f in part.groupby("arm"):
            f = f.reset_index(drop=True)
            groups = P.index_groups((f,), "case_id", clusters)
            tls[int(threshold)][arm] = {"arrays": int(len(f)), "aggregates": int(f.n_ref.sum()), "predicted": int(f.n_pred.sum()),
                                        **P.bootstrap_arm((f,), groups, tls_metric, clusters, boot)}
    results["tls_level"] = tls
    # Paired differences at the primary threshold (2,000 CD20+/mm²), shared draws, sign-flip permutation.
    perm = P.flips(len(clusters))
    primary = m[m.threshold == 2000]
    paired = []
    arms_avail = set(primary.arm)
    for a in ["Two-stage (HE2CELL)", "B density (HE2CELL)"]:
        for b in ["Nuclei only (frozen ESCA, 11000)", "Nuclei only (extended grid, 16000)", "HookNet-TLS",
                  "HoVer-Net inflammatory density", "Pathologist Lymphoid nodule (ceiling)", "B density (HE2CELL)"]:
            if a == b or b not in arms_avail or a not in arms_avail:
                continue
            fa, fb = primary[primary.arm == a], primary[primary.arm == b]
            common = sorted(set(fa.array) & set(fb.array))
            fa = fa[fa.array.isin(common)].reset_index(drop=True)
            fb = fb[fb.array.isin(common)].reset_index(drop=True)
            base = pd.concat([fa, fb], ignore_index=True)
            groups = P.index_groups((base,), "case_id", clusters)
            ga = {c: (g[0][g[0] < len(fa)],) for c, g in groups.items()}
            gb = {c: (g[0][g[0] >= len(fa)],) for c, g in groups.items()}
            r = P.paired_comparison((base,), ga, gb, tls_metric, clusters, boot, perm)
            paired.append({"a": a, "b": b, "arrays": len(common), "metrics": r})
    results["tls_level_paired_2000"] = paired
    comp = pd.read_csv(OUT / "composition.csv")
    comp = comp[~comp.CD2_slide]
    if len(comp) >= 5:
        rng = np.random.default_rng(20261025)
        cases = comp.case_id.unique()
        groups = {c: comp[comp.case_id == c] for c in cases}
        boots = []
        for _ in range(2000):
            s = pd.concat([groups[c] for c in rng.choice(cases, len(cases))])
            boots.append(spearman(s.HE_B_share_BT, s.IHC_CD20_share))
        results["composition"] = {"objects": int(len(comp)), "patients": int(len(cases)),
                                  "rho_HE_B_share_vs_IHC_CD20_share": float(spearman(comp.HE_B_share_BT, comp.IHC_CD20_share)),
                                  "CI95": np.nanpercentile(boots, [2.5, 97.5]).tolist(),
                                  "median_HE_B_share_BT": float(comp.HE_B_share_BT.median()),
                                  "median_IHC_CD20_share": float(comp.IHC_CD20_share.median())}
    C.save_json(results, OUT / "results.json")
    print(json.dumps(results, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
