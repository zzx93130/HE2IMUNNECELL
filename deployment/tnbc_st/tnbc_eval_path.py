"""Stage 4a: TLS detection against the pathologist annotation on the same H&E section (claim 2).

Rules are those of preregistration.json (A): reference TLS = Lymphoid-nodule components >= 20,000 µm² in the
valid region; negatives = Lymphocyte + High TIL stroma (inflamed) and other annotated tissue; object matching
by esca_analysis' 10 %-overlap rule; 96 µm tiles; frozen ESCA detector parameters. Paired arm differences use
esca_paired_stats (shared patient-cluster draws, sign-flip permutation).

Writes analysis/pathologist/{objects.csv, tiles.csv.gz, per_arm.json, paired.json}.
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
import esca_analysis as A  # noqa: E402
import esca_paired_stats as P  # noqa: E402

OUT = C.OUT / "analysis" / "pathologist"
PREREG = json.loads((C.OUT / "preregistration.json").read_text())
FROZEN = PREREG["frozen_detectors"]
MIN_REF_UM2 = 20000.0
TILE = 12  # grid cells = 96 µm
ARMS = ["Two-stage (HE2CELL)", "B density (HE2CELL)", "Nuclei only (frozen ESCA, 11000)",
        "Nuclei only (extended grid, 16000)", "HookNet-TLS", "HoVer-Net inflammatory density"]


def objects(mask, min_cells=0):
    labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
    sl = ndimage.find_objects(labels)
    out = []
    for i, s in enumerate(sl, 1):
        if s is None:
            continue
        if (labels[s] == i).sum() >= min_cells:
            m = np.zeros(mask.shape, bool)
            m[s] = labels[s] == i
            out.append(m)
    return out


def detector_masks(x):
    """Arm -> (TLS mask, tile-score map). Score maps are reduced per tile by max."""
    masks = {}
    p = FROZEN["B density (HE2CELL)"]
    m, d = A.detect(x, ["B"], **p)
    masks["B density (HE2CELL)"] = (m, d)
    for name in ["Nuclei only (frozen ESCA, 11000)", "Nuclei only (extended grid, 16000)"]:
        m, d = A.detect(x, None, **FROZEN[name])
        masks[name] = (m, d)
    p = FROZEN["Two-stage (HE2CELL)"]
    proposals, _ = A.detect(x, None, **{k: p[k] for k in ["sigma_um", "threshold", "min_area_um2", "min_cells"]})
    total = sum(x["counts"].values())
    keep = np.zeros_like(proposals)
    share_map = np.zeros(proposals.shape, np.float32)
    for obj in objects(proposals):
        share = x["counts"]["B"][obj].sum() / max(total[obj].sum(), 1)
        share_map[obj] = share
        if share >= p["b_share"]:
            keep |= obj
    masks["Two-stage (HE2CELL)"] = (keep, share_map)
    if "hooknet" in x:
        masks["HookNet-TLS"] = (x["hooknet"]["tls"] & x["valid"], x["hooknet"]["confidence"])
    if "hovernet" in x:
        hx = {"counts": {"inflammatory": x["hovernet"]["inflammatory"]}, "valid": x["valid"]}
        m, d = A.detect(hx, ["inflammatory"], **FROZEN["HoVer-Net inflammatory density"])
        masks["HoVer-Net inflammatory density"] = (m, d)
    return masks


def reference(x):
    ln = (x["labels"] == G.LN) & x["valid"]
    refs = [o for o in objects(ln) if o.sum() * G.GRID ** 2 >= MIN_REF_UM2]
    inflamed = np.isin(x["labels"], [G.LYMPHOCYTE, G.HIGH_TIL]) & x["valid"]
    return refs, ln, inflamed


def match(pred_mask, refs, ln):
    pred = objects(pred_mask)
    overlaps = lambda a, b: (a & b).sum() >= A.MATCH_FRACTION * min(a.sum(), b.sum())  # noqa: E731
    tp_ref = sum(any(overlaps(r, p) for p in pred) for r in refs)
    tp_pred = sum(any(overlaps(p, r) for r in refs) for p in pred)
    tp_pred_lenient = sum((p & ln).sum() >= A.MATCH_FRACTION * p.sum() for p in pred)
    return {"n_ref": len(refs), "n_pred": len(pred), "tp_ref": tp_ref, "tp_pred": tp_pred, "tp_pred_lenient": tp_pred_lenient}


def tiles(x, score):
    n = x["n"]
    k = n // TILE
    rows = []
    lab, valid = x["labels"], x["valid"]
    for i in range(k):
        for j in range(k):
            s = (slice(i * TILE, (i + 1) * TILE), slice(j * TILE, (j + 1) * TILE))
            v = valid[s]
            if v.mean() < .5:
                continue
            l = lab[s][v]
            ln = (l == G.LN).mean()
            inflamed = np.isin(l, [G.LYMPHOCYTE, G.HIGH_TIL]).mean()
            if ln >= .5:
                label = "TLS"
            elif ln == 0 and inflamed >= .5:
                label = "inflamed"
            elif ln == 0:
                label = "other"
            else:
                continue
            rows.append((i, j, label, {a: float(np.max(sc[s][v])) for a, sc in score.items()}))
    return rows


def per_array(array):
    x = G.load(array)
    refs, ln, inflamed = reference(x)
    masks = detector_masks(x)
    objs, tile_rows = [], []
    for arm, (mask, _) in masks.items():
        mask = mask & x["valid"]
        objs.append({"array": array, "case_id": x["case_id"], "arm": arm, **match(mask, refs, ln),
                     "inflamed_um2": float(inflamed.sum() * G.GRID ** 2),
                     "inflamed_called_um2": float((inflamed & mask).sum() * G.GRID ** 2),
                     "predicted_um2": float(mask.sum() * G.GRID ** 2), "valid_mm2": float(x["valid"].sum() * G.GRID ** 2 / 1e6),
                     "ln_um2": float(ln.sum() * G.GRID ** 2)})
    for i, j, label, scores in tiles(x, {a: s for a, (_, s) in masks.items()}):
        for arm, value in scores.items():
            tile_rows.append({"array": array, "case_id": x["case_id"], "arm": arm, "tile": f"{i}_{j}", "label": label, "score": value})
    return objs, tile_rows


def auc(labels, scores):
    labels = np.asarray(labels, bool)
    if labels.all() or (~labels).all():
        return np.nan
    ranks = stats.rankdata(scores)
    n1 = labels.sum()
    return (ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * (~labels).sum())


def metric(parts):
    obj, til = parts
    recall = obj.tp_ref.sum() / max(obj.n_ref.sum(), 1)
    precision = obj.tp_pred.sum() / max(obj.n_pred.sum(), 1)
    lenient = obj.tp_pred_lenient.sum() / max(obj.n_pred.sum(), 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    t = til[til.label.isin(["TLS", "inflamed"])]
    return {"object_F1": f1, "object_recall": recall, "object_precision": precision, "object_precision_lenient": lenient,
            "tile_AUROC_TLS_vs_inflamed": auc(t.label == "TLS", t.rank_score),
            "tile_AUROC_TLS_vs_all": auc(til.label == "TLS", til.rank_score),
            "inflamed_area_called_TLS": obj.inflamed_called_um2.sum() / max(obj.inflamed_um2.sum(), 1)}


def frames(obj, til, arm, clusters, offset_obj=0, offset_til=0):
    o = obj[obj.arm == arm].reset_index(drop=True)
    t = til[til.arm == arm].reset_index(drop=True)
    t["rank_score"] = t.score.rank(pct=True)  # within-arm ranks so swaps never mix score scales
    return o, t


def groups_for(o, t, clusters, offset_o, offset_t):
    go = pd.Series(np.arange(len(o)) + offset_o, index=o.case_id.to_numpy())
    gt = pd.Series(np.arange(len(t)) + offset_t, index=t.case_id.to_numpy())
    pick = lambda s, c: s.loc[[c]].to_numpy() if c in s.index else np.empty(0, int)  # noqa: E731
    return {c: (pick(go, c), pick(gt, c)) for c in clusters}


def analyse(obj, til, arms, pairs, label):
    """Per-arm estimates and paired differences on the arrays every compared arm has."""
    clusters = sorted(obj.case_id.unique())
    boot, perm = P.draws(len(clusters)), P.flips(len(clusters))
    per_arm = {}
    for arm in arms:
        o, t = frames(obj, til, arm, clusters)
        if o.empty:
            continue
        per_arm[arm] = {"arrays": int(len(o)), "patients": int(o.case_id.nunique()), "reference_TLS": int(o.n_ref.sum()),
                        **P.bootstrap_arm((o, t), groups_for(o, t, clusters, 0, 0), metric, clusters, boot)}
    comparisons = []
    for a, b in pairs:
        if a not in per_arm or b not in per_arm:
            continue
        oa, ta = frames(obj, til, a, clusters)
        ob, tb = frames(obj, til, b, clusters)
        common = set(oa.array) & set(ob.array)
        oa, ob = oa[oa.array.isin(common)], ob[ob.array.isin(common)]
        ta, tb = ta[ta.array.isin(common)], tb[tb.array.isin(common)]
        base_o = pd.concat([oa, ob], ignore_index=True)
        base_t = pd.concat([ta, tb], ignore_index=True)
        ga = groups_for(oa.reset_index(drop=True), ta.reset_index(drop=True), clusters, 0, 0)
        gb = groups_for(ob.reset_index(drop=True), tb.reset_index(drop=True), clusters, len(oa), len(ta))
        result = P.paired_comparison((base_o, base_t), ga, gb, metric, clusters, boot, perm)
        comparisons.append({"a": a, "b": b, "arrays": len(common), "metrics": result})
        print(f"[{label}] {a} - {b} ({len(common)}): " + ", ".join(
            f"{k} {r['difference']:+.3f} [{r['CI95_difference'][0]:+.3f}, {r['CI95_difference'][1]:+.3f}] p={r['p_sign_flip']:.3g}"
            for k, r in result.items()), flush=True)
    return {"clusters": len(clusters), "resamples": len(boot), "permutations": len(perm), "per_arm": per_arm, "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", nargs="*")
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    inv = G.inventory()
    if not args.stats_only:
        arrays = args.arrays or [a for a, e in inv.items() if e["annotated"] and (C.OUT / "cells" / a / "cells.csv.gz").exists()]
        objs, tile_rows = [], []
        for array in arrays:
            o, t = per_array(array)
            objs += o
            tile_rows += t
            print(array, {r["arm"]: (r["n_ref"], r["n_pred"], r["tp_ref"], r["tp_pred"]) for r in o}, flush=True)
        pd.DataFrame(objs).to_csv(OUT / "objects.csv", index=False)
        pd.DataFrame(tile_rows).to_csv(OUT / "tiles.csv.gz", index=False)
    obj = pd.read_csv(OUT / "objects.csv")
    til = pd.read_csv(OUT / "tiles.csv.gz")
    primary = "Two-stage (HE2CELL)"
    baselines = ["Nuclei only (frozen ESCA, 11000)", "Nuclei only (extended grid, 16000)", "HookNet-TLS", "HoVer-Net inflammatory density"]
    pairs = [(primary, b) for b in baselines] + [("B density (HE2CELL)", b) for b in baselines] + [(primary, "B density (HE2CELL)")]
    no_hovernet = [a for a in ARMS if a != "HoVer-Net inflammatory density"]
    full = analyse(obj, til, no_hovernet, [p for p in pairs if "HoVer-Net" not in p[1]], "all annotated")
    subset_arrays = set(obj[obj.arm == "HoVer-Net inflammatory density"].array)
    result = {"all_annotated": full}
    ref_arrays = set(obj[(obj.arm == primary) & (obj.n_ref > 0)].array)
    result["reference_arrays_only"] = analyse(obj[obj.array.isin(ref_arrays)], til[til.array.isin(ref_arrays)], no_hovernet,
                                              [p for p in pairs if "HoVer-Net" not in p[1]], "arrays with a reference TLS")
    if subset_arrays:
        result["hovernet_subset"] = analyse(obj[obj.array.isin(subset_arrays)], til[til.array.isin(subset_arrays)], ARMS, pairs, "HoVer-Net subset")
    C.save_json(result, OUT / "results.json")


if __name__ == "__main__":
    main()
