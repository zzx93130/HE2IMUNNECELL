"""Step 2 evaluation (CPU): held-out arm labels -> composition, IHC tile Spearman, ST RNA, TLS detection, primary test,
FFPE sanity. Labels (IHC, RNA, annotation) are used only here, plus the preregistered training-fold A4 base-arm choice.

Writes outputs/tnbc_frozen_adapt_20261101/{tables/*.csv, summary.json, figures/*.png}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, stats
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frozen_adapt_common as F  # noqa: E402
import frozen_adapt_step2 as S  # noqa: E402
import tnbc_grids as G  # noqa: E402
import tnbc_eval_ihc as EI  # noqa: E402
import tnbc_eval_path as EP  # noqa: E402
import tnbc_eval_rna as ER  # noqa: E402

EVAL_OUT = F.OUT / "work" / "test_eval" if S.TEST else F.OUT
TABLES = EVAL_OUT / "tables"
FIG = EVAL_OUT / "figures"
ARMS = ["stored", "A0", "A1", "A2", "A3", "A4", "A5"]
CLASSES = ["Lymphoid nodule", "Lymphocyte", "High TIL stroma", "Tumor", "Low TIL stroma"]
NB = 2000
CACHE = F.OUT / "work" / "eval_cache"


def labels_from(p):
    return np.array(F.TYPES)[p.argmax(1)]


def load_probs(array):
    z = np.load(S.PROBS / f"{array}.npz")
    return {k: z[k] for k in z.files}


def case(array):
    return f"TNBC{F.inventory()[array]['sample']}"


def ihc_ok(array):
    r = F.MAIN / "registration" / f"{array}.json"
    return r.exists() and json.loads(r.read_text())["accept"] and (F.MAIN / "ihc" / case(array) / "points.npz").exists()


_X = {}


def base_x(array):
    if array not in _X:
        x = G.load(array, hovernet=False, hooknet=False)
        x["ihc"] = EI.mapped_ihc(array, x) if ihc_ok(array) else None
        _X.clear()
        _X[array] = x
    return _X[array]


def with_labels(array, xy_um, ct, weight):
    x = dict(base_x(array))
    n = x["n"]
    x["cells_xy_um"], x["celltype"] = xy_um, ct
    x["counts"] = {k: G.counts_grid(xy_um[ct == k, 0], xy_um[ct == k, 1], n) * weight for k in F.TYPES}
    return x


def tile_rhos(x, tile_um):
    ihc = x["ihc"]
    overlap = x["tissue"] & ~np.isin(x["labels"], [G.NAMES.index("Artefacts"), G.NAMES.index("Hole (whitespace)")]) & ihc["tissue"]
    t = EI.tile_table(x, ihc, overlap, tile_um, None)
    if len(t) < 10:
        return {}
    r = {k: EI.spearman(t[a], t[b]) for k, (a, b) in {"rho_B_CD20": ("HE_B", "IHC_CD20"), "rho_B_CD3": ("HE_B", "IHC_CD3"),
                                                       "rho_T_CD3": ("HE_T", "IHC_CD3"), "rho_T_CD20": ("HE_T", "IHC_CD20")}.items()}
    r["disc_B"] = r["rho_B_CD20"] - r["rho_B_CD3"]
    r["disc_T"] = r["rho_T_CD3"] - r["rho_T_CD20"]
    r["tiles"] = len(t)
    return r


def tls(x):
    xv = dict(x)
    out = {}
    if x["ihc"] is not None:
        ihc = x["ihc"]
        overlap = x["tissue"] & ~np.isin(x["labels"], [G.NAMES.index("Artefacts"), G.NAMES.index("Hole (whitespace)")]) & ihc["tissue"]
        xv["valid"] = overlap & x["valid"]
    masks = EP.detector_masks(xv)
    refs, ln, _ = EP.reference(x)
    for arm in ["Two-stage (HE2CELL)", "B density (HE2CELL)"]:
        m = masks[arm][0] & x["valid"]
        out[f"path|{arm}"] = EP.match(m, refs, ln)
    if x["ihc"] is not None:
        aggs, _, _ = EI.ihc_aggregates(x, x["ihc"], overlap, 2000, x["sample"] not in EI.CD2)
        for arm in ["Two-stage (HE2CELL)", "B density (HE2CELL)"]:
            pred = EP.objects(masks[arm][0] & xv["valid"])
            po, ro = EI.match_tolerance(pred, aggs)
            out[f"ihc|{arm}"] = {"n_ref": len(aggs), "n_pred": len(pred), "tp_ref": int(sum(ro)), "tp_pred": int(sum(po))}
    return out


_EXPR = {}


def rna(array, x):
    entry = G.inventory()[array]
    key = array
    if key not in _EXPR:
        p = CACHE / f"{array}_spots.pkl"
        if p.exists():
            _EXPR.clear()
            _EXPR[key] = pd.read_pickle(p)
        else:
            spots = pd.read_csv(F.ROOT / entry["spot_table_all"], sep="\t")
            spots = spots[spots.selected == 1].copy()
            s = G.scale()
            spots["x_um"], spots["y_um"] = spots.pixel_x * s["large_mpp"], spots.pixel_y * s["large_mpp"]
            gi = np.clip((spots.y_um / G.GRID).astype(int), 0, x["n"] - 1)
            gj = np.clip((spots.x_um / G.GRID).astype(int), 0, x["n"] - 1)
            spots = spots[x["valid"][gi, gj]].copy()
            spots["spot"] = spots.x.astype(str) + "x" + spots.y.astype(str)
            spots = spots.merge(ER.expression(entry)[["T", "B"]], left_on="spot", right_index=True, how="inner")
            CACHE.mkdir(parents=True, exist_ok=True)
            spots.to_pickle(p)
            _EXPR.clear()
            _EXPR[key] = spots
    spots = _EXPR[key]
    hits = cKDTree(x["cells_xy_um"]).query_ball_point(spots[["x_um", "y_um"]].to_numpy(), ER.RADIUS_UM)
    ct = x["celltype"]
    b = np.array([(ct[h] == "B").sum() for h in hits])
    t = np.array([(ct[h] == "T").sum() for h in hits])
    return {"rna_B_vs_Bgenes": EI.spearman(b, spots.B.to_numpy()), "rna_T_vs_Tgenes": EI.spearman(t, spots["T"].to_numpy()), "spots": len(spots)}


def composition(array, xy_um, ct, arm):
    x = base_x(array)
    if x["labels"] is None:
        return []
    gx = np.clip((xy_um[:, 0] / G.GRID).astype(int), 0, x["n"] - 1)
    gy = np.clip((xy_um[:, 1] / G.GRID).astype(int), 0, x["n"] - 1)
    lab = x["labels"][gy, gx]
    rows = []
    for cls in CLASSES:
        sel = lab == G.NAMES.index(cls)
        if sel.sum() < 20:
            continue
        vc = pd.Series(ct[sel]).value_counts()
        rows.append({"array": array, "case_id": case(array), "arm": arm, "class": cls, "cells": int(sel.sum()),
                     **{f"share_{t}": vc.get(t, 0) / sel.sum() for t in F.TYPES}})
    return rows


def arm_labels(arrays):
    """array -> {arm: labels on the eval subsample (found cells)}, xy, plus the A4 selection log."""
    corr = F.prior_tau()
    from model_eval import apply_prior
    data, train_scores = {}, []
    for a in arrays:
        cells = F.load_cells(a)
        ev, _ = S.eval_subsample(a, cells)
        p = load_probs(a)
        if S.TEST:
            ev &= cells.global_instance_id.isin(p["global_instance_id"]).to_numpy()
        sub = cells[ev].reset_index(drop=True)
        assert (p["global_instance_id"] == sub.global_instance_id.to_numpy()).all()
        f = p["found"]
        xy = np.c_[sub.x_hd, sub.y_hd][f] * F.hd_mpp()
        d = {"xy": xy, "stored_sub": sub.celltype.to_numpy()[f], "found": float(f.mean()), "p": {}, "train": {}}
        for arm in ["A0", "A1", "A2", "A3", "A5"]:
            d["p"][arm] = p[arm][f]
            d[arm] = labels_from(apply_prior(p[arm][f], corr))
        trs = p["train_subset"][f]
        d["train_xy"] = xy[trs]
        for k in F.folds():
            for arm in ["A1", "A3"]:
                if f"train_{arm}_{k}" in p:
                    d["train"][(arm, k)] = _train(p, f, arm, k)
            d["train"][("A2", k)] = p["A2"][f][trs]
        data[a] = d
    # A4: per fold, pick the A1-A3 arm by training-patient mean rho(B, CD20) at 200 um, then EM prior on training cells
    sel_log = {}
    for k, test in F.folds().items():
        train = [a for a in arrays if case(a) not in test]
        scores = {}
        for arm in ["A1", "A2", "A3"]:
            vals = []
            for a in train:
                if (arm, k) not in data[a]["train"] or not ihc_ok(a):
                    continue
                pp = data[a]["train"][(arm, k)]
                ct = labels_from(apply_prior(pp, corr))
                r = tile_rhos(with_labels(a, data[a]["train_xy"], ct, 9.0), 200)
                if r:
                    vals.append(r["rho_B_CD20"])
                    train_scores.append({"fold": k, "arm": arm, "array": a, "rho_B_CD20": r["rho_B_CD20"]})
            scores[arm] = float(np.nanmean(vals)) if vals else np.nan
        if not train:
            sel_log[k] = {"train_arrays": 0}
            continue
        best = max(scores, key=lambda s: -np.inf if np.isnan(scores[s]) else scores[s])
        post = np.concatenate([data[a]["train"][(best, k)] for a in train if (best, k) in data[a]["train"]])
        pi, it = F.saerens_em(post)
        sel_log[k] = {"train_arrays": len(train), "train_scores": scores, "chosen": best, "em_prior_Other_T_Mac_B": pi.tolist(),
                      "em_iterations": it, "train_cells": int(len(post))}
        for a in arrays:
            if case(a) in test:
                data[a]["A4"] = labels_from(F.apply_em(data[a]["p"][best], pi))
                data[a]["A4_base"] = best
    return data, sel_log, pd.DataFrame(train_scores)


def _train(p, f, arm, k):
    """training-scoring probs restricted to found cells (train_subset was saved over all eval cells)."""
    trs_all = p["train_subset"]
    return p[f"train_{arm}_{k}"][f[trs_all]]


def mean_ci(v, n=NB, seed=F.SEED):
    v = pd.Series(v).dropna()
    if v.empty:
        return {"mean": np.nan, "CI95": [np.nan, np.nan], "patients": 0}
    rng = np.random.default_rng(seed)
    b = v.to_numpy()[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return {"mean": float(v.mean()), "CI95": np.percentile(b, [2.5, 97.5]).tolist(), "patients": int(len(v))}


def ratio_ci(frame, num, den, n=NB, seed=F.SEED):
    """pooled ratio sum(num)/sum(den) over patients with patient bootstrap."""
    g = frame.groupby("case_id")[[num, den]].sum()
    if g.empty or g[den].sum() == 0:
        return {"value": np.nan, "CI95": [np.nan, np.nan]}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (n, len(g)))
    b = g[num].to_numpy()[idx].sum(1) / np.maximum(g[den].to_numpy()[idx].sum(1), 1)
    return {"value": float(g[num].sum() / g[den].sum()), "CI95": np.percentile(b, [2.5, 97.5]).tolist(), "patients": int(len(g))}


def paired(a, b, n=NB, perm=10000, seed=F.SEED):
    d = (pd.Series(a) - pd.Series(b)).dropna()
    if d.empty:
        return {}
    rng = np.random.default_rng(seed)
    boots = d.to_numpy()[rng.integers(0, len(d), (n, len(d)))].mean(1)
    signs = rng.choice([-1, 1], (perm, len(d)))
    null = np.abs((signs * d.to_numpy()).mean(1))
    return {"mean_diff": float(d.mean()), "CI95": np.percentile(boots, [2.5, 97.5]).tolist(), "patients": int(len(d)),
            "positive": int((d > 0).sum()), "sign_flip_p": float((1 + (null >= abs(d.mean()) - 1e-12).sum()) / (perm + 1))}


def ffpe_summary():
    path = F.OUT / ("work/test_ffpe" if S.TEST else "ffpe") / "predictions.csv.gz"
    if not path.exists():
        return None
    f = pd.read_csv(path)

    def macro(g):
        return np.mean([2 * ((g.pred == c) & (g.label == c)).sum() / max((g.pred == c).sum() + (g.label == c).sum(), 1) for c in range(4)])
    per = f.groupby(["patient", "arm"]).apply(macro).unstack()
    out = {"patients": int(len(per)), "cells_per_patient": int(f.groupby(["patient", "arm"]).size().median())}
    for arm in ["clean", "A1", "A2", "A3"]:
        out[f"macroF1_{arm}"] = mean_ci(per[arm])
    pooled = {arm: macro(f[f.arm == arm]) for arm in ["clean", "A1", "A2", "A3"]}
    out["pooled_macroF1"] = pooled
    for arm in ["A1", "A2", "A3"]:
        out[f"delta_{arm}_minus_clean"] = paired(per[arm], per["clean"])
        out[f"flag_{arm}_drop_gt_0.02"] = bool(out[f"delta_{arm}_minus_clean"]["mean_diff"] < -0.02)
    per.to_csv(TABLES / "ffpe_per_patient_macroF1.csv")
    return out


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    arrays = [a for a in F.priority() if (S.PROBS / f"{a}.npz").exists()]
    data, sel_log, train_scores = arm_labels(arrays)
    train_scores.to_csv(TABLES / "a4_training_fold_scores.csv", index=False)
    rows, comp, tlsrows, checks = [], [], [], []
    for a in arrays:
        d = data[a]
        x0 = base_x(a)
        checks.append({"array": a, "case_id": case(a), "crops_found": d["found"], "cells_eval": len(d["xy"]),
                       "A0_vs_stored_agreement": float((d["A0"] == d["stored_sub"]).mean()),
                       "A0_B_count": int((d["A0"] == "B").sum()), "stored_B_count_sub": int((d["stored_sub"] == "B").sum())})
        for arm in ARMS:
            if arm != "stored" and arm not in d:
                continue
            if arm == "stored":
                xy, ct, w = x0["cells_xy_um"], x0["celltype"], 1.0
            else:
                xy, ct, w = d["xy"], d[arm], 1 / S.FRACTION
            x = with_labels(a, xy, ct, w)
            r = {"array": a, "case_id": case(a), "fold": F.fold_of(case(a)), "arm": arm, "ihc": x["ihc"] is not None,
                 "CD2_slide": x["sample"] in EI.CD2, "cells": len(ct), **{f"share_{t}": float((ct == t).mean()) for t in F.TYPES}}
            if x["ihc"] is not None:
                for size in (200, 500):
                    r.update({f"{k}_{size}": v for k, v in tile_rhos(x, size).items()})
            r.update(rna(a, x))
            rows.append(r)
            comp += composition(a, xy, ct, arm)
            if x["labels"] is not None:
                for key, m in tls(x).items():
                    src, det = key.split("|")
                    tlsrows.append({"array": a, "case_id": case(a), "arm": arm, "reference": src, "detector": det, **m})
        print(a, "evaluated", flush=True)
    per = pd.DataFrame(rows)
    comp = pd.DataFrame(comp)
    tl = pd.DataFrame(tlsrows)
    per.to_csv(TABLES / "per_array_arm.csv", index=False)
    comp.to_csv(TABLES / "composition_by_class.csv", index=False)
    tl.to_csv(TABLES / "tls_detection.csv", index=False)
    pd.DataFrame(checks).to_csv(TABLES / "a0_reproduction.csv", index=False)
    summary = {"arrays": arrays, "patients": int(per.case_id.nunique()), "a4_selection": sel_log,
               "a0_reproduction": {"median_agreement": float(pd.DataFrame(checks).A0_vs_stored_agreement.median()),
                                   "min_agreement": float(pd.DataFrame(checks).A0_vs_stored_agreement.min()),
                                   "crops_found_min": float(pd.DataFrame(checks).crops_found.min())}, "arms": {}}
    for arm in ARMS:
        f = per[per.arm == arm]
        e = {}
        cd3 = f[~f.CD2_slide]
        for col, frame in [("rho_B_CD20_200", f), ("rho_T_CD3_200", cd3), ("rho_B_CD3_200", cd3), ("disc_B_200", cd3),
                           ("rho_B_CD20_500", f), ("rho_T_CD3_500", cd3), ("rna_B_vs_Bgenes", f), ("rna_T_vs_Tgenes", f)]:
            if col in frame:
                e[col] = mean_ci(frame.groupby("case_id")[col].mean())
        c = comp[(comp.arm == arm)]
        for cls in CLASSES:
            cc = c[c["class"] == cls].assign(B=lambda g: g.share_B * g.cells, T=lambda g: g.share_T * g.cells, O=lambda g: g.share_Other * g.cells)
            if len(cc):
                e[f"B_share_{cls}"] = ratio_ci(cc, "B", "cells")
                e[f"T_share_{cls}"] = ratio_ci(cc, "T", "cells")
                e[f"Other_share_{cls}"] = ratio_ci(cc, "O", "cells")
        for (src, det), g in tl[tl.arm == arm].groupby(["reference", "detector"]):
            e[f"TLS_recall_{src}_{det}"] = ratio_ci(g, "tp_ref", "n_ref")
            e[f"TLS_precision_{src}_{det}"] = ratio_ci(g, "tp_pred", "n_pred")
            e[f"TLS_counts_{src}_{det}"] = {k: int(g[k].sum()) for k in ["n_ref", "n_pred", "tp_ref", "tp_pred"]}
        summary["arms"][arm] = e
    pv = per.pivot_table(index="case_id", columns="arm", values="rho_B_CD20_200", dropna=False).reindex(columns=ARMS)
    pv3 = per[~per.CD2_slide].pivot_table(index="case_id", columns="arm", values="rho_T_CD3_200", dropna=False).reindex(columns=ARMS)
    summary["primary_A3_vs_A0_rho_B_CD20_200"] = paired(pv["A3"], pv["A0"])
    summary["secondary_paired_vs_A0"] = {} if "A4" not in pv else {arm: {"rho_B_CD20_200": paired(pv[arm], pv["A0"]),
                                               "rho_T_CD3_200": paired(pv3[arm], pv3["A0"])}
                                         for arm in ["A1", "A2", "A4", "A5"]}
    summary["ffpe_sanity"] = ffpe_summary()
    F.C.save_json(summary, EVAL_OUT / "summary.json")
    figures(per, comp)
    print(json.dumps(summary["primary_A3_vs_A0_rho_B_CD20_200"]))


def figures(per, comp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"Other": "#B8B8B8", "T": "#2E8B57", "Macrophage": "#E0A400", "B": "#C0392B"}
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(18, 4), sharey=True)
    for ax, cls in zip(axes, CLASSES):
        c = comp[comp["class"] == cls]
        bottom = np.zeros(len(ARMS))
        for t in F.TYPES:
            v = np.array([np.average(c[c.arm == a][f"share_{t}"], weights=c[c.arm == a].cells) if len(c[c.arm == a]) else 0 for a in ARMS])
            ax.bar(ARMS, v, bottom=bottom, color=colors[t], label=t, width=.75)
            bottom += v
        ax.set_title(cls)
        ax.tick_params(axis="x", rotation=45)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("label share (cells pooled over held-out patients)")
    axes[-1].legend(frameon=False, bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(FIG / "composition_by_class.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["rho_B_CD20_200", "rho_T_CD3_200", "disc_B_200"]):
        f = per[per.arm.isin(ARMS)]
        if col not in f:
            continue
        if col != "rho_B_CD20_200":
            f = f[~f.CD2_slide]
        pv = f.pivot_table(index="case_id", columns="arm", values=col).reindex(columns=[a for a in ARMS if a in f.arm.unique()])
        for _, row in pv.iterrows():
            ax.plot(range(len(pv.columns)), row.values, color="#999", lw=.6, alpha=.6)
        ax.plot(range(len(pv.columns)), pv.mean().values, "o-", color="#C0392B", lw=2)
        ax.set_xticks(range(len(pv.columns)), pv.columns)
        ax.axhline(0, color="k", lw=.5)
        ax.set_title(col + " (per held-out patient; red = mean)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "tile_spearman_by_arm.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
