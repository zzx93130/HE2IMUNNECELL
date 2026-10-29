"""Paired method comparisons on the TCGA-ESCA held-out test patients (frozen dev-tuned parameters).

For every pair of methods: the difference of each metric with a patient-cluster bootstrap CI that uses
the same 2000 resamples of the 85 test patients for both arms, and a two-sided cluster sign-flip
permutation p-value (each patient's results are swapped between the two arms with probability 1/2).
Pairs are scored on the windows both arms have. Window scores are rank-normalised within each arm before
a swap, so AUROC permutations do not mix the scales of different density maps.

Metrics reuse esca_analysis.prf / auc: object precision, recall and F1 against HookNet-TLS; window AUROC
(peak density, ambiguous windows dropped); AUROC TLS vs dense hard negatives; and specificity on the
`dense` hard-negative windows (HookNet-negative dense windows without any predicted object).
HookNet-TLS is the reference on ESCA, so it is not a comparison arm here (it is on Visium).

The cluster machinery (draws, paired_comparison) is shared with visium_paired_stats.py.
"""
from __future__ import annotations

import argparse
from itertools import combinations, product
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import esca_analysis as A  # noqa: E402
from esca_common import OUT, ROOT, save_json  # noqa: E402

N_RESAMPLES, N_PERMUTATIONS, SEED = 2000, 2000, 20261025
ADOPTED = OUT / "analysis_he2cell_ft_tau05"
# name -> (analysis folder, method in per_window_results.csv)
ARMS = {"B density": (ADOPTED, "B (HE2CELL)"),
        "Two-stage": (ADOPTED, A.TWO_STAGE),
        "Nuclei only (no classifier)": (ADOPTED, "All nuclei (no classifier)"),
        "Original checkpoint (B density)": (OUT / "analysis", "B (HE2CELL)")}
KEY = ["barcode", "window_id"]
METRICS = ["object_F1", "object_precision", "object_recall", "window_AUROC", "AUROC_vs_dense", "dense_specificity"]


# ---------------------------------------------------------------- cluster resampling (shared)
def draws(n_clusters, n=N_RESAMPLES, seed=SEED):
    """Bootstrap draws as cluster indices, shape (n, n_clusters); reused for every arm and pair."""
    return np.random.default_rng(seed).integers(0, n_clusters, (n, n_clusters))


def flips(n_clusters, n=N_PERMUTATIONS, seed=SEED + 1):
    """Sign-flip patterns (True = swap the arms in that cluster); exhaustive when 2^k <= n."""
    if 2 ** n_clusters <= n:
        return np.array(list(product([False, True], repeat=n_clusters)))
    return np.random.default_rng(seed).random((n, n_clusters)) < .5


def index_groups(bases, column, clusters, offset=0):
    """cluster -> tuple of row-index arrays, one per base frame (rows of that cluster)."""
    out = {c: [] for c in clusters}
    for base in bases:
        positions = pd.Series(np.arange(len(base)), index=base[column].to_numpy())
        for c in clusters:
            out[c].append(positions.loc[[c]].to_numpy() if c in positions.index else np.empty(0, int))
    return {c: tuple(v) for c, v in out.items()}


def combine(bases, parts):
    """Rows of the chosen clusters (with repeats) from each base frame."""
    return tuple(base.iloc[np.concatenate([p[i] for p in parts])] for i, base in enumerate(bases))


def paired_comparison(bases, groups_a, groups_b, metric, clusters, boot, perm):
    """bases: tuple of frames holding both arms' rows; groups_x: cluster -> tuple of row-index arrays of
    one arm into bases; metric: tuple of frames -> {name: value}.

    Returns {metric: {a, b, difference, CI95_a, CI95_b, CI95_difference, p_sign_flip}} with difference = a - b.
    """
    def evaluate(groups, ids):
        return metric(combine(bases, [groups[clusters[i]] for i in ids]))
    everyone = np.arange(len(clusters))
    a, b = evaluate(groups_a, everyone), evaluate(groups_b, everyone)
    boot_a, boot_b = [], []
    for ids in boot:
        boot_a.append(evaluate(groups_a, ids)); boot_b.append(evaluate(groups_b, ids))
    null = []
    for swap in perm:
        x = combine(bases, [(groups_b if s else groups_a)[c] for c, s in zip(clusters, swap)])
        y = combine(bases, [(groups_a if s else groups_b)[c] for c, s in zip(clusters, swap)])
        mx, my = metric(x), metric(y)
        null.append({k: mx[k] - my[k] for k in mx})
    out = {}
    for k in a:
        va, vb = np.array([r[k] for r in boot_a], float), np.array([r[k] for r in boot_b], float)
        d = a[k] - b[k]
        nd = np.array([r[k] for r in null], float)
        nd = nd[np.isfinite(nd)]
        exhaustive = len(perm) == 2 ** len(clusters)
        exceed = int((np.abs(nd) >= abs(d) - 1e-12).sum())
        p = exceed / len(nd) if exhaustive else (1 + exceed) / (1 + len(nd))
        out[k] = {"a": float(a[k]), "b": float(b[k]), "difference": float(d),
                  "CI95_a": np.nanpercentile(va, [2.5, 97.5]).tolist(), "CI95_b": np.nanpercentile(vb, [2.5, 97.5]).tolist(),
                  "CI95_difference": np.nanpercentile(va - vb, [2.5, 97.5]).tolist(),
                  "p_sign_flip": float(p) if len(nd) and np.isfinite(d) else float("nan"),
                  "permutations": "exhaustive" if exhaustive else len(nd)}
    return out


def excludes_zero(entry):
    low, high = entry["CI95_difference"]
    return bool(low > 0 or high < 0)


# ---------------------------------------------------------------- ESCA arms and metrics
def load_arm(folder, method, split="test"):
    frame = pd.read_csv(Path(folder) / "per_window_results.csv")
    frame = frame[(frame.method == method) & (frame.split == split)].copy()
    if frame.empty:
        raise ValueError(f"{method} has no {split} rows in {folder}")
    frame["rank_score"] = frame.max_density.rank(pct=True)
    return frame


def esca_metric(parts):
    """Metrics of one arm on a (resampled) set of test windows."""
    (frame,) = parts
    precision, recall, f1 = A.prf(frame)
    labelled = frame[frame.hooknet_label != "ambiguous"]
    hard = labelled[labelled.category.isin(["tls", "dense"])]
    dense = frame[(frame.category == "dense") & (frame.hooknet_label == "none")]
    return {"object_F1": f1, "object_precision": precision, "object_recall": recall,
            "window_AUROC": A.auc(labelled.hooknet_label == "TLS", labelled.rank_score),
            "AUROC_vs_dense": A.auc(hard.hooknet_label == "TLS", hard.rank_score),
            "dense_specificity": float((dense.n_pred == 0).mean()) if len(dense) else np.nan}


def bootstrap_arm(bases, groups, metric, clusters, boot):
    """Point estimate and percentile CI of one arm, on the shared draws."""
    values = metric(bases)
    samples = [metric(combine(bases, [groups[clusters[i]] for i in ids])) for ids in boot]
    return {k: {"value": float(v), "CI95": np.nanpercentile(np.array([s[k] for s in samples], float), [2.5, 97.5]).tolist()}
            for k, v in values.items()}


def esca_pair(frame_a, frame_b, clusters):
    """One base frame with both arms' rows on their common windows, and each arm's cluster index."""
    keys = frame_a[KEY].merge(frame_b[KEY], on=KEY)
    a, b = frame_a.merge(keys, on=KEY), frame_b.merge(keys, on=KEY)
    base = pd.concat([a, b], ignore_index=True)
    groups = index_groups((base,), "case_id", clusters)
    split = len(a)
    groups_a = {c: (g[0][g[0] < split],) for c, g in groups.items()}
    groups_b = {c: (g[0][g[0] >= split],) for c, g in groups.items()}
    return (base,), groups_a, groups_b, len(keys)


def compare_arms(arms, pairs, clusters=None):
    """arms: name -> per-window test frame. Returns per-arm estimates and per-pair comparisons."""
    clusters = clusters or sorted(set().union(*(set(f.case_id) for f in arms.values())))
    boot, perm = draws(len(clusters)), flips(len(clusters))
    per_arm = {}
    for name, frame in arms.items():
        bases = (frame.reset_index(drop=True),)
        per_arm[name] = {"windows": len(frame), "patients": int(frame.case_id.nunique()),
                         **bootstrap_arm(bases, index_groups(bases, "case_id", clusters), esca_metric, clusters, boot)}
    comparisons = []
    for a, b in pairs:
        bases, groups_a, groups_b, windows = esca_pair(arms[a], arms[b], clusters)
        result = paired_comparison(bases, groups_a, groups_b, esca_metric, clusters, boot, perm)
        comparisons.append({"a": a, "b": b, "windows": windows, "metrics": result})
        print(f"{a} - {b} ({windows} windows): " + ", ".join(
            f"{k} {r['difference']:+.3f} [{r['CI95_difference'][0]:+.3f}, {r['CI95_difference'][1]:+.3f}] p={r['p_sign_flip']:.3g}"
            for k, r in result.items()), flush=True)
    return {"clusters": len(clusters), "resamples": len(boot), "permutations": len(perm), "seed": SEED,
            "per_arm": per_arm, "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "outputs" / "ablations_20261025" / "esca_paired_stats.json"))
    args = parser.parse_args()
    arms = {name: load_arm(*spec) for name, spec in ARMS.items()}
    result = compare_arms(arms, list(combinations(arms, 2)))
    result["arms"] = {name: {"analysis": str(folder.relative_to(ROOT)), "method": method} for name, (folder, method) in ARMS.items()}
    result["design"] = __doc__
    save_json(result, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
