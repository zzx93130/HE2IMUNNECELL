"""Paired method comparisons on the 8 USZ Visium sections against pathologist spot labels.

Per pair of methods: difference of each metric with a section-cluster bootstrap CI (the same 2000
resamples of the 8 sections for both arms) and a two-sided sign-flip permutation p-value enumerated
exhaustively over all 2^8 = 256 section swaps (smallest attainable p = 2/256 = 0.0078).
Metrics (visium_eval definitions): spot AUROC TLS vs all negatives and TLS vs INFL, object recall of
pathologist TLS groups (detected when any member spot is >= 50% covered), INFL-called-TLS rate
(INFL spots >= 50% covered), and spot F1 at 50% coverage. Spot scores are rank-normalised within each
arm before a swap. Arms in the published spot table use the frozen ESCA dev parameters; nothing here
is tuned on Visium labels.

arm_spot_table() rebuilds the same spot table for any classifier / prior / parameter set, for the
ablations (deployment/ablations/ablation_report.py).
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "esca_tls"))
import visium_eval as VE  # noqa: E402
import esca_analysis as A  # noqa: E402
import esca_paired_stats as P  # noqa: E402

V = VE.V
SPOT_TABLE = V.OUT / "analysis" / "spot_table.csv.gz"
# name -> (classifier rows in the spot table, spot score column, detection coverage column)
ARMS = {"B density": ("finetuned_tau05", "B_density_at_spot", "cover_B density"),
        "Two-stage": ("finetuned_tau05", "cover_Two-stage", "cover_Two-stage"),
        "Nuclei only (no classifier)": ("finetuned_tau05", "nuclei_density_at_spot", "cover_All nuclei (no classifier)"),
        "HookNet-TLS": ("finetuned_tau05", "cover_HookNet-TLS", "cover_HookNet-TLS"),
        "Original checkpoint (B density)": ("original", "B_density_at_spot", "cover_B density")}
METRICS = ["AUROC_TLS_vs_all", "AUROC_TLS_vs_INFL", "object_recall", "INFL_called_TLS", "spot_F1"]


def tls_object_members(spots):
    """(sample, barcode) -> pathologist TLS object id, from labels only (identical for every arm)."""
    rows = []
    for sample, part in spots[spots.in_tissue == 1].groupby("sample"):
        part = part.reset_index(drop=True)
        groups = VE.tls_objects(part, 100 / V.native_mpp(sample))
        rows.append(pd.DataFrame({"sample": sample, "barcode": part.loc[groups.index, "barcode"].to_numpy(),
                                  "object": [f"{sample}:{g}" for g in groups.to_numpy()]}))
    return pd.concat(rows, ignore_index=True)


def arm_frames(table, score, cover, members):
    """(scored spots, TLS objects) of one arm from a spot table restricted to that arm's rows."""
    tissue = table[table.in_tissue == 1]
    spots = tissue[tissue.label.isin(VE.POSITIVE | VE.NEGATIVE)][["sample", "barcode", "label"]].copy()
    values = tissue.set_index(["sample", "barcode"])
    key = pd.MultiIndex.from_frame(spots[["sample", "barcode"]])
    spots["score"] = values.loc[key, score].fillna(0).to_numpy()
    spots["rank_score"] = spots.score.rank(pct=True)
    spots["cover"] = values.loc[key, cover].to_numpy()
    detected = members.assign(hit=values.loc[pd.MultiIndex.from_frame(members[["sample", "barcode"]]), cover].to_numpy() >= .5)
    objects = detected.groupby(["sample", "object"], as_index=False).hit.any()
    return spots.reset_index(drop=True), objects


def visium_metric(parts):
    spots, objects = parts
    infl = spots[spots.label.isin(VE.POSITIVE | {"INFL"})]
    return {"AUROC_TLS_vs_all": A.auc(spots.label.isin(VE.POSITIVE), spots.rank_score),
            "AUROC_TLS_vs_INFL": A.auc(infl.label.isin(VE.POSITIVE), infl.rank_score),
            "object_recall": float(objects.hit.mean()) if len(objects) else np.nan,
            "INFL_called_TLS": float((spots[spots.label == "INFL"].cover >= .5).mean()),
            "spot_F1": VE.binary(spots, "cover")["F1"]}


def visium_pair(frames_a, frames_b, clusters):
    """Shared base frames holding both arms and each arm's per-section row indices."""
    bases = tuple(pd.concat([x, y], ignore_index=True) for x, y in zip(frames_a, frames_b))
    groups = P.index_groups(bases, "sample", clusters)
    sizes = [len(x) for x in frames_a]
    pick = lambda first: {c: tuple(g[i][(g[i] < sizes[i]) == first] for i in range(len(bases))) for c, g in groups.items()}
    return bases, pick(True), pick(False)


def compare_arms(arms, pairs):
    """arms: name -> (spots, objects) frames. Per-arm estimates and pairwise comparisons."""
    clusters = sorted(set().union(*(set(s["sample"]) for s, _ in arms.values())))
    boot, perm = P.draws(len(clusters)), P.flips(len(clusters))
    per_arm = {}
    for name, frames in arms.items():
        frames = tuple(f.reset_index(drop=True) for f in frames)
        per_arm[name] = {"spots": len(frames[0]), "TLS_objects": len(frames[1]),
                         **P.bootstrap_arm(frames, P.index_groups(frames, "sample", clusters), visium_metric, clusters, boot)}
    comparisons = []
    for a, b in pairs:
        bases, groups_a, groups_b = visium_pair(arms[a], arms[b], clusters)
        result = P.paired_comparison(bases, groups_a, groups_b, visium_metric, clusters, boot, perm)
        comparisons.append({"a": a, "b": b, "metrics": result})
        print(f"{a} - {b}: " + ", ".join(
            f"{k} {r['difference']:+.3f} [{r['CI95_difference'][0]:+.3f}, {r['CI95_difference'][1]:+.3f}] p={r['p_sign_flip']:.3g}"
            for k, r in result.items()), flush=True)
    return {"clusters": len(clusters), "resamples": len(boot), "permutations": f"exhaustive ({len(perm)})",
            "seed": P.SEED, "per_arm": per_arm, "comparisons": comparisons}


def arm_spot_table(classifier, params, cells_dir="cells_0.25um_v2", priors=None):
    """Spot table (visium_eval columns) for one classifier and one frozen ESCA parameter set.

    classifier: a celltype_<name> column of the cells table, or a key of priors -> (base, prior json).
    The spot scores use each detector's own frozen sigma. Also returns each section's cell-type shares.
    """
    VE.CELLS_DIR = cells_dir
    VE.PRIORS.update(priors or {})
    rows, shares = [], {}
    for sample in V.SAMPLES:
        x = VE.sample_inputs(sample, classifier)
        spots, mpp, shape = x["spots"], x["mpp"], x["shape"]
        disks = VE.spot_disks(spots, mpp, shape)
        masks = VE.detectors(x, params)
        total = sum(x["counts"].values())
        b_density = A.density(x["counts"], ["B"], params["B (HE2CELL)"]["sigma_um"])
        nuclei_density = A.density(x["counts"], None, params["All nuclei (no classifier)"]["sigma_um"])
        shares[sample] = x["cells"][f"celltype_{classifier}"].value_counts(normalize=True).to_dict()
        for i, s in enumerate(spots.itertuples()):
            ys, xs = disks[i]
            row = {"sample": sample, "classifier": classifier, "barcode": s.barcode, "label": s.label, "in_tissue": s.in_tissue,
                   "cells": total[ys, xs].sum(), "B": x["counts"]["B"][ys, xs].sum(),
                   "B_density_at_spot": float(b_density[ys, xs].mean()), "nuclei_density_at_spot": float(nuclei_density[ys, xs].mean())}
            for name, mask in masks.items():
                row[f"cover_{name}"] = float(mask[ys, xs].mean())
            rows.append(row)
    return pd.DataFrame(rows), shares


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(V.ROOT / "outputs" / "ablations_20261025" / "visium_paired_stats.json"))
    args = parser.parse_args()
    table = pd.read_csv(SPOT_TABLE)
    members = tls_object_members(table[table.classifier == "original"])
    arms = {name: arm_frames(table[table.classifier == rows], score, cover, members) for name, (rows, score, cover) in ARMS.items()}
    result = compare_arms(arms, list(combinations(arms, 2)))
    result["arms"] = {name: {"spot_table": str(SPOT_TABLE.relative_to(V.ROOT)), "classifier_rows": rows, "score": score, "coverage": cover}
                      for name, (rows, score, cover) in ARMS.items()}
    result["design"] = __doc__
    Path(args.out).write_text(json.dumps(result, indent=1, default=float))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
