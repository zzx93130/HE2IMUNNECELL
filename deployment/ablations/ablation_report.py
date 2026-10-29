"""Item 3 of deployment/tls_validation_plan.md: paired method comparisons and ablations, one report.

Every arm's detector parameters were tuned on ESCA dev patients with the esca_analysis grid and frozen;
each arm is scored on the 85 ESCA test patients and on the 8 Visium sections. Paired differences use the
shared cluster machinery of esca_paired_stats / visium_paired_stats (same resamples for both arms,
sign-flip permutation p). Nothing is tuned on test patients or Visium labels.

Writes outputs/ablations_<date>/{Ablation_report.html, summary.json, figures/, visium_spot_tables/}.
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deployment" / "esca_tls"))
sys.path.insert(0, str(ROOT / "deployment" / "visium_tls"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import esca_analysis as A  # noqa: E402
import esca_paired_stats as P  # noqa: E402
import visium_paired_stats as VP  # noqa: E402
from esca_common import save_json  # noqa: E402
from esca_sigma_threshold import THRESHOLDS  # noqa: E402  (same grid as the ESCA threshold curve)

VE, V = VP.VE, VP.V
OUT = ROOT / "outputs" / "ablations_20261025"
FIG = OUT / "figures"
ESCA = A.OUT
PRIORS = OUT / "priors"
NUCLEI, B = "All nuclei (no classifier)", "B (HE2CELL)"
ADOPTED = "Adopted: B density"
DETECTOR = {B: ("B_density_at_spot", "cover_B density"), A.TWO_STAGE: ("cover_Two-stage", "cover_Two-stage"),
            NUCLEI: ("nuclei_density_at_spot", "cover_All nuclei (no classifier)")}


def analysis_params(folder):
    results = json.loads((folder / "detection_results.json").read_text())["results"]
    return {method: entry["dev"]["parameters"] for method, entry in results.items()}


def json_params(path):
    chosen = json.loads(path.read_text())["chosen_parameters"]
    return {**analysis_params(ESCA / "analysis_he2cell_ft_tau05"), **chosen}  # unused detectors keep adopted values


def prior(base, tau):
    return {f"{base}_{tau}": (base, PRIORS / f"prior_{tau}.json")}


# name -> group, ESCA (per-window folder, method), parameter source, Visium (classifier, priors) or "HookNet"
ARMS = {
    ADOPTED: ("method", ESCA / "analysis_he2cell_ft_tau05", B, None, ("finetuned_tau05", prior("finetuned", "tau05"))),
    "Two-stage (adopted classifier)": ("method", ESCA / "analysis_he2cell_ft_tau05", A.TWO_STAGE, None, ("finetuned_tau05", prior("finetuned", "tau05"))),
    "Nuclei only (no classifier)": ("method", ESCA / "analysis_he2cell_ft_tau05", NUCLEI, None, ("finetuned_tau05", prior("finetuned", "tau05"))),
    "Nuclei only, extended threshold grid": ("method", OUT / "esca_nuclei_extended", NUCLEI, OUT / "esca_nuclei_extended" / "chosen_parameters.json",
                                             ("finetuned_tau05", prior("finetuned", "tau05"))),
    "HookNet-TLS": ("method", None, None, None, "HookNet"),
    "Original checkpoint": ("classifier", ESCA / "analysis", B, None, ("original", {})),
    "Weak augmentation (control fine-tune)": ("augmentation", ESCA / "analysis_he2cell_ftweak_tau05", B, None,
                                              ("finetuned_weakaug_tau05", prior("finetuned_weakaug", "tau05"))),
    "No TTA (identity view)": ("TTA", ESCA / "analysis_he2cell_ft_ttanone_tau05", B, None, ("finetuned_ttanone_tau05", prior("finetuned_ttanone", "tau05"))),
    "Prior τ 0 (none)": ("prior", ESCA / "analysis_he2cell_ft", B, None, ("finetuned", {})),
    "Prior τ 1.0": ("prior", ESCA / "analysis_he2cell_ft_tau10", B, None, ("finetuned_tau10", prior("finetuned", "tau10"))),
    **{f"σ {s} µm": ("sigma", OUT / f"esca_sigma{s}", B, OUT / f"esca_sigma{s}" / "chosen_parameters.json",
                     ("finetuned_tau05", prior("finetuned", "tau05"))) for s in [10, 15, 25, 40]},
}
# (a, b): difference a - b. The decision rule uses ADOPTED - nuclei-only.
PAIRS = ([(ADOPTED, "Nuclei only (no classifier)"), (ADOPTED, "Nuclei only, extended threshold grid"),
          ("Two-stage (adopted classifier)", "Nuclei only (no classifier)"),
          ("Two-stage (adopted classifier)", "Nuclei only, extended threshold grid"), (ADOPTED, "Two-stage (adopted classifier)"),
          (ADOPTED, "HookNet-TLS"), ("Nuclei only (no classifier)", "HookNet-TLS")]
         + [(name, ADOPTED) for name, spec in ARMS.items() if spec[0] in {"classifier", "augmentation", "TTA", "prior", "sigma"}])


def visium_cells_dir():
    ablate = V.OUT / "cells_0.25um_ablate"
    return "cells_0.25um_ablate" if all((ablate / s / "cells.csv.gz").exists() for s in V.SAMPLES) else "cells_0.25um_v2"


def available(name):
    group, folder, method, params, visium = ARMS[name]
    esca_ok = folder is None or (folder / "per_window_results.csv").exists()
    if visium == "HookNet" or not esca_ok:
        return esca_ok
    cells = pd.read_csv(V.OUT / visium_cells_dir() / V.SAMPLES[0] / "cells.csv.gz", nrows=1).columns
    base = next(iter(visium[1].values()))[0] if visium[1] else visium[0]
    return f"prob_B_{base}" in cells or f"celltype_{base}" in cells


def arm_parameters(name):
    group, folder, method, params, visium = ARMS[name]
    if visium == "HookNet":
        return None
    return json_params(params) if params else analysis_params(folder)


def visium_arm(name, members, cells_dir):
    """(spots, objects) frames and per-section cell-type shares of one arm on Visium."""
    group, folder, method, params, visium = ARMS[name]
    if visium == "HookNet":
        table = pd.read_csv(VP.SPOT_TABLE)
        return VP.arm_frames(table[table.classifier == "finetuned_tau05"], "cover_HookNet-TLS", "cover_HookNet-TLS", members), None
    classifier, priors = visium
    cache = OUT / "visium_spot_tables" / f"{cells_dir}__{name.replace(' ', '_').replace('/', '_')}.csv.gz"
    shares_path = cache.with_name(cache.name.replace(".csv.gz", ".shares.json"))
    if cache.exists():
        table, shares = pd.read_csv(cache), json.loads(shares_path.read_text())
    else:
        table, shares = VP.arm_spot_table(classifier, arm_parameters(name), cells_dir, priors)
        cache.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(cache, index=False)
        shares_path.write_text(json.dumps(shares, indent=1))
    score, cover = DETECTOR[method]
    return VP.arm_frames(table, score, cover, members), shares


def visium_threshold_curve(cells_dir, members):
    """INFL-called-TLS rate, object recall and spot F1 vs threshold on Visium (pooled sections, no tuning)."""
    params = analysis_params(ESCA / "analysis_he2cell_ft_tau05")
    VE.CELLS_DIR = cells_dir
    VE.PRIORS.update(prior("finetuned", "tau05"))
    rows = []
    for sample in V.SAMPLES:
        x = VE.sample_inputs(sample, "finetuned_tau05")
        disks = VE.spot_disks(x["spots"], x["mpp"], x["shape"])
        for method, thresholds in THRESHOLDS.items():
            types = A.SCORES[method]
            for threshold in thresholds:
                mask, _ = A.detect(x, types, **{**params[method], "threshold": float(threshold)})
                cover = [float(mask[ys, xs].mean()) for ys, xs in disks]
                rows.append(pd.DataFrame({"sample": sample, "barcode": x["spots"].barcode, "label": x["spots"].label,
                                          "in_tissue": x["spots"].in_tissue, "cover": cover, "method": method,
                                          "threshold": float(threshold), "frozen_threshold": params[method]["threshold"]}))
    table = pd.concat(rows, ignore_index=True)
    out = []
    for (method, threshold), part in table.groupby(["method", "threshold"]):
        spots, objects = VP.arm_frames(part.assign(score=part.cover), "score", "cover", members)
        m = VP.visium_metric((spots, objects))
        out.append({"method": method, "threshold": threshold, "frozen_threshold": part.frozen_threshold.iat[0],
                    "INFL_called_TLS": m["INFL_called_TLS"], "object_recall": m["object_recall"], "spot_F1": m["spot_F1"]})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- report helpers
def png(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def fmt(v, d=3):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def ci(entry, d=3):
    if entry is None:
        return "–"
    low, high = entry["CI95"]
    return f"{fmt(entry['value'], d)} <span class=muted>[{fmt(low, d)}, {fmt(high, d)}]</span>"


def diff_cell(entry, d=3):
    if entry is None:
        return "–"
    low, high = entry["CI95_difference"]
    mark = " class=sig" if P.excludes_zero(entry) else ""
    return (f"<td{mark}>{entry['difference']:+.{d}f} <span class=muted>[{low:+.{d}f}, {high:+.{d}f}]</span>"
            f"<br><span class=muted>p = {entry['p_sign_flip']:.3g}</span></td>")


INK, MUTED, GRID, POS, NEG, NULL = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834", "#9a9890"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "figure.dpi": 150})
FOREST = [("ESCA", "object_F1", "ESCA object F1"), ("ESCA", "dense_specificity", "ESCA dense-window specificity"),
          ("ESCA", "object_recall", "ESCA object recall"), ("Visium", "AUROC_TLS_vs_INFL", "Visium AUROC TLS vs INFL"),
          ("Visium", "INFL_called_TLS", "Visium INFL called TLS"), ("Visium", "object_recall", "Visium object recall")]


def forest(comparisons, path):
    labels = [f"{c['a']}  −  {c['b']}" for c in comparisons]
    fig, axes = plt.subplots(1, len(FOREST), figsize=(15, .36 * len(labels) + 1.4), sharey=True)
    y = np.arange(len(labels))[::-1]
    for ax, (cohort, metric, title) in zip(axes, FOREST):
        for yi, c in zip(y, comparisons):
            entry = c.get(cohort, {}).get(metric) if c.get(cohort) else None
            if entry is None or not np.isfinite(entry["difference"]):
                continue
            low, high = entry["CI95_difference"]
            colour = (POS if low > 0 else NEG) if P.excludes_zero(entry) else NULL
            ax.plot([low, high], [yi, yi], color=colour, lw=1.6)
            ax.plot(entry["difference"], yi, "o", color=colour, ms=4)
        ax.axvline(0, color=INK, lw=.8)
        ax.set_title(title, fontsize=8.5)
        ax.tick_params(axis="x", labelsize=7)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=7.5)
    fig.text(.995, .005, "difference a − b with 95% cluster-bootstrap CI; blue = CI above 0, orange = CI below 0, grey = includes 0 (colour is the sign, not benefit: higher INFL-called-TLS is worse)", ha="right", fontsize=7, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def threshold_figure(esca_curve, visium_curve, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    colours = {B: POS, NUCLEI: "#1baf7a"}
    names = {B: "B density", NUCLEI: "Nuclei only"}
    for method, colour in colours.items():
        e = esca_curve[esca_curve.method == method]
        for split, style in [("dev", ":"), ("test", "-")]:
            part = e[e.split == split]
            axes[0].plot(part.threshold, part.object_F1, style, color=colour, label=f"{names[method]}, {split}")
            axes[1].plot(part.threshold, part.dense_specificity, style, color=colour, label=f"{names[method]}, {split}")
        if visium_curve is not None:
            v = visium_curve[visium_curve.method == method]
            axes[2].plot(v.threshold, v.INFL_called_TLS, "-", color=colour, label=f"{names[method]}: INFL called TLS")
            axes[2].plot(v.threshold, v.object_recall, "--", color=colour, label=f"{names[method]}: object recall")
        for ax in axes:
            ax.axvline(e.frozen_threshold.iat[0], color=colour, lw=.8, alpha=.6)
    for ax, title in zip(axes, ["ESCA object F1", "ESCA dense-window specificity", "Visium (pooled sections)"]):
        ax.set_xscale("log"); ax.set_title(title, fontsize=9); ax.set_xlabel("density threshold (cells/mm²)"); ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    cells_dir = visium_cells_dir()
    names = [n for n in ARMS if available(n)]
    missing = [n for n in ARMS if n not in names]
    print("arms:", names, "missing:", missing, "Visium cells:", cells_dir, flush=True)
    spot_table = pd.read_csv(VP.SPOT_TABLE)
    members = VP.tls_object_members(spot_table[spot_table.classifier == "original"])
    esca_arms, visium_arms, shares = {}, {}, {}
    for name in names:
        group, folder, method, params, visium = ARMS[name]
        if folder is not None:
            esca_arms[name] = P.load_arm(folder, method)
        visium_arms[name], shares[name] = visium_arm(name, members, cells_dir)
    dev_f1 = {name: float(A.prf(P.load_arm(ARMS[name][1], ARMS[name][2], "dev"))[2]) for name in esca_arms}
    pairs = [(a, b) for a, b in PAIRS if a in names and b in names]
    esca = P.compare_arms(esca_arms, [(a, b) for a, b in pairs if a in esca_arms and b in esca_arms])
    visium = VP.compare_arms(visium_arms, pairs)
    comparisons = []
    for a, b in pairs:
        e = next((c for c in esca["comparisons"] if (c["a"], c["b"]) == (a, b)), None)
        v = next(c for c in visium["comparisons"] if (c["a"], c["b"]) == (a, b))
        comparisons.append({"a": a, "b": b, "ESCA": e["metrics"] if e else None, "ESCA_windows": e["windows"] if e else None,
                            "Visium": v["metrics"]})
    other_share = {n: (float(np.mean([s.get("Other", 0) for s in shares[n].values()])) if shares[n] and ARMS[n][2] != NUCLEI else None)
                   for n in names}

    # Threshold curves
    esca_curve = pd.read_csv(OUT / "esca_threshold_curve.csv")
    curve_path = OUT / "visium_threshold_curve.csv"
    if not curve_path.exists():
        visium_threshold_curve(cells_dir, members).to_csv(curve_path, index=False)
    visium_curve = pd.read_csv(curve_path)
    forest(comparisons, FIG / "paired_differences.png")
    threshold_figure(esca_curve, visium_curve, FIG / "threshold_sensitivity.png")

    # Decision rule (plan section 3a)
    def rule(a, b):
        c = next((c for c in comparisons if (c["a"], c["b"]) == (a, b)), None)
        if c is None or c["ESCA"] is None:
            return None
        dense, infl = c["ESCA"]["dense_specificity"], c["Visium"]["AUROC_TLS_vs_INFL"]
        recall_esca, recall_visium = c["ESCA"]["object_recall"], c["Visium"]["object_recall"]
        criteria = {"ESCA dense-window specificity higher, CI excludes 0": dense["CI95_difference"][0] > 0,
                    "Visium AUROC TLS vs INFL higher, CI excludes 0": infl["CI95_difference"][0] > 0,
                    "ESCA object recall not lost (CI upper bound >= 0)": recall_esca["CI95_difference"][1] >= 0,
                    "Visium object recall not lost (CI upper bound >= 0)": recall_visium["CI95_difference"][1] >= 0}
        return {"comparison": f"{a} − {b}", "criteria": criteria, "classifier_adds": all(criteria.values()),
                "permutation_p": {"ESCA dense specificity": dense["p_sign_flip"], "Visium AUROC vs INFL": infl["p_sign_flip"],
                                  "ESCA recall": recall_esca["p_sign_flip"], "Visium recall": recall_visium["p_sign_flip"]}}
    decisions = [d for d in (rule(a, b) for a, b in [(ADOPTED, "Nuclei only (no classifier)"), (ADOPTED, "Nuclei only, extended threshold grid"),
                                                     ("Two-stage (adopted classifier)", "Nuclei only (no classifier)"),
                                                     ("Two-stage (adopted classifier)", "Nuclei only, extended threshold grid")]) if d]
    excluded = [{"comparison": f"{c['a']} − {c['b']}", "cohort": cohort, "metric": metric,
                 "difference": entry["difference"], "CI95": entry["CI95_difference"], "p_sign_flip": entry["p_sign_flip"]}
                for c in comparisons for cohort in ["ESCA", "Visium"] if c[cohort]
                for metric, entry in c[cohort].items() if P.excludes_zero(entry)]
    paired_3a = {"ESCA": json.loads((OUT / "esca_paired_stats.json").read_text()),
                 "Visium": json.loads((OUT / "visium_paired_stats.json").read_text())}
    for cohort in paired_3a.values():
        cohort.pop("design", None)
    sigma_edges = {f"σ {s} µm": json.loads((OUT / f"esca_sigma{s}" / "chosen_parameters.json").read_text())["chosen_value_at_grid_edge"]
                   for s in [10, 15, 25, 40] if (OUT / f"esca_sigma{s}" / "chosen_parameters.json").exists()}
    summary = {"design": __doc__, "visium_cells": cells_dir, "missing_arms": missing,
               "arms": {n: {"group": ARMS[n][0], "esca_source": str(ARMS[n][1].relative_to(ROOT)) if ARMS[n][1] else None,
                            "detector": ARMS[n][2], "parameters": arm_parameters(n),
                            "visium": ARMS[n][4] if ARMS[n][4] == "HookNet" else {"classifier": ARMS[n][4][0],
                                                                                  "prior": {k: str(v[1].relative_to(ROOT)) for k, v in ARMS[n][4][1].items()}},
                            "ESCA_dev_object_F1": dev_f1.get(n), "ESCA": esca["per_arm"].get(n), "Visium": visium["per_arm"][n], "Visium_Other_share_mean_of_sections": other_share[n],
                            "Visium_cell_type_shares": shares[n]} for n in names},
               "comparisons": comparisons, "decision_rule": decisions, "differences_excluding_zero": excluded,
               "grid_edge_flags": sigma_edges, "paired_3a": paired_3a,
               "resampling": {"ESCA": {"clusters": esca["clusters"], "resamples": esca["resamples"], "permutations": esca["permutations"]},
                              "Visium": {"clusters": visium["clusters"], "resamples": visium["resamples"], "permutations": visium["permutations"]},
                              "seed": P.SEED}}
    save_json(summary, OUT / "summary.json")
    write_html(summary, esca_curve, visium_curve)


def write_html(s, esca_curve, visium_curve):
    arms = s["arms"]
    groups = {"method": "Methods", "classifier": "Classifier", "augmentation": "Augmentation", "TTA": "Test-time augmentation",
              "prior": "Prior correction", "sigma": "Smoothing σ (dev-tuned threshold per σ)"}
    rows = ""
    for group, title in groups.items():
        members = [n for n, a in arms.items() if a["group"] == group]
        if not members:
            continue
        rows += f"<tr class=group><td colspan=8>{title}</td></tr>"
        for n in members:
            a = arms[n]
            e, v = a["ESCA"], a["Visium"]
            rows += (f"<tr><td>{html.escape(n)}</td><td>{ci(e['object_F1']) if e else 'reference'}</td>"
                     f"<td>{ci(e['dense_specificity']) if e else '–'}</td><td>{ci(e['object_recall']) if e else '–'}</td>"
                     f"<td>{ci(v['AUROC_TLS_vs_INFL'])}</td><td>{ci(v['INFL_called_TLS'])}</td><td>{ci(v['object_recall'])}</td>"
                     f"<td>{fmt(a['Visium_Other_share_mean_of_sections'])}</td></tr>")
    secondary = "".join(
        f"<tr><td>{html.escape(n)}</td><td>{fmt(a['ESCA_dev_object_F1'])}</td><td>{ci(a['ESCA']['object_precision']) if a['ESCA'] else '–'}</td>"
        f"<td>{ci(a['ESCA']['window_AUROC']) if a['ESCA'] else '–'}</td><td>{ci(a['ESCA']['AUROC_vs_dense']) if a['ESCA'] else '–'}</td>"
        f"<td>{ci(a['Visium']['AUROC_TLS_vs_all'])}</td><td>{ci(a['Visium']['spot_F1'])}</td>"
        f"<td>{html.escape(json.dumps(a['parameters'].get(a['detector'] or '', {})) if a['parameters'] else '–')}</td></tr>"
        for n, a in arms.items())
    diff_rows = "".join(
        f"<tr><td>{html.escape(c['a'])} − {html.escape(c['b'])}</td>"
        + "".join(diff_cell(c[cohort][metric]) if c[cohort] else "<td>–</td>" for cohort, metric, _ in FOREST) + "</tr>"
        for c in s["comparisons"])

    def pair_rows(cohort, metrics):
        out = ""
        for c in s["paired_3a"][cohort]["comparisons"]:
            out += f"<tr><td>{html.escape(c['a'])} − {html.escape(c['b'])}</td>" + "".join(diff_cell(c["metrics"][m]) for m in metrics) + "</tr>"
        return out
    esca_metrics = ["object_F1", "object_precision", "object_recall", "window_AUROC", "AUROC_vs_dense", "dense_specificity"]
    visium_metrics = ["AUROC_TLS_vs_all", "AUROC_TLS_vs_INFL", "object_recall", "INFL_called_TLS", "spot_F1"]
    decision_html = ""
    for d in s["decision_rule"]:
        items = "".join(f"<li>{'✔' if ok else '✘'} {html.escape(k)}</li>" for k, ok in d["criteria"].items())
        p = ", ".join(f"{k} p = {v:.3g}" for k, v in d["permutation_p"].items())
        decision_html += (f"<div class=card><b>{html.escape(d['comparison'])}: "
                          f"{'classifier adds (all criteria met)' if d['classifier_adds'] else 'rule NOT met'}</b><ul>{items}</ul>"
                          f"<p class=muted>Sign-flip permutation: {p}</p></div>")
    excluded = "".join(f"<li>{html.escape(e['comparison'])} — {e['cohort']} {e['metric']}: {e['difference']:+.3f} "
                       f"[{e['CI95'][0]:+.3f}, {e['CI95'][1]:+.3f}], p = {e['p_sign_flip']:.3g}</li>" for e in s["differences_excluding_zero"])
    missing = (f"<p class=warn>Not yet available and therefore not in this report: {html.escape(', '.join(s['missing_arms']))}.</p>"
               if s["missing_arms"] else "")
    edges = "".join(f"<li>{k}: " + html.escape(", ".join(f"{m.split(' (')[0]} {p}" for m, f in v.items() for p, at in f.items() if at)) + "</li>"
                    for k, v in s["grid_edge_flags"].items() if any(any(f.values()) for f in v.values()))
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TLS Ablation Report</title><style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--line:#e4e3df;--card:#fff;--pos:#1d5fae;--warn:#a04a12}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--line:#3a3936;--card:#232321;--pos:#7fb2f0;--warn:#f0a070}}}}
:root[data-theme="dark"]{{--bg:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--line:#3a3936;--card:#232321;--pos:#7fb2f0;--warn:#f0a070}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:28px 16px 64px}} h1{{font-size:26px;margin:0 0 4px}} h2{{margin-top:36px;border-top:1px solid var(--line);padding-top:18px}}
.muted{{color:var(--muted)}} table{{border-collapse:collapse;width:100%;font-size:12.5px;display:block;overflow-x:auto}}
th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;white-space:nowrap;vertical-align:top}} img.fig{{max-width:100%;background:#fff;border-radius:8px;border:1px solid var(--line)}}
tr.group td{{font-weight:600;background:var(--card);padding-top:12px}} td.sig{{color:var(--pos);font-weight:600}} .warn{{color:var(--warn)}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 14px;margin:10px 0}} .card ul{{margin:6px 0}}
</style></head><body><main>
<h1>TLS detection: paired comparisons and ablations</h1>
<p class="muted">Item 3 of the validation plan. ESCA: 85 held-out TCGA-ESCA test patients, reference HookNet-TLS (published filter). Visium: 8 USZ sections, reference pathologist spot labels.
Every arm was tuned on ESCA dev patients with the same grid, frozen, then scored once; nothing is tuned on test patients or Visium labels.</p>
{missing}
<h2>Statistics</h2>
<p>Paired differences (a − b) use a cluster bootstrap with the <i>same</i> 2000 resamples for both arms (patients on ESCA, sections on Visium; 95% percentile intervals) and a two-sided sign-flip permutation test that swaps the two arms within whole clusters
(ESCA: 2000 random flips over 85 patients; Visium: all 2<sup>8</sup> = 256 flips, so the smallest attainable p is 0.0078). Pairs are scored on the windows both arms have (the original checkpoint has 517 of the 580 test windows, 76 patients).
With 8 Visium sections the percentile bootstrap is optimistic; where the bootstrap CI excludes 0 but the permutation p is above 0.05, treat the Visium difference as suggestive.</p>

<h2>Decision rule</h2>
<p>Stated in advance in the plan: the classifier <b>adds</b> if B density beats nuclei-only on hard negatives — ESCA dense-window specificity and Visium TLS-vs-INFL AUROC, each with CI excluding 0 — without losing recall.
"Without losing recall" is operationalised here as the 95% CI of the recall difference (ESCA object recall; Visium object recall) not lying entirely below 0. The rule is checked against the frozen nuclei-only baseline and against a nuclei-only baseline re-tuned with an extended threshold grid (its adopted threshold sat at the top of the original grid).</p>
{decision_html}

<h2>Headline table</h2>
<p>B density detector unless the row says otherwise. Visium Other share is the mean over sections of the fraction of nuclei called Other (n/a for detectors without a classifier).</p>
<table><tr><th>Arm</th><th>ESCA object F1 [CI]</th><th>ESCA dense-window specificity</th><th>ESCA object recall</th><th>Visium AUROC TLS vs INFL</th><th>Visium INFL called TLS</th><th>Visium object recall</th><th>Visium Other share</th></tr>{rows}</table>

<h2>Paired differences</h2>
<p>Colour marks the sign of a difference whose CI excludes 0 (blue above, orange below), not whether it is better: for "INFL called TLS" a positive difference means more inflammation called TLS. Detection rates (INFL called, recall) are single operating points of each arm's frozen threshold; the threshold-free comparison is the AUROC.</p>
<img class="fig" src="{png(FIG / 'paired_differences.png')}" alt="Forest plot of paired differences">
<table><tr><th>a − b</th>{''.join(f'<th>{t}</th>' for _, _, t in FOREST)}</tr>{diff_rows}</table>
<p>Differences whose 95% CI excludes 0 (bold blue in the table):</p><ul>{excluded}</ul>

<h2>All pairs of methods (section 3a)</h2>
<h3>ESCA test patients</h3>
<table><tr><th>a − b</th>{''.join(f'<th>{m}</th>' for m in esca_metrics)}</tr>{pair_rows('ESCA', esca_metrics)}</table>
<p class="muted">HookNet-TLS is the ESCA reference, so it is compared only on Visium. ESCA rows of the original checkpoint use its own dev-tuned run (analysis/).</p>
<h3>Visium sections</h3>
<table><tr><th>a − b</th>{''.join(f'<th>{m}</th>' for m in visium_metrics)}</tr>{pair_rows('Visium', visium_metrics)}</table>
<p class="muted">Spot scores: B density = smoothed predicted-B density over the spot; nuclei = smoothed nuclear density; two-stage and HookNet = fraction of the spot covered by detections (a near-binary score, so their AUROCs are not directly comparable with the density scores).</p>

<h2>Secondary metrics and frozen parameters</h2>
<p>ESCA dev F1 is the tuning-set value: the arm a dev-only selection would prefer. Test and Visium values were never used to choose between arms.</p>
<table><tr><th>Arm</th><th>ESCA dev F1</th><th>ESCA test precision</th><th>ESCA window AUROC</th><th>ESCA AUROC vs dense</th><th>Visium AUROC TLS vs all</th><th>Visium spot F1</th><th>Frozen detector parameters</th></tr>{secondary}</table>
{f'<p class=warn>Dev-tuned values at the edge of the existing grid (true optimum may lie outside):</p><ul>{edges}</ul>' if edges else ''}

<h2>Threshold sensitivity</h2>
<p>Every other adopted parameter frozen; vertical lines mark the frozen thresholds. ESCA dotted = dev, solid = test. Visium curves pool all sections.</p>
<img class="fig" src="{png(FIG / 'threshold_sensitivity.png')}" alt="Threshold sensitivity curves">

<h2>Provenance</h2>
<p class="muted">Visium cells: outputs/visium_tls_validation/{html.escape(s['visium_cells'])}. ESCA arms: per-window results under outputs/esca_tls_20261019/analysis_* and outputs/ablations_20261025/esca_*. Code: deployment/esca_tls/esca_paired_stats.py, deployment/visium_tls/visium_paired_stats.py, deployment/ablations/. Numbers: summary.json next to this report.</p>
</main></body></html>"""
    (OUT / "Ablation_report.html").write_text(page)
    print("wrote", OUT / "Ablation_report.html")


if __name__ == "__main__":
    main()
