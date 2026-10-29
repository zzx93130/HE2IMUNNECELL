"""HTML report for the 4-fold patient-level cross-validation of the classifier recipe (reads summary.json and
per_patient_metrics.csv written by classifier_cv.py metrics, and the fold training histories)."""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import classifier_cv as C  # noqa: E402

OUT, FIG, NAMES, ROOT = C.OUT, C.OUT / "figures", C.NAMES, C.ROOT
COL = {"CV1": "#2a78d6", "CV2": "#eb6834", "A1": "#8a8984", "A2": "#c3c2b7", "A3": "#1baf7a"}
FOLD_COL = {"fold0": "#2a78d6", "fold1": "#eb6834", "fold2": "#1baf7a", "fold3": "#a35bd6"}
SHORT = {"CV1": "CV held-out + prior (recipe)", "CV2": "CV held-out, no prior", "A1": "A1 deployed + prior (training set)",
         "A2": "A2 deployed, no prior (training set)", "A3": "A3 original (training set)"}
SEV = {"clean": 0.0, "sev1": 0.5, "sev2": 1.0, "sev3": 1.5}
SEV_LABEL = {"clean": "clean", "sev1": "λ 0.5 (mild)", "sev2": "λ 1.0 (training range)", "sev3": "λ 1.5 (beyond training)"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "figure.dpi": 130})


def png(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def fmt(v, d=3):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def vci(m, d=3):
    lo, hi = m["CI95"]
    return f"{fmt(m['value'], d)} <span class=muted>[{fmt(lo, d)}–{fmt(hi, d)}]</span>"


def dcell(m, d=3, key="difference"):
    lo, hi = m["CI95"]
    sig = lo is not None and (lo > 0 or hi < 0)
    p = m.get("p_bootstrap")
    ptxt = "" if p is None else (" p&lt;0.001" if p == 0 else f" p={p:.3f}")
    return f"<td class='{'sig' if sig else ''}'>{m[key]:+.{d}f} <span class=muted>[{lo:+.{d}f}, {hi:+.{d}f}]{ptxt}</span></td>"


def fig_training(s):
    tr = s["training"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, t in tr.items():
        h = [r for r in t["history"] if r["epoch"] > 0]
        ep = [r["epoch"] for r in h]
        axes[0].plot(ep, [r["holdout_macro_F1"] for r in h], color=FOLD_COL[name], marker="o", ms=4, lw=1.8, label=name)
        axes[1].plot(ep, [r.get("ESCA_dev_AUROC_PB_TLS_vs_dense", np.nan) for r in h], color=FOLD_COL[name], marker="o", ms=4, lw=1.8, label=name)
        for ax, key in zip(axes, ["holdout_macro_F1", "ESCA_dev_AUROC_PB_TLS_vs_dense"]):
            sel = next(r for r in h if r["epoch"] == t["selected_epoch"])
            ax.scatter([sel["epoch"]], [sel.get(key, np.nan)], s=120, facecolor="none", edgecolor=FOLD_COL[name], lw=1.8, zorder=5)
    axes[0].set_title("Selection-patient macro-F1 by epoch", fontsize=10); axes[0].set_xlabel("Epoch")
    axes[1].set_title("ESCA-dev AUROC P(B), TLS vs dense, by epoch", fontsize=10); axes[1].set_xlabel("Epoch")
    axes[0].legend(fontsize=7.5, frameon=False, title="circle = selected epoch", title_fontsize=7.5)
    fig.tight_layout(); fig.savefig(FIG / "training_curves.png", facecolor="white"); plt.close(fig)


def fig_per_patient(pp):
    metrics = [("F1_B", "B F1"), ("F1_T", "T F1"), ("macro_F1", "Macro-F1"), ("AUROC_B_vs_T", "AUROC P(B), B vs T")]
    clean = pp[(pp.condition == "clean")]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6))
    for ax, (m, title) in zip(axes, metrics):
        wide = clean.pivot(index="patient", columns="arm", values=m)
        folds = clean[clean.arm == "CV1"].set_index("patient").fold
        for q, row in wide.iterrows():
            ax.plot([0, 1], row[["CV1", "CV2"]].to_numpy(), color=GRID, lw=0.8, zorder=1)
        for i, arm in enumerate(["CV1", "CV2"]):
            jitter = np.random.default_rng(i).uniform(-.08, .08, len(wide))
            ax.scatter(np.full(len(wide), i) + jitter, wide[arm], s=16, c=[FOLD_COL[folds[q]] for q in wide.index], zorder=2,
                       edgecolor="white", linewidth=0.5)
            ax.plot([i - .25, i + .25], [wide[arm].median()] * 2, color=INK, lw=2, zorder=3)
        ax.set_xticks([0, 1], ["CV1", "CV2"]); ax.set_title(title, fontsize=10)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f"{f} ({C.FOLDS[f][0]}–{C.FOLDS[f][-1]})") for f, c in FOLD_COL.items()]
    axes[-1].legend(handles=handles, fontsize=7, frameon=False, loc="lower right")
    fig.suptitle("Per held-out patient (40 Immunocto patients, each scored by the model that never saw it; black bar = median)", fontsize=9, color=MUTED)
    fig.tight_layout(); fig.savefig(FIG / "per_patient.png", facecolor="white"); plt.close(fig)


def fig_confusion(s):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    for ax, arm in zip(axes, C.ARMS):
        cm = np.array(s["results"][arm]["clean"]["confusion_matrix_true_rows_pred_cols"], float)
        rn = cm / cm.sum(1, keepdims=True)
        ax.imshow(rn, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{rn[i, j]:.2f}\n{int(cm[i, j]):,}", ha="center", va="center", fontsize=7.5, color="white" if rn[i, j] > .55 else INK)
        ax.set_xticks(range(4), NAMES, rotation=30); ax.set_yticks(range(4), NAMES); ax.set_xlabel("Predicted"); ax.set_title(SHORT[arm], fontsize=10)
    axes[0].set_ylabel("IF label (true)")
    fig.tight_layout(); fig.savefig(FIG / "confusion_matrices.png", facecolor="white"); plt.close(fig)


def fig_reliability(s):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, n in zip(axes, ["B", "T"]):
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="perfect calibration")
        for arm in C.ARMS:
            r = s["reliability_pooled"][arm][n]
            x, y = np.array(r["mean_predicted"], float), np.array(r["observed_fraction"], float)
            lo = np.array([c[0] if c[0] is not None else np.nan for c in r["observed_CI95"]], float)
            hi = np.array([c[1] if c[1] is not None else np.nan for c in r["observed_CI95"]], float)
            ok = np.array(r["count"]) > 0
            ece = s["results"][arm]["clean"]["pooled"][f"ECE_{n}"]
            ax.errorbar(x[ok], y[ok], yerr=[y[ok] - lo[ok], hi[ok] - y[ok]], color=COL[arm], lw=2, marker="o", ms=5, capsize=2,
                        label=f"{SHORT[arm]} (ECE {ece['value']:.3f})")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel(f"Predicted P({n}) (bin mean)"); ax.set_ylabel(f"Observed fraction IF-{n}")
        ax.set_title(f"Reliability, P({n}), held-out cells pooled", fontsize=10)
        ax.legend(fontsize=7.5, loc="upper left" if n == "B" else "lower right", frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "reliability.png", facecolor="white"); plt.close(fig)


def fig_robustness(s):
    metrics = [("macro_F1", "Macro-F1"), ("recall_B", "B recall"), ("F1_B", "B F1"), ("AUROC_B_vs_T", "AUROC P(B), B vs T")]
    R40, R30 = s["results"], s["comparison_CRC11_40"]["results"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
    for ax, (m, title) in zip(axes, metrics):
        for arm in C.ARMS:
            conds = [c for c in SEV if c in R40[arm]]
            x = [SEV[c] for c in conds]
            v = [R40[arm][c]["pooled"][m]["value"] for c in conds]
            lo = [R40[arm][c]["pooled"][m]["CI95"][0] for c in conds]; hi = [R40[arm][c]["pooled"][m]["CI95"][1] for c in conds]
            ax.fill_between(x, lo, hi, color=COL[arm], alpha=.15, lw=0)
            ax.plot(x, v, color=COL[arm], lw=2, marker="o", ms=5, label=f"{SHORT[arm]} (40 pts)")
        for arm in ["A1", "A3"]:
            if arm in R30:
                conds = [c for c in SEV if c in R30[arm]]
                ax.plot([SEV[c] for c in conds], [R30[arm][c]["pooled"][m]["value"] for c in conds], color=COL[arm], lw=1.4, ls="--", marker="s", ms=3.5,
                        label=f"{SHORT[arm]}, CRC11–40")
        ax.set_xticks([0, .5, 1, 1.5], ["clean", "0.5", "1.0", "1.5"]); ax.set_xlabel("Perturbation severity λ"); ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "robustness.png", facecolor="white"); plt.close(fig)


def fig_prevalence(s):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    w = .25
    true = [s["results"]["CV1"]["clean"]["pooled"][f"true_prevalence_{n}"]["value"] for n in NAMES]
    ax.bar(np.arange(4) - w, true, w - .02, color="#8a8984", label="IF labels (truth)")
    for k, arm in enumerate(C.ARMS):
        r = s["results"][arm]["clean"]["pooled"]
        v = [r[f"predicted_prevalence_{n}"]["value"] for n in NAMES]
        lo = [v[i] - r[f"predicted_prevalence_{n}"]["CI95"][0] for i, n in enumerate(NAMES)]
        hi = [r[f"predicted_prevalence_{n}"]["CI95"][1] - v[i] for i, n in enumerate(NAMES)]
        ax.bar(np.arange(4) + k * w, v, w - .02, color=COL[arm], yerr=[lo, hi], capsize=2, label=SHORT[arm], error_kw={"ecolor": MUTED, "lw": 0.8})
    ax.set_xticks(range(4), NAMES); ax.set_ylabel("Share of evaluated cells"); ax.legend(fontsize=7.5, frameon=False)
    ax.set_title("Class prevalence: IF truth vs predicted (40 held-out patients pooled, clean)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "prevalence.png", facecolor="white"); plt.close(fig)


def fig_cv_vs_a1(pp):
    """Same 30 patients: CV1 (held out) against A1 (training set), per patient."""
    a1 = pd.read_csv(C.A1_OUT / "per_patient_metrics.csv")
    a1 = a1[(a1.arm == "A1") & (a1.condition == "clean")].set_index("patient")
    cv = pp[(pp.arm == "CV1") & (pp.condition == "clean")].set_index("patient")
    common = [q for q in C.E.PATIENTS if q in cv.index and q in a1.index]
    metrics = [("F1_B", "B F1"), ("F1_T", "T F1"), ("macro_F1", "Macro-F1"), ("AUROC_B_vs_T", "AUROC P(B), B vs T")]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    for ax, (m, title) in zip(axes, metrics):
        x, y = a1.loc[common, m].to_numpy(), cv.loc[common, m].to_numpy()
        lo = np.nanmin(np.concatenate([x, y])) - .03; hi = 1.0
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--")
        ax.scatter(x, y, s=18, c=[FOLD_COL[cv.loc[q, "fold"]] for q in common], edgecolor="white", linewidth=0.5, zorder=3)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_xlabel("A1 deployed (training set)"); ax.set_ylabel("CV1 held out"); ax.set_title(title, fontsize=10)
    fig.suptitle("CRC11–CRC40, same crops: held-out recipe (y) vs deployed checkpoint scored on its training patients (x)", fontsize=9, color=MUTED)
    fig.tight_layout(); fig.savefig(FIG / "cv_vs_a1_per_patient.png", facecolor="white"); plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    s = json.loads((OUT / "summary.json").read_text())
    pp = pd.read_csv(OUT / "per_patient_metrics.csv")
    fig_training(s); fig_per_patient(pp); fig_confusion(s); fig_reliability(s); fig_robustness(s); fig_prevalence(s); fig_cv_vs_a1(pp)
    R, R30 = s["results"], s["comparison_CRC11_40"]["results"]
    pub = json.loads((C.A1_OUT / "summary.json").read_text())["results"]
    samp, folds, tr, ver = s["sampling"], s["folds"]["folds"], s["training"], s["verification"]
    cv1, cv2 = R["CV1"]["clean"]["pooled"], R["CV2"]["clean"]["pooled"]

    head_metrics = [("F1_T", "T F1"), ("F1_B", "B F1"), ("F1_Macrophage", "Macrophage F1"), ("F1_Other", "Other F1"), ("macro_F1", "Macro-F1"),
                    ("balanced_accuracy", "Balanced accuracy"), ("AUROC_B_vs_T", "AUROC P(B), B vs T"), ("ECE_B", "ECE P(B)"), ("ECE_T", "ECE P(T)")]
    headline = "".join(f"<tr><td>{html.escape(SHORT[a])}</td>" + "".join(f"<td>{vci(R[a]['clean']['pooled'][m])}</td>" for m, _ in head_metrics) + "</tr>" for a in C.ARMS)
    pmean = "".join(f"<tr><td>{html.escape(SHORT[a])}</td>" + "".join(
        f"<td>{vci(R[a]['clean']['patient_mean'][m])}<br><span class=muted>median {fmt(R[a]['clean']['patient_mean'][m]['patient_median'])} "
        f"(IQR {fmt(R[a]['clean']['patient_mean'][m]['patient_IQR'][0])}–{fmt(R[a]['clean']['patient_mean'][m]['patient_IQR'][1])})</span></td>"
        for m, _ in head_metrics) + "</tr>" for a in C.ARMS)
    # side by side: published A arms (30 pts), CV on the same 30 (paired diff), CV on 40
    side_metrics = [("F1_T", "T F1"), ("F1_B", "B F1"), ("recall_B", "B recall"), ("precision_B", "B precision"), ("macro_F1", "Macro-F1"),
                    ("AUROC_B_vs_T", "AUROC B vs T"), ("ECE_B", "ECE P(B)")]
    side = ""
    for a in ["A1", "A2", "A3"]:
        side += f"<tr><td>{html.escape(SHORT[a])}, CRC11–40 <span class=muted>(published)</span></td>" + "".join(f"<td>{vci(pub[a]['clean']['pooled'][m])}</td>" for m, _ in side_metrics) + "</tr>"
    for a in C.ARMS:
        if a in R30:
            side += f"<tr><td>{html.escape(SHORT[a])}, same 30 patients CRC11–40</td>" + "".join(f"<td>{vci(R30[a]['clean']['pooled'][m])}</td>" for m, _ in side_metrics) + "</tr>"
    for a in C.ARMS:
        side += f"<tr class=hl><td>{html.escape(SHORT[a])}, all 40 patients</td>" + "".join(f"<td>{vci(R[a]['clean']['pooled'][m])}</td>" for m, _ in side_metrics) + "</tr>"
    paired30 = s["comparison_CRC11_40"]["paired_differences_clean"]
    pdiff = "".join(f"<tr><td>{pair.replace('-', ' − ')}</td>" + "".join(dcell(blk[m]) if m in blk else "<td>–</td>" for m, _ in side_metrics) + "</tr>"
                    for pair, blk in paired30.items())
    cvdiff = ""
    for pair, blk in s["paired_differences_clean"].items():
        cvdiff += f"<tr><td>{pair.replace('-', ' − ')} (40 pts, pooled)</td>" + "".join(dcell(blk['pooled'][m]) if m in blk['pooled'] else "<td>–</td>" for m, _ in side_metrics) + "</tr>"
    perfold = ""
    fold_metrics = [("F1_T", "T F1"), ("F1_B", "B F1"), ("macro_F1", "Macro-F1"), ("AUROC_B_vs_T", "AUROC B vs T"), ("ECE_B", "ECE P(B)"),
                    ("predicted_prevalence_B", "Pred. B share"), ("true_prevalence_B", "IF B share")]
    for a in C.ARMS:
        perfold += f"<tr class=group><td colspan={len(fold_metrics) + 3}>{html.escape(SHORT[a])}</td></tr>"
        for name, f in folds.items():
            pf = R[a]["clean"]["per_fold"][name]
            t = tr.get(name, {})
            perfold += (f"<tr><td>{name}: {f['test_patients'][0]}–{f['test_patients'][-1]}</td><td>{', '.join(f['selection_patients'])}</td>"
                        f"<td>{t.get('selected_epoch', '–')}</td>" + "".join(f"<td>{vci(pf[m])}</td>" for m, _ in fold_metrics) + "</tr>")
    perclass = ""
    for a in C.ARMS:
        r = R[a]["clean"]["pooled"]
        perclass += f"<tr class=group><td colspan=4>{html.escape(SHORT[a])}</td></tr>"
        for n in NAMES:
            perclass += f"<tr><td>{n}</td><td>{vci(r[f'precision_{n}'])}</td><td>{vci(r[f'recall_{n}'])}</td><td>{vci(r[f'F1_{n}'])}</td></tr>"
    cms = ""
    for a in C.ARMS:
        cm = R[a]["clean"]["confusion_matrix_true_rows_pred_cols"]
        cms += (f"<div class=card><b>{html.escape(SHORT[a])}</b><table><tr><th>true \\ predicted</th>" + "".join(f"<th>{n}</th>" for n in NAMES) + "<th>n</th></tr>"
                + "".join(f"<tr><td>{NAMES[i]}</td>" + "".join(f"<td>{c:,}</td>" for c in row) + f"<td>{sum(row):,}</td></tr>" for i, row in enumerate(cm)) + "</table></div>")
    rob = s["robustness"]; rob30 = s["comparison_CRC11_40"]["robustness"]
    rob_metrics = [("macro_F1", "Macro-F1"), ("recall_B", "B recall"), ("F1_B", "B F1"), ("F1_T", "T F1"), ("AUROC_B_vs_T", "AUROC B vs T"), ("predicted_prevalence_B", "Predicted B share")]
    rob_rows = ""
    for a in C.ARMS:
        rob_rows += f"<tr class=group><td colspan={len(rob_metrics) + 1}>{html.escape(SHORT[a])}, 40 held-out patients</td></tr>"
        rob_rows += "<tr><td>clean (value)</td>" + "".join(f"<td>{vci(R[a]['clean']['pooled'][m])}</td>" for m, _ in rob_metrics) + "</tr>"
        for cond in ["sev1", "sev2", "sev3"]:
            if cond in rob["drop_clean_minus_perturbed"].get(a, {}):
                rob_rows += f"<tr><td>drop at {SEV_LABEL[cond]}</td>" + "".join(dcell(rob["drop_clean_minus_perturbed"][a][cond]["pooled"][m]) for m, _ in rob_metrics) + "</tr>"
    rob30_rows = ""
    for a in ["CV1", "A1", "A3"]:
        if a not in rob30["drop_clean_minus_perturbed"]:
            continue
        rob30_rows += f"<tr class=group><td colspan={len(rob_metrics) + 1}>{html.escape(SHORT[a])}, CRC11–40</td></tr>"
        rob30_rows += "<tr><td>clean (value)</td>" + "".join(f"<td>{vci(R30[a]['clean']['pooled'][m])}</td>" for m, _ in rob_metrics) + "</tr>"
        for cond in ["sev1", "sev2", "sev3"]:
            if cond in rob30["drop_clean_minus_perturbed"][a]:
                rob30_rows += f"<tr><td>drop at {SEV_LABEL[cond]}</td>" + "".join(dcell(rob30["drop_clean_minus_perturbed"][a][cond][m]) for m, _ in rob_metrics) + "</tr>"
    rob_pairs = ""
    for pair, conds in rob30["drop_difference_between_arms"].items():
        if not pair.startswith("CV1"):
            continue
        for cond, blk in conds.items():
            rob_pairs += f"<tr><td>{pair.replace('-', ' − ')}</td><td>{SEV_LABEL[cond]}</td>" + "".join(dcell(blk[m], key="difference_of_drops") for m, _ in rob_metrics) + "</tr>"
    prev = "<tr><td>IF labels, evaluated sample</td>" + "".join(f"<td>{cv1[f'true_prevalence_{n}']['value']:.3f}</td>" for n in NAMES) + "</tr>"
    prev += "<tr><td>IF labels, all crops of the 40 patients</td>" + "".join(f"<td>{samp['population_prevalence'][n]:.3f}</td>" for n in NAMES) + "</tr>"
    for a in C.ARMS:
        prev += f"<tr><td>Predicted, {html.escape(SHORT[a])}</td>" + "".join(f"<td>{vci(R[a]['clean']['pooled'][f'predicted_prevalence_{n}'])}</td>" for n in NAMES) + "</tr>"
    clean = pp[pp.condition == "clean"]
    ppm = ["F1_T", "F1_B", "recall_B", "precision_B", "macro_F1", "balanced_accuracy", "AUROC_B_vs_T", "ECE_B", "ECE_T"]
    pp_rows = ""
    for q in C.ALL:
        sub = clean[clean.patient == q].set_index("arm")
        pp_rows += (f"<tr><td>{q}</td><td>{C.fold_of(q)}</td><td>{int(sub.loc['CV1', 'n_cells']):,}</td>"
                    f"<td>{int(round(sub.loc['CV1', 'true_prevalence_B'] * sub.loc['CV1', 'n_cells'])):,}</td>"
                    + "".join("<td>" + " / ".join(fmt(sub.loc[a, m], 2) for a in C.ARMS) + "</td>" for m in ppm) + "</tr>")
    train_rows = ""
    for name, t in tr.items():
        f = folds[name]
        ep = [r for r in t["history"] if r["epoch"] == t["selected_epoch"]][0]
        train_rows += (f"<tr><td>{name}</td><td>{f['test_patients'][0]}–{f['test_patients'][-1]}</td><td>{len(f['fit_patients'])}</td><td>{', '.join(f['selection_patients'])}</td>"
                       f"<td>{t['selected_epoch']} / {len(t['history']) - 1}</td><td>{fmt(ep['holdout_macro_F1'])}</td><td>{fmt(ep.get('ESCA_dev_AUROC_PB_TLS_vs_dense'))}</td>"
                       f"<td>{' / '.join(f'{v:.3f}' for v in t['prior_Other_T_Macrophage_B'])}</td><td class=muted>{t['checkpoint_sha256'][:12]}…</td></tr>")
    dep_prior = json.loads((C.DEPLOYED / "prior_correction_tau05.json").read_text())["prior_Other_T_Macrophage_B"]
    ver_rows = "".join(f"<li>{name}: " + "; ".join(f"{a}: {x['identical_predictions']}/{ver['n_per_fold']} identical, max |Δp| {x['max_abs_prob_diff']:.1e}, τ {x['prior_correction_tau']}"
                                                  for a, x in v.items()) + "</li>" for name, v in ver["folds"].items()) if ver else "<li>not run</li>"
    chk = s["comparison_CRC11_40"]["recomputation_check_vs_published"]
    chk_txt = ", ".join(f"{a} max |Δ| {v['max_abs_diff']:.1e}" for a, v in chk.items())
    a1 = pub["A1"]["clean"]["pooled"]
    d30 = paired30.get("CV1-A1", {})

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Classifier Cross-Validation</title><style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--line:#e4e3df;--card:#fff;--pos:#1d5fae;--warn:#a04a12;--hl:#eef4fc}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--line:#3a3936;--card:#232321;--pos:#7fb2f0;--warn:#f0a070;--hl:#1f2a3a}}}}
:root[data-theme="dark"]{{--bg:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--line:#3a3936;--card:#232321;--pos:#7fb2f0;--warn:#f0a070;--hl:#1f2a3a}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:28px 16px 64px}} h1{{font-size:26px;margin:0 0 4px}} h2{{margin-top:36px;border-top:1px solid var(--line);padding-top:18px}}
.muted{{color:var(--muted)}} table{{border-collapse:collapse;width:100%;font-size:12.5px;display:block;overflow-x:auto}}
th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;white-space:nowrap;vertical-align:top}} img.fig{{max-width:100%;background:#fff;border-radius:8px;border:1px solid var(--line)}}
tr.group td{{font-weight:600;background:var(--card);padding-top:12px}} tr.hl td{{background:var(--hl);font-weight:600}} td.sig{{color:var(--pos);font-weight:600}} .warn{{color:var(--warn)}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 14px;margin:10px 0}} .card ul,.card ol{{margin:6px 0}}
</style></head><body><main>
<h1>Held-out T and B cell accuracy of the HE2CELL classifier recipe: 4-fold patient-level cross-validation</h1>
<p class="muted">Immunocto CRC01–CRC40 (40 patients, {samp['evaluated_crops']:,} evaluated crops of {samp['population_crops']:,}). Ground truth: multiplex-immunofluorescence labels transferred to H&amp;E (other / CD4 / CD8 / macrophage / B; T = CD4 + CD8).
Each patient is scored by a model fitted on the other 30 patients with the deployed fine-tuning recipe, from ImageNet weights.</p>

<div class=card><b>Headline (held out, pooled over 40 patients; 95% patient-bootstrap CIs).</b> Recipe with prior correction (CV1): T F1 {vci(cv1['F1_T'])}, B F1 {vci(cv1['F1_B'])},
macro-F1 {vci(cv1['macro_F1'])}, B-vs-T AUROC {vci(cv1['AUROC_B_vs_T'])}, ECE P(B) {vci(cv1['ECE_B'])}. Without the prior (CV2): T F1 {vci(cv2['F1_T'])}, B F1 {vci(cv2['F1_B'])}, macro-F1 {vci(cv2['macro_F1'])}.
<br>For comparison, the deployed checkpoint scored on its own training patients CRC11–40 (A1, published): T F1 {vci(a1['F1_T'])}, B F1 {vci(a1['F1_B'])}, macro-F1 {vci(a1['macro_F1'])}, AUROC {vci(a1['AUROC_B_vs_T'])}.
{('On the same 30 patients the held-out recipe differs from A1 by macro-F1 ' + dcell(d30['macro_F1']).replace('<td', '<span').replace('</td>', '</span>') + ', B F1 ' + dcell(d30['F1_B']).replace('<td', '<span').replace('</td>', '</span>') + ', T F1 ' + dcell(d30['F1_T']).replace('<td', '<span').replace('</td>', '</span>') + ' (paired, same crops).') if d30 else ''}
</div>

<div class=card><b>What these numbers mean.</b>
<ol>
<li><b>Out-of-patient estimate of the recipe.</b> Every one of the 40 patients is scored by a model that never saw it, neither in fitting nor in checkpoint selection. The CV numbers therefore estimate how the deployed <i>recipe</i> performs on new patients of this domain (Immunocto-style colorectal H&amp;E with IF-derived labels).</li>
<li><b>The deployed checkpoint itself cannot be scored out of patient</b>: it is initialised from <code>best_model2.pth</code>, which was trained on all 40 patients. It was produced by the same recipe on fewer patients (8 fitted + 2 selection) from that stronger, in-domain initialisation. On new patients of this domain it is expected to perform about like the CV estimate; its training-set numbers (A1) are an upper bound.</li>
<li><b>ImageNet initialisation decision.</b> Any fold initialised from <code>best_model2.pth</code> would leak its held-out patients, so each fold starts from torchvision's ImageNet ResNet-50 weights (<code>finetune_classifier.py --init imagenet</code>; mask encoder and 4-class head random). This is the one deliberate departure from the deployed fine-tune; the other difference is the training set size (28 fitted patients instead of 8). Everything else — augmentation, 10 epochs × 200,000 class-balanced samples, batch 256, AdamW 1e-4 one-cycle, label smoothing 0.05, checkpoint selection by selection-patient macro-F1 + ESCA-dev AUROC, D4 TTA, τ 0.5 prior from the fitted patients' class counts — is identical.</li>
<li><b>In-domain only.</b> This is not a test on ESCA or Visium stains; the stain-perturbation section addresses colour shift within the augmentation family only.</li>
</ol></div>

<h2>Design</h2>
<table><tr><th>Fold</th><th>Held-out patients</th><th>Fitted</th><th>Selection (early stopping)</th><th>Selected epoch / of</th><th>Selection macro-F1</th><th>ESCA-dev AUROC</th><th>Prior Other / T / Mac / B</th><th>Checkpoint</th></tr>{train_rows}</table>
<p class=muted>Selection patients were drawn with seed [20261026, fold] from the 30 training patients; the deployed run used CRC03 and CRC09 of CRC01–10. Deployed prior: {' / '.join(f'{v:.3f}' for v in dep_prior)} (8 fitted patients). Each fold's prior is the normalised class count of its 28 fitted patients, written as <code>best_finetuned.prior.json</code> next to the checkpoint so <code>model_eval.py</code> applies it automatically.
Crops: the CRC11–40 index of <code>outputs/cell_validation_20261026</code> is reused unchanged; CRC01–10 are sampled with the same rule (≤ 50,000 per patient, seed 20261026); CRC03, CRC05, CRC21 and CRC25 have fewer and are used in full.</p>
<img class="fig" src="{png(FIG / 'training_curves.png')}" alt="Training curves per fold">
<p class=muted>Inference reproduces <code>script/src/model_eval.py</code> (PIL RGB/255, ImageNet normalisation, mask/max, D4 mean softmax, sidecar prior). Check on 250 held-out crops per fold against model_eval.py run as a subprocess with <code>--prior_correction auto</code> (CV1) and <code>none</code> (CV2):</p><ul>{ver_rows}</ul>
<p class=muted>Re-computing the published A1/A2/A3 clean metrics from their stored predictions with this script's bootstrap: {chk_txt}.</p>

<h2>Statistics</h2>
<p>Pooled metrics pool every evaluated cell; "patient mean" averages per-patient values. 95% CIs are percentile intervals from a patient-cluster bootstrap (2000 resamples of the 40 patients, seed 20261026; of the 30 patients CRC11–40 in the comparison with A1–A3, where the resamples are the ones of the published evaluation, so differences are paired on the same crops and perturbation draws). Per-fold CIs resample the fold's 10 patients. p values are two-sided bootstrap p (0 printed as p&lt;0.001). Blue bold = CI excludes 0 (sign only; for ECE lower is better).
AUROC: P(B) separating IF-B from IF-T cells. ECE: one-vs-rest, 10 equal-width bins, all cells.</p>

<h2>Headline table (held out, pooled cells, clean)</h2>
<table><tr><th>Arm</th>{''.join(f'<th>{t}</th>' for _, t in head_metrics)}</tr>{headline}</table>
<h3>Mean of per-patient values</h3>
<table><tr><th>Arm</th>{''.join(f'<th>{t}</th>' for _, t in head_metrics)}</tr>{pmean}</table>
<img class="fig" src="{png(FIG / 'per_patient.png')}" alt="Per-patient F1 and AUROC by arm, coloured by fold">
<h3>Prior correction (CV1 − CV2, paired)</h3>
<table><tr><th>a − b</th>{''.join(f'<th>{t}</th>' for _, t in side_metrics)}</tr>{cvdiff}</table>

<h2>Per fold</h2>
<table><tr><th>Fold (held-out)</th><th>Selection patients</th><th>Epoch</th>{''.join(f'<th>{t}</th>' for _, t in fold_metrics)}</tr>{perfold}</table>

<h2>Side by side with the existing training-set numbers</h2>
<p>A1–A3 (<code>outputs/cell_validation_20261026</code>) score the deployed and original checkpoints on CRC11–40, which both saw in training. CV1/CV2 on the same 30 patients use the same crops and are held out; the paired difference measures the optimism of the training-set numbers for the recipe.</p>
<table><tr><th>Arm, patients</th>{''.join(f'<th>{t}</th>' for _, t in side_metrics)}</tr>{side}</table>
<h3>Paired differences on CRC11–40 (held-out CV − training-set A; pooled cells)</h3>
<table><tr><th>a − b</th>{''.join(f'<th>{t}</th>' for _, t in side_metrics)}</tr>{pdiff}</table>
<img class="fig" src="{png(FIG / 'cv_vs_a1_per_patient.png')}" alt="Per-patient CV1 vs A1 scatter">

<h2>Confusion matrices (40 held-out patients)</h2>
<img class="fig" src="{png(FIG / 'confusion_matrices.png')}" alt="Row-normalised confusion matrices">
<details><summary>Counts</summary>{cms}</details>

<h2>Per-class precision, recall and F1</h2>
<table><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th></tr>{perclass}</table>

<h2>Calibration</h2>
<img class="fig" src="{png(FIG / 'reliability.png')}" alt="Reliability curves for P(B) and P(T)">

<h2>Class prevalence</h2>
<img class="fig" src="{png(FIG / 'prevalence.png')}" alt="True vs predicted class prevalence">
<table><tr><th></th>{''.join(f'<th>{n}</th>' for n in NAMES)}</tr>{prev}</table>

<h2>Stain robustness</h2>
<p>Every evaluated crop was re-scored under the LAB colour-style shift + HED stain perturbation of the fine-tune's training augmentation at λ = 0.5, 1.0 (a full-strength training draw) and 1.5 (beyond the training range), with the same seeded direction per crop as in the A1–A3 evaluation (so CRC11–40 drops are paired with A1–A3).
Caveat as before: the recipe is trained on this perturbation family, so this measures invariance to the augmentation family, not to unseen stain processes.</p>
<img class="fig" src="{png(FIG / 'robustness.png')}" alt="Metrics versus perturbation severity">
<h3>Drop from clean (clean − perturbed; positive = worse under perturbation, except predicted B share), 40 held-out patients</h3>
<table><tr><th></th>{''.join(f'<th>{t}</th>' for _, t in rob_metrics)}</tr>{rob_rows}</table>
<h3>Drops on CRC11–40: held-out recipe vs the published arms (same crops and draws)</h3>
<table><tr><th></th>{''.join(f'<th>{t}</th>' for _, t in rob_metrics)}</tr>{rob30_rows}</table>
<h3>Difference of drops (drop<sub>CV1</sub> − drop<sub>A</sub>; negative = the held-out recipe degrades less)</h3>
<table><tr><th>a − b</th><th>Severity</th>{''.join(f'<th>{t}</th>' for _, t in rob_metrics)}</tr>{rob_pairs}</table>

<h2>Per patient (clean)</h2>
<p class="muted">Each cell: CV1 / CV2. Full table for all conditions: per_patient_metrics.csv.</p>
<table><tr><th>Patient</th><th>Fold</th><th>Cells</th><th>IF-B cells</th>{''.join(f'<th>{m}</th>' for m in ppm)}</tr>{pp_rows}</table>

<h2>Limitations</h2>
<ul>
<li>The CV estimates the recipe, not the deployed weights; the deployed checkpoint's own out-of-patient accuracy is unknowable from Immunocto and is expected to be about this (its initialisation is stronger, its fitted set smaller).</li>
<li>Same cohort, lab and scanner across folds: in-domain generalisation to new patients, not to ESCA, Visium or other stains and scanners.</li>
<li>Immunocto labels are IF-derived and automatically transferred; label noise caps the attainable score equally for all arms.</li>
<li>Crops are Immunocto 64-px crops with Immunocto masks, not Cellpose crops from the HE2CELL pipeline.</li>
<li>Contiguous-block folds were pre-specified; patient blocks may differ in composition (e.g. B-cell share), which the per-fold table shows.</li>
<li>Subsampling to 50,000 cells per patient makes pooled prevalence differ from the cohort's crop-weighted prevalence (both shown).</li>
</ul>

<h2>Provenance</h2>
<p class="muted">Code: deployment/cell_validation/classifier_cv.py (folds, train, prior, index, verify, infer, metrics), classifier_cv_report.py, run_classifier_cv.sh; deployment/finetune/finetune_classifier.py --init imagenet.
Outputs: outputs/classifier_cv_20261026/{{folds.json, fold0-3/{{best_finetuned.pth, best_finetuned.prior.json, settings.json, history.json, class_counts.csv, train.log}}, crops.csv.gz, crop_index/ (CRC01–10), verification.json,
predictions/cv_&lt;clean|sev1|sev2|sev3&gt;/CRC##.csv.gz (uncorrected D4 probabilities), summary.json, per_patient_metrics.csv, figures/}}. The published A1–A3 arms are read from outputs/cell_validation_20261026 and not modified.</p>
</main></body></html>"""
    (OUT / "CV_report.html").write_text(page)
    print("wrote", OUT / "CV_report.html")


if __name__ == "__main__":
    main()
