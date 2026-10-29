"""HTML report for the Immunocto CRC11-CRC40 cell-level validation (not held out; see caveats) (reads summary.json written by
immunocto_holdout_eval.py metrics, per_patient_metrics.csv and examples/*.npz)."""
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
import immunocto_holdout_eval as E  # noqa: E402

OUT = E.OUT
FIG = OUT / "figures"
NAMES = E.NAMES
COL = {"A1": "#2a78d6", "A2": "#eb6834", "A3": "#1baf7a"}  # categorical slots 1-3 (validated all-pairs)
SHORT = {"A1": "A1 fine-tune + prior", "A2": "A2 fine-tune, no prior", "A3": "A3 original"}
SEV = {"clean": 0.0, "sev1": 0.5, "sev2": 1.0, "sev3": 1.5}
SEV_LABEL = {"clean": "clean", "sev1": "λ 0.5 (mild)", "sev2": "λ 1.0 (training range)", "sev3": "λ 1.5 (beyond training)"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.dpi": 130})


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
    ptxt = "" if p is None else (f" p&lt;0.001" if p == 0 else f" p={p:.3f}")
    return (f"<td class='{'sig' if sig else ''}'>{m[key]:+.{d}f} <span class=muted>[{lo:+.{d}f}, {hi:+.{d}f}]{ptxt}</span></td>")


def fig_confusion(s):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, arm in zip(axes, E.ARMS):
        cm = np.array(s["results"][arm]["clean"]["confusion_matrix_true_rows_pred_cols"], float)
        rn = cm / cm.sum(1, keepdims=True)
        ax.imshow(rn, cmap="Blues", vmin=0, vmax=1)
        ax.grid(False)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{rn[i, j]:.2f}\n{int(cm[i, j]):,}", ha="center", va="center", fontsize=7.5,
                        color="white" if rn[i, j] > .55 else INK)
        ax.set_xticks(range(4), NAMES, rotation=30)
        ax.set_yticks(range(4), NAMES)
        ax.set_xlabel("Predicted")
        ax.set_title(SHORT[arm], fontsize=10)
    axes[0].set_ylabel("IF label (true)")
    fig.tight_layout()
    fig.savefig(FIG / "confusion_matrices.png", facecolor="white")
    plt.close(fig)


def fig_reliability(s):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, n in zip(axes, ["B", "T"]):
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="perfect calibration")
        for arm in E.ARMS:
            r = s["reliability_pooled"][arm][n]
            x, y = np.array(r["mean_predicted"], float), np.array(r["observed_fraction"], float)
            lo = np.array([c[0] if c[0] is not None else np.nan for c in r["observed_CI95"]], float)
            hi = np.array([c[1] if c[1] is not None else np.nan for c in r["observed_CI95"]], float)
            ok = np.array(r["count"]) > 0
            ece = s["results"][arm]["clean"]["pooled"][f"ECE_{n}"]
            ax.errorbar(x[ok], y[ok], yerr=[y[ok] - lo[ok], hi[ok] - y[ok]], color=COL[arm], lw=2, marker="o", ms=5,
                        capsize=2, label=f"{SHORT[arm]} (ECE {ece['value']:.3f})")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel(f"Predicted P({n}) (bin mean)")
        ax.set_ylabel(f"Observed fraction IF-{n}")
        ax.set_title(f"Reliability, P({n}), all cells pooled", fontsize=10)
        ax.legend(fontsize=7.5, loc="upper left" if n == "B" else "lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "reliability.png", facecolor="white")
    plt.close(fig)


def fig_per_patient(pp):
    metrics = [("F1_B", "B F1"), ("F1_T", "T F1"), ("macro_F1", "Macro-F1"), ("AUROC_B_vs_T", "AUROC P(B), B vs T")]
    clean = pp[pp.condition == "clean"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6))
    for ax, (m, title) in zip(axes, metrics):
        wide = clean.pivot(index="patient", columns="arm", values=m)
        for _, row in wide.iterrows():
            ax.plot(range(3), row[list(E.ARMS)].to_numpy(), color=GRID, lw=0.8, zorder=1)
        for i, arm in enumerate(E.ARMS):
            ax.scatter(np.full(len(wide), i) + np.random.default_rng(i).uniform(-.08, .08, len(wide)), wide[arm],
                       s=14, color=COL[arm], zorder=2, edgecolor="white", linewidth=0.5)
            ax.plot([i - .25, i + .25], [wide[arm].median()] * 2, color=INK, lw=2, zorder=3)
        ax.set_xticks(range(3), ["A1", "A2", "A3"])
        ax.set_title(title, fontsize=10)
    fig.suptitle("Per patient (Immunocto CRC11–CRC40, not held out; black bar = median; grey lines join the same patient)", fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(FIG / "per_patient.png", facecolor="white")
    plt.close(fig)


def fig_robustness(s):
    metrics = [("macro_F1", "Macro-F1"), ("recall_B", "B recall"), ("F1_B", "B F1"), ("AUROC_B_vs_T", "AUROC P(B), B vs T")]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
    for ax, (m, title) in zip(axes, metrics):
        for arm in E.ARMS:
            conds = [c for c in SEV if c in s["results"][arm]]
            x = [SEV[c] for c in conds]
            v = [s["results"][arm][c]["pooled"][m]["value"] for c in conds]
            lo = [s["results"][arm][c]["pooled"][m]["CI95"][0] for c in conds]
            hi = [s["results"][arm][c]["pooled"][m]["CI95"][1] for c in conds]
            ax.fill_between(x, lo, hi, color=COL[arm], alpha=.15, lw=0)
            ax.plot(x, v, color=COL[arm], lw=2, marker="o", ms=5, label=SHORT[arm])
        ax.set_xticks([0, .5, 1, 1.5], ["clean", "0.5", "1.0", "1.5"])
        ax.set_xlabel("Perturbation severity λ")
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "robustness.png", facecolor="white")
    plt.close(fig)


def fig_prevalence(s):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    w = .2
    true = [s["results"]["A1"]["clean"]["pooled"][f"true_prevalence_{n}"]["value"] for n in NAMES]
    ax.bar(np.arange(4) - 1.5 * w, true, w - .02, color="#8a8984", label="IF labels (truth)")
    for k, arm in enumerate(E.ARMS):
        r = s["results"][arm]["clean"]["pooled"]
        v = [r[f"predicted_prevalence_{n}"]["value"] for n in NAMES]
        lo = [v[i] - r[f"predicted_prevalence_{n}"]["CI95"][0] for i, n in enumerate(NAMES)]
        hi = [r[f"predicted_prevalence_{n}"]["CI95"][1] - v[i] for i, n in enumerate(NAMES)]
        ax.bar(np.arange(4) + (k - .5) * w, v, w - .02, color=COL[arm], yerr=[lo, hi], capsize=2, label=SHORT[arm],
               error_kw={"ecolor": MUTED, "lw": 0.8})
    ax.set_xticks(range(4), NAMES)
    ax.set_ylabel("Share of evaluated cells")
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_title("Class prevalence: IF truth vs predicted (pooled, clean)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "prevalence.png", facecolor="white")
    plt.close(fig)


def fig_examples():
    files = sorted((OUT / "examples").glob("*.npz"))[:3]
    if not files:
        return False
    rows = []
    for f in files:
        z = np.load(f)
        for i in range(0, len(z["label"]), 2):
            rows.append((f.stem, int(z["label"][i]), [z[c][i] for c in SEV]))
    fig, axes = plt.subplots(len(rows), 4, figsize=(4.6, 1.2 * len(rows)))
    for r, (patient, label, imgs) in enumerate(rows):
        for c, img in enumerate(imgs):
            ax = axes[r, c]
            ax.imshow(img); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if r == 0:
                ax.set_title(["clean", "λ 0.5", "λ 1.0", "λ 1.5"][c], fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{patient}\n{NAMES[label]}", fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "perturbation_examples.png", facecolor="white")
    plt.close(fig)
    return True


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    s = json.loads((OUT / "summary.json").read_text())
    pp = pd.read_csv(OUT / "per_patient_metrics.csv")
    fig_confusion(s); fig_reliability(s); fig_per_patient(pp); fig_robustness(s); fig_prevalence(s)
    has_examples = fig_examples()
    R = s["results"]
    samp = s["sampling"]
    v = s["verification"]

    head_metrics = [("F1_T", "T F1"), ("F1_B", "B F1"), ("F1_Macrophage", "Macrophage F1"), ("F1_Other", "Other F1"),
                    ("macro_F1", "Macro-F1"), ("balanced_accuracy", "Balanced accuracy"), ("AUROC_B_vs_T", "AUROC P(B), B vs T"),
                    ("ECE_B", "ECE P(B)"), ("ECE_T", "ECE P(T)")]
    headline = "".join(f"<tr><td>{html.escape(SHORT[a])}</td>" + "".join(f"<td>{vci(R[a]['clean']['pooled'][m])}</td>" for m, _ in head_metrics) + "</tr>"
                       for a in E.ARMS)
    pmean = "".join(f"<tr><td>{html.escape(SHORT[a])}</td>" + "".join(
        f"<td>{vci(R[a]['clean']['patient_mean'][m])}<br><span class=muted>median {fmt(R[a]['clean']['patient_mean'][m]['patient_median'])} "
        f"(IQR {fmt(R[a]['clean']['patient_mean'][m]['patient_IQR'][0])}–{fmt(R[a]['clean']['patient_mean'][m]['patient_IQR'][1])})</span></td>"
        for m, _ in head_metrics) + "</tr>" for a in E.ARMS)
    perclass = ""
    for a in E.ARMS:
        r = R[a]["clean"]["pooled"]
        perclass += f"<tr class=group><td colspan=4>{html.escape(SHORT[a])}</td></tr>"
        for n in NAMES:
            perclass += (f"<tr><td>{n}</td><td>{vci(r[f'precision_{n}'])}</td><td>{vci(r[f'recall_{n}'])}</td>"
                         f"<td>{vci(r[f'F1_{n}'])}</td></tr>")
    cms = ""
    for a in E.ARMS:
        cm = R[a]["clean"]["confusion_matrix_true_rows_pred_cols"]
        cms += (f"<div class=card><b>{html.escape(SHORT[a])}</b><table><tr><th>true \\ predicted</th>"
                + "".join(f"<th>{n}</th>" for n in NAMES) + "<th>n</th></tr>"
                + "".join(f"<tr><td>{NAMES[i]}</td>" + "".join(f"<td>{c:,}</td>" for c in row) + f"<td>{sum(row):,}</td></tr>"
                          for i, row in enumerate(cm)) + "</table></div>")
    diff_metrics = [("F1_T", "T F1"), ("F1_B", "B F1"), ("recall_B", "B recall"), ("precision_B", "B precision"),
                    ("macro_F1", "Macro-F1"), ("balanced_accuracy", "Balanced acc."), ("AUROC_B_vs_T", "AUROC B vs T"),
                    ("ECE_B", "ECE P(B)"), ("ECE_T", "ECE P(T)")]
    diffs = ""
    for scope in ["pooled", "patient_mean"]:
        diffs += f"<tr class=group><td colspan={len(diff_metrics) + 1}>{'Pooled cells' if scope == 'pooled' else 'Mean of per-patient values'}</td></tr>"
        for pair, blk in s["paired_differences_clean"].items():
            diffs += f"<tr><td>{pair.replace('-', ' − ')}</td>" + "".join(dcell(blk[scope][m]) if m in blk[scope] else "<td>–</td>" for m, _ in diff_metrics) + "</tr>"
    rob = s["robustness"]
    rob_metrics = [("macro_F1", "Macro-F1"), ("recall_B", "B recall"), ("F1_B", "B F1"), ("F1_T", "T F1"),
                   ("AUROC_B_vs_T", "AUROC B vs T"), ("predicted_prevalence_B", "Predicted B share")]
    rob_rows = ""
    for a in E.ARMS:
        rob_rows += f"<tr class=group><td colspan={len(rob_metrics) + 1}>{html.escape(SHORT[a])}</td></tr>"
        rob_rows += "<tr><td>clean (value)</td>" + "".join(f"<td>{vci(R[a]['clean']['pooled'][m])}</td>" for m, _ in rob_metrics) + "</tr>"
        for cond in ["sev1", "sev2", "sev3"]:
            if cond in rob["drop_clean_minus_perturbed"].get(a, {}):
                blk = rob["drop_clean_minus_perturbed"][a][cond]["pooled"]
                rob_rows += f"<tr><td>drop at {SEV_LABEL[cond]}</td>" + "".join(dcell(blk[m]) for m, _ in rob_metrics) + "</tr>"
    rob_pairs = ""
    for pair, conds in rob["drop_difference_between_arms"].items():
        for cond, blk in conds.items():
            rob_pairs += (f"<tr><td>{pair.replace('-', ' − ')}</td><td>{SEV_LABEL[cond]}</td>"
                          + "".join(dcell(blk[m], key="difference_of_drops") for m, _ in rob_metrics) + "</tr>")
    prev = "<tr><td>IF labels, evaluated sample</td>" + "".join(f"<td>{R['A1']['clean']['pooled'][f'true_prevalence_{n}']['value']:.3f}</td>" for n in NAMES) + "</tr>"
    prev += "<tr><td>IF labels, all crops of the 30 patients</td>" + "".join(f"<td>{samp['population_prevalence'][n]:.3f}</td>" for n in NAMES) + "</tr>"
    for a in E.ARMS:
        prev += f"<tr><td>Predicted, {html.escape(SHORT[a])}</td>" + "".join(f"<td>{vci(R[a]['clean']['pooled'][f'predicted_prevalence_{n}'])}</td>" for n in NAMES) + "</tr>"
    for a in E.ARMS:
        for cond in ["sev1", "sev2", "sev3"]:
            if cond in R[a]:
                prev += (f"<tr><td class=muted>Predicted, {html.escape(SHORT[a])}, {SEV_LABEL[cond]}</td>"
                         + "".join(f"<td class=muted>{R[a][cond]['pooled'][f'predicted_prevalence_{n}']['value']:.3f}</td>" for n in NAMES) + "</tr>")
    clean = pp[pp.condition == "clean"]
    ppm = ["n_cells", "F1_T", "F1_B", "recall_B", "precision_B", "macro_F1", "balanced_accuracy", "AUROC_B_vs_T", "ECE_B", "ECE_T"]
    pp_rows = ""
    for q in E.PATIENTS:
        sub = clean[clean.patient == q].set_index("arm")
        pp_rows += f"<tr><td>{q}</td><td>{int(sub.loc['A1', 'n_cells']):,}</td><td>{int(round(sub.loc['A1', 'true_prevalence_B'] * sub.loc['A1', 'n_cells'])):,}</td>" + "".join(
            "<td>" + " / ".join(fmt(sub.loc[a, m], 2) for a in E.ARMS) + "</td>" for m in ppm[1:]) + "</tr>"
    ver = "".join(f"<li>{a}: {x['identical_predictions']}/{v['n_crops']} identical predictions, max |Δp| {x['max_abs_prob_diff']:.1e}</li>"
                  for a, x in v["arms"].items()) if v else "<li>not run</li>"

    a1, a3 = R["A1"]["clean"]["pooled"], R["A3"]["clean"]["pooled"]
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Immunocto Cell Validation</title><style>
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
<h1>Cell-level validation: T and B identification on Immunocto CRC11–CRC40 (not held out: see Caveats)</h1>
<p class="muted">{len(E.PATIENTS)} Immunocto patients (CRC11–CRC40). The adopted fine-tune's own fine-tuning step used CRC01–CRC10 only (CRC03/CRC09 for checkpoint selection), but its starting checkpoint saw all 40 patients (see Caveats).
Ground truth: Immunocto multiplex-immunofluorescence labels transferred to H&amp;E (other / CD4 / CD8 / macrophage / B; T = CD4 + CD8). The 40-patient fine-tune (outputs/classifier_finetune_40pt_20261022/) is not evaluated.</p>
<div class="card warn" style="border-color:var(--warn)"><b>Caveats: these patients are not held out.</b> Source: user statement, 2026-10-26: models/best_model2.pth was trained on <b>all 40</b> Immunocto patients (CRC01–CRC40).
<ol>
<li><b>Not held out for A3 (original):</b> its clean-crop numbers are training-set performance.</li>
<li><b>Not strictly held out for A1/A2:</b> the fine-tune was initialised from best_model2.pth, so CRC11–CRC40 are unseen only by the fine-tuning step. The A1-vs-A3 clean comparison (macro-F1 −0.036 etc.) compares two models on crops both have seen in some form and cannot speak to generalisation.</li>
<li><b>Still supported:</b> (a) the stain-perturbation robustness ranking (macro-F1 drop at λ 1.0: A1 +0.022 vs A3 +0.260; differences of drops exclude 0), with the caveat that the perturbation family equals the fine-tune's training augmentation family; (b) the prior-correction trade-off A1 vs A2 (B recall −0.196 for higher B precision and better calibration), since both share the same training data; (c) the calibration and prevalence match.</li>
<li><b>Not supported:</b> any absolute held-out T/B accuracy claim for either model. A genuine held-out estimate needs external CD20/CD3 data (plan item 2) or a from-scratch patient-level cross-validation with the same recipe; neither was run.</li>
</ol></div>

<div class=card><b>Headline (in-sample for A3, see Caveats; not a held-out accuracy estimate).</b> Adopted model (A1): T F1 {vci(a1['F1_T'])}, B F1 {vci(a1['F1_B'])}, macro-F1 {vci(a1['macro_F1'])}, B-vs-T AUROC {vci(a1['AUROC_B_vs_T'])}.
Original (A3): T F1 {vci(a3['F1_T'])}, B F1 {vci(a3['F1_B'])}, macro-F1 {vci(a3['macro_F1'])}, B-vs-T AUROC {vci(a3['AUROC_B_vs_T'])}.</div>

<h2>Data and arms</h2>
<ul>
<li>{samp['population_crops']:,} labelled crops in the 30 patients; {samp['evaluated_crops']:,} evaluated. Because the total exceeds 1.5M, each patient is subsampled to at most {E.N_PER_PATIENT:,} crops by simple random sampling (seed {E.SEED}; CRC21 and CRC25 have fewer and are used in full). Class shares are therefore those of each patient; pooled shares weight patients almost equally.</li>
<li><b>A1</b> {html.escape(E.ARMS['A1']['label'])}: models/he2cell_esca_finetuned.pth, D4 mean softmax × prior<sup>0.5</sup> (sidecar), renormalised. <b>A2</b>: the same probabilities without the prior correction. <b>A3</b>: models/best_model2.pth (no sidecar). A1 and A2 share one forward pass.</li>
<li>Inference reproduces script/src/model_eval.py exactly (PIL RGB/255, ImageNet normalisation, mask/max, D4 average, batch 512, mps). Checked on 1000 crops against model_eval.py run as a subprocess:<ul>{ver}</ul></li>
</ul>

<h2>Statistics</h2>
<p>Pooled metrics pool every evaluated cell; "patient mean" averages the per-patient values. 95% CIs are percentile intervals from a patient-cluster bootstrap (2000 resamples of the 30 patients, seed {E.SEED}); every arm and perturbation level uses the same resamples, so differences are paired.
p values are two-sided bootstrap p (2 × the smaller share of resampled differences on either side of 0; 0 printed as p&lt;0.001). Blue bold = CI excludes 0 (sign only; not necessarily "better" — for ECE lower is better).
AUROC: P(B) separating IF-B from IF-T cells (lymphocytes only). Calibration: one-vs-rest, 10 equal-width bins over all cells, ECE = Σ<sub>bins</sub> (n<sub>b</sub>/N)·|mean P − observed fraction|.</p>

<h2>Headline table (pooled cells, clean; not held out, see Caveats)</h2>
<table><tr><th>Arm</th>{''.join(f'<th>{t}</th>' for _, t in head_metrics)}</tr>{headline}</table>
<h3>Mean of per-patient values</h3>
<table><tr><th>Arm</th>{''.join(f'<th>{t}</th>' for _, t in head_metrics)}</tr>{pmean}</table>
<img class="fig" src="{png(FIG / 'per_patient.png')}" alt="Per-patient F1 and AUROC by arm">

<h2>Confusion matrices</h2>
<p>Rows = IF label, columns = prediction; cells show the row share and the count.</p>
<img class="fig" src="{png(FIG / 'confusion_matrices.png')}" alt="Row-normalised confusion matrices for the three arms">
<details><summary>Counts</summary>{cms}</details>

<h2>Per-class precision, recall and F1</h2>
<table><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th></tr>{perclass}</table>

<h2>Paired differences between arms (clean)</h2>
<p class=warn>A1/A2 vs A3 on clean crops compares models that have both seen these patients in some form (A3 directly, A1/A2 through initialisation); it does not measure generalisation. A1 vs A2 shares the same training data and is interpretable.</p>
<table><tr><th>a − b</th>{''.join(f'<th>{t}</th>' for _, t in diff_metrics)}</tr>{diffs}</table>

<h2>Calibration</h2>
<p>Pooled reliability curves; error bars are bootstrap 95% CIs of the observed fraction per bin.</p>
<img class="fig" src="{png(FIG / 'reliability.png')}" alt="Reliability curves for P(B) and P(T)">

<h2>Class prevalence</h2>
<img class="fig" src="{png(FIG / 'prevalence.png')}" alt="True vs predicted class prevalence">
<table><tr><th></th>{''.join(f'<th>{n}</th>' for n in NAMES)}</tr>{prev}</table>

<h2>Stain robustness</h2>
<p>Every evaluated crop was re-scored under the LAB colour-style shift and HED stain perturbation of <code>finetune_classifier.py</code>'s training augmentation (no other augmentation).
Each crop draws one seeded random direction: a target LAB mean in the training STYLE range (L 110–190, a 135–175, b 100–132), an LAB std scale (0.75–1.35), HED scale factors (±15%) and HED shifts (±0.012).
Severity λ scales the move from the crop's own colour (patient LAB statistics, 400 crops): target mean = mean + λ(target − mean), std scale = 1 + λ(s − 1), HED scale = 1 + λ(f − 1), HED shift = λ·shift.
λ 1.0 is an ordinary full-strength training draw for the fine-tune; λ 1.5 goes beyond its training range. The same draws are used for both checkpoints. Note that A1/A2 were trained on exactly this family of perturbations, so the test shows invariance to the augmentation family, not to unseen stain processes.</p>
{f'<img class="fig" style="max-width:420px" src="{png(FIG / "perturbation_examples.png")}" alt="Example crops at each severity">' if has_examples else ''}
<img class="fig" src="{png(FIG / 'robustness.png')}" alt="Metrics versus perturbation severity">
<h3>Drop from clean (clean − perturbed; positive = worse under perturbation, except predicted B share)</h3>
<table><tr><th></th>{''.join(f'<th>{t}</th>' for _, t in rob_metrics)}</tr>{rob_rows}</table>
<h3>Difference of drops between arms (drop<sub>a</sub> − drop<sub>b</sub>; negative = arm a degrades less)</h3>
<table><tr><th>a − b</th><th>Severity</th>{''.join(f'<th>{t}</th>' for _, t in rob_metrics)}</tr>{rob_pairs}</table>

<h2>Per patient (clean)</h2>
<p class="muted">Each cell: A1 / A2 / A3. Full table for all conditions: per_patient_metrics.csv.</p>
<table><tr><th>Patient</th><th>Cells</th><th>IF-B cells</th>{''.join(f'<th>{m}</th>' for m in ppm[1:])}</tr>{pp_rows}</table>

<h2>Limitations</h2>
<ul>
<li>best_model2.pth was trained on all 40 patients (see Caveats); A3 is scored on its training data and A1/A2 inherit that exposure through initialisation.</li>
<li>CRC11–CRC40 come from the same Immunocto CRC cohort, lab and scanner as the training patients; even ignoring the training overlap this would be in-domain, not a test on ESCA or other stains. The colour perturbations approximate stain shift within the augmentation family the fine-tune was trained on.</li>
<li>Immunocto labels are IF-derived and automatically transferred to H&amp;E nuclei; label noise (mis-registration, marker thresholds) caps every arm's attainable score equally.</li>
<li>Crops are 64-px Immunocto crops with Immunocto masks, not Cellpose crops from the HE2CELL pipeline; mask shape and scale differ slightly from deployment.</li>
<li>Subsampling to 50,000 cells per patient makes pooled prevalence differ from the cohort's crop-weighted prevalence (both are shown).</li>
</ul>

<h2>Provenance</h2>
<p class="muted">Code: deployment/cell_validation/immunocto_holdout_eval.py (index, verify, infer, metrics) and immunocto_holdout_report.py. Crop list: crops.csv.gz. Predictions (uncorrected D4 probabilities per checkpoint and condition, per patient): predictions/&lt;ft|orig&gt;_&lt;condition&gt;/CRC##.csv.gz.
Numbers: summary.json and per_patient_metrics.csv next to this report.</p>
</main></body></html>"""
    (OUT / "Cell_validation_report.html").write_text(page)
    print("wrote", OUT / "Cell_validation_report.html")


if __name__ == "__main__":
    main()
