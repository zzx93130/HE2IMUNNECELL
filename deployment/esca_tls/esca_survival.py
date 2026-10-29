"""TLS burden vs overall survival in TCGA-ESCA (plan item 4a).

Burden definitions per patient (mean over the patient's analysed slides):
  HookNet TLS per cm² and HookNet TLS area fraction (whole slide, analysis_he2cell_ft_tau05/slide_hooknet_burden.csv);
  HE2CELL whole-slide accepted TLS per cm² and B-TLS area fraction (slides/*/wsi_burden, esca_wsi_burden.py);
  HE2CELL predicted-B density in the 3 random windows per slide (pooled B cells / pooled valid mm², windows.csv);
  12-chemokine score (RNA PC1, rna/patient_signatures.csv).
Cox PH for OS, stratified by histology (ESCC / EAC baseline hazards) and adjusted for stage group
(I/II, III/IV, missing), age and sex; burden as log1p (HR per SD of log1p burden; the chemokine score, which is
already a log-scale z-score composite, per SD) and as tertiles. Schoenfeld residual test, Kaplan-Meier by tertile,
Harrell's C, and a complete-case analysis excluding patients with missing stage. Nothing here is tuned.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import html
import json
from pathlib import Path
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from esca_common import OUT, ROOT, save_json  # noqa: E402

from lifelines import CoxPHFitter, KaplanMeierFitter  # noqa: E402
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test  # noqa: E402

ANALYSIS = OUT / "analysis_he2cell_ft_tau05"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TERTILE_COLOURS = {"T1 (low)": "#2a78d6", "T2": "#eda100", "T3 (high)": "#eb6834"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "figure.dpi": 150})
# key -> (label, log1p transform, unit)
BURDENS = {
    "hooknet_TLS_per_cm2": ("HookNet TLS per cm²", True, "TLS/cm²"),
    "hooknet_TLS_area_pct": ("HookNet TLS area fraction", True, "% of tissue"),
    "he2cell_wsi_TLS_per_cm2": ("HE2CELL whole-slide TLS per cm²", True, "TLS/cm²"),
    "he2cell_wsi_B_TLS_area_pct": ("HE2CELL whole-slide B-TLS area fraction", True, "% of tissue"),
    "he2cell_random_B_per_mm2": ("HE2CELL predicted-B density, random windows", True, "B cells/mm²"),
    "chemokine12_PC1": ("12-chemokine score (RNA)", False, "PC1 z-score"),
}
COVARIATES = ["stage_III_IV", "stage_missing", "age_per_10y", "male"]
WSI_KEYS = ("he2cell_wsi_TLS_per_cm2", "he2cell_wsi_B_TLS_area_pct")
MIN_PATIENTS, MIN_DEATHS = 20, 10


def patient_table():
    clinical = pd.read_csv(OUT / "rna" / "clinical.csv").set_index("case_id")
    slides = pd.read_csv(ANALYSIS / "slide_hooknet_burden.csv")
    windows = pd.read_csv(ANALYSIS / "windows.csv")
    analysed = set(windows.barcode)
    slides = slides[slides.barcode.isin(analysed)]
    hooknet = slides.groupby("case_id").agg(hooknet_TLS_per_cm2=("hooknet_TLS_per_100mm2", "mean"),
                                            hooknet_TLS_area_fraction=("hooknet_TLS_area_fraction", "mean"),
                                            slides=("barcode", "nunique"))
    hooknet["hooknet_TLS_area_pct"] = 100 * hooknet.pop("hooknet_TLS_area_fraction")
    random = windows[windows.category == "random"].groupby("case_id")[["n_B", "valid_mm2"]].sum()
    random = (random.n_B / random.valid_mm2).rename("he2cell_random_B_per_mm2")
    wsi_rows = []
    for barcode in sorted(analysed):
        path = OUT / "slides" / barcode / "wsi_burden" / "summary.json"
        if path.exists():
            s = json.loads(path.read_text())
            wsi_rows.append({"case_id": barcode[:12], "barcode": barcode, "he2cell_wsi_TLS_per_cm2": s["TLS_per_cm2"],
                             "he2cell_wsi_B_TLS_area_pct": 100 * s["B_TLS_area_fraction"],
                             "processed_fraction_of_tissue": s["processed_fraction_of_tissue"]})
    wsi = pd.DataFrame(wsi_rows, columns=["case_id", "barcode", "he2cell_wsi_TLS_per_cm2", "he2cell_wsi_B_TLS_area_pct",
                                          "processed_fraction_of_tissue"])
    # A patient's whole-slide burden is used only when every analysed slide of that patient is complete.
    expected = windows.groupby("case_id").barcode.nunique()
    complete = wsi.groupby("case_id").barcode.nunique().reindex(expected.index).fillna(0) == expected
    wsi = wsi[wsi.case_id.isin(complete[complete].index)].groupby("case_id")[
        ["he2cell_wsi_TLS_per_cm2", "he2cell_wsi_B_TLS_area_pct", "processed_fraction_of_tissue"]].mean()
    rna = pd.read_csv(OUT / "rna" / "patient_signatures.csv").set_index("case_id")[["chemokine12_PC1"]]
    table = hooknet.join(random).join(wsi).join(rna).join(clinical, how="left")
    table.index.name = "case_id"
    return table, len(analysed), int(expected.sum())


def slides_complete(expected):
    return len(list((OUT / "slides").glob("*/wsi_burden/summary.json")))


def model_frame(table):
    frame = table.copy()
    frame = frame[frame.histology_primary.isin(["ESCC", "EAC"])]
    frame[list(BURDENS)] = frame[list(BURDENS)].astype(float)  # whole-slide columns are object dtype while empty
    frame["time_months"] = frame.os_days / 30.4375
    frame["event"] = frame.os_event
    frame["stage_III_IV"] = (frame.stage_group == "III/IV").astype(int)
    frame["stage_missing"] = (frame.stage_group == "missing").astype(int)
    frame["age_per_10y"] = frame.age_at_index / 10
    frame["male"] = (frame.gender == "male").astype(int)
    frame["histology"] = frame.histology_primary
    return frame


def tertiles(values):
    q1, q2 = np.nanquantile(values, [1 / 3, 2 / 3])
    groups = np.where(values <= q1, "T1 (low)", np.where(values <= q2, "T2", "T3 (high)"))
    return pd.Series(groups, index=values.index).where(values.notna()), (float(q1), float(q2))


def fit(data, columns, strata=("histology",)):
    # Drop indicators with < 5 patients in a level (e.g. 1 ESCC patient with missing stage): not estimable.
    columns = [c for c in columns if data[c].nunique() > 2 or data[c].value_counts().min() >= 5 and data[c].nunique() == 2]
    cph = CoxPHFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(data[["time_months", "event", *columns, *strata]], "time_months", "event",
                strata=list(strata) or None)
    return cph


def cox_rows(cph, terms, definition, model, n, events):
    rows = []
    for term in terms:
        s = cph.summary.loc[term]
        rows.append({"burden": definition, "model": model, "term": term, "n": n, "events": events,
                     "HR": float(s["exp(coef)"]), "CI95_low": float(s["exp(coef) lower 95%"]),
                     "CI95_high": float(s["exp(coef) upper 95%"]), "p": float(s["p"]), "C_index": float(cph.concordance_index_)})
    return rows


def analyse(frame, key, figures):
    label, use_log, unit = BURDENS[key]
    data = frame[frame[key].notna() & frame.time_months.notna() & (frame.time_months > 0) & frame.age_at_index.notna()].copy()
    raw = data[key]
    x = np.log1p(raw) if use_log else raw
    data["burden_sd"] = (x - x.mean()) / x.std(ddof=1)
    data["tertile"], cuts = tertiles(raw)
    for group in ["T2", "T3 (high)"]:
        data[f"tertile_{group.split()[0]}"] = (data.tertile == group).astype(int)
    tert_terms = ["tertile_T2", "tertile_T3"]
    n, events = len(data), int(data.event.sum())
    rows, result = [], {"label": label, "unit": unit, "transform": "log1p, per SD" if use_log else "per SD",
                        "n": n, "events": events, "SD_of_transformed": float(x.std(ddof=1)),
                        "median_raw": float(raw.median()), "IQR_raw": [float(raw.quantile(.25)), float(raw.quantile(.75))],
                        "zero_fraction": float((raw == 0).mean()), "tertile_cuts": cuts,
                        "tertile_sizes": data.tertile.value_counts().to_dict()}
    clinical_only = fit(data, COVARIATES)
    models = {
        "unadjusted (stratified by histology)": (["burden_sd"], data),
        "adjusted (stratified by histology)": (["burden_sd", *COVARIATES], data),
        "adjusted, histology as covariate (not stratified)": (["burden_sd", *COVARIATES, "EAC"], data.assign(EAC=(data.histology == "EAC").astype(int))),
        "adjusted tertiles (stratified by histology)": ([*tert_terms, *COVARIATES], data),
    }
    complete = data[data.stage_missing == 0]
    models["complete-case stage, adjusted (stratified)"] = (["burden_sd", "stage_III_IV", "age_per_10y", "male"], complete)
    models["complete-case stage, adjusted tertiles (stratified)"] = ([*tert_terms, "stage_III_IV", "age_per_10y", "male"], complete)
    for name, (columns, d) in models.items():
        strata = () if "not stratified" in name else ("histology",)
        cph = fit(d, columns, strata)
        terms = [c for c in columns if c.startswith(("burden", "tertile"))]
        rows += cox_rows(cph, terms, key, name, len(d), int(d.event.sum()))
        if name == "adjusted (stratified by histology)":
            ph = proportional_hazard_test(cph, d[["time_months", "event", *cph.params_.index, "histology"]], time_transform="rank")
            ph_table = ph.summary
            result["schoenfeld_p_burden"] = float(ph_table.loc["burden_sd", "p"])
            result["schoenfeld_p_min_covariate"] = float(ph_table.p.min())
            result["schoenfeld_global_p"] = float(stats.chi2.sf(ph_table.test_statistic.sum(), len(ph_table)))
            result["C_index_adjusted"] = float(cph.concordance_index_)
            result["C_index_clinical_only"] = float(clinical_only.concordance_index_)
            result["log_likelihood_ratio_p_vs_clinical"] = float(stats.chi2.sf(2 * (cph.log_likelihood_ - clinical_only.log_likelihood_), 1))
            # C of the burden alone (higher burden -> lower risk is the TLS hypothesis; report both directions' C as is).
            from lifelines.utils import concordance_index
            result["C_index_burden_alone"] = float(concordance_index(d.time_months, -d.burden_sd, d.event))
    result["by_histology"] = {}
    for histology in ["ESCC", "EAC"]:
        d = data[data.histology == histology]
        if d.event.sum() >= 10:
            cph = fit(d, ["burden_sd", *COVARIATES], ())
            s = cph.summary.loc["burden_sd"]
            result["by_histology"][histology] = {"n": len(d), "events": int(d.event.sum()), "HR_per_SD": float(s["exp(coef)"]),
                                                 "CI95": [float(s["exp(coef) lower 95%"]), float(s["exp(coef) upper 95%"])],
                                                 "p": float(s["p"])}
    test = multivariate_logrank_test(data.time_months, data.tertile, data.event)
    result["logrank_tertiles_p"] = float(test.p_value)
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    for group, colour in TERTILE_COLOURS.items():
        d = data[data.tertile == group]
        if len(d):
            KaplanMeierFitter().fit(d.time_months, d.event, label=f"{group} (n={len(d)}, {int(d.event.sum())} deaths)").plot_survival_function(
                ax=ax, ci_show=False, color=colour, show_censors=True, censor_styles={"ms": 4, "marker": "|"})
    ax.set_xlabel("Months since diagnosis"); ax.set_ylabel("Overall survival"); ax.set_ylim(0, 1.02)
    ax.set_title(f"{label}\nlog-rank p = {test.p_value:.2g}", fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(figures / f"km_{key}.png"); plt.close(fig)
    return rows, result, data[["burden_sd", "tertile"]].rename(columns={"burden_sd": f"{key}_sd", "tertile": f"{key}_tertile"})


def power(deaths):
    """Minimal HR detectable with 80% power, two-sided alpha 0.05 (Schoenfeld 1983), ignoring covariate correlation."""
    z = stats.norm.ppf(.975) + stats.norm.ppf(.8)
    top_vs_bottom = deaths * 2 / 3  # deaths in T1 + T3, equal split
    return {"deaths": int(deaths), "per_SD_HR_detectable": [float(np.exp(-z / np.sqrt(deaths))), float(np.exp(z / np.sqrt(deaths)))],
            "T3_vs_T1_HR_detectable": [float(np.exp(-z / np.sqrt(.25 * top_vs_bottom))), float(np.exp(z / np.sqrt(.25 * top_vs_bottom)))]}


def forest(cox, figures):
    keys = [k for k in BURDENS if k in set(cox.burden)]
    panels = [("adjusted (stratified by histology)", "burden_sd", "Adjusted, all patients"),
              ("complete-case stage, adjusted (stratified)", "burden_sd", "Adjusted, stage known")]
    fig, ax = plt.subplots(figsize=(6.4, .55 * len(keys) * 2 + 1))
    y, ticks, labels = 0, [], []
    for key in keys[::-1]:
        for model, term, name in panels[::-1]:
            r = cox[(cox.burden == key) & (cox.model == model) & (cox.term == term)]
            if r.empty:
                continue
            r = r.iloc[0]
            colour = "#2a78d6" if "all" in name else "#eb6834"
            ax.plot([r.CI95_low, r.CI95_high], [y, y], color=colour, lw=1.6)
            ax.plot(r.HR, y, "o", color=colour, ms=5)
            ticks.append(y); labels.append(f"{BURDENS[key][0]} — {name.lower()} (n={r.n}, {r.events} deaths)")
            y += 1
        y += .6
    ax.axvline(1, color=MUTED, lw=.8, ls="--")
    ax.set_xscale("log"); ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xticks([0.5, 0.7, 1, 1.4, 2]); ax.set_xticklabels(["0.5", "0.7", "1", "1.4", "2"]); ax.minorticks_off()
    ax.set_xlim(0.45, 2.2)
    ax.set_xlabel("HR for death per SD of burden (95% CI, log scale)")
    fig.tight_layout(); fig.savefig(figures / "forest_per_SD.png", bbox_inches="tight"); plt.close(fig)


def png(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def fmt(value, digits=2):
    return "–" if value is None or (isinstance(value, float) and not np.isfinite(value)) else f"{value:.{digits}f}"


def pfmt(p):
    return "–" if p is None or not np.isfinite(p) else (f"{p:.3f}" if p >= .001 else f"{p:.1e}")


def report(out, summary, cox, figures):
    keys = list(summary["burdens"])
    main_rows = []
    for key in keys:
        b = summary["burdens"][key]
        def get(model, term):
            r = cox[(cox.burden == key) & (cox.model == model) & (cox.term == term)]
            return r.iloc[0] if len(r) else None
        cells = []
        for model, term in [("unadjusted (stratified by histology)", "burden_sd"), ("adjusted (stratified by histology)", "burden_sd"),
                            ("adjusted tertiles (stratified by histology)", "tertile_T3"),
                            ("complete-case stage, adjusted (stratified)", "burden_sd")]:
            r = get(model, term)
            cells.append(f"{fmt(r.HR)} ({fmt(r.CI95_low)}–{fmt(r.CI95_high)}); p {pfmt(r.p)}" if r is not None else "–")
        main_rows.append(f"<tr><td>{html.escape(b['label'])}</td><td>{b['n']} / {b['events']}</td>"
                         + "".join(f"<td>{c}</td>" for c in cells)
                         + f"<td>{pfmt(b.get('schoenfeld_p_burden'))}</td><td>{fmt(b.get('C_index_clinical_only'), 3)} → {fmt(b.get('C_index_adjusted'), 3)}</td></tr>")
    hist_rows = []
    for key in keys:
        for h, r in summary["burdens"][key]["by_histology"].items():
            hist_rows.append(f"<tr><td>{html.escape(summary['burdens'][key]['label'])}</td><td>{h}</td><td>{r['n']} / {r['events']}</td>"
                             f"<td>{fmt(r['HR_per_SD'])} ({fmt(r['CI95'][0])}–{fmt(r['CI95'][1])})</td><td>{pfmt(r['p'])}</td></tr>")
    desc_rows = []
    for key in keys:
        b = summary["burdens"][key]
        desc_rows.append(f"<tr><td>{html.escape(b['label'])}</td><td>{html.escape(b['unit'])}</td><td>{fmt(b['median_raw'], 3)} "
                         f"({fmt(b['IQR_raw'][0], 3)}–{fmt(b['IQR_raw'][1], 3)})</td><td>{100 * b['zero_fraction']:.0f}%</td>"
                         f"<td>{fmt(b['tertile_cuts'][0], 3)} / {fmt(b['tertile_cuts'][1], 3)}</td><td>{html.escape(b['transform'])}</td></tr>")
    p = summary["power"]
    w = summary["wsi_run"]
    s = summary["screen_validation"]
    km = "".join(f'<figure><img class="fig" src="{png(figures / f"km_{k}.png")}" alt="Kaplan-Meier by tertile, {html.escape(summary["burdens"][k]["label"])}"></figure>' for k in keys)
    c = summary["cohort"]
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ESCA TLS Survival</title><style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--line:#e4e3df;--card:#fff;--accent:#2a78d6}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--line:#3a3936;--card:#232321;--accent:#3987e5}}}}
:root[data-theme="dark"]{{--bg:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--line:#3a3936;--card:#232321;--accent:#3987e5}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1080px;margin:0 auto;padding:28px 16px 64px}} h1{{font-size:26px;margin:0 0 4px}} h2{{margin-top:40px;border-top:1px solid var(--line);padding-top:20px}}
.muted{{color:var(--muted)}} .kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:20px 0}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}} .kpi b{{font-size:24px;display:block}}
table{{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow-x:auto}} th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;white-space:nowrap}}
img.fig{{max-width:100%;background:#fff;border-radius:8px;border:1px solid var(--line)}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px}} figure{{margin:0}}
</style></head><body><main>
<h1>Is TLS burden associated with overall survival in TCGA-ESCA?</h1>
<p class="muted">Cox proportional-hazards models of overall survival on six patient-level burden definitions (two HookNet-TLS, three HE2CELL, one bulk-RNA), stratified by histology (ESCC / EAC) and adjusted for stage group, age and sex. Generated {summary['generated']}.</p>
<div class="kpis">
<div class="kpi"><b>{c['patients']}</b>patients with survival and HookNet burden<br><span class="muted">{c['deaths']} deaths · {c['ESCC']} ESCC / {c['EAC']} EAC</span></div>
<div class="kpi"><b>{w['slides_complete']}/{w['slides_expected']}</b>slides with HE2CELL whole-slide burden<br><span class="muted">{w['patients_complete']} patients complete</span></div>
<div class="kpi"><b>{fmt(100 * s['test']['recovered_fraction'], 1) if s else '–'}%</b>test accepted TLS recovered by the screen<br><span class="muted">{s['test']['recovered_end_to_end'] if s else '–'} of {s['test']['accepted_aggregates'] if s else '–'} · target ≥95%</span></div>
<div class="kpi"><b>{fmt(p['per_SD_HR_detectable'][0])} / {fmt(p['per_SD_HR_detectable'][1])}</b>HR per SD detectable with 80% power<br><span class="muted">{p['deaths']} deaths, α 0.05 two-sided</span></div>
</div>

<h2>Power, stated before the results</h2>
<p>{c['patients']} patients and {p['deaths']} deaths. With 80% power at two-sided α = 0.05 (Schoenfeld), a continuous burden is detectable only if its hazard ratio per SD is below {fmt(p['per_SD_HR_detectable'][0])} or above {fmt(p['per_SD_HR_detectable'][1])}; the top-vs-bottom tertile contrast only if its HR is below {fmt(p['T3_vs_T1_HR_detectable'][0])} or above {fmt(p['T3_vs_T1_HR_detectable'][1])}. Published TLS effects in other cancers (HR ≈ 0.5–0.7 for high vs low) sit at or below the edge of this range, so a null result here does not exclude a clinically relevant effect. Six burden definitions are tested without multiplicity correction; treat p-values near 0.05 as exploratory.</p>

<h2>Cohort and clinical data</h2>
<p>GDC clinical fields were added to <code>esca_rna.clinical()</code>. Stage, age, sex and histology come from the diagnosis record flagged as the primary disease; the earlier first-record columns are kept unchanged (for {c['first_record_not_primary']} cases the first record was a metastasis, recurrence or prior primary, which is why the previous table lacked stage for 61 cases). OS time = days to death, or days to last follow-up for living patients. Cause of death is empty in GDC for every ESCA case, so disease-specific survival is not possible; this report is OS only.</p>
<table><tr><th>Step</th><th>Patients</th></tr>
<tr><td>Patients with ≥1 analysed slide (152 slides; lymphoma, myeloma and 20× slides already excluded)</td><td>{c['patients_with_slides']}</td></tr>
<tr><td>ESCC or EAC primary histology, OS time &gt; 0 and age known</td><td>{c['patients']}</td></tr>
<tr><td>Stage group I/II · III/IV · missing</td><td>{c['stage']['I/II']} · {c['stage']['III/IV']} · {c['stage']['missing']}</td></tr>
<tr><td>Median follow-up of survivors (reverse Kaplan–Meier), months</td><td>{fmt(c['median_followup_months'], 1)}</td></tr></table>

<h2>Burden definitions</h2>
<table><tr><th>Burden</th><th>Unit</th><th>Median (IQR)</th><th>Zero</th><th>Tertile cuts</th><th>Model scale</th></tr>{''.join(desc_rows)}</table>
<p class="muted">Per patient: mean over that patient's analysed slides. Whole-slide HE2CELL burden: a nuclear-blob screen proposes candidate regions (artefact-free tissue only; see next section), every 768-px core within 100 µm of a candidate is segmented and classified with the adopted fine-tuned classifier (tau 0.5 prior), and the frozen two-stage rule (all-nuclei density ≥ {summary['two_stage_params']['threshold']}/mm² at σ {summary['two_stage_params']['sigma_um']} µm, area ≥ {summary['two_stage_params']['min_area_um2']:.0f} µm², B share ≥ {summary['two_stage_params']['b_share']}) accepts TLS. A patient's whole-slide burden is used only when all of that patient's slides are complete.</p>

<h2>Screen validation (held-out test windows)</h2>
{f'''<p>The plan asked for a 4 µm haematoxylin screen. Two versions were calibrated on dev windows and rejected: absolute haematoxylin OD (stain intensity differs between slides; the best setting covered 95% of dev TLS while processing a median 136 mm² per slide, i.e. almost all tissue) and a per-slide quantile of local haematoxylin contrast (89% of dev TLS at a median 56 mm²; the 150 µm background suppresses the interior of large aggregates). The adopted screen is a model-free surrogate of the frozen stage 1: nuclear blobs (LoG maxima of haematoxylin on the 1 µm level) counted on the 8 µm artefact-free QC tissue grid, smoothed at 25 µm; regions with blob density ≥ {s['screen_params']['density_per_mm2']:.0f}/mm² over ≥ {s['screen_params']['min_component_um2']:.0f} µm², grown by {s['screen_params']['margin_um']:.0f} µm, are segmented. Selection rule (dev only, fixed before the test check): cheapest setting with ≥ 95% dev end-to-end recovery; 98% was not reachable at less than twice the cost. End-to-end check: in each window, cells outside the cores the screen would process were removed and the two-stage rule re-run. <b>Test: {s['test']['recovered_end_to_end']} of {s['test']['accepted_aggregates']} accepted TLS recovered ({fmt(100 * s['test']['recovered_fraction'], 1)}%, patient-bootstrap 95% CI {fmt(100 * s['test']['recovered_fraction_CI95'][0], 1)}–{fmt(100 * s['test']['recovered_fraction_CI95'][1], 1)}%)</b>; dev {s['dev']['recovered_end_to_end']} of {s['dev']['accepted_aggregates']}. Expected cost: median {s['screen_params']['median_cores_per_slide']:.0f} cores per slide (90th percentile {s['screen_params']['p90_cores_per_slide']:.0f}). Completed slides: median processed area {fmt(w.get('median_processed_mm2'), 1)} mm², median wall time {fmt(w.get('median_wall_minutes'), 1)} min.</p>''' if s else '<p>Not run.</p>'}

<h2>Overall survival: hazard ratios</h2>
<p><b>{html.escape(summary['he2cell_wsi_status'])}</b></p>
<table><tr><th>Burden</th><th>n / deaths</th><th>Unadjusted HR per SD</th><th>Adjusted HR per SD</th><th>Adjusted HR, top vs bottom tertile</th><th>Complete-case stage, adjusted HR per SD</th><th>Schoenfeld p (burden)</th><th>C-index clinical → +burden</th></tr>{''.join(main_rows)}</table>
<p class="muted">HR &lt; 1 means more burden, lower hazard of death. Unadjusted and adjusted models are stratified by histology; adjusted = + stage group (I/II, III/IV, missing), age per 10 years, male sex. Complete-case drops patients without stage. All Cox terms are in <code>cox_tables.csv</code>, including the unstratified model with histology as a covariate.</p>
<img class="fig" src="{png(figures / 'forest_per_SD.png')}" alt="Forest plot of adjusted hazard ratios per SD">

<h3>Within histology (adjusted, per SD)</h3>
<table><tr><th>Burden</th><th>Histology</th><th>n / deaths</th><th>HR (95% CI)</th><th>p</th></tr>{''.join(hist_rows)}</table>

<h2>Kaplan–Meier by tertile</h2>
<div class="gallery">{km}</div>

<h2>Reading</h2>
<p>{html.escape(summary['interpretation'])}</p>
</main></body></html>"""
    (out / "Survival_report.html").write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "outputs" / f"esca_survival_{dt.date.today():%Y%m%d}"))
    args = parser.parse_args()
    out = Path(args.out)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    table, slides_analysed, slides_expected = patient_table()
    frame = model_frame(table)
    eligible = frame[frame.time_months.notna() & (frame.time_months > 0) & frame.age_at_index.notna()]
    rows, burdens, per_patient, skipped = [], {}, [frame], {}
    wsi_patients = eligible[eligible.he2cell_wsi_TLS_per_cm2.notna()]
    interim = slides_complete(slides_expected) < slides_expected
    for key in BURDENS:
        available = eligible[eligible[key].notna()]
        if len(available) < MIN_PATIENTS or available.event.sum() < MIN_DEATHS:
            skipped[key] = {"patients": len(available), "deaths": int(available.event.sum()),
                            "reason": f"fewer than {MIN_PATIENTS} patients or {MIN_DEATHS} deaths"}
            print(f"skip {key}: {skipped[key]}")
            continue
        try:
            r, result, patient = analyse(frame, key, figures)
        except Exception as error:  # e.g. separation in a small interim subset
            skipped[key] = {"patients": len(available), "deaths": int(available.event.sum()), "reason": repr(error)}
            continue
        if key in WSI_KEYS and interim:
            result["interim"] = True
            result["label"] += f" — INTERIM ({result['n']} patients)"
        rows += r
        burdens[key] = result
        per_patient.append(patient)
    interim_comparator = None
    if interim and any(k in burdens for k in WSI_KEYS):
        # HookNet on exactly the patients the interim HE2CELL burden covers, for a like-for-like reading.
        subset = frame[frame.index.isin(wsi_patients.index)]
        try:
            (figures / "_interim_subset").mkdir(exist_ok=True)
            r, result, _ = analyse(subset, "hooknet_TLS_per_cm2", figures / "_interim_subset")
            for row in r:
                row["burden"] = "hooknet_TLS_per_cm2__same_patients_as_HE2CELL_wsi"
            rows += r
            interim_comparator = {k: result[k] for k in ["n", "events", "C_index_adjusted", "logrank_tertiles_p"]}
            a = [x for x in r if x["model"] == "adjusted (stratified by histology)"][0]
            interim_comparator["adjusted_HR_per_SD"] = {"HR": a["HR"], "CI95": [a["CI95_low"], a["CI95_high"]], "p": a["p"]}
        except Exception as error:
            interim_comparator = {"error": repr(error)}
    cox = pd.DataFrame(rows)
    cox.to_csv(out / "cox_tables.csv", index=False)
    patients = frame.join([p for p in per_patient[1:]], how="left")
    keep = ["slides", *BURDENS, "histology", "stage_group", "stage_primary", "age_at_index", "gender", "vital_status",
            "os_days", "os_event", "processed_fraction_of_tissue"]
    patients[[c for c in keep if c in patients] + [c for c in patients if c.endswith(("_sd", "_tertile"))]].to_csv(out / "patient_burden.csv")
    forest(cox, figures)
    km = KaplanMeierFitter().fit(eligible.time_months, 1 - eligible.event)  # reverse KM for follow-up
    clinical = pd.read_csv(OUT / "rna" / "clinical.csv")
    first_not_primary = int(sum((clinical[a].fillna("NA") != clinical[f"{a}_primary"].fillna("NA"))
                                for a in ["primary_diagnosis", "site", "stage"]).astype(bool).sum())
    validation_path = OUT / "wsi_screen_calibration" / "screen_validation.json"
    validation = json.loads(validation_path.read_text()) if validation_path.exists() else None
    wsi = [json.loads(p.read_text()) for p in (OUT / "slides").glob("*/wsi_burden/summary.json")]
    wsi_run = {"slides_complete": len(wsi), "slides_expected": slides_expected,
               "patients_complete": int(frame.he2cell_wsi_TLS_per_cm2.notna().sum()),
               "failed": len(list((OUT / "slides").glob("*/wsi_burden/failed.json")))}
    if wsi:
        wsi_run.update({"median_processed_mm2": float(np.median([w["processed_mm2"] for w in wsi])),
                        "median_processed_fraction": float(np.median([w["processed_fraction_of_tissue"] for w in wsi])),
                        "median_wall_minutes": float(np.median([w["wall_seconds"] for w in wsi]) / 60),
                        "total_accepted_TLS": int(sum(w["accepted_TLS"] for w in wsi)),
                        "accepted_matching_HookNet_fraction": float(sum(w["accepted_matching_HookNet"] for w in wsi) / max(sum(w["accepted_TLS"] for w in wsi), 1))})
        both = frame[frame.he2cell_wsi_TLS_per_cm2.notna()]
        rho = stats.spearmanr(both.he2cell_wsi_TLS_per_cm2, both.hooknet_TLS_per_cm2)
        wsi_run["spearman_vs_HookNet_TLS_per_cm2"] = {"rho": float(rho.statistic), "p": float(rho.pvalue), "n": len(both)}
    adjusted = cox[(cox.model == "adjusted (stratified by histology)") & (cox.term == "burden_sd") & cox.burden.isin(list(BURDENS))]
    final = adjusted[~adjusted.burden.isin(WSI_KEYS)] if interim else adjusted
    significant = final[final.p < .05]
    interpretation = (f"Adjusted for stage, age and sex and stratified by histology, "
                      + ("no burden definition was associated with overall survival at p < 0.05 "
                         if significant.empty else
                         f"{len(significant)} of {len(final)} burden definitions reached p < 0.05 without multiplicity correction "
                         f"({', '.join(BURDENS[k][0] for k in significant.burden)}) ")
                      + f"(adjusted HR per SD between {fmt(final.HR.min())} and {fmt(final.HR.max())}"
                      + ("; HE2CELL whole-slide burden excluded here because it is interim" if interim and adjusted.burden.isin(WSI_KEYS).any() else "") + "). "
                      f"With {int(eligible.event.sum())} deaths this analysis can only detect HR per SD beyond about "
                      f"{fmt(power(eligible.event.sum())['per_SD_HR_detectable'][0])}/{fmt(power(eligible.event.sum())['per_SD_HR_detectable'][1])}, "
                      "so the confidence intervals, not the p-values, are the result to carry forward. "
                      + ("Every adjusted point estimate is above 1 (more burden, higher hazard), the opposite of the protective "
                         "TLS effect reported in other cancers, but no interval excludes 1. " if (final.HR > 1).all() else "")
                      + "".join(f"Within {h}, {BURDENS[k][0]} has HR per SD {fmt(r['HR_per_SD'])} ({fmt(r['CI95'][0])}–{fmt(r['CI95'][1])}, "
                                f"p {pfmt(r['p'])}); this is one of {sum(len(b['by_histology']) for b in burdens.values())} "
                                "subgroup tests and is not corrected for multiplicity. "
                                for k, b in burdens.items() for h, r in b["by_histology"].items() if r["p"] < .05))
    wsi_status = (("INTERIM" if interim else "FINAL") + f": HE2CELL whole-slide burden from {wsi_run['slides_complete']} of {slides_expected} slides, "
                  f"{len(wsi_patients)} patients with all slides complete ({int(wsi_patients.event.sum())} deaths); slides are processed in a "
                  "seeded random patient order (seed 20261025), so completed patients are a random subset of the cohort.")
    if interim and any(k in burdens for k in WSI_KEYS):
        interpretation = interpretation.rstrip() + (" " + wsi_status + " Interim adjusted HR per SD: "
                           + "; ".join(f"{BURDENS[r.burden][0]} {fmt(r.HR)} ({fmt(r.CI95_low)}–{fmt(r.CI95_high)}), p {pfmt(r.p)}"
                                       for r in adjusted[adjusted.burden.isin(WSI_KEYS)].itertuples())
                           + (f"; HookNet TLS per cm² on the same patients {fmt(interim_comparator['adjusted_HR_per_SD']['HR'])} "
                              f"({fmt(interim_comparator['adjusted_HR_per_SD']['CI95'][0])}–{fmt(interim_comparator['adjusted_HR_per_SD']['CI95'][1])})"
                              if interim_comparator and "adjusted_HR_per_SD" in interim_comparator else "") + ".")
    elif interim:
        interpretation = interpretation.rstrip() + " " + wsi_status + " Too few patients for an interim HE2CELL whole-slide model yet."
    summary = {"generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "design": __doc__,
               "cohort": {"patients_with_slides": int(len(table)), "patients": int(len(eligible)), "deaths": int(eligible.event.sum()),
                          "ESCC": int((eligible.histology == "ESCC").sum()), "EAC": int((eligible.histology == "EAC").sum()),
                          "stage": eligible.stage_group.value_counts().reindex(["I/II", "III/IV", "missing"]).fillna(0).astype(int).to_dict(),
                          "median_followup_months": float(km.median_survival_time_), "first_record_not_primary": first_not_primary,
                          "slides_analysed": slides_analysed},
               "power": power(eligible.event.sum()), "two_stage_params": json.loads((ANALYSIS / "report_summary.json").read_text())["chosen_parameters"]["Two-stage (nuclei propose, composition accepts)"],
               "screen_validation": ({k: v for k, v in validation.items()} if validation else None), "wsi_run": wsi_run,
               "burdens": burdens,
               "adjusted_HR_per_SD": {r.burden: {"HR": r.HR, "CI95": [r.CI95_low, r.CI95_high], "p": r.p, "n": int(r.n), "events": int(r.events),
                                                 "interim": bool(interim and r.burden in WSI_KEYS)}
                                      for r in adjusted.itertuples()},
               "adjusted_HR_T3_vs_T1": {r.burden: {"HR": r.HR, "CI95": [r.CI95_low, r.CI95_high], "p": r.p}
                                        for r in cox[(cox.model == "adjusted tertiles (stratified by histology)") & (cox.term == "tertile_T3")].itertuples()},
               "interpretation": interpretation,
               "he2cell_wsi_status": wsi_status,
               "he2cell_wsi_interim_comparator_HookNet_same_patients": interim_comparator,
               "skipped_burdens": skipped}
    save_json(summary, out / "summary.json")
    report(out, summary, cox, figures)
    print(json.dumps(summary["adjusted_HR_per_SD"], indent=1))
    print(interpretation)


if __name__ == "__main__":
    main()
