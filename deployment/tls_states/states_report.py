"""HTML report and summary.json for the TLS-state analysis (Parts A-C).

Reads outputs/tls_states_20261026/parts/{esca_states, visium_states, survival_states}.json, the tables and the
figures written by esca_tls_states.py, visium_tls_states.py and esca_state_survival.py. Re-run after
esca_state_survival.py when the whole-slide run ends.
"""
from __future__ import annotations

import base64
import datetime as dt
import html
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import esca_tls_states as S  # noqa: E402

PROGRAM_ORDER = ["chemokine12", "GC", "Tfh", "plasma", "mature_DC", "cytotoxic", "checkpoint", "B_cell", "T_cell", "macrophage"]


def png(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def fmt(v, d=2):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def pfmt(p):
    return "–" if p is None or not np.isfinite(p) else (f"{p:.3f}" if p >= .001 else f"{p:.1e}")


def ci(v, d=2):
    return f"[{fmt(v[0], d)}, {fmt(v[1], d)}]" if v else "–"


def esc(s):
    return html.escape(str(s))


def table(header, rows):
    return ("<table><tr>" + "".join(f"<th>{esc(h)}</th>" for h in header) + "</tr>" +
            "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table>")


def figures_b(objects, path):
    plt = S._plt()
    pairs = [("B_share", "B_genes", "B share (H&E)", "MS4A1/CD79A/CD19"), ("T_share", "T_genes", "T share (H&E)", "CD3E/CD3D/CD2"),
             ("Macrophage_share", "Macrophage_genes", "Macrophage share (H&E)", "CD68/CD163/C1QA")]
    colours = dict(zip(sorted(objects["sample"].unique()), ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#b3261e", "#52514e", "#0fa3b1"]))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    for ax, (x, y, xl, yl) in zip(axes, pairs):
        for s, g in objects.groupby("sample"):
            ax.scatter(g[x], g[y], s=np.clip(g.spots * 6, 8, 80), color=colours[s], alpha=.75, label=s, lw=0)
        ax.set_xlabel(xl)
        ax.set_ylabel(f"log1p CP10k, {yl}")
    axes[0].legend(fontsize=6, frameon=False, ncol=2)
    fig.suptitle("Visium pathologist TLS objects (marker size = spots in object)", fontsize=9, x=.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def rna_heatmap(assoc, exposures, labels, path, title):
    plt = S._plt()
    a = assoc[(assoc.subset == "all") & assoc.exposure.isin(exposures)]
    m = a.pivot(index="exposure", columns="program", values="rho").reindex(index=exposures, columns=PROGRAM_ORDER)
    lo = a.pivot(index="exposure", columns="program", values="CI95").reindex(index=exposures, columns=PROGRAM_ORDER)
    fig, ax = plt.subplots(figsize=(8.2, .5 * len(exposures) + 1.6))
    im = ax.imshow(m.to_numpy(float), cmap="RdBu_r", vmin=-.5, vmax=.5, aspect="auto")
    for (i, j), v in np.ndenumerate(m.to_numpy(float)):
        c = lo.iat[i, j]
        star = "*" if c is not None and (c[0] > 0 or c[1] < 0) else ""
        ax.text(j, i, f"{v:+.2f}{star}", ha="center", va="center", fontsize=7)
    ax.set_yticks(range(len(exposures)))
    ax.set_yticklabels(labels)
    ax.set_xticks(range(len(PROGRAM_ORDER)))
    ax.set_xticklabels([S.PROGRAM_NAMES[p] for p in PROGRAM_ORDER], rotation=40, ha="right")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=.025, label="Spearman ρ")
    ax.set_title(title + "  (* bootstrap 95% CI excludes 0)", fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def key_findings(a, b, c, model):
    """Short, number-driven summary (recomputed on every run)."""
    out = []
    prof = {w: {p["state_name"]: p for p in a[w]["profiles"]} for w in ["primary", "k3"]}
    def comp(p):
        return f"B {p['median_B_share']:.0%}, T {p['median_T_share']:.0%}, macrophage {p['median_Macrophage_share']:.1%}, other {p['median_Other_share']:.0%}"
    out.append(f"BIC picks k = {model['primary']['k']} ({model['primary']['covariance_type']}): " + "; ".join(
        f"<b>{esc(n)}</b> {p['objects']} TLS ({comp(p)})" for n, p in prof["primary"].items()) + ". Fixed k = 3: " + "; ".join(
        f"<b>{esc(n)}</b> {p['objects']} ({comp(p)})" for n, p in prof["k3"].items()) + ".")
    st = a["stability"]
    out.append(f"The partitions are not stable under patient resampling (median ARI {st['primary']['median_ARI']:.2f} for the primary model, "
               f"{st['k3']['median_ARI']:.2f} for k = 3): TLS composition behaves like a continuum, and the frozen states are reproducible "
               "labels on it (deterministic given the JSON), not discrete biological types.")
    gc = {n: p["with_HookNet_GC_1pct"] for n, p in prof["k3"].items()}
    out.append("HookNet germinal-centre pixels (≥ 1% of the TLS) are not most frequent in the B-follicular state: " +
               ", ".join(f"{esc(n)} {v:.0%}" for n, v in gc.items()) +
               ". Composition states do not encode germinal-centre maturity (as in item 5a): 'B-follicular' means a B-rich aggregate with B concentrated in its core, not a secondary follicle.")
    h = {r["exposure"]: r for r in a["histology"]}
    k3b, k3t = h.get(f"k3frac_{model['k3']['state_names'][0]}"), h.get("k3frac_T-dominant")
    if k3b and k3t:
        out.append(f"Histology (k = 3, patient means of window state fractions): B-follicular EAC {k3b['EAC_mean']:.2f} vs ESCC {k3b['ESCC_mean']:.2f} "
                   f"(difference {k3b['EAC_minus_ESCC']:+.2f} {ci(k3b['CI95'])}); T-dominant EAC {k3t['EAC_mean']:.2f} vs ESCC {k3t['ESCC_mean']:.2f} "
                   f"({k3t['EAC_minus_ESCC']:+.2f} {ci(k3t['CI95'])}).")
    assoc = pd.DataFrame(a["rna_associations"])
    allp = assoc[assoc.subset == "all"]
    sig = allp[allp.CI95.apply(lambda v: v[0] > 0 or v[1] < 0) & allp.exposure.str.startswith(("frac_", "k3frac_"))]
    bf = allp[allp.exposure == f"frac_{model['primary']['state_names'][0]}"]
    out.append(f"Bulk RNA (patient-level, window-sampled, n = {int(bf.n.max())}): the primary B-follicular fraction is not associated with any program "
               f"(|ρ| ≤ {bf.rho.abs().max():.2f}). Associations whose bootstrap CI excludes 0: " +
               ("; ".join(f"{esc(r.exposure.replace('k3frac_', 'k=3 ').replace('frac_', ''))} vs {esc(S.PROGRAM_NAMES[r.program])} ρ {r.rho:+.2f} {ci(r.CI95)}"
                          for r in sig.itertuples()) or "none") + " (uncorrected).")
    if b:
        comp_ = {(r["level"], r["x"]): r for r in b["composition_vs_genes"]}
        o = [comp_[("object", x)] for x in ["B_share", "T_share", "Macrophage_share"]]
        earlier = {e["score"]: e for e in b["earlier_spot_level"]}
        out.append(f"Visium, {b['objects']} pathologist TLS objects: B share vs MS4A1/CD79A/CD19 ρ {o[0]['rho']:+.2f} {ci(o[0]['CI95'])} pooled, "
                   f"median within-section ρ {o[0]['median_within_section_rho']:+.2f} (earlier all-spot result {earlier['B']['median_spearman_across_samples']:.2f}); "
                   f"T share vs CD3E/CD3D/CD2 {o[1]['rho']:+.2f} {ci(o[1]['CI95'])} (within {o[1]['median_within_section_rho']:+.2f}); "
                   f"macrophage share vs CD68/CD163/C1QA {o[2]['rho']:+.2f} {ci(o[2]['CI95'])} (within {o[2]['median_within_section_rho']:+.2f}). "
                   f"Only B agrees, and mainly within sections; at single spots inside TLS (median {b['median_umi_per_spot']:.0f} UMI) there is no signal.")
        pk = [r for r in b["programs_vs_state"] if r["model"] == "agg_k3"]
        best = max(pk, key=lambda r: abs(r["rho"]))
        out.append(f"The frozen model labels {b['state_counts']['agg_primary'].get(model['primary']['state_names'][0], 0)}/{b['objects']} Visium objects B-follicular (primary). "
                   f"With k = 3, the B-follicular posterior tracks expression programs weakly (strongest: {esc(S.PROGRAM_NAMES.get(best['program'], best['program']))} "
                   f"ρ {best['rho']:+.2f} {ci(best['CI95'])}).")
    if c:
        p = c["primary"]
        out.append(f"Survival ({esc(c['label'])}): " + "; ".join(
            f"{h} HR/SD {v['HR']:.2f} {ci(v['CI95'])}, p {pfmt(v['p'])}" if v else f"{h} not estimable yet" for h, v in p.items()) + ".")
    return out


def main():
    parts = {k: json.loads((S.PARTS / f"{k}.json").read_text()) for k in ["esca_states", "visium_states", "survival_states"]
             if (S.PARTS / f"{k}.json").exists()}
    a, b, c = parts["esca_states"], parts.get("visium_states"), parts.get("survival_states")
    model = S.load_model()
    fig = S.FIG
    # ---------------- Part A tables
    grid_rows = [[r["k"], r["covariance_type"], f"{r['BIC']:.0f}", r["n_parameters"], f"{r['min_state_fraction']:.0%}",
                  "yes" if r["eligible"] else "no"] for r in a["model_selection"]]
    def state_rows(which):
        return [[f"<b>{esc(p['state_name'])}</b>", p["objects"], f"{p['fraction']:.0%}", p["patients"],
                 f"{p['median_B_share']:.0%}", f"{p['median_T_share']:.0%}", f"{p['median_Macrophage_share']:.1%}",
                 f"{p['median_Other_share']:.0%}", f"{p['median_area_um2'] / 1e3:.0f}k", f"{p['median_nuclei_per_mm2']:,.0f}",
                 fmt(p["median_B_core_enrichment"]), fmt(p["median_T_rim_enrichment"]), fmt(p["median_B_radial_gradient"]),
                 fmt(p["median_solidity"]), fmt(p["median_compactness"]), f"{p['with_HookNet_GC_1pct']:.0%}", f"{p['matches_HookNet']:.0%}"]
                for p in a[which]["profiles"]]
    state_header = ["State", "TLS", "share", "patients", "B", "T", "Mac", "Other", "area µm²", "nuclei/mm²", "B-core enr.",
                    "T-rim enr.", "B radial grad.", "solidity", "compactness", "HookNet GC ≥1%", "HookNet agrees"]
    hist_rows = [[esc(r["exposure"].replace("frac_", "fraction ").replace("k3frac_", "k=3 fraction ")), r["ESCC_n"], r["EAC_n"],
                  fmt(r["ESCC_mean"]), fmt(r["EAC_mean"]), f"{fmt(r['EAC_minus_ESCC'])} {ci(r['CI95'])}", pfmt(r["mann_whitney_p"])]
                 for r in a["histology"]]
    assoc = pd.DataFrame(a["rna_associations"])
    names_p, names_k3 = model["primary"]["state_names"], model["k3"]["state_names"]
    exposures = [f"frac_{names_p[0]}", *[f"k3frac_{n}" for n in names_k3], "mean_p_B_follicular"]
    labels = [f"Fraction {names_p[0]} (primary, k={model['primary']['k']})", *[f"Fraction {n} (k=3)" for n in names_k3], "Mean B-follicular posterior"]
    rna_heatmap(assoc, exposures, labels, fig / "esca_rna_heatmap.png", "ESCA patients: TLS-state fraction vs bulk RNA program (all patients)")
    key_assoc = assoc[(assoc.exposure.isin([f"frac_{names_p[0]}", f"k3frac_{names_k3[0]}"]))]
    assoc_rows = [[esc(r.exposure.replace("k3frac_", "k=3 ").replace("frac_", "primary ")), esc(S.PROGRAM_NAMES[r.program]), r.subset, r.n,
                   f"{fmt(r.rho)} {ci(r.CI95)}", pfmt(r.p)] for r in key_assoc.itertuples() if r.program in PROGRAM_ORDER]
    gallery_html = ""
    for state, tiles in (a.get("gallery") or {}).items():
        gallery_html += f"<h4>{esc(state)}</h4><div class='gallery'>" + "".join(
            f"<figure><img src='{png(S.OUT / t['path'])}'><figcaption>{esc(t['caption'])}</figcaption></figure>" for t in tiles) + "</div>"
    # ---------------- Part B
    b_html = "<p class='muted'>Part B not run.</p>"
    if b:
        objects = pd.read_csv(S.TABLES / "visium_tls_objects.csv")
        figures_b(objects, fig / "visium_object_scatter.png")
        comp_rows = [[esc(r["level"]), esc(r["x"]), esc(r["y"]), r["n"], f"<b>{fmt(r['rho'])}</b> {ci(r['CI95'])}", pfmt(r["p"]),
                      fmt(r["median_within_section_rho"]), esc(", ".join(f"{v:+.2f}" for v in r["within_section_rho"]))]
                     for r in b["composition_vs_genes"]]
        earlier = {e["score"] + "|" + e["genes"]: e for e in b["earlier_spot_level"]}
        e_b, e_t = earlier.get("B|MS4A1+CD79A+CD19", {}), earlier.get("T|CD3E+CD3D", {})
        prog_rows = []
        for r in b["programs_vs_state"]:
            if r["model"] != "agg_primary":
                continue
            med = r["median_by_state"]
            prog_rows.append([esc(S.PROGRAM_NAMES.get(r["program"], r["program"])),
                              *[fmt(med.get(n)) for n in names_p], f"{fmt(r['rho'])} {ci(r['CI95'])}", fmt(r["median_within_section_rho"]), pfmt(r["kruskal_p"])])
        prog_k3 = []
        for r in b["programs_vs_state"]:
            if r["model"] != "agg_k3":
                continue
            med = r["median_by_state"]
            prog_k3.append([esc(S.PROGRAM_NAMES.get(r["program"], r["program"])), *[fmt(med.get(n)) for n in names_k3],
                            f"{fmt(r['rho'])} {ci(r['CI95'])}", pfmt(r["kruskal_p"])])
        comp_prog = [[esc(S.PROGRAM_NAMES.get(r["program"], r["program"])), esc(r["x"]), f"{fmt(r['rho'])} {ci(r['CI95'])}", fmt(r["median_within_section_rho"])]
                     for r in b.get("programs_vs_composition", [])]
        mat = b["state_vs_pathologist_maturity"]["agg_primary"]
        mat_rows = [[esc(col), *[mat[col].get(n, 0) for n in names_p]] for col in mat]
        auc = b["AUROC_pBfol_for_mature"]
        b_html = f"""
<p>{b['objects']} pathologist TLS objects on {b['sections']} sections ({b['mature_objects']} mature = any MTLS/GC spot), {b['spots_in_objects']} spots inside them. Expression: {esc(b['expression_source'])}. Sequencing depth is low: median {b['median_umi_per_spot']:.0f} UMI per spot, {b['median_umi_per_object']:.0f} per object. Cells: {esc(b['classifier'])}. State features were measured on {esc(', '.join(f'{k}: {v}' for k, v in b['aggregate_source'].items()))}.</p>
<h3>Composition vs marker genes</h3>
{table(['Level', 'H&E', 'Genes', 'n', 'Spearman ρ pooled [section bootstrap CI]', 'p', 'median within-section ρ', 'per section'], comp_rows)}
<p>Earlier spot-level result (all in-tissue spots, <code>visium_tls_validation/analysis/results.json</code>, same classifier): B cells vs MS4A1+CD79A+CD19 median within-section ρ {fmt(e_b.get('median_spearman_across_samples'))}; T cells vs CD3E+CD3D {fmt(e_t.get('median_spearman_across_samples'))}. The rows above restrict to pathologist TLS and aggregate over each object.</p>
<img class="fig" src="{png(fig / 'visium_object_scatter.png')}">
<h3>Frozen ESCA state model on Visium objects</h3>
<p>Assigned states (aggregate features): {esc(b['state_counts']['agg_primary'])}; k=3: {esc(b['state_counts']['agg_k3'])}. Aggregate vs spot-tile features give the same primary state for {b['agg_vs_tile_state_agreement']:.0%} of objects. Program score = mean per-gene z across objects (12-chemokine: mean z of 11 genes; CCL3 absent from the Visium panel). ρ is the Spearman correlation of the program with the B-follicular posterior.</p>
{table(['Program', *[f'median, {n}' for n in names_p], 'ρ vs p(B-follicular) [CI]', 'median within-section ρ', 'Kruskal p'], prog_rows)}
<h4>k = 3 model</h4>
{table(['Program', *[f'median, {n}' for n in names_k3], 'ρ vs p(B-follicular) [CI]', 'Kruskal p'], prog_k3)}
<h4>Programs vs continuous composition (object level)</h4>
{table(['Program', 'H&E share', 'ρ [CI]', 'median within-section ρ'], comp_prog)}
<h4>States vs pathologist maturity</h4>
{table(['Pathologist', *names_p], mat_rows)}
<p>AUROC of the B-follicular posterior for mature (MTLS/GC) vs early: primary {fmt(auc['agg_primary']['AUROC'])} {ci(auc['agg_primary']['CI95'])}; k=3 {fmt(auc['agg_k3']['AUROC'])} {ci(auc['agg_k3']['CI95'])}; spot-tile features {fmt(auc['tile_primary']['AUROC'])} {ci(auc['tile_primary']['CI95'])}.</p>
<p class="muted">Resolution caveat: the Visium H&E is 0.79 µm/px, bicubically upsampled to 0.25 µm/px before Cellpose and the classifier; nuclei are softer than in 0.25 µm/px TCGA scans, and Visium objects are bounded by 100 µm-pitch spot tiles, so shape features are coarse. The state model was fitted on ESCA only; the Visium objects include lung and kidney cancer.</p>"""
    # ---------------- Part C
    c_html = "<p class='muted'>Part C not run.</p>"
    if c:
        pre = c["preregistration"]
        pre_rows = [[h, esc(v["definition"]), esc(v["expected_direction"])] for h, v in pre["primary_hypotheses"].items()]
        prim_rows = []
        for h, v in c["primary"].items():
            spec = pre["primary_hypotheses"][h]
            prim_rows.append([h, esc(spec["exposure"]), "–" if not v else f"{v['n']} / {v['events']}",
                              "not estimable (too few patients/deaths)" if not v else f"{fmt(v['HR'])} {ci(v['CI95'])}",
                              "–" if not v else pfmt(v["p"]), "–" if not v else pfmt(v.get("p_holm"))])
        cox = pd.read_csv(S.OUT / "survival" / "cox_tables.csv") if (S.OUT / "survival" / "cox_tables.csv").exists() else pd.DataFrame()
        sec_rows = []
        for key, r in c["results"].items():
            sub = cox[(cox.burden == key)] if len(cox) else cox
            def get(model, term):
                x = sub[(sub.model == model) & (sub.term == term)]
                return f"{fmt(x.HR.iloc[0])} {ci([x.CI95_low.iloc[0], x.CI95_high.iloc[0]])}; p {pfmt(x.p.iloc[0])}" if len(x) else "–"
            sec_rows.append([esc(r["label"]), f"{r['n']} / {r['events']}", get("unadjusted (stratified by histology)", "burden_sd"),
                             get("adjusted (stratified by histology)", "burden_sd"), get("adjusted tertiles (stratified by histology)", "tertile_T3"),
                             pfmt(r.get("schoenfeld_p_burden")), pfmt(r.get("logrank_tertiles_p"))])
        skipped = "; ".join(f"{k}: {v['patients']} patients, {v['deaths']} deaths ({v['reason']})" for k, v in c["skipped"].items())
        kms = "".join(f"<figure><img src='{png(S.OUT / 'survival' / 'figures' / f'km_{k}.png')}'></figure>"
                      for k in ["total_TLS_per_cm2", "Bfol_TLS_per_cm2", "Bfol_fraction"] if (S.OUT / "survival" / "figures" / f"km_{k}.png").exists())
        powr = c.get("power") or {}
        c_html = f"""
<div class="verdict"><b>{esc(c['label'])}</b>. Whole-slide accepted TLS so far: {c['accepted_wsi_TLS']} ({esc(c['wsi_TLS_by_state'])}; k=3: {esc(c['wsi_TLS_by_state_k3'])}).
{'' if c['final'] else 'These numbers are descriptive only and will change; slides are processed in a seeded random patient order, so the completed prefix is an unbiased subset.'}</div>
<h3>Pre-registered hypotheses (fixed {esc(pre['date_fixed'])}, before {esc(pre['fixed_before'])}; file <code>preregistration.json</code>, written {esc(pre.get('written', ''))})</h3>
{table(['', 'Exposure', 'Expected'], pre_rows)}
<p class="muted">Model: {esc(pre['model'])}. Multiplicity: {esc(pre['multiplicity'])}. State model: {esc(pre['state_model'])}. Secondary: {esc('; '.join(pre['secondary']))}. {esc(pre['interim_rule'])}.</p>
<h3>Primary results</h3>
{table(['', 'Exposure', 'n / deaths', 'Adjusted HR per SD [95% CI]', 'p', 'Holm p'], prim_rows)}
<p class="muted">Power at this death count: HR per SD detectable with 80% power ≤ {fmt((powr.get('per_SD_HR_detectable') or [np.nan])[0])} or ≥ {fmt((powr.get('per_SD_HR_detectable') or [np.nan, np.nan])[1])}.</p>
<h3>All fitted exposures</h3>
{table(['Exposure', 'n / deaths', 'Unadjusted HR/SD', 'Adjusted HR/SD', 'Adjusted T3 vs T1', 'Schoenfeld p', 'Log-rank tertiles p'], sec_rows)}
<p class="muted">Not fitted: {esc(skipped) or 'none'}.</p>
<div class="gallery">{kms}</div>"""
    findings = key_findings(a, b, c, model)
    stab = a["stability"]
    gcs = a["gc_feature_sensitivity"]
    p_names = ", ".join(f"<b>{esc(n)}</b>" for n in names_p)
    html_text = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TLS States Report</title><style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--line:#e4e3df;--card:#fff;--accent:#2a78d6}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--line:#3a3936;--card:#232321;--accent:#7fb2ee}}}}
:root[data-theme="dark"]{{--bg:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--line:#3a3936;--card:#232321;--accent:#7fb2ee}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1080px;margin:0 auto;padding:28px 16px 64px}} h1{{font-size:26px;margin:0 0 4px}} h2{{margin-top:36px;border-top:1px solid var(--line);padding-top:18px}}
.muted{{color:var(--muted)}} table{{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow-x:auto;margin:8px 0}}
th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;white-space:nowrap}} img.fig{{max-width:100%;background:#fff;border-radius:8px;border:1px solid var(--line);margin:6px 0}}
.verdict{{border:2px solid var(--accent);border-radius:8px;padding:12px 16px;background:var(--card)}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:10px}} figure{{margin:0}} figure img{{width:100%;border-radius:6px;background:#fff}} figcaption{{font-size:11px;color:var(--muted)}}
</style></head><body><main>
<h1>TLS states: composition, function and prognosis hook</h1>
<p class="muted">HE2CELL accepted two-stage TLS (fine-tuned classifier + τ 0.5 prior, frozen ESCA parameters). Generated {dt.date.today().isoformat()}.</p>
<div class="verdict">Primary state model (BIC, k = {model['primary']['k']}, {model['primary']['covariance_type']} covariance): {p_names}. Fixed k = 3: {', '.join(esc(n) for n in names_k3)}.
Bootstrap stability (patient resamples, ARI with the frozen assignment): primary median {fmt(stab['primary']['median_ARI'])}, k = 3 median {fmt(stab['k3']['median_ARI'])}.
{'' if not c else 'Survival: ' + esc(c['label']) + '.'}</div>

<h2>Key findings</h2>
<ul>{''.join(f'<li>{f}</li>' for f in findings)}</ul>

<h2>A. TCGA-ESCA TLS states</h2>
<p>Objects: the {a['objects']} accepted two-stage aggregates in the ESCA windows (dev {a['by_split'].get('dev', 0)}, test {a['by_split'].get('test', 0)}; {a['patients']} patients, {a['slides']} slides; window category {esc(a['by_window_category'])}). Stage 1: σ {a['stage1']['sigma_um']} µm, ≥ {a['stage1']['threshold']:,} nuclei/mm², ≥ {a['stage1']['min_area_um2']:,} µm²; stage 2: B share ≥ {a['b_share_cut']:.0%}. Per TLS: area, cells, nuclear density, B/T/macrophage/other shares, B-core and T-rim enrichment (as in <code>esca_analysis.proposal_features</code>), radial B/T/macrophage density in 30 µm rings (0–150 µm from the centroid, artefact-free tissue inside and outside the TLS), compactness and solidity of the hole-filled mask, and HookNet GC pixel fraction.</p>
<p>Clustering features (pre-specified, standardised, medians imputed): {esc(', '.join(S.FEATURE_NAMES[f] for f in S.STATE_FEATURES))}. HookNet GC pixel fraction was kept out of the clustering (zero-inflated output of an independent model) and is used to describe states; adding it as a 12th feature changes the primary assignment little (ARI {fmt(gcs['ARI_vs_primary'])}). The mixture is fitted by EM in numpy (scikit-learn is not installed in <code>.venv</code>); the frozen model is <code>tls_state_model.json</code> (+ <code>.pkl</code>) and <code>esca_tls_states.assign_states()</code> applies it. State names follow a fixed rule on the state means: {esc(model['naming_rule'])}</p>
<h3>Model selection</h3>
{table(['k', 'covariance', 'BIC', 'parameters', 'smallest state', '≥ 10% each'], grid_rows)}
<h3>Primary states (k = {model['primary']['k']})</h3>
{table(state_header, state_rows('primary'))}
<img class="fig" src="{png(fig / 'heatmap_primary.png')}">
<img class="fig" src="{png(fig / 'radial_primary.png')}">
<h3>Fixed k = 3 (interpretability)</h3>
{table(state_header, state_rows('k3'))}
<img class="fig" src="{png(fig / 'heatmap_k3.png')}">
<img class="fig" src="{png(fig / 'radial_k3.png')}">
<h3>Stability</h3>
<p>Each eligible (k, covariance) was refitted on 100 patient-cluster bootstrap resamples; the refit's assignment of all {a['objects']} TLS is compared with the full-cohort fit (adjusted Rand index; 1 = identical partition, 0 = chance).</p>
{table(['k', 'covariance', 'BIC', 'median ARI', 'IQR'], [[r['k'], r['covariance_type'], f"{r['BIC']:.0f}", fmt(r['median_ARI']), ci([r['IQR_low'], r['IQR_high']])] for r in a.get('stability_grid', [])])}
<p class="muted">Primary vs k = 3 ARI {fmt(a['primary_vs_k3_ARI'])}; cross-tab {esc(a['primary_vs_k3_crosstab'])}. Bootstrap ARI IQR: primary {ci(stab['primary']['IQR_ARI'])}, k = 3 {ci(stab['k3']['IQR_ARI'])} ({stab['primary']['resamples']} resamples).</p>
<h3>Examples (highest posterior, one per patient)</h3>
<p class="muted"><code>esca_report.gallery_tile</code>: black = the TLS, green = HookNet-TLS, orange dots = predicted B, blue dots = predicted T.</p>
{gallery_html}
<h3>Histology (patient-level state fractions)</h3>
{table(['State fraction', 'ESCC n', 'EAC n', 'ESCC mean', 'EAC mean', 'EAC − ESCC [patient bootstrap CI]', 'Mann–Whitney p'], hist_rows)}
<p class="muted">TLS-level counts by histology: {esc(a['aggregate_state_by_histology'])}.</p>
<h3>Bulk RNA programs</h3>
<p>Programs are patient medians of log2(TPM+1) over primary-tumour aliquots (TCGA-ESCA STAR TPM, GDC Xena), z-scored within ESCA, averaged over genes: GC = AICDA, BCL6, RGS13, MKI67; Tfh = CXCL13, PDCD1, ICOS, CXCR5; plasma = MZB1, JCHAIN, IGHG1; mature DC = LAMP3, CCL19, CCL21; cytotoxic = CD8A, GZMB, PRF1; checkpoint = PDCD1, CTLA4, LAG3, HAVCR2, CD274; 12-chemokine = existing PC1. These are <b>patient-level, window-sampled associations</b>: state fractions come from at most 8 windows per slide, mostly centred on HookNet TLS, and {a['patients_with_aggregates']} patients have any accepted window TLS; bulk RNA is a different tissue piece. Spearman with 2000 patient bootstrap resamples; no multiplicity correction.</p>
<img class="fig" src="{png(fig / 'esca_rna_heatmap.png')}">
{table(['Exposure', 'Program', 'Subset', 'n', 'ρ [CI]', 'p'], assoc_rows)}

<h2>B. Visium functional validation</h2>
{b_html}

<h2>C. Prognosis hook: TLS states and overall survival</h2>
{c_html}

<h2>Files and commands</h2>
<pre>.venv/bin/python deployment/tls_states/esca_tls_states.py      # Part A (features cached; --recompute)
.venv/bin/python deployment/tls_states/visium_tls_states.py     # Part B
.venv/bin/python deployment/tls_states/esca_state_survival.py   # Part C (re-run when the whole-slide run ends)
.venv/bin/python deployment/tls_states/states_report.py         # this report + summary.json</pre>
<p class="muted">Tables in <code>outputs/tls_states_20261026/tables/</code>; survival in <code>survival/</code>; frozen model <code>tls_state_model.json</code>.</p>
</main></body></html>"""
    (S.OUT / "TLS_states_report.html").write_text(html_text)
    summary = {"date": dt.date.today().isoformat(), "state_model": {w: {"k": model[w]["k"], "covariance_type": model[w]["covariance_type"],
                                                                         "state_names": model[w]["state_names"]} for w in ["primary", "k3"]},
               "esca": {k: a[k] for k in ["objects", "patients", "slides", "primary_vs_k3_ARI", "stability", "gc_feature_sensitivity", "histology"]},
               "esca_profiles": {w: [{k: p[k] for k in ["state_name", "objects", "fraction", "patients", "median_B_share", "median_T_share",
                                                         "median_Macrophage_share", "median_Other_share", "median_area_um2", "median_nuclei_per_mm2",
                                                         "median_B_core_enrichment", "median_solidity", "with_HookNet_GC_1pct"]} for p in a[w]["profiles"]]
                                 for w in ["primary", "k3"]},
               "esca_rna_key": key_assoc[key_assoc.subset == "all"].to_dict("records"),
               "visium": None if not b else {k: b[k] for k in ["objects", "sections", "composition_vs_genes", "earlier_spot_level", "state_counts",
                                                               "AUROC_pBfol_for_mature", "median_umi_per_spot", "median_umi_per_object"]}
               | ({"programs_vs_state_primary": [r for r in b["programs_vs_state"] if r["model"] == "agg_primary"]} if b else {}),
               "survival": c}
    S.save_json(summary, S.OUT / "summary.json")
    print("report written:", S.OUT / "TLS_states_report.html")


if __name__ == "__main__":
    main()
