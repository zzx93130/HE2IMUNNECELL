"""TNBC_ST report: outputs/tnbc_st_20261101/{TNBC_report.html, summary.json} from the Stage 0-4 outputs.

Cohort: frozen-section ST H&E (Zenodo 14204217, Venet 2024; outside HE2CELL's FFPE training domain) with
serial-section CD3/CD20 IHC, same-section pathologist annotations and ST spot RNA. CIs are patient-cluster
bootstraps (2000 resamples); paired differences and sign-flip p from esca_paired_stats. "Better" is used only when
the paired CI excludes 0 and p < 0.05. p values are nominal except the Holm-adjusted primary family.
"""
from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tnbc_common as C  # noqa: E402

A = C.OUT / "analysis"
CAVEAT = ("Precision caveat: the 'inflamed' negatives (Lymphocyte, High TIL stroma) contain IHC CD20 follicles "
          "(e.g. TNBC45: 48 IHC aggregates, no Lymphoid-nodule pixel), so precision and F1 against Lymphoid nodules are pessimistic for every arm.")
TWO, BD = "Two-stage (HE2CELL)", "B density (HE2CELL)"
N11, N16, HK, HN = "Nuclei only (frozen ESCA, 11000)", "Nuclei only (extended grid, 16000)", "HookNet-TLS", "HoVer-Net inflammatory density"


def load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def fmt(v, ci=None, d=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    s = f"{v:.{d}f}"
    if ci is not None and all(c is not None and np.isfinite(c) for c in ci):
        s += f" [{ci[0]:.{d}f}, {ci[1]:.{d}f}]"
    return s


def fp(p):
    return "–" if p is None or not np.isfinite(p) else ("< 0.001" if p < .001 else f"{p:.3f}")


def table(rows, header, cls=""):
    out = [f"<table class='{cls}'><thead><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in header) + "</tr></thead><tbody>"]
    out += ["<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows]
    return "\n".join(out + ["</tbody></table>"])


def img_b64(path, mime="image/jpeg", alt="figure"):
    p = Path(path)
    return f"<img alt='{alt}' src='data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}'>" if p.exists() else ""


def figure(fig, alt):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    return f"<img alt='{alt}' src='data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}'>"


def verdict(x):
    lo, hi = x["CI95_difference"]
    sig = (lo > 0 or hi < 0) and x["p_sign_flip"] < .05
    if sig:
        return "a better" if x["difference"] > 0 else "b better"
    if (lo > 0 or hi < 0) != (x["p_sign_flip"] < .05):
        return "CI and permutation disagree: no direction claimed"
    return "no difference shown"


def holm(ps):
    order = np.argsort(ps)
    adj, running = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def primary_family(path):
    """Pre-registered primary comparisons (two-stage vs 4 baselines) x primary metrics (object F1, TLS-vs-inflamed AUROC)."""
    rows = []
    for key, b in [("all_annotated", N11), ("all_annotated", N16), ("all_annotated", HK), ("hovernet_subset", HN)]:
        comp = next(c for c in path[key]["comparisons"] if c["a"] == TWO and c["b"] == b)
        for m in ["object_F1", "tile_AUROC_TLS_vs_inflamed"]:
            x = comp["metrics"][m]
            rows.append({"comparison": f"Two-stage − {b}", "set": key, "arrays": comp["arrays"], "metric": m, "difference": x["difference"],
                         "CI95_difference": x["CI95_difference"], "p_nominal": x["p_sign_flip"], "verdict": verdict(x)})
    adj = holm(np.array([r["p_nominal"] for r in rows]))
    for r, a in zip(rows, adj):
        r["p_holm"] = float(a)
        lo, hi = r["CI95_difference"]
        r["verdict_holm"] = ("a better" if r["difference"] > 0 else "b better") if ((lo > 0 or hi < 0) and a < .05) else \
            "not significant after Holm" if r["verdict"].endswith("better") else r["verdict"]
    return rows


def per_arm_table(r, caveat=True):
    metrics = [("object_F1", "Object F1*"), ("object_recall", "Recall"), ("object_precision", "Precision*"),
               ("object_precision_lenient", "Precision, lenient*"), ("tile_AUROC_TLS_vs_inflamed", "Tile AUROC TLS vs inflamed"),
               ("tile_AUROC_TLS_vs_all", "Tile AUROC TLS vs all"), ("inflamed_area_called_TLS", "Inflamed area called TLS")]
    rows = [[html.escape(arm), e["arrays"], e["patients"], e["reference_TLS"]] + [fmt(e[m]["value"], e[m]["CI95"], 2) for m, _ in metrics]
            for arm, e in r["per_arm"].items()]
    return table(rows, ["Arm", "Arrays", "Patients", "Reference TLS"] + [h for _, h in metrics]) + \
        (f"<p class='caveat'>* {CAVEAT}</p>" if caveat else "")


def paired_table(r, metrics=("object_F1", "object_recall", "object_precision", "tile_AUROC_TLS_vs_inflamed", "inflamed_area_called_TLS")):
    rows = []
    for c in r["comparisons"]:
        for m in metrics:
            x = c["metrics"][m]
            rows.append([html.escape(c["a"]), html.escape(c["b"]), c["arrays"], m + ("*" if m in ("object_F1", "object_precision") else ""),
                         fmt(x["difference"], x["CI95_difference"]), fp(x["p_sign_flip"]), verdict(x)])
    return table(rows, ["a", "b", "Arrays", "Metric", "a − b [95% CI]", "Sign-flip p (nominal)", "Reading"]) + f"<p class='caveat'>* {CAVEAT}</p>"


def diag_section(d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = pd.DataFrame(d["tnbc_frozen_section_by_class"])
    e = pd.DataFrame(d["esca_ffpe_test_reference"])
    v = pd.DataFrame(d["visium_ffpe_reference"])
    groups = [("TNBC frozen", f"{r['class']}", r["share_T"], r["share_B"], r["share_Other"] + r["share_Macrophage"], r["median_patient_nuclei_per_mm2"])
              for _, r in t.iterrows()]
    groups += [("ESCA FFPE", f"{r['region']}", r["share_T"], r["share_B"], r["share_Other"] + r["share_Macrophage"], r["median_patient_nuclei_per_mm2"])
               for _, r in e.iterrows()]
    groups += [("Visium FFPE", f"{r['region']}", r["share_T"], r["share_B"], r["share_Other_or_Macrophage"], r["median_patient_nuclei_per_mm2"])
               for _, r in v.iterrows()]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    y = np.arange(len(groups))
    T = np.array([g[2] for g in groups]); B = np.array([g[3] for g in groups]); O = np.array([g[4] for g in groups])
    ax.barh(y, B, color="#c0392b", label="B")
    ax.barh(y, T, left=B, color="#2e8b57", label="T")
    ax.barh(y, O, left=B + T, color="#c9c9c9", label="Other (+ Macrophage)")
    ax.set_yticks(y, [f"{g[0]}: {g[1]}" for g in groups], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of HE2CELL nuclei (same checkpoint, tau 0.5 prior, D4)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("HE2CELL labels by region: frozen-section TNBC vs FFPE references", fontsize=10)
    rows = [[html.escape(g[0]), html.escape(g[1]), f"{g[2]:.2f}", f"{g[3]:.3f}", f"{g[4]:.2f}", f"{g[5]:,.0f}"] for g in groups]
    tab = table(rows, ["Cohort", "Region / class", "T share", "B share", "Other (+Mac) share", "Median nuclei/mm² per patient"])
    crops = "".join(f"<figure>{img_b64(p, alt=p.stem)}<figcaption>{html.escape(p.stem.replace('_', ' '))}</figcaption></figure>"
                    for p in sorted((C.OUT / "qc" / "diagnostic").glob("*.jpg")))
    return figure(fig, "label composition by class") + tab + \
        "<details><summary>QC crops, 2 per class (T green, B red, macrophage yellow; Other unmarked)</summary><div class='crops'>" + crops + "</div></details>"


def main():
    scale, prereg = load(C.OUT / "scale.json"), load(C.OUT / "preregistration.json")
    inv = load(C.OUT / "inventory.json")["summary"]
    diag, path = load(A / "diagnostic" / "summary.json"), load(A / "pathologist" / "results.json")
    ihc, rna, hk = load(A / "ihc" / "results.json"), load(A / "rna" / "results.json"), load(A / "hooknet_qc.json")
    reg = pd.read_csv(C.OUT / "registration" / "summary.csv")
    cells = sorted(p.parent.name for p in (C.OUT / "cells").glob("*/cells.csv.gz"))
    deviation = prereg["protocol_deviations"][0]
    family = primary_family(path)
    i5, i2 = ihc["tiles_500um"], ihc["tiles_200um"]
    t2 = ihc["tls_level"]["2000"]
    allp = path["all_annotated"]["per_arm"]
    sub = path["hovernet_subset"]["per_arm"]
    proc_reg = reg[reg.array.isin(cells)]

    def comp(key, a, b, m):
        return next(c for c in path[key]["comparisons"] if c["a"] == a and c["b"] == b)["metrics"][m]

    def ihc_pair(a, b, m="F1"):
        return next(c for c in ihc["tls_level_paired_2000"] if c["a"] == a and c["b"] == b)["metrics"][m]

    hn_auc = comp("hovernet_subset", TWO, HN, "tile_AUROC_TLS_vs_inflamed")
    conclusions = {
        "domain": "Frozen-section ST H&E (TNBC_ST), outside HE2CELL's FFPE training domain. 33 of 94 annotated arrays (pre-specified priority block).",
        "label_shift": f"About three quarters of nuclei in pathologist Lymphoid nodules and Lymphocyte regions are called Other "
                       f"(0.76 / 0.75) versus 0.46 inside HookNet TLS on FFPE ESCA; B share in Lymphoid nodules 0.12 vs 0.19 (ESCA TLS) and 0.45 (Visium TLS).",
        "claim1_T_B_identity": "NOT supported on frozen sections: region-level agreement with serial IHC is no better than nuclear density "
                               f"(500 µm: T vs CD3 {fmt(i5['rho_T_CD3']['mean'], i5['rho_T_CD3']['CI95'])}, B vs CD20 {fmt(i5['rho_B_CD20']['mean'], i5['rho_B_CD20']['CI95'])}, "
                               f"nuclei vs CD3+CD20 {fmt(i5['rho_nuclei_total']['mean'], i5['rho_nuclei_total']['CI95'])}; T+B − nuclei {fmt(i5['TB_minus_nuclei']['mean'], i5['TB_minus_nuclei']['CI95'])}); "
                               f"B discriminant not met ({fmt(i5['disc_B']['mean'], i5['disc_B']['CI95'])}); T discriminant met but small ({fmt(i5['disc_T']['mean'], i5['disc_T']['CI95'])}); "
                               f"spot RNA B vs B genes {fmt(rna['B_vs_Bgenes']['mean'], rna['B_vs_Bgenes']['CI95'])} vs nuclei {fmt(rna['nuclei_vs_Bgenes']['mean'], rna['nuclei_vs_Bgenes']['CI95'])}.",
        "claim2_TLS_detection": "PARTIAL. Against same-section pathologist Lymphoid nodules, the two-stage composition filter has higher object F1 than nuclei-only "
                                f"(11000: {fmt(comp('all_annotated', TWO, N11, 'object_F1')['difference'], comp('all_annotated', TWO, N11, 'object_F1')['CI95_difference'])}; "
                                f"16000: {fmt(comp('all_annotated', TWO, N16, 'object_F1')['difference'], comp('all_annotated', TWO, N16, 'object_F1')['CI95_difference'])}) and HookNet-TLS "
                                f"({fmt(comp('all_annotated', TWO, HK, 'object_F1')['difference'], comp('all_annotated', TWO, HK, 'object_F1')['CI95_difference'])}), all Holm p < 0.01; "
                                f"versus HoVer-Net (20-array subset) the F1 difference {fmt(comp('hovernet_subset', TWO, HN, 'object_F1')['difference'], comp('hovernet_subset', TWO, HN, 'object_F1')['CI95_difference'])} "
                                f"has nominal p {comp('hovernet_subset', TWO, HN, 'object_F1')['p_sign_flip']:.3f} but Holm p {next(r['p_holm'] for r in family if r['comparison'].endswith(HN) and r['metric'] == 'object_F1'):.3f}, so it is not claimed; "
                                "two-stage TLS-vs-inflamed AUROC over nuclei-only is not significant after Holm. B density has higher "
                                f"TLS-vs-inflamed tile AUROC than nuclei-only ({fmt(comp('all_annotated', BD, N11, 'tile_AUROC_TLS_vs_inflamed')['difference'], comp('all_annotated', BD, N11, 'tile_AUROC_TLS_vs_inflamed')['CI95_difference'])}, secondary comparison, nominal p {comp('all_annotated', BD, N11, 'tile_AUROC_TLS_vs_inflamed')['p_sign_flip']:.3f}); "
                                "precision/F1 against nodules are pessimistic because the inflamed negatives contain IHC follicles. Against IHC CD20 aggregates there is no F1 advantage over nuclei-only "
                                f"({fmt(ihc_pair(TWO, N11)['difference'], ihc_pair(TWO, N11)['CI95_difference'])}) or HoVer-Net ({fmt(ihc_pair(TWO, HN)['difference'], ihc_pair(TWO, HN)['CI95_difference'])}): "
                                f"HE2CELL is higher-precision ({fmt(t2[TWO]['precision']['value'], t2[TWO]['precision']['CI95'])}) and lower-recall ({fmt(t2[TWO]['recall']['value'], t2[TWO]['recall']['CI95'])}). "
                                f"HoVer-Net has the highest TLS-vs-inflamed tile AUROC point estimate on its 20-array subset ({fmt(sub[HN]['tile_AUROC_TLS_vs_inflamed']['value'], sub[HN]['tile_AUROC_TLS_vs_inflamed']['CI95'])}); "
                                f"two-stage − HoVer-Net {fmt(hn_auc['difference'], hn_auc['CI95_difference'])}, p {hn_auc['p_sign_flip']:.3f}: the CI and the permutation test disagree at n = 20, so no direction is claimed.",
        "claim3_composition": f"Supporting evidence only: in {ihc['composition']['objects']} two-stage TLS matched to IHC aggregates ({ihc['composition']['patients']} patients), HE2CELL B/(B+T) "
                              f"tracks IHC CD20/(CD3+CD20) (rho {fmt(ihc['composition']['rho_HE_B_share_vs_IHC_CD20_share'], ihc['composition']['CI95'], 2)}), but under-calls the B share "
                              f"(median {ihc['composition']['median_HE_B_share_BT']:.2f} vs {ihc['composition']['median_IHC_CD20_share']:.2f}).",
        "hooknet": f"HookNet-TLS fails on frozen sections: it calls a median {hk['median_tissue_fraction_called_TLS']:.0%} of tissue TLS (IQR {hk['IQR'][0]:.0%}–{hk['IQR'][1]:.0%}); "
                   f"object F1 vs Lymphoid nodules {fmt(allp[HK]['object_F1']['value'], allp[HK]['object_F1']['CI95'], 2)}.",
        "multiplicity": "preregistration.json fixed no correction but lists a primary family (two-stage vs 4 baselines x object F1 and TLS-vs-inflamed AUROC); Holm-adjusted p is reported for those 8 tests, every other p is nominal.",
    }
    summary = {"conclusions": conclusions, "cohort": "frozen-section ST H&E (TNBC_ST, Zenodo 14204217)", "arrays_with_HE2CELL": cells,
               "protocol_deviation": deviation, "preregistration_amendments": prereg.get("amendments", []),
               "primary_family_holm": family, "scale": scale, "inventory": inv,
               "registration": {"accepted_all": int(reg.accept.sum()), "total_all": int(len(reg)),
                                "processed_arrays_with_ihc": int(len(proc_reg)), "processed_accepted": int(proc_reg.accept.sum()),
                                "failed": reg[~reg.accept].array.tolist()},
               "classifier": {"checkpoint": "models/he2cell_esca_finetuned.pth", "sha256": "1778b5dd336db6e58f59b3a2533c2b15631d8f9b136d539072fd0038cb2a5f1d",
                              "prior_tau": 0.5, "tta": "d4"},
               "diagnostic": diag, "pathologist": path, "ihc": ihc, "rna": rna, "hooknet_failure_mode": hk}
    C.save_json(summary, C.OUT / "summary.json")

    deviations = [
        f"Stage 1 ran on 33/94 annotated arrays: the pre-specified priority block (order fixed at Gate A, not by results), stopped after the interim checkpoint for GPU cost. The 61 arrays not run are listed below; recorded in preregistration.json → protocol_deviations.",
        "IHC stain vectors: spherical k-means collapsed onto the grey H–DAB mixture band, so stains are OD-chromaticity endmembers with a cohort-default fallback (> 8° or cond > 20); the red vector fell back on 65/86 slides.",
        "IHC concentration maps are saved at 2 µm/px, not 1 µm/px (disk); blobs are detected at 0.457 µm/px.",
        "QuPath artefact contours (.qpdata) are not used; annotated arrays use the annotation's Artefacts and Hole classes.",
        "The ST spot-lattice scale estimate only holds under an assumed 100 µm array unit (66×63 checkerboard, y/x unit 1.113 while the H&E is isotropic); the registration median (imagesHD 0.2715 µm/px) is the adopted scale.",
        "HoVer-Net runs on 3328-px blocks of 4×4 HE2CELL patches (same resampling, nuclei kept in processed patch cores), not on each patch.",
        "Pre-registration amendments made before any H&E output was opened: equal 7/7/6 HoVer-Net subset allocation; annotation colours matched within 20 levels.",
        "Added at the coordinator's request after Gate A/B: a reference-arrays-only sensitivity analysis, partial correlations given nuclear density, paired statistics for the IHC-aggregate level.",
        "HookNet objects are its published-filter polygons cut by the valid region, so they fragment (3,765 components); no post-filtering beyond the published filter.",
        "Lymphoid-nodule reference after valid-region masking: 54 objects in 22 patients (raw annotation: 65 in 25).",
        "IHC quantification and registration were run before Gate A (needed for the pixel size).",
        "Another session (tnbc-frozen-adapt) shared the GPU and deployment/tnbc_st/; this report uses only the default checkpoint (sha256 1778b5dd…) and none of that session's outputs.",
    ]
    fam_rows = [[html.escape(r["comparison"]), r["set"].replace("_", " "), r["arrays"], r["metric"] + ("*" if r["metric"] == "object_F1" else ""),
                 fmt(r["difference"], r["CI95_difference"]), fp(r["p_nominal"]), fp(r["p_holm"]), r["verdict_holm"]] for r in family]

    # B table
    def brow(label, key, d500, d200, cd2=True):
        a, b = d500[key], d200[key]
        s = d500.get(key + "_incl_CD2_as_CD3")
        return [label, fmt(a["mean"], a["CI95"]), a["samples"], fmt(b["mean"], b["CI95"]), fmt(s["mean"], s["CI95"]) if (cd2 and s) else "–"]
    b_rows = [brow("HE2CELL T vs CD3", "rho_T_CD3", i5, i2), brow("HE2CELL B vs CD20", "rho_B_CD20", i5, i2, False),
              brow("HE2CELL B vs CD3", "rho_B_CD3", i5, i2), brow("HE2CELL T vs CD20", "rho_T_CD20", i5, i2),
              brow("HE2CELL T+B vs CD3+CD20", "rho_TB_total", i5, i2), brow("Nuclei-only vs CD3+CD20", "rho_nuclei_total", i5, i2),
              brow("HoVer-Net inflammatory vs CD3+CD20 (subset)", "rho_HN_total", i5, i2),
              brow("(T+B) − nuclei", "TB_minus_nuclei", i5, i2), brow("(T+B) − HoVer-Net (subset)", "TB_minus_HN", i5, i2),
              brow("Discriminant B: ρ(B,CD20) − ρ(B,CD3)", "disc_B", i5, i2), brow("Discriminant T: ρ(T,CD3) − ρ(T,CD20)", "disc_T", i5, i2),
              brow("Partial ρ(B,CD20 | nuclei)", "partial_B_CD20_given_nuclei", i5, i2, False),
              brow("Partial ρ(T,CD3 | nuclei)", "partial_T_CD3_given_nuclei", i5, i2),
              brow("Partial ρ(B,CD20 | HE T+B)", "partial_B_CD20_given_TB", i5, i2, False),
              brow("Partial ρ(T,CD3 | HE T+B) (mechanically negative)", "partial_T_CD3_given_TB", i5, i2),
              brow("Partial ρ(B,CD20 | IHC CD3+CD20)", "partial_B_CD20_given_IHCtotal", i5, i2),
              brow("Partial ρ(T,CD3 | IHC CD3+CD20)", "partial_T_CD3_given_IHCtotal", i5, i2)]
    c_rows = []
    for thr in ["2000", "1000", "4000"]:
        for arm, e in ihc["tls_level"][thr].items():
            c_rows.append([thr, html.escape(arm), e["arrays"], e["aggregates"], e["predicted"], fmt(e["recall"]["value"], e["recall"]["CI95"], 2),
                           fmt(e["precision"]["value"], e["precision"]["CI95"], 2), fmt(e["F1"]["value"], e["F1"]["CI95"], 2)])
    cp_rows = []
    for c in ihc["tls_level_paired_2000"]:
        for m in ["F1", "recall", "precision"]:
            x = c["metrics"][m]
            cp_rows.append([html.escape(c["a"]), html.escape(c["b"]), c["arrays"], m, fmt(x["difference"], x["CI95_difference"]), fp(x["p_sign_flip"]), verdict(x)])
    d_rows = [[k, fmt(v["mean"], v["CI95"]), v["samples"]] for k, v in rna.items() if isinstance(v, dict) and "mean" in v]
    reg_fail = reg[~reg.accept].sort_values("median_residual_um")
    reg_rows = [[r.array, f"{r.median_residual_um:.0f}", f"{r.tiles_converged}/{r.tiles_total}", f"{r.ncc:.2f}"] for r in reg_fail.itertuples()]
    style = """:root{--bg:#fff;--fg:#1f2328;--muted:#59636e;--line:#d0d7de;--head:#f6f8fa;--note:#fff8e1;--noteline:#d4a017;--cav:#fdecea;--cavline:#c0392b}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0d1117;--fg:#e6edf3;--muted:#9198a1;--line:#30363d;--head:#161b22;--note:#2b2410;--noteline:#d4a017;--cav:#2d1414;--cavline:#e06c5a}}
:root[data-theme="dark"]{--bg:#0d1117;--fg:#e6edf3;--muted:#9198a1;--line:#30363d;--head:#161b22;--note:#2b2410;--noteline:#d4a017;--cav:#2d1414;--cavline:#e06c5a}
body{background:var(--bg);color:var(--fg);font-family:-apple-system,Helvetica,Arial,sans-serif;max-width:1150px;margin:24px auto;padding:0 16px;line-height:1.45}
table{border-collapse:collapse;font-size:12px;margin:8px 0 14px;display:block;overflow-x:auto}td,th{border:1px solid var(--line);padding:3px 6px;text-align:left;vertical-align:top}
th{background:var(--head)}h2{border-bottom:1px solid var(--line);padding-bottom:4px;margin-top:32px}img{max-width:100%;background:#fff}
.note{background:var(--note);padding:8px 12px;border-left:4px solid var(--noteline)}.caveat{background:var(--cav);padding:6px 10px;border-left:4px solid var(--cavline);font-size:12px}
.crops{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:8px}figure{margin:0}figcaption{font-size:11px;color:var(--muted)}
ul.conclusions li{margin-bottom:6px}"""
    P = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
         f"<title>TNBC ST validation</title><style>{style}</style></head><body>",
         "<h1>HE2CELL on TNBC_ST: frozen-section ST H&amp;E vs serial CD3/CD20 IHC, pathologist annotations and spot RNA</h1>",
         "<p class='note'><b>Cohort first:</b> these are <b>frozen-section</b> spatial-transcriptomics H&amp;E arrays (Venet 2024, Zenodo 14204217), "
         "outside HE2CELL's FFPE training domain. HE2CELL = Cellpose cpsam_v2 + models/he2cell_esca_finetuned.pth (sha256 1778b5dd…, tau 0.5 prior, D4 TTA), "
         f"raw H&amp;E at {scale['hd_mpp']} µm/px resampled to 0.25 µm/px; frozen ESCA detector parameters; reference rules pre-registered. "
         f"{len(cells)} of 94 annotated arrays (33 patients) were processed (pre-specified priority block; see deviations).</p>",
         "<h2>Conclusions</h2><ul class='conclusions'>" + "".join(f"<li><b>{html.escape(k.replace('_', ' '))}:</b> {html.escape(v)}</li>" for k, v in conclusions.items()) + "</ul>",
         "<h2>1. Domain shift: HE2CELL labels by pathologist class (descriptive; lead result)</h2>", diag_section(diag),
         "<h2>2. Section A — pathologist Lymphoid nodules, same section (primary for claim 2)</h2>",
         "<p>Reference: 8-connected Lymphoid-nodule components ≥ 20,000 µm² in the valid region (54 objects, 22 patients). Negatives: Lymphocyte + High TIL stroma ('inflamed'). "
         "Matching: ≥ 10 % overlap of the smaller object. Tiles 96 µm. All 33 arrays (precision counts false objects in arrays without a nodule).</p>",
         per_arm_table(path["all_annotated"]), "<h3>Paired differences (all 33 arrays)</h3>", paired_table(path["all_annotated"]),
         "<h3>Pre-registered primary family (Holm-adjusted)</h3>",
         table(fam_rows, ["Comparison", "Set", "Arrays", "Metric", "Difference [95% CI]", "p nominal", "p Holm (8 tests)", "Reading (CI excludes 0 and Holm p < 0.05)"]),
         f"<p class='caveat'>* {CAVEAT}</p>",
         f"<p>Two-stage − HoVer-Net, TLS-vs-inflamed AUROC: {fmt(hn_auc['difference'], hn_auc['CI95_difference'])}, sign-flip p {hn_auc['p_sign_flip']:.3f}. "
         "The bootstrap CI excludes 0 but the permutation test does not reject at n = 20 patients, so neither direction is claimed.</p>",
         "<h3>HoVer-Net subset (20 arrays; pre-registered stratified draw)</h3>", per_arm_table(path["hovernet_subset"]), paired_table(path["hovernet_subset"]),
         "<h3>Sensitivity: arrays with a Lymphoid-nodule reference only (22 patients)</h3>", per_arm_table(path["reference_arrays_only"]),
         paired_table(path["reference_arrays_only"], ("object_F1", "object_precision", "tile_AUROC_TLS_vs_inflamed")),
         "<h2>3. Section C — IHC CD20 aggregates on the serial section (same weight as A)</h2>",
         f"<p>Reference: CD20+ blob density (σ 25 µm) ≥ 2,000/mm², ≥ 20,000 µm², ≥ 20 CD20+ and ≥ 20 CD3+ cells within 50 µm (CD3 rule skipped on CD2 slides), "
         f"mapped to H&amp;E with accepted registrations ({int(proc_reg.accept.sum())} of {len(proc_reg)} processed arrays with IHC); match within 100 µm. "
         "Pathologist Lymphoid nodules scored the same way give the agreement ceiling. Sensitivity thresholds 1,000 and 4,000/mm².</p>",
         table(c_rows, ["CD20/mm²", "Arm", "Arrays", "IHC aggregates", "H&E objects", "Recall", "Precision", "F1"]),
         "<h3>Paired differences at 2,000/mm²</h3>", table(cp_rows, ["a", "b", "Arrays", "Metric", "a − b [95% CI]", "Sign-flip p (nominal)", "Reading"]),
         "<h2>4. Section B — IHC region level (claim 1)</h2>",
         "<p>Mean per-sample Spearman over tiles with ≥ 50 % registered overlap (samples with ≥ 10 tiles), patient bootstrap. "
         "CD2 slides (TNBC 9, 10, 78, 82, 83, 84) are excluded from every CD3 term; last column: CD2 treated as CD3.</p>",
         table(b_rows, ["Quantity", "500 µm (primary)", "Samples", "200 µm", "500 µm, CD2 as CD3"]),
         "<h2>5. Section D — ST spot RNA (secondary)</h2>",
         f"<p>{rna['arrays']} arrays, {rna['spots']:,} spots under tissue; 50 µm-radius disks; log1p(CP10k) panel means; IHC rows are the ceiling on registered arrays.</p>",
         table(d_rows, ["Quantity", "Mean per-array Spearman [95% CI]", "Arrays"]),
         "<h2>6. Section E — composition in matched TLS (claim 3)</h2>",
         f"<p>{ihc['composition']['objects']} two-stage TLS matched to IHC aggregates in {ihc['composition']['patients']} patients (CD2 slides excluded): "
         f"Spearman of HE2CELL B/(B+T) vs IHC CD20/(CD3+CD20) = {fmt(ihc['composition']['rho_HE_B_share_vs_IHC_CD20_share'], ihc['composition']['CI95'], 2)}; "
         f"median shares {ihc['composition']['median_HE_B_share_BT']:.2f} (HE2CELL) vs {ihc['composition']['median_IHC_CD20_share']:.2f} (IHC). Supporting evidence only; HE2CELL under-calls B.</p>",
         "<h2>7. HookNet-TLS failure mode on frozen sections</h2>",
         f"<p>Published HookNet-TLS (published filter only) calls a median {hk['median_tissue_fraction_called_TLS']:.0%} (IQR {hk['IQR'][0]:.0%}–{hk['IQR'][1]:.0%}) "
         f"of frozen-section tissue TLS ({hk['filtered_TLS_polygons_total']} polygons), against a median Lymphoid-nodule fraction of {hk['median_LN_fraction_of_tissue']:.1%}. "
         "Blue: HookNet TLS; yellow: pathologist Lymphoid nodules; arrays at the 25th/50th/75th/90th percentile.</p>",
         img_b64(C.OUT / "qc" / "hooknet_failure_mode.jpg", alt="HookNet failure mode"),
         "<h2>8. Data, scale and registration QC</h2>",
         f"<p>Inventory: {inv['arrays_in_ids']} arrays / {inv['samples']} patients; {inv['annotated_arrays']} annotated; {inv['samples_with_he_and_ihc']} patients with IHC. "
         f"Pixel size: imagesHD {scale['hd_mpp']} µm/px = median of {scale['registration']['pairs_accepted']} accepted H&amp;E–IHC registrations "
         f"(lattice cross-check {scale['cross_check']['spot_lattice_hd_mpp']:.4f}; {html.escape(scale['cross_check']['anisotropy'])}). "
         f"Registration (median 1 mm tile residual ≤ 100 µm, ≥ 3 and ≥ 50 % tiles converged): {int(reg.accept.sum())}/{len(reg)} pairs accepted overall, "
         f"{int(proc_reg.accept.sum())}/{len(proc_reg)} among processed arrays. Failed pairs:</p>",
         table(reg_rows, ["Array", "Median residual µm", "Tiles converged", "Coarse NCC"]),
         "<p>Chromogen assignment (red = CD20, brown/DAB = CD3) from morphology: qc/stage0/chromogen_assignment.png.</p>",
         "<h2>9. Deviations from the plan</h2><ol>" + "".join(f"<li>{html.escape(d)}</li>" for d in deviations) + "</ol>",
         f"<p><b>Arrays not run ({deviation['n_not_run']}):</b> {html.escape(', '.join(deviation['arrays_not_run']))}.</p>",
         "<p><b>Resume:</b> <code>deployment/tnbc_st/run_stage1.sh</code> (reads outputs/tnbc_st_20261101/stage1_order.txt, skips finished arrays, one GPU-lock job per array), "
         "then re-run <code>tnbc_eval_path.py</code>, <code>tnbc_eval_ihc.py</code>, <code>tnbc_eval_rna.py</code>, <code>tnbc_diagnostic.py</code>, <code>tnbc_report.py</code>.</p>",
         "<h2>Files</h2><p>outputs/tnbc_st_20261101/{inventory.json, scale.json, preregistration.json, cells/, baselines/hovernet/, hooknet/, ihc/, registration/, "
         "analysis/{pathologist, ihc, rna, diagnostic, interim}/, qc/, summary.json}. Code: deployment/tnbc_st/.</p></body></html>"]
    (C.OUT / "TNBC_report.html").write_text("\n".join(P))
    print("wrote", C.OUT / "TNBC_report.html")
    print(json.dumps(conclusions, indent=1))


if __name__ == "__main__":
    main()
