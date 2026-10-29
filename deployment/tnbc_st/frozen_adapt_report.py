"""outputs/tnbc_frozen_adapt_20261101/Frozen_adapt_report.html from summary.json, step1/metrics.json, tables and figures."""
from __future__ import annotations

import base64
import html
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frozen_adapt_common as F  # noqa: E402

O = F.OUT


def img(path, width="100%"):
    if not Path(path).exists():
        return ""
    mime = "image/png" if str(path).endswith(".png") else "image/jpeg"
    return f'<img src="data:{mime};base64,{base64.b64encode(Path(path).read_bytes()).decode()}" style="max-width:{width}">'


def ci(e, digits=3):
    if not e or e.get("mean", e.get("value")) is None:
        return "–"
    v = e.get("mean", e.get("value"))
    lo, hi = e.get("CI95", [None, None])
    if v != v:
        return "–"
    return f"{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def main():
    s = json.loads((O / "summary.json").read_text())
    m = json.loads((O / "step1" / "metrics.json").read_text())
    pre = json.loads((O / "preregistration.json").read_text())
    extra = json.loads((O / "report_notes.json").read_text()) if (O / "report_notes.json").exists() else {}
    arms = list(s["arms"])
    cols = [("rho_B_CD20_200", "B vs CD20 (200 µm)"), ("rho_T_CD3_200", "T vs CD3 (200 µm)"), ("disc_B_200", "B discriminant"),
            ("B_share_Lymphoid nodule", "B share, Lymphoid nodule"), ("Other_share_Lymphoid nodule", "Other share, Lymphoid nodule"),
            ("TLS_recall_path_Two-stage (HE2CELL)", "TLS recall vs pathologist (two-stage)"),
            ("TLS_recall_path_B density (HE2CELL)", "TLS recall vs pathologist (B density)"),
            ("TLS_recall_ihc_Two-stage (HE2CELL)", "recall vs IHC CD20 aggregates (two-stage)"),
            ("TLS_recall_ihc_B density (HE2CELL)", "recall vs IHC CD20 aggregates (B density)"),
            ("rna_B_vs_Bgenes", "spot B vs MS4A1/CD79A/CD19"), ("rna_T_vs_Tgenes", "spot T vs CD3E/CD3D/CD2")]
    rows = "".join(f"<tr><td><b>{a}</b></td>" + "".join(f"<td>{ci(s['arms'][a].get(c))}</td>" for c, _ in cols) + "</tr>" for a in arms)
    head = "<tr><th>arm</th>" + "".join(f"<th>{html.escape(n)}</th>" for _, n in cols) + "</tr>"
    p = s["primary_A3_vs_A0_rho_B_CD20_200"]
    ff = s.get("ffpe_sanity") or {}
    ffrows = "".join(f"<tr><td>{k}</td><td>{ci(ff.get(f'macroF1_{k}'))}</td><td>{ci({'mean': ff.get(f'delta_{k}_minus_clean', {}).get('mean_diff', float('nan')), 'CI95': ff.get(f'delta_{k}_minus_clean', {}).get('CI95', [0, 0])}) if k != 'clean' else ''}</td><td>{'FLAG' if ff.get(f'flag_{k}_drop_gt_0.02') else ''}</td></tr>" for k in ["clean", "A1", "A2", "A3"]) if ff else ""
    r = m["ratios_TNBC_over_ESCA"]
    s1 = "".join(f"<tr><td>{k}</td><td>{ci(v.get('TNBC', v.get('TNBC_LN_plus_Lymphocyte')))}</td><td>{ci(v.get('ESCA_TLS', v.get('ESCA_HookNet_TLS')))}</td><td>{r.get(k, float('nan')):.2f}</td></tr>"
                 for k, v in m["metrics"].items() if isinstance(v, dict) and ("TNBC" in v or "TNBC_LN_plus_Lymphocyte" in v))
    overlays = "".join(img(x, "48%") for x in sorted((O / "step1").glob("overlay_*.jpg")))
    sel = "".join(f"<tr><td>{k}</td><td>{v.get('chosen', '–')}</td><td>{json.dumps({a: round(b, 3) if b == b else None for a, b in v.get('train_scores', {}).items()})}</td><td>{[round(x, 3) for x in v.get('em_prior_Other_T_Mac_B', [])]}</td></tr>" for k, v in s["a4_selection"].items())
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Frozen-section adaptation</title><style>
:root{{--bg:#fff;--fg:#1b1b1b;--muted:#666;--line:#ddd;--accent:#8b1e1e}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#151515;--fg:#e8e8e8;--muted:#aaa;--line:#333;--accent:#e07a7a}}}}
:root[data-theme="dark"]{{--bg:#151515;--fg:#e8e8e8;--muted:#aaa;--line:#333;--accent:#e07a7a}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:1200px;margin:auto;padding:16px}}
table{{border-collapse:collapse;font-size:13px;display:block;overflow-x:auto}}td,th{{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:top}}
h1,h2{{color:var(--accent)}} .muted{{color:var(--muted)}} img{{height:auto;margin:4px 0}}
.verdict{{border-left:4px solid var(--accent);padding:6px 12px;font-size:16px}}</style></head><body>
<h1>TNBC_ST frozen-section adaptation (steps 1-2)</h1>
<p class="muted">Cohort-adapted add-on; the frozen default result in outputs/tnbc_st_20261101 stays primary. Preregistered {pre['written']} (+ amendment_1 {pre['amendment_1']['written']}).
Arrays: {len(s['arrays'])} ({s['patients']} patients). Values are patient means (or pooled ratios) with patient-bootstrap 95 % CI (2000).</p>
<h2>Step 1 verdict</h2><div class="verdict">{html.escape(extra.get('verdict_text', m['verdict']))}</div>
<table><tr><th>metric (LN + Lymphocyte regions)</th><th>TNBC</th><th>ESCA TLS</th><th>ratio</th></tr>{s1}</table>
<p>Cellpose overlays on dense Lymphoid-nodule windows (96 µm, 0.25 µm/px):</p>{overlays}
<p>Random crops with masks, TNBC (top two rows) vs ESCA TLS (bottom two rows):</p>{img(O / 'step1' / 'crops_tnbc_vs_esca.jpg', '70%')}
<p>Macenko to ESCA (top raw, bottom normalised):</p>{img(O / 'figures' / 'macenko_examples_CN10_D2.jpg', '70%')}
<h2>Step 2: per-arm results (held-out patients)</h2>
<p>stored = full-array stage-1 labels; A0 = recomputed default on the 1/3 evaluation subsample (agreement with stored labels: median {s['a0_reproduction']['median_agreement']:.4f}).</p>
<table>{head}{rows}</table>
<h3>Primary: A3 vs A0, B-vs-CD20 200 µm tile Spearman, paired by patient</h3>
<p>mean difference {p.get('mean_diff', float('nan')):+.3f}, 95 % CI [{p.get('CI95', [float('nan')] * 2)[0]:+.3f}, {p.get('CI95', [float('nan')] * 2)[1]:+.3f}], sign-flip p = {p.get('sign_flip_p', float('nan')):.4f}, {p.get('positive', 0)}/{p.get('patients', 0)} patients improved.</p>
<h3>A4 base-arm selection (training folds) and EM priors</h3><table><tr><th>fold</th><th>chosen</th><th>training mean rho(B,CD20)</th><th>EM prior (Other,T,Mac,B)</th></tr>{sel}</table>
{img(O / 'figures' / 'composition_by_class.png')}{img(O / 'figures' / 'tile_spearman_by_arm.png')}
<h2>FFPE sanity check (Immunocto CV held-out folds)</h2><table><tr><th>arm</th><th>macro-F1</th><th>delta vs clean</th><th></th></tr>{ffrows}</table>
<h2>Notes</h2><ul>{''.join(f'<li>{html.escape(n)}</li>' for n in extra.get('notes', []))}</ul>
</body></html>"""
    (O / "Frozen_adapt_report.html").write_text(page)


if __name__ == "__main__":
    main()
