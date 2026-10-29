"""HTML report for the HoVer-Net (PanNuke) inflammatory-cell TLS baseline (reads summary.json from hovernet_eval.py).

Writes outputs/baselines_hovernet_20261026/Baseline_report.html (same layout as the ESCA / Visium / ablation
reports) and figures/. Run with .venv/bin/python after hovernet_eval.py.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "baselines_hovernet_20261026"
ADOPTED, NUCLEI = "HE2CELL B density (adopted)", "Nuclei only (no classifier)"
DENSITY_ARM, DBSCAN_ARM = "HoVer-Net inflammatory density", "HoVer-Net inflammatory DBSCAN"
INK, MUTED, GRID, POS, NEG, NULL = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834", "#9a9890"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "figure.dpi": 150})
FOREST = [("ESCA", "object_F1", "ESCA object F1"), ("ESCA", "object_recall", "ESCA object recall"),
          ("ESCA", "dense_specificity", "ESCA dense-window specificity"), ("ESCA", "window_AUROC", "ESCA window AUROC"),
          ("Visium", "AUROC_TLS_vs_all", "Visium AUROC TLS vs all"), ("Visium", "AUROC_TLS_vs_INFL", "Visium AUROC TLS vs INFL"),
          ("Visium", "object_recall", "Visium object recall"), ("Visium", "INFL_called_TLS", "Visium INFL called TLS")]


def png(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def fmt(v, d=3):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def ci(entry, d=3):
    if not entry:
        return "–"
    low, high = entry["CI95"]
    return f"{fmt(entry['value'], d)} <span class=muted>[{fmt(low, d)}, {fmt(high, d)}]</span>"


def excludes_zero(entry):
    low, high = entry["CI95_difference"]
    return bool(low > 0 or high < 0)


def diff_cell(entry, d=3):
    if not entry or not np.isfinite(entry["difference"]):
        return "<td>–</td>"
    low, high = entry["CI95_difference"]
    mark = " class=sig" if excludes_zero(entry) else ""
    return (f"<td{mark}>{entry['difference']:+.{d}f} <span class=muted>[{low:+.{d}f}, {high:+.{d}f}]</span>"
            f"<br><span class=muted>p = {entry['p_sign_flip']:.3g}</span></td>")


def comparisons(s):
    """(a, b) -> {"ESCA": metrics, "Visium": metrics}."""
    out = {}
    for cohort in ["ESCA", "Visium"]:
        for c in (s.get(cohort) or {}).get("comparisons", []):
            out.setdefault((c["a"], c["b"]), {})[cohort] = c["metrics"]
    return out


def forest(pairs, path):
    labels = [f"{a}  −  {b}" for a, b in pairs]
    fig, axes = plt.subplots(1, len(FOREST), figsize=(17, .5 * len(labels) + 1.6), sharey=True)
    y = np.arange(len(labels))[::-1]
    for ax, (cohort, metric, title) in zip(axes, FOREST):
        for yi, metrics in zip(y, pairs.values()):
            entry = (metrics.get(cohort) or {}).get(metric)
            if not entry or not np.isfinite(entry["difference"]):
                continue
            low, high = entry["CI95_difference"]
            colour = (POS if low > 0 else NEG) if excludes_zero(entry) else NULL
            ax.plot([low, high], [yi, yi], color=colour, lw=1.6)
            ax.plot(entry["difference"], yi, "o", color=colour, ms=4)
        ax.axvline(0, color=INK, lw=.8)
        ax.set_title(title, fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=7.5)
    fig.text(.995, .005, "difference a − b, 95% cluster-bootstrap CI; blue/orange = CI above/below 0, grey = includes 0 "
             "(colour is the sign, not benefit: higher INFL-called-TLS is worse)", ha="right", fontsize=7, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def spot_scatter(counts, path):
    samples = sorted(counts["sample"].unique())
    cols = min(4, len(samples))
    rows = int(np.ceil(len(samples) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 2.9 * rows), squeeze=False)
    for ax, sample in zip(axes.ravel(), samples):
        part = counts[counts["sample"] == sample]
        ax.scatter(part.he2cell_lymphocytes + np.random.default_rng(0).uniform(-.3, .3, len(part)),
                   part.hovernet_inflammatory + np.random.default_rng(1).uniform(-.3, .3, len(part)), s=2, alpha=.25, color=POS, lw=0)
        top = max(part.he2cell_lymphocytes.max(), part.hovernet_inflammatory.max(), 1)
        ax.plot([0, top], [0, top], color=MUTED, lw=.7)
        ax.set_title(sample, fontsize=8.5)
        ax.set_xlabel("HE2CELL T+B per spot", fontsize=7.5)
        ax.set_ylabel("HoVer-Net inflammatory per spot", fontsize=7.5)
    for ax in axes.ravel()[len(samples):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    out = Path(args.out)
    fig = out / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    s = json.loads((out / "summary.json").read_text())
    pairs = comparisons(s)
    if pairs:
        forest(pairs, fig / "paired_differences.png")
    counts_path = out / "visium_spot_cell_counts.csv.gz"
    if counts_path.exists():
        spot_scatter(pd.read_csv(counts_path), fig / "visium_spot_counts.png")
    write_html(s, pairs, out, fig)


def arm_rows(s, cohort, metrics):
    block = s.get(cohort)
    if not block:
        return "<tr><td colspan=9 class=muted>not yet available</td></tr>"
    rows = ""
    for name, entry in block["per_arm"].items():
        size = f"{entry.get('windows', entry.get('spots'))} / {entry.get('patients', entry.get('TLS_objects'))}"
        rows += f"<tr><td>{html.escape(name)}</td><td>{size}</td>" + "".join(f"<td>{ci(entry.get(m))}</td>" for m in metrics) + "</tr>"
    return rows


def write_html(s, pairs, out, fig):
    prog = s["progress"]
    status = ("<p class=warn><b>INTERIM.</b> HoVer-Net inference is incomplete: "
              f"{prog['esca_windows_done']} of {prog['esca_windows_total']} ESCA windows and "
              f"{len(prog['visium_sections_done'])} of {prog['visium_sections_total']} Visium sections "
              f"({', '.join(prog['visium_sections_done']) or 'none'}). Every arm, including the adopted HE2CELL arm, is "
              "restricted to the finished windows / sections; parameters tuned on the finished dev windows only. "
              "Numbers will change; do not cite them.</p>") if s["interim"] else \
             "<p>Complete: all ESCA windows and all 8 Visium sections.</p>"
    weights = s.get("weights") or {}
    provenance = (f"TIAToolbox {html.escape(str(weights.get('tiatoolbox_version')))} (torch {html.escape(str(weights.get('torch_version')))}), "
                  f"<code>{html.escape(str(weights.get('call')))}</code>; weights <code>{html.escape(str(weights.get('weights_url')))}</code>, "
                  f"sha256 <code>{html.escape(str(weights.get('weights_sha256')))}</code>.") if weights else "provenance pending"
    chosen = s.get("chosen_parameters") or {}
    edges = s.get("chosen_value_at_grid_edge") or {}
    params = "".join(f"<tr><td>{html.escape(a)}</td><td><code>{html.escape(json.dumps(p))}</code></td>"
                     f"<td>{fmt((s.get('dev_object_F1') or {}).get(a))}</td>"
                     f"<td>{html.escape(', '.join(k for k, v in edges.get(a, {}).items() if v) or 'none')}</td></tr>"
                     for a, p in chosen.items())
    esca_metrics = ["object_F1", "object_precision", "object_recall", "dense_specificity", "window_AUROC", "AUROC_vs_dense"]
    visium_metrics = ["AUROC_TLS_vs_all", "AUROC_TLS_vs_INFL", "object_recall", "INFL_called_TLS", "spot_F1"]
    diff_rows = "".join(f"<tr><td>{html.escape(a)} − {html.escape(b)}</td>"
                        + "".join(diff_cell((m.get(cohort) or {}).get(metric)) for cohort, metric, _ in FOREST) + "</tr>"
                        for (a, b), m in pairs.items())
    cell = s.get("Visium_cell_level")
    cell_html = "<p class=muted>Not yet available.</p>"
    if cell:
        agree = cell["counts_agreement"]
        rows = "".join(f"<tr><td>{html.escape(r['score'])}</td><td>{r['gene']}</td><td>{fmt(r['mean_spearman'])} "
                       f"<span class=muted>[{fmt(r['CI95'][0])}, {fmt(r['CI95'][1])}]</span></td>"
                       f"<td class=muted>{html.escape(', '.join(f'{k} {v:.2f}' for k, v in r['per_section'].items()))}</td></tr>"
                       for r in cell["vs_RNA"])
        paired = "".join(f"<tr><td>{html.escape(r['a'])} − {html.escape(r['b'])}</td><td>{r['gene']}</td>"
                         + diff_cell({**r, "p_sign_flip": float('nan')}).replace("<br><span class=muted>p = nan</span>", "")
                         + f"<td>{r['sections_a_higher']} of {len(cell['sections'])}</td></tr>" for r in cell["paired_HoVerNet_minus_HE2CELL"])
        cell_html = f"""<p>{cell['spots']} in-tissue spots in {len(cell['sections'])} sections. Counts are nuclei whose centroid falls in the
55 µm spot disk (8 µm grid, as visium_eval). RNA: log1p counts per 10k. ρ is computed within each section and averaged;
95% CI from the same 2000 section-cluster resamples (paired for differences).</p>
<p><b>HoVer-Net inflammatory vs HE2CELL T+B per spot:</b> mean Spearman ρ {fmt(agree['mean_spearman'])}
<span class=muted>[{fmt(agree['CI95'][0])}, {fmt(agree['CI95'][1])}]</span>
<span class=muted>({html.escape(', '.join(f'{k} {v:.2f}' for k, v in agree['per_section'].items()))})</span></p>
<table><tr><th>Per-spot count</th><th>Gene</th><th>Mean Spearman ρ [CI]</th><th>Per section</th></tr>{rows}</table>
<h3>Paired: HoVer-Net minus HE2CELL (ρ with the same gene)</h3>
<table><tr><th>a − b</th><th>Gene</th><th>Δρ [CI]</th><th>Sections where a is higher</th></tr>{paired}</table>
{f'<img class="fig" src="{png(fig / "visium_spot_counts.png")}" alt="Per-spot counts">' if (fig / "visium_spot_counts.png").exists() else ''}"""
    shares = (s.get("Visium") or {}).get("hovernet_type_shares") or {}
    share_rows = "".join(f"<tr><td>{k}</td>" + "".join(f"<td>{v.get(t, 0):.3f}</td>" for t in ["inflammatory", "neoplastic", "connective", "epithelial", "dead", "no_label"]) + "</tr>"
                         for k, v in shares.items())
    esca_shares = s.get("ESCA_hovernet_type_shares_by_category") or {}
    share_rows += "".join(f"<tr><td>ESCA {k} windows</td>" + "".join(f"<td>{v.get(t, 0):.3f}</td>" for t in ["inflammatory", "neoplastic", "connective", "epithelial", "dead", "no_label"]) + "</tr>"
                          for k, v in esca_shares.items())
    ev = s.get("esca_windows_evaluated", {})
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HoVer-Net TLS Baseline</title><style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--line:#e4e3df;--card:#fff;--pos:#1d5fae;--warn:#a04a12}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--line:#3a3936;--card:#232321;--pos:#7fb2f0;--warn:#f0a070}}}}
:root[data-theme="dark"]{{--bg:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--line:#3a3936;--card:#232321;--pos:#7fb2f0;--warn:#f0a070}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:28px 16px 64px}} h1{{font-size:26px;margin:0 0 4px}} h2{{margin-top:36px;border-top:1px solid var(--line);padding-top:18px}}
.muted{{color:var(--muted)}} table{{border-collapse:collapse;width:100%;font-size:12.5px;display:block;overflow-x:auto}}
th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;white-space:nowrap;vertical-align:top}} img.fig{{max-width:100%;background:#fff;border-radius:8px;border:1px solid var(--line)}}
td.sig{{color:var(--pos);font-weight:600}} .warn{{color:var(--warn)}} code{{font-size:12px;word-break:break-all}}
</style></head><body><main>
<h1>TLS detection: HoVer-Net inflammatory-cell baseline</h1>
<p class="muted">A second, independent, published cell-level method. HoVer-Net (PanNuke) classifies every nucleus; its "inflammatory" class is the
standard lymphocyte proxy. TLS candidates are regions of dense inflammatory cells, found either with HE2CELL's own smoothed-density machinery or by
DBSCAN clustering (Barmpoutis-style lymphocyte clustering). ESCA: TCGA-ESCA windows, reference HookNet-TLS. Visium: 8 USZ sections, reference pathologist spot labels.</p>
{status}

<h2>Method</h2>
<ul>
<li><b>Model.</b> {provenance} HoVer-Net "fast" mode, 256-px input / 164-px output at 0.25 µm/px; classes neoplastic, inflammatory, connective, dead, non-neoplastic epithelial.</li>
<li><b>Images.</b> Pixel-identical to HE2CELL. ESCA: each window regenerated from the SVS at native level 0 as the 3328-px square covered by
esca_windows' 4×4 patches of 1024 px at stride 768 (no stain normalisation), segmented as one image, nuclei kept in the 3072-px window core (= the union of
the 16 patch cores). Visium: tissue upsampled bicubically to 0.25 µm/px by <code>visium_cells.patch_grid</code>, one call per 1024-px patch, nuclei kept in the
768-px patch core. HoVer-Net's own 256&nbsp;→&nbsp;164 px sliding window is unchanged; its post-processing tile is enlarged to 3444 px so each image is a single tile
(TIAToolbox's default 1024-px tile re-infers seam strips, 2.6× the compute; outputs were identical on a test set of 8 patches).</li>
<li><b>(a) Inflammatory density.</b> <code>esca_analysis.detect</code> unchanged on the inflammatory-count grid (8 µm grid, Gaussian σ, threshold, minimum area, minimum cells),
grid σ ∈ {{15, 25}} µm × threshold × min area ∈ {{5000, 10000, 20000}} µm² × min cells ∈ {{20, 50}}.</li>
<li><b>(b) Inflammatory DBSCAN.</b> DBSCAN on inflammatory centroids in valid tissue, eps 30 µm, min_samples ∈ {{20, 50, 100}}; a cluster whose convex hull is ≥ 20 000 µm² is a TLS candidate
(hull rasterised on the 8 µm grid, clipped to valid tissue). Window score = largest cluster size; Visium spot score = fraction of the spot covered (near-binary, like two-stage/HookNet).</li>
<li><b>Tuning.</b> Best ESCA dev object F1 (dev = <code>split == "dev"</code> patients), frozen, then scored once on ESCA test and on Visium. Nothing is tuned on test patients or Visium labels.</li>
<li><b>Statistics.</b> <code>esca_paired_stats</code> / <code>visium_paired_stats</code>: per-arm patient- (ESCA) or section- (Visium) cluster bootstrap CIs, paired differences on the same 2000 resamples, sign-flip permutation p (Visium exhaustive over 256 flips; minimum p 0.0078).</li>
</ul>

<h2>Frozen parameters (dev-tuned)</h2>
<p class=muted>ESCA windows with HoVer-Net cells: {ev.get('all', 0)} (dev {ev.get('dev', 0)}, test {ev.get('test', 0)}; {ev.get('test_patients', 0)} test patients).</p>
<table><tr><th>Arm</th><th>Parameters</th><th>ESCA dev F1</th><th>At grid edge</th></tr>{params or '<tr><td colspan=4 class=muted>not yet tuned</td></tr>'}</table>

<h2>ESCA test patients (reference HookNet-TLS)</h2>
<table><tr><th>Arm</th><th>windows / patients</th>{''.join(f'<th>{m}</th>' for m in esca_metrics)}</tr>{arm_rows(s, 'ESCA', esca_metrics)}</table>

<h2>Visium sections (reference pathologist labels)</h2>
<table><tr><th>Arm</th><th>spots / TLS objects</th>{''.join(f'<th>{m}</th>' for m in visium_metrics)}</tr>{arm_rows(s, 'Visium', visium_metrics)}</table>
<p class=muted>Density arms are scored by smoothed density over the spot; DBSCAN, HookNet by spot coverage (near-binary), so their AUROCs are not directly comparable with density scores.</p>

<h2>Paired differences</h2>
{f'<img class="fig" src="{png(fig / "paired_differences.png")}" alt="Forest plot">' if (fig / "paired_differences.png").exists() else ''}
<table><tr><th>a − b</th>{''.join(f'<th>{t}</th>' for _, _, t in FOREST)}</tr>{diff_rows or '<tr><td class=muted>not yet available</td></tr>'}</table>

<h2>Cell-level comparison on Visium</h2>
{cell_html}

<h2>HoVer-Net class shares</h2>
<table><tr><th>Set</th><th>inflammatory</th><th>neoplastic</th><th>connective</th><th>epithelial</th><th>dead</th><th>no label</th></tr>{share_rows or '<tr><td class=muted>–</td></tr>'}</table>

<h2>Provenance</h2>
<p class="muted">Cells: outputs/baselines_hovernet_20261026/cells/ (provenance.json, timing.csv). Code: deployment/baselines/{{hovernet_cells.py, hovernet_eval.py, hovernet_report.py, run_hovernet_queue.sh}}.
Adopted HE2CELL arms: outputs/esca_tls_20261019/analysis_he2cell_ft_tau05/per_window_results.csv and outputs/visium_tls_validation/analysis/spot_table.csv.gz (read only). Numbers: summary.json next to this report.</p>
</main></body></html>"""
    (out / "Baseline_report.html").write_text(page)
    print("wrote", out / "Baseline_report.html")


if __name__ == "__main__":
    main()
