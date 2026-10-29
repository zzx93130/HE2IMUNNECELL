"""Figures and HTML report for the Visium pathologist-label validation."""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "esca_tls"))
import visium_cells as V  # noqa: E402
import visium_eval as E  # noqa: E402
import esca_analysis as A  # noqa: E402

ANALYSIS = V.OUT / E.ANALYSIS_DIR
FIG = ANALYSIS / "figures"
COLOURS = {"finetuned": "#2a78d6", "original": "#eb6834", "none": "#1baf7a", "HookNet-TLS (reference)": "#6250d6", "finetuned_tau05": "#eda100", "finetuned40": "#008300", "finetuned40_tau05": "#e87ba4"}
NAMES = {"finetuned": "Fine-tuned classifier", "original": "Original classifier", "none": "Nuclei only (no classifier)",
         "finetuned_tau05": "Fine-tuned 10-pt + prior correction (τ 0.5)",
         "finetuned40": "Fine-tuned 40-pt", "finetuned40_tau05": "Fine-tuned 40-pt + prior correction (τ 0.5)",
         "HookNet-TLS (reference)": "HookNet-TLS (reference)"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "figure.dpi": 150})


def png(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def jpg(image):
    ok, data = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
    return "data:image/jpeg;base64," + base64.b64encode(data.tobytes()).decode()


def fmt(v, d=2):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def roc(ax, sub, column, colour, label):
    y = sub.label.isin(E.POSITIVE).to_numpy()
    s = sub[column].fillna(0).to_numpy()
    order = np.argsort(-s, kind="stable")
    tpr = np.r_[0, np.cumsum(y[order]) / max(y.sum(), 1)]
    fpr = np.r_[0, np.cumsum(~y[order]) / max((~y).sum(), 1)]
    ax.plot(fpr, tpr, color=colour, lw=2, label=f"{label}  {A.auc(y, s):.2f}")


def figures(spots):
    FIG.mkdir(parents=True, exist_ok=True)
    scored = spots[spots.label.isin(E.POSITIVE | E.NEGATIVE) & (spots.in_tissue == 1)]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9))
    for ax, (title, keep) in zip(axes, [("TLS vs normal, tumour, inflammation", E.POSITIVE | E.NEGATIVE),
                                        ("TLS vs inflammation only", E.POSITIVE | {"INFL"})]):
        for classifier in ["finetuned_tau05", "finetuned", "original"]:
            sub = scored[(scored.classifier == classifier) & scored.label.isin(keep)]
            roc(ax, sub, "B_density_at_spot", COLOURS[classifier], f"B density, {NAMES[classifier].lower()}")
        sub = scored[(scored.classifier == "original") & scored.label.isin(keep)]
        roc(ax, sub, "nuclei_density_at_spot", COLOURS["none"], "Nuclear density, no classifier")
        hk = sub[sub["cover_HookNet-TLS"].notna()]
        roc(ax, hk, "cover_HookNet-TLS", COLOURS["HookNet-TLS (reference)"], "HookNet-TLS coverage")
        ax.plot([0, 1], [0, 1], color=MUTED, lw=.8, ls=":")
        ax.set(title=title, xlabel="False-positive rate (spots)", ylabel="True-positive rate (spots)", xlim=(0, 1), ylim=(0, 1.01))
        ax.legend(frameon=False, fontsize=7, loc="lower right", title="AUROC", title_fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "spot_roc.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    order = ["MTLS", "GC", "ETLS", "LN", "INFL", "NOR", "TUM"]
    part = spots[(spots.in_tissue == 1) & spots.label.isin(order)]
    width = .38
    for k, classifier in enumerate(["finetuned", "original"]):
        data = [part[(part.classifier == classifier) & (part.label == l)].B_fraction.dropna().to_numpy() for l in order]
        positions = np.arange(len(order)) + (k - .5) * width
        box = ax.boxplot(data, positions=positions, widths=width * .85, showfliers=False, patch_artist=True,
                         medianprops={"color": INK})
        for patch in box["boxes"]:
            patch.set(facecolor=COLOURS[classifier], alpha=.75, edgecolor=MUTED)
    ax.set_xticks(range(len(order)), [f"{l}\n(n={int(((part.classifier == 'original') & (part.label == l)).sum())})" for l in order])
    ax.set_ylabel("Predicted B fraction in spot")
    ax.legend([plt.Rectangle((0, 0), 1, 1, fc=COLOURS[c], alpha=.75) for c in ["finetuned", "original"]],
              [NAMES[c] for c in ["finetuned", "original"]], frameon=False, fontsize=7.5, loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "b_fraction_by_label.png"); plt.close(fig)


def gallery(spots, max_tiles=8):
    """Crops around pathologist TLS and INFL spot groups: spots coloured by label, predicted B/T dots."""
    colours = {"ETLS": (255, 140, 0), "MTLS": (200, 0, 0), "GC": (150, 0, 150), "INFL": (0, 120, 255), "TUM": (0, 150, 0)}
    tiles = {"TLS spots": [], "Inflammation spots": []}
    rng = np.random.default_rng(0)
    for sample in spots["sample"].unique():
        image = tifffile.imread(V.DATA / "tif_slides" / f"{sample}.tif")[..., :3]
        cells = pd.read_csv(V.OUT / E.CELLS_DIR / sample / "cells.csv.gz",
                            usecols=["x_native", "y_native", "celltype_finetuned", "celltype_original"])
        sub = spots[(spots["sample"] == sample) & (spots.classifier == "finetuned") & (spots.in_tissue == 1)]
        for group, labels in [("TLS spots", {"MTLS", "GC", "ETLS"}), ("Inflammation spots", {"INFL"})]:
            candidates = sub[sub.label.isin(labels)]
            if candidates.empty:
                continue
            centre = candidates.iloc[rng.integers(len(candidates))]
            half = 250
            x0, y0 = int(centre.x) - half, int(centre.y) - half
            if x0 < 0 or y0 < 0 or x0 + 2 * half > image.shape[1] or y0 + 2 * half > image.shape[0]:
                continue
            panels = []
            for classifier in ["original", "finetuned"]:
                crop = image[y0:y0 + 2 * half, x0:x0 + 2 * half].copy()
                near = cells[(cells.x_native.between(x0, x0 + 2 * half)) & (cells.y_native.between(y0, y0 + 2 * half))]
                for name, colour in [("B", (235, 104, 52)), ("T", (42, 120, 214))]:
                    for c in near[near[f"celltype_{classifier}"] == name].itertuples():
                        cv2.circle(crop, (int(c.x_native - x0), int(c.y_native - y0)), 2, colour, -1)
                for s in sub[(sub.x.between(x0, x0 + 2 * half)) & (sub.y.between(y0, y0 + 2 * half))].itertuples():
                    if s.label in colours:
                        cv2.circle(crop, (int(s.x - x0), int(s.y - y0)), int(E.SPOT_RADIUS_UM / V.native_mpp(sample)), colours[s.label], 2)
                panels.append(crop)
            tiles[group].append((sample, centre.label, np.concatenate(panels, 1)))
    html_parts = []
    for group, items in tiles.items():
        figures_html = "".join(f'<figure><img src="{jpg(t)}" alt="{s} {l}"><figcaption>{s} · centre spot {l} · left original, right fine-tuned</figcaption></figure>'
                               for s, l, t in items[:max_tiles])
        html_parts.append(f"<h3>{group}</h3><div class='gallery'>{figures_html}</div>")
    return "".join(html_parts)


def main():
    spots = pd.read_csv(ANALYSIS / "spot_table.csv.gz")
    results = json.loads((ANALYSIS / "results.json").read_text())
    figures(spots)
    gallery_html = gallery(spots)
    summaries = {s: json.loads((V.OUT / E.CELLS_DIR / s / "summary.json").read_text()) for s in results["samples"]}
    auroc_rows = "".join(
        f"<tr><td>{html.escape(NAMES.get(r['classifier'], r['classifier']))}</td><td>{html.escape(r['score'])}</td><td>{html.escape(r['contrast'])}</td>"
        f"<td>{fmt(r['AUROC'])}</td><td>{fmt(r['CI95'][0])}–{fmt(r['CI95'][1])}</td></tr>"
        for r in results["spot_AUROC"] if r["score"] in ("Smoothed B density", "B fraction in spot", "Nuclear density (no classifier)", "HookNet TLS coverage", "Smoothed B+T density"))
    detection_rows = "".join(
        f"<tr><td>{html.escape(NAMES.get(d['classifier'], d['classifier']))}</td><td>{html.escape(d['detector'])}</td><td>{fmt(d['sensitivity'])}</td>"
        f"<td>{fmt(d['specificity'])}</td><td>{fmt(d['PPV'])}</td><td>{fmt(d['F1'])} [{fmt(d['CI95_F1'][0])}–{fmt(d['CI95_F1'][1])}]</td>"
        f"<td>{fmt(d['INFL_called_TLS'])}</td></tr>" for d in results["spot_detection_frozen"])
    object_rows = "".join(
        f"<tr><td>{html.escape(NAMES.get(o['classifier'], o['classifier']))}</td><td>{html.escape(o['detector'])}</td><td>{o['TLS_objects']}</td>"
        f"<td>{fmt(o['recall_all'])}</td><td>{fmt(o['recall_early'])}</td><td>{fmt(o['recall_mature'])}</td>"
        f"<td>{o['components_on_TLS'] if o['components_on_TLS'] is not None else '–'} / {o['components_on_negative_only'] if o['components_on_negative_only'] is not None else '–'}</td></tr>"
        for o in results["object_level_frozen"] if not (o["detector"] in ("HookNet-TLS", "All nuclei (no classifier)") and o["classifier"] == "finetuned"))
    maturity_rows = "".join(
        f"<tr><td>{NAMES[c]}</td><td>{html.escape(f)}</td><td>{fmt(v['median_early'], 3)}</td><td>{fmt(v['median_mature'], 3)}</td><td>{fmt(v['AUROC_mature'])}</td></tr>"
        for c, features in results["maturity"].items() for f, v in features.items())
    rna_rows = "".join(
        f"<tr><td>{NAMES[g['classifier']]}</td><td>{html.escape(g['score'])}</td><td>{html.escape(g['genes'])}</td><td>{fmt(g['median_spearman_across_samples'])}</td>"
        f"<td>{', '.join(fmt(v) for v in g['per_sample'])}</td></tr>" for g in results["spot_RNA_check"])
    sample_rows = "".join(
        f"<tr><td>{s}</td><td>{fmt(v['native_mpp'], 3)}</td><td>{v['upsampling']:.2f}×</td><td>{v['cells']:,}</td>"
        f"<td>{v['counts']['original'].get('B', 0) / v['cells']:.1%} / {v['counts']['finetuned'].get('B', 0) / v['cells']:.1%}</td>"
        f"<td>{v['counts']['original'].get('Other', 0) / v['cells']:.1%} / {v['counts']['finetuned'].get('Other', 0) / v['cells']:.1%}</td></tr>"
        for s, v in summaries.items())
    counts = results["spots"]
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Visium TLS Validation</title><style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--line:#e4e3df;--card:#fff}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--line:#3a3936;--card:#232321}}}}
:root[data-theme="dark"]{{--bg:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--line:#3a3936;--card:#232321}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1080px;margin:0 auto;padding:28px 16px 64px}} h1{{font-size:26px;margin:0 0 4px}} h2{{margin-top:36px;border-top:1px solid var(--line);padding-top:18px}}
.muted{{color:var(--muted)}} table{{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow-x:auto}}
th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;white-space:nowrap}} img.fig{{max-width:100%;background:#fff;border-radius:8px;border:1px solid var(--line)}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:10px}} figure{{margin:0}} figure img{{width:100%;border-radius:6px}} figcaption{{font-size:11px;color:var(--muted)}}
</style></head><body><main>
<h1>HE2CELL TLS identification vs pathologist Visium labels</h1>
<p class="muted">External validation on 8 USZ Visium sections (5 lung, 3 kidney cancer; Zenodo 14620362). Truth is the pathologists' per-spot annotation (manual_anno_tls). Every detector setting was frozen on TCGA-ESCA; nothing was tuned on these images or labels.</p>
<p>Labelled in-tissue spots used: {', '.join(f'{k} {v}' for k, v in counts.items())}. TLS = ETLS + MTLS + GC; negatives = NOR + TUM + INFL; LN and unassigned spots are excluded from scoring.</p>

<h2>Important caveat: image resolution</h2>
<p>The released H&amp;E images are about 0.78–0.80 µm/px (≈12.5× magnification), not 40×. HE2CELL classifies 64-px crops at 0.25 µm/px, so tissue was upsampled about 3.1× with bicubic interpolation before segmentation. Nuclear detail is therefore much poorer than in TCGA slides, and these results measure robustness to low-resolution input as well as TLS identification.</p>
<table><tr><th>Sample</th><th>Native µm/px</th><th>Upsampling</th><th>Cells</th><th>B share, original / fine-tuned</th><th>Other share, original / fine-tuned</th></tr>{sample_rows}</table>

<h2>Spot-level discrimination</h2>
<p>Scores are taken at each 55 µm spot. B density is the smoothed (σ = 25 µm) predicted-B density over the spot; HookNet coverage is the fraction of the spot inside published-filter HookNet-TLS polygons. Brackets are 95% bootstrap intervals over samples (n = 8, so intervals are wide).</p>
<img class="fig" src="{png(FIG / 'spot_roc.png')}" alt="Spot-level ROC curves">
<table><tr><th>Model</th><th>Score</th><th>Contrast</th><th>AUROC</th><th>95% CI</th></tr>{auroc_rows}</table>
<img class="fig" src="{png(FIG / 'b_fraction_by_label.png')}" alt="Predicted B fraction per pathologist label">

<h2>Frozen detectors, spot level</h2>
<p>A spot is called TLS when at least half of it lies inside a detected region. Thresholds are the TCGA-ESCA values.</p>
<table><tr><th>Model</th><th>Detector</th><th>Sensitivity</th><th>Specificity</th><th>PPV</th><th>F1 [95% CI]</th><th>INFL spots called TLS</th></tr>{detection_rows}</table>

<h2>Frozen detectors, TLS level</h2>
<p>A pathologist TLS is a connected group of TLS-labelled spots; it counts as found when any of its spots is called. Detector regions are classed by the labelled spots they cover (≥ 50% of a spot).</p>
<table><tr><th>Model</th><th>Detector</th><th>Pathologist TLS</th><th>Recall, all</th><th>Recall, early</th><th>Recall, mature</th><th>Regions on TLS / on negatives only</th></tr>{object_rows}</table>

<h2>Early versus mature TLS</h2>
<p>Per pathologist TLS group: mature if MTLS + GC spots are at least as many as ETLS spots. AUROC is how well each feature ranks mature groups above early ones.</p>
<table><tr><th>Model</th><th>Feature</th><th>Median, early</th><th>Median, mature</th><th>AUROC (mature)</th></tr>{maturity_rows}</table>

<h2>Cell types against the spot's own RNA</h2>
<p>Spearman correlation, per sample, between predicted cell counts in each in-tissue spot and the spot's mean log-normalised marker expression. "cells" (all nuclei) is the control: a cell-type call is informative only if it beats total cellularity.</p>
<table><tr><th>Model</th><th>Predicted</th><th>Genes</th><th>Median ρ</th><th>Per sample</th></tr>{rna_rows}</table>

<h2>Examples</h2>
<p class="muted">Pathologist spots: early TLS orange, mature TLS red, GC purple, inflammation blue, tumour green. Dots: predicted B orange, predicted T blue.</p>
{gallery_html}
</main></body></html>"""
    (ANALYSIS / "Visium_TLS_validation_report.html").write_text(page)
    print("Report written", ANALYSIS / "Visium_TLS_validation_report.html")


if __name__ == "__main__":
    main()
