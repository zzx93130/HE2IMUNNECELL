"""Patient-level checks, figures, review gallery and HTML report for the ESCA TLS validation."""
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
from PIL import Image
from scipy import ndimage, stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import esca_analysis as A  # noqa: E402
from esca_common import OUT, ROOT, EXCLUDED_DIAGNOSES, save_json  # noqa: E402
from esca_qc import slide_qc  # noqa: E402

ANALYSIS = OUT / ("analysis" if A.RESULT == "he2cell" else f"analysis_{A.RESULT}")
FIG = ANALYSIS / "figures"
METHOD_COLOURS = {"B (HE2CELL)": "#2a78d6", "B+T (HE2CELL)": "#eb6834", "All nuclei (no classifier)": "#1baf7a", A.TWO_STAGE: "#eda100"}
CELL_COLOURS = {"B": "#eb6834", "T": "#2a78d6"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "figure.dpi": 150})


def hooknet_slide_table():
    """Slide-level HookNet TLS burden on artifact-free valid tissue, for every processed slide."""
    rows = []
    for summary_path in sorted((OUT / "slides").glob("*/hooknet/summary.json")):
        folder = summary_path.parents[1]
        summary = json.loads(summary_path.read_text())
        qc = slide_qc(folder)
        ratio = summary["source_dimensions"][0] / summary["dimensions_at_0p5um"][0]
        features = [f for f in json.loads((folder / "hooknet" / "native_polygons.geojson").read_text())["features"]
                    if f["properties"]["predicted_label"] == "tls" and f["properties"]["passes_published_filter"]]
        kept, area = 0, 0.0
        for f in features:
            ring = np.asarray(f["geometry"]["coordinates"][0], float) / 16  # 0.5 µm px -> 8 µm px
            canvas = np.zeros(qc["valid"].shape, np.uint8)
            cv2.fillPoly(canvas, [np.round(ring).astype(np.int32)], 1)
            inside = canvas.astype(bool)
            if inside.sum() and (qc["artifact"][inside] | ~qc["valid"][inside]).mean() <= .25:
                kept += 1
                area += f["properties"]["area_um2"]
        valid_mm2 = qc["valid"].sum() * 64 / 1e6
        rows.append({"barcode": folder.name, "case_id": folder.name[:12], "mpp": summary["source_mpp"],
                     "magnification": summary.get("aperio_AppMag"), "otsu_tissue_mm2": summary["tissue_area_mm2"],
                     "valid_tissue_mm2": valid_mm2,
                     "artifact_mm2": qc["artifact"].sum() * 64 / 1e6, "hooknet_TLS_filtered": len(features),
                     "hooknet_TLS_artifact_free": kept, "hooknet_TLS_area_mm2": area / 1e6,
                     "hooknet_TLS_per_100mm2": 100 * kept / max(valid_mm2, 1e-9),
                     "hooknet_TLS_area_fraction": area / 1e6 / max(valid_mm2, 1e-9),
                     "hooknet_GC_filtered": summary["native"]["filtered_GC_polygons"]})
    return pd.DataFrame(rows)


def spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 8:
        return {"rho": np.nan, "p": np.nan, "n": int(m.sum())}
    r = stats.spearmanr(x[m], y[m])
    return {"rho": float(r.statistic), "p": float(r.pvalue), "n": int(m.sum())}


def png(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def jpg_data(image, quality=82):
    ok, data = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(data.tobytes()).decode()


def gallery_tile(barcode, window_id, x, predicted, rejected=None):
    image = np.asarray(Image.open(OUT / "slides" / barcode / "he2cell" / f"{window_id}.jpg").convert("RGB")).copy()
    scale = image.shape[0] / x["n"]
    layers = [(x["reference"], (27, 175, 122)), (predicted, (11, 11, 11))]
    if rejected is not None:
        layers.append((rejected, (255, 255, 255)))  # stage-1 aggregates that stage 2 rejected
    for mask, colour in layers:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [(c * scale).astype(np.int32) for c in contours]
        cv2.drawContours(image, contours, -1, colour, 3)
    px_per_um = image.shape[0] / (x["n"] * A.GRID_UM)
    for name, hexcode in CELL_COLOURS.items():
        colour = tuple(int(hexcode[i:i + 2], 16) for i in (1, 3, 5))
        for cx, cy in x["cells_xy_um"][x["celltype"] == name]:
            cv2.circle(image, (int(cx * px_per_um), int(cy * px_per_um)), 2, colour, -1)
    return cv2.resize(image, (360, 360), interpolation=cv2.INTER_AREA)


def main():
    windows, inputs, chosen = A.main()
    analysis = ANALYSIS
    FIG.mkdir(parents=True, exist_ok=True)
    results = json.loads((analysis / "detection_results.json").read_text())
    per_window = pd.read_csv(analysis / "per_window_results.csv")
    rna = pd.read_csv(OUT / "rna" / "patient_signatures.csv").set_index("case_id")
    clinical = pd.read_csv(OUT / "rna" / "clinical.csv").set_index("case_id")

    # ---------- Cohort flow ----------
    listed = [l for l in (OUT / "tar_listing.txt").read_text().splitlines() if l.strip()]
    flow = {"svs_in_archive": sum(l.endswith(".svs") for l in listed),
            "incomplete_downloads_in_archive": sum(l.endswith(".svs.partial") for l in listed),
            "hooknet_complete": len(list((OUT / "slides").glob("*/hooknet/summary.json"))),
            "hooknet_failed": len(list((OUT / "slides").glob("*/hooknet/failed.json"))),
            "he2cell_complete": len(list((OUT / "slides").glob(f"*/{A.RESULT}/cells.csv.gz"))),
            "excluded": {}}
    for path in (OUT / "slides").glob(f"*/{A.RESULT}/excluded.json"):
        reason = json.loads(path.read_text())["reason"]
        flow["excluded"][reason] = flow["excluded"].get(reason, 0) + 1

    # ---------- Patient-level: HookNet reference vs RNA ----------
    slides = hooknet_slide_table()
    slides = slides[~slides.case_id.map(clinical.primary_diagnosis).isin(EXCLUDED_DIAGNOSES)]
    slides.to_csv(analysis / "slide_hooknet_burden.csv", index=False)
    per_case = slides.groupby("case_id")[["hooknet_TLS_per_100mm2", "hooknet_TLS_area_fraction"]].mean().join(rna, how="inner")
    rna_checks = {}
    for target in ["chemokine12_PC1", "log2TPM_CXCL13", "B_cell_mean_z", "log2TPM_MS4A1", "log2TPM_LAMP3"]:
        rna_checks[f"HookNet TLS/100mm2 vs {target}"] = spearman(per_case.hooknet_TLS_per_100mm2.to_numpy(), per_case[target].to_numpy())

    # ---------- Patient-level: HE2CELL random windows vs RNA ----------
    b_params = chosen["B (HE2CELL)"]
    random_rows = []
    for w in windows[windows.category == "random"].itertuples():
        x = inputs[(w.barcode, w.window_id)]
        predicted, _ = A.detect(x, ["B"], **b_params)
        random_rows.append({"case_id": w.case_id, "valid_mm2": w.valid_mm2, "B": w.n_B, "T": w.n_T,
                            "all": w.n_B + w.n_T + w.n_Other + w.n_Macrophage,
                            "model_TLS_mm2": predicted.sum() * A.GRID_UM ** 2 / 1e6})
    random_frame = pd.DataFrame(random_rows).groupby("case_id").sum()
    random_frame["B_per_mm2"] = random_frame.B / random_frame.valid_mm2
    random_frame["T_per_mm2"] = random_frame["T"] / random_frame.valid_mm2
    random_frame["B_fraction"] = random_frame.B / random_frame["all"]
    random_frame["model_TLS_area_fraction"] = random_frame.model_TLS_mm2 / random_frame.valid_mm2
    model_case = random_frame.join(rna, how="inner").join(per_case[["hooknet_TLS_per_100mm2"]], how="left")
    model_case.to_csv(analysis / "patient_random_window_model_vs_rna.csv")
    for score, targets in {"B_per_mm2": ["log2TPM_MS4A1", "log2TPM_CD19", "log2TPM_CD79A", "B_cell_mean_z"],
                           "T_per_mm2": ["log2TPM_CD3E", "log2TPM_CD8A"],
                           "B_fraction": ["log2TPM_MS4A1", "B_cell_mean_z"],
                           "model_TLS_area_fraction": ["chemokine12_PC1", "log2TPM_CXCL13"]}.items():
        for target in targets:
            rna_checks[f"HE2CELL random-window {score} vs {target}"] = spearman(model_case[score].to_numpy(), model_case[target].to_numpy())
    # The model's own whole-slide-free TLS signal: max B density over all of a patient's windows vs HookNet burden.
    b_rows = per_window[per_window.method == "B (HE2CELL)"]
    case_max = b_rows.groupby("case_id").max_density.max().to_frame("max_B_density_any_window").join(per_case, how="inner")
    rna_checks["HE2CELL max B density (all windows) vs chemokine12_PC1"] = spearman(case_max.max_B_density_any_window.to_numpy(), case_max.chemokine12_PC1.to_numpy())
    rna_checks["HE2CELL max B density (all windows) vs CXCL13"] = spearman(case_max.max_B_density_any_window.to_numpy(), case_max.log2TPM_CXCL13.to_numpy())

    # ---------- Composition inside independently defined HookNet TLS (not circular) ----------
    comp_rows = []
    for w in windows.itertuples():
        x = inputs[(w.barcode, w.window_id)]
        total = sum(x["counts"].values())
        for region, mask in [("HookNet TLS", x["reference"]), ("outside TLS", x["valid"] & ~x["reference"])]:
            n = total[mask].sum()
            if n >= 50:
                comp_rows.append({"barcode": w.barcode, "split": w.split, "category": w.category, "region": region, "cells": n,
                                  "B": x["counts"]["B"][mask].sum() / n, "T": x["counts"]["T"][mask].sum() / n})
    composition = pd.DataFrame(comp_rows)
    composition.to_csv(analysis / "composition_inside_vs_outside_HookNet_TLS.csv", index=False)
    org_stats = {}
    inside = composition[composition.region == "HookNet TLS"].groupby("barcode")[["B", "T"]].mean()
    outside = composition[(composition.region == "outside TLS") & (composition.category == "random")].groupby("barcode")[["B", "T"]].mean()
    paired = inside.join(outside, lsuffix="_TLS", rsuffix="_random", how="inner")
    if len(paired) >= 8:
        org_stats = {"slides": len(paired),
                     "median_B_in_TLS": float(paired.B_TLS.median()), "median_B_random": float(paired.B_random.median()),
                     "median_T_in_TLS": float(paired.T_TLS.median()), "median_T_random": float(paired.T_random.median()),
                     "slides_B_TLS_gt_random": int((paired.B_TLS > paired.B_random).sum()),
                     "B_wilcoxon_p": float(stats.wilcoxon(paired.B_TLS, paired.B_random, alternative="greater").pvalue),
                     "T_wilcoxon_p": float(stats.wilcoxon(paired.T_TLS, paired.T_random, alternative="greater").pvalue),
                     "slides_with_B_lt_2pct_in_TLS": int((paired.B_TLS < .02).sum())}

    # ---------- Figures ----------
    test = per_window[(per_window.split == "test") & (per_window.hooknet_label != "ambiguous")]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    for ax, subset, title in [(axes[0], test, "All test windows"),
                              (axes[1], test[test.category.isin(["tls", "dense"])], "HookNet-TLS vs dense hard negatives")]:
        for method, colour in METHOD_COLOURS.items():
            part = subset[subset.method == method]
            y = (part.hooknet_label == "TLS").to_numpy()
            if y.all() or (~y).all():
                continue
            order = np.argsort(-part.max_density.to_numpy())
            tpr = np.concatenate([[0], np.cumsum(y[order]) / y.sum()])
            fpr = np.concatenate([[0], np.cumsum(~y[order]) / (~y).sum()])
            ax.plot(fpr, tpr, color=colour, lw=2, label=f"{method}  AUROC {A.auc(y, part.max_density):.2f}")
        ax.plot([0, 1], [0, 1], color=MUTED, lw=.8, ls=":")
        ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title=title, xlim=(0, 1), ylim=(0, 1.01))
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "roc.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    metrics = [("object_precision", "Precision", "CI95_object_precision"), ("object_recall", "Recall", "CI95_object_recall"),
               ("object_F1", "F1", "CI95_object_F1")]
    width = .2
    for i, (method, colour) in enumerate(METHOD_COLOURS.items()):
        entry = results["results"][method]["test"]
        values = [entry[m] for m, _, _ in metrics]
        errors = np.array([[v - entry[c][0], entry[c][1] - v] for (m, _, c), v in zip(metrics, values)]).T
        ax.bar(np.arange(3) + (i - 1.5) * width, values, width - .03, color=colour, label=method, yerr=errors,
               error_kw={"elinewidth": 1, "ecolor": MUTED, "capsize": 2})
    ax.set_xticks(np.arange(3), [label for _, label, _ in metrics]); ax.set_ylim(0, 1)
    ax.set_ylabel("Held-out test patients"); ax.legend(frameon=False, fontsize=7, ncols=2, loc="upper center", bbox_to_anchor=(.5, 1.22))
    ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "object_metrics.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    b_all = per_window[(per_window.method == "B (HE2CELL)")].copy()
    groups = [("tls", "TLS", "HookNet-TLS windows"), ("random", "TLS", "Random, HookNet TLS"),
              ("random", "none", "Random, no TLS"), ("dense", "none", "Dense hard negatives")]
    data = [b_all[(b_all.category == c) & (b_all.hooknet_label == l)].max_density.to_numpy() for c, l, _ in groups]
    ax.boxplot([np.log10(d + 10) for d in data], widths=.5, showfliers=False, medianprops={"color": INK})
    rng = np.random.default_rng(0)
    for i, d in enumerate(data, 1):
        ax.scatter(i + rng.uniform(-.15, .15, len(d)), np.log10(d + 10), s=9, color="#2a78d6", alpha=.55, linewidths=0)
    ax.set_xticks(range(1, 5), [f"{g}\n(n={len(d)})" for (_, _, g), d in zip(groups, data)], fontsize=7.5)
    ax.set_ylabel("Peak predicted-B density\nlog10(cells/mm² + 10)")
    ax.axhline(np.log10(b_params["threshold"] + 10), color=MUTED, ls="--", lw=.8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "b_density_by_category.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    panels = [(per_case, "hooknet_TLS_per_100mm2", "log2TPM_CXCL13", "HookNet TLS per 100 mm²", "CXCL13 log2(TPM+1)", "HookNet TLS/100mm2 vs log2TPM_CXCL13"),
              (model_case, "B_per_mm2", "log2TPM_MS4A1", "B cells/mm², random windows", "MS4A1 log2(TPM+1)", "HE2CELL random-window B_per_mm2 vs log2TPM_MS4A1"),
              (case_max, "max_B_density_any_window", "chemokine12_PC1", "Peak B density, cells/mm²", "12-chemokine score (PC1)", "HE2CELL max B density (all windows) vs chemokine12_PC1")]
    for ax, (frame, xcol, ycol, xl, yl, key) in zip(axes, panels):
        xv = frame[xcol].to_numpy(float)
        ax.scatter(np.log10(xv + (1 if "per" in xcol else 10)), frame[ycol], s=12, color="#2a78d6", alpha=.7, linewidths=0)
        r = rna_checks[key]
        ax.set(xlabel=f"{xl}, log10 scale", ylabel=yl, title=f"Spearman ρ={r['rho']:.2f}, p={r['p']:.1g}, n={r['n']}")
        ax.title.set_fontsize(8.5)
    fig.tight_layout(); fig.savefig(FIG / "patient_level_rna.png"); plt.close(fig)

    # ---------- Review gallery (held-out test windows), organised by two-stage outcome ----------
    ts_params = chosen[A.TWO_STAGE]
    stage1 = {k: ts_params[k] for k in ["sigma_um", "threshold", "min_area_um2", "min_cells"]}
    proposals = pd.read_csv(analysis / "two_stage_proposals.csv")
    proposals = proposals[proposals.split == "test"]
    unproposed = pd.read_csv(analysis / "two_stage_unproposed_HookNet_TLS.csv")
    unproposed = unproposed[(unproposed.split == "test") & (unproposed.HookNet_TLS_not_proposed > 0)]
    order = ["aggregate found, composition rejected; HookNet calls TLS", "TLS, HE2CELL only", "TLS, HookNet agrees",
             "aggregate found, composition rejected; HookNet agrees not TLS"]
    titles = {order[0]: "Aggregate found, composition rejected — HookNet calls TLS (pathologist review priority)",
              order[1]: "Accepted as TLS by HE2CELL only (HookNet negative)",
              order[2]: "Accepted as TLS, HookNet agrees",
              order[3]: "Aggregate found, composition rejected — HookNet agrees not TLS"}
    gallery_html = []
    for outcome in order + ["not proposed"]:
        if outcome == "not proposed":
            frame = unproposed.drop_duplicates(["barcode", "window_id"])
            frame = frame.sample(min(6, len(frame)), random_state=0)
            title = "HookNet TLS with no stage-1 aggregate (nuclear density too low)"
        else:
            # Seeded random sample so the gallery is representative, not just the largest aggregates.
            frame = proposals[proposals.outcome == outcome].drop_duplicates(["barcode", "window_id"])
            frame = frame.sample(min(6, len(frame)), random_state=0)
            title = titles[outcome]
        tiles = []
        for r in frame.itertuples():
            x = inputs[(r.barcode, r.window_id)]
            props, _ = A.proposal_features(x, **stage1)
            kept = np.zeros((x["n"], x["n"]), bool); dropped = kept.copy()
            for prop in props:
                if A.accept(prop, ts_params["b_share"], ts_params.get("require_b_core", False))[0]:
                    kept |= prop["mask"]
                else:
                    dropped |= prop["mask"]
            caption = (f"{html.escape(r.barcode[:12])} · {r.window_id}" if outcome == "not proposed" else
                       f"{html.escape(r.barcode[:12])} · {r.window_id} · B {r.B_share:.0%} · core−periphery B {r.B_core_enrichment:+.0%}"
                       + (f" · {html.escape(r.rejection_reason)}" if isinstance(r.rejection_reason, str) and r.rejection_reason else ""))
            tiles.append(f'<figure><img src="{jpg_data(gallery_tile(r.barcode, r.window_id, x, kept, dropped))}" alt="{r.barcode} {r.window_id}">'
                         f'<figcaption>{caption}</figcaption></figure>')
        gallery_html.append(f"<h3>{html.escape(title)}</h3><div class='gallery'>{''.join(tiles) or '<p class=muted>None in the test set.</p>'}</div>")

    # ---------- Head-to-head with the original classifier (Reinhard arm) on shared test windows ----------
    head_to_head = {}
    original = OUT / "analysis" / "per_window_results.csv"
    if A.RESULT != "he2cell" and original.exists():
        old = pd.read_csv(original)
        for method in ["B (HE2CELL)", "B+T (HE2CELL)", A.TWO_STAGE]:
            o = old[(old.method == method) & (old.split == "test")]
            if o.empty:
                continue
            n = per_window[(per_window.method == method) & (per_window.split == "test")]
            keys = set(zip(o.barcode, o.window_id)) & set(zip(n.barcode, n.window_id))
            entry = {}
            for name, frame in [("original + Reinhard", o), ("fine-tuned, raw", n)]:
                f = frame[[k in keys for k in zip(frame.barcode, frame.window_id)]]
                lab = f[f.hooknet_label != "ambiguous"]
                pr, rc, f1 = A.prf(f)
                entry[name] = {"windows": len(f), "patients": int(f.case_id.nunique()), "precision": pr, "recall": rc, "F1": f1,
                               "AUROC": A.auc(lab.hooknet_label == "TLS", lab.max_density)}
            head_to_head[method] = entry
    finetune = {}
    history_path = ROOT / "outputs" / "classifier_finetune_20261019" / "history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
        trained = [h for h in history if h["epoch"] > 0]
        finetune = {"history": [{k: v for k, v in h.items() if not isinstance(v, dict)} | {"holdout_recall": h["holdout_recall"], "holdout_precision": h["holdout_precision"]} for h in history],
                    "chosen_epoch": max(trained, key=lambda h: h["selection_score"])["epoch"] if trained else None,
                    "settings": json.loads((history_path.parent / "settings.json").read_text()) if (history_path.parent / "settings.json").exists() else {}}
    summary = {"flow": flow, "rna_checks": rna_checks, "organisation": org_stats, "chosen_parameters": chosen,
               "head_to_head": head_to_head, "finetune": finetune, "classifier_run": A.RESULT,
               "detection": results}
    save_json(summary, analysis / "report_summary.json")
    write_html(summary, slides, windows, gallery_html)


def fmt(value, digits=2):
    return "–" if value is None or (isinstance(value, float) and not np.isfinite(value)) else f"{value:.{digits}f}"


def write_html(summary, slides, windows, gallery_html):
    det = summary["detection"]["results"]
    b = det["B (HE2CELL)"]["test"]
    t2 = det[A.TWO_STAGE]["test"]
    breakdown = summary["detection"].get("two_stage_breakdown", {})
    rows = []
    for method, entry in det.items():
        t = entry["test"]
        rows.append(f"<tr><td>{html.escape(method)}</td><td>{fmt(t['object_precision'])} [{fmt(t['CI95_object_precision'][0])}–{fmt(t['CI95_object_precision'][1])}]</td>"
                    f"<td>{fmt(t['object_recall'])} [{fmt(t['CI95_object_recall'][0])}–{fmt(t['CI95_object_recall'][1])}]</td>"
                    f"<td>{fmt(t['object_F1'])} [{fmt(t['CI95_object_F1'][0])}–{fmt(t['CI95_object_F1'][1])}]</td>"
                    f"<td>{fmt(t['precision_lenient_any_HookNet_TLS_pixel'])}</td>"
                    f"<td>{fmt(t['window_AUROC_max_density'])} [{fmt(t['CI95_window_AUROC'][0])}–{fmt(t['CI95_window_AUROC'][1])}]</td>"
                    f"<td>{fmt(t['window_AUROC_vs_dense_hard_negatives'])} [{fmt(t['CI95_AUROC_vs_dense'][0])}–{fmt(t['CI95_AUROC_vs_dense'][1])}]</td>"
                    f"<td>{fmt(t['median_dice_in_HookNet_TLS_windows'])}</td></tr>")
    rna_rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{fmt(v['rho'])}</td><td>{v['p']:.2g}</td><td>{v['n']}</td></tr>"
                       for k, v in summary["rna_checks"].items() if np.isfinite(v["rho"]))
    org = summary["organisation"]
    p = summary["chosen_parameters"]["B (HE2CELL)"]
    ts = summary["chosen_parameters"].get(A.TWO_STAGE)
    tse = det.get(A.TWO_STAGE, {}).get("test", {})
    two_stage_text = ""
    if ts:
        rule = (f"predicted-B share ≥ {ts['b_share']:.0%}" + (" and more predicted B in the core than the periphery"
                if ts.get("require_b_core") else "; the B-core requirement was tested on dev patients and not selected"))
        outcomes = breakdown.get("outcomes", {})
        orow = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in outcomes.items())
        orow += f"<tr><td>HookNet TLS with no stage-1 aggregate</td><td>{breakdown.get('HookNet_TLS_not_proposed_by_stage1', 0)}</td></tr>"
        reasons = ", ".join(f"{k}: {v}" for k, v in breakdown.get("rejection_reasons", {}).items()) or "none"
        comparison = breakdown.get("HookNet_TLS_accepted_vs_rejected_median", {})
        crow = "".join(f"<tr><td>{html.escape(f)}</td><td>{fmt(v[0])}</td><td>{fmt(v[1])}</td></tr>" for f, v in comparison.items())
        mat = breakdown.get("maturity_proxy", {})
        mrow = "".join(f"<tr><td>{html.escape(f)}</td><td>{fmt(v['median_with_GC'])}</td><td>{fmt(v['median_without_GC'])}</td><td>{fmt(v['AUROC_for_GC'])}</td><td>{v['mann_whitney_p']:.2g}</td></tr>"
                       for f, v in mat.items() if isinstance(v, dict))
        two_stage_text = f"""<h3>Two-stage detector</h3>
<p><b>Stage 1</b> proposes aggregates from all Cellpose nuclei, with no classifier (σ = {ts['sigma_um']} µm, ≥ {ts['threshold']:,} nuclei/mm², area ≥ {ts['min_area_um2']:,} µm², ≥ {ts['min_cells']} cells). <b>Stage 2</b> accepts a proposal as TLS only if the classifier's composition supports it: {rule}. Core is the inner half of each aggregate by distance to its edge; periphery is the outer half plus a {int(A.RIM_UM)} µm ring. Window score for AUROC: highest B share among proposals.</p>
<table><tr><th>Outcome of every stage-1 aggregate (test patients)</th><th>Aggregates</th></tr>{orow}</table>
<p class="muted">Rejection reasons: {html.escape(reasons)}. Each category is listed per aggregate in <code>two_stage_proposals.csv</code>; "composition rejected — HookNet calls TLS" is the main pathologist review set.</p>
{f'<table><tr><th>HookNet-matched aggregates: median feature</th><th>accepted</th><th>rejected</th></tr>{crow}</table>' if crow else ''}
<h3>Toward maturity: germinal-centre proxy</h3>
<p>{f"Of {mat.get('accepted_aggregates', 0)} accepted test aggregates, {mat.get('with_HookNet_GC_pixels', 0)} contain HookNet germinal-centre-class pixels (≥ 1% of area). The HE2CELL composition features below are compared between aggregates with and without them; AUROC is how well each feature alone ranks GC-containing aggregates first. HookNet's GC class is unfiltered model output, so this is a feasibility check, not a maturity classifier." if mat else ""}</p>
{f'<table><tr><th>Feature</th><th>median, GC</th><th>median, no GC</th><th>AUROC for GC</th><th>p</th></tr>{mrow}</table>' if mrow else '<p class="muted">Too few aggregates with and without GC pixels for this comparison.</p>'}"""
    flow = summary["flow"]
    split_counts = windows.groupby("split").case_id.nunique().to_dict()
    diag = json.loads((OUT / "diagnostics" / "diagnostics.json").read_text())
    def cnt(d, k):
        return d.get(k, 0)
    drows = []
    for key, value in diag.items():
        if key.startswith("LUAD A"):
            drows.append(f"<tr><td>Lung {key[5:]} (original pipeline → window path)</td><td>{cnt(value['original_pipeline'],'B')} → {cnt(value['window_path'],'B')}</td><td>{cnt(value['original_pipeline'],'T')} → {cnt(value['window_path'],'T')}</td><td>{cnt(value['original_pipeline'],'Other')} → {cnt(value['window_path'],'Other')}</td></tr>")
        elif "raw" in value:
            drows.append(f"<tr><td>{html.escape(key[:12])} {key.split()[-1]} (raw → normalised)</td><td>{cnt(value['raw'],'B')} → {cnt(value['reinhard_to_LUAD'],'B')}</td><td>{cnt(value['raw'],'T')} → {cnt(value['reinhard_to_LUAD'],'T')}</td><td>{cnt(value['raw'],'Other')} → {cnt(value['reinhard_to_LUAD'],'Other')}</td></tr>")
    diagnostics_table = "<table><tr><th>Window (original classifier)</th><th>B</th><th>T</th><th>Other</th></tr>" + "".join(drows) + "</table>"
    finetune_html, head_html = "", ""
    ft = summary.get("finetune") or {}
    if ft:
        st = ft.get("settings", {})
        erows = "".join(
            f"<tr{' style=\"font-weight:600\"' if h['epoch'] == ft['chosen_epoch'] else ''}><td>{h['epoch']}{' (original)' if h['epoch'] == 0 else ''}{' ← chosen' if h['epoch'] == ft['chosen_epoch'] else ''}</td>"
            f"<td>{fmt(h['holdout_macro_F1'])}</td><td>{fmt(h['holdout_recall']['B'])} / {fmt(h['holdout_precision']['B'])}</td>"
            f"<td>{fmt(h['holdout_recall']['T'])} / {fmt(h['holdout_precision']['T'])}</td><td>{fmt(h.get('ESCA_dev_AUROC_PB_TLS_vs_dense'))}</td>"
            f"<td>{fmt(h.get('ESCA_dev_B_fraction_in_TLS'))}</td><td>{fmt(h.get('ESCA_dev_B_fraction_in_dense'))}</td>"
            f"<td>{h.get('ESCA_dev_slides_with_B_fraction_in_TLS_lt_0.02', '–')}/{h.get('ESCA_dev_slides', '–')}</td></tr>"
            for h in ft["history"])
        finetune_html = f"""<h2>Fine-tuning with stain, colour and scale augmentation</h2>
<p>Starting from <code>best_model2.pth</code>, all layers were fine-tuned on the original colorectal training crops (patients {', '.join(sorted(set(f'CRC{i:02d}' for i in range(1, 11)) - set(st.get('holdout', []))))}; {', '.join(st.get('holdout', []))} held out) for {st.get('epochs', '–')} × {st.get('samples_per_epoch', 0):,} class-balanced samples. Augmentation: LAB colour-style shifts across a broad H&amp;E range, HED stain scaling ±{int(100*st.get('hed_alpha', 0))}%, brightness/contrast/saturation/hue/gamma jitter, blur and noise, ×{st.get('scale_range', ['–','–'])[0]}–{st.get('scale_range', ['–','–'])[1]} rescaling (covers 0.325 → 0.25 µm/px), mask dilation/erosion (Cellpose masks are larger than the training masks), and D4 symmetries. The checkpoint was chosen by a rule fixed in advance: held-out macro-F1 + dev-ESCA AUROC of P(B) for cells inside HookNet TLS versus dense non-TLS windows. Only dev patients were used; test patients were never seen.</p>
<table><tr><th>Epoch</th><th>Held-out macro-F1</th><th>B recall / precision</th><th>T recall / precision</th><th>Dev-ESCA AUROC</th><th>B share in TLS</th><th>B share in dense</th><th>B-empty dev slides</th></tr>{erows}</table>
<p class="muted">Held-out metrics are on a class-balanced sample (≤ 6,000 per class) of the two held-out patients. Dev-ESCA figures use raw crops, identity view only.</p>"""
    hh = summary.get("head_to_head") or {}
    if hh:
        hrows = "".join(f"<tr><td>{html.escape(m)}</td><td>{html.escape(k)}</td><td>{v['windows']} / {v['patients']}</td><td>{fmt(v['precision'])}</td><td>{fmt(v['recall'])}</td><td>{fmt(v['F1'])}</td><td>{fmt(v['AUROC'])}</td></tr>"
                        for m, e in hh.items() for k, v in e.items())
        head_html = f"""<h3>Original vs fine-tuned classifier on the same test windows</h3>
<table><tr><th>Score</th><th>Classifier</th><th>Windows / patients</th><th>Precision</th><th>Recall</th><th>F1</th><th>Window AUROC</th></tr>{hrows}</table>
<p class="muted">Restricted to test windows scored by both runs. Each run's detector parameters were tuned on its own dev patients.</p>"""
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ESCA TLS Validation</title><style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--line:#e4e3df;--card:#fff;--accent:#2a78d6}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--line:#3a3936;--card:#232321;--accent:#3987e5}}}}
:root[data-theme="dark"]{{--bg:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--line:#3a3936;--card:#232321;--accent:#3987e5}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1080px;margin:0 auto;padding:28px 16px 64px}} h1{{font-size:26px;margin:0 0 4px}} h2{{margin-top:40px;border-top:1px solid var(--line);padding-top:20px}}
.muted{{color:var(--muted)}} .kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:20px 0}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}} .kpi b{{font-size:24px;display:block}}
table{{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow-x:auto}} th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;white-space:nowrap}}
img.fig{{max-width:100%;background:#fff;border-radius:8px;border:1px solid var(--line)}}
.flow{{display:flex;flex-wrap:wrap;gap:8px;align-items:stretch}} .flow div{{flex:1 1 160px;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px;font-size:13px}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px}} figure{{margin:0}} figure img{{width:100%;border-radius:6px}}
figcaption{{font-size:11px;color:var(--muted)}} .legend span{{display:inline-block;width:12px;height:12px;border-radius:2px;margin:0 4px -2px 12px}}
</style></head><body><main>
<h1>Can HE2CELL find tertiary lymphoid structures in TCGA-ESCA?</h1>
<p class="muted">Cell-level predictions (B / T / Macrophage / Other) from the user's ResNet50 classifier on Cellpose cpsam_v2 nuclei, aggregated into TLS calls and compared with an independent H&amp;E model (HookNet-TLS) and with bulk RNA from the same patients.</p>
<div class="kpis">
<div class="kpi"><b>{fmt(t2['object_F1'])}</b>Two-stage detector, object F1 vs HookNet-TLS<br><span class="muted">precision {fmt(t2['object_precision'])} · recall {fmt(t2['object_recall'])} · 95% CI {fmt(t2['CI95_object_F1'][0])}–{fmt(t2['CI95_object_F1'][1])}</span></div>
<div class="kpi"><b>{fmt(t2['window_AUROC_max_density'])}</b>Two-stage window AUROC<br><span class="muted">vs dense hard negatives {fmt(t2['window_AUROC_vs_dense_hard_negatives'])}</span></div>
<div class="kpi"><b>{fmt(b['object_F1'])}</b>B-density detector, object F1<br><span class="muted">95% CI {fmt(b['CI95_object_F1'][0])}–{fmt(b['CI95_object_F1'][1])} · AUROC {fmt(b['window_AUROC_max_density'])}</span></div>
<div class="kpi"><b>{flow['he2cell_complete']}</b>slides analysed<br><span class="muted">{len(windows)} windows · {windows.case_id.nunique()} patients</span></div>
</div>

<h2>Workflow</h2>
<div class="flow">
<div><b>1 · Inventory &amp; QC</b><br>{flow['svs_in_archive']} complete SVS ({flow['incomplete_downloads_in_archive']} partial downloads skipped). Otsu tissue mask, then removal of pen ink, black ink/folds and grey mounting medium.</div>
<div><b>2 · Independent reference</b><br>Published HookNet-TLS (DIAGNijmegen) on every slide, authored filters; TLS polygons on ink/folds discarded.</div>
<div><b>3 · Molecular context</b><br>TCGA-ESCA STAR TPM: 12-chemokine TLS score, CXCL13, MS4A1/CD19/CD79A, LAMP3, CD3E.</div>
<div><b>4 · HE2CELL windows</b><br>Per 40x slide: ≤3 HookNet-TLS, 3 random tissue, 2 densest non-TLS windows (0.76 mm), chosen before any HE2CELL output. {"Raw, un-normalised patches → Cellpose → <b>fine-tuned</b> classifier (D4 TTA)." if summary.get("classifier_run") == "he2cell_ft" else "Per-slide Reinhard normalisation to the LUAD slide → Cellpose → original classifier (D4 TTA)."}</div>
<div><b>5 · Two-stage TLS detector</b><br>Cellpose nuclear density proposes aggregates; HE2CELL's B share (and optionally B-core enrichment) accepts or rejects each. HookNet is never used to find or accept regions. Tuned on {split_counts.get('dev', 0)} dev patients, frozen, scored on {split_counts.get('test', 0)} test patients.</div>
<div><b>6 · Checks</b><br>Baselines without the classifier, B/T composition inside independently defined HookNet TLS, patient-level RNA correlation, review gallery.</div>
</div>

<h2>Why the classifier had to be fine-tuned</h2>
<p>With the original checkpoint, B calls in ESCA depended on colour rendering rather than morphology. On raw slides some TLS were almost B-empty. Reinhard normalisation to the lung slide rescued those slides but emptied others that worked raw, and Macenko normalisation made it worse. The same window code reproduces the original whole-slide lung predictions, so the pipeline is not the cause.</p>
{diagnostics_table}
<p class="muted">Cell counts are window cores. The original training recipe used a weak stain perturbation (HED σ = 0.012) on pale, restained colorectal crops at about 0.325 µm/px; TCGA slides are deeper pink-purple at 0.25 µm/px.</p>
{finetune_html}
<h2>Detection on held-out patients</h2>
{head_html}
<p>Frozen B-detector parameters: σ = {p['sigma_um']} µm, B density ≥ {p['threshold']} cells/mm², area ≥ {p['min_area_um2']:,} µm², ≥ {p['min_cells']} B cells. Each baseline was tuned on the same dev patients with the same grid size.</p>
<table><tr><th>Score</th><th>Precision</th><th>Recall</th><th>F1</th><th>Precision (any HookNet TLS pixel)</th><th>Window AUROC</th><th>AUROC vs dense</th><th>Median Dice</th></tr>{''.join(rows)}</table>
<p>{two_stage_text}</p>
<p class="muted">Object matches need overlap ≥ {int(A.MATCH_FRACTION*100)}% of the smaller region; HookNet regions &lt; {int(A.MIN_REF_UM2):,} µm² inside a window are edge slivers and are not counted. Windows with only a sliver are excluded from AUROC. Brackets are 95% patient-level bootstrap intervals.</p>
<img class="fig" src="{png(FIG/'roc.png')}" alt="ROC curves">
<img class="fig" src="{png(FIG/'object_metrics.png')}" alt="Object-level precision, recall, F1">
<img class="fig" src="{png(FIG/'b_density_by_category.png')}" alt="Peak B density by window category">

<h2>Cell composition inside HookNet-defined TLS</h2>
<p>{f"TLS regions here are defined by HookNet alone, so this test does not depend on the HE2CELL detector. Across {org['slides']} slides, the median predicted-B share is {fmt(org['median_B_in_TLS'])} inside HookNet TLS versus {fmt(org['median_B_random'])} in the same slide's random windows ({org['slides_B_TLS_gt_random']}/{org['slides']} slides higher inside; paired one-sided Wilcoxon p = {org['B_wilcoxon_p']:.2g}). For T cells the shares are {fmt(org['median_T_in_TLS'])} versus {fmt(org['median_T_random'])} (p = {org['T_wilcoxon_p']:.2g}). {org['slides_with_B_lt_2pct_in_TLS']} slide(s) have under 2% predicted B inside HookNet TLS." if org else "Too few slides with both HookNet TLS and random windows."}</p>

<h2>Patient-level agreement with RNA</h2>
<p>RNA comes from a different piece of the same tumour, so these are cohort-level consistency checks, not spatial confirmation.</p>
<table><tr><th>Comparison</th><th>Spearman ρ</th><th>p</th><th>n</th></tr>{rna_rows}</table>
<img class="fig" src="{png(FIG/'patient_level_rna.png')}" alt="Patient-level RNA scatter plots">

<h2>Review gallery (held-out test windows)</h2>
<p class="legend muted">HookNet-TLS contour<span style="background:#1baf7a"></span> HE2CELL accepted TLS<span style="background:#0b0b0b"></span> aggregate rejected by composition<span style="background:#ffffff;border:1px solid #888"></span> predicted B<span style="background:#eb6834"></span> predicted T<span style="background:#2a78d6"></span></p>
{''.join(gallery_html)}

<h2>Limitations</h2>
<ul>
<li>HookNet-TLS is a published model, not pathologist ground truth. It also errs, for example on pen ink (handled by QC) and on lymphoid tissue that is not TLS. Agreement measures concordance between two independent H&amp;E methods; HE2CELL-only detections in the gallery need expert review before being counted as errors.</li>
<li>Windows sample each slide; slide-level TLS counts from HE2CELL would need whole-slide runs (about 1–2 h per ESCA slide at 40x on this Mac).</li>
<li>Only 40x slides (≤ 0.30 µm/px) are analysed, because the classifier sees native-resolution 64-px crops. Translucent red marker over tissue cannot be separated from eosin and is not masked.</li>
<li>The fine-tuned classifier was selected partly on dev-ESCA slides using HookNet as the reference, so dev-set numbers are optimistic; only test-patient results should be quoted. Its in-domain evaluation uses two held-out colorectal patients, and no ESCA cell-level ground truth (for example CD20 IHC) exists.</li>
<li>Patches use a 768-px stride with a 128-px core margin (the validated pipeline uses 512). Cells and crops are produced the same way; only the segmentation context overlap differs.</li>
<li>Lymphoma and myeloma cases are excluded. Normal mucosal lymphoid follicles are not separated from TLS by either method.</li>
</ul>
<p class="muted">Cohort: {flow['hooknet_complete']} slides with HookNet, {flow['he2cell_complete']} with HE2CELL windows; exclusions {html.escape(json.dumps(flow['excluded']))}. Code: deployment/esca_tls/. Data: outputs/esca_tls_20261019/.</p>
</main></body></html>"""
    (ANALYSIS / "ESCA_TLS_validation_report.html").write_text(page)
    print("Report written", ANALYSIS / "ESCA_TLS_validation_report.html")


if __name__ == "__main__":
    main()
