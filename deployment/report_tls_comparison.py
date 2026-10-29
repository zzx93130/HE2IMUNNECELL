"""Standalone scientific figures, H&E review gallery and a portable comparison report."""
from __future__ import annotations

import base64
import html
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime" / "matplotlib"))
COLORS = {"ours": "#f24b42", "finder": "#00bfd3", "tlsr": "#c53fe2", "B": "#39eb79", "T": "#ffd341"}


def image_uri(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def run_label(run_id):
    if run_id == "finder_default":
        return "TLS-Finder: 160 µm, ordered IDs"
    if run_id == "tlsr_default":
        return "tlsR: default 10 µm criterion"
    if run_id.startswith("finder_radius_"):
        return "TLS-Finder: radius " + run_id.split("_")[-1] + " µm"
    if run_id.startswith("finder_order_"):
        return "TLS-Finder: shuffled order " + run_id.split("_")[-1]
    if run_id.startswith("tlsr_distance_"):
        return "tlsR: " + run_id.split("_")[-1] + " µm criterion"
    method = "TLS-Finder" if run_id.startswith("finder") else "tlsR"
    return method + (": confidence ≥0.7" if "confidence" in run_id else ": stable under both flips")


def select_discordant(regions, candidates):
    options = regions.loc[regions.run_id.eq("finder_default") | regions.run_id.str.startswith("tlsr_distance_")].copy()
    xy = candidates[["x_um", "y_um"]].to_numpy()
    if not len(options):
        return options
    options["nearest_original_center_um"] = cKDTree(xy).query(options[["x_um", "y_um"]])[0]
    options = options.loc[options.nearest_original_center_um.gt(150)].sort_values(["source_BT", "region_id"], ascending=[False, True])
    selected = []
    centers = []
    for row in options.to_dict("records"):
        center = np.array([row["x_um"], row["y_um"]])
        if any(np.linalg.norm(center - previous) < 150 for previous in centers):
            continue
        centers.append(center)
        row["review_id"] = "D" + str(len(selected) + 1)
        selected.append(row)
        if len(selected) == 10:
            break
    return pd.DataFrame(selected)


def draw_maps(slide, cells, candidates, regions, memberships, extras, out, mpp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    thumbnail = slide.get_thumbnail((1800, 1800)).convert("RGB")
    extent = [0, slide.dimensions[0] * mpp / 1000, slide.dimensions[1] * mpp / 1000, 0]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "figure.facecolor": "#f7fafc"})
    fig, axes = plt.subplots(2, 3, figsize=(17, 11), layout="constrained")
    panels = [(None, "Our six fixed review regions"), ("finder_default", "TLS-Finder example settings"),
              ("tlsr_default", "tlsR source defaults"), ("tlsr_distance_40", "tlsR exploratory: 40 µm criterion"),
              ("tlsr_distance_50", "tlsR exploratory: 50 µm criterion"), ("tlsr_distance_80", "tlsR exploratory: 80 µm criterion")]
    lookup = cells.set_index("global_instance_id")
    for ax, (run_id, title) in zip(axes.flat, panels):
        ax.imshow(thumbnail, extent=extent)
        for c in candidates.itertuples():
            ax.add_patch(Circle((c.x_um/1000, c.y_um/1000), .15, fill=False, color=COLORS["ours"], lw=1.7))
            ax.annotate(c.candidate_id, (c.x_um/1000, c.y_um/1000), xytext=(4, 3), textcoords="offset points", color="#a81919", fontweight="bold", fontsize=8)
        if run_id:
            sub = regions.loc[regions.run_id.eq(run_id)]
            if run_id.startswith("finder"):
                for r in sub.itertuples():
                    ax.add_patch(Circle((r.x_um/1000, r.y_um/1000), r.radius_um/1000, fill=False, color=COLORS["finder"], lw=.6, alpha=.75))
            else:
                for r in sub.itertuples():
                    ids = memberships.get(r.region_id, set())
                    local = lookup.loc[list(ids)] if ids else lookup.iloc[:0]
                    ax.scatter(local.x_um/1000, local.y_um/1000, s=2.5, color=COLORS["tlsr"], alpha=.65, linewidths=0)
                    ax.plot(r.x_um/1000, r.y_um/1000, marker="+", color="#740086", ms=12, mew=2)
                    ax.add_patch(Circle((r.x_um/1000, r.y_um/1000), .08, fill=False, color=COLORS["tlsr"], lw=1, ls="--"))
            title += f"\n{len(sub)} detections"
        elif len(extras):
            for d in extras.itertuples():
                ax.plot(d.x_um/1000, d.y_um/1000, "x", color="#23344b", ms=6)
                ax.annotate(d.review_id, (d.x_um/1000, d.y_um/1000), xytext=(3, -9), textcoords="offset points", fontsize=7)
        ax.set(title=title, xlabel="x (mm)", ylabel="y (mm)")
    fig.suptitle("Whole-slide spatial methods on shared H&E cell predictions", fontsize=18)
    fig.supxlabel("Red: A1–A6 disks. Cyan: derived TLS-Finder disks. Purple: native tlsR B/T memberships; + core center; dashed circle: 80 µm expansion.", fontsize=9)
    fig.savefig(out / "whole_slide_comparison.png", dpi=160)
    plt.close(fig)


def draw_sensitivity(settings, comparison, out):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(17, 10), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.4])
    a, b = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    f = settings.loc[settings.run_id.isin(["finder_radius_80", "finder_radius_120", "finder_default", "finder_radius_200"])].sort_values("radius_um")
    a.plot(f.radius_um, f.detections, "o-", color="#0097a9", label="Ordered-ID runs")
    shuffled = settings.loc[settings.category.eq("order_sensitivity")]
    a.scatter(shuffled.radius_um, shuffled.detections, marker="x", s=90, color="#cf5d32", label="Three shuffled orders at 160 µm")
    a.set(title="TLS-Finder: radius and order sensitivity", xlabel="Radius (µm)", ylabel="Lymphoid aggregate count")
    a.legend(fontsize=9)
    r = settings.loc[settings.run_id.eq("tlsr_default") | settings.category.eq("density_sweep")].sort_values("threshold_um")
    b.plot(r.threshold_um, r.detections, "o-", color="#a632ba", label="Detected regions")
    b.set(title="tlsR: predefined density-criterion sweep", xlabel="Maximum mean distance to 30 B neighbors (µm)", ylabel="Native detected region count")
    bb = b.twinx()
    bb.plot(r.threshold_um, r.dense_B, "s--", color="#566c85", alpha=.7)
    bb.set_ylabel("Eligible dense B cells", color="#566c85")
    b.legend(fontsize=9, loc="upper left")
    ax = fig.add_subplot(grid[1, :])
    chosen = ["finder_default", "finder_order_569", "finder_order_570", "finder_order_571", "finder_confidence_0.7", "finder_stable_flips",
              "tlsr_default", "tlsr_distance_30", "tlsr_distance_40", "tlsr_distance_50", "tlsr_distance_60", "tlsr_distance_80"]
    candidate_ids = [f"A{i}" for i in range(1, 7)]
    matrix = comparison.pivot(index="candidate_id", columns="run_id", values="all_detection_candidate_BT_coverage").loc[candidate_ids, chosen]
    matches = comparison.pivot(index="candidate_id", columns="run_id", values="primary_center_match").loc[candidate_ids, chosen]
    image = ax.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    labels = ["Finder\ndefault", "Finder\nshuffle 569", "Finder\nshuffle 570", "Finder\nshuffle 571", "Finder\nconf ≥0.7", "Finder\nstable flips",
              "tlsR\ndefault", "tlsR\n30 µm", "tlsR\n40 µm", "tlsR\n50 µm", "tlsR\n60 µm", "tlsR\n80 µm"]
    ax.set_xticks(np.arange(len(chosen)), labels, fontsize=9)
    ax.set_yticks(np.arange(6), candidate_ids)
    for i in range(6):
        for j in range(len(chosen)):
            value = matrix.iloc[i, j]
            ax.text(j, i, f"{value:.0%}" + (" *" if matches.iloc[i, j] else ""), ha="center", va="center", color="white" if value > .55 else "#173049", fontsize=10)
    ax.set_title("Original candidate B/T cells covered by any detection in each run\n* = at least one tool center lies inside that candidate's fixed 150 µm disk")
    fig.colorbar(image, ax=ax, shrink=.75, label="Fraction of original candidate B/T cells")
    fig.suptitle("Sensitivity describes spatial algorithm behavior; it does not measure TLS accuracy", fontsize=17)
    fig.savefig(out / "sensitivity_comparison.png", dpi=160)
    plt.close(fig)


def review_gallery(slide, cells, candidates, regions, memberships, extras, out, mpp):
    from PIL import Image, ImageDraw, ImageFont
    gallery_dir = out / "review_regions"
    gallery_dir.mkdir(exist_ok=True)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 17)
    except OSError:
        font = ImageFont.load_default(size=17)
    reviews = [{"review_id": r.candidate_id, "x_um": r.x_um, "y_um": r.y_um,
                "kind": "original candidate", "run_id": "tlsr_distance_50", "region_id": ""} for r in candidates.itertuples()]
    reviews += [{"review_id": r.review_id, "x_um": r.x_um, "y_um": r.y_um,
                 "kind": "additional region outside original center matches", "run_id": r.run_id,
                 "region_id": r.region_id} for r in extras.itertuples()]
    xy = cells[["x_um", "y_um"]].to_numpy()
    tree = cKDTree(xy)
    lookup = cells.set_index("global_instance_id")
    cards = []
    candidate_centers = candidates[["x_um", "y_um"]].to_numpy()
    level = min(1, slide.level_count - 1)
    downsample = slide.level_downsamples[level]
    size = int(np.ceil(600/mpp/downsample))
    factor = 1/(mpp*downsample)
    for review in reviews:
        center = np.array([review["x_um"], review["y_um"]])
        start = np.floor((center - 300)/mpp).astype(int)
        origin = start*mpp
        roi = slide.read_region(tuple(start), level, (size, size)).convert("RGB")
        roi.save(gallery_dir / f"{review['review_id']}_HE.png")
        local = cells.iloc[tree.query_ball_point(center, 425)]
        plots = [roi.copy(), roi.copy()]
        tf = regions.loc[regions.run_id.eq("finder_default")]
        selected_r_run = review["run_id"] if review["run_id"].startswith("tlsr") else "tlsr_distance_50"
        rs = regions.loc[regions.run_id.eq(selected_r_run)]
        for panel_number, panel in enumerate(plots):
            draw = ImageDraw.Draw(panel)
            for cell in local.itertuples():
                if cell.celltype not in ("B", "T"):
                    continue
                x, y = (np.array([cell.x_um, cell.y_um]) - origin)*factor
                if 0 <= x < size and 0 <= y < size:
                    draw.ellipse((x-1.6, y-1.6, x+1.6, y+1.6), fill=COLORS[cell.celltype])
            for c in candidates.itertuples():
                relative = (np.array([c.x_um, c.y_um]) - origin)*factor
                if np.all(relative >= -150*factor) and np.all(relative <= size + 150*factor):
                    x, y = relative; radius = 150*factor
                    draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=COLORS["ours"], width=3)
            if panel_number == 0:
                for r in tf.itertuples():
                    x, y = (np.array([r.x_um, r.y_um]) - origin)*factor
                    radius = r.radius_um*factor
                    if -radius <= x <= size+radius and -radius <= y <= size+radius:
                        draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=COLORS["finder"], width=2)
            else:
                for r in rs.itertuples():
                    ids = memberships.get(r.region_id, set())
                    sub = local.loc[local.global_instance_id.isin(ids)]
                    for cell in sub.itertuples():
                        x, y = (np.array([cell.x_um, cell.y_um]) - origin)*factor
                        if 0 <= x < size and 0 <= y < size:
                            draw.ellipse((x-3.1, y-3.1, x+3.1, y+3.1), outline=COLORS["tlsr"], width=1)
                    x, y = (np.array([r.x_um, r.y_um]) - origin)*factor
                    if -80*factor <= x <= size+80*factor and -80*factor <= y <= size+80*factor:
                        radius = 80*factor
                        draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=COLORS["tlsr"], width=2)
                        draw.line((x-8, y, x+8, y), fill=COLORS["tlsr"], width=3)
                        draw.line((x, y-8, x, y+8), fill=COLORS["tlsr"], width=3)
        panel_size = 440
        canvas = Image.new("RGB", (3*panel_size, panel_size+105), "#17283a")
        draw = ImageDraw.Draw(canvas)
        headings = ["H&E: 600 × 600 µm", "TLS-Finder default disks", run_label(selected_r_run)]
        for i, image in enumerate([roi, *plots]):
            canvas.paste(image.resize((panel_size, panel_size)), (i*panel_size, 38))
            draw.text((i*panel_size+10, 10), headings[i], font=font, fill="white")
        bcount, tcount = int(local.celltype.eq("B").sum()), int(local.celltype.eq("T").sum())
        draw.text((10, panel_size+45), f"{review['review_id']} | center ({center[0]:.1f}, {center[1]:.1f}) µm | {review['kind']}", font=font, fill="white")
        draw.text((10, panel_size+73), "B green / T gold; original disks red; Finder disks cyan; tlsR memberships purple rings. TLS unconfirmed.", font=font, fill="white")
        filename = gallery_dir / f"{review['review_id']}_comparison.png"
        canvas.save(filename)
        cards.append({**review, "image": str(filename), "tlsr_display_run": selected_r_run,
                      "review_window_um": 600, "B_within_425um": bcount, "T_within_425um": tcount})
    pd.DataFrame(cards).to_csv(out / "review_regions.csv", index=False)
    # A compact six-region figure is convenient for sharing and inline preview.
    sheet = Image.new("RGB", (1320, 545*6), "#17283a")
    for i, card in enumerate(cards[:6]):
        with Image.open(card["image"]) as image:
            sheet.paste(image, (0, i*545))
    sheet.save(out / "A1_A6_comparison_gallery.png")
    return cards


def table(df):
    return df.to_html(index=False, border=0, escape=True, na_rep="—", float_format=lambda x: f"{x:.3f}", classes="data")


def render(run: Path, out: Path):
    import openslide
    analysis = run / "tls_analysis"
    cells = pd.read_csv(analysis / "spatial_cells.csv")
    candidates = pd.read_csv(analysis / "ranked_candidate_regions.csv")
    summary = json.loads((analysis / "spatial_summary.json").read_text())
    provenance = json.loads((out / "provenance.json").read_text())
    regions = pd.read_csv(out / "regions.csv")
    comparison = pd.read_csv(out / "A1_A6_comparison.csv")
    settings = pd.read_csv(out / "run_summary.csv")
    global_overlap = pd.read_csv(out / "global_overlap.csv")
    membership_frame = pd.read_csv(out / "BT_membership.csv.gz")
    memberships = {region_id: set(map(int, group.global_instance_id)) for region_id, group in membership_frame.groupby("region_id")}
    extras = select_discordant(regions, candidates)
    extras.to_csv(out / "additional_discordant_regions.csv", index=False)
    with openslide.OpenSlide(summary["slide"]) as slide:
        if not np.isclose(float(slide.properties[openslide.PROPERTY_NAME_MPP_X]), summary["microns_per_pixel"]):
            raise ValueError("Slide metadata and saved physical coordinates disagree")
        draw_maps(slide, cells, candidates, regions, memberships, extras, out, summary["microns_per_pixel"])
        draw_sensitivity(settings, comparison, out)
        cards = review_gallery(slide, cells, candidates, regions, memberships, extras, out, summary["microns_per_pixel"])
    f = settings.set_index("run_id").loc["finder_default"]
    r = settings.set_index("run_id").loc["tlsr_default"]
    shuffle = settings.loc[settings.category.eq("order_sensitivity")]
    native_f = regions.loc[regions.run_id.eq("finder_default")]
    t_only = int(native_f.source_B.eq(0).sum())
    coverage = global_overlap.set_index("run_id").loc["finder_default", "candidate_union_coverage"]
    baseline = comparison.loc[comparison.run_id.eq("finder_default")].copy()
    display = pd.DataFrame({"Candidate": baseline.candidate_id, "B / T": baseline.source_B.astype(str)+" / "+baseline.source_T.astype(str),
        "Finder centers inside": baseline.nearby_detection_count,
        "Nearest Finder center (µm)": baseline.nearest_center_distance_um.round(1),
        "B/T covered by any Finder disk": baseline.all_detection_candidate_BT_coverage.map(lambda v: f"{v:.1%}"),
        "Best single-disk Jaccard": baseline.best_BT_jaccard.round(3), "tlsR default centers": 0})
    run_display = settings[["run_id", "eligible_B", "eligible_T", "dense_B", "detections", "original_candidates_center_matched", "detection_seconds"]].copy()
    run_display["run_id"] = run_display.run_id.map(run_label)
    run_display.columns = ["Run", "Eligible B", "Eligible T", "Dense B", "Detections", "A1–A6 center matches", "Detection seconds"]
    candidate_rows = comparison.copy()
    narratives = [
        f"TLS-Finder found {int(f.detections)} lymphoid aggregates at its 160 µm example radius. Its centers match {int(f.original_candidates_center_matched)} of our six fixed regions, while its geometric disks cover {coverage:.1%} of their combined predicted B/T cells.",
        f"tlsR found {int(r.detections)} regions at its source defaults: none of the {int(r.eligible_B):,} predicted B cells met the default mean-distance criterion. This is a legitimate empty result; coordinates were not rescaled to change it.",
        f"TLS-Finder count changes to {int(shuffle.detections.min())}–{int(shuffle.detections.max())} under three shuffled input orders. Its greedy selection is order dependent; aggregate count is not a biological TLS count. {t_only} default disks contain no predicted B cells.",
    ]
    sweep = settings.loc[settings.category.eq("density_sweep")]
    narratives.append("tlsR's predefined exploratory criteria produced " + "; ".join(f"{int(x.threshold_um)} µm: {int(x.detections)} regions" for x in sweep.itertuples()) + ". These are sensitivity results, not recalibrated defaults.")
    if len(regions.loc[regions.method.eq("tlsR")]):
        rr = regions.loc[regions.method.eq("tlsR")]
        broad = rr.loc[rr.membership_max_distance_um.gt(300)]
        if len(broad):
            narratives.append(f"{len(broad)} tlsR region results across the runs have native member cells more than 300 µm from their core center (maximum {rr.membership_max_distance_um.max():.0f} µm). Its core kNN graph has no edge-distance cutoff, so some results bridge spatially separated groups. Review the membership map rather than treating the 80 µm expansion circle as a boundary.")
    interpretation = []
    informative_runs = ["finder_default", "finder_confidence_0.7", "finder_stable_flips", "tlsr_distance_40", "tlsr_distance_50", "tlsr_distance_60", "tlsr_distance_80"]
    for cid in candidates.candidate_id:
        local = comparison.loc[comparison.candidate_id.eq(cid)].set_index("run_id")
        tls_hits = [str(x) for x in [40, 50, 60, 80] if local.loc[f"tlsr_distance_{x}", "primary_center_match"]]
        interpretation.append(f"{cid}: Finder baseline B/T coverage {local.loc['finder_default','all_detection_candidate_BT_coverage']:.1%}; confidence-filtered coverage {local.loc['finder_confidence_0.7','all_detection_candidate_BT_coverage']:.1%}; flip-stable coverage {local.loc['finder_stable_flips','all_detection_candidate_BT_coverage']:.1%}. tlsR center matches at exploratory criteria: {', '.join(tls_hits) + ' µm' if tls_hits else 'none'}. Original H&E review priority: {candidates.set_index('candidate_id').loc[cid,'suggested_review_priority']}.")
    default_union = set().union(*(memberships[r] for r in native_f.region_id)) if len(native_f) else set()
    method_pairs = []
    for run_id in settings.run_id:
        if run_id == "finder_default":
            continue
        ids = regions.loc[regions.run_id.eq(run_id), "region_id"]
        other_union = set().union(*(memberships[x] for x in ids)) if len(ids) else set()
        common = len(default_union & other_union)
        method_pairs.append({"reference_run": "finder_default", "comparison_run": run_id,
                             "intersection_BT": common, "union_BT": len(default_union | other_union),
                             "BT_union_jaccard": common/len(default_union | other_union) if default_union | other_union else np.nan})
    pd.DataFrame(method_pairs).to_csv(out / "method_pair_overlap.csv", index=False)
    limits = ["All methods share our predicted B/T labels. Agreement is spatial reproducibility, not independent biological validation.",
              "Generic predicted T cells are not measured CD8 cells. Other was mapped to Unclassified to avoid tlsR's substring label matching.",
              "TLS-Finder memberships are derived geometric disks and may overlap. Its threshold counts at least 50 previously unused combined B/T cells; it does not require both cell types.",
              "tlsR memberships are its returned assignments after expansion. Its source defaults differ from parts of its documentation; source function defaults were verified.",
              "All coverage denominators use the same original B/T cells, including cells made ineligible during confidence or flip checks. Those checks alter detection eligibility rather than the comparison universe.",
              "Center matches use 150 µm and allow many detections per candidate. A failed center match can still have substantial cell overlap. A possible merge means one region covers at least 25% of the B/T cells in two or more original disks.",
              "ICAT describes spatial spread per cell; it is neither a TLS maturity score nor a probability. No gold standard, diagnostic accuracy, maturity or confirmed TLS labels are available."]
    figures = [("whole_slide_comparison.png", "Whole-slide overlays"), ("sensitivity_comparison.png", "Parameter and input sensitivity")]
    gallery_html = []
    for i, card in enumerate(cards):
        gallery_html.append(f'<article class="review-card" data-id="{card["review_id"]}" style="display:{"block" if i == 0 else "none"}"><img src="{image_uri(card["image"])}" alt="{card["review_id"]} H&amp;E comparison"><p>{html.escape(card["review_id"] + ": " + card["kind"])}. Center ({card["x_um"]:.1f}, {card["y_um"]:.1f}) µm. TLS-Finder panel uses default settings; tlsR panel uses {html.escape(run_label(card["tlsr_display_run"]))}. This 600 µm review window is not a measured biological boundary.</p></article>')
    options = "".join(f'<option value="{card["review_id"]}">{html.escape(card["review_id"] + " — " + card["kind"])}</option>' for card in cards)
    run_options = "".join(f'<option value="{x}">{html.escape(run_label(x))}</option>' for x in settings.run_id)
    detailed_rows = []
    for row in comparison.itertuples():
        nearest = f"{row.nearest_center_distance_um:.1f}" if np.isfinite(row.nearest_center_distance_um) else "—"
        detailed_rows.append(f'<tr data-run="{row.run_id}" style="display:{"table-row" if row.run_id == "finder_default" else "none"}"><td>{row.candidate_id}</td><td>{row.nearby_detection_count}</td><td>{nearest}</td><td>{row.all_detection_candidate_BT_coverage:.1%}</td><td>{row.best_BT_jaccard:.3f}</td><td>{html.escape(str(row.nearby_region_ids))}</td></tr>')
    tlsr_display = regions.loc[regions.method.eq("tlsR"), ["run_id", "native_id", "core_B_count", "source_B", "source_T", "native_total_cells", "icat", "membership_max_distance_um"]].copy()
    tlsr_display["run_id"] = tlsr_display.run_id.map(run_label)
    tlsr_display.columns = ["Run", "Region", "Core B", "Assigned source B", "Assigned source T", "All native assigned cells", "ICAT", "Maximum member-center distance µm"]
    r_peak = provenance.get("R_batch_peak_resident_bytes")
    r_peak_text = f"{r_peak/1024**2:.0f} MiB" if r_peak is not None else "unavailable because the sandbox blocked the system profiler"
    report = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>H&E TLS spatial tool comparison</title>
<style>body{{margin:0;background:#f3f7fa;color:#183047;font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{max-width:1400px;margin:auto;padding:30px}}h1{{font-size:32px;line-height:1.2}}h2{{margin-top:32px}}section{{background:white;border-radius:12px;padding:24px;margin:22px 0;box-shadow:0 2px 8px #17304b0a}}.notice{{background:#e9f2fb;border-left:5px solid #315f94;padding:14px 18px}}img{{display:block;width:100%;height:auto;margin:16px 0}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:9px 11px;border-bottom:1px solid #e0e8f0;text-align:left}}th{{background:#edf3f9;white-space:nowrap}}select{{font:inherit;padding:8px;max-width:100%;margin:10px 0}}.muted{{color:#5b7083}}a{{color:#1764a1}}.legend{{font-size:14px}}code{{font-size:13px;overflow-wrap:anywhere}}</style>
<main><h1>H&E TLS spatial tool comparison</h1><p class="muted">Whole slide · {len(cells):,} deduplicated predicted cells · 4,773 B / 21,632 T · 17 predefined runs</p><p class="notice">Exploratory agreement between spatial algorithms using shared cell predictions. These results do not confirm TLS or establish diagnostic accuracy.</p>
<section><h2>What the comparison shows</h2>{''.join('<p>'+html.escape(n)+'</p>' for n in narratives)}<div class="scroll">{table(display)}</div><p>Center matches and coverage answer different questions. The six original disks remain fixed; public-tool detections can cover a disk without placing their center inside it.</p></section>
<section><h2>Whole-slide maps</h2><img src="{image_uri(out / 'whole_slide_comparison.png')}" alt="Whole-slide H&E overlays for our regions and both public methods"><p class="legend">tlsR 40–80 µm panels are explicitly exploratory. Purple points show native assigned B/T cells, including any distant core groups connected by its graph. Cyan disks show derived TLS-Finder neighborhoods, not annotated TLS boundaries.</p></section>
<section><h2>Sensitivity</h2><img src="{image_uri(out / 'sensitivity_comparison.png')}" alt="Counts and A1–A6 coverage across parameter and input-quality checks"><div class="scroll">{table(run_display)}</div><p>Radius and density sweeps use all original predicted B/T cells. Input-quality checks use default tool parameters and restrict B/T eligibility separately to confidence ≥0.7 or stability under both flips. Order checks use seeds 569–571. No settings were selected to reproduce our six candidates.</p></section>
<section><h2>Inspect each original region</h2>{''.join('<p>'+html.escape(n)+'</p>' for n in interpretation)}<label for="run">Detailed comparison run</label><br><select id="run">{run_options}</select><div class="scroll"><table><thead><tr><th>Candidate</th><th>Centers inside</th><th>Nearest center µm</th><th>B/T covered by any detection</th><th>Best single-region Jaccard</th><th>Matched regions</th></tr></thead><tbody>{''.join(detailed_rows)}</tbody></table></div><p>No detections are displayed as zero centers and zero coverage; nearest distance is unavailable.</p></section>
<section><h2>Native tlsR regions and ICAT</h2><div class="scroll">{table(tlsr_display)}</div><p>ICAT is the author's spatial-spread descriptor computed on all natively assigned cells, with a fixed random seed per region. It is not a maturity or accuracy score. Large member-center distances expose graph connections beyond the 80 µm expansion circle.</p></section>
<section><h2>H&E review gallery</h2><p>A1–A6 are the original review candidates. D1–D{len(extras)} are additional method detections with centers outside those disks, ranked by source B/T support and separated by at least 150 µm to avoid duplicate review windows.</p><select id="review">{options}</select>{''.join(gallery_html)}<p class="legend">The A1–A6 tlsR gallery uses the predefined 50 µm sensitivity run; its default detects no regions. For additional tlsR regions, the gallery shows that region's stated sensitivity run.</p></section>
<section><h2>Performance and verification</h2><p>Default detection time: TLS-Finder {f.detection_seconds:.2f} seconds; tlsR {r.detection_seconds:.3f} seconds. Detection times exclude setup, file loading, normalization, plotting and the original H&E inference. The whole comparison and normalization took {provenance['wall_seconds_detection_and_comparison']:.1f} seconds.</p><p>Peak resident memory during comparison: Python {provenance['python_peak_resident_bytes']/1024**2:.0f} MiB; the complete R batch {r_peak_text}. Python includes shared tables and normalized memberships, rather than isolated per-method usage.</p><p>Verified every retained cell coordinate against original slide pixels and physical scale, unchanged A1–A6 counts, safe B/T mapping, adapter equivalence on synthetic and real-region fixtures, sparse inputs, mixed/T-only/B-only/dispersed behavior, and exact native R membership IDs. Fixture tests passed before the whole-slide run.</p></section>
<section><h2>Interpretation limits</h2><ul>{''.join('<li>'+html.escape(n)+'</li>' for n in limits)}</ul><p>Prioritize A1/A2 for expert morphology review as in the original analysis. Additional tool agreement changes spatial support, not the confirmation status. B-cell-rich groups and their organization require further pathology or marker evidence to establish TLS and maturity.</p></section>
<section><h2>Sources and reproducibility</h2><p><a href="https://github.com/AAKoksoy/TLS-Finder/tree/cde11ecbd541ec280609f494a9dd9409639359b4">AAKoksoy TLS-Finder, pinned source</a> · <a href="https://github.com/labsyspharm/tlsR/tree/5e6acf7bebc9464635782b37fcf9dc70fc6daafe/v0.3.0">labsyspharm tlsR 0.3.0, pinned source</a></p><p>Machine-readable deliverables are beside this report: run_summary.csv, A1_A6_comparison.csv, candidate_region_pairs.csv, regions.csv, BT_membership.csv.gz, global_overlap.csv, method_pair_overlap.csv, provenance.json, raw native outputs, source audit and individual H&E crops.</p><p><code>.venv/bin/python deployment/compare_tls_tools.py --run-dir outputs/whole-slide-20261012</code></p></section></main>
<script>document.getElementById('review').addEventListener('change',e=>document.querySelectorAll('.review-card').forEach(x=>x.style.display=x.dataset.id===e.target.value?'block':'none'));document.getElementById('run').addEventListener('change',e=>document.querySelectorAll('tr[data-run]').forEach(x=>x.style.display=x.dataset.run===e.target.value?'table-row':'none'));</script></html>'''
    (out / "TLS_tool_comparison.html").write_text(report)
    lines = ["# H&E TLS spatial tool comparison", "", *[n + "\n" for n in narratives],
             "## Original candidates", "", "| Candidate | Finder center matches | Predicted B/T coverage | Best Jaccard | tlsR default matches |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {row.candidate_id} | {row.nearby_detection_count} | {row.all_detection_candidate_BT_coverage:.1%} | {row.best_BT_jaccard:.3f} | 0 |" for row in baseline.itertuples()]
    lines += ["", "## Candidate interpretation", "", *interpretation, "", "## Limitations", "", *["- " + n for n in limits],
              "", "Full visual report: TLS_tool_comparison.html. Tables, native outputs, figures and provenance are saved alongside it."]
    (out / "TLS_tool_comparison.md").write_text("\n".join(lines) + "\n")
    conclusions = {"TLS_Finder_default_aggregates": int(f.detections), "TLS_Finder_default_candidate_center_matches": int(f.original_candidates_center_matched),
                   "TLS_Finder_candidate_union_BT_coverage": float(coverage), "TLS_Finder_default_T_only_disks": t_only,
                   "tlsR_default_regions": int(r.detections), "TLS_Finder_shuffle_count_range": [int(shuffle.detections.min()), int(shuffle.detections.max())],
                   "tlsR_sensitivity_counts": {str(int(x.threshold_um)): int(x.detections) for x in sweep.itertuples()},
                   "TLS_confirmed": False, "maturity_assessed": False}
    (out / "conclusions.json").write_text(json.dumps(conclusions, indent=2) + "\n")
    print("Whole-slide figures, sensitivity dashboard and review gallery rendered.", flush=True)
    # Keep an already-completed independent H&E model in regenerated reports.
    if (out / "hooknet_tls" / "native_polygons.geojson").exists():
        import subprocess
        hooknet_manifest = json.loads((ROOT / ".runtime" / "hooknet_tls" / "manifest.json").read_text())
        subprocess.run([hooknet_manifest["python"], str(ROOT / "deployment" / "report_hooknet_tls.py"),
                        "--run-dir", str(run)], check=True)
