"""Compare pinned TLS-Finder and tlsR against fixed whole-slide H&E candidates.

Run with the project Python environment. No segmentation or cell classification
is repeated. Both methods receive coordinates in micrometers and shared labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from tls_tools import ROOT, RUNTIME, finder_input, load_finder, r_input


REGION_COLUMNS = ["method", "run_id", "region_id", "native_id", "x_um", "y_um", "radius_um",
                  "source_B", "source_T", "eligible_B", "eligible_T", "native_total_cells",
                  "native_unused_BT_count", "core_B_count", "icat", "membership_max_distance_um",
                  "membership_basis", "xmin_um", "xmax_um", "ymin_um", "ymax_um"]


def file_hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save_json(value, path: Path):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate_inputs(cells, candidates, summary, run):
    required = {"global_instance_id", "celltype", "confidence", "x_um", "y_um", "stable_both_flips"}
    if not required.issubset(cells.columns) or cells.global_instance_id.duplicated().any():
        raise ValueError("Invalid spatial cell table or duplicate global IDs")
    if not np.isfinite(cells[["x_um", "y_um", "confidence"]].to_numpy()).all():
        raise ValueError("Invalid cell coordinates or confidence")
    if not pd.api.types.is_bool_dtype(cells.stable_both_flips):
        raise ValueError("stable_both_flips must be boolean")
    if len(cells) != summary["analysis_cells"] or cells.celltype.value_counts().to_dict() != summary["analysis_counts"]:
        raise ValueError("Spatial table differs from the frozen analysis")
    if list(candidates.candidate_id) != [f"A{i}" for i in range(1, 7)]:
        raise ValueError("Expected the original A1–A6 candidates")
    manifest = json.loads((run / "run.json").read_text())
    source = pd.read_csv(manifest["slides"][0]["results"], usecols=["global_instance_id", "centroid_x_global", "centroid_y_global"])
    source = source.set_index("global_instance_id").loc[cells.global_instance_id]
    expected = source[["centroid_x_global", "centroid_y_global"]].to_numpy() * summary["microns_per_pixel"]
    np.testing.assert_allclose(cells[["x_um", "y_um"]], expected, atol=1e-8, rtol=0)
    tree = cKDTree(cells[["x_um", "y_um"]])
    for candidate in candidates.itertuples():
        local = cells.iloc[tree.query_ball_point([candidate.x_um, candidate.y_um], 150)]
        assert int(local.celltype.eq("B").sum()) == candidate.B_150um
        assert int(local.celltype.eq("T").sum()) == candidate.T_150um
    return {"canonical_cells": len(cells), "counts": cells.celltype.value_counts().to_dict(),
            "coordinate_units": "micrometers", "coordinate_check": "All retained IDs equal original level-0 centroids × microns_per_pixel",
            "candidate_count_check": "A1–A6 B/T disk counts unchanged"}


def adapter_checks(cells, candidates, adapted, reference):
    checks = []
    for number, row in candidates.head(2).iterrows():
        xy = cells[["x_um", "y_um"]].to_numpy()
        closest = np.argsort(np.linalg.norm(xy - [row.x_um, row.y_um], axis=1))[:180]
        small = cells.iloc[closest].copy()
        for seed in [None, 569]:
            generic = finder_input(small, seed=seed)
            marker_alias = generic.copy()
            marker_alias["Phenotype"] = marker_alias.Phenotype.replace({"B": "CD20", "T": "CD8"})
            pd.testing.assert_frame_equal(adapted(generic, 160), reference(marker_alias, 160))
            checks.append({"fixture": row.candidate_id + "_180_nearest_cells", "seed": seed, "identical_to_upstream": True})
    return checks


def normalize_region(method, run_id, native_id, center, indices, cells, eligible, **extra):
    sub = cells.iloc[indices]
    xy = sub[["x_um", "y_um"]].to_numpy()
    counts = sub.celltype.value_counts()
    record = {"method": method, "run_id": run_id, "region_id": f"{run_id}:{native_id}", "native_id": int(native_id),
              "x_um": float(center[0]), "y_um": float(center[1]),
              "source_B": int(counts.get("B", 0)), "source_T": int(counts.get("T", 0)),
              "eligible_B": int(np.count_nonzero((sub.celltype.to_numpy() == "B") & eligible[indices])),
              "eligible_T": int(np.count_nonzero((sub.celltype.to_numpy() == "T") & eligible[indices])),
              "xmin_um": float(xy[:, 0].min()) if len(xy) else np.nan,
              "xmax_um": float(xy[:, 0].max()) if len(xy) else np.nan,
              "ymin_um": float(xy[:, 1].min()) if len(xy) else np.nan,
              "ymax_um": float(xy[:, 1].max()) if len(xy) else np.nan, **extra}
    members = set(map(int, sub.loc[sub.celltype.isin(["B", "T"]), "global_instance_id"]))
    return record, members


def run_finder(cells, out, adapted):
    specs = [("finder_default", "defaults", 160, "all", None)]
    specs += [(f"finder_radius_{r}", "radius_sweep", r, "all", None) for r in [80, 120, 200]]
    specs += [(f"finder_order_{s}", "order_sensitivity", 160, "all", s) for s in [569, 570, 571]]
    specs += [(f"finder_{v}", "input_quality", 160, v, None) for v in ["confidence_0.7", "stable_flips"]]
    native_dir = out / "native" / "TLS-Finder"
    native_dir.mkdir(parents=True, exist_ok=True)
    xy = cells[["x_um", "y_um"]].to_numpy()
    tree = cKDTree(xy)
    id_to_index = pd.Series(np.arange(len(cells)), index=cells.global_instance_id)
    regions, memberships, settings = [], {}, []
    for run_id, category, radius, variant, seed in specs:
        df = finder_input(cells, variant, seed)
        start = time.perf_counter()
        result = adapted(df, radius)
        seconds = time.perf_counter() - start
        result.to_csv(native_dir / f"{run_id}.csv", index=False)
        eligible_ids = set(map(int, df.loc[df.Phenotype.isin(["B", "T"]), "global_instance_id"]))
        eligible = cells.global_instance_id.isin(eligible_ids).to_numpy()
        # Replay accepted disks to audit the native >=50 unused B/T criterion.
        input_tree = cKDTree(df[["X", "Y"]])
        used = set()
        for ordinal, record in enumerate(result.itertuples(index=False), 1):
            center = record.Aggregate_Center_Coordinates
            input_neighbors = input_tree.query_ball_point(center, radius)
            unused_bt = sum(i not in used and df.Phenotype.iat[i] in ("B", "T") for i in input_neighbors)
            assert unused_bt >= 50
            used.update(input_neighbors)
            center_global_id = int(df.global_instance_id.iat[int(record.Aggregate_Center_Index)])
            np.testing.assert_allclose(xy[id_to_index.at[center_global_id]], center)
            indices = np.asarray(tree.query_ball_point(center, radius), dtype=int)
            entry, members = normalize_region("TLS-Finder", run_id, ordinal, center, indices, cells, eligible,
                radius_um=radius, native_total_cells=len(indices), native_unused_BT_count=unused_bt,
                core_B_count=np.nan, icat=np.nan,
                membership_max_distance_um=float(np.linalg.norm(xy[indices] - center, axis=1).max()),
                membership_basis="geometrically derived disk; overlapping disks allowed")
            regions.append(entry)
            memberships[entry["region_id"]] = members
        settings.append({"method": "TLS-Finder", "run_id": run_id, "category": category, "variant": variant,
                         "radius_um": radius, "minimum_unused_BT": 50, "order_seed": seed,
                         "threshold_um": np.nan, "eligible_B": int((eligible & cells.celltype.eq("B")).sum()),
                         "eligible_T": int((eligible & cells.celltype.eq("T")).sum()),
                         "detections": len(result), "dense_B": np.nan, "detection_seconds": seconds})
        print(f"{run_id}: {len(result)} aggregates in {seconds:.2f}s", flush=True)
    return regions, memberships, settings


def run_tlsr(cells, out, runtime_manifest):
    native_dir = out / "native" / "tlsR"
    native_dir.mkdir(parents=True, exist_ok=True)
    input_dir = out / "inputs"
    input_dir.mkdir(exist_ok=True)
    inputs = {}
    eligible = {}
    for variant in ["all", "confidence_0.7", "stable_flips"]:
        frame = r_input(cells, variant)
        inputs[variant] = input_dir / f"tlsr_{variant}.csv"
        frame.to_csv(inputs[variant], index=False)
        eligible_ids = set(frame.loc[frame.phenotype.isin(["B cell", "T cell"]), "global_instance_id"])
        eligible[variant] = cells.global_instance_id.isin(eligible_ids).to_numpy()
    specs = [("tlsr_default", "defaults", 10, "all")]
    specs += [(f"tlsr_distance_{r}", "density_sweep", r, "all") for r in [30, 40, 50, 60, 80]]
    specs += [(f"tlsr_{v}", "input_quality", 10, v) for v in ["confidence_0.7", "stable_flips"]]
    jobs = pd.DataFrame([{"run_id": r, "threshold": t, "variant": v, "input_csv": str(inputs[v])} for r, c, t, v in specs])
    jobs_path = input_dir / "tlsr_jobs.csv"
    jobs.to_csv(jobs_path, index=False)
    env = os.environ.copy()
    env["R_LIBS_USER"] = runtime_manifest["R_library"]
    env["PATH"] = str(Path(runtime_manifest["Rscript"]).parent) + os.pathsep + env.get("PATH", "")
    command = ["/usr/bin/time", "-l", runtime_manifest["Rscript"], str(ROOT / "deployment" / "run_tlsr.R"), str(jobs_path), str(native_dir)]
    with (native_dir / "execution.log").open("w") as log:
        completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    log_text = (native_dir / "execution.log").read_text()
    if completed.returncode:
        # macOS time -l can fail after a successful R run when sandbox policy
        # blocks sysctl. Accept only that specific profiler failure after every
        # requested job has been confirmed in this run's cumulative metrics.
        metrics_path = native_dir / "tlsr_metrics.csv"
        profiler_only = "time: sysctl kern.clockrate: Operation not permitted" in log_text and "Error" not in log_text
        finished_ids = pd.read_csv(metrics_path).run_id.tolist() if metrics_path.exists() else []
        if not profiler_only or finished_ids != jobs.run_id.tolist():
            raise subprocess.CalledProcessError(completed.returncode, command)
    print((native_dir / "execution.log").read_text(), flush=True)
    metrics = pd.read_csv(native_dir / "tlsr_metrics.csv").set_index("run_id")
    xy = cells[["x_um", "y_um"]].to_numpy()
    regions, memberships, settings = [], {}, []
    id_to_index = pd.Series(np.arange(len(cells)), index=cells.global_instance_id)
    for run_id, category, threshold, variant in specs:
        native = pd.read_csv(native_dir / f"{run_id}_cells.csv.gz")
        assert len(native) == len(cells) and not native.global_instance_id.duplicated().any()
        np.testing.assert_array_equal(np.sort(native.global_instance_id), np.sort(cells.global_instance_id))
        region_data = pd.read_csv(native_dir / f"{run_id}_regions.csv")
        for region in region_data.itertuples():
            ids = native.loc[native.tls_id_knn.eq(region.native_id), "global_instance_id"]
            indices = id_to_index.loc[ids].to_numpy(dtype=int)
            entry, members = normalize_region("tlsR", run_id, region.native_id, [region.x_um, region.y_um],
                indices, cells, eligible[variant], radius_um=np.nan, native_total_cells=len(indices),
                native_unused_BT_count=np.nan, core_B_count=region.core_B_count, icat=region.icat,
                membership_max_distance_um=region.membership_max_distance_um,
                membership_basis="native tls_id_knn after center-based expansion; no distance limit on core graph")
            regions.append(entry)
            memberships[entry["region_id"]] = members
        metric = metrics.loc[run_id]
        settings.append({"method": "tlsR", "run_id": run_id, "category": category, "variant": variant,
                         "radius_um": np.nan, "threshold_um": threshold, "order_seed": None,
                         "k": 30, "minimum_dense_B": 50, "minimum_T_nearby": 10,
                         "T_search_radius_um": 50, "expansion_radius_um": 80,
                         "eligible_B": int(metric.eligible_B), "eligible_T": int(metric.eligible_T),
                         "dense_B": int(metric.dense_B), "detections": len(region_data),
                         "detection_seconds": float(metric.detection_seconds)})
    assert metrics.loc["tlsr_default", "dense_B"] == 0 and metrics.loc["tlsr_default", "detections"] == 0
    match = re.search(r"(\d+)\s+maximum resident set size", (native_dir / "execution.log").read_text())
    r_peak = int(match.group(1)) if match else None
    return regions, memberships, settings, r_peak


def compare(cells, candidates, regions, memberships, settings, out):
    tree = cKDTree(cells[["x_um", "y_um"]])
    original_sets = {}
    for row in candidates.itertuples():
        sub = cells.iloc[tree.query_ball_point([row.x_um, row.y_um], 150)]
        original_sets[row.candidate_id] = set(map(int, sub.loc[sub.celltype.isin(["B", "T"]), "global_instance_id"]))
    rows, edges, global_rows = [], [], []
    original_union = set().union(*original_sets.values())
    for setting in settings:
        run_id = setting["run_id"]
        sub = regions.loc[regions.run_id.eq(run_id)]
        union = set().union(*(memberships[r] for r in sub.region_id)) if len(sub) else set()
        overlap = len(union & original_union)
        global_rows.append({"run_id": run_id, "method": setting["method"], "source_BT_covered": len(union),
                            "source_BT_covered_fraction": len(union) / int(cells.celltype.isin(["B", "T"]).sum()),
                            "our_candidate_union_BT": len(original_union), "intersection_BT": overlap,
                            "candidate_union_coverage": overlap / len(original_union),
                            "global_jaccard_with_candidates": overlap / len(union | original_union)})
        for candidate in candidates.itertuples():
            ids = original_sets[candidate.candidate_id]
            location_ids = []
            pair_rows = []
            for region in sub.itertuples():
                members = memberships[region.region_id]
                distance = float(np.hypot(region.x_um - candidate.x_um, region.y_um - candidate.y_um))
                common = len(ids & members)
                jaccard = common / len(ids | members) if ids | members else 0
                edge = {"candidate_id": candidate.candidate_id, "method": setting["method"], "run_id": run_id,
                        "region_id": region.region_id, "center_distance_um": distance, "center_match": distance <= 150,
                        "intersection_BT": common, "candidate_BT_coverage": common / len(ids),
                        "region_BT_covered_by_candidate": common / len(members) if members else 0,
                        "BT_jaccard": jaccard}
                pair_rows.append(edge)
                if common or distance <= 150:
                    edges.append(edge)
                if distance <= 150:
                    location_ids.append(region.region_id)
            nearest = min(pair_rows, key=lambda r: r["center_distance_um"]) if pair_rows else None
            best = max(pair_rows, key=lambda r: r["BT_jaccard"]) if pair_rows else None
            if best and best["intersection_BT"] == 0:
                best = None
            matched_union = set().union(*(memberships[r] for r in location_ids)) if location_ids else set()
            rows.append({"candidate_id": candidate.candidate_id, "method": setting["method"], "run_id": run_id,
                         "source_B": candidate.B_150um, "source_T": candidate.T_150um,
                         "primary_center_match": bool(location_ids), "nearby_detection_count": len(location_ids),
                         "nearby_region_ids": json.dumps(location_ids), "split_location_match": len(location_ids) > 1,
                         "nearest_region_id": nearest["region_id"] if nearest else "",
                         "nearest_center_distance_um": nearest["center_distance_um"] if nearest else np.nan,
                         "best_overlap_region_id": best["region_id"] if best else "",
                         "best_BT_jaccard": best["BT_jaccard"] if best else 0,
                         "best_region_candidate_BT_coverage": best["candidate_BT_coverage"] if best else 0,
                         "all_detection_candidate_BT_coverage": len(ids & union) / len(ids),
                         "center_matched_candidate_BT_coverage": len(ids & matched_union) / len(ids)})
    comparison = pd.DataFrame(rows)
    pairs = pd.DataFrame(edges, columns=["candidate_id", "method", "run_id", "region_id", "center_distance_um",
                      "center_match", "intersection_BT", "candidate_BT_coverage", "region_BT_covered_by_candidate", "BT_jaccard"])
    merge_ids = pairs.loc[pairs.candidate_BT_coverage.ge(.25)].groupby("region_id").candidate_id.agg(list)
    regions["candidates_covered_at_least_25_percent"] = regions.region_id.map(lambda r: json.dumps(merge_ids.get(r, [])))
    regions["possible_merge_of_candidate_regions"] = regions.region_id.map(lambda r: len(merge_ids.get(r, [])) >= 2)
    regions["source_BT"] = regions.source_B + regions.source_T
    regions["b_fraction_source_BT"] = regions.source_B / regions.source_BT.replace(0, np.nan)
    regions.to_csv(out / "regions.csv", index=False)
    comparison.to_csv(out / "A1_A6_comparison.csv", index=False)
    pairs.to_csv(out / "candidate_region_pairs.csv", index=False)
    pd.DataFrame(global_rows).to_csv(out / "global_overlap.csv", index=False)
    settings_df = pd.DataFrame(settings)
    match_counts = comparison.groupby("run_id").primary_center_match.sum()
    settings_df["original_candidates_center_matched"] = settings_df.run_id.map(match_counts)
    settings_df.to_csv(out / "run_summary.csv", index=False)
    membership_rows = []
    lookup = cells.set_index("global_instance_id")
    for region_id, ids in memberships.items():
        run_id = region_id.split(":")[0]
        for cell_id in sorted(ids):
            membership_rows.append((run_id, region_id, cell_id, lookup.at[cell_id, "celltype"]))
    pd.DataFrame(membership_rows, columns=["run_id", "region_id", "global_instance_id", "source_celltype"]).to_csv(out / "BT_membership.csv.gz", index=False)
    return comparison, pairs, settings_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs" / "whole-slide-20261012")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--setup", action="store_true", help="Fetch pinned tools and install isolated R dependencies first")
    parser.add_argument("--report-only", action="store_true", help="Rebuild figures/report from saved comparison results")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    out = (args.output_dir or run / "tls_tool_comparison").resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        from report_tls_comparison import render
        render(run, out)
        provenance_path = out / "provenance.json"
        provenance = json.loads(provenance_path.read_text())
        provenance["status"] = "complete"
        save_json(provenance, provenance_path)
        return
    if args.setup:
        from setup_tls_tools import setup
        setup()
    manifest_path = RUNTIME / "manifest.json"
    if not manifest_path.exists():
        parser.error("Tools are not installed. Run with --setup first.")
    runtime_manifest = json.loads(manifest_path.read_text())
    assert runtime_manifest["sources"]["TLS-Finder"]["revision"] == "cde11ecbd541ec280609f494a9dd9409639359b4"
    assert runtime_manifest["sources"]["tlsR"]["revision"] == "5e6acf7bebc9464635782b37fcf9dc70fc6daafe"
    started = time.perf_counter()
    analysis = run / "tls_analysis"
    cells = pd.read_csv(analysis / "spatial_cells.csv").sort_values("global_instance_id").reset_index(drop=True)
    candidates = pd.read_csv(analysis / "ranked_candidate_regions.csv")
    summary = json.loads((analysis / "spatial_summary.json").read_text())
    validation = validate_inputs(cells, candidates, summary, run)
    adapted, reference = load_finder(out / "source_audit")
    validation["upstream_equivalence_checks"] = adapter_checks(cells, candidates, adapted, reference)
    print("Coordinates, frozen candidate counts and upstream adapter equivalence verified.", flush=True)
    finder_regions, finder_members, finder_settings = run_finder(cells, out, adapted)
    r_regions, r_members, r_settings, r_peak = run_tlsr(cells, out, runtime_manifest)
    regions = pd.DataFrame(finder_regions + r_regions, columns=REGION_COLUMNS)
    compare(cells, candidates, regions, finder_members | r_members, finder_settings + r_settings, out)
    provenance = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "detection_complete",
        "slide": summary["slide"], "microns_per_pixel": summary["microns_per_pixel"],
        "inputs": {name: {"path": str(analysis / name), "sha256": file_hash(analysis / name)} for name in
                   ["spatial_cells.csv", "ranked_candidate_regions.csv", "candidate_coordinates.json", "spatial_summary.json"]},
        "tool_setup": runtime_manifest, "validation": validation,
        "python": sys.version, "python_dependencies": {"numpy": np.__version__, "pandas": pd.__version__},
        "wall_seconds_detection_and_comparison": time.perf_counter() - started,
        "python_peak_resident_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "R_batch_peak_resident_bytes": r_peak,
        "comparison_contract": {"primary_center_match_radius_um": 150, "many_to_many": True,
            "membership_universe": "Original deduplicated B/T cells, fixed across input-quality checks",
            "TLS_Finder_membership": "Geometric disks; not exclusive native assignments",
            "tlsR_membership": "Native post-expansion tls_id_knn; exact core centers recovered by official no-expansion call",
            "merge_descriptor": "One region covers >=25% of B/T cells in at least two original candidate disks"},
        "limitations": ["Both tools share the H&E cell classifier's predictions; neither validates cell phenotypes",
            "Generic T predictions are not CD8 measurements", "No ground-truth TLS annotations or maturity markers",
            "Sensitivity parameters are predefined exploratory checks, not tuned to reproduce A1–A6",
            "TLS-Finder is order dependent and accepts T-only aggregates",
            "tlsR's core kNN graph has no distance cutoff and can bridge distant aggregates",
            "tlsR expansion uses distance from center, despite boundary-expansion wording in its documentation"],
    }
    import scipy
    provenance["python_dependencies"]["scipy"] = scipy.__version__
    save_json(provenance, out / "provenance.json")
    for name in ["compare_tls_tools.py", "tls_tools.py", "run_tlsr.R", "setup_tls_tools.py", "test_tls_tools.py", "report_tls_comparison.py"]:
        shutil.copy2(ROOT / "deployment" / name, out / "source_audit" / name)
    for tool in ["TLS-Finder", "tlsR"]:
        source = Path(runtime_manifest["sources"][tool]["directory"])
        for license_path in source.glob("*LICENSE*"):
            shutil.copy2(license_path, out / "source_audit" / f"{tool}_{license_path.name}")
    verification = ROOT / ".runtime" / "tls_tools_verification.log"
    if verification.exists():
        shutil.copy2(verification, out / "source_audit" / "fixture_verification.log")
    from report_tls_comparison import render
    render(run, out)
    provenance["status"] = "complete"
    provenance["wall_seconds_including_report"] = time.perf_counter() - started
    save_json(provenance, out / "provenance.json")
    print(f"Comparison complete: {out / 'TLS_tool_comparison.html'}", flush=True)


if __name__ == "__main__":
    main()
