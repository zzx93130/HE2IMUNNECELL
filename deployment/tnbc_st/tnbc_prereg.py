"""Write outputs/tnbc_st_20261101/preregistration.json: every reference rule, detector parameter, metric and
statistical procedure of Stage 4, fixed before any H&E (HE2CELL / HookNet / HoVer-Net) output is opened.

Detector parameters are copied from the frozen ESCA analyses (not re-tuned): two-stage and B density from
outputs/esca_tls_20261019/analysis_he2cell_ft_tau05/detection_results.json, the extended-grid nuclei-only
baseline from outputs/ablations_20261025/summary.json, HoVer-Net inflammatory density from
outputs/baselines_hovernet_20261026/summary.json. Refuses to overwrite an existing file.
"""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tnbc_common as C  # noqa: E402

ROOT = C.ROOT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    target = C.OUT / "preregistration.json"
    if target.exists():
        raise SystemExit(f"{target} exists; preregistration is write-once")
    esca = ROOT / "outputs/esca_tls_20261019/analysis_he2cell_ft_tau05/detection_results.json"
    ablations = ROOT / "outputs/ablations_20261025/summary.json"
    hovernet = ROOT / "outputs/baselines_hovernet_20261026/summary.json"
    results = json.loads(esca.read_text())["results"]
    arms_ab = json.loads(ablations.read_text())["arms"]
    hn = json.loads(hovernet.read_text())["chosen_parameters"]
    frozen = {
        "Two-stage (HE2CELL)": results["Two-stage (nuclei propose, composition accepts)"]["test"]["parameters"],
        "B density (HE2CELL)": results["B (HE2CELL)"]["test"]["parameters"],
        "Nuclei only (frozen ESCA, 11000)": results["All nuclei (no classifier)"]["test"]["parameters"],
        "Nuclei only (extended grid, 16000)": arms_ab["Nuclei only, extended threshold grid"]["parameters"]["All nuclei (no classifier)"],
        "HoVer-Net inflammatory density": hn["HoVer-Net inflammatory density"],
        "HookNet-TLS": "published model and postprocessing (deployment/esca_tls/esca_hooknet.py); TLS = published-filter TLS polygons; tile score = max TLS confidence",
    }
    prereg = {
        "written": datetime.datetime.now().isoformat(timespec="seconds"),
        "statement": "Written before any HE2CELL, HookNet-TLS or HoVer-Net output on TNBC_ST was opened. Only Stage-0 "
                     "material (inventory, scale, IHC stain/blob QC on 6 slides, registration trials) had been inspected.",
        "sources": {str(p.relative_to(ROOT)): sha(p) for p in [esca, ablations, hovernet]},
        "classifier": {"checkpoint": "models/he2cell_esca_finetuned.pth",
                       "sha256": "1778b5dd336db6e58f59b3a2533c2b15631d8f9b136d539072fd0038cb2a5f1d",
                       "prior": "models/he2cell_esca_finetuned.prior.json (tau 0.5, applied by model_eval.py)", "tta": "d4",
                       "stain_normalisation": "none"},
        "frozen_detectors": frozen,
        "grid_um": 8.0, "match_fraction": 0.10,
        "patient": "sample id (TNBC<n>); one patient per sample; CN<k> is the ST glass slide",
        "statistics": {"bootstrap": "patient-cluster, 2000 resamples, esca_paired_stats.draws (seed 20261025), same draws for all arms",
                       "paired": "esca_paired_stats.paired_comparison: difference with paired cluster-bootstrap CI and sign-flip permutation p (2000 flips or exhaustive)",
                       "multiplicity": "none; the primary comparisons are listed; everything else is descriptive"},
        "A_pathologist": {
            "arrays": "the annotated array of each sample (94; imageAnnotations PNG on the same H&E section)",
            "valid_region": "HE2CELL tissue mask (tnbc_cells.tissue_mask) on the 8 µm grid, minus annotation classes Nothing (unannotated), Artefacts and Hole (whitespace)",
            "reference_TLS": "8-connected components of class 'Lymphoid nodule' (#80801a) within the valid region, area >= 20,000 µm²",
            "ambiguous": "Lymphoid nodule components < 20,000 µm²: not counted as false negatives; a prediction overlapping only them is a false positive for strict precision and a true positive for lenient precision",
            "negatives": "inflamed = 'Lymphocyte' (#c4417f) + 'High TIL stroma' (#e9e900); all other annotated tissue = other negatives",
            "object_metrics": "esca_analysis.match_counts rule (overlap >= 10 % of the smaller object): recall, strict precision, lenient precision, F1; micro-pooled over arrays",
            "tiles": "96 µm tiles (12 x 12 grid cells) with >= 50 % valid pixels. Label TLS if Lymphoid nodule >= 50 % of valid pixels; inflamed if Lymphocyte + High TIL stroma >= 50 % and no Lymphoid nodule pixel; other negative if no Lymphoid nodule pixel and not inflamed; all remaining tiles excluded",
            "tile_scores": {"B density": "max smoothed predicted-B density in the tile", "Nuclei only": "max smoothed nuclear density",
                            "Two-stage": "highest B share among stage-1 aggregates intersecting the tile (0 without one)",
                            "HoVer-Net": "max smoothed inflammatory density", "HookNet-TLS": "max TLS confidence"},
            "tile_metrics": "AUROC TLS vs inflamed (primary), TLS vs all negatives",
            "inflamed_called_TLS": "area of inflamed annotation covered by the detector's TLS mask / inflamed area, pooled",
            "arms": list(frozen),
            "HoVer-Net_subset": "20 annotated arrays drawn with seed 20261101, stratified by reference-TLS count (0 / 1-2 / >= 3, proportional allocation); HoVer-Net comparisons restricted to them, all other arms also reported on the same subset",
            "primary_comparisons": ["Two-stage vs Nuclei only (frozen ESCA, 11000)", "Two-stage vs Nuclei only (extended grid, 16000)",
                                    "Two-stage vs HookNet-TLS", "Two-stage vs HoVer-Net inflammatory density (subset)",
                                    "B density vs the same baselines (secondary)"],
            "primary_metrics": ["object_F1", "tile AUROC TLS vs inflamed"],
        },
        "B_IHC_region": {
            "pairs": "annotated array <-> IHC slide of the same sample; only pairs whose registration passes QC (median 1 mm tile residual <= 100 µm, >= 3 tiles and >= 50 % of tiles converged); failures listed",
            "overlap": "H&E valid tissue mapped to the IHC frame AND IHC tissue mask; computed on the H&E grid",
            "tiles_um": {"primary": 500, "secondary": 200}, "tile_min_overlap": 0.5,
            "densities": "cells per mm² of overlap: HE2CELL T, B, T+B, all nuclei; HoVer-Net inflammatory (subset); IHC CD3+ and CD20+ LoG blobs (tnbc_ihc.py constants), CD3+ + CD20+",
            "per_sample": "Spearman rho over tiles; samples with >= 10 tiles",
            "summary": "mean of per-sample rho with patient bootstrap CI; pooled-tile rho secondary",
            "discriminant_validity": "per sample rho(B,CD20) - rho(B,CD3) and rho(T,CD3) - rho(T,CD20); mean, bootstrap CI; supported if the CI excludes 0 in the expected direction",
            "partial": "partial Spearman (rank residuals) rho(B,CD20 | HE T+B density) and rho(T,CD3 | HE T+B density)",
            "baselines": "rho(all nuclei, CD3+CD20), rho(HoVer-Net inflammatory, CD3+CD20), compared with rho(HE2CELL T+B, CD3+CD20) by paired per-sample differences",
            "CD2_slides": "TNBC 9, 10, 78, 82, 83, 84 are labelled CD2/CD20: excluded from every CD3 analysis (T vs CD3, discriminant validity, CD3 rim rule); kept for CD20 analyses; sensitivity analysis with CD2 treated as CD3",
        },
        "C_IHC_TLS": {
            "reference": "CD20+ blob counts on the 8 µm grid, Gaussian sigma 25 µm, density >= 2,000 CD20+ cells/mm², 8-connected components >= 20,000 µm² with >= 20 CD20+ cells, and >= 20 CD3+ cells within the component dilated by 50 µm (the CD3 rule is skipped on CD2 slides); sensitivity at 1,000 and 4,000 cells/mm²",
            "mapping": "IHC aggregates mapped onto the H&E grid with the accepted registration; only the overlap region is scored",
            "matching": "an H&E object and an IHC aggregate match when their distance is <= 100 µm (reference dilated by 100 µm overlaps the prediction)",
            "metrics": "recall (IHC aggregates matched) and precision (H&E objects matched), patient bootstrap; each arm of A; ceiling = pathologist Lymphoid nodule objects (>= 20,000 µm²) vs IHC aggregates by the same rule",
        },
        "D_ST_RNA": {
            "arrays": "arrays with HE2CELL cells, a byArray spot table and a raw count matrix",
            "spots": "spot_data-all pixel_x/pixel_y (imagesLarge frame), selected == 1, disk of 50 µm radius (100 µm spots) inside the valid tissue",
            "expression": "raw counts -> log1p(counts / library x 1e4); panel score = mean over genes present",
            "panels": {"T": ["CD3E", "CD3D", "CD2"], "B": ["MS4A1", "CD79A", "CD19"],
                       "TLS12": ["CCL2", "CCL3", "CCL4", "CCL5", "CCL8", "CCL18", "CCL19", "CCL21", "CXCL9", "CXCL10", "CXCL11", "CXCL13"]},
            "tests": "Spearman per array: HE2CELL T count vs T panel; B count vs B panel; B count and two-stage TLS cover vs TLS12; baseline all-nuclei count vs each panel; mean per-array rho with patient bootstrap",
            "ceiling": "IHC CD3+/CD20+ counts in the same disks (registered arrays only) vs the same panels",
        },
        "E_composition": {
            "objects": "two-stage TLS objects matched to IHC aggregates (C rule)",
            "HE": "B/(B+T) in the H&E object dilated by 50 µm", "IHC": "CD20+/(CD3+ + CD20+) in the matched aggregate dilated by 50 µm",
            "RNA": "log((MS4A1+1)/(CD3E+1)) of summed counts over spots centred inside the H&E object",
            "metric": "Spearman across objects, patient bootstrap; CD2 slides excluded",
        },
    }
    C.save_json(prereg, target)
    print("wrote", target)


if __name__ == "__main__":
    main()
