"""Write outputs/tnbc_frozen_adapt_20261101/preregistration.json (before any adapted-arm output exists)."""
import json, datetime
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "outputs" / "tnbc_st_20261101"
OUT = ROOT / "outputs" / "tnbc_frozen_adapt_20261101"
inv = json.loads((MAIN / "inventory.json").read_text())["arrays"]
priority = (MAIN / "stage1_priority.txt").read_text().split()
patients = sorted({f"TNBC{inv[a]['sample']}" for a in priority}, key=lambda p: int(p[4:]))
perm = np.random.default_rng(569).permutation(len(patients))
folds = {f"fold{k}": sorted([patients[i] for i in part], key=lambda p: int(p[4:]))
         for k, part in enumerate(np.array_split(perm, 4))}
pre = {
 "written": datetime.datetime.now().isoformat(timespec="seconds"),
 "plan": "deployment/tnbc_st/frozen_adaptation_plan.md (steps 1-2)",
 "status": "written before any adapted-arm (A1-A5) prediction was computed",
 "checkpoint": {"path": "models/he2cell_esca_finetuned.pth", "sha256": "1778b5dd336db6e58f59b3a2533c2b15631d8f9b136d539072fd0038cb2a5f1d",
                "prior_sidecar": "models/he2cell_esca_finetuned.prior.json (tau 0.5) in A0-A3, A5"},
 "cells": "stage-1 Cellpose cpsam_v2 cells of outputs/tnbc_st_20261101/cells/<array>; crops (64 px RGB + mask at 0.25 um/px) rebuilt "
          "from imagesHD with the same bicubic patch resampling (visium_cells.patch_grid_image arithmetic) and the stage-1 Cellpose "
          "label masks captured read-only from the stage-1 temp dir; no new segmentation. Arrays whose masks were deleted before "
          "capture cannot be re-classified and are excluded from step 2 (reported).",
 "folds": {"unit": "patient (TNBC<n>)", "k": 4, "seed": 569, "population": "patients of stage1_priority.txt (one array each)",
           "rule": "numpy default_rng(569).permutation over patients sorted by number, np.array_split into 4", "assignment": folds},
 "arms": {
  "A0": "default: raw crops, stored BN stats, tau 0.5 prior, D4 TTA (reproduces stored labels; agreement reported)",
  "A1": "AdaBN: all BatchNorm running mean/var reset and re-estimated (cumulative average, model otherwise in eval) on >= 50,000 "
        "unlabelled crops sampled uniformly (seed 569) from the training-fold patients' arrays (all classes, one random D4 view per crop); "
        "if the processed training-fold arrays hold fewer crops, all are used and the shortfall is reported",
  "A2": "Macenko normalisation of each crop's RGB to the ESCA reference (stain matrix + 99th-percentile concentrations pooled over "
        "pixels of TCGA-ESCA dev cropbank crops, outputs/esca_tls_20261019/cropbank_dev); source stain matrix and max concentrations "
        "estimated per TNBC array from pixels of 3,000 random crops of that array (no labels, no other patient); Io 240, alpha 1, beta 0.15",
  "A3": "A2 then A1 (AdaBN statistics estimated on Macenko-normalised training-fold crops)",
  "A4": "the A1-A3 arm with the highest training-fold mean per-patient B-vs-CD20 200 um tile Spearman (chosen per fold) + Saerens-2002 EM "
        "label-shift prior: training prior taken as uniform (class-balanced fine-tuning), EM on the training-fold cells' D4 posteriors "
        "of that arm (tol 1e-6, max 1000 iter); adjusted posteriors p*pi_EM/0.25 renormalised replace the tau 0.5 correction",
  "A5": "nucleus-size matching: each 64-px crop taken from a 64/0.88 = 72.7 px window around the centroid, resized to 64 px (RGB bicubic, "
        "mask nearest), i.e. content scaled by 0.88 (nucleus area x0.77, undoing the +28 % vs ESCA); stored BN stats, tau 0.5",
  "common": "same weights, D4 softmax averaging; only BN statistics or the inputs change"},
 "evaluation": {
  "held_out": "every adapted-arm number on a patient comes from the fold where that patient is held out (BN stats, EM prior and A4 "
              "arm choice fitted on the other three folds)",
  "composition": "label shares per pathologist class: Lymphoid nodule, Lymphocyte, High TIL stroma, Tumor, Low TIL stroma",
  "tile_spearman": "200 um tiles, main-analysis registration, IHC blobs and overlap rule (tnbc_eval_ihc.tile_table, >= 50 % overlap); "
                   "per patient: rho(T, CD3), rho(B, CD20), rho(B, CD3), discriminant rho(B,CD20)-rho(B,CD3); patient bootstrap 2000",
  "st_rna": "per spot (tnbc_eval_rna geometry): B count vs MS4A1/CD79A panel, T count vs CD3E/CD3D panel (secondary)",
  "tls": "frozen ESCA two-stage and B-density detectors (tnbc_eval_path.detector_masks) vs pathologist Lymphoid nodules "
         "(tnbc_eval_path.reference/match) and IHC CD20 aggregates (2,000/mm2, 100 um tolerance): counts, recall, precision",
  "primary": "A3 vs A0 on the per-patient B-vs-CD20 200 um tile Spearman (patients with registered IHC), paired by patient: mean "
             "difference, patient bootstrap 95 % CI (2000 draws, seed 569) and two-sided sign-flip permutation p (10,000)",
  "arm_selection": "training-fold B-vs-CD20 Spearman (used only to pick A4's base arm); all arms reported regardless",
  "ffpe_sanity": "A1 (BN stats re-estimated on TNBC crops of a training fold), A2 (Macenko to ESCA, source estimated per Immunocto "
                 "patient) and A3 applied to each Immunocto CV fold model on its held-out patients (seeded subsample of crops); "
                 "macro-F1 drop > 0.02 vs the unadapted fold model is flagged",
  "labels": "TNBC IHC, RNA and annotations are used for evaluation only (annotations also choose where step-1 diagnostics run); "
            "the A4 base-arm choice uses training-fold IHC Spearman as specified by the plan"},
 "step1_verdict_rule": "segmentation is the main failure if, inside Lymphoid nodule/Lymphocyte regions, nuclei per 1000 um2 < 0.6x ESCA "
                       "TLS, or the merged-nucleus rate (solidity < 0.85 and area > 2x median) > 2x ESCA, or border-touch fraction > 2x ESCA; "
                       "appearance if those are within range but mean crop intensity / haematoxylin OD differ from ESCA by > 15 %; both if both",
}
OUT.mkdir(parents=True, exist_ok=True)
assert not (OUT / "preregistration.json").exists()
(OUT / "preregistration.json").write_text(json.dumps(pre, indent=1))
print(json.dumps(folds))
