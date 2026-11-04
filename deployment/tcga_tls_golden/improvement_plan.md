# Pan-cancer TLS adaptation on the TCGA golden regions (LUSC, BLCA, KIRC)

Written 2026-11-04, before any adapted output exists. It is an add-on to the frozen evaluation in this folder.

## Goal
Make HE2CELL identify the human-annotated TLS regions in LUSC, BLCA and KIRC. The method must stay one model and
one set of settings for every cancer, with no per-cancer tuning or colour normalisation. It must not lose accuracy
on ESCA, Visium or Immunocto.

## Rules
- The frozen run (`outputs/tcga_tls_golden_20261102/`) is untouched and stays the primary external result.
  Adapted arms go to `outputs/tcga_tls_adapt_<date>/`, which only reads the frozen outputs.
- **Patient split, fixed before any adapted output:**
  - Per cohort, 50 % of patients go to dev and 50 % to a locked test set, assigned by sha256(`tls-adapt-20261104` + patient).
  - If the Zenodo readme defines the HookNet-TLS train/test split, use its test patients as the locked test set instead.
    That allows a fair HookNet comparison on test.
- **Everything is fitted on dev only:** thresholds, region scorer, fine-tuning and arm selection. Test is scored once,
  after the arm is chosen.
- **Generalisation check:** within dev, leave-one-cancer-out (fit on two cohorts + ESCA dev, score the third).
  This estimates performance on a cancer type never seen.
- **Regression gates:**
  - ESCA test two-stage F1 drops by no more than 0.02 vs 0.854.
  - Visium TLS-vs-INFL spot AUROC drops by no more than 0.02 vs 0.923.
  - Immunocto CV macro-F1 drops by no more than 0.02 (classifier arms only).

## Step 0: failure analysis (CPU, saved outputs only)
Uses `slides/*/cells_*.csv.gz`, `regions.csv` and `predictions_8um.npz` on dev patients. For each cohort, it compares
missed and detected TLS annotations on:
- area;
- B, T, macrophage and all-nuclei density, and B share;
- whether a stage-1 nuclei aggregate exists;
- B share inside the aggregate vs the b_share cut-off.

False positives are broken down by negative label (inflammation, lymph node, tumour, other). Each miss is put in one bucket:
| Bucket | Meaning | Fix |
|---|---|---|
| no stage-1 aggregate | density/size settings too strict for this tissue | step 1 |
| aggregate, low B share | B under-called by the classifier, or a T-rich TLS | step 2 (features) or step 3 (classifier) |
| detected, coverage < 10 % | aggregate smaller than the polygon | step 1 (growth/σ) |

A gallery of 20 misses and 20 false positives per cohort (H&E with overlays of predicted cell types) is checked by eye.
It shows whether "low B share" means wrong cells or truly B-poor TLS.

## Step 1: pan-cancer recalibration (CPU)
Re-run the existing ESCA grid search for the two-stage and B-density detectors (σ, density threshold, min area,
b_share). Run it once on pooled dev patients (ESCA dev + LUSC + BLCA + KIRC dev), cluster-weighted so each cohort
counts equally. The objective is region recall at a negative-region specificity no lower than the frozen value, plus
ESCA dev F1. Report leave-one-cancer-out as well as pooled results.

## Step 2: learned region scorer (CPU)
Candidates are the stage-1 nuclei aggregates, at a relaxed threshold chosen on dev for ≥ 97 % TLS-candidate recall.
Per candidate, features come from the saved cell table only:
- **Composition:** B, T, macrophage and other densities and shares; B/T ratio.
- **Architecture:**
  - B core vs T rim (radial B and T profiles);
  - B clustering (Ripley's K at 20/50 µm);
  - the existing GC cues.
- **Context:** area, compactness, nuclear-density contrast against a 100 µm ring, and tumour-cell (Other) density in the ring.
- **Confounders:**
  - a lymph-node capsule/sinus proxy (aggregate area > 2 mm², peripheral position);
  - diffuse inflammation (high T, low B clustering).

The model is a monotone-constrained gradient-boosted tree (or logistic regression as a check). Labels: candidates
overlapping a TLS polygon by at least 10 % are positive. Candidates inside explicit negative polygons are negative.
Candidates outside all polygons are unlabelled and excluded. The operating threshold is fixed on dev. ESCA dev is
added with HookNet-TLS labels, at the same weight as one cohort.

## Step 3: classifier adaptation (GPU), only if step 0 shows B under-calling
Crops are re-extracted from the SVS for dev patients. The fine-tune restarts from `he2cell_esca_finetuned.pth` with:
- **Immunocto cell labels:** cross-entropy as now, so the cell labels stay anchored.
- **Weak region labels on TCGA dev:**
  - a learning-from-label-proportions loss: predicted B share inside TLS polygons ≥ the dev median of
    correctly detected TLS, and B share inside inflammation polygons ≤ the 90th percentile of true negatives;
  - no per-cell TCGA labels.
- **Same stain/scale augmentation** as now. Add JPEG/blur augmentation for TCGA scanner variation.

A no-label AdaBN arm on pooled TCGA dev crops is the cheap control. Checkpoints are selected on dev.
Leave-one-cancer-out fine-tunes run only if the pooled arm passes.

## Step 4 (optional): image features
A pathology foundation-model patch embedding (licence permitting) plus the cell-density maps feed a small
segmentation head on the 8 µm grid. This is reported only as a comparison: it answers whether cell
composition is the limiting factor.

## Evaluation (locked test patients, scored once)
- **Primary:** pooled TLS region recall at matched negative-region specificity (the frozen arm's specificity),
  adapted arm vs frozen two-stage. Paired by patient; 2000-resample patient cluster bootstrap; sign-flip permutation.
- **Secondary:**
  - per-cohort recall;
  - false-positive rate by negative label (lymph node and inflammation separately);
  - region-score AUROC;
  - leave-one-cancer-out recall;
  - HookNet-TLS on the same test patients, if its split allows.
- **Whole-slide check:**
  - Run the WSI screen and the adopted arm on 20 random test slides per cohort, including slides with no annotations.
  - Report calls per slide outside annotations, and review the top 10 unannotated calls per cohort by eye.
  - The annotations are not exhaustive, so this is descriptive. It is not a precision estimate.

## Adoption
The adapted arm replaces the default only if all of these hold:
- the primary endpoint improves, with a bootstrap CI lower bound above 0;
- every regression gate passes;
- leave-one-cancer-out recall is no lower than the frozen arm in every cohort.

Otherwise the frozen result stands and the arms are reported as negative results.
