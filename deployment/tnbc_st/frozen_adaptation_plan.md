# TNBC_ST frozen-section adaptation (add-on to the TNBC_ST analysis)

Written 2026-11-01, before any adapted model output exists.

## Why
On frozen-section ST H&E the default classifier (`models/he2cell_esca_finetuned.pth` + tau 0.5 prior) calls most
dense lymphoid-nodule cells "Other". Example: CN43_C2 is 90 % Other with 84 B cells, while its IHC shows about 5k CD20+.
A scale probe ruled out pixel size. This add-on tests cheap fixes that need no labels.

## Rules
- The frozen default arm in `outputs/tnbc_st_20261101/` is untouched and stays the primary external result.
- Adapted arms are reported as a separate, explicitly cohort-adapted result.
- Anything estimated from TNBC images is estimated only on training-fold patients and evaluated on held-out patients:
  - batch-norm statistics
  - class priors
  - normalisation targets, if any are taken from TNBC (the default target is ESCA)
- Folds: 4-fold split by patient (TNBC<n>), seeded 569.
- No TNBC labels (IHC, RNA, annotations) are used to fit anything in steps 1–2. They are used for evaluation only.
- Outputs go to `outputs/tnbc_frozen_adapt_20261101/`. The add-on only reads `outputs/tnbc_st_20261101/`, never writes to it.

## Step 1: diagnosis (CPU)
On the arrays already processed, inside pathologist Lymphoid-nodule and Lymphocyte regions:
- Segmentation QC:
  - Cellpose mask overlays on 6 dense crops.
  - Nuclei per 1000 µm² vs ESCA lymphoid aggregates.
  - Mask-area distribution, and the merged-nucleus rate (solidity < 0.85, area > 2× median).
- Crop QC:
  - Mean crop intensity and haematoxylin OD vs ESCA.
  - Fraction of crops in which the nucleus touches the crop border.
- Verdict: is the main failure segmentation, appearance, or both?
  - If segmentation is the main failure, report it. Re-segmentation is a later step.

## Step 2: label-free adaptation arms (classification only, same Cellpose cells)
Re-classify the existing crops of the processed priority arrays (no new segmentation):

| Arm | What changes |
|---|---|
| A0 | default (reproduce the stored labels as a check) |
| A1 | AdaBN: BatchNorm running stats re-estimated on unlabelled training-fold TNBC crops (≥ 50k, all classes) |
| A2 | Macenko stain normalisation of crops to the ESCA reference, before the model |
| A3 | A1 + A2 |
| A4 | best of A1–A3 + EM label-shift prior (Saerens 2002) re-estimated on training folds, replacing tau 0.5 |
| A5 | nucleus-size matching: crops resampled by 0.88 (nucleus area +28 % vs ESCA) |

All arms use D4 TTA and the same checkpoint weights. Only BN statistics or the inputs change.

## Evaluation (held-out patients only)
- **Label composition by annotation class:** Lymphoid nodule, Lymphocyte, High TIL stroma, Tumor, Low TIL stroma.
- **200 µm tile Spearman, per patient, then patient bootstrap 2000:**
  - T vs CD3+ density and B vs CD20+ density, using the main analysis registration and IHC blobs.
  - Discriminant: B vs CD20 minus B vs CD3.
- **ST RNA per spot:** B count vs MS4A1/CD79A, T count vs CD3E/CD3D.
- **TLS detection:** frozen ESCA two-stage and B-density detectors vs pathologist Lymphoid nodules and IHC CD20 aggregates, as counts plus recall/precision.
- **Primary comparison:** A3 vs A0 on the B-vs-CD20 tile Spearman, paired by patient, with bootstrap CI and sign-flip permutation.
- **Arm selection:** chosen by the training-fold B-vs-CD20 Spearman. All arms are reported regardless.
- **Sanity check:** the A1/A2 transforms are also applied to the Immunocto CV held-out folds. A drop in FFPE macro-F1 of more than 0.02 is flagged.

## Not in this step
Retraining with frozen augmentations, ST-RNA label-proportion fine-tuning, and frozen-to-FFPE style transfer.
These are proposed after step 2 results.
