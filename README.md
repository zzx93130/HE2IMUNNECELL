# H&E cell-type inference on this Mac

## External validation: USZ Visium pathologist labels (October 2026)

`deployment/visium_tls/` scores HE2CELL against the per-spot pathologist annotation (`manual_anno_tls`) in 8 Visium sections (5 lung, 3 kidney; Zenodo 14620362, `data/TLS_VISIUM_USZ/`). TLS = ETLS + MTLS + GC; negatives = normal, tumour and inflammation. Every detector setting stays frozen at its TCGA-ESCA value.

- **Coordinates.** The `.h5ad` `x_pixel`/`y_pixel` are not in the TIFF frame. `visium_cells.spots()` maps barcode → array row/column (`tissue_positions.csv`) → TIFF pixels (`*_manual_loupe_alignment.json`).
- **Resolution.** The TIFFs are ~0.78–0.80 µm/px. Tissue under in-tissue spots is bicubically upsampled ~3.1× to 0.25 µm/px, segmented once, and classified by both checkpoints (`visium_cells.py`). HookNet-TLS runs as a reference (`visium_hooknet.py`). `visium_eval.py` and `visium_report.py` write `outputs/visium_tls_validation/analysis/`.

Results over 11,008 labelled spots (925 TLS) and 108 pathologist TLS:

| Method | Spot AUROC, TLS vs all negatives | Spot AUROC, TLS vs inflammation |
|---|---|---|
| Fine-tuned classifier, smoothed B density | 0.99 | 0.91 [0.86–0.95] |
| Nuclear density, no classifier | 0.98 | 0.85 |
| HookNet-TLS | 0.93 | 0.80 |
| Original classifier, B density | 0.84 | 0.43 |

- **Frozen B-density detector (fine-tuned):** spot sensitivity 0.97, specificity 0.97, PPV 0.72. It finds 94% of pathologist TLS (93% early, 95% mature), with 3 regions over negative-only spots.
- **Inflammation is the weak point.** The frozen threshold calls 55% of inflammation spots TLS, versus 39% for nuclei only.
- **No maturity signal:** early and mature TLS are not separated (AUROC ≤ 0.62).
- **Weak spot RNA agreement:** per-spot B counts barely beat total cellularity against B-cell genes (median ρ 0.15 vs 0.12).
- **Over-called immune classes.** On these upsampled images the fine-tuned model calls only 16–54% of cells "Other". Treat its per-cell labels as unreliable at this resolution, even though aggregate-level TLS discrimination is strong.

## TCGA-ESCA TLS validation (October 2026)

Workflow code is in `deployment/esca_tls/` and `deployment/finetune/`; results are in `outputs/esca_tls_20261019/`. The report is `outputs/esca_tls_20261019/analysis_he2cell_ft/ESCA_TLS_validation_report.html` (the original-classifier baseline is in `analysis/`).

1. **Slides.** 156 diagnostic SVS from `/Volumes/KESU 1/TCGA-ESCA.tar`, extracted to `data/TCGA-ESCA_raw/` (`esca_extract.py` resumes per member). `TCGA-LN-A4A2` is corrupt; one 20x slide, one lymphoma case and one slide with no artifact-free window are excluded.
2. **QC.** `esca_qc.py` removes pen ink, black ink/folds and grey mounting medium from the Otsu tissue mask. Translucent red marker is not removed.
3. **Reference.** `esca_hooknet.py` streams the published HookNet-TLS over every slide (output identical to the earlier LUAD run). TLS polygons on artifacts are discarded.
4. **RNA.** `esca_rna.py` pulls TCGA-ESCA STAR TPM (12-chemokine score, CXCL13, MS4A1/CD19/CD79A, CD3E) and GDC histology.
5. **Windows.** `esca_windows.py` selects up to 3 HookNet-TLS, 3 random and 2 dense non-TLS windows (0.76 mm) per 40x slide, before any HE2CELL output. It then runs Cellpose and the classifier (D4 TTA) on 1024-px patches at a 768-px stride. Environment variables select the classifier (`ESCA_CLASSIFIER`), stain mode (`ESCA_STAIN=none|reinhard`), output folder (`ESCA_RESULT`) and crop retention (`ESCA_KEEP_CROPS=1`).
6. **Detectors and evaluation.** `esca_analysis.py` scores four detectors against HookNet. Parameters are tuned on dev patients (40%, split by patient hash), frozen, and scored once on test patients.
   - **Two-stage (primary):** stage 1 proposes aggregates from Cellpose nuclear density with no classifier. It uses the all-nuclei detector's own dev-tuned settings (σ 25 µm, ≥ 11,000 nuclei/mm², ≥ 20,000 µm²). Stage 2 accepts an aggregate as TLS only if HE2CELL's predicted-B share reaches a dev-tuned cut (8%). An optional "more B in core than periphery" rule was tested on dev patients and not selected. Every aggregate is written to `two_stage_proposals.csv` with its features and outcome. "Aggregate found, composition rejected; HookNet calls TLS" is its own category and the main pathologist review set.
   - **Comparisons:** smoothed predicted-B density, B+T density, and all nuclei with no classifier.

   `esca_report.py` adds the RNA correlations, B/T composition inside HookNet TLS, a germinal-centre feasibility check (composition features vs HookNet GC pixels) and an outcome-organised review gallery.

The original checkpoint did not transfer: its B calls depended on slide colour, and Reinhard or Macenko normalisation fixed some slides while breaking others. `deployment/finetune/finetune_classifier.py` fine-tunes it on the CRC training crops (`data/immunocto/`, from `/Volumes/KESU 1/CRC01–10.tar.gz`). The augmentation adds LAB colour-style shifts, HED ±15%, colour jitter, 0.9–1.45x rescaling and mask jitter; CRC03 and CRC09 are held out. Epoch 1 was selected by a pre-specified rule: held-out macro-F1 plus dev-ESCA B AUROC. The result is `outputs/classifier_finetune_20261019/best_finetuned.pth`; use it on raw, un-normalised TCGA slides.

On 85 held-out test patients (580 windows), scored against HookNet-TLS:

| Detector | Precision | Recall | F1 [95% CI] | Window AUROC |
|---|---|---|---|---|
| Two-stage (nuclei propose, B share accepts) | 0.83 | 0.93 | 0.88 [0.83–0.92] | 0.975 |
| B density, fine-tuned classifier | 0.93 | 0.89 | 0.91 [0.86–0.94] | 0.992 |
| All nuclei, no classifier | 0.66 | 0.95 | 0.77 [0.71–0.84] | 0.967 |
| B density, original classifier + Reinhard (517 shared windows) | 0.48 | 0.53 | 0.50 | 0.73 |

The composition step raises precision over nuclear density alone from 0.66 to 0.83 at nearly unchanged recall. On test patients, 250 stage-1 aggregates fall into four outcomes, and 10 HookNet TLS had no aggregate at all:

| Outcome | Aggregates |
|---|---|
| Accepted, HookNet agrees | 162 |
| Accepted by HE2CELL only | 34 |
| Rejected, HookNet agrees (mostly dense tumour sheets) | 52 |
| Rejected although HookNet calls TLS | 2 |

Germinal-centre presence is predicted by aggregate area (AUROC 0.79) but not by B share or core/periphery composition (AUROC 0.42–0.51). This suggests the classifier does not label germinal-centre B cells as B, so maturity grading needs more than these composition features.

Predicted B makes up a median 50% of cells inside HookNet TLS versus 1.6% in random tissue (97/98 slides). Random-window B density tracks CD79A/MS4A1 RNA (ρ 0.28/0.24) and T density tracks CD3E (ρ 0.31). HookNet is a model, not pathologist ground truth, and no ESCA cell-level labels exist.

## Revised TLS methods

The current report is `outputs/whole-slide-20261012/tls_methods_v2/TLS_methods_v2_report.html`. It leads with the cell-label audit, eight-view softmax-averaged predictions, and connected morphological regions. The original raw predictions, A1–A6 disks and earlier tool outputs are preserved. The old dashboard links to this revised report.

To repeat the saved-slide workflow, reusing verified classification and mask-feature caches:

```bash
.venv/bin/python deployment/revise_tls_methods.py --offline
```

Omit `--offline` to retrieve or resume the public RNA/reference checks. Individual stages are available through `--stage tta|features|analysis|report|external|verify`; `--force-tta` repeats all eight views. Geometry and reports reuse the existing isolated HookNet environment, and tlsR uses its project R library. The default classifier used for new slides also averages D4 views; use `--classification-tta none` only to reproduce identity-view behavior. Both identities and averaged probabilities are saved.

For the saved 129,470 cells, TTA changes 7,395 calls: 4,243 B, 21,548 T, 24 Macrophage and 103,655 Other. These are model predictions. The audit contains 50 original Other calls from each of A1/A2, 50 original T calls in five epithelial-appearing H&E polygons, all 63 original macrophage calls and 20 near misses, with blinded review cards and an exportable review form. Expert review and biological cell-label accuracy remain pending. Full-mask diagnostics reveal 649 truncated classifier crops; saved Cellpose objects are treated as nuclear-size proxies.

Morphological selection uses all objects on a 10-µm grid, Gaussian σ=25 µm, tissue-corrected density above an Otsu log-density threshold, and local mean object area below the slide's 40th percentile. Eight-connected components undergo 10-µm closing and fill holes up to 2,500 µm²; retained regions require 50 objects and 5,000 µm². The baseline gives 45 review regions, with 57/36 at σ=20/35. Ranking uses density, small size, uniform size and solidity; B/T labels do not select or rank regions. Areas, diameters, polygons, rejected regions, sensitivity overlaps and comparisons to every A1–A6 candidate are saved. These are not TLS counts. The active methods omit core/annulus odds ranking and slide-level B–B permutation tests.

TTA-input TLS-Finder gives 342 geometric aggregates. Native tlsR gives zero at defaults and three regions each at density ceilings 40/50 µm. Cached HookNet image predictions are preserved, with TTA cell membership recomputed. Native outputs, safe phenotype mappings, memberships and source revisions are recorded.

The RNA context uses a public UCSC GDC STAR TPM mirror: 14 requested genes, verified log₂(TPM+1) units, primary-tumor sample barcodes, and patient-level medians across 516 LUAD patients. This case is near the 70th percentile for the cohort-specific 12-chemokine PC1, 97th for CXCL13 and 79th for CD19/MS4A1. Xena averages aliquots within a sample before transformation. The adaptation is not the original microarray score scale and cannot confirm TLS spatially or validate cell labels. Direct GDC target metadata also identifies TCGA-55-7725-01A.

The public annotation archive is checksummed and five unique LUSC patients are selected with seed 569. Original-SVS-coordinate XMLs are paired with exact GDC image barcodes and checksums; annotated thumbnails never enter inference. The release was used for HookNet development and supplies no held-out split. Available images receive HookNet annotation-concordance checks only inside explicit TLS/other annotations; missing transfers are marked pending, and neither independent accuracy nor external validation of the morphology screen is claimed.

The Linux pipeline has been adapted for native Apple Silicon execution. It extracts tissue patches, segments cells using Cellpose, creates 64 × 64 RGB and mask crops, applies the supplied four-class ResNet50 classifier, and writes cell tables and slide overlays. Classification now defaults to averaging softmax probabilities over eight joint RGB/mask orientations (D4 TTA). Identity-view outputs are retained in `raw_*` columns; `--classification-tta none` reproduces single-view classification.

## Model and example slide

**Default classifier (since October 24, 2026):** `models/he2cell_esca_finetuned.pth`, the stain-robust fine-tune of your original checkpoint. Both launchers select it automatically, and it should be run on raw slides, without stain normalisation. Its sidecar, `models/he2cell_esca_finetuned.prior.json`, holds a class-prior correction that classification applies automatically: probabilities × training-class prior^0.5, renormalised after D4 averaging. The correction offsets the over-calling of B cells and macrophages caused by class-balanced fine-tuning.

- **Outputs.** Predictions, `prob_*` and `raw_*` are corrected. The uncorrected D4 probabilities are kept as `uncorrected_prob_*`, `prior_correction_tau` records the setting, and the `.inference.json` metadata lists the weights.
- **Overrides.** Pass `--prior-correction none` to skip the correction, or `--classifier models/best_model2.pth` to use the original checkpoint, which has no sidecar and is therefore uncorrected. A user-supplied `models/best_model.pth` still takes precedence over both.
- **Why this variant.** On held-out TCGA-ESCA patients (two-stage object F1 0.88 vs HookNet-TLS) and on Visium pathologist labels (spot AUROC 0.92 for TLS vs inflammation; frozen B-density detector F1 0.85, calling 42% of inflammation spots TLS versus 55% uncorrected), it was the best all-round choice. That comparison covered the uncorrected 10-patient fine-tune and a 40-patient fine-tune with and without correction. See the external-validation sections above and `outputs/classifier_finetune_40pt_20261022/` for the 40-patient comparison.
- **Verification.** On one ESCA slide's 28,882 cells, the installed default reproduced the validated corrected predictions exactly, and `--prior-correction none` reproduced the uncorrected run.

The original checkpoint remains at `models/best_model2.pth`. The example SVS slide is in `data/`. Both checkpoints use the four-class `SAM_ConvNet` architecture and label order from your source code:

| Class ID | Label |
| --- | --- |
| 0 | Other |
| 1 | T |
| 2 | Macrophage |
| 3 | B |

This checkpoint does not distinguish CD4 and CD8 T cells. Those labels appear only in commented-out code.

The default segmentation model is the official Cellpose `cpsam_v2`. Its public weights are cached under `models/cellpose_cache/`. If the Linux file named `cpsam_v2` was fine-tuned, copy **that exact file** to `models/cpsam_v2`; the launcher will select it automatically. A custom file can also be specified with `--cellpose-model /absolute/path/to/model`. Missing custom model paths stop the run instead of silently substituting a public model.

Only load classifier checkpoints from your trusted training run; they can contain Python training metadata.

## Run from Finder

Double-click `Run_HE2CELL.command` and select an SVS or TIFF slide. Your installed checkpoint is selected automatically. The pipeline uses the Apple GPU automatically. Results are saved to a dated folder inside `outputs/`, which opens when the run finishes.

Place custom segmentation weights in `models/cpsam_v2` before launching if applicable. The Finder launcher otherwise uses the official public model.

## Run from Terminal

From this project directory:

```sh
./he2cell doctor
./he2cell run --input "/absolute/path/to/slide.svs" --output "outputs/my_first_run"
```

To run your installed example over all selected tissue patches:

```sh
./he2cell run --input data --output outputs/example_full_slide --device mps
```

For a quick preview, add `--max-patches 8`. This samples eight patches evenly across the tissue patch index and records the limited scope in `run.json`. Preview counts apply only to those regions. The supplied example contains 1,410 eligible patches at the default settings, so a full run takes substantially longer than a preview.

To process a directory, provide the directory as `--input`. It reads SVS, TIF, and TIFF files directly inside that directory. A run requires an empty output directory so it cannot overwrite prior results.

```sh
./he2cell run \
  --input "/absolute/path/to/slides" \
  --classifier "/absolute/path/to/best_model.pth" \
  --cellpose-model "/absolute/path/to/custom/cpsam_v2" \
  --output "outputs/custom_model_run" \
  --device mps
```

Use `--device cpu` if needed. Default batch sizes are 32 for segmentation and 128 for classification. They can be adjusted with `--segmentation-batch` and `--classification-batch`. Defaults retain the Linux script's 1024-pixel patches, 512-pixel stride, 64-pixel cell crops, normalization, and four-class label order.

## Outputs

New runs default to **on-demand archiving**, including the Finder launcher. Each slide gets a subfolder containing `results.h5`. After inference and overlays finish, the pipeline writes the archive directly, checks every reconstructed patch/cell image and mask against its original, verifies ancillary file checksums, and only then removes the archived loose files. A failed verification leaves the loose outputs in place. The original slide and model weights are retained. Existing runs are not migrated or deleted.

The archive contains predictions, counts, review images, full segmentation labels, and the information needed to reconstruct crops. Keep the original slide with the archive; it is checked by SHA-256 when crops are read. For example:

```sh
# Recover tables, counts, QC and overlays; the original slide is not needed.
./he2cell export --archive outputs/my_first_run/SLIDE_NAME/results.h5 \
  --output outputs/exported_tables --tables-only

# Recover all loose files for older analysis scripts.
./he2cell export --archive outputs/my_first_run/SLIDE_NAME/results.h5 \
  --output outputs/exported_slide
```

Use `--source-slide /new/location/slide.svs` when the original slide has moved. Export requires an empty output directory and updates CSV paths to the restored files. Predictions and geometry are preserved; image reconstruction does not run either model. The Python `SlideResults` reader in `script/src/result_store.py` reads individual cells directly without exporting the full slide. Archived CSV paths record the original loose files and should be accessed through the reader or export command.

Pass `--storage files` to retain the original loose-output layout for workflows that require it. In that mode, each slide's subfolder contains:

- `cell_types.csv`: each cell's type, crop paths, and coordinates in the original slide.
- `cell_counts.csv`: counts by predicted type.
- `overlay/`: a colored cell-box thumbnail.
- `patches/`: source patches, coordinates, and tissue quality-control images.
- `cells/`: segmentation masks and individual cell crops.

The run-level `run.json` records inputs, model paths, device, settings, counts, completion or failure, and the archive path/format and verification totals. Blank slides produce an explicit no-tissue entry and a valid empty archive. Slides with no detected cells retain their patch labels and empty result table in the archive, with no overlay. During processing the pipeline still uses temporary PNG/TIFF/CSV files, so peak working disk space is higher than the final archive size.

Default archiving was checked on October 16, 2026 with a four-region real-slide run: all 126 cell crops/masks and four patches matched before cleanup, and a full export preserved their pixels, predictions and coordinates. Reader/lifecycle tests cover failed verification, interrupted cleanup, wrong source slides, relocation, empty results and keeping loose files. Actual blank-slide runs passed in both storage modes. Evidence is in `deployment/verification-archive-default.json` and `outputs/archive-default-validation-20261016/`. If cleanup is interrupted, the published archive retains the verified data; an incomplete `.partial.h5` is never used as a completed result.

## Installed environment and changes

The isolated `.venv` uses Python 3.12 and native ARM packages: PyTorch 2.14.1, torchvision 0.29.1, Cellpose 4.2.1.1, and bundled OpenSlide libraries. Other installed versions are pinned in `requirements-mac.lock.txt`. It does not modify existing Conda environments.

`./setup_mac.sh` reinstalls the locked packages and checks the runtime. If rebuilding on another Mac, install Python 3.12 or set `HE2CELL_PYTHON` to a Python 3.12 executable first.

Original versions of the three modified pipeline scripts are preserved in `deployment/original_src/` and in the supplied `script/src.zip`. Changes include Apple GPU selection, smaller batches, CPU-safe checkpoint loading and DataParallel prefix support, no unnecessary ImageNet weight download, preservation of slide sample names, handling of empty cell tables, and ordinary TIFF support. Cellpose uses float32 on this deployment for compatibility and predictable numerical behavior.

The original patch filter counted **zero-valued** mask pixels as tissue even though dark-mode tissue was encoded as **255**. The adapted filter counts the tissue pixels, uses `dark` for H&E, retains pixels equal to the Otsu threshold, and rejects uniform blank thumbnails. This corrects a preprocessing bug, so the original Linux invocation's `--tissue_mode light` should become `--tissue-mode dark` in this launcher. Review the saved tissue QC images with a real slide before a large run.

The original overlapping-patch core-selection method is retained. Slide pixel resolution, crop size, and staining should match the Linux training/inference conditions; deployment does not rescale slides to a new physical pixel size. Despite the original script filename, there is no CZI reader in this project.

## Verification

`deployment/verify_mac.py` checks TIFF extraction, blank-slide handling, crop padding, RGB/mask preprocessing, checkpoint loading, classifier execution, and CPU-versus-Apple-GPU agreement. Its full mode also runs the official Cellpose model and the complete output pipeline.

```sh
.venv/bin/python deployment/verify_mac.py --device mps
```

The smoke test uses artificial images and an **untrained, temporary** classifier checkpoint which is deleted afterwards. Reports in `deployment/verification-*.json` describe the checks actually completed. These checks establish software execution only. Reference Linux predictions or labeled cells are needed to assess prediction accuracy.

The complete Apple GPU smoke test passed on this Mac on October 11, 2026: the official Cellpose model found 36 synthetic cells, crop creation and classifier execution succeeded, and CSV counts and a slide overlay were generated. CPU and Apple GPU classifier logits agreed within a maximum absolute difference of approximately 0.00001 for the test input. Temporary images, classifier weights, and results were removed after verification.

On October 12, 2026, your `best_model2.pth` was tested on eight tissue regions from the real SVS slide. The Apple GPU pipeline completed, segmenting and classifying 682 cells: 578 Other, 87 T, 17 B, and 0 Macrophage. Results, original slide coordinates, 64 × 64 crops, and overlays are saved in `outputs/example-validation-20261012/`. `prediction_preview.png` shows the sampled regions at readable magnification. These are model predictions for eight regions, not whole-slide counts or verified biological labels. File integrity and a 32-cell CPU comparison are recorded in `deployment/verification-real-slide.json`.

## Completed whole-slide run

The example slide was processed in full on October 12, 2026: all 1,410 selected tissue patches, with 130,233 retained detections classified in 69.59 minutes. Results are in `outputs/whole-slide-20261012/`. Segmentation plus crop creation took 67.42 minutes; classification took 87.19 seconds. Confidence and class probabilities are included in `cell_types.csv`.

See `outputs/whole-slide-20261012/performance_report/performance_report.html` for figures, sample cell images, performance measurements, and limitations. The same folder contains the Markdown report and structured metrics. CPU/GPU classification agreement was 100% on a 447-cell stratified sample, while flips revealed substantial sensitivity in Macrophage and B predictions. These diagnostics do not establish biological accuracy.

To regenerate the report after a completed full-slide run with loose files (use `--storage files` for new runs intended for this legacy report workflow):

```sh
OMP_NUM_THREADS=8 .venv/bin/python deployment/report_whole_slide.py outputs/whole-slide-20261012 --device mps
```

## TLS-related spatial analysis

The predictions were screened for TLS-related B/T spatial patterns. Six regions pass an exploratory aggregate screen; A1/A2 are the first H&E review priorities, followed by A3/A4. This is a shortlist for validation, not a confirmed TLS count or maturity classification.

The analysis uses physical coordinates, B/T density maps, nearest-neighbor random-label controls, core-to-annulus composition, confidence sensitivity, and flip consistency. Original predictions are preserved; the spatial analysis conservatively excludes 763 likely overlap repeats and records the excluded IDs.

See `outputs/whole-slide-20261012/tls_analysis/TLS_pattern_report.html`, the H&E gallery, and `ranked_candidate_regions.csv` for results and original-slide coordinates. To regenerate the analysis and figures for this saved example:

```sh
OMP_NUM_THREADS=8 .venv/bin/python deployment/tls_flip_check.py
.venv/bin/python deployment/analyze_tls_spatial.py
.venv/bin/python deployment/plot_tls_spatial.py
.venv/bin/python deployment/report_tls_spatial.py
```

These scripts target the saved example slide and its verified 0.252-µm pixel spacing. The flip check requires Apple GPU access. Reference cell labels and TLS annotations are needed before treating these features as a validated TLS detector.

## Public TLS tool comparison

The saved whole-slide predictions can also be analyzed using [AAKoksoy TLS-Finder](https://github.com/AAKoksoy/TLS-Finder) and [labsyspharm tlsR](https://github.com/labsyspharm/tlsR). Both are spatial cell-coordinate methods; they reuse our predicted phenotypes and do not independently classify the H&E image.

```sh
# First use: download pinned public sources and install missing R packages locally.
.venv/bin/python deployment/compare_tls_tools.py --setup

# Subsequent runs reuse the installed tools and saved whole-slide predictions.
.venv/bin/python deployment/compare_tls_tools.py

# Rebuild figures and the portable report without rerunning detection.
.venv/bin/python deployment/compare_tls_tools.py --report-only

# Verify adapters and actual upstream algorithms on synthetic fixtures.
.venv/bin/python deployment/test_tls_tools.py
```

The single entry point accepts `--run-dir` and `--output-dir`; the default run is `outputs/whole-slide-20261012/`. It verifies every retained cell's physical coordinates against the original prediction table and freezes A1–A6. No image inference is repeated. Outputs go into `tls_tool_comparison/`, preserving earlier results: an interactive HTML report, whole-slide maps, H&E close-ups, parameter/input sensitivity plots, candidate overlap and matching tables, native tool outputs, source adaptations, fixture checks and provenance.

TLS-Finder is pinned to `cde11ecbd541ec280609f494a9dd9409639359b4`. Its notebook example uses a 160-µm radius and at least 50 previously unused combined B/T cells. The adapter substitutes generic B/T selection labels for its internal CD20/CD8 literals and removes an unused graph, with exact upstream equivalence checks. Its geometric disks are not native exclusive cell assignments. Ordered global IDs define the baseline; three shuffled orders expose its greedy order dependence. Radius checks use 80/120/160/200 µm.

tlsR 0.3.0 is pinned to `5e6acf7bebc9464635782b37fcf9dc70fc6daafe`. Its actual source defaults are checked at runtime: 30 B neighbors, a 10-µm mean-distance ceiling, at least 50 dense B cells, at least 10 T cells within 50 µm of the core center, and 80-µm center-based expansion. Other is relabeled Unclassified for its input because its substring matching would otherwise identify Other as T. Native returned memberships are retained. A separate official no-expansion call recovers exact core centers. Its core neighbor graph has no edge-distance limit and can connect separated groups; membership extent and ICAT spatial-spread values are reported. Predefined mean-distance sensitivity ceilings are 30/40/50/60/80 µm, without changing the default baseline.

For both tools, separate default-parameter checks restrict B/T detection eligibility to confidence ≥0.7 or stability under both image flips. Coverage uses the fixed original B/T universe. Center matches use the original 150-µm disks and allow splitting or merging; overlap is reported separately. Parameter sensitivity is exploratory and is not optimized to recover A1–A6. These results cannot establish accuracy, confirmed TLS counts or maturity.

Downloaded sources, archives and missing packages live under `.runtime/tls_tools/`. R uses `/opt/miniconda3/envs/singlecell/bin/Rscript`, with added packages installed exclusively into the project's `R-library` rather than the Conda environment. `deployment/setup_tls_tools.py --rscript /absolute/path/to/Rscript` can select another installed R when rebuilding. Setup needs internet access; subsequent detection and report generation run locally. macOS process-memory profiling can be unavailable inside a sandbox; that limitation does not change native detections.

## Independent H&E model: HookNet-TLS

[DIAGNijmegen HookNet-TLS](https://github.com/DIAGNijmegen/pathology-hooknet-tls) adds a trained whole-slide image segmentation method that receives H&E pixels directly. Our predicted B/T labels are used only for the subsequent comparison. Source `573680a5b962b1acae88cdeb6f8b1c9597570539`, its HookNet/WholeSlideData dependencies, and the public [Zenodo pretrained weights](https://zenodo.org/records/10614942) are pinned and checksummed. The separate runtime is `.runtime/hooknet_tls/venv`; it does not change the original classifier environment.

```sh
# Install pinned source, model weights and an isolated Mac runtime, then run.
.venv/bin/python deployment/add_hooknet_tls.py --setup

# Reuse the installed runtime and cached physical slide pyramid.
.venv/bin/python deployment/add_hooknet_tls.py

# Rebuild comparisons and update the main report from completed predictions.
.venv/bin/python deployment/add_hooknet_tls.py --report-only

# Check the actual authored filter functions, crop and lossless pyramid alignment.
.runtime/hooknet_tls/venv/bin/python deployment/test_hooknet_tls.py
```

The Mac port uses TensorFlow 2.18.1 with legacy Keras and Apple Metal. Three recorded compatibility changes update the Rescaling/Tensor import locations and remove a training-only compile argument. All checkpoint arrays load against the authored topology. Raw RGB inputs use the model's internal 1/255 rescaling. The native reader replaces Linux ASAP/JPEG preprocessing with lossless, bilinear physical resampling to 0.5 µm and a 2.0-µm context pyramid, Otsu tissue masking and the original 1244-pixel/220-overlap grid. The network's actual output is 1030 pixels; its original upper-left 1024-pixel writer crop and centered tissue-mask crop are preserved, rather than silently recentered. CPU/GPU inference and authored filtering fixtures are verified. No same-input Linux reference output is available to establish equivalence to the original Docker/ASAP implementation.

The published postprocessing uses 4-µm polygonization and retains TLS regions with at least 147² bounding-box pixels of 8-bit confidence ≥210. GC retention requires 60² pixels ≥245 and at least 50% GC area inside a retained TLS **bounding rectangle**. The comparison additionally reports actual GC/TLS polygon overlap, because the bounding-box criterion can retain a GC outside the TLS contour. No TLS or maturity confirmation is assigned.

Results and native masks, confidence TIFFs, contours, H&E review images and provenance are in `outputs/whole-slide-20261012/tls_tool_comparison/hooknet_tls/`. The main HTML report is extended in place, with its original spatial report preserved as `TLS_tool_comparison_before_HookNet.html`. Combined tables use `_with_hooknet`; prior spatial tables are untouched. The report generator retains this extension when rebuilding the original report.

On the saved example, all 120 tissue tiles completed on Apple GPU. Published filters retained seven TLS predictions and one GC prediction. Polygons overlap A1–A4; the original 150-µm centroid rule matches A1–A3. A4's nearest centroid is 157.7 µm away and its original B/T coverage is 53.8%. A5/A6 have no overlap. The GC passed bounding-rectangle association but has zero overlap with all retained TLS polygons, so it is shown separately for review. These are model comparisons without gold-standard annotations.

## Faster segmentation defaults

On October 13, 2026, the default segmentation path was accelerated with the newer PyTorch runtime, optimized SAM attention with cached relative-position embeddings, a 32-tile batch, and more efficient cell-mask cropping. Float32 precision and model weights are retained. Six-patch inference benchmarking measured 19.1% less time; a 751-cell crop-writing test measured 1.7× speed. A full pipeline check across 24 regions matched all 1,993 cell crops and predicted labels and every patch label mask exactly against the earlier whole-slide run.

See `outputs/acceleration-20261013/acceleration_report.md` for measured results, scope, and reproduction instructions. The projected 54–55 minute full-slide segmentation/crop runtime is an estimate, not a measured full-slide rerun. To use original attention with the current runtime, pass `--segmentation-attention original --segmentation-batch 8`. The previous runtime pins are preserved in `deployment/requirements-pre-acceleration.lock.txt`.

## Per-slide storage and on-demand crops

`deployment/compare_storage.py` benchmarks a completed slide without deleting its original files. It creates three lossless, per-slide HDF5 alternatives: decoded arrays, original compressed PNG bytes in indexed streams, and an on-demand archive that keeps full segmentation labels and reconstructs crops from the original slide. Predictions, original CSV bytes, review files, and run provenance are retained. The experiment includes every retained cell and every selected patch, including patches without retained cells.

```sh
.venv/bin/python deployment/compare_storage.py \
  --run-dir outputs/whole-slide-20261012 \
  --output-dir outputs/storage-example

# Recover one cell without running either model. Row 0 is the first prediction CSV row.
.venv/bin/python script/src/result_store.py outputs/storage-example/slide_on_demand.h5 \
  --row 0 --output-dir outputs/recovered-cell

.venv/bin/python deployment/test_result_store.py
```

For a relocated slide, supply `--source-slide /path/to/slide.svs`; the reader verifies its SHA-256. Both full benchmark archives can retrieve cells without the original slide. The Python `SlideResults` reader exposes `read_cell(row)`, `table()`, and `file_bytes(relative_path)` and supports a context manager. Original absolute paths in preserved CSVs are provenance; retrieval uses archive indices. The comparison tool remains a non-destructive experiment on existing loose outputs. New pipeline runs now use the production on-demand archive directly; older classifier and TLS scripts can use a full export or a run made with `--storage files`.

The measured comparison is in `outputs/storage-20261016/storage_report.md`, with exact bytes and repeated retrieval timings in `storage_metrics.json` and a chart in `storage_comparison.png`. Validation checks every cell RGB/mask pair, every complete patch label mask, every patch RGB image, and ancillary file bytes. On-demand reconstruction preserves the existing level-0, raw RGB, white-padding, centroid-rounding behavior. The builder currently requires a nonempty result, fixed crop/patch sizes, positive labels and successful crops; it fails verification for incompatible image-generation settings.

On the completed 130,233-cell slide, original result files total 4.225 GiB. Decoded-array HDF5 increased that to 4.808 GiB; preserving PNG bytes reduced it to 3.816 GiB (9.7% fewer logical bytes and 24.4% fewer allocated bytes). On-demand results use 106.77 MiB, or 185.52 MiB including the required original slide. All cell and patch pixels matched exactly. Random crop retrieval averaged 3.065 ms/cell for on-demand versus 0.260 ms for original PNGs; sequential reads were 0.242 versus 0.164 ms/cell. These are medians of four 1,024-cell passes with warm filesystem caches, excluding opening/checksum time. On-demand is the smallest tested retention option; the PNG-preserving archive provides self-contained reuse with less reconstruction work.

Storage totals cover the selected slide's pipeline directory. Separate run-level TLS analyses, model weights and `.DS_Store` files are outside this benchmark. Keep the source slide with an on-demand archive. File lengths and allocated filesystem blocks are reported separately; read benchmarks use warm caches. Original PNG/TIFF compression is measured rather than assumed to improve under HDF5. A new output directory is required unless `--resume` is passed to reuse completed archives and repeat validation; incomplete `.partial.h5` files are not treated as valid results.

## Upstream references

- [Cellpose installation and Python 3.12 guidance](https://pypi.org/project/cellpose/)
- [Cellpose model names and built-in models](https://cellpose.readthedocs.io/en/latest/models.html)
- [PyTorch Apple GPU backend](https://docs.pytorch.org/docs/main/notes/mps.html)
- [OpenSlide Python](https://openslide.org/api/python/)

## TLS maturity: germinal-centre morphology score (plan item 5a, failed)

On October 25, 2026, a germinal-centre (GC) morphology score was tested as a fix for maturity grading. Composition (B share, B-core enrichment) does not separate mature from early TLS. The score measured four cues inside each accepted two-stage aggregate, around the palest 60 µm disk of haematoxylin OD (colour deconvolution, σ 8 µm) that lay at least 30 µm inside the hole-filled aggregate. The cues were mantle-ring/centre pale contrast, nuclear-size gradient, the radial B-density dip and the nuclear-density dip. A logistic combination was fitted on 81 ESCA dev aggregates, with HookNet GC pixels (≥ 1% of the aggregate) as a weak proxy label, and then frozen. It was tested on 166 ESCA test aggregates (57 patients) and on 108 Visium pathologist TLS objects (47 mature = any MTLS/GC spot, 61 early). The Visium objects came from 8 sections.

The score did not meet the acceptance rule (Visium ≥ 0.75 with CI lower bound > 0.6, and ESCA ≥ 0.70). Visium mature vs early AUROC was 0.45 [0.35–0.55], and ESCA test vs HookNet GC was 0.58 [0.49–0.66]. CIs come from patient and section cluster bootstraps with 2000 resamples. No single cue passed either. On ESCA test, pale contrast reached 0.69 and the density dip 0.67, but on Visium they reached only 0.47 and 0.48. Aggregate size beat every morphology cue (area AUROC 0.80 on ESCA and 0.68 on Visium), and the existing composition baseline reached 0.50 on ESCA and 0.65 on Visium for B share. Post-hoc candidate radii of 50 and 75 µm were also tried; the best Visium AUROC was 0.63. The gallery shows that the palest disk often sits on a vessel, a stroma island or the gap between follicles rather than on a GC. No `GC_score` column was added to `two_stage_proposals.csv`. Limitation wording for the papers (plan 5b): "Composition and density grade TLS presence, not maturity. GC-aware morphology features were tested and did not reach the pre-specified acceptance (AUROC 0.45 for pathologist mature vs early on Visium; 0.58 against HookNet GC on TCGA-ESCA)." A GC-B class needs Ki67/BCL6/CD21 or pathologist-marked GCs (item 2 or the item 1 SFL reads).

```sh
.venv/bin/python deployment/esca_tls/esca_maturity.py features   # ESCA aggregates + dev fit (CPU, ~1 min)
.venv/bin/python deployment/visium_tls/visium_maturity.py         # Visium TLS objects (~1 min)
.venv/bin/python deployment/esca_tls/esca_maturity.py report     # evaluation, gallery, HTML (~5 min)
```

The outputs are in `outputs/maturity_20261025/`: `Maturity_report.html`, `summary.json`, `esca_aggregate_features.csv`, `visium_object_features.csv`, `frozen_gc_score.json` and `gallery/`.

## TLS burden and overall survival in TCGA-ESCA (plan item 4a, October 2026)

Question: is TLS burden associated with overall survival in TCGA-ESCA after adjustment for stage? Code: `deployment/esca_tls/esca_survival.py` (Cox models, report) and `deployment/esca_tls/esca_wsi_burden.py` (HE2CELL whole-slide burden). `esca_rna.clinical()` now also pulls vital status, days to death, days to last follow-up, age and sex (GDC `sex_at_birth`). Stage, sex and histology come from the diagnosis record flagged as the primary disease (new `*_primary` and `stage_group` columns), because for 1 in 3 cases GDC's first record is a metastasis, recurrence or prior primary. The original five columns of `clinical.csv` are unchanged. Cause of death is empty for every ESCA case, so the analysis is OS only.

```sh
.venv/bin/python deployment/esca_tls/esca_rna.py --clinical-only               # rewrite rna/clinical.csv only
.venv/bin/python deployment/esca_tls/esca_wsi_burden.py --calibrate           # screen grid on dev windows (CPU)
.venv/bin/python deployment/esca_tls/esca_wsi_burden.py --calibrate-e2e       # dev end-to-end choice (CPU)
.venv/bin/python deployment/esca_tls/esca_wsi_burden.py --validate            # test-window check (CPU)
.venv/bin/python deployment/esca_tls/esca_wsi_burden.py --run                 # whole slides, GPU lock, resumable
.venv/bin/python deployment/esca_tls/esca_survival.py --out outputs/esca_survival_20261025
```

Cohort: 150 patients (152 analysed slides), 63 deaths, 86 ESCC / 64 EAC, stage I/II 79, III/IV 53, missing 18. With 63 deaths, 80% power at α 0.05 requires an HR per SD below 0.70 or above 1.42. Adjusted (stage group, age, sex; stratified by histology) HR per SD of log1p burden:

| Burden | HR per SD (95% CI) | p |
|---|---|---|
| HookNet TLS per cm² | 1.19 (0.91–1.56) | 0.21 |
| HookNet TLS area fraction | 1.27 (0.98–1.64) | 0.075 |
| HE2CELL predicted-B density, random windows | 1.13 (0.87–1.47) | 0.37 |
| 12-chemokine score (RNA) | 1.13 (0.87–1.46) | 0.35 |

No burden definition is associated with OS. The point estimates lie on the "harmful" side, but every interval includes 1; tertile, complete-case and unstratified models agree. Schoenfeld tests show no PH violation. Adding burden does not raise the C-index above the clinical model (0.69). One of eight within-histology tests is nominal (ESCC, HookNet area fraction, HR 1.49, 1.04–2.14) and is not corrected for multiplicity.

HE2CELL whole-slide burden. The window pipeline never covered whole slides. A 4 µm haematoxylin screen was tried and rejected on dev windows: absolute OD would have processed almost all tissue, and per-slide local contrast covered only 89% of dev TLS. The adopted screen is a model-free surrogate of the frozen stage 1: nuclear blobs (LoG on the 1 µm level) counted on the 8 µm artefact-free QC grid. Regions with ≥ 9000 blobs/mm² over ≥ 5000 µm², grown by 100 µm, are segmented with the `esca_windows` patch/stride/core convention. They are classified with `he2cell_esca_finetuned.pth` (+ prior), and the frozen two-stage rule is applied per region. The setting was chosen on dev only, as the cheapest with ≥ 95% end-to-end recovery; 98% would have doubled the cost. On held-out test windows the screen recovers 162 of 166 accepted TLS (97.6%, 95% CI 95.2–99.4%). The expected cost is a median 274 cores per slide (90th percentile 1374). The run started 2026-10-25 12:49 and waits for item 3's GPU queue to finish (`outputs/_gpu.lock`); the full cohort is roughly 2–3 days of GPU time. Slides are processed in a seeded random patient order (seed 20261025), so any completed prefix is an unbiased subset. Every 10 completed slides, and at the end, the run rewrites the survival report and `summary.json`. HE2CELL whole-slide results are labelled INTERIM, with patient and death counts, next to HookNet on the same patients; they are modelled from 20 patients and 10 deaths. A patient counts only when all their slides are complete. Per-slide results go to `slides/<barcode>/wsi_burden/{cells.csv.gz, tls.csv, summary.json}`, progress to `outputs/esca_tls_20261019/wsi_burden.log`. Calibration tables, including the rejected screens, are in `outputs/esca_tls_20261019/wsi_screen_calibration/`.

Outputs: `outputs/esca_survival_20261025/{Survival_report.html, summary.json, patient_burden.csv, cox_tables.csv, figures/}`.

## Paired comparisons and ablations (plan item 3, October 2026)

Question: does the classifier add anything over nuclear density, and which parts of the recipe matter? Every arm was tuned on ESCA dev patients with the `esca_analysis` grid, frozen, and scored on the 85 ESCA test patients (reference HookNet-TLS) and the 8 Visium sections (pathologist labels). Paired differences use a cluster bootstrap with the same 2000 resamples for both arms (patients / sections) and a sign-flip permutation p. On Visium the sign-flip enumerates all 256 section swaps, so the minimum p is 0.0078.

- The plan's decision rule is **not met**: B density does not beat nuclei-only on both hard-negative tests without losing recall.
  - Against the frozen nuclei-only baseline, B density gains ESCA dense-window specificity (+0.108 [0.054, 0.171]) but loses ESCA object recall (−0.175).
  - The frozen nuclei-only threshold (11000) was the top of its grid. Re-tuned on dev with an extended grid (16000), nuclei-only reaches dense specificity 0.98, and B density's specificity gain is no longer significant (+0.018, p 0.23).
  - Visium TLS-vs-INFL AUROC is +0.069 [0.002, 0.136] for B density over nuclear density, but the permutation p is 0.094.
  - ESCA object F1 is higher for B density than for either nuclei-only baseline (+0.08, CI excludes 0).
- **Original checkpoint and weak-augmentation control are much worse on ESCA.** Both lose about 0.3 ESCA F1 against the stain-augmented fine-tune (−0.34 and −0.30). The control used the same 10 patients and schedule with only the original flips + HED ±0.012 (`finetune_classifier.py --augmentation weak --patients CRC01 … CRC10`); its ESCA F1 is 0.56 vs 0.85. On Visium it calls 95% of INFL spots TLS, although its spot AUROC is unchanged (0.914). Stain augmentation is the component that matters.
- **No TTA and prior τ 0 score higher on ESCA test** (F1 0.903 / 0.905 vs 0.854), but call more Visium INFL spots TLS (0.48 / 0.55 vs 0.42). Dev F1 did not prefer either (0.838 / 0.851 vs 0.846), so they are not re-adopted. τ 1.0 is clearly worse (F1 0.55).
- **σ 10–40 µm changes ESCA F1 by < 0.04 and Visium AUROC by < 0.03.** Several σ arms and the min-area parameter land on grid edges (flagged in the report). B-density F1 is flat (0.83–0.85 on dev) for thresholds from 150 to 1300 cells/mm².

Code: `deployment/esca_tls/esca_paired_stats.py`, `deployment/visium_tls/visium_paired_stats.py`, `deployment/ablations/{esca_sigma_threshold.py, ablation_report.py, run_gpu_queue.sh, gpu_locked.sh}`. Options added (defaults unchanged): `esca_rescore.py --tta`, and `visium_cells.py --classifiers … --identity`. The `--identity` option keeps the identity view of the D4 pass as `<name>_ttanone`.

Visium no-TTA and weak-aug arms needed one re-segmentation, because the v2 crops had been deleted. The new run (`cells_0.25um_ablate`) reproduces v2 exactly (cell positions and fine-tuned/original calls). New ESCA result folders: `he2cell_ft_tau10`, `he2cell_ft_ttanone(_tau05)` and `he2cell_ftweak(_tau05)` with matching `analysis_*`. The control checkpoint is in `outputs/classifier_finetune_weakaug_20261025/`.

Outputs: `outputs/ablations_20261025/{Ablation_report.html, summary.json, esca_paired_stats.json, visium_paired_stats.json, figures/}`.

## TLS states: composition, function and survival hook (October 2026)

Question: do HE2CELL-detected TLS fall into composition states, do those states match expression, and are they prognostic? Code is in `deployment/tls_states/`, outputs are in `outputs/tls_states_20261026/` (`TLS_states_report.html`, `summary.json`, `tables/`, `survival/`, the frozen model `tls_state_model.json`/`.pkl` and `preregistration.json`). Everything runs on CPU. Nothing is re-tuned: the frozen two-stage parameters and the fine-tuned classifier with the τ 0.5 prior are used throughout.

```sh
.venv/bin/python deployment/tls_states/esca_tls_states.py      # A: features of 247 ESCA window TLS, state model, RNA (~8 min)
.venv/bin/python deployment/tls_states/visium_tls_states.py     # B: 108 Visium pathologist TLS objects (~5 min)
.venv/bin/python deployment/tls_states/esca_state_survival.py   # C: whole-slide TLS states + Cox; re-run when the WSI run ends
.venv/bin/python deployment/tls_states/states_report.py         # report + summary.json
```

- **States (Part A).** For each accepted two-stage aggregate the script computes 11 features: area, nuclear density, B, T and macrophage shares, B-core and T-rim enrichment, B and T radial gradients from 30 µm rings, solidity and compactness. A Gaussian mixture is fitted on them with numpy EM, because scikit-learn is not installed in `.venv`.
  - BIC picks k = 2 with full covariances: **B-follicular** (160 TLS; median B 20%, T 40%, Other 36%) and **lymphoid-poor/Other-rich** (87; B 8%, Other 61%).
  - With fixed k = 3 the states are B-follicular (94; B 29%), **T-dominant** (105; T 46%) and lymphoid-poor (48).
  - Macrophages are 0.5–1.4% in every state, so no state is macrophage-rich.
  - The partitions are **not stable**: median patient-bootstrap ARI is 0.08 for k = 2 and 0.22 for k = 3, while diagonal-covariance k = 2 reaches 0.71. Treat the states as frozen labels on a continuum.
  - HookNet GC pixels are not enriched in the B-follicular state.
  - The state names are set by a rule on state means. The rule was amended once after the first fit, because it had mislabelled states; only the names changed.
- **ESCA associations.** These are patient-level and window-sampled (89–90 patients). EAC has more k = 3 B-follicular TLS than ESCC (+0.25 [0.09, 0.39]) and fewer T-dominant TLS. The primary B-follicular fraction is not associated with any RNA program (|ρ| ≤ 0.14). The k = 3 T-dominant fraction correlates with the GC program (ρ 0.38 [0.20, 0.55]), Tfh, mature DC, checkpoint and the 12-chemokine score (ρ 0.22–0.26). The lymphoid-poor fraction correlates negatively with most programs. None of these is corrected for multiplicity. Twenty-one extra genes were pulled from GDC Xena into `tables/rna_program_genes_log2tpm.csv`; `rna/` is unchanged.
- **Visium (Part B).** The h5ad `X` holds raw UMI counts (integers, identical to `filtered_feature_bc_matrix`, about 500 UMI per spot). Counts are summed per object and converted to log1p(CP10k). Composition comes from cells inside the 55 µm capture discs.
  - At the object level, B share vs MS4A1/CD79A/CD19 gives ρ 0.26 [−0.18, 0.58] pooled and a median within-section ρ of 0.50. The earlier all-spot result was 0.16.
  - T share vs CD3E/CD3D/CD2 gives −0.08 (within-section −0.27), and macrophage share vs CD68/CD163/C1QA gives −0.08 (within-section 0.11).
  - Spot level inside TLS shows no signal.
  - The frozen model calls 106 of 108 objects B-follicular. Under k = 3, the B-follicular posterior tracks B genes (ρ 0.32 [0.05, 0.47]) and mature DC (0.19 [0.02, 0.30]), and other programs only weakly.
  - Caveat: the images are 0.79 µm/px upsampled to 0.25 µm/px, the objects are bounded by spot tiles, and the sections are lung and kidney cancer.
- **Survival (Part C, INTERIM).** The script re-runs the frozen whole-slide detection from the saved cells to recover TLS masks. It checks the reproduction against `wsi_burden/tls.csv` and caches features in `wsi_features/`. It then assigns states and uses `esca_survival.analyse` unchanged.
  - Pre-registered primary hypotheses, fixed before any estimate and Holm-corrected: H1 total TLS per cm², H2 B-follicular-state TLS per cm², H3 fraction of TLS in the B-follicular state (patients with ≥ 3 TLS).
  - Interim, 27 patients and 12 deaths (28/152 slides): H1 HR per SD 1.10 [0.58, 2.08], H2 1.27 [0.63, 2.57]. H3 is not yet estimable (17 patients).
  - These numbers are descriptive only. Re-run the last two commands when the whole-slide run finishes.

## Second independent baseline: HoVer-Net (PanNuke) inflammatory cells (October 2026)

Question: does HE2CELL's predicted-B density beat a published cell-level method used in the same way? The baseline is HoVer-Net in "fast" mode, trained on PanNuke. Its "inflammatory" class is the standard lymphocyte proxy, and it is used for TLS detection as lymphocyte-density clustering.

- **Model.** TIAToolbox 1.6.0 runs `NucleusInstanceSegmentor(pretrained_model="hovernet_fast-pannuke")` in its own venv, `.runtime/tiatoolbox/venv` (Python 3.12, torch 2.5.1, `numcodecs<0.16` for zarr 2; `pip freeze` is in `requirements.lock.txt`).
  - Weights: `https://tiatoolbox.dcs.warwick.ac.uk/models/seg/hovernet_fast-pannuke.pth`, cached in `~/.tiatoolbox/models/`, sha256 `789b2be8…f57e8`, 151,109,291 bytes.
  - Provenance is recorded in `outputs/baselines_hovernet_20261026/cells/provenance.json`.
- **Images.** The input pixels are the same as HE2CELL's.
  - ESCA: each of the 988 adopted windows is re-read from the SVS at level 0 as the 3328-px square that `esca_windows`' 4×4 patches of 1024 px at stride 768 cover. Nuclei are kept in the 3072-px core.
  - Visium: the 8 sections are upsampled to 0.25 µm/px by `visium_cells.patch_grid`. This generator was factored out of `process()` and verified pixel-identical on KC2, 554 patches. Nuclei are kept in each patch's 768-px core.
  - HoVer-Net's post-processing tile is raised from 1024 to 3444 px, so each image is one tile. The default tiling re-runs inference on seam strips, about 2.6× the compute. On a test set the cells matched to 98–99% with identical types; the default seams drop about 0.5% of nuclei.
  - Cell tables are in `cells/esca/<barcode>/<window>.csv.gz` (level-0 coordinates) and `cells/visium/<sample>.csv.gz` (0.25 µm frame plus `x_native`/`y_native`). Columns: type, type_name, type_prob, area_um2.
- **Detectors.** Both are tuned on ESCA dev patients only, then frozen.
  - (a) Inflammatory density: `esca_analysis.detect` unchanged, with grid σ {15, 25} µm × threshold 1000–15000 cells/mm² × minimum area × minimum cells.
  - (b) DBSCAN: inflammatory centroids, eps 30 µm, min_samples {20, 50, 100}. A cluster counts as a candidate if its convex hull is ≥ 20,000 µm². DBSCAN is implemented in scipy because `.venv` has no scikit-learn; `hovernet_eval.py --check-dbscan` verifies it against sklearn.
- **Evaluation.** The same as the item-3 arms: `esca_paired_stats` / `visium_paired_stats` by import, with differences taken against the adopted HE2CELL B-density arm. A cell-level check per Visium spot compares HoVer-Net inflammatory counts with HE2CELL T+B counts, and each with PTPRC, CD3E and MS4A1 RNA (Spearman per section, section-cluster bootstrap).
- **Compute.** Timed on CPU (4 workers × 4 threads): about 475 s per ESCA window and about 51 s per Visium patch. KC2 (554 patches) took 2.9 h of wall time. The full run is about 230 CPU-worker-hours, which is more than a day.
  - `run_hovernet_queue.sh` started 4 CPU workers on 2026-10-26 10:53: Visium first, then ESCA.
  - It also queued an MPS worker for ESCA behind `outputs/_gpu.lock` (`gpu_locked.sh`, which now accepts a `GPU_LOCK_OWNER` label).
  - Workers share units through mkdir claims and skip finished ones, so re-running the script resumes.
  - The script writes an INTERIM report when Visium completes, again every 12 h, and a final report when every unit is done. The log is `outputs/baselines_hovernet_20261026/queue.log`.
  - Expected finish: about 2026-10-28 evening on CPU alone, earlier if the GPU lock frees.

```sh
deployment/baselines/run_hovernet_queue.sh            # (re)start or resume inference + chained evaluation/report
.venv/bin/python deployment/baselines/hovernet_eval.py && .venv/bin/python deployment/baselines/hovernet_report.py   # report from finished units
```

Outputs: `outputs/baselines_hovernet_20261026/{Baseline_report.html, summary.json, analysis/, figures/, cells/, queue.log}`. The current report covers only the timed sample (10 ESCA windows and KC2). It is marked INTERIM and must not be cited: parameters were tuned on 5 dev windows, and every CI comes from a single section.

## Cell-level validation on Immunocto CRC11–CRC40, not held out (October 2026)

Question: how well are T and B cells identified at the cell level, against immunofluorescence-derived labels, on Immunocto CRC11–CRC40? The fine-tuning step used CRC01–CRC10 only (CRC03 and CRC09 for checkpoint selection), and the 40-patient fine-tune is not evaluated.

**These patients are not held out.** `models/best_model2.pth` was trained on all 40 Immunocto patients (CRC01–CRC40; user statement 2026-10-26), and the fine-tune is initialised from it.
1. For A3 (original), CRC11–CRC40 are training data: its clean-crop numbers are training-set performance.
2. For A1/A2 they are unseen only by the fine-tuning step. The A1-vs-A3 clean comparison compares two models on crops both have seen in some form and cannot speak to generalisation.
3. Still supported: the stain-robustness ranking, the A1-vs-A2 prior-correction trade-off (same training data), and the calibration and prevalence match.
4. Not supported: any absolute held-out T/B accuracy claim for either model. A genuine held-out estimate needs external CD20/CD3 data (plan item 2) or a from-scratch patient-level cross-validation with the same recipe; neither was run.

Because 5,663,535 crops exceed the 1.5M cap, each patient was randomly subsampled to at most 50,000 crops (seed 20261026). That leaves 1,485,734 crops; CRC21 and CRC25 are used in full. Inference reproduces `model_eval.py` exactly: on 1000 crops, every prediction was identical and the largest probability difference was 3e-8. CIs come from a patient-cluster bootstrap with 2000 resamples, shared by all arms. Pooled cells, clean crops (not held-out accuracy; in-sample for A3):

| Arm | T F1 | B F1 | Macrophage F1 | Macro-F1 | Balanced acc. | AUROC P(B), B vs T | ECE P(B) / P(T) |
|---|---|---|---|---|---|---|---|
| A1 fine-tune + prior (default) | 0.819 [0.797–0.839] | 0.610 [0.549–0.651] | 0.619 [0.575–0.653] | 0.748 [0.731–0.759] | 0.734 | 0.857 [0.835–0.874] | 0.011 / 0.039 |
| A2 fine-tune, no prior | 0.775 [0.748–0.799] | 0.612 [0.547–0.662] | 0.566 [0.506–0.615] | 0.720 [0.701–0.735] | 0.790 | 0.862 [0.839–0.879] | 0.053 / 0.063 |
| A3 original `best_model2.pth` | 0.849 [0.827–0.867] | 0.657 [0.571–0.715] | 0.671 [0.635–0.699] | 0.783 [0.762–0.798] | 0.769 | 0.907 [0.885–0.924] | 0.013 / 0.015 |

- **On clean crops the original scores higher, on its own training patients.** A1 − A3: macro-F1 −0.036 [−0.047, −0.021], B F1 −0.047 [−0.078, −0.003], B-vs-T AUROC −0.050 [−0.063, −0.035]. A3 is scored on its training data and A1 inherits that exposure, so this says nothing about generalisation.
- **Prior correction trades B recall for precision.** A1 − A2: B recall 0.531 vs 0.727 (−0.196), B precision 0.717 vs 0.529, B F1 unchanged, T F1 +0.044 and calibration better. A1's predicted prevalence (Other / T / Macrophage / B 0.68 / 0.21 / 0.07 / 0.04) is closest to the IF prevalence of the sample (0.67 / 0.21 / 0.07 / 0.05). A2 over-calls macrophages (0.13) and B cells (0.075).
- **Stain robustness.** Each crop was re-scored under the LAB style shift + HED perturbation from `finetune_classifier.py`, at λ = 0.5, 1.0 (a full training draw) and 1.5 (beyond the training range). Each crop gets one seeded direction, shared by both checkpoints. Drops in macro-F1 (clean − perturbed):

  | Arm | λ 0.5 | λ 1.0 | λ 1.5 |
  |---|---|---|---|
  | A1 | +0.003 | +0.022 | +0.066 |
  | A2 | +0.005 | +0.021 | +0.063 |
  | A3 | +0.123 | +0.260 | +0.356 |

  B recall changes as follows:
  - **A1:** 0.531 → 0.546 / 0.540 / 0.496.
  - **A3:** 0.615 → 0.530 / 0.412 / 0.276.
  - **A1 − A3 difference of drops at λ 1.0:** macro-F1 −0.238 [−0.252, −0.222]; B recall −0.211 [−0.258, −0.152].

  From λ 0.5 onward the original is worse than the fine-tune: macro-F1 0.660 vs 0.745. This ranks the two models under colour shift and remains valid despite the training overlap, because both see the same crops. Caveat: the fine-tune was trained on this perturbation family, so this shows invariance to it, not to every stain process.

```sh
.venv/bin/python deployment/cell_validation/immunocto_holdout_eval.py all      # index, verify, infer (resumable, mps, no GPU lock), metrics
.venv/bin/python deployment/cell_validation/immunocto_holdout_report.py
```

Outputs: `outputs/cell_validation_20261026/{Cell_validation_report.html, summary.json, per_patient_metrics.csv, verification.json, crops.csv.gz, predictions/<ft|orig>_<clean|sev1|sev2|sev3>/CRC##.csv.gz}`. The predictions are uncorrected D4 probabilities; the prior correction is applied in `metrics`.

## Patient-level cross-validation of the classifier recipe, held-out T/B accuracy (October 2026)

Question: how accurately does the fine-tune recipe identify T and B cells in patients it has never seen? The deployed classifier cannot answer this, because its initialisation `models/best_model2.pth` was trained on all 40 Immunocto patients. So the recipe was re-run from scratch in four patient-level folds and every patient was scored by a model that saw neither it nor its fold during fitting or checkpoint selection.

Design (`outputs/classifier_cv_20261026/folds.json`):
1. **Folds:** CRC01–10, CRC11–20, CRC21–30, CRC31–40 are held out in turn; each fold trains on the other 30 patients.
2. **Recipe:** same as `outputs/classifier_finetune_20261019/settings.json` (strong augmentation, 10 epochs × 200,000 samples, batch 256, lr 1e-4, class-balanced sampling, label smoothing 0.05) via `finetune_classifier.py --init imagenet`. Two training patients per fold, drawn with seed [20261026, fold], form the early-stopping set (fold0 CRC28+CRC34, fold1 CRC35+CRC39, fold2 CRC07+CRC31, fold3 CRC12+CRC20). Checkpoint selection is the same rule as the deployed run (holdout macro-F1 plus ESCA-dev AUROC of P(B), TLS vs dense). Selected epochs: 7, 10, 7, 2.
3. **Prior:** τ 0.5 sidecar fitted from the class counts of the 28 fitted training patients of each fold, the same rule as the deployed `prior_correction_tau05.json` (checked to reproduce it exactly).
4. **Evaluation:** `immunocto_holdout_eval.py` machinery, at most 50,000 crops per patient (seed 20261026), D4 TTA, 1,970,538 crops over 40 patients. On 250 crops per fold, the predictions equal `model_eval.py` exactly (largest probability difference 3e-8). CIs are a patient-cluster bootstrap, 2000 resamples (the same resamples as the earlier validation when restricted to CRC11–40).

**How the recipe differs from the deployed fine-tune.**
- It starts from ImageNet weights, not `best_model2.pth`. This is the one leakage-free choice.
- It fits 28 patients per fold (the deployed one fitted 8 for the prior and used 10 for fine-tuning), so each fold is trained on more patients but from a weaker start.
- Checkpoint-selection patients are drawn by seed, not CRC03 and CRC09.

The deployed checkpoint is therefore not itself evaluated. These numbers estimate the recipe's out-of-patient accuracy, and the deployed checkpoint is expected to perform about like that on new patients of this domain (colorectal H&E, immunofluorescence-derived labels). They do not cover other organs, scanners or ESCA tissue; that needs external CD20/CD3 data.

Pooled cells over 40 held-out patients, clean crops (CV1 is the recipe with prior, the deployed configuration; CV2 has no prior):

| Arm | T F1 | B F1 | Macrophage F1 | Macro-F1 | AUROC P(B), B vs T | ECE P(B) |
|---|---|---|---|---|---|---|
| CV1 held-out, recipe + prior | 0.804 [0.784–0.822] | 0.589 [0.527–0.634] | 0.560 [0.524–0.594] | 0.721 [0.705–0.734] | 0.847 [0.823–0.865] | 0.020 [0.014–0.025] |
| CV2 held-out, no prior | 0.781 [0.758–0.800] | 0.577 [0.511–0.628] | 0.519 [0.470–0.568] | 0.695 [0.677–0.711] | 0.853 [0.829–0.871] | 0.054 [0.048–0.059] |

- **T cells:** CV1 precision 0.778 [0.753–0.801], recall 0.833 [0.811–0.851].
- **B cells:** CV1 precision 0.630 [0.576–0.674], recall 0.554 [0.478–0.609]. B cells are identified much less accurately than T cells. The patient-mean B F1 is 0.483 [0.424–0.537] against the pooled 0.589, so the pooled number is carried by patients with many B cells; per-patient B F1 ranges from about 0.03 to 0.78.
- **Prior correction:** it trades B recall for precision and fixes calibration and prevalence. CV1 predicts a B share of 0.042 [0.029–0.057] (IF 0.048 [0.035–0.063]); CV2 predicts 0.064 [0.047–0.082]. The prior is worth keeping for macro-F1 and T F1, not for B-vs-T ranking (AUROC equal within the CIs).
- **By fold (CV1 macro-F1):** 0.706 [0.684–0.729], 0.717 [0.678–0.741], 0.743 [0.707–0.769], 0.688 [0.668–0.702] (10-patient bootstrap); the weakest fold has the lowest T F1 (0.753).

**Comparison with the earlier numbers.** On the same 30 patients (CRC11–40, same crops, paired bootstrap), the training-set numbers for A1 (deployed) are optimistic. A1 macro-F1 0.748 against CV1 0.722 [0.703–0.737]; CV1 − A1: macro-F1 −0.026 [−0.033, −0.019], T F1 −0.024 [−0.031, −0.018], B F1 −0.005 [−0.019, 0.009], B-vs-T AUROC −0.013 [−0.022, −0.004]. The optimism is small and B F1 is unchanged. The original `best_model2.pth` (A3) is 0.061 macro-F1 higher than CV1, but is scored on its own training patients. The recomputed A1/A2/A3 numbers equal the published ones exactly.

**Stain robustness on held-out folds** (macro-F1 clean − perturbed, pooled, 40 patients; A rows are the earlier training-set numbers on CRC11–40):

| Arm | λ 0.5 | λ 1.0 | λ 1.5 |
|---|---|---|---|
| CV1 | +0.002 [0.001–0.004] | +0.010 [0.008–0.011] | +0.035 [0.032–0.038] |
| CV2 | +0.007 [0.006–0.010] | +0.015 [0.012–0.018] | +0.038 [0.034–0.043] |
| A1 (CRC11–40) | +0.003 | +0.022 | +0.066 |
| A3 (CRC11–40) | +0.123 | +0.260 | +0.356 |

The robustness of the recipe is not a training-set artefact: held-out drops are as small as, or smaller than, the A1 ones. B recall barely moves (CV1 0.554 → 0.561 / 0.557 / 0.537), and the original model's collapse is confirmed. As before, the fine-tune was trained on this perturbation family, so this shows invariance to it, not to every stain process.

```sh
.venv/bin/python deployment/cell_validation/classifier_cv.py folds
deployment/cell_validation/run_classifier_cv.sh      # trains 4 folds under the GPU lock (~67 min each, ~4.5 h in one hold), then prior, index, verify, infer (mps, no lock), metrics, report
.venv/bin/python deployment/cell_validation/classifier_cv_report.py
```

Outputs: `outputs/classifier_cv_20261026/{CV_report.html, summary.json, per_patient_metrics.csv, folds.json, verification.json, crops.csv.gz, figures/, fold{0..3}/{best_finetuned.pth, best_finetuned.prior.json, settings.json, history.json, train.log}, predictions/cv_<clean|sev1|sev2|sev3>/CRC##.csv.gz}`. `deployment/finetune/finetune_classifier.py` gained `--init imagenet` (default unchanged).
