# TLS validation: plan for the five improvements (drafted 2026-10-25)

Each item below is written so that an Opus 5.5 agent session can execute it from this file alone,
once its inputs exist. Items 3 and 5 need no new data; item 4a (ESCA survival) needs only GDC clinical
fields; items 1, 2 and 4b wait for data you supply. Drop-folders are listed per item; put nothing else in them.

Shared rules for every agent
- Code goes under `deployment/<item_folder>/`, outputs under `outputs/<item_folder>_<YYYYMMDD>/`, a section in
  `README.md`, and an HTML report with the same layout as the ESCA/Visium reports.
- Python: `.venv/bin/python` (HookNet: `.runtime/hooknet_tls/venv/bin/python`). Device `mps`.
- Default classifier: `models/he2cell_esca_finetuned.pth` + `models/he2cell_esca_finetuned.prior.json` (tau 0.5).
  Comparison arms: original `models/best_model2.pth`; nuclei-only (no classifier); HookNet-TLS; two-stage.
- Frozen parameters: `outputs/esca_tls_20261019/analysis_he2cell_ft_tau05/report_summary.json` → `chosen_parameters`.
  Nothing is re-tuned on test patients or on external labels. Any new parameter (ablations, maturity score)
  is tuned on ESCA dev patients only (`split == "dev"` in `windows.csv`), then frozen.
- Uncertainty: patient-cluster bootstrap (ESCA, 85 test patients) or section-cluster bootstrap (Visium, 8 sections),
  2000 resamples, 95% percentile CIs. Paired comparisons use the same resamples for both arms.
- Write `summary.json` next to every report so later items can read numbers instead of HTML.

---

## 1. Blinded pathologist review of ESCA test windows  (waiting: labels)

Goal: human reference for HE2CELL (B density and two-stage), nuclei-only and HookNet on the held-out ESCA test set,
plus inter-observer agreement.

### 1a. Packaging (agent does this before the review; no model output visible to reviewers)
Source tables: `analysis_he2cell_ft_tau05/two_stage_proposals.csv` (test rows, `outcome`),
`two_stage_unproposed_HookNet_TLS.csv`, `windows.csv`, `per_window_results.csv`.
Slides: `data/TCGA-ESCA_raw/TCGA-ESCA/*.svs` (OpenSlide, level 0 = 0.25 µm/px).

Stratified sample, ~300 items (test split counts in brackets; take all when below the cap):

| stratum | source | cap |
|---|---|---|
| A both agree (TLS, HookNet agrees) | proposals [149] | 60 |
| B HE2CELL only | proposals [17] | all |
| C aggregate rejected by composition, HookNet calls TLS | proposals [15] | all |
| D aggregate rejected, HookNet agrees not TLS | proposals [69] | 50 |
| E HookNet TLS not proposed by stage 1 | unproposed table [10] | all |
| F random tissue, no aggregate and no HookNet TLS in window | windows (`random`) | 80 |
| G dense hard negatives, no aggregate, no HookNet TLS | windows (`dense`) | 40 |

Sampling: seeded (`seed = 20261025`), at most 3 items per patient per stratum, record the sampling
probability of every item (needed for weighted estimates). Unit = one candidate region. Each item shows the same
thing regardless of stratum: a 1.0 mm field at 1 µm/px (overview) and a 0.5 mm field at 0.5 µm/px (detail),
centred on the candidate (for F/G: a random tissue point in the window), with a thin grey dashed circle of fixed
radius (250 µm) marking the region in question. No contours, no colours, no barcode. Items get random IDs
(`R0001`…). The key (`item_id → barcode, window_id, stratum, model calls, sampling weight`) is written to
`outputs/pathologist_review_<date>/_key/review_key.csv` and never copied into reviewer material.

Reviewer package: one self-contained HTML per reviewer (`review_<reviewer>.html`, images embedded, keyboard
shortcuts, progress saved in localStorage, "Export answers CSV" button) plus a one-page instruction sheet.
Item order is shuffled per reviewer. Questions per item:
1. Lymphoid aggregate present inside the circle? yes / no / cannot assess (artefact, out of focus)
2. If yes: TLS? yes / no (e.g. diffuse infiltrate, normal lymphoid tissue, lymph node)
3. If TLS: maturity — E-TLS (dense aggregate, no follicle) / PFL (primary follicle-like, dense B zone, no GC) /
   SFL (secondary follicle-like, germinal centre visible) — the Posch/Siliņa scheme.
4. Confidence 1–3, optional free text.

### 1b. Analysis (after both CSVs are back; drop in `data/pathologist_review/answers/<reviewer>.csv`)
- Inter-observer: Cohen's κ for TLS yes/no, weighted κ for maturity (E-TLS < PFL < SFL), per-stratum agreement table.
- Reference: consensus (both say TLS / both say not TLS). Disagreements: primary analysis excludes them;
  sensitivity analyses (i) either-reviewer-positive, (ii) adjudicated if you provide a third read.
- Per method (HE2CELL B density, two-stage, nuclei-only, HookNet, original checkpoint): sensitivity, specificity,
  PPV, NPV, accuracy against consensus, with patient-cluster bootstrap CIs. Report (a) raw within-sample values and
  (b) estimates re-weighted by inverse sampling probability so they describe the full test set.
- Paired method differences (sens, spec, accuracy) with bootstrap CIs and McNemar on discordant pairs.
- Maturity: HE2CELL maturity score (from item 5, if adopted) vs pathologist E-TLS/PFL/SFL; AUROC SFL vs rest.
- Report sections: sampling table; κ; method table; the strata B/C/E as explicit "who was right" tables
  (HE2CELL-only, composition-rejected-but-HookNet, HookNet-missed-by-nuclei) with example tiles.

Deliverables: `deployment/pathologist_review/{review_sample.py, review_package.py, review_analysis.py}`,
`outputs/pathologist_review_<date>/{reviewer packages, _key/, analysis/Pathologist_review_report.html}`.

---

## 2. Cell-level validation with immunostaining  (waiting: data)

Goal: per-cell accuracy of the B / T / Macrophage / Other calls against CD20 / CD3 (± CD68), outside colorectal cancer.

Inputs (drop in `data/ihc_validation/<case>/`): H&E image and the immunostain(s) of the same or a serial section,
pixel size, tumour type, and whether same-section (restained / mIF) or serial. Preferred: same-section mIF
(DAPI + CD20 + CD3 ± CD68) or sequential IHC, 40×, ≥5 cases, ≥2 tumour types, your own lab.
Leakage check before anything else: the Immunocto crops come from the Orion CRC same-section H&E/IF series
(Lin et al.); any Orion-derived data, or any public set the agent cannot trace to a non-Orion source, is excluded.
The agent writes the provenance check into the report.

Method
- Same-section: register IF→H&E on nuclei (DAPI vs haematoxylin channel from colour deconvolution), rigid then
  affine, residual error reported (target < 2 µm, must be < 1 nucleus radius). Serial sections: region-level
  comparison only (B fraction and T fraction per 100 µm tile), no per-cell matching.
- Reference per cell: IF nucleus segmentation (Cellpose on DAPI), marker positivity by Otsu on the membrane ring
  with a manual spot-check of 50 cells per case; CD20+ → B, CD3+ → T, CD68+ → Macrophage, else Other; double
  positives excluded.
- HE2CELL: standard pipeline on the H&E (segment once; score with finetuned, original, and finetuned without prior
  correction). Match predicted cells to IF cells within 4 µm (Hungarian on centroids).
- Metrics: confusion matrix, per-class precision/recall/F1, balanced accuracy, calibration (reliability curve of
  `prob_B`), by case and pooled, case-cluster bootstrap. Per-tile B fraction and T fraction: Spearman and
  Bland–Altman. Also inside vs outside lymphoid aggregates, because the TLS claim rests on aggregate-level composition.
- Report the Visium-style sanity table (share of Other, T, B, Macrophage) so over-calling is visible.

Deliverables: `deployment/ihc_validation/{ihc_register.py, ihc_reference.py, ihc_eval.py, ihc_report.py}`,
`outputs/ihc_validation_<date>/`.

---

## 3. Statistical comparisons and ablations  (ready now)

Question: does the classifier add anything over nuclear density, and which parts of the recipe matter?

### 3a. Paired comparisons (both cohorts, frozen parameters)
- ESCA test (85 patients): B density vs nuclei-only vs two-stage vs HookNet — object F1, precision, recall,
  window AUROC; hard-negative subset = `dense` windows (specificity there is the question).
- Visium (8 sections): AUROC TLS-vs-all and TLS-vs-INFL, object recall, INFL-called-TLS rate.
- For every pair of methods: paired difference with cluster-bootstrap CI (same resamples), and a cluster
  permutation p-value (sign-flip over patients/sections). Say explicitly which differences exclude 0.
- Decision rule stated in the report: the classifier "adds" if B density beats nuclei-only on hard negatives
  (ESCA dense-window specificity; Visium TLS-vs-INFL AUROC) with CI excluding 0, without losing recall.

### 3b. Ablations
Each ablation is re-tuned on ESCA dev with the same grid, frozen, then scored on ESCA test and Visium.
| ablation | arms | how |
|---|---|---|
| augmentation | fine-tuned (strong aug) vs control fine-tune with the same 10 patients, same schedule, weak aug (original HED only) vs original | one extra training run with `deployment/finetune/finetune_classifier.py` (add `--augmentation weak`), ~hours on MPS |
| TTA | d4 vs none | `esca_rescore.py` on the kept `he2cell_ft/crops` with `--tta none`; Visium: `visium_cells.py` scores with `--tta none` (one extra model_eval pass, no re-segmentation) |
| prior correction | tau 0, 0.5, 1.0 | `esca_prior_adjust.py` (probabilities saved) |
| σ | 10, 15, 25, 40 µm | `esca_analysis.py` grid, dev-tuned threshold per σ |
| threshold | sensitivity curve of F1 / INFL rate vs threshold | from existing densities |

Present one table: rows = ablation arm, columns = ESCA F1 [CI], ESCA dense-window specificity, Visium AUROC vs INFL,
Visium INFL-called-TLS rate, Visium Other share. Plus a figure of paired differences.

Deliverables: `deployment/esca_tls/esca_paired_stats.py`, `deployment/visium_tls/visium_paired_stats.py`,
`deployment/finetune/finetune_classifier.py --augmentation weak`, `outputs/ablations_<date>/Ablation_report.html`.

---

## 4. Clinical relevance  (4a ready now; 4b waiting: slides)

### 4a. TLS burden vs overall survival in TCGA-ESCA
- Clinical: extend `esca_rna.clinical()` with `demographic.vital_status, demographic.days_to_death,
  diagnoses.days_to_last_follow_up, demographic.age_at_index, demographic.gender`. Stage grouped I/II vs III/IV;
  61 of 185 cases have no stage — report them as a "missing" stratum and run a complete-case sensitivity analysis.
- Burden definitions (per patient, averaged over slides):
  1. HookNet whole-slide TLS count per cm² tissue and TLS area fraction (exists: `slide_hooknet_burden.csv`).
  2. HE2CELL whole-slide burden. The window pipeline does not cover whole slides, so add a screening step:
     haematoxylin-density proposals at 4 µm/px over the whole slide (sensitive, cheap), then run HE2CELL
     segmentation + classification only inside proposals (+100 µm margin), apply the frozen two-stage acceptance,
     and count accepted TLS per cm² and B-cell TLS area fraction. Budget ~5–15 min per slide on MPS; run in the
     background with the existing slide-lock pattern. Validate the screen on the test windows first (it must
     propose ≥95% of accepted TLS).
  3. Mean predicted-B density in the random windows (already available; a cheap infiltration covariate).
  4. 12-chemokine score (exists) as the RNA comparator.
- Models: Cox PH for OS, burden as log1p-continuous and as tertiles; adjusted for stage group, age, sex and
  histology (ESCC vs EAC), and stratified by histology because the two diseases differ. Schoenfeld test for PH;
  KM curves by tertile; C-index. Lifelines in `.venv` (install if missing). ~150 patients and ~70 deaths: state
  power up front; a null result is reportable.
- Also disease-specific survival if `days_to_death` plus cause is available; otherwise OS only.

### 4b. Other TCGA cancers you supply
Drop each cohort as `data/TCGA-<CODE>_raw/` (svs files, plus `gdc_sample_sheet.<CODE>.csv`). The agent runs the
ESCA pipeline unchanged with `ESCA_RAW`/`OUT` pointed at the cohort (QC → HookNet → windows → HE2CELL →
analysis → report), then 4a's burden and survival, and archives to `outputs/tcga_<CODE>_tls_<date>/`
(cells.csv.gz per slide, analysis/, report, summary.json). Frozen ESCA parameters throughout; a cohort-specific
re-tune is reported separately only as a sensitivity analysis. Suggested cohorts with known TLS prognostic
signal: LUAD, LUSC, BLCA, HNSC, STAD, SKCM. Avoid COAD/READ as a generalisation claim (training domain).
Disk: ~134 GB free on SSD2; a cohort of 150 slides needs ~60–80 GB raw; remove raw slides after archiving cells.

Deliverables: `deployment/esca_tls/esca_survival.py`, `deployment/esca_tls/esca_wsi_burden.py`,
`deployment/tcga_cohort/run_cohort.py` (wrapper), `outputs/esca_survival_<date>/Survival_report.html`.

---

## 5. Maturity: try to fix it  (ready now)

Current state: B share, B-core enrichment and B density do not separate mature from early TLS on either cohort
(AUROC 0.50 for HookNet-GC on ESCA; 0.56–0.57 for pathologist MTLS/GC vs ETLS on Visium).

### 5a. Germinal-centre morphology score (first attempt, no new labels)
Compute inside each accepted aggregate from the slide image and the cell table:
- pale-centre contrast: haematoxylin OD (colour deconvolution) smoothed at 8 µm; score = (median OD in mantle
  ring) − (minimum OD of a ≥60 µm-diameter interior region); GCs are paler than the mantle.
- nuclear size gradient: median Cellpose nuclear area in the interior vs the mantle (centroblasts are larger).
- radial B profile: B density in concentric rings; a mantle peak with an interior dip indicates a follicle.
- cell-density dip: nuclei/mm² interior vs mantle.
Combine by logistic regression trained on ESCA dev aggregates with HookNet GC pixels as a weak label (proxy only),
freeze, then test on (i) ESCA test aggregates vs HookNet GC and (ii) Visium TLS objects labelled mature
(any MTLS or GC spot) vs early (ETLS only) — the pathologist labels are the real test. Also test each feature alone.
Acceptance: Visium AUROC mature vs early ≥ 0.75 with CI lower bound > 0.6; ESCA ≥ 0.70. Otherwise 5b.

### 5b. If 5a fails: GC-B class or stated limitation
A GC-B class needs labels (Ki67/BCL6/CD21 or pathologist-marked GCs); it can be trained only when item 2 data
or the item 1 SFL reads exist. If neither is available, the papers' maturity section becomes a stated
limitation: "composition and density grade presence, not maturity; GC-aware features were tested and did not
reach …", with the 5a numbers.

Deliverables: `deployment/esca_tls/esca_maturity.py`, `deployment/visium_tls/visium_maturity.py`,
`outputs/maturity_<date>/Maturity_report.html`; if adopted, the score is added to `two_stage_proposals.csv`
and to item 1's maturity analysis.

---

## Execution order and agents

| order | item | needs from you | agent model | session |
|---|---|---|---|---|
| now | 3 (stats + ablations) | nothing | Opus 5.5 | own session; runs the control fine-tune in background |
| now | 5a (maturity score) | nothing | Opus 5.5 | own session; independent of 3 |
| now | 4a (ESCA survival, incl. WSI burden) | nothing | Opus 5.5 | own session; the WSI burden run is the long job |
| on delivery | 1a packaging | your go-ahead and reviewer names | Opus 5.5 | produces the packages; waits |
| on delivery | 1b analysis | `data/pathologist_review/answers/*.csv` | Opus 5.5 | after 5a (uses its score if adopted) |
| on delivery | 2 | `data/ihc_validation/` | Opus 5.5 | after the leakage check |
| on delivery | 4b | `data/TCGA-<CODE>_raw/` | Opus 5.5 | one session per cohort, sequential (disk, GPU) |

Items 3, 5a and 4a write to disjoint folders and can run at the same time, except that the MPS GPU should hold
one training/segmentation job at a time: 3's control fine-tune and 4a's WSI burden run must be queued, not parallel.
