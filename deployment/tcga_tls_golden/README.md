# Frozen TCGA TLS region evaluation

Inputs: `/Volumes/One Touch/TCGA_TLS_golden`. Outputs:
`outputs/tcga_tls_golden_20261102/`.

The checked annotation release is Zenodo 10614928, MD5
144b0f70df48eb8ea975f84c89f7dec3. Use original-SVS-coordinate XMLs,
not approximate-0.5-µm XMLs, thumbnails, or unverified raster masks.
The original release contains 375 LUSC, 345 BLCA, and 299 KIRC slides.
One LUSC image, TCGA-33-4587-01Z-00-DX1, is missing locally.

The primary method is the deployed fine-tuned classifier with its tau=0.5
prior correction and D4 averaging, followed by the frozen two-stage detector.
Baselines are frozen predicted-B density and all-nuclei density.
All settings come from the saved ESCA chosen_parameters JSON, not README prose.
No threshold is fitted to this cohort. No HookNet comparison is claimed:
these human annotations were used in HookNet development.

This is **annotation-guided region evaluation**, not blind whole-slide
screening. Every TLS and explicit negative polygon is processed, with 128 µm
context, at 0.25 µm/px. GC polygons are nested annotations, not extra TLS objects.
The global 768-pixel core grid and 1024-pixel patches avoid duplicate nuclei.
An annotation is detected when at least 10% of its non-conflicting area is
covered by a predicted region. This measures regional recovery, not one-to-one
object matching. Pixel counts are retained separately. TLS/negative overlap
pixels are excluded. Negative labels are other, inflammation, tumor and lymph
node; label-specific records are retained for stratified analysis.

Empty annotated slides are recorded but not scored, as whole-slide inference
is not performed. Whole-slide specificity, precision, object F1, and false
positives per slide cannot be inferred from these outputs. An unbiased
whole-slide benchmark must additionally run the deployed slide screen on all
tissue, including the empty slides, and assess annotation completeness.

`manifest.json`, `protocol.json`, and `audit.json` record matching and frozen
settings. Slide order is a fixed barcode hash, independent of model output.
`REPORT.md` and `summary.json` update after completed slides. Interim intervals
are descriptive, with 2,000 patient-cluster resamples; a single patient has no
reported interval. Do not interpret sequential interim results as confirmatory.

KIRC uses indexed reads from the uncompressed tar, staging one SVS at a time
and deleting the staged copy after a completed slide. Source files are never
modified. At least 8 GB of workspace free space is required. Cell batches are
saved after 64 patches, enabling restart without repeating completed batches.
The shared GPU lock prevents overlap with other cooperating project jobs.

Run from the repository root with Apple GPU access:

```sh
.venv/bin/python deployment/tcga_tls_golden/evaluate.py audit
.venv/bin/python -m unittest discover -s deployment/tcga_tls_golden -p 'test_*.py' -v
.venv/bin/python deployment/tcga_tls_golden/evaluate.py run
.venv/bin/python deployment/tcga_tls_golden/evaluate.py report
```

Do not rerun `audit` or change frozen code/settings mid-evaluation. Inference
stops on errors instead of silently excluding a patient. Inspect per-slide
`error.json` and `pipeline.log` before resuming. The current process must finish
or be stopped cleanly before restarting; never remove a live GPU lock.
