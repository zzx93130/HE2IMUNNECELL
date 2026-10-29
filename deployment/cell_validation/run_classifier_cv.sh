#!/bin/zsh
# 4-fold patient-level cross-validation of the classifier recipe (deployment/cell_validation/classifier_cv.py).
#
#   deployment/cell_validation/run_classifier_cv.sh        (resumable; re-run to continue after an interruption)
#
# 1. folds.json; 2. the four fold trainings under the shared GPU lock (outputs/_gpu.lock via
#    deployment/ablations/gpu_locked.sh, owner "classifier-cv"; ONE hold for all four folds, ~35 min each on MPS;
#    "waiting" lines are thinned to one per hour); 3. prior sidecars, crop index, verification, held-out
#    inference (mps, no lock: classification-only inference shares the GPU, as cell_validation does), metrics;
#    4. the HTML report. Log: outputs/classifier_cv_20261026/cv_run.log.
ROOT=${0:A:h:h:h}
cd $ROOT
OUT=$ROOT/outputs/classifier_cv_20261026
mkdir -p $OUT
exec >> $OUT/cv_run.log 2>&1
PY=$ROOT/.venv/bin/python
CV=$ROOT/deployment/cell_validation/classifier_cv.py
ts() { date '+%F %T' }
echo "$(ts) run start (pid $$)"
$PY $CV folds || exit 1
GPU_LOCK_OWNER=classifier-cv $ROOT/deployment/ablations/gpu_locked.sh $PY $CV train \
  > >(awk '/waiting for/ { if (++n % 60 != 1) next } { print; fflush() }') 2>&1
rc=$?
echo "$(ts) training stage finished with status $rc (lock released)"
[[ $rc == 0 ]] || exit $rc
$PY $CV prior index verify infer metrics || { echo "$(ts) evaluation failed"; exit 1; }
$PY $ROOT/deployment/cell_validation/classifier_cv_report.py
echo "$(ts) run done"
