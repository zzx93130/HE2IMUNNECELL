#!/bin/zsh
# Item-3 GPU queue, run after the weak-augmentation fine-tune. Every step takes outputs/_gpu.lock on its
# own (Visium: one section per lock) so other sessions' jobs can interleave; CPU steps follow each one.
cd ${0:A:h:h:h}
PY=.venv/bin/python
LOCKED=deployment/ablations/gpu_locked.sh
FT=outputs/classifier_finetune_20261019/best_finetuned.pth
WEAK=outputs/classifier_finetune_weakaug_20261025/best_finetuned.pth
PRIOR=outputs/ablations_20261025/priors/prior_tau05.json
set -e
[[ -f $WEAK ]] || { echo "missing $WEAK"; exit 1; }
# ESCA: the kept he2cell_ft crops, no re-segmentation; prior tau 0.5 applied afterwards as for the adopted run.
$LOCKED $PY deployment/esca_tls/esca_rescore.py --classifier $FT --result he2cell_ft_ttanone --tta none
$PY deployment/esca_tls/esca_prior_adjust.py --source he2cell_ft_ttanone --result he2cell_ft_ttanone_tau05 --prior-json $PRIOR
ESCA_RESULT=he2cell_ft_ttanone_tau05 $PY deployment/esca_tls/esca_analysis.py > outputs/ablations_20261025/esca_ttanone.log 2>&1 &
$LOCKED $PY deployment/esca_tls/esca_rescore.py --classifier $WEAK --result he2cell_ftweak
$PY deployment/esca_tls/esca_prior_adjust.py --source he2cell_ftweak --result he2cell_ftweak_tau05 --prior-json $PRIOR
ESCA_RESULT=he2cell_ftweak_tau05 $PY deployment/esca_tls/esca_analysis.py > outputs/ablations_20261025/esca_weakaug.log 2>&1 &
# Visium: crops of the v2 run were not kept, so one re-segmentation scored by all needed checkpoints
# (identity view of the same D4 pass = TTA none).
for s in LC1 LC2 LC3 LC4 LC5 KC1 KC2 KC3; do
  VISIUM_RUN=ablate $LOCKED $PY deployment/visium_tls/visium_cells.py --samples $s \
    --classifiers finetuned finetuned_weakaug original --identity
done
wait
echo "queue done $(date '+%F %T')"
