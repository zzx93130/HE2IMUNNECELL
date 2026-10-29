#!/bin/zsh
# Driver: CPU prep as stage-1 arrays complete; GPU stages (gpu_locked.sh, owner tnbc-frozen-adapt) only while the
# stage-1 queue is not running; one array per lock job; then CPU evaluation.
ROOT=${0:A:h:h:h}
cd $ROOT
OUT=$ROOT/outputs/tnbc_frozen_adapt_20261101
LOG=$OUT/logs; mkdir -p $LOG
PY=$ROOT/.venv/bin/python
S=deployment/tnbc_st/frozen_adapt_step2.py
export GPU_LOCK_OWNER=tnbc-frozen-adapt
queue_busy() { pgrep -f run_stage1.sh >/dev/null || pgrep -f "run_cellcrops.py|model_eval.py|tnbc_cells.py|tnbc_hooknet.py" >/dev/null; }
disk_ok() { (( $(df -g $ROOT | tail -1 | awk '{print $4}') >= 8 )) || { echo "$(date '+%F %T') STOP: disk < 8 GB" >> $LOG/driver.log; exit 1; } }
done_list() { $PY -c "import sys;sys.path.insert(0,'deployment/tnbc_st');import frozen_adapt_common as F;print(' '.join(F.done_arrays()))" 2>/dev/null; }
gpu() { # name, args...
  local name=$1; shift
  until ! queue_busy; do sleep 120; done
  disk_ok
  local t0=$(date +%s)
  deployment/ablations/gpu_locked.sh $PY "$@" >> $LOG/$name.log 2>&1
  echo "$name $t0 $(date +%s) status $?" >> $LOG/gpu_jobs.log
}
# phase 1: prep while the queue runs
while queue_busy; do
  disk_ok
  $PY $S prep --arrays $(done_list) >> $LOG/prep.log 2>&1
  sleep 300
done
sleep 60
$PY $S prep --arrays $(done_list) >> $LOG/prep.log 2>&1
# phase 2: GPU
gpu bn $S bn
for a in $(done_list); do gpu infer_$a $S infer --arrays $a; done
gpu ffpe $S ffpe
# phase 3: evaluation
$PY deployment/tnbc_st/frozen_adapt_eval.py > $LOG/eval.log 2>&1
echo "$(date '+%F %T') driver finished" >> $LOG/driver.log
