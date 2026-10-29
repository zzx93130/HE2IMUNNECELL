#!/bin/zsh
# Stage 1 (+ HookNet-TLS) queue for the TNBC_ST arrays listed in $1 (one array id per line), in order.
# Each array is one GPU job under the shared MPS lock (gpu_locked.sh, owner tnbc-st): HE2CELL cells, then HookNet-TLS.
# Resumable: finished arrays are skipped by both scripts; stops when free disk < 8 GB (checked by the scripts).
ROOT=${0:A:h:h:h}
LIST=${1:-$ROOT/outputs/tnbc_st_20261101/stage1_order.txt}
LOG=$ROOT/outputs/tnbc_st_20261101/logs
mkdir -p $LOG
export GPU_LOCK_OWNER=tnbc-st
for array in $(cat $LIST); do
  if [[ -f $ROOT/outputs/tnbc_st_20261101/cells/$array/cells.csv.gz && -f $ROOT/outputs/tnbc_st_20261101/hooknet/$array/summary.json ]]; then
    continue
  fi
  free=$(df -g $ROOT | tail -1 | awk '{print $4}')
  if (( free < 8 )); then echo "$(date '+%F %T') STOP: ${free} GB free"; break; fi
  $ROOT/deployment/ablations/gpu_locked.sh zsh -c "cd $ROOT && .venv/bin/python deployment/tnbc_st/tnbc_cells.py --arrays $array && .runtime/hooknet_tls/venv/bin/python deployment/tnbc_st/tnbc_hooknet.py --arrays $array" >> $LOG/stage1.log 2>&1
  echo "$(date '+%F %T') $array done (status $?)" >> $LOG/stage1.log
done
echo "$(date '+%F %T') queue finished" >> $LOG/stage1.log
