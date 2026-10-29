#!/bin/zsh
# HoVer-Net (PanNuke) baseline: inference -> evaluation -> report, chained, resumable, self-contained.
#
#   deployment/baselines/run_hovernet_queue.sh [CPU_WORKERS]      (default 4 CPU workers)
#
# Timed on 2026-10-26 (M3 Ultra, CPU, 4 workers x 4 torch threads, batch 8, single post-processing tile):
# one ESCA window (3328 px) takes 443-499 s per worker and one Visium 1024-px patch ~51 s per worker
# (KC2, 554 patches). Full run = 988 windows + 6835 patches ~= 228 CPU-worker-hours, i.e. ~57 h on 4 workers:
# more than the ~24 h budget. A brief MPS check gave identical cells but was not faster while the GPU was
# shared with the Cellpose job. Therefore:
#   * CPU workers start now, without the GPU lock, on the Visium sections first, then help with ESCA;
#   * one MPS worker queues for the shared GPU lock (outputs/_gpu.lock, via deployment/ablations/gpu_locked.sh:
#     mkdir lock, retry every 60 s, removed when done) and, once it has the lock, works on ESCA (then any
#     Visium left). All workers claim units (ESCA slide / Visium chunk) with mkdir locks, so they never
#     duplicate work, and finished units are skipped: re-running this script resumes.
#   * When Visium is complete an INTERIM evaluation + report is written (then every 12 h while ESCA runs);
#     when every unit is done, the final one. If the CPU workers finish everything before the GPU lock
#     frees, the MPS waiter is withdrawn.
# Log: outputs/baselines_hovernet_20261026/queue.log (worker logs: workers/).
ROOT=${0:A:h:h:h}
cd $ROOT
NCPU=${1:-4}
TIA=$ROOT/.runtime/tiatoolbox/venv/bin/python
PY=$ROOT/.venv/bin/python
OUT=$ROOT/outputs/baselines_hovernet_20261026
CELLS=$OUT/cells
LOG=$OUT/queue.log
mkdir -p $OUT/workers
exec >> $LOG 2>&1
ts() { date '+%F %T' }

# Single instance: a stale lock (dead pid) is taken over.
QLOCK=$OUT/_queue.lock
if ! mkdir $QLOCK 2>/dev/null; then
  old=$(cat $QLOCK/pid 2>/dev/null)
  if [[ -n $old ]] && kill -0 $old 2>/dev/null; then
    echo "$(ts) queue already running (pid $old); exiting"; exit 0
  fi
  echo "$(ts) taking over stale queue lock (pid $old)"
fi
echo $$ > $QLOCK/pid
trap 'rm -rf $QLOCK' EXIT
echo "$(ts) queue start (pid $$, $NCPU CPU workers + 1 MPS worker under the GPU lock)"

visium_done() { for s in LC1 LC2 LC3 LC4 LC5 KC1 KC2 KC3; do [[ -f $CELLS/visium/$s.csv.gz ]] || return 1; done }
esca_done() {
  local n=$($PY -c "
import pandas as pd, os
w = pd.read_csv('outputs/esca_tls_20261019/analysis_he2cell_ft_tau05/windows.csv')
print(sum(not os.path.exists(f'$CELLS/esca/{b}/{i}.csv.gz') for b, i in zip(w.barcode, w.window_id)))")
  [[ $n == 0 ]]
}
evaluate() {
  echo "$(ts) evaluation ($1) start"
  $PY deployment/baselines/hovernet_eval.py > $OUT/workers/eval_last.log 2>&1 \
    && $PY deployment/baselines/hovernet_report.py >> $OUT/workers/eval_last.log 2>&1
  local rc=$?
  grep -E "^progress|^wrote|Error|Traceback" $OUT/workers/eval_last.log
  echo "$(ts) evaluation ($1) finished with status $rc"
}

# CPU workers: Visium first, then ESCA.
CPU_PIDS=()
for i in $(seq 1 $NCPU); do
  OMP_NUM_THREADS=4 $TIA deployment/baselines/hovernet_cells.py visium esca --device cpu --threads 4 \
    --batch-size 8 --postproc-workers 2 >> $OUT/workers/cpu$i.log 2>&1 &
  CPU_PIDS+=($!)
  sleep 30
done
echo "$(ts) CPU workers: $CPU_PIDS"

# MPS worker: waits for the shared GPU lock (waiting lines thinned to one per hour), then ESCA first.
GPU_LOCK_OWNER=baselines-hovernet deployment/ablations/gpu_locked.sh \
  $TIA deployment/baselines/hovernet_cells.py esca visium --device mps --threads 4 --batch-size 8 --postproc-workers 4 \
  > >(awk '/waiting for/ { if (++n % 60 != 1) next } { print; fflush() }' >> $OUT/workers/mps.log) 2>&1 &
MPS=$!
echo "$(ts) MPS worker (lock waiter) pid $MPS"

# Interim evaluations: once when Visium completes, then every 12 h until all CPU workers have exited.
interim_visium=0; last=0
while true; do
  alive=0; for p in $CPU_PIDS; do kill -0 $p 2>/dev/null && alive=1; done
  (( alive )) || break
  if (( ! interim_visium )) && visium_done; then
    interim_visium=1; last=$(date +%s); evaluate "interim, Visium complete"
  elif (( interim_visium && $(date +%s) - last > 43200 )); then
    last=$(date +%s); evaluate "interim"
  fi
  sleep 600
done
echo "$(ts) CPU workers finished"
if visium_done && esca_done; then
  if ! grep -q "lock taken" $OUT/workers/mps.log 2>/dev/null; then
    echo "$(ts) all units done on CPU; withdrawing the MPS lock waiter"; kill $MPS 2>/dev/null
  fi
fi
wait $MPS 2>/dev/null
echo "$(ts) MPS worker finished"
# A worker can die (e.g. reboot) with units unclaimed: finish them on CPU before the final evaluation.
if ! (visium_done && esca_done); then
  echo "$(ts) units left after the workers exited; finishing on CPU"
  OMP_NUM_THREADS=4 $TIA deployment/baselines/hovernet_cells.py visium esca --device cpu --threads 4 >> $OUT/workers/cpu_final.log 2>&1
fi
if visium_done && esca_done; then evaluate final; else evaluate "INCOMPLETE after all workers"; fi
echo "$(ts) queue done"
