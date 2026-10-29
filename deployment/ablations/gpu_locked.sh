#!/bin/zsh
# Run one GPU job under the shared MPS lock (outputs/_gpu.lock, a directory): wait, retrying every
# 60 s, while another session holds it; release it when the job ends, whatever its exit status.
# GPU_LOCK_OWNER sets the label written to $LOCK/owner (default: item3-ablations).
ROOT=${0:A:h:h:h}
LOCK=$ROOT/outputs/_gpu.lock
until mkdir "$LOCK" 2>/dev/null; do
  echo "$(date '+%F %T') waiting for $LOCK ($(cat $LOCK/owner 2>/dev/null))"; sleep 60
done
echo "${GPU_LOCK_OWNER:-item3-ablations} pid $$: $*" > "$LOCK/owner"
trap 'rm -rf "$LOCK"' EXIT INT TERM
echo "$(date '+%F %T') lock taken: $*"
"$@"
rc=$?
echo "$(date '+%F %T') job finished with status $rc"
exit $rc
