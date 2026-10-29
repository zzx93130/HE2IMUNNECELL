#!/bin/zsh
# Read-only capture of the stage-1 Cellpose label masks and mask tables (the other agent deletes its temp dir per
# array; the classifier needs the mask channel, so crops cannot be rebuilt from centroids alone).
ROOT=${0:A:h:h:h}
SRC=$ROOT/outputs/tnbc_st_20261101/_tmp_cells
DST=$ROOT/outputs/tnbc_frozen_adapt_20261101/seg_capture
while true; do
  for d in $SRC/*(N/); do
    a=${d:t}
    [[ -d $d/cells ]] || continue
    mkdir -p $DST/$a
    rsync -a --exclude single_cells $d/cells/patch_masks $d/cells/mask_tables $DST/$a/ 2>/dev/null
    cp $d/${a}_patch_index.csv $DST/$a/ 2>/dev/null
  done
  sleep 45
done
