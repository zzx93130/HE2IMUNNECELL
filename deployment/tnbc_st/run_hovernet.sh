#!/bin/zsh
# HoVer-Net (CPU) workers for the pre-registered 20-array TNBC subset; they poll until HE2CELL has each array's patches.
ROOT=${0:A:h:h:h}
cd $ROOT
SUBSET=($(.venv/bin/python -c "import json;print(' '.join(json.load(open('outputs/tnbc_st_20261101/preregistration.json'))['amendments'][0]['subset']))"))
mkdir -p outputs/tnbc_st_20261101/logs
for i in $(seq 1 ${1:-6}); do
  nohup .runtime/tiatoolbox/venv/bin/python -B deployment/tnbc_st/tnbc_hovernet.py --wait --threads 4 --arrays $SUBSET \
    > outputs/tnbc_st_20261101/logs/hovernet_w$i.log 2>&1 &
  sleep 5
done
