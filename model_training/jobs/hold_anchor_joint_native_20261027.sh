#!/usr/bin/env bash
set -euo pipefail
source /home/geadmin/N1GE/default/common/settings.sh
source ~/1.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ulimit -v unlimited
ROOT=/work/wangyy/virtual_query_matched_5k_20261025
test "$(git -C "$ROOT/code" rev-parse HEAD)" = "$1"
test -z "$(git -C "$ROOT/code" status --porcelain)"
test "$NSLOTS" = 16
exec /home/wangyy/miniconda3/envs/mygs/bin/python -B \
    "$ROOT/code/model_training/jobs/hold_virtual_pair_allocation.py" \
    --directory "$ROOT/allocation_binding_native_20261027"
