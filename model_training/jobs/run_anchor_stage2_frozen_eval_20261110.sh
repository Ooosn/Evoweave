#!/usr/bin/env bash
# SLOTS: 8
set -euo pipefail
[[ "${JOB_ID:-}" == 131942123 && "${NSLOTS:-}" == 16 && "$(hostname -s)" == pcg01 ]]
[[ "$(ulimit -v)" == unlimited ]]
source /home/geadmin/N1GE/default/common/settings.sh
source ~/1.sh
cd /work/wangyy/tokenrig_alignment_20261105/code_anchor_stage2_frozen_eval_20261110
export PYTHONDONTWRITEBYTECODE=1 GIT_TERMINAL_PROMPT=0
exec /home/wangyy/miniconda3/envs/mygs/bin/python -B server_ops/hgc/chain_anchor_stage2_20261109.py \
    --config model_training/experiments/anchor_stage2_frozen_eval_20261110.json
