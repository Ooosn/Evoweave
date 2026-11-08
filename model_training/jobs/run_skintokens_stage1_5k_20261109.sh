#!/usr/bin/env bash
# SLOTS: 8
set -euo pipefail
[[ "${JOB_ID:-}" == 131942123 && "${NSLOTS:-}" == 16 && "$(hostname -s)" == pcg01 ]]
[[ "$(ulimit -v)" == unlimited ]]
source /home/geadmin/N1GE/default/common/settings.sh
source ~/1.sh
[[ -f "$RUNNER_ROOT/done/skintokens_static_5k_20261109_train.sh" ]]
cd /work/wangyy/tokenrig_alignment_20261105/code_skintokens_stage1_5k_20261109
export PYTHONDONTWRITEBYTECODE=1 GIT_TERMINAL_PROMPT=0
exec /home/wangyy/miniconda3/envs/mygs/bin/python -B server_ops/hgc/chain_skintokens_stage1_20261109.py run
