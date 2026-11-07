#!/usr/bin/env bash
# SLOTS: 8
set -euo pipefail
root=/work/wangyy/tokenrig_alignment_20261105
code=$root/code_native_cont400k_visual12_20261107
source /home/geadmin/N1GE/default/common/settings.sh
source ~/1.sh
[[ "${JOB_ID:-}" == 131942123 && "${NSLOTS:-}" == 16 ]]
[[ "$(ulimit -v)" == unlimited ]]
[[ -f "$RUNNER_ROOT/done/tokenrig_native_cont400k_20261107_train.sh" ]]
cd "$code"
export PYTHONDONTWRITEBYTECODE=1
exec /home/wangyy/miniconda3/envs/mygs/bin/python -B server_ops/hgc/chain_tokenrig_native_followup_20261108.py
