#!/usr/bin/env bash
# SLOTS: 8
set -euo pipefail
root=/work/wangyy/tokenrig_alignment_20261105
code=$root/code_native_visual12_20261107
py=/home/wangyy/miniconda3/envs/mygs/bin/python
source /home/geadmin/N1GE/default/common/settings.sh
source ~/1.sh
[[ "${JOB_ID:-}" == 131351736 && "${NSLOTS:-}" == 16 ]]
[[ "$(ulimit -v)" == unlimited ]]
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
cd "$code"
export PYTHONDONTWRITEBYTECODE=1
"$py" -B model_training/tools/agent_work_guard.py begin > "$root/context_native_visual12_20261107.log"
config="$code/model_training/experiments/tokenrig_native_visual12_20261107.json"
mode=${EVOWEAVE_VISUAL_MODE:-audit}
[[ "$mode" == audit || "$mode" == capacity ]]
"$py" -B model_training/jobs/launch_virtual_pair.py --mode "$mode" --config "$config" \
  --output "$root/outputs_native_visual12_20261107/$mode" > "$root/native_visual12_20261107_$mode.log" 2>&1
