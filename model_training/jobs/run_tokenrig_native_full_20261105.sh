#!/usr/bin/env bash
# SLOTS: 8
set -euo pipefail
root=/work/wangyy/tokenrig_alignment_20261105
code=$root/code_native_train
py=/home/wangyy/miniconda3/envs/mygs/bin/python
source /home/geadmin/N1GE/default/common/settings.sh
source ~/1.sh
[[ "${JOB_ID:-}" == 131351736 && "${NSLOTS:-}" == 16 ]]
[[ "$(ulimit -v)" == unlimited ]]
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
[[ ! -e "$root/outputs_native_train" ]]
cd "$code"
export PYTHONDONTWRITEBYTECODE=1
"$py" -B model_training/tools/agent_work_guard.py begin > "$root/context_native_train.log"
config="$code/model_training/experiments/tokenrig_native_full_20261105.json"
trap 'tar -czf /home/wangyy/evorig/outputs/tokenrig_alignment_20261105/native_train_receipts.tar.gz --exclude="*.pt" --exclude="*.pt.tmp" -C "$root" outputs_native_train native_train.log' EXIT
printf '%s fresh_native_train\n' "$(date -Ins)"
"$py" -B model_training/jobs/launch_virtual_pair.py --mode train \
    --config "$config" --output "$root/outputs_native_train/train" > "$root/native_train.log" 2>&1
