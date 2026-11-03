#!/usr/bin/env bash
# SLOTS: 8
set -euo pipefail
root=/work/wangyy/tokenrig_full_20261104
code=$root/code
py=/home/wangyy/miniconda3/envs/mygs/bin/python
source /home/geadmin/N1GE/default/common/settings.sh
source ~/1.sh
[[ "${JOB_ID:-}" == 131351736 && "${NSLOTS:-}" == 16 ]]
[[ "$(ulimit -v)" == unlimited ]]
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
cd "$code"
export PYTHONDONTWRITEBYTECODE=1
"$py" -B model_training/tools/agent_work_guard.py begin > "$root/context.log"
config="$code/model_training/experiments/tokenrig_full_20261104.json"
for stage in audit train; do
    printf '%s %s\n' "$(date -Ins)" "$stage"
    "$py" -B "$code/model_training/jobs/launch_virtual_pair.py" \
        --mode "$stage" --config "$config" --output "$root/outputs/$stage" \
        > "$root/${stage}.log" 2>&1
done
