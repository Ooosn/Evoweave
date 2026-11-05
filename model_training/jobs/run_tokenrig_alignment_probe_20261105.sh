#!/usr/bin/env bash
# SLOTS: 8
set -euo pipefail
root=/work/wangyy/tokenrig_alignment_20261105
code=$root/code_probe_r4
py=/home/wangyy/miniconda3/envs/mygs/bin/python
source /home/geadmin/N1GE/default/common/settings.sh
source ~/1.sh
[[ "${JOB_ID:-}" == 131351736 && "${NSLOTS:-}" == 16 ]]
[[ "$(ulimit -v)" == unlimited ]]
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
[[ ! -e "$root/outputs_probe_r4" ]]
cd "$code"
export PYTHONDONTWRITEBYTECODE=1
"$py" -B model_training/tools/agent_work_guard.py begin > "$root/context_probe_r4.log"
config="$code/model_training/experiments/tokenrig_alignment_probe_20261105_r4.json"
for mode in audit capacity; do
    printf '%s %s\n' "$(date -Ins)" "$mode"
    "$py" -B model_training/jobs/launch_virtual_pair.py --mode "$mode" \
        --config "$config" --output "$root/outputs_probe_r4/$mode" > "$root/probe_r4_$mode.log" 2>&1
done
tar -czf /home/wangyy/evorig/outputs/tokenrig_alignment_20261105/probe_r4_receipts.tar.gz \
    -C "$root" outputs_probe_r4 probe_r4_audit.log probe_r4_capacity.log
