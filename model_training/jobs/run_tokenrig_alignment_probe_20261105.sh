#!/usr/bin/env bash
# SLOTS: 8
set -euo pipefail
root=/work/wangyy/tokenrig_alignment_20261105
code=$root/code_probe
py=/home/wangyy/miniconda3/envs/mygs/bin/python
source /home/geadmin/N1GE/default/common/settings.sh
source ~/1.sh
[[ "${JOB_ID:-}" == 131351736 && "${NSLOTS:-}" == 16 ]]
[[ "$(ulimit -v)" == unlimited ]]
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
[[ ! -e "$root/outputs_probe" ]]
cd "$code"
export PYTHONDONTWRITEBYTECODE=1
"$py" -B model_training/tools/agent_work_guard.py begin > "$root/context_probe.log"
config="$code/model_training/experiments/tokenrig_alignment_probe_20261105.json"
for mode in audit capacity; do
    printf '%s %s\n' "$(date -Ins)" "$mode"
    "$py" -B model_training/jobs/launch_virtual_pair.py --mode "$mode" \
        --config "$config" --output "$root/outputs_probe/$mode" > "$root/probe_$mode.log" 2>&1
done
tar -czf /home/wangyy/evorig/outputs/tokenrig_alignment_20261105/probe_receipts.tar.gz \
    -C "$root" outputs_probe probe_audit.log probe_capacity.log
