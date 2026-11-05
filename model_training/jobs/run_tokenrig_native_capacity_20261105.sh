#!/usr/bin/env bash
# SLOTS: 8
set -euo pipefail
root=/work/wangyy/tokenrig_alignment_20261105
code=$root/code_native_capacity
py=/home/wangyy/miniconda3/envs/mygs/bin/python
source /home/geadmin/N1GE/default/common/settings.sh
source ~/1.sh
[[ "${JOB_ID:-}" == 131351736 && "${NSLOTS:-}" == 16 ]]
[[ "$(ulimit -v)" == unlimited ]]
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
[[ ! -e "$root/outputs_native_capacity" ]]
cd "$code"
export PYTHONDONTWRITEBYTECODE=1
"$py" -B model_training/tools/agent_work_guard.py begin > "$root/context_native_capacity.log"
config="$code/model_training/experiments/tokenrig_native_capacity_20261105.json"
archive() {
    tar -czf /home/wangyy/evorig/outputs/tokenrig_alignment_20261105/native_capacity_receipts.tar.gz \
        --exclude='*.pt' --exclude='*.pt.tmp' -C "$root" \
        outputs_native_capacity native_audit.log native_capacity.log
}
trap archive EXIT
for mode in audit capacity; do
    printf '%s %s\n' "$(date -Ins)" "$mode"
    "$py" -B model_training/jobs/launch_virtual_pair.py --mode "$mode" \
        --config "$config" --output "$root/outputs_native_capacity/$mode" > "$root/native_$mode.log" 2>&1
done
