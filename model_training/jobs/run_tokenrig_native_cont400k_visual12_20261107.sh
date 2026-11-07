#!/usr/bin/env bash
# SLOTS: 8
set -euo pipefail
root=/work/wangyy/tokenrig_alignment_20261105
phase=tokenrig_native_cont400k_visual12_20261107
code=$root/code_native_cont400k_visual12_20261107
out=$root/outputs_native_cont400k_visual12_20261107
durable=/home/wangyy/evorig/outputs/tokenrig_alignment_20261105/native_cont400k_20261107/visual12
py=/home/wangyy/miniconda3/envs/mygs/bin/python
case "$(basename -- "$0")" in
    tokenrig_native_cont400k_visual12_20261107_audit.sh) mode=audit ;;
    tokenrig_native_cont400k_visual12_20261107_infer.sh) mode=capacity ;;
    *) exit 2 ;;
esac
source /home/geadmin/N1GE/default/common/settings.sh
source ~/1.sh
[[ "${JOB_ID:-}" == 131942123 && "${NSLOTS:-}" == 16 ]]
[[ "$(ulimit -v)" == unlimited ]]
gpus=GPU-1fb4221b-91b0-0df6-8615-b36fbdd55bf7,GPU-0544830a-4dac-9fe3-a099-95ddc2240b72
apps=$(nvidia-smi -i "$gpus" --query-compute-apps=pid --format=csv,noheader)
[[ -z "$apps" ]]
cd "$code"
export PYTHONDONTWRITEBYTECODE=1
"$py" -B model_training/tools/agent_work_guard.py begin > "$root/context_${phase}_$mode.log"
rc=0
"$py" -B model_training/jobs/launch_virtual_pair.py --mode "$mode" \
    --config "$code/model_training/experiments/$phase.json" --output "$out/$mode" \
    > "$root/${phase}_$mode.log" 2>&1 || rc=$?
mkdir -p "$durable/receipts"
cp -n "$root/${phase}_$mode.log" "$durable/receipts/$mode.log"
if [[ -f "$out/${mode}_resources.json" ]]; then
    cp -n "$out/${mode}_resources.json" "$durable/receipts/${mode}_resources.json"
fi
if [[ "$mode" == audit && -f "$out/audit/result.json" ]]; then
    cp -n "$out/audit/result.json" "$durable/receipts/audit_result.json"
    cp -n "$out/audit/development_matrix.json" "$durable/receipts/screening.json"
fi
exit "$rc"
