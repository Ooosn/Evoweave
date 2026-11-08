#!/usr/bin/env bash
# SLOTS: 8
set -euo pipefail
[[ "${JOB_ID:-}" == 131942123 && "${NSLOTS:-}" == 16 && "$(hostname -s)" == pcg01 ]]
[[ "$(ulimit -v)" == unlimited ]]
source /home/geadmin/N1GE/default/common/settings.sh
source ~/1.sh
task=$(basename "$0" .sh)
mode=${task##*_}
case "$mode" in audit|capacity|train) ;; *) exit 2 ;; esac
root=/work/wangyy/tokenrig_alignment_20261105/code_skintokens_static_5k_20261109
out=/work/wangyy/tokenrig_alignment_20261105/outputs_skintokens_static_5k_20261109
receipts=/home/wangyy/evorig/outputs/tokenrig_alignment_20261105/skintokens_static_5k_20261109/receipts
python=/home/wangyy/miniconda3/envs/mygs/bin/python
cd "$root"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$receipts"
[[ ! -e "$receipts/$mode.log" ]]
"$python" -B model_training/tools/agent_work_guard.py begin >/dev/null
finish() {
    local rc=$?
    if [[ -f "$out/${mode}_resources.json" ]]; then
        cp --no-clobber "$out/${mode}_resources.json" "$receipts/${mode}_resources.json"
    fi
    if [[ -f "$out/$mode/result.json" ]]; then
        cp --no-clobber "$out/$mode/result.json" "$receipts/${mode}_result.json"
    fi
    return "$rc"
}
trap finish EXIT
"$python" -B model_training/jobs/launch_virtual_pair.py --mode "$mode" \
    --config "$root/model_training/experiments/skintokens_static_5k_20261109.json" \
    --output "$out/$mode" >"$receipts/$mode.log" 2>&1
