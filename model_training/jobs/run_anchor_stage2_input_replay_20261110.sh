#!/usr/bin/env bash
# SLOTS: 8
set -euo pipefail
[[ "${JOB_ID:-}" == 131942123 && "${NSLOTS:-}" == 16 && "$(hostname -s)" == pcg01 ]]
[[ "$(ulimit -v)" == unlimited ]]
source /home/geadmin/N1GE/default/common/settings.sh
source ~/1.sh
scope=anchor_stage2_input_replay_20261110
root=/work/wangyy/tokenrig_alignment_20261105
durable=/home/wangyy/evorig/outputs/tokenrig_alignment_20261105/$scope
output="$root/outputs_$scope/probe"
cd "$root/code_$scope"
export PYTHONDONTWRITEBYTECODE=1 GIT_TERMINAL_PROMPT=0
python=/home/wangyy/miniconda3/envs/mygs/bin/python
[[ ! -e "$durable" && ! -e "$output" ]]
mkdir "$durable"
"$python" -B model_training/tools/agent_work_guard.py begin > "$durable/admission.log"
rc=0
timeout --signal=TERM --kill-after=30s 15m "$python" -B model_training/jobs/launch_virtual_pair.py \
  --mode capacity --config "$PWD/model_training/experiments/$scope.json" --output "$output" \
  > "$durable/probe.log" 2>&1 || rc=$?
for path in "$output"/*.json "$output"_resources.json; do
  if [[ -f "$path" ]]; then
    cp --no-clobber "$path" "$durable/"
    cmp "$path" "$durable/${path##*/}"
  fi
done
exit "$rc"
