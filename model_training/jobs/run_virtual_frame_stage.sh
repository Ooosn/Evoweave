#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 5 || ! $1 =~ ^[a-z0-9_-]+$ ]]; then
    printf 'usage: run_virtual_frame_stage.sh LABEL MODE PLAN CANDIDATE GPU [GPU]\n' >&2
    exit 2
fi
label=$1
mode=$2
plan=$3
candidate=$4
shift 4
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
job_root=/home/wangyy/jobs/shared_virtual_frame_20261012
mkdir -p "$job_root/logs" "$job_root/results"
trap 'code=$?; printf "%s\n" "$code" > "$job_root/results/$label.exit"' EXIT
ulimit -v unlimited
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo/model_training/rigweave/src:$repo/model_training/rigweave/scripts"
/home/wangyy/miniconda3/envs/mygs/bin/python -B \
    "$repo/model_training/rigweave/scripts/run_virtual_frame_experiment.py" run \
    --config "$repo/model_training/experiments/shared_virtual_frame_20261012.json" \
    --mode "$mode" --plan "$plan" --candidate "$candidate" --label "$label" --devices "$@" \
    >> "$job_root/logs/${label}_controller.log" 2>&1
