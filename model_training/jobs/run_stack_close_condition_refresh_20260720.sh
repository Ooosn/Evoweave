#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"

export JOB_RUN_NAME="${JOB_RUN_NAME:-stack_close_condition_refresh_20260720}"
export JOB_CONDITION_REFRESH_LAYERS="${JOB_CONDITION_REFRESH_LAYERS:-7,15,23}"
export JOB_CONDITION_REFRESH_DIM="${JOB_CONDITION_REFRESH_DIM:-256}"
export JOB_CONDITION_REFRESH_HEADS="${JOB_CONDITION_REFRESH_HEADS:-8}"
export JOB_LR_REFRESH="${JOB_LR_REFRESH:-0.0001}"

if [[ "${JOB_CONDITION_REFRESH_LAYERS}" != "7,15,23" ]]; then
  echo "[stack_close_refresh] ERROR: formal route requires layers 7,15,23" >&2
  exit 2
fi
if [[ "${JOB_CONDITION_REFRESH_DIM}" != "256" ]]; then
  echo "[stack_close_refresh] ERROR: formal route requires refresh dim 256" >&2
  exit 2
fi
if [[ "${JOB_CONDITION_REFRESH_HEADS}" != "8" ]]; then
  echo "[stack_close_refresh] ERROR: formal route requires 8 refresh heads" >&2
  exit 2
fi

exec bash "${SCRIPT_DIR}/run_stack_close_sibling_perturb_20260720.sh"
