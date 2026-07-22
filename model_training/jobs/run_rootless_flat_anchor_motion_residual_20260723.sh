#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
export JOB_CONDITION_FUSION=anchor_motion_residual_zero
export JOB_RUN_NAME="${JOB_RUN_NAME:-flat_anchor_motion_residual_20260723}"
exec bash "${SCRIPT_DIR}/run_rootless_flat_static_motion_residual_20260723.sh"
