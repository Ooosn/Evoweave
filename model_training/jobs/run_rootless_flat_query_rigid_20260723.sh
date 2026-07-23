#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
export JOB_MOTION_ALIGNMENT_POLICY=query_rigid
export JOB_RUN_NAME="${JOB_RUN_NAME:-flat_query_rigid_20260723}"
exec bash "${SCRIPT_DIR}/run_rootless_flat_unirig_motion_baseline_20260706.sh"
