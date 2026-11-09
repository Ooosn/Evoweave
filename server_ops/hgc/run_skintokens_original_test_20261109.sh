#!/usr/bin/env bash
# SLOTS: 8
set -euo pipefail
source /home/geadmin/N1GE/default/common/settings.sh
source ~/1.sh
export GIT_TERMINAL_PROMPT=0 PYTHONDONTWRITEBYTECODE=1
root=/work/wangyy/skintokens_original_test_20261109
code=$root/code
py=$root/venv/bin/python
entry=server_ops/hgc/run_skintokens_original_test_20261109.py
[[ "${JOB_ID:-}" == 131942123 && "${NSLOTS:-}" == 16 && "$(ulimit -v)" == unlimited ]]
[[ -f "$RUNNER_ROOT/done/skintokens_original_flash_headers_20261109.sh" ]]
cd "$code"
"$py" -B model_training/tools/agent_work_guard.py begin >/dev/null
"$py" -B model_training/tools/agent_work_guard.py check --operation matched_eval --entrypoint "$entry" >/dev/null
exec "$py" -B "$entry" supervise
