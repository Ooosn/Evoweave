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
wheel=$root/flash_build_headers/flash_attn-2.8.3-cp311-cp311-linux_x86_64.whl
[[ "$(sha256sum "$wheel" | cut -d' ' -f1)" == 431729697a4191c33a1189a4e96884905d4e754af6ccdd35c010338d535cf21b ]]
CUDA_VISIBLE_DEVICES='' "$py" -B -c 'import torch, flash_attn; assert torch.__version__=="2.7.0+cu128" and flash_attn.__version__=="2.8.3" and not torch.cuda.is_initialized()'
CUDA_VISIBLE_DEVICES='' "$py" -B -c 'import bpy; assert bpy.app.version[:2]==(4,3)'
cd "$code"
"$py" -B model_training/tools/agent_work_guard.py begin >/dev/null
"$py" -B model_training/tools/agent_work_guard.py check --operation matched_eval --entrypoint "$entry" >/dev/null
exec "$py" -B "$entry" supervise
