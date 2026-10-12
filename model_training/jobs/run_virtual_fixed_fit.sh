#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
job=/home/wangyy/jobs/shared_virtual_frame_20261012
label=fixed_pair_step_scale_v1
py=/home/wangyy/miniconda3/envs/mygs/bin/python
source "$HOME/1.sh"
case "$(hostname)" in pcg01|pcg01i) ;; *) exit 2 ;; esac
qstat -j 129547842 | grep -Eq 'granted_devices.*pcg01i: /dev/nvidia5 /dev/nvidia7'
while read -r used; do
    [[ $used =~ ^[0-9]+$ && $used -le 256 ]] || exit 3
done < <(nvidia-smi -i 5,7 --query-gpu=memory.used --format=csv,noheader,nounits)
test -z "$(git -C "$repo" status --porcelain)"
test ! -e "$job/logs/$label.log"
exec > "$job/logs/$label.log" 2>&1
trap 'code=$?; printf "%s\n" "$code" > "$job/results/$label.exit"' EXIT
ulimit -v unlimited
export CUDA_VISIBLE_DEVICES=5,7 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export EVOWEAVE_UNIRIG_ROOT=/home/wangyy/evorig/model_assets/UniRig
export EVOWEAVE_OPT_CONFIG_ROOT=/home/wangyy/evorig/model_assets/hf_configs/facebook-opt-350m
export RIGWEAVE_DISABLE_OPEN3D=1 RIGWEAVE_DISABLE_LIGHTNING_IMPORT=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$repo/model_training/rigweave/src:$repo/model_training/rigweave/scripts:$EVOWEAVE_UNIRIG_ROOT:/home/wangyy/evorig/runtime/puppeteer_py310_hgc"
export PATH="/home/wangyy/miniconda3/envs/mygs/bin:$PATH"
cd "$repo"
date -Is
"$py" -B model_training/tools/agent_work_guard.py begin > "$job/logs/${label}_begin.log"
"$py" -B model_training/tools/agent_work_guard.py check --operation preflight > "$job/logs/${label}_guard.log"
"$py" -B -m torch.distributed.run --standalone --nproc_per_node=2 \
    model_training/rigweave/scripts/diagnose_virtual_fitting.py \
    --config "$repo/model_training/experiments/virtual_fixed_step_scale_20261012.json"
date -Is
