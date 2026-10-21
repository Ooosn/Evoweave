#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
py=/home/wangyy/miniconda3/envs/mygs/bin/python
mode=${1:?mode required}
config=${2:?configuration path required}
output=${3:?output path required}
jobs=/work/wangyy/structure_xyz_fill_20261022/jobs
label=$(basename "$output")
mkdir -p "$jobs"
test ! -e "$output"
test ! -e "$jobs/$label.log"
test -z "$(git -C "$repo" status --porcelain)"

source "$HOME/1.sh"
export PATH="/home/wangyy/miniconda3/envs/mygs/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export EVOWEAVE_UNIRIG_ROOT=/home/wangyy/evorig/model_assets/UniRig
export EVOWEAVE_OPT_CONFIG_ROOT=/home/wangyy/evorig/model_assets/hf_configs/facebook-opt-350m
export RIGWEAVE_DISABLE_OPEN3D=1 RIGWEAVE_DISABLE_LIGHTNING_IMPORT=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_MODULE_LOADING=LAZY
export PYTHONPATH="$repo/model_training/rigweave/src:$repo/model_training/rigweave/scripts:$EVOWEAVE_UNIRIG_ROOT:/home/wangyy/evorig/runtime/puppeteer_py310_hgc"
ulimit -v unlimited

mapfile -t settings < <("$py" -B -c 'import json,sys; c=json.load(open(sys.argv[1])); r=c["runtime"]; print(r["allocation"]); print(r["host"]); print(",".join(map(str,r["physical_gpus"]))); print(r["world_size"])' "$config")
test "${#settings[@]}" -eq 4
allocation=${settings[0]}
host=${settings[1]}
devices=${settings[2]}
world=${settings[3]}
case "$(hostname)" in "$host"|"${host%i}") ;; *) exit 2 ;; esac

if [[ "$mode" == audit || "$mode" == initialize ]]; then
    export CUDA_VISIBLE_DEVICES=
else
    exec 9> "$jobs/.gpu_work.lock"
    flock -n 9
    qstat -j "$allocation" > "$jobs/${label}_grant.txt"
    "$py" -B - "$config" "$jobs/${label}_grant.txt" <<'PY'
import json, re, sys
runtime = json.load(open(sys.argv[1]))['runtime']
grant = open(sys.argv[2]).read()
lines = [line for line in grant.splitlines() if line.startswith('granted_devices')]
assert len(lines) == 1 and runtime['host'] + ':' in lines[0], 'grant host changed'
actual = sorted(int(value) for value in re.findall(r'/dev/nvidia(\d+)', lines[0]))
assert actual == sorted(runtime['physical_gpus']), 'granted physical GPUs changed'
assert re.search(r'job_state\s+\d+:\s+r\b', grant), 'allocation is not running'
assert not re.search(r'hard_resource_list:.*s_vmem', grant), 'CUDA virtual memory limit is forbidden'
PY
    memory=$(nvidia-smi -i "$devices" --query-gpu=memory.used --format=csv,noheader,nounits)
    checked=0
    while read -r used; do
        [[ $used =~ ^[0-9]+$ && $used -le 256 ]] || exit 3
        checked=$((checked + 1))
    done <<< "$memory"
    [[ $checked -eq $world ]] || exit 3
    export CUDA_VISIBLE_DEVICES="$devices"
fi

exec > "$jobs/$label.log" 2>&1
trap 'code=$?; printf "%s\n" "$code" > "$jobs/$label.exit"' EXIT
cd "$repo"
date -Is
printf 'MODE=%s HOST=%s CUDA_VISIBLE_DEVICES=%s ULIMIT_V=%s\n' "$mode" "$(hostname)" "${CUDA_VISIBLE_DEVICES:-none}" "$(ulimit -v)"
"$py" -B model_training/tools/agent_work_guard.py begin > "$jobs/${label}_begin.log"
if [[ "$mode" == audit || "$mode" == initialize ]]; then
    "$py" -B model_training/rigweave/scripts/train_structure_xyz_fill.py --mode "$mode" --config "$config" --output "$output"
else
    "$py" -B -m torch.distributed.run --standalone --nproc_per_node="$world" \
        model_training/rigweave/scripts/train_structure_xyz_fill.py --mode "$mode" --config "$config" --output "$output"
fi
date -Is
