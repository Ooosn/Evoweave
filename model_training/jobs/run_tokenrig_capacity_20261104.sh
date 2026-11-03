#!/usr/bin/env bash
# SLOTS: 8
set -euo pipefail

case "${0##*/}" in
    tokenrig_batch_frames12.sh) profile=frames12 ;;
    tokenrig_batch_frames24.sh) profile=frames24 ;;
    *) printf '%s\n' 'Unexpected capacity task name' >&2; exit 2 ;;
esac
root=/work/wangyy/tokenrig_batch_20261104/code
out=/work/wangyy/tokenrig_batch_20261104/$profile
py=/home/wangyy/miniconda3/envs/mygs/bin/python
gpu_a=GPU-1fb4221b-91b0-0df6-8615-b36fbdd55bf7
gpu_b=GPU-0544830a-4dac-9fe3-a099-95ddc2240b72
expected_job=131351736
[[ "${JOB_ID:-}" == "$expected_job" && "${NSLOTS:-}" == 16 ]]
for group in cpuset devices; do
    grep -Eq "^[0-9]+:${group}:/AGE/${expected_job}\.1/" /proc/self/cgroup
done
[[ "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | sort)" == "$(printf '%s\n' "$gpu_a" "$gpu_b" | sort)" ]]
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
[[ "$(ulimit -v)" == unlimited ]]
mkdir "$out"
cat /proc/self/cgroup > "$out/cgroups.txt"
export CUDA_VISIBLE_DEVICES="$gpu_a,$gpu_b"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export EVOWEAVE_UNIRIG_ROOT=/home/wangyy/evorig/model_assets/UniRig
export PYTHONPATH="$root/model_training/rigweave/src:$EVOWEAVE_UNIRIG_ROOT:/work/wangyy/paper_baselines_20261102/runtime_skin:/home/wangyy/evorig/runtime/puppeteer_py310_hgc"
cd "$root"
taskset -c 6,7,9,10,11,12,14,15,61,63,64,65,66,67,68,69 true
timeout -k 25s 30m taskset -c 6,7,9,10,11,12,14,15,61,63,64,65,66,67,68,69 \
    "$py" -u -m torch.distributed.run --standalone --nproc-per-node=2 \
    model_training/rigweave/scripts/probe_tokenrig_capacity.py \
    --config model_training/experiments/anchor_full_20261030_v1.json \
    --decoder model_training/experiments/tokenrig_decoder_20261104.json --output "$out" --profile "$profile" \
    > "$out/probe.log" 2>&1 &
pid=$!
trap 'kill -TERM "$pid" 2>/dev/null || true; wait "$pid" || true' EXIT
while kill -0 "$pid" 2>/dev/null; do
    sample=$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits)
    printf '%s\n' "$sample" | awk -v t="$(date -Ins)" '{print t ", " $0}' >> "$out/nvml.csv"
    if ! printf '%s\n' "$sample" | awk -F, 'NF != 2 || $2 >= 78000 {bad=1} END {exit (bad || NR != 2)}'; then
        printf '%s\n' 'GPU memory reached the 78000 MiB stop threshold or telemetry became invalid.' > "$out/resource_stop.txt"
        exit 1
    fi
    sleep 1
done
wait "$pid"
