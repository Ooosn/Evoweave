#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-/ssdwork/liuhaohan/evorig/evoweave_model_training_20260706}"
MANIFEST_ROOT="${EVOWEAVE_MANIFEST_ROOT:-/ssdwork/liuhaohan/evorig/evoweave_rebuild_rootless_v3_20260706/quality_distributions/rootless_bbox_consistency/final_manifests}"
RUN_NAME="${JOB_RUN_NAME:-rootless_flat_unirig_motion_fullft_20260707_hxr4gpu}"
JOB_LOG_DIR="${JOB_LOG_DIR:-/ssdwork/liuhaohan/jobs/evoweave_rootless_flat_unirig_motion_fullft_20260707/logs}"

TRAIN_MANIFEST="${MANIFEST_ROOT}/train_manifest.jsonl"
VALID_MANIFEST="${MANIFEST_ROOT}/valid_manifest.jsonl"
LAUNCHER="${MODEL_ROOT}/jobs/run_rootless_flat_unirig_motion_baseline_20260706.sh"

mkdir -p "${JOB_LOG_DIR}"
exec > >(tee -a "${JOB_LOG_DIR}/job.log") 2>&1

echo "[westlake flat baseline] start=$(date -Is)"
echo "[westlake flat baseline] host=$(hostname)"
echo "[westlake flat baseline] model_root=${MODEL_ROOT}"
echo "[westlake flat baseline] manifest_root=${MANIFEST_ROOT}"
echo "[westlake flat baseline] train_manifest=${TRAIN_MANIFEST}"
echo "[westlake flat baseline] valid_manifest=${VALID_MANIFEST}"
echo "[westlake flat baseline] run_name=${RUN_NAME}"

if [[ ! -f "${LAUNCHER}" ]]; then
  echo "[westlake flat baseline] ERROR: missing launcher ${LAUNCHER}" >&2
  exit 2
fi
if [[ ! -f "${TRAIN_MANIFEST}" || ! -f "${VALID_MANIFEST}" ]]; then
  echo "[westlake flat baseline] ERROR: missing final train/valid manifests" >&2
  exit 2
fi

train_rows="$(wc -l < "${TRAIN_MANIFEST}")"
valid_rows="$(wc -l < "${VALID_MANIFEST}")"
echo "[westlake flat baseline] train_rows=${train_rows}"
echo "[westlake flat baseline] valid_rows=${valid_rows}"
if [[ "${train_rows}" != "15903" || "${valid_rows}" != "857" ]]; then
  echo "[westlake flat baseline] ERROR: manifest row count mismatch; refusing to train on wrong data." >&2
  exit 3
fi

gpu_rows="$(nvidia-smi -L | wc -l | tr -d ' ')"
echo "[westlake flat baseline] visible_gpus=${gpu_rows}"
nvidia-smi -L
if [[ "${gpu_rows}" != "4" ]]; then
  echo "[westlake flat baseline] ERROR: formal baseline requires exactly 4 visible GPUs." >&2
  exit 4
fi

export JOB_RUN_NAME="${RUN_NAME}"
export EVOWEAVE_MANIFEST_ROOT="${MANIFEST_ROOT}"
export JOB_TRAIN_MANIFEST="${TRAIN_MANIFEST}"
export JOB_VAL_MANIFEST="${VALID_MANIFEST}"
export JOB_TEST_MANIFEST=""

export JOB_NPROC=4
export JOB_BATCH_SIZE=3
export JOB_GRAD_ACCUM=4
export JOB_MAX_STEPS="${JOB_MAX_STEPS:-1667}"
export JOB_SAMPLE_MILESTONES="${JOB_SAMPLE_MILESTONES:-5000,10000,20000,30000,50000,80000}"
export JOB_SAVE_EVERY="${JOB_SAVE_EVERY:-0}"
export JOB_VAL_EVERY="${JOB_VAL_EVERY:-200}"
export JOB_VAL_STEPS="${JOB_VAL_STEPS:-16}"
export RIGWEAVE_REQUIRE_CUDA=1

bash "${LAUNCHER}"

echo "[westlake flat baseline] done=$(date -Is)"
