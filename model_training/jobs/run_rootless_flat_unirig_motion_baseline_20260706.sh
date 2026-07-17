#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
DEFAULT_MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_ROOT="${MODEL_ROOT:-${DEFAULT_MODEL_ROOT}}"
MANIFEST_ROOT="${EVOWEAVE_MANIFEST_ROOT:-${EVOWEAVE_ROOTLESS_ROOT:-/ssdwork/liuhaohan/evorig/evoweave_rebuild_rootless_v3_20260706/quality_distributions/rootless_bbox_consistency/final_manifests}}"
RUN_NAME="${JOB_RUN_NAME:-rootless_flat_unirig_motion_fullft_20260706}"
OUTPUT_BASE="${EVOWEAVE_MODEL_OUTPUT_BASE:-/ssdwork/liuhaohan/evoweave/outputs/dynamic_rig_runs}"

CONDA_SH="${JOB_CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
CONDA_ENV="${JOB_CONDA_ENV:-evoweave}"
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[flat baseline] ERROR: conda initialization script does not exist: ${CONDA_SH}" >&2
  exit 2
fi
source "${CONDA_SH}"
set +u
conda activate "${CONDA_ENV}"
set -u

cd "${MODEL_ROOT}"

export HF_HOME="${HF_HOME:-/ssdwork/liuhaohan/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export RIGWEAVE_DISABLE_OPEN3D=1
export RIGWEAVE_DISABLE_LIGHTNING_IMPORT=1

if [[ -z "${MANIFEST_ROOT}" && ( -z "${JOB_TRAIN_MANIFEST:-}" || -z "${JOB_VAL_MANIFEST:-}" ) ]]; then
  echo "[flat baseline] ERROR: set EVOWEAVE_MANIFEST_ROOT, or set JOB_TRAIN_MANIFEST and JOB_VAL_MANIFEST explicitly." >&2
  exit 2
fi

export EVOWEAVE_MANIFEST_ROOT="${MANIFEST_ROOT}"
export EVOWEAVE_TRAIN_MANIFEST="${JOB_TRAIN_MANIFEST:-${MANIFEST_ROOT}/train_manifest.jsonl}"
export EVOWEAVE_VAL_MANIFEST="${JOB_VAL_MANIFEST:-${MANIFEST_ROOT}/valid_manifest.jsonl}"
export EVOWEAVE_TEST_MANIFEST="${JOB_TEST_MANIFEST:-}"
export EVOWEAVE_UNIRIG_ROOT="${JOB_UNIRIG_ROOT:-/ssdwork/liuhaohan/evorig/evoweave_repo/external/UniRig}"
export EVOWEAVE_MODEL_CONFIG="${JOB_MODEL_CONFIG:-${EVOWEAVE_UNIRIG_ROOT}/configs/model/unirig_ar_350m_1024_81920_float32.yaml}"
export EVOWEAVE_TOKENIZER_CONFIG="${JOB_TOKENIZER_CONFIG:-${EVOWEAVE_UNIRIG_ROOT}/configs/tokenizer/tokenizer_parts_articulationxl_256.yaml}"
export EVOWEAVE_UNIRIG_CKPT="${JOB_UNIRIG_CKPT:-/ssdwork/liuhaohan/evorig/evoweave_repo/external/UniRig_hf/skeleton/articulation-xl_quantization_256/model.ckpt}"
export EVOWEAVE_INIT_CHECKPOINT="${JOB_INIT_CHECKPOINT:-}"
export EVOWEAVE_OUTPUT_DIR="${JOB_OUTPUT_DIR:-${OUTPUT_BASE}/${RUN_NAME}}"
export EVOWEAVE_ENV_FILE="${JOB_ENV_FILE:-}"
export EVOWEAVE_TRAIN_ROWS="${JOB_TRAIN_ROWS:-$(wc -l < "${EVOWEAVE_TRAIN_MANIFEST}")}"
export PYTHONPATH="${MODEL_ROOT}/rigweave/src:${EVOWEAVE_UNIRIG_ROOT}:${PYTHONPATH:-}"

# UniRig-line training profile. These defaults mirror the old dynamic UniRig
# wrapper: LR=1e-4, AdamW weight decay=0.04, OneCycle schedule, and effective
# batch 48. On four A100s that is micro batch 3 with grad accumulation 4.
export RIGWEAVE_NPROC="${JOB_NPROC:-4}"
export RIGWEAVE_BATCH_SIZE="${JOB_BATCH_SIZE:-3}"
export RIGWEAVE_GRAD_ACCUM="${JOB_GRAD_ACCUM:-4}"
export RIGWEAVE_MAX_STEPS="${JOB_MAX_STEPS:-1667}"
export RIGWEAVE_SAMPLE_MILESTONES="${JOB_SAMPLE_MILESTONES:-5000,10000,20000,30000,50000,80000}"
export RIGWEAVE_SAVE_EVERY="${JOB_SAVE_EVERY:-0}"
export RIGWEAVE_LOG_EVERY="${JOB_LOG_EVERY:-10}"
export RIGWEAVE_VAL_EVERY="${JOB_VAL_EVERY:-200}"
export RIGWEAVE_VAL_STEPS="${JOB_VAL_STEPS:-16}"
export RIGWEAVE_NUM_WORKERS="${JOB_NUM_WORKERS:-0}"

export RIGWEAVE_TARGET_ROOT_POLICY=legacy
export RIGWEAVE_TARGET_START_POLICY=joint0

# Old-style flat UniRig baseline.
export RIGWEAVE_BRANCH_PRIOR_PROPOSALS=0
export RIGWEAVE_BRANCH_PRIOR_LOSS_WEIGHT=0.0
export RIGWEAVE_BRANCH_PRIOR_COORD_LOSS_WEIGHT=0.0
export RIGWEAVE_FREEZE_BRANCH_PRIOR=1

export RIGWEAVE_PREFIX_TOKEN_RECOVERY_WEIGHT=0.0
export RIGWEAVE_PREFIX_DECISION_RECOVERY_WEIGHT=0.0
export RIGWEAVE_PREFIX_ACTION_RECOVERY_WEIGHT=0.0
export RIGWEAVE_CONDITION_CONTROL_CE_WEIGHT=0.0
export RIGWEAVE_CONDITION_CONTROL_CE_EVERY=0
export RIGWEAVE_DECISION_LOSS_WEIGHT=0.0
export RIGWEAVE_LOOP_RECOVERY_LOSS_WEIGHT=0.0
export RIGWEAVE_STRUCTURE_COUNT_LOSS_WEIGHT=0.0
export RIGWEAVE_STRUCTURE_ACTION_LOSS_WEIGHT=0.0

export RIGWEAVE_USE_GRAMMAR_STATE_EMBEDDING=0
export RIGWEAVE_USE_ACTION_GROUP_BIAS=0
export RIGWEAVE_USE_CONDITION_ACTION_GROUP_BIAS=0
export RIGWEAVE_CONDITION_FUSION=dynamic
export RIGWEAVE_RESET_CONDITION_FUSER=0

export RIGWEAVE_LR_AR="${JOB_LR_AR:-1e-4}"
export RIGWEAVE_LR_MOTION="${JOB_LR_MOTION:-1e-4}"
export RIGWEAVE_LR_SURFACE="${JOB_LR_SURFACE:-1e-4}"
export RIGWEAVE_WEIGHT_DECAY="${JOB_WEIGHT_DECAY:-0.04}"
export RIGWEAVE_SCHEDULER="${JOB_SCHEDULER:-onecycle}"
export RIGWEAVE_ONECYCLE_PCT_START="${JOB_ONECYCLE_PCT_START:-0.1}"
export RIGWEAVE_ONECYCLE_DIV_FACTOR="${JOB_ONECYCLE_DIV_FACTOR:-5.0}"
export RIGWEAVE_ONECYCLE_FINAL_DIV_FACTOR="${JOB_ONECYCLE_FINAL_DIV_FACTOR:-10.0}"
export RIGWEAVE_TRAIN_SURFACE_TOKENIZER=1
export RIGWEAVE_FREEZE_AR=0
export RIGWEAVE_FREEZE_CONDITIONER=0

mkdir -p "${EVOWEAVE_OUTPUT_DIR}"
cp -f "${SCRIPT_PATH}" "${EVOWEAVE_OUTPUT_DIR}/launch.sh"
python3 - <<'PY_ENV' > "${EVOWEAVE_OUTPUT_DIR}/resolved_env.txt"
import os

prefixes = ("EVOWEAVE", "RIGWEAVE", "JOB_")
exact = {"MODEL_ROOT", "MANIFEST_ROOT"}
for key in sorted(os.environ):
    if key in exact or key.startswith(prefixes):
        print(f"{key}={os.environ[key]}")
PY_ENV

echo "[flat baseline] start $(date -Is)"
echo "[flat baseline] model_root=${MODEL_ROOT}"
echo "[flat baseline] manifest_root=${MANIFEST_ROOT}"
echo "[flat baseline] train_manifest=${EVOWEAVE_TRAIN_MANIFEST}"
echo "[flat baseline] valid_manifest=${EVOWEAVE_VAL_MANIFEST}"
echo "[flat baseline] output=${EVOWEAVE_OUTPUT_DIR}"
echo "[flat baseline] effective_batch=$((RIGWEAVE_NPROC * RIGWEAVE_BATCH_SIZE * RIGWEAVE_GRAD_ACCUM))"
echo "[flat baseline] profile=unirig lr_ar=${RIGWEAVE_LR_AR} lr_motion=${RIGWEAVE_LR_MOTION} lr_surface=${RIGWEAVE_LR_SURFACE} weight_decay=${RIGWEAVE_WEIGHT_DECAY} scheduler=${RIGWEAVE_SCHEDULER}"
echo "[flat baseline] decoder=UniRig flat AR"
bash rigweave/scripts/run_dynamic_ar_train.sh 2>&1 | tee "${EVOWEAVE_OUTPUT_DIR}/train_frontend.log"
echo "[flat baseline] done $(date -Is)"
