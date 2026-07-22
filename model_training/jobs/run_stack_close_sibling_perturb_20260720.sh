#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
MODEL_ROOT="${MODEL_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
REPO_ROOT="$(cd "${MODEL_ROOT}/.." && pwd)"

MANIFEST_ROOT="${EVOWEAVE_MANIFEST_ROOT:-/ssdwork/liuhaohan/evorig/evoweave_rebuild_rootless_v3_20260706/quality_distributions/rootless_bbox_consistency/final_manifests}"
TRAIN_MANIFEST="${JOB_TRAIN_MANIFEST:-${MANIFEST_ROOT}/train_manifest.jsonl}"
VAL_MANIFEST="${JOB_VAL_MANIFEST:-${MANIFEST_ROOT}/valid_manifest.jsonl}"
UNIRIG_ROOT="${JOB_UNIRIG_ROOT:-/ssdwork/liuhaohan/evorig/Evoweave/model_training/third_party_references/UniRig}"
UNIRIG_CKPT="${JOB_UNIRIG_CKPT:-/ssdwork/liuhaohan/evorig/Evoweave/model_training/third_party_references/UniRig_hf/skeleton/articulation-xl_quantization_256/model.ckpt}"
MODEL_CONFIG="${JOB_MODEL_CONFIG:-${UNIRIG_ROOT}/configs/model/unirig_ar_350m_1024_81920_float32.yaml}"
TOKENIZER_CONFIG="${JOB_TOKENIZER_CONFIG:-${UNIRIG_ROOT}/configs/tokenizer/tokenizer_parts_articulationxl_256.yaml}"
OUTPUT_BASE="${EVOWEAVE_MODEL_OUTPUT_BASE:-/ssdwork/liuhaohan/evoweave/outputs/dynamic_rig_runs}"
RUN_NAME="${JOB_RUN_NAME:-stack_close_sibling_perturb_20260720}"
OUTPUT_DIR="${JOB_OUTPUT_DIR:-${OUTPUT_BASE}/${RUN_NAME}}"

CONDA_SH="${JOB_CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
CONDA_ENV="${JOB_CONDA_ENV:-evoweave}"
NPROC="${JOB_NPROC:-2}"
BATCH_SIZE="${JOB_BATCH_SIZE:-3}"
if [[ -n "${JOB_GRAD_ACCUM:-}" ]]; then
  GRAD_ACCUM="${JOB_GRAD_ACCUM}"
elif [[ "${NPROC}" == "2" ]]; then
  GRAD_ACCUM=8
else
  GRAD_ACCUM=16
fi
EXPECTED_GPUS="${JOB_EXPECTED_GPUS:-${NPROC}}"
USE_MOTION_FEATURES="${JOB_USE_MOTION_FEATURES:-1}"
USE_TIME_EMBEDDING="${JOB_USE_TIME_EMBEDDING:-1}"
RANDOM_SIBLING_ORDER="${JOB_RANDOM_SIBLING_ORDER:-1}"
INITIALIZE_STACK_CHECKPOINT="${JOB_INITIALIZE_STACK_CHECKPOINT:-}"
FREEZE_BASE_FOR_STACK_ACTION="${JOB_FREEZE_BASE_FOR_STACK_ACTION:-0}"

for flag_name in \
  USE_MOTION_FEATURES \
  USE_TIME_EMBEDDING \
  RANDOM_SIBLING_ORDER \
  FREEZE_BASE_FOR_STACK_ACTION; do
  flag_value="${!flag_name}"
  if [[ "${flag_value}" != "0" && "${flag_value}" != "1" ]]; then
    echo "[stack_close] ERROR: ${flag_name} must be 0 or 1, got ${flag_value}" >&2
    exit 2
  fi
done

if [[ "${FREEZE_BASE_FOR_STACK_ACTION}" == "1" && -z "${INITIALIZE_STACK_CHECKPOINT}" ]]; then
  echo "[stack_close] ERROR: freezing the base requires JOB_INITIALIZE_STACK_CHECKPOINT" >&2
  exit 2
fi

for required in \
  "${TRAIN_MANIFEST}" \
  "${VAL_MANIFEST}" \
  "${UNIRIG_CKPT}" \
  "${MODEL_CONFIG}" \
  "${TOKENIZER_CONFIG}" \
  "${CONDA_SH}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[stack_close] ERROR: missing required path ${required}" >&2
    exit 2
  fi
done
if [[ -n "${INITIALIZE_STACK_CHECKPOINT}" && ! -f "${INITIALIZE_STACK_CHECKPOINT}" ]]; then
  echo "[stack_close] ERROR: missing initialization checkpoint ${INITIALIZE_STACK_CHECKPOINT}" >&2
  exit 2
fi

train_rows="$(wc -l < "${TRAIN_MANIFEST}" | tr -d ' ')"
val_rows="$(wc -l < "${VAL_MANIFEST}" | tr -d ' ')"
if [[ "${JOB_LIMIT_TRAIN:-0}" == "0" && "${train_rows}" != "15903" ]]; then
  echo "[stack_close] ERROR: train rows=${train_rows}, expected 15903" >&2
  exit 3
fi
if [[ "${JOB_LIMIT_VAL:-0}" == "0" && "${val_rows}" != "857" ]]; then
  echo "[stack_close] ERROR: valid rows=${val_rows}, expected 857" >&2
  exit 3
fi

visible_gpus="$(nvidia-smi -L | wc -l | tr -d ' ')"
if [[ "${visible_gpus}" != "${EXPECTED_GPUS}" ]]; then
  echo "[stack_close] ERROR: visible GPUs=${visible_gpus}, expected ${EXPECTED_GPUS}" >&2
  nvidia-smi -L >&2
  exit 4
fi

if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  echo "[stack_close] ERROR: source worktree is dirty; refusing an unversioned run" >&2
  git -C "${REPO_ROOT}" status --short >&2
  exit 5
fi
GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
if [[ -n "${JOB_EXPECTED_COMMIT:-}" && "${GIT_COMMIT}" != "${JOB_EXPECTED_COMMIT}" ]]; then
  echo "[stack_close] ERROR: commit=${GIT_COMMIT}, expected ${JOB_EXPECTED_COMMIT}" >&2
  exit 5
fi

if [[ -e "${OUTPUT_DIR}" && -z "${JOB_RESUME_CHECKPOINT:-}" ]]; then
  echo "[stack_close] ERROR: output already exists and no resume checkpoint was supplied: ${OUTPUT_DIR}" >&2
  exit 6
fi

source "${CONDA_SH}"
set +u
conda activate "${CONDA_ENV}"
set -u

cd "${MODEL_ROOT}"
mkdir -p "${OUTPUT_DIR}"
cp -f "${SCRIPT_PATH}" "${OUTPUT_DIR}/launch.sh"
printf '%s\n' "${GIT_COMMIT}" > "${OUTPUT_DIR}/git_commit.txt"

export EVOWEAVE_UNIRIG_ROOT="${UNIRIG_ROOT}"
export PYTHONPATH="${MODEL_ROOT}/rigweave/src:${UNIRIG_ROOT}:${PYTHONPATH:-}"
export RIGWEAVE_DISABLE_OPEN3D=1
export RIGWEAVE_DISABLE_LIGHTNING_IMPORT=1
export HF_HOME="${HF_HOME:-/ssdwork/liuhaohan/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CMD=(
  torchrun
  --standalone
  --nproc_per_node "${NPROC}"
  rigweave/scripts/train_stack_close_dynamic_rig.py
  --train-manifest "${TRAIN_MANIFEST}"
  --val-manifest "${VAL_MANIFEST}"
  --tokenizer-config "${TOKENIZER_CONFIG}"
  --model-config "${MODEL_CONFIG}"
  --unirig-checkpoint "${UNIRIG_CKPT}"
  --output-dir "${OUTPUT_DIR}"
  --frames "${JOB_FRAMES:-24}"
  --surface-samples "${JOB_SURFACE_SAMPLES:-65536}"
  --vertex-samples "${JOB_VERTEX_SAMPLES:-8192}"
  --query-tokens 1024
  --register-tokens "${JOB_REGISTER_TOKENS:-96}"
  --motion-depth "${JOB_MOTION_DEPTH:-12}"
  --motion-heads "${JOB_MOTION_HEADS:-8}"
  --motion-fps-ratio "${JOB_MOTION_FPS_RATIO:-0.7}"
  --motion-vertex-samples "${JOB_MOTION_VERTEX_SAMPLES:-512}"
  --batch-size "${BATCH_SIZE}"
  --grad-accum-steps "${GRAD_ACCUM}"
  --num-workers "${JOB_NUM_WORKERS:-0}"
  --max-steps "${JOB_MAX_STEPS:-1667}"
  --sample-milestones "${JOB_SAMPLE_MILESTONES:-5000,10000,20000,30000,50000,80000}"
  --save-every "${JOB_SAVE_EVERY:-0}"
  --log-every "${JOB_LOG_EVERY:-10}"
  --val-every "${JOB_VAL_EVERY:-200}"
  --val-steps "${JOB_VAL_STEPS:-16}"
  --lr-motion "${JOB_LR_MOTION:-0.0001}"
  --lr-ar "${JOB_LR_AR:-0.0001}"
  --lr-surface "${JOB_LR_SURFACE:-0.0001}"
  --lr-refresh "${JOB_LR_REFRESH:-0.0001}"
  --lr-stack-action "${JOB_LR_STACK_ACTION:-0.0001}"
  --weight-decay "${JOB_WEIGHT_DECAY:-0.04}"
  --onecycle-pct-start "${JOB_ONECYCLE_PCT_START:-0.1}"
  --onecycle-div-factor "${JOB_ONECYCLE_DIV_FACTOR:-5.0}"
  --onecycle-final-div-factor "${JOB_ONECYCLE_FINAL_DIV_FACTOR:-10.0}"
  --perturb-row-probability "${JOB_PERTURB_ROW_PROBABILITY:-0.5}"
  --perturb-axial-fraction-max "${JOB_PERTURB_AXIAL_FRACTION_MAX:-0.05}"
  --perturb-radial-fraction-max "${JOB_PERTURB_RADIAL_FRACTION_MAX:-0.05}"
  --perturb-max-joints "${JOB_PERTURB_MAX_JOINTS:-4}"
  --perturb-max-joint-fraction "${JOB_PERTURB_MAX_JOINT_FRACTION:-0.08}"
  --perturb-warmup-samples "${JOB_PERTURB_WARMUP_SAMPLES:-5000}"
  --perturb-ramp-samples "${JOB_PERTURB_RAMP_SAMPLES:-15000}"
  --stack-action-loss-weight "${JOB_STACK_ACTION_LOSS_WEIGHT:-0.0}"
  --condition-refresh-layers "${JOB_CONDITION_REFRESH_LAYERS:-}"
  --condition-refresh-dim "${JOB_CONDITION_REFRESH_DIM:-256}"
  --condition-refresh-heads "${JOB_CONDITION_REFRESH_HEADS:-8}"
  --limit-train "${JOB_LIMIT_TRAIN:-0}"
  --limit-val "${JOB_LIMIT_VAL:-0}"
  --seed "${JOB_SEED:-20260720}"
  --amp-dtype "${JOB_AMP_DTYPE:-bf16}"
)
if [[ "${USE_MOTION_FEATURES}" == "1" ]]; then
  CMD+=(--use-motion-features)
else
  CMD+=(--no-use-motion-features)
fi
if [[ "${USE_TIME_EMBEDDING}" == "1" ]]; then
  CMD+=(--use-time-embedding)
else
  CMD+=(--no-use-time-embedding)
fi
if [[ "${RANDOM_SIBLING_ORDER}" == "1" ]]; then
  CMD+=(--random-sibling-order)
else
  CMD+=(--no-random-sibling-order)
fi
if [[ -n "${JOB_RESUME_CHECKPOINT:-}" ]]; then
  CMD+=(--resume-checkpoint "${JOB_RESUME_CHECKPOINT}")
fi
if [[ -n "${INITIALIZE_STACK_CHECKPOINT}" ]]; then
  CMD+=(--initialize-stack-checkpoint "${INITIALIZE_STACK_CHECKPOINT}")
fi
if [[ "${FREEZE_BASE_FOR_STACK_ACTION}" == "1" ]]; then
  CMD+=(--freeze-base-for-stack-action)
fi

route="stack_close"
if [[ "${JOB_STACK_ACTION_LOSS_WEIGHT:-0.0}" != "0" && "${JOB_STACK_ACTION_LOSS_WEIGHT:-0.0}" != "0.0" ]]; then
  route="${route}_action"
fi
if [[ -n "${JOB_CONDITION_REFRESH_LAYERS:-}" ]]; then
  route="${route}_condition_refresh"
fi
if [[ "${RANDOM_SIBLING_ORDER}" == "1" ]]; then
  route="${route}_random_sibling"
else
  route="${route}_canonical_sibling"
fi
if [[ "${JOB_PERTURB_ROW_PROBABILITY:-0.5}" != "0" && "${JOB_PERTURB_ROW_PROBABILITY:-0.5}" != "0.0" ]]; then
  route="${route}_perturb"
fi

{
  echo "route=${route}"
  echo "git_commit=${GIT_COMMIT}"
  echo "train_manifest=${TRAIN_MANIFEST}"
  echo "val_manifest=${VAL_MANIFEST}"
  echo "initialization=${INITIALIZE_STACK_CHECKPOINT:-${UNIRIG_CKPT}}"
  if [[ -n "${INITIALIZE_STACK_CHECKPOINT}" ]]; then
    echo "dynamic_checkpoint_loaded=true"
  else
    echo "dynamic_checkpoint_loaded=false"
  fi
  echo "freeze_base_for_stack_action=${FREEZE_BASE_FOR_STACK_ACTION}"
  echo "visible_gpus=${visible_gpus}"
  echo "effective_batch=$((NPROC * BATCH_SIZE * GRAD_ACCUM))"
  echo "perturb_axial_fraction_max=${JOB_PERTURB_AXIAL_FRACTION_MAX:-0.05}"
  echo "perturb_radial_fraction_max=${JOB_PERTURB_RADIAL_FRACTION_MAX:-0.05}"
  echo "condition_refresh_layers=${JOB_CONDITION_REFRESH_LAYERS:-}"
  echo "condition_refresh_dim=${JOB_CONDITION_REFRESH_DIM:-256}"
  echo "condition_refresh_heads=${JOB_CONDITION_REFRESH_HEADS:-8}"
  echo "use_motion_features=${USE_MOTION_FEATURES}"
  echo "use_time_embedding=${USE_TIME_EMBEDDING}"
  echo "random_sibling_order=${RANDOM_SIBLING_ORDER}"
  echo "stack_action_loss_weight=${JOB_STACK_ACTION_LOSS_WEIGHT:-0.0}"
} > "${OUTPUT_DIR}/resolved_contract.txt"

printf '[stack_close] command:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}" 2>&1 | tee "${OUTPUT_DIR}/train_frontend.log"
