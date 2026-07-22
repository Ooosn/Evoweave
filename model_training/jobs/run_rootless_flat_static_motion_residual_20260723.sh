#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
MODEL_ROOT="${MODEL_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
REPO_ROOT="$(cd "${MODEL_ROOT}/.." && pwd)"

MANIFEST_ROOT="${EVOWEAVE_MANIFEST_ROOT:-/ssdwork/liuhaohan/evorig/evoweave_rebuild_rootless_v3_20260706/quality_distributions/rootless_bbox_consistency/final_manifests}"
TRAIN_MANIFEST="${JOB_TRAIN_MANIFEST:-${MANIFEST_ROOT}/train_manifest.jsonl}"
VAL_MANIFEST="${JOB_VAL_MANIFEST:-${MANIFEST_ROOT}/valid_manifest.jsonl}"
ASSET_ROOT="${JOB_MODEL_ASSET_ROOT:-/ssdwork/liuhaohan/evorig/Evoweave/model_training/third_party_references}"
UNIRIG_ROOT="${JOB_UNIRIG_ROOT:-${ASSET_ROOT}/UniRig}"
UNIRIG_CKPT="${JOB_UNIRIG_CKPT:-${ASSET_ROOT}/UniRig_hf/skeleton/articulation-xl_quantization_256/model.ckpt}"
MODEL_CONFIG="${JOB_MODEL_CONFIG:-${UNIRIG_ROOT}/configs/model/unirig_ar_350m_1024_81920_float32.yaml}"
TOKENIZER_CONFIG="${JOB_TOKENIZER_CONFIG:-${UNIRIG_ROOT}/configs/tokenizer/tokenizer_parts_articulationxl_256.yaml}"
OUTPUT_BASE="${EVOWEAVE_MODEL_OUTPUT_BASE:-/ssdwork/liuhaohan/evoweave/outputs/dynamic_rig_runs}"
RUN_NAME="${JOB_RUN_NAME:-flat_static_motion_residual_20260723}"
OUTPUT_DIR="${JOB_OUTPUT_DIR:-${OUTPUT_BASE}/${RUN_NAME}}"
CONDITION_FUSION="${JOB_CONDITION_FUSION:-static_cross_attn_zero}"

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
EXPECTED_EFFECTIVE_BATCH="${JOB_EXPECTED_EFFECTIVE_BATCH:-48}"
EXPECTED_TRAIN_ROWS="${JOB_EXPECTED_TRAIN_ROWS:-15903}"
EXPECTED_VAL_ROWS="${JOB_EXPECTED_VAL_ROWS:-857}"
EFFECTIVE_BATCH=$((NPROC * BATCH_SIZE * GRAD_ACCUM))

if [[ -n "${JOB_INIT_CHECKPOINT:-}" || -n "${JOB_RESUME_CHECKPOINT:-}" ]]; then
  echo "[static_motion_residual] ERROR: dynamic init/resume checkpoints are forbidden" >&2
  exit 2
fi
if [[ "${CONDITION_FUSION}" != "static_cross_attn_zero" && "${CONDITION_FUSION}" != "anchor_motion_residual_zero" ]]; then
  echo "[static_motion_residual] ERROR: unsupported condition fusion ${CONDITION_FUSION}" >&2
  exit 2
fi
if [[ "${EFFECTIVE_BATCH}" != "${EXPECTED_EFFECTIVE_BATCH}" ]]; then
  echo "[static_motion_residual] ERROR: effective batch=${EFFECTIVE_BATCH}, expected ${EXPECTED_EFFECTIVE_BATCH}" >&2
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
    echo "[static_motion_residual] ERROR: missing required path ${required}" >&2
    exit 2
  fi
done

train_rows="$(wc -l < "${TRAIN_MANIFEST}" | tr -d ' ')"
val_rows="$(wc -l < "${VAL_MANIFEST}" | tr -d ' ')"
if [[ "${train_rows}" != "${EXPECTED_TRAIN_ROWS}" ]]; then
  echo "[static_motion_residual] ERROR: train rows=${train_rows}, expected ${EXPECTED_TRAIN_ROWS}" >&2
  exit 3
fi
if [[ "${val_rows}" != "${EXPECTED_VAL_ROWS}" ]]; then
  echo "[static_motion_residual] ERROR: valid rows=${val_rows}, expected ${EXPECTED_VAL_ROWS}" >&2
  exit 3
fi

visible_gpus="$(nvidia-smi -L | wc -l | tr -d ' ')"
if [[ "${visible_gpus}" != "${EXPECTED_GPUS}" ]]; then
  echo "[static_motion_residual] ERROR: visible GPUs=${visible_gpus}, expected ${EXPECTED_GPUS}" >&2
  nvidia-smi -L >&2
  exit 4
fi

if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  echo "[static_motion_residual] ERROR: source worktree is dirty" >&2
  git -C "${REPO_ROOT}" status --short >&2
  exit 5
fi
GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
if [[ -n "${JOB_EXPECTED_COMMIT:-}" && "${GIT_COMMIT}" != "${JOB_EXPECTED_COMMIT}" ]]; then
  echo "[static_motion_residual] ERROR: commit=${GIT_COMMIT}, expected ${JOB_EXPECTED_COMMIT}" >&2
  exit 5
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "[static_motion_residual] ERROR: output already exists: ${OUTPUT_DIR}" >&2
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

export EVOWEAVE_ROOT="${MODEL_ROOT}"
export EVOWEAVE_TRAIN_MANIFEST="${TRAIN_MANIFEST}"
export EVOWEAVE_VAL_MANIFEST="${VAL_MANIFEST}"
export EVOWEAVE_OUTPUT_DIR="${OUTPUT_DIR}"
export EVOWEAVE_UNIRIG_ROOT="${UNIRIG_ROOT}"
export EVOWEAVE_UNIRIG_CKPT="${UNIRIG_CKPT}"
export EVOWEAVE_MODEL_CONFIG="${MODEL_CONFIG}"
export EVOWEAVE_TOKENIZER_CONFIG="${TOKENIZER_CONFIG}"
export EVOWEAVE_TRAIN_ROWS="${train_rows}"
export EVOWEAVE_INIT_CHECKPOINT=""
export EVOWEAVE_RESUME_CHECKPOINT=""
export PYTHONPATH="${MODEL_ROOT}/rigweave/src:${UNIRIG_ROOT}:${PYTHONPATH:-}"
export RIGWEAVE_DISABLE_OPEN3D=1
export RIGWEAVE_DISABLE_LIGHTNING_IMPORT=1
export HF_HOME="${HF_HOME:-/ssdwork/liuhaohan/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export RIGWEAVE_NPROC="${NPROC}"
export RIGWEAVE_BATCH_SIZE="${BATCH_SIZE}"
export RIGWEAVE_GRAD_ACCUM="${GRAD_ACCUM}"
export RIGWEAVE_MAX_STEPS="${JOB_MAX_STEPS:-1667}"
export RIGWEAVE_STOP_AFTER_SAMPLES="${JOB_STOP_AFTER_SAMPLES:-0}"
export RIGWEAVE_SAMPLE_MILESTONES="${JOB_SAMPLE_MILESTONES:-5000,10000,20000,30000,50000,80000}"
export RIGWEAVE_SAVE_EVERY="${JOB_SAVE_EVERY:-0}"
export RIGWEAVE_LOG_EVERY="${JOB_LOG_EVERY:-10}"
export RIGWEAVE_VAL_EVERY="${JOB_VAL_EVERY:-200}"
export RIGWEAVE_VAL_STEPS="${JOB_VAL_STEPS:-16}"
export RIGWEAVE_NUM_WORKERS="${JOB_NUM_WORKERS:-0}"
export RIGWEAVE_FRAMES="${JOB_FRAMES:-24}"
export RIGWEAVE_SURFACE_SAMPLES="${JOB_SURFACE_SAMPLES:-65536}"
export RIGWEAVE_VERTEX_SAMPLES="${JOB_VERTEX_SAMPLES:-8192}"
export RIGWEAVE_QUERY_TOKENS=1024
export RIGWEAVE_REGISTER_TOKENS="${JOB_REGISTER_TOKENS:-96}"
export RIGWEAVE_MOTION_DEPTH="${JOB_MOTION_DEPTH:-12}"
export RIGWEAVE_MOTION_HEADS="${JOB_MOTION_HEADS:-8}"
export RIGWEAVE_MOTION_FPS_RATIO="${JOB_MOTION_FPS_RATIO:-0.7}"
export RIGWEAVE_MOTION_VERTEX_SAMPLES="${JOB_MOTION_VERTEX_SAMPLES:-512}"
export RIGWEAVE_REQUIRE_CUDA=1

export RIGWEAVE_TARGET_ROOT_POLICY=legacy
export RIGWEAVE_TARGET_START_POLICY=joint0
export RIGWEAVE_BRANCH_PRIOR_PROPOSALS=0
export RIGWEAVE_BRANCH_PRIOR_LOSS_WEIGHT=0.0
export RIGWEAVE_BRANCH_PRIOR_COORD_LOSS_WEIGHT=0.0
export RIGWEAVE_FREEZE_BRANCH_PRIOR=1
export RIGWEAVE_EXPLICIT_TREE_LOSS_WEIGHT=0.0
export RIGWEAVE_EXPLICIT_TREE_GENERATED_PREFIX_WEIGHT=0.0
export RIGWEAVE_EXPLICIT_TREE_ORACLE_PREFIX_WEIGHT=0.0
export RIGWEAVE_EXPLICIT_TREE_PREFIX_JITTER_WEIGHT=0.0
export RIGWEAVE_PREFIX_TOKEN_RECOVERY_WEIGHT=0.0
export RIGWEAVE_PREFIX_DECISION_RECOVERY_WEIGHT=0.0
export RIGWEAVE_PREFIX_ACTION_RECOVERY_WEIGHT=0.0
export RIGWEAVE_GENERATED_PREFIX_RECOVERY_WEIGHT=0.0
export RIGWEAVE_CONDITION_CONTROL_CE_WEIGHT=0.0
export RIGWEAVE_CONDITION_CONTROL_CE_EVERY=0
export RIGWEAVE_DECISION_LOSS_WEIGHT=0.0
export RIGWEAVE_LOOP_RECOVERY_LOSS_WEIGHT=0.0
export RIGWEAVE_STRUCTURE_COUNT_LOSS_WEIGHT=0.0
export RIGWEAVE_STRUCTURE_ACTION_LOSS_WEIGHT=0.0
export RIGWEAVE_USE_GRAMMAR_STATE_EMBEDDING=0
export RIGWEAVE_USE_ACTION_GROUP_BIAS=0
export RIGWEAVE_USE_CONDITION_ACTION_GROUP_BIAS=0

export RIGWEAVE_CONDITION_FUSION="${CONDITION_FUSION}"
export RIGWEAVE_CONDITION_FUSION_HEADS=8
export RIGWEAVE_CONDITION_FUSION_GATE_INIT=0.25
export RIGWEAVE_CONDITION_FUSION_DEPTH=1
export RIGWEAVE_CONDITION_STATIC_BLEND_WEIGHT=0.0
export RIGWEAVE_RESET_CONDITION_FUSER=1

export RIGWEAVE_LR_AR="${JOB_LR_AR:-0.0001}"
export RIGWEAVE_LR_MOTION="${JOB_LR_MOTION:-0.0001}"
export RIGWEAVE_LR_SURFACE="${JOB_LR_SURFACE:-0.0001}"
export RIGWEAVE_WEIGHT_DECAY="${JOB_WEIGHT_DECAY:-0.04}"
export RIGWEAVE_SCHEDULER=onecycle
export RIGWEAVE_ONECYCLE_PCT_START="${JOB_ONECYCLE_PCT_START:-0.1}"
export RIGWEAVE_ONECYCLE_DIV_FACTOR="${JOB_ONECYCLE_DIV_FACTOR:-5.0}"
export RIGWEAVE_ONECYCLE_FINAL_DIV_FACTOR="${JOB_ONECYCLE_FINAL_DIV_FACTOR:-10.0}"
export RIGWEAVE_TRAIN_SURFACE_TOKENIZER=1
export RIGWEAVE_FREEZE_AR=0
export RIGWEAVE_FREEZE_CONDITIONER=0
export RIGWEAVE_MOTION_CHECKPOINTING=1
export RIGWEAVE_NO_SAVE_OPTIMIZER="${JOB_NO_SAVE_OPTIMIZER:-0}"

{
  echo "route=${CONDITION_FUSION}"
  echo "git_commit=${GIT_COMMIT}"
  echo "train_manifest=${TRAIN_MANIFEST}"
  echo "val_manifest=${VAL_MANIFEST}"
  echo "train_rows=${train_rows}"
  echo "val_rows=${val_rows}"
  echo "initialization=${UNIRIG_CKPT}"
  echo "dynamic_checkpoint_loaded=false"
  echo "visible_gpus=${visible_gpus}"
  echo "effective_batch=${EFFECTIVE_BATCH}"
  echo "condition_fusion=${CONDITION_FUSION}"
  echo "condition_fusion_gate_init=0.25"
  echo "condition_fusion_depth=1"
  if [[ "${CONDITION_FUSION}" == "anchor_motion_residual_zero" ]]; then
    echo "initial_fused_condition=exact_trackable_frame0_identity"
  else
    echo "initial_fused_condition=exact_static_unirig_identity"
  fi
  echo "use_motion_features=false"
  echo "use_time_embedding=false"
  echo "train_random_query=true"
  echo "surface_tokenizer=trainable"
  echo "motion_encoder=trainable"
  echo "ar_decoder=trainable"
  echo "condition_fuser=trainable"
  echo "branch_prior_proposals=0"
  echo "explicit_tree_loss_weight=0.0"
  echo "all_recovery_losses=0.0"
  echo "scheduler=onecycle"
  echo "lr_ar=${RIGWEAVE_LR_AR}"
  echo "lr_motion=${RIGWEAVE_LR_MOTION}"
  echo "lr_surface=${RIGWEAVE_LR_SURFACE}"
  echo "weight_decay=${RIGWEAVE_WEIGHT_DECAY}"
} > "${OUTPUT_DIR}/resolved_contract.txt"

python rigweave/scripts/audit_static_motion_residual_contract.py \
  --manifest "${VAL_MANIFEST}" \
  --tokenizer-config "${TOKENIZER_CONFIG}" \
  --model-config "${MODEL_CONFIG}" \
  --unirig-checkpoint "${UNIRIG_CKPT}" \
  --frames "${RIGWEAVE_FRAMES:-24}" \
  --surface-samples "${RIGWEAVE_SURFACE_SAMPLES:-65536}" \
  --vertex-samples "${RIGWEAVE_VERTEX_SAMPLES:-8192}" \
  --query-tokens "${RIGWEAVE_QUERY_TOKENS:-1024}" \
  --register-tokens "${RIGWEAVE_REGISTER_TOKENS:-96}" \
  --motion-depth "${RIGWEAVE_MOTION_DEPTH:-12}" \
  --motion-heads "${RIGWEAVE_MOTION_HEADS:-8}" \
  --motion-fps-ratio "${RIGWEAVE_MOTION_FPS_RATIO:-0.7}" \
  --motion-vertex-samples "${RIGWEAVE_MOTION_VERTEX_SAMPLES:-512}" \
  --gate-init "${RIGWEAVE_CONDITION_FUSION_GATE_INIT}" \
  --condition-fusion "${CONDITION_FUSION}" \
  --output "${OUTPUT_DIR}/static_motion_residual_preflight.json"

echo "[static_motion_residual] preflight passed commit=${GIT_COMMIT} effective_batch=${EFFECTIVE_BATCH}"
bash rigweave/scripts/run_dynamic_ar_train.sh 2>&1 | tee "${OUTPUT_DIR}/train_frontend.log"
