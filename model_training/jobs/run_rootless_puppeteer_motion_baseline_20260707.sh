#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
DEFAULT_MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_ROOT="${MODEL_ROOT:-${DEFAULT_MODEL_ROOT}}"
MANIFEST_ROOT="${EVOWEAVE_MANIFEST_ROOT:-${EVOWEAVE_ROOTLESS_ROOT:-/ssdwork/liuhaohan/evorig/evoweave_rebuild_rootless_v3_20260706/quality_distributions/rootless_bbox_consistency/final_manifests}}"
RUN_NAME="${JOB_RUN_NAME:-rootless_puppeteer_motion_fullft_20260707}"
OUTPUT_BASE="${EVOWEAVE_MODEL_OUTPUT_BASE:-/ssdwork/liuhaohan/evoweave/outputs/dynamic_rig_runs}"

source /opt/conda/etc/profile.d/conda.sh
set +u
conda activate evoweave
set -u

cd "${MODEL_ROOT}"

export HF_HOME="${HF_HOME:-/ssdwork/liuhaohan/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export RIGWEAVE_DISABLE_OPEN3D=1
export RIGWEAVE_DISABLE_LIGHTNING_IMPORT=1

if [[ -z "${MANIFEST_ROOT}" && ( -z "${JOB_TRAIN_MANIFEST:-}" || -z "${JOB_VAL_MANIFEST:-}" ) ]]; then
  echo "[puppeteer baseline] ERROR: set EVOWEAVE_MANIFEST_ROOT, or set JOB_TRAIN_MANIFEST and JOB_VAL_MANIFEST explicitly." >&2
  exit 2
fi

export EVOWEAVE_MANIFEST_ROOT="${MANIFEST_ROOT}"
export EVOWEAVE_TRAIN_MANIFEST="${JOB_TRAIN_MANIFEST:-${MANIFEST_ROOT}/train_manifest.jsonl}"
export EVOWEAVE_VAL_MANIFEST="${JOB_VAL_MANIFEST:-${MANIFEST_ROOT}/valid_manifest.jsonl}"
export EVOWEAVE_UNIRIG_ROOT="${JOB_UNIRIG_ROOT:-${MODEL_ROOT}/third_party_references/UniRig}"
export EVOWEAVE_MODEL_CONFIG="${JOB_MODEL_CONFIG:-${EVOWEAVE_UNIRIG_ROOT}/configs/model/unirig_ar_350m_1024_81920_float32.yaml}"
export EVOWEAVE_TOKENIZER_CONFIG="${JOB_TOKENIZER_CONFIG:-${EVOWEAVE_UNIRIG_ROOT}/configs/tokenizer/tokenizer_parts_articulationxl_256.yaml}"
export EVOWEAVE_UNIRIG_CKPT="${JOB_UNIRIG_CKPT:-${MODEL_ROOT}/third_party_references/UniRig_hf/skeleton/articulation-xl_quantization_256/model.ckpt}"
export PUPPETEER_ROOT="${JOB_PUPPETEER_ROOT:-${MODEL_ROOT}/third_party_references/Puppeteer}"
export PUPPETEER_CHECKPOINT="${JOB_PUPPETEER_CHECKPOINT:-${PUPPETEER_CHECKPOINT:-}}"
export PUPPETEER_LLM="${JOB_PUPPETEER_LLM:-facebook/opt-350m}"
export EVOWEAVE_OUTPUT_DIR="${JOB_OUTPUT_DIR:-${OUTPUT_BASE}/${RUN_NAME}}"
export EVOWEAVE_TRAIN_ROWS="${JOB_TRAIN_ROWS:-$(wc -l < "${EVOWEAVE_TRAIN_MANIFEST}")}"
export PYTHONPATH="${MODEL_ROOT}/rigweave/src:${EVOWEAVE_UNIRIG_ROOT}:${PUPPETEER_ROOT}/skeleton:${PYTHONPATH:-}"

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

# Joint-token decoder profile. This is not a conservative low-LR finetune:
# the decoder and Evoweave condition modules use the same baseline LR whether
# initialized from Puppeteer weights or trained from scratch.
export RIGWEAVE_LR_SURFACE="${JOB_LR_SURFACE:-1e-4}"
export RIGWEAVE_LR_MOTION="${JOB_LR_MOTION:-1e-4}"
export RIGWEAVE_LR_PREFIX="${JOB_LR_PREFIX:-1e-4}"
export RIGWEAVE_LR_DECODER="${JOB_LR_DECODER:-1e-4}"
export RIGWEAVE_WEIGHT_DECAY="${JOB_WEIGHT_DECAY:-0.04}"
export RIGWEAVE_SCHEDULER="${JOB_SCHEDULER:-onecycle}"
export RIGWEAVE_ONECYCLE_PCT_START="${JOB_ONECYCLE_PCT_START:-0.1}"
export RIGWEAVE_ONECYCLE_DIV_FACTOR="${JOB_ONECYCLE_DIV_FACTOR:-5.0}"
export RIGWEAVE_ONECYCLE_FINAL_DIV_FACTOR="${JOB_ONECYCLE_FINAL_DIV_FACTOR:-10.0}"

export RIGWEAVE_N_DISCRETE_SIZE="${JOB_N_DISCRETE_SIZE:-128}"
# The default keeps the current checkpoint-compatible data subset. It is a
# route configuration, not a data-processing rule.
export RIGWEAVE_N_MAX_JOINTS="${JOB_N_MAX_JOINTS:-101}"
export RIGWEAVE_TARGET_COORD_SCALE="${JOB_TARGET_COORD_SCALE:-0.25}"
export RIGWEAVE_COND_LENGTH="${JOB_COND_LENGTH:-257}"
export RIGWEAVE_PREFLIGHT_CONTRACT_MAX_POSITIONS="${RIGWEAVE_PREFLIGHT_CONTRACT_MAX_POSITIONS:-32}"
# FlashAttention bf16 is not bit-exact between full teacher-forcing and
# incremental-prefix sequence lengths. Real prefix/index bugs are much larger
# than this tolerance; measured one-token shifts are about 0.4 max logit diff.
export RIGWEAVE_PREFLIGHT_CONTRACT_MAX_DIFF="${RIGWEAVE_PREFLIGHT_CONTRACT_MAX_DIFF:-3.0e-2}"

export RIGWEAVE_FRAMES="${JOB_FRAMES:-24}"
export RIGWEAVE_SURFACE_SAMPLES="${JOB_SURFACE_SAMPLES:-65536}"
export RIGWEAVE_VERTEX_SAMPLES="${JOB_VERTEX_SAMPLES:-8192}"
export RIGWEAVE_QUERY_TOKENS="${JOB_QUERY_TOKENS:-1024}"
export RIGWEAVE_REGISTER_TOKENS="${JOB_REGISTER_TOKENS:-96}"
export RIGWEAVE_MOTION_DEPTH="${JOB_MOTION_DEPTH:-12}"
export RIGWEAVE_MOTION_HEADS="${JOB_MOTION_HEADS:-8}"
export RIGWEAVE_MOTION_FPS_RATIO="${JOB_MOTION_FPS_RATIO:-0.7}"
export RIGWEAVE_MOTION_VERTEX_SAMPLES="${JOB_MOTION_VERTEX_SAMPLES:-512}"

export RANDOM_INIT="${RANDOM_INIT:-${RANDOM_INIT_SMOKE:-0}}"
export TINY_RANDOM_DECODER="${TINY_RANDOM_DECODER:-0}"
if [[ "${RANDOM_INIT}" != "1" && -z "${PUPPETEER_CHECKPOINT}" ]]; then
  echo "[puppeteer baseline] ERROR: set PUPPETEER_CHECKPOINT/JOB_PUPPETEER_CHECKPOINT for optional pretrained init, or RANDOM_INIT=1 for from-scratch training." >&2
  exit 2
fi

mkdir -p "${EVOWEAVE_OUTPUT_DIR}"
cp -f "${SCRIPT_PATH}" "${EVOWEAVE_OUTPUT_DIR}/launch.sh"
python3 - <<'PY_ENV' > "${EVOWEAVE_OUTPUT_DIR}/resolved_env.txt"
import os

prefixes = ("EVOWEAVE", "RIGWEAVE", "JOB_", "PUPPETEER", "RANDOM_INIT", "TINY_RANDOM")
exact = {"MODEL_ROOT", "MANIFEST_ROOT", "RUN_NAME"}
for key in sorted(os.environ):
    if key in exact or key.startswith(prefixes):
        print(f"{key}={os.environ[key]}")
PY_ENV

CMD=(
  torchrun
  --standalone
  --nproc_per_node "${RIGWEAVE_NPROC}"
  rigweave/scripts/train_puppeteer_dynamic_rig.py
  --train-manifest "${EVOWEAVE_TRAIN_MANIFEST}"
  --val-manifest "${EVOWEAVE_VAL_MANIFEST}"
  --tokenizer-config "${EVOWEAVE_TOKENIZER_CONFIG}"
  --model-config "${EVOWEAVE_MODEL_CONFIG}"
  --unirig-checkpoint "${EVOWEAVE_UNIRIG_CKPT}"
  --puppeteer-root "${PUPPETEER_ROOT}"
  --puppeteer-llm "${PUPPETEER_LLM}"
  --output-dir "${EVOWEAVE_OUTPUT_DIR}"
  --frames "${RIGWEAVE_FRAMES}"
  --surface-samples "${RIGWEAVE_SURFACE_SAMPLES}"
  --vertex-samples "${RIGWEAVE_VERTEX_SAMPLES}"
  --query-tokens "${RIGWEAVE_QUERY_TOKENS}"
  --register-tokens "${RIGWEAVE_REGISTER_TOKENS}"
  --motion-depth "${RIGWEAVE_MOTION_DEPTH}"
  --motion-heads "${RIGWEAVE_MOTION_HEADS}"
  --motion-fps-ratio "${RIGWEAVE_MOTION_FPS_RATIO}"
  --motion-vertex-samples "${RIGWEAVE_MOTION_VERTEX_SAMPLES}"
  --batch-size "${RIGWEAVE_BATCH_SIZE}"
  --grad-accum-steps "${RIGWEAVE_GRAD_ACCUM}"
  --num-workers "${RIGWEAVE_NUM_WORKERS}"
  --max-steps "${RIGWEAVE_MAX_STEPS}"
  --sample-milestones "${RIGWEAVE_SAMPLE_MILESTONES}"
  --log-every "${RIGWEAVE_LOG_EVERY}"
  --val-every "${RIGWEAVE_VAL_EVERY}"
  --val-steps "${RIGWEAVE_VAL_STEPS}"
  --save-every "${RIGWEAVE_SAVE_EVERY}"
  --lr-surface "${RIGWEAVE_LR_SURFACE}"
  --lr-motion "${RIGWEAVE_LR_MOTION}"
  --lr-prefix "${RIGWEAVE_LR_PREFIX}"
  --lr-decoder "${RIGWEAVE_LR_DECODER}"
  --weight-decay "${RIGWEAVE_WEIGHT_DECAY}"
  --scheduler "${RIGWEAVE_SCHEDULER}"
  --onecycle-pct-start "${RIGWEAVE_ONECYCLE_PCT_START}"
  --onecycle-div-factor "${RIGWEAVE_ONECYCLE_DIV_FACTOR}"
  --onecycle-final-div-factor "${RIGWEAVE_ONECYCLE_FINAL_DIV_FACTOR}"
  --n-discrete-size "${RIGWEAVE_N_DISCRETE_SIZE}"
  --n-max-joints "${RIGWEAVE_N_MAX_JOINTS}"
  --target-coord-scale "${RIGWEAVE_TARGET_COORD_SCALE}"
  --cond-length "${RIGWEAVE_COND_LENGTH}"
  --amp-dtype "${RIGWEAVE_AMP_DTYPE:-bf16}"
)

if [[ -n "${PUPPETEER_CHECKPOINT}" ]]; then
  CMD+=(--puppeteer-checkpoint "${PUPPETEER_CHECKPOINT}")
fi
if [[ "${RANDOM_INIT}" == "1" ]]; then
  CMD+=(--random-init)
fi
if [[ "${TINY_RANDOM_DECODER}" == "1" ]]; then
  CMD+=(--tiny-random-decoder)
fi
if [[ "${RIGWEAVE_MOTION_CHECKPOINTING:-1}" == "1" ]]; then
  CMD+=(--motion-checkpointing)
fi
if [[ "${RIGWEAVE_DECODER_CHECKPOINTING:-0}" == "1" ]]; then
  CMD+=(--decoder-checkpointing)
fi
if [[ "${RIGWEAVE_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  CMD+=(--preflight-only)
fi
if [[ "${RIGWEAVE_PREFLIGHT_FORWARD:-0}" == "1" ]]; then
  CMD+=(--preflight-forward)
fi
if [[ "${RIGWEAVE_PREFLIGHT_CONTRACT_SANITY:-0}" == "1" ]]; then
  CMD+=(--preflight-contract-sanity)
  CMD+=(--preflight-contract-max-positions "${RIGWEAVE_PREFLIGHT_CONTRACT_MAX_POSITIONS}")
  CMD+=(--preflight-contract-max-diff "${RIGWEAVE_PREFLIGHT_CONTRACT_MAX_DIFF}")
fi

echo "[puppeteer baseline] start $(date -Is)"
echo "[puppeteer baseline] model_root=${MODEL_ROOT}"
echo "[puppeteer baseline] manifest_root=${MANIFEST_ROOT}"
echo "[puppeteer baseline] train_manifest=${EVOWEAVE_TRAIN_MANIFEST}"
echo "[puppeteer baseline] valid_manifest=${EVOWEAVE_VAL_MANIFEST}"
echo "[puppeteer baseline] output=${EVOWEAVE_OUTPUT_DIR}"
echo "[puppeteer baseline] checkpoint=${PUPPETEER_CHECKPOINT:-<random-init>}"
echo "[puppeteer baseline] effective_batch=$((RIGWEAVE_NPROC * RIGWEAVE_BATCH_SIZE * RIGWEAVE_GRAD_ACCUM))"
echo "[puppeteer baseline] profile=SkeletonOPT joint-token parent-index; tail_tokens=off; full_finetune=on"
printf '[puppeteer baseline] command:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}" 2>&1 | tee "${EVOWEAVE_OUTPUT_DIR}/train_frontend.log"
echo "[puppeteer baseline] done $(date -Is)"
