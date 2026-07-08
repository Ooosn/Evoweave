#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-/ssdwork/liuhaohan/evorig/evoweave_model_training_20260706}"
RUN_DIR="${RUN_DIR:-/ssdwork/liuhaohan/evoweave/outputs/dynamic_rig_runs/rootless_flat_unirig_motion_fullft_20260707_hxr4gpu}"
MANIFEST_ROOT="${MANIFEST_ROOT:-/ssdwork/liuhaohan/evorig/evoweave_rebuild_rootless_v3_20260706/quality_distributions/rootless_bbox_consistency/final_manifests}"
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/checkpoint_best_val.pt}"

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

export EVOWEAVE_OUTPUT_DIR="${RUN_DIR}"
export EVOWEAVE_TRAIN_MANIFEST="${MANIFEST_ROOT}/train_manifest.jsonl"
export EVOWEAVE_VAL_MANIFEST="${MANIFEST_ROOT}/valid_manifest.jsonl"
export EVOWEAVE_TEST_MANIFEST="${EVOWEAVE_VAL_MANIFEST}"
export EVOWEAVE_TRAIN_ROWS=15903
export EVOWEAVE_UNIRIG_ROOT="${EVOWEAVE_UNIRIG_ROOT:-/ssdwork/liuhaohan/evorig/evoweave_repo/external/UniRig}"
export EVOWEAVE_MODEL_CONFIG="${EVOWEAVE_MODEL_CONFIG:-${EVOWEAVE_UNIRIG_ROOT}/configs/model/unirig_ar_350m_1024_81920_float32.yaml}"
export EVOWEAVE_TOKENIZER_CONFIG="${EVOWEAVE_TOKENIZER_CONFIG:-${EVOWEAVE_UNIRIG_ROOT}/configs/tokenizer/tokenizer_parts_articulationxl_256.yaml}"
export EVOWEAVE_UNIRIG_CKPT="${EVOWEAVE_UNIRIG_CKPT:-/ssdwork/liuhaohan/evorig/evoweave_repo/external/UniRig_hf/skeleton/articulation-xl_quantization_256/model.ckpt}"
export EVOWEAVE_TRAIN_LOG="${RUN_DIR}/train_frontend.log"
export PYTHONPATH="${MODEL_ROOT}/rigweave/src:${EVOWEAVE_UNIRIG_ROOT}:${PYTHONPATH:-}"

export RIGWEAVE_NPROC=1
export RIGWEAVE_EVAL_CE_LIMIT="${RIGWEAVE_EVAL_CE_LIMIT:-857}"
export RIGWEAVE_EVAL_EOS_LIMIT="${RIGWEAVE_EVAL_EOS_LIMIT:-857}"
export RIGWEAVE_EVAL_GENERATION_LIMIT="${RIGWEAVE_EVAL_GENERATION_LIMIT:-64}"
export RIGWEAVE_EVAL_DIAGNOSE_LIMIT="${RIGWEAVE_EVAL_DIAGNOSE_LIMIT:-32}"
export RIGWEAVE_EVAL_MAX_NEW_TOKENS="${RIGWEAVE_EVAL_MAX_NEW_TOKENS:-1400}"
export RIGWEAVE_EVAL_CONTROLS="${RIGWEAVE_EVAL_CONTROLS:-normal}"
export RIGWEAVE_EVAL_DYNAMIC_DECODE_MODE="${RIGWEAVE_EVAL_DYNAMIC_DECODE_MODE:-both}"
export EVOWEAVE_EVAL_OUTPUT_DIR="${EVOWEAVE_EVAL_OUTPUT_DIR:-${RUN_DIR}/eval_valid64plus/best_val_normal}"

echo "[eval rootless flat] checkpoint=${CHECKPOINT}"
echo "[eval rootless flat] output=${EVOWEAVE_EVAL_OUTPUT_DIR}"
echo "[eval rootless flat] val_manifest=${EVOWEAVE_VAL_MANIFEST}"
echo "[eval rootless flat] limits ce=${RIGWEAVE_EVAL_CE_LIMIT} eos=${RIGWEAVE_EVAL_EOS_LIMIT} gen=${RIGWEAVE_EVAL_GENERATION_LIMIT} diagnose=${RIGWEAVE_EVAL_DIAGNOSE_LIMIT}"
echo "[eval rootless flat] controls=${RIGWEAVE_EVAL_CONTROLS}"

bash rigweave/scripts/run_dynamic_ar_eval_suite.sh "${CHECKPOINT}"
