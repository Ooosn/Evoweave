#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   conda activate gs3
#   source rigweave/configs/evoweave_current.env
#   bash rigweave/scripts/run_dynamic_ar_train.sh
#
# Optional:
#   bash rigweave/scripts/run_dynamic_ar_train.sh /path/to/env_file

ENV_FILE_ARG="${1:-}"
if [[ $# -gt 0 ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE_ARG}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVOWEAVE_ROOT="${EVOWEAVE_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${EVOWEAVE_ROOT}"

: "${EVOWEAVE_TRAIN_MANIFEST:?set EVOWEAVE_TRAIN_MANIFEST}"
: "${EVOWEAVE_VAL_MANIFEST:?set EVOWEAVE_VAL_MANIFEST}"
: "${EVOWEAVE_OUTPUT_DIR:?set EVOWEAVE_OUTPUT_DIR}"
: "${EVOWEAVE_UNIRIG_ROOT:?set EVOWEAVE_UNIRIG_ROOT}"
: "${EVOWEAVE_UNIRIG_CKPT:?set EVOWEAVE_UNIRIG_CKPT}"
: "${EVOWEAVE_TRAIN_ROWS:?set EVOWEAVE_TRAIN_ROWS}"

export EVOWEAVE_UNIRIG_ROOT
export PYTHONPATH="${EVOWEAVE_ROOT}/rigweave/src:${EVOWEAVE_UNIRIG_ROOT}:${PYTHONPATH:-}"
export RIGWEAVE_DISABLE_OPEN3D="${RIGWEAVE_DISABLE_OPEN3D:-1}"
export RIGWEAVE_DISABLE_LIGHTNING_IMPORT="${RIGWEAVE_DISABLE_LIGHTNING_IMPORT:-0}"
export RIGWEAVE_LOG_PROCESS_MEMORY="${RIGWEAVE_LOG_PROCESS_MEMORY:-0}"

NPROC="${RIGWEAVE_NPROC:-2}"
BATCH_SIZE="${RIGWEAVE_BATCH_SIZE:-3}"
GRAD_ACCUM="${RIGWEAVE_GRAD_ACCUM:-8}"
MAX_STEPS="${RIGWEAVE_MAX_STEPS:-1667}"
STOP_AFTER_SAMPLES="${RIGWEAVE_STOP_AFTER_SAMPLES:-0}"
SAMPLE_MILESTONES="${RIGWEAVE_SAMPLE_MILESTONES:-5000,10000,20000,30000,50000,80000}"
MODEL_CONFIG="${EVOWEAVE_MODEL_CONFIG:-${EVOWEAVE_UNIRIG_ROOT}/configs/model/unirig_ar_350m_1024_81920_float32.yaml}"
TOKENIZER_CONFIG="${EVOWEAVE_TOKENIZER_CONFIG:-${EVOWEAVE_UNIRIG_ROOT}/configs/tokenizer/tokenizer_parts_articulationxl_256.yaml}"

if [[ "${RIGWEAVE_SKIP_PREFLIGHT:-0}" != "1" ]]; then
  echo "[evoweave train] preflight"
  PREFLIGHT_CMD=(python rigweave/scripts/preflight_dynamic_ar_run.py)
  if [[ "${RIGWEAVE_REQUIRE_CUDA:-}" == "1" || ( "${RIGWEAVE_PREFLIGHT_ONLY:-0}" != "1" && "${RIGWEAVE_REQUIRE_CUDA:-1}" != "0" ) ]]; then
    PREFLIGHT_CMD+=(--require-cuda --min-cuda-devices "${NPROC}")
  elif [[ "${RIGWEAVE_PREFLIGHT_REQUIRE_CUDA:-0}" == "1" ]]; then
    PREFLIGHT_CMD+=(--require-cuda --min-cuda-devices "${NPROC}")
  fi
  "${PREFLIGHT_CMD[@]}"
fi

if [[ "${RIGWEAVE_SKIP_DATALOADER_CHECK:-0}" != "1" ]]; then
  echo "[evoweave train] dataloader check"
  python rigweave/scripts/check_dynamic_dataloader.py \
    --env-file "${EVOWEAVE_ENV_FILE:-${ENV_FILE_ARG:-rigweave/configs/evoweave_current.env}}" \
    --manifest "${EVOWEAVE_TRAIN_MANIFEST}" \
    --sample-mode "${RIGWEAVE_DATALOADER_CHECK_SAMPLE_MODE:-linspace}" \
    --limit "${RIGWEAVE_DATALOADER_CHECK_LIMIT:-12}" \
    --batch-size "${RIGWEAVE_DATALOADER_CHECK_BATCH_SIZE:-4}" \
    --frames "${RIGWEAVE_FRAMES:-24}"
fi

if [[ "${RIGWEAVE_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "[evoweave train] preflight-only mode; not starting torchrun"
  exit 0
fi

CMD=(
  torchrun
  --standalone
  --nproc_per_node "${NPROC}"
  rigweave/scripts/train_dynamic_rig.py
  --train-manifest "${EVOWEAVE_TRAIN_MANIFEST}"
  --val-manifest "${EVOWEAVE_VAL_MANIFEST}"
  --tokenizer-config "${TOKENIZER_CONFIG}"
  --model-config "${MODEL_CONFIG}"
  --unirig-checkpoint "${EVOWEAVE_UNIRIG_CKPT}"
  --output-dir "${EVOWEAVE_OUTPUT_DIR}"
  --frames "${RIGWEAVE_FRAMES:-24}"
  --surface-samples "${RIGWEAVE_SURFACE_SAMPLES:-65536}"
  --vertex-samples "${RIGWEAVE_VERTEX_SAMPLES:-8192}"
  --query-tokens "${RIGWEAVE_QUERY_TOKENS:-1024}"
  --register-tokens "${RIGWEAVE_REGISTER_TOKENS:-96}"
  --motion-depth "${RIGWEAVE_MOTION_DEPTH:-12}"
  --motion-heads "${RIGWEAVE_MOTION_HEADS:-8}"
  --batch-size "${BATCH_SIZE}"
  --grad-accum-steps "${GRAD_ACCUM}"
  --num-workers "${RIGWEAVE_NUM_WORKERS:-0}"
  --lr-motion "${RIGWEAVE_LR_MOTION:-0.0001}"
  --lr-ar "${RIGWEAVE_LR_AR:-0.0001}"
  --lr-surface "${RIGWEAVE_LR_SURFACE:-0.0001}"
  --weight-decay "${RIGWEAVE_WEIGHT_DECAY:-0.04}"
  --scheduler "${RIGWEAVE_SCHEDULER:-onecycle}"
  --onecycle-pct-start "${RIGWEAVE_ONECYCLE_PCT_START:-0.1}"
  --onecycle-div-factor "${RIGWEAVE_ONECYCLE_DIV_FACTOR:-5.0}"
  --onecycle-final-div-factor "${RIGWEAVE_ONECYCLE_FINAL_DIV_FACTOR:-10.0}"
  --max-steps "${MAX_STEPS}"
  --stop-after-samples "${STOP_AFTER_SAMPLES}"
  --sample-milestones "${SAMPLE_MILESTONES}"
  --log-every "${RIGWEAVE_LOG_EVERY:-10}"
  --val-every "${RIGWEAVE_VAL_EVERY:-200}"
  --val-steps "${RIGWEAVE_VAL_STEPS:-16}"
  --save-every "${RIGWEAVE_SAVE_EVERY:-0}"
  --motion-fps-ratio "${RIGWEAVE_MOTION_FPS_RATIO:-0.7}"
  --motion-vertex-samples "${RIGWEAVE_MOTION_VERTEX_SAMPLES:-512}"
  --motion-alignment-policy "${RIGWEAVE_MOTION_ALIGNMENT_POLICY:-none}"
  --target-start-policy "${RIGWEAVE_TARGET_START_POLICY:-joint0}"
  --target-root-policy "${RIGWEAVE_TARGET_ROOT_POLICY:-legacy}"
  --branch-prior-proposals "${RIGWEAVE_BRANCH_PRIOR_PROPOSALS:-32}"
  --branch-prior-heads "${RIGWEAVE_BRANCH_PRIOR_HEADS:-8}"
  --branch-prior-loss-weight "${RIGWEAVE_BRANCH_PRIOR_LOSS_WEIGHT:-1.0}"
  --branch-prior-coord-loss-weight "${RIGWEAVE_BRANCH_PRIOR_COORD_LOSS_WEIGHT:-0.0}"
  --explicit-tree-loss-weight "${RIGWEAVE_EXPLICIT_TREE_LOSS_WEIGHT:-0.0}"
  --explicit-tree-generated-prefix-weight "${RIGWEAVE_EXPLICIT_TREE_GENERATED_PREFIX_WEIGHT:-0.0}"
  --explicit-tree-generated-prefix-states "${RIGWEAVE_EXPLICIT_TREE_GENERATED_PREFIX_STATES:-4}"
  --explicit-tree-generated-prefix-max-steps "${RIGWEAVE_EXPLICIT_TREE_GENERATED_PREFIX_MAX_STEPS:-64}"
  --explicit-tree-generated-prefix-max-rows "${RIGWEAVE_EXPLICIT_TREE_GENERATED_PREFIX_MAX_ROWS:-4}"
  --explicit-tree-generated-prefix-every "${RIGWEAVE_EXPLICIT_TREE_GENERATED_PREFIX_EVERY:-1}"
  --explicit-tree-oracle-prefix-weight "${RIGWEAVE_EXPLICIT_TREE_ORACLE_PREFIX_WEIGHT:-0.0}"
  --explicit-tree-oracle-prefix-states "${RIGWEAVE_EXPLICIT_TREE_ORACLE_PREFIX_STATES:-4}"
  --explicit-tree-oracle-prefix-max-steps "${RIGWEAVE_EXPLICIT_TREE_ORACLE_PREFIX_MAX_STEPS:-64}"
  --explicit-tree-oracle-prefix-max-rows "${RIGWEAVE_EXPLICIT_TREE_ORACLE_PREFIX_MAX_ROWS:-4}"
  --explicit-tree-prefix-jitter-weight "${RIGWEAVE_EXPLICIT_TREE_PREFIX_JITTER_WEIGHT:-0.0}"
  --explicit-tree-prefix-jitter-std "${RIGWEAVE_EXPLICIT_TREE_PREFIX_JITTER_STD:-0.0}"
  --explicit-tree-depth "${RIGWEAVE_EXPLICIT_TREE_DEPTH:-4}"
  --explicit-tree-heads "${RIGWEAVE_EXPLICIT_TREE_HEADS:-8}"
  --explicit-tree-topology-mode "${RIGWEAVE_EXPLICIT_TREE_TOPOLOGY_MODE:-geometry}"
  --explicit-tree-coordinate-mode "${RIGWEAVE_EXPLICIT_TREE_COORDINATE_MODE:-absolute}"
  --explicit-tree-action-eos-loss-weight "${RIGWEAVE_EXPLICIT_TREE_ACTION_EOS_LOSS_WEIGHT:-1.0}"
  --explicit-tree-action-child-loss-weight "${RIGWEAVE_EXPLICIT_TREE_ACTION_CHILD_LOSS_WEIGHT:-1.0}"
  --explicit-tree-action-branch-loss-weight "${RIGWEAVE_EXPLICIT_TREE_ACTION_BRANCH_LOSS_WEIGHT:-1.0}"
  --explicit-tree-xyz-loss-weight "${RIGWEAVE_EXPLICIT_TREE_XYZ_LOSS_WEIGHT:-1.0}"
  --eos-loss-weight "${RIGWEAVE_EOS_LOSS_WEIGHT:-1.0}"
  --decision-loss-weight "${RIGWEAVE_DECISION_LOSS_WEIGHT:-0.0}"
  --loop-recovery-loss-weight "${RIGWEAVE_LOOP_RECOVERY_LOSS_WEIGHT:-0.0}"
  --loop-recovery-repeats "${RIGWEAVE_LOOP_RECOVERY_REPEATS:-4}"
  --prefix-decision-recovery-weight "${RIGWEAVE_PREFIX_DECISION_RECOVERY_WEIGHT:-0.0}"
  --prefix-decision-recovery-states "${RIGWEAVE_PREFIX_DECISION_RECOVERY_STATES:-4}"
  --prefix-decision-recovery-variants "${RIGWEAVE_PREFIX_DECISION_RECOVERY_VARIANTS:-1}"
  --prefix-decision-recovery-jitter "${RIGWEAVE_PREFIX_DECISION_RECOVERY_JITTER:-4}"
  --prefix-token-recovery-weight "${RIGWEAVE_PREFIX_TOKEN_RECOVERY_WEIGHT:-0.0}"
  --prefix-token-recovery-states "${RIGWEAVE_PREFIX_TOKEN_RECOVERY_STATES:-4}"
  --prefix-token-recovery-variants "${RIGWEAVE_PREFIX_TOKEN_RECOVERY_VARIANTS:-1}"
  --prefix-token-recovery-jitter "${RIGWEAVE_PREFIX_TOKEN_RECOVERY_JITTER:-4}"
  --prefix-token-recovery-max-rows "${RIGWEAVE_PREFIX_TOKEN_RECOVERY_MAX_ROWS:-4}"
  --prefix-action-recovery-weight "${RIGWEAVE_PREFIX_ACTION_RECOVERY_WEIGHT:-0.0}"
  --prefix-action-recovery-states "${RIGWEAVE_PREFIX_ACTION_RECOVERY_STATES:-4}"
  --prefix-action-recovery-variants "${RIGWEAVE_PREFIX_ACTION_RECOVERY_VARIANTS:-1}"
  --prefix-action-recovery-jitter "${RIGWEAVE_PREFIX_ACTION_RECOVERY_JITTER:-4}"
  --prefix-action-recovery-max-rows "${RIGWEAVE_PREFIX_ACTION_RECOVERY_MAX_ROWS:-4}"
  --generated-prefix-recovery-weight "${RIGWEAVE_GENERATED_PREFIX_RECOVERY_WEIGHT:-0.0}"
  --generated-prefix-recovery-states "${RIGWEAVE_GENERATED_PREFIX_RECOVERY_STATES:-4}"
  --generated-prefix-recovery-max-new-tokens "${RIGWEAVE_GENERATED_PREFIX_RECOVERY_MAX_NEW_TOKENS:-128}"
  --generated-prefix-recovery-max-rows "${RIGWEAVE_GENERATED_PREFIX_RECOVERY_MAX_ROWS:-4}"
  --generated-prefix-recovery-every "${RIGWEAVE_GENERATED_PREFIX_RECOVERY_EVERY:-1}"
  --structure-count-loss-weight "${RIGWEAVE_STRUCTURE_COUNT_LOSS_WEIGHT:-0.0}"
  --structure-action-loss-weight "${RIGWEAVE_STRUCTURE_ACTION_LOSS_WEIGHT:-0.0}"
  --condition-control-ce-weight "${RIGWEAVE_CONDITION_CONTROL_CE_WEIGHT:-0.0}"
  --condition-control-ce-controls "${RIGWEAVE_CONDITION_CONTROL_CE_CONTROLS:-zero,shuffle}"
  --condition-control-ce-every "${RIGWEAVE_CONDITION_CONTROL_CE_EVERY:-1}"
  --condition-fusion "${RIGWEAVE_CONDITION_FUSION:-dynamic}"
  --condition-fusion-heads "${RIGWEAVE_CONDITION_FUSION_HEADS:-8}"
  --condition-fusion-gate-init "${RIGWEAVE_CONDITION_FUSION_GATE_INIT:-0.25}"
  --condition-fusion-depth "${RIGWEAVE_CONDITION_FUSION_DEPTH:-1}"
  --condition-static-blend-weight "${RIGWEAVE_CONDITION_STATIC_BLEND_WEIGHT:-0.0}"
  --amp-dtype "${RIGWEAVE_AMP_DTYPE:-bf16}"
)

if [[ -n "${EVOWEAVE_INIT_CHECKPOINT:-}" ]]; then
  CMD+=(--init-checkpoint "${EVOWEAVE_INIT_CHECKPOINT}")
fi
if [[ -n "${EVOWEAVE_RESUME_CHECKPOINT:-}" ]]; then
  CMD+=(--resume-checkpoint "${EVOWEAVE_RESUME_CHECKPOINT}")
fi
if [[ -n "${EVOWEAVE_INIT_CHECKPOINT:-}" && -n "${EVOWEAVE_RESUME_CHECKPOINT:-}" ]]; then
  echo "[evoweave train] ERROR: EVOWEAVE_INIT_CHECKPOINT and EVOWEAVE_RESUME_CHECKPOINT are mutually exclusive." >&2
  exit 2
fi

if [[ "${RIGWEAVE_MOTION_CHECKPOINTING:-1}" == "1" ]]; then
  CMD+=(--motion-checkpointing)
fi

if [[ "${RIGWEAVE_TRAIN_SURFACE_TOKENIZER:-1}" == "0" ]]; then
  CMD+=(--freeze-surface-tokenizer)
fi

if [[ "${RIGWEAVE_FREEZE_AR:-0}" == "1" ]]; then
  CMD+=(--freeze-ar)
fi

if [[ "${RIGWEAVE_FREEZE_CONDITIONER:-0}" == "1" ]]; then
  CMD+=(--freeze-conditioner)
fi

if [[ "${RIGWEAVE_FREEZE_BRANCH_PRIOR:-0}" == "1" ]]; then
  CMD+=(--freeze-branch-prior)
fi

if [[ "${RIGWEAVE_USE_GRAMMAR_STATE_EMBEDDING:-0}" == "1" ]]; then
  CMD+=(--use-grammar-state-embedding)
fi

if [[ "${RIGWEAVE_USE_ACTION_GROUP_BIAS:-0}" == "1" ]]; then
  CMD+=(--use-action-group-bias)
fi

if [[ "${RIGWEAVE_USE_CONDITION_ACTION_GROUP_BIAS:-0}" == "1" ]]; then
  CMD+=(--use-condition-action-group-bias)
fi

if [[ "${RIGWEAVE_RESET_CONDITION_FUSER:-0}" == "1" ]]; then
  CMD+=(--reset-condition-fuser)
fi

if [[ "${RIGWEAVE_NO_SAVE_OPTIMIZER:-0}" == "1" ]]; then
  CMD+=(--no-save-optimizer)
fi

echo "[evoweave train] root=${EVOWEAVE_ROOT}"
echo "[evoweave train] output=${EVOWEAVE_OUTPUT_DIR}"
echo "[evoweave train] nproc=${NPROC} micro_batch=${BATCH_SIZE} grad_accum=${GRAD_ACCUM}"
echo "[evoweave train] effective_batch=$((NPROC * BATCH_SIZE * GRAD_ACCUM))"
echo "[evoweave train] sample_milestones=${SAMPLE_MILESTONES}"
echo "[evoweave train] stop_after_samples=${STOP_AFTER_SAMPLES} scheduler_horizon_steps=${MAX_STEPS}"
echo "[evoweave train] explicit_tree_loss=${RIGWEAVE_EXPLICIT_TREE_LOSS_WEIGHT:-0.0} explicit_tree_gpr=${RIGWEAVE_EXPLICIT_TREE_GENERATED_PREFIX_WEIGHT:-0.0}:every${RIGWEAVE_EXPLICIT_TREE_GENERATED_PREFIX_EVERY:-1} explicit_tree_oracle=${RIGWEAVE_EXPLICIT_TREE_ORACLE_PREFIX_WEIGHT:-0.0} topology=${RIGWEAVE_EXPLICIT_TREE_TOPOLOGY_MODE:-geometry} coord=${RIGWEAVE_EXPLICIT_TREE_COORDINATE_MODE:-absolute} condition_action_group_bias=${RIGWEAVE_USE_CONDITION_ACTION_GROUP_BIAS:-0}"
echo "[evoweave train] generated_prefix_weight=${RIGWEAVE_GENERATED_PREFIX_RECOVERY_WEIGHT:-0.0} condition_control_ce=${RIGWEAVE_CONDITION_CONTROL_CE_WEIGHT:-0.0}:${RIGWEAVE_CONDITION_CONTROL_CE_CONTROLS:-zero,shuffle}:every${RIGWEAVE_CONDITION_CONTROL_CE_EVERY:-1}"
echo "[evoweave train] prefix_recovery token=${RIGWEAVE_PREFIX_TOKEN_RECOVERY_WEIGHT:-0.0} decision=${RIGWEAVE_PREFIX_DECISION_RECOVERY_WEIGHT:-0.0} action=${RIGWEAVE_PREFIX_ACTION_RECOVERY_WEIGHT:-0.0} loop=${RIGWEAVE_LOOP_RECOVERY_LOSS_WEIGHT:-0.0} structure_count=${RIGWEAVE_STRUCTURE_COUNT_LOSS_WEIGHT:-0.0}"
echo "[evoweave train] condition_fusion=${RIGWEAVE_CONDITION_FUSION:-dynamic} gate=${RIGWEAVE_CONDITION_FUSION_GATE_INIT:-0.25} depth=${RIGWEAVE_CONDITION_FUSION_DEPTH:-1} static_blend=${RIGWEAVE_CONDITION_STATIC_BLEND_WEIGHT:-0.0} reset=${RIGWEAVE_RESET_CONDITION_FUSER:-0}"
printf '[evoweave train] command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
