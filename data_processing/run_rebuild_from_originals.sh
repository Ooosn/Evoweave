#!/usr/bin/env bash
set -euo pipefail

# Rebuild Evoweave data from preserved original files.
#
# Layering:
# 0. source manifests: no filtering, only original-file bookkeeping.
# 1. data reading / Pass1: Blender reads originals and writes raw dynamic NPZ.
#    Export failures are the first reject layer.
# 2. dynamic sequence dedupe.
# 3. Pass1 precheck: schema/shape/finite/skin/LBS readability checks only.
#    Root policy and training-target quality are not decided on raw Pass1.
# 4. combine accepted precheck manifests across source buckets.
# 5. rootless rewrite: remove unskinned top root/RT chains, reject rootless
#    forests, and keep only mesh components controlled by the retained
#    skeleton. This writes rootless v3 training-target NPZ without per-bone tail
#    endpoint coordinate fields; deleted root/RT transforms are record-only.
# 6. rootless target strict: first formal hard reject on the actual training
#    target fields.
# 7. final histogram screening: dynamic quality scores plus compact
#    skeleton/mesh bbox consistency distributions. The final screened manifests
#    are train/valid only; original test rows are merged into train.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

TRANSFER_ROOT="${EVOWEAVE_TRANSFER_ROOT:-/ssdwork/liuhaohan/evoweave_full_transfer_download/evoweave_full_transfer_20260623/evorig}"
TRANSFER_MANIFEST="${EVOWEAVE_TRANSFER_MANIFEST:-${TRANSFER_ROOT}/transfer/evoweave_full_transfer_20260623/full_transfer_manifest.csv}"
REBUILD_ROOT="${EVOWEAVE_REBUILD_ROOT:-/ssdwork/liuhaohan/evorig/evoweave_rebuild_clean}"

PYTHON="${PYTHON:-/opt/conda/bin/python3.8}"
BLENDER="${BLENDER:-blender}"
EVOWEAVE_UNIRIG_ROOT="${EVOWEAVE_UNIRIG_ROOT:-${REPO_ROOT}/../external/UniRig}"
export RIGWEAVE_DISABLE_OPEN3D="${RIGWEAVE_DISABLE_OPEN3D:-1}"
export PYTHONPATH="${REPO_ROOT}/src:${EVOWEAVE_UNIRIG_ROOT}/src:${PYTHONPATH:-}"

SEQUENCE_FRAMES="${EVOWEAVE_SEQUENCE_FRAMES:-40}"
MOTION_FPS_DESCRIPTOR_VERTICES="${EVOWEAVE_MOTION_FPS_DESCRIPTOR_VERTICES:-1024}"
DEDUP_FINGERPRINT_FRAMES="${EVOWEAVE_DEDUP_FINGERPRINT_FRAMES:-8}"
LIMIT="${EVOWEAVE_REBUILD_LIMIT:-0}"
TEXVERSE_WORKERS="${EVOWEAVE_TEXVERSE_PASS1_WORKERS:-4}"
TEXVERSE_PASS0_WORKERS="${EVOWEAVE_TEXVERSE_PASS0_WORKERS:-${TEXVERSE_WORKERS}}"
TEXVERSE_PASS0_TIMEOUT_SEC="${EVOWEAVE_TEXVERSE_PASS0_TIMEOUT_SEC:-180}"
TEXVERSE_PASS0_MAX_CANDIDATES="${EVOWEAVE_TEXVERSE_PASS0_MAX_CANDIDATES:-8}"
OXL_WORKERS="${EVOWEAVE_OXL_PASS1_WORKERS:-4}"
PRECHECK_WORKERS="${EVOWEAVE_PASS1_PRECHECK_WORKERS:-8}"
ROOTLESS_WORKERS="${EVOWEAVE_ROOTLESS_WORKERS:-12}"
AUDIT_WORKERS="${EVOWEAVE_AUDIT_WORKERS:-12}"
BLENDER_THREADS="${EVOWEAVE_BLENDER_THREADS:-2}"
PASS1_TIMEOUT_SEC="${EVOWEAVE_PASS1_TIMEOUT_SEC:-360}"
PASS1_TIMEOUT_RETRIES="${EVOWEAVE_PASS1_TIMEOUT_RETRIES:-1}"
MAX_JOINTS="${EVOWEAVE_MAX_JOINTS:-256}"
RAW_MIN_VERTICES="${EVOWEAVE_RAW_MIN_VERTICES:-1}"
MAX_VERTICES="${EVOWEAVE_MAX_VERTICES:-300000}"
MAX_FACES="${EVOWEAVE_MAX_FACES:-600000}"
RAW_BBOX_RATIO_HARD_MIN="${EVOWEAVE_RAW_BBOX_RATIO_HARD_MIN:-0.0}"
RAW_BBOX_RATIO_HARD_MAX="${EVOWEAVE_RAW_BBOX_RATIO_HARD_MAX:-0.0}"
RAW_MIN_MOTION_P95_BBOX="${EVOWEAVE_RAW_MIN_MOTION_P95_BBOX:-0.0}"
ACTIVE_SKIN_THRESHOLD="${EVOWEAVE_ACTIVE_SKIN_THRESHOLD:-1e-4}"
RAW_ACTIVE_SKIN_THRESHOLD="${EVOWEAVE_RAW_ACTIVE_SKIN_THRESHOLD:-0.0}"
MIN_COMPONENT_CONTROLLED_RATIO="${EVOWEAVE_MIN_COMPONENT_CONTROLLED_RATIO:-0.95}"
ROOTLESS_MIN_TARGET_JOINT_MESH_BBOX_CONSISTENCY="${EVOWEAVE_ROOTLESS_MIN_TARGET_JOINT_MESH_BBOX_CONSISTENCY:-0.4}"

TOKENIZER_CONFIG="${EVOWEAVE_TOKENIZER_CONFIG:-${EVOWEAVE_UNIRIG_ROOT}/configs/tokenizer/tokenizer_parts_articulationxl_256.yaml}"
if [[ ! -f "${TOKENIZER_CONFIG}" ]]; then
  echo "Missing UniRig tokenizer config: ${TOKENIZER_CONFIG}" >&2
  exit 3
fi
if [[ -n "${EVOWEAVE_7Z:-}" && ! -x "${EVOWEAVE_7Z}" ]]; then
  echo "EVOWEAVE_7Z is set but not executable: ${EVOWEAVE_7Z}" >&2
  exit 3
fi

SOURCE_DIR="${REBUILD_ROOT}/00_source_manifests"
mkdir -p "${SOURCE_DIR}"
cd "${REPO_ROOT}"

echo "[0] source manifests from transfer manifest"
"${PYTHON}" scripts/build_rebuild_source_manifests.py \
  --transfer-manifest-csv "${TRANSFER_MANIFEST}" \
  --transfer-root "${TRANSFER_ROOT}" \
  --out-dir "${SOURCE_DIR}" \
  --fail-on-missing-original

TEX_PASS0_LIMIT_ARGS=()
if [[ "${LIMIT}" -gt 0 ]]; then
  TEX_PASS0_LIMIT_ARGS+=(--sample-size "${LIMIT}" --seed 2026)
fi

run_source() {
  local source_name="$1"
  local pass1_manifest="$2"
  local pass1_root="$3"
  local dedup_root="$4"
  local precheck_root="$5"

  if [[ -f "${precheck_root}/summary.json" \
     && -f "${precheck_root}/accepted.jsonl" \
     && -f "${precheck_root}/rejected.jsonl" ]]; then
    existing_stage="$("${PYTHON}" - "${precheck_root}/summary.json" <<'PY'
import json
import sys
try:
    print(json.load(open(sys.argv[1], encoding="utf-8")).get("stage", ""))
except Exception:
    print("")
PY
)"
    if [[ "${existing_stage}" == "pass1_precheck" ]]; then
      echo "[2-3] ${source_name}: pass1 precheck output already complete, skipping dedup/precheck"
      return
    fi
    echo "[2-3] ${source_name}: existing precheck output is stage='${existing_stage}', rebuilding"
  fi

  echo "[2] ${source_name}: dedup dynamic sequence fingerprints"
  "${PYTHON}" scripts/dedupe_dynamic_sequences.py \
    --manifest-jsonl "${pass1_manifest}" \
    --fingerprint-frames "${DEDUP_FINGERPRINT_FRAMES}" \
    --out-keep-jsonl "${dedup_root}/sequence_unique_manifest.jsonl" \
    --out-duplicate-jsonl "${dedup_root}/sequence_duplicate_manifest.jsonl"

  echo "[3] ${source_name}: Pass1 readability precheck before rootless"
  "${PYTHON}" scripts/build_pass1_precheck_dataset.py \
    --manifest-jsonl "${dedup_root}/sequence_unique_manifest.jsonl" \
    --output-root "${precheck_root}" \
    --target-assets 0 \
    --min-frames "${SEQUENCE_FRAMES}" \
    --min-vertices 100 \
    --min-faces 20 \
    --min-joints 4 \
    --max-joints "${MAX_JOINTS}" \
    --max-lbs-recon-p95-bbox 0.05 \
    --max-lbs-recon-p99-bbox 0.15 \
    --max-lbs-recon-max-bbox 0.50 \
    --split-map-csv "${SOURCE_DIR}/split_map.csv" \
    --workers "${PRECHECK_WORKERS}"
}

TEX_ROOT="${REBUILD_ROOT}/texverse"
TEX_PASS0_AUDIT="${SOURCE_DIR}/texverse_pass0_audit.jsonl"
TEX_PASS0_SUMMARY="${SOURCE_DIR}/texverse_pass0_summary"

echo "[1a] texverse: Pass0 audit and source selection from zip originals"
"${PYTHON}" scripts/texverse_quality_audit.py \
  --model-paths "${SOURCE_DIR}/texverse_model_paths.txt" \
  --animation-ids "${SOURCE_DIR}/texverse_animation_ids.txt" \
  --download-dir "${TRANSFER_ROOT}/evoweave_texverse_full/pass0_wide/zips" \
  --extract-root "${TEX_ROOT}/pass0_extract" \
  --output-jsonl "${TEX_PASS0_AUDIT}" \
  --blender "${BLENDER}" \
  --blender-threads "${BLENDER_THREADS}" \
  --timeout-sec "${TEXVERSE_PASS0_TIMEOUT_SEC}" \
  --workers "${TEXVERSE_PASS0_WORKERS}" \
  --max-candidates-per-zip "${TEXVERSE_PASS0_MAX_CANDIDATES}" \
  --resume \
  "${TEX_PASS0_LIMIT_ARGS[@]}"
"${PYTHON}" scripts/summarize_texverse_audit.py \
  --input-jsonl "${TEX_PASS0_AUDIT}" \
  --out-dir "${TEX_PASS0_SUMMARY}" \
  --name "texverse_pass0"

echo "[1b] texverse: Blender export raw dynamic NPZ from selected source files"
TEX_PASS1="${TEX_ROOT}/pass1_motionfps_sequence${SEQUENCE_FRAMES}"
mkdir -p "${TEX_PASS1}"
"${PYTHON}" scripts/export_texverse_pass1_batch.py \
  --audit-jsonl "${TEX_PASS0_AUDIT}" \
  --zip-dir "${TRANSFER_ROOT}/evoweave_texverse_full/pass0_wide/zips" \
  --out-root "${TEX_PASS1}" \
  --extract-root "${TEX_PASS1}/extract" \
  --manifest-jsonl "${TEX_PASS1}/manifest.jsonl" \
  --blender "${BLENDER}" \
  --blender-threads "${BLENDER_THREADS}" \
  --frames "${SEQUENCE_FRAMES}" \
  --motion-fps-descriptor-vertices "${MOTION_FPS_DESCRIPTOR_VERTICES}" \
  --timeout-sec "${PASS1_TIMEOUT_SEC}" \
  --workers "${TEXVERSE_WORKERS}" \
  --limit "${LIMIT}" \
  --max-joints "${MAX_JOINTS}" \
  --min-vertices "${RAW_MIN_VERTICES}" \
  --max-vertices "${MAX_VERTICES}" \
  --max-faces "${MAX_FACES}" \
  --bbox-ratio-hard-min "${RAW_BBOX_RATIO_HARD_MIN}" \
  --bbox-ratio-hard-max "${RAW_BBOX_RATIO_HARD_MAX}" \
  --min-motion-p95-bbox "${RAW_MIN_MOTION_P95_BBOX}" \
  --active-skin-threshold "${RAW_ACTIVE_SKIN_THRESHOLD}" \
  --shuffle \
  --seed 2026
run_source "texverse" "${TEX_PASS1}/manifest.jsonl" "${TEX_PASS1}" "${TEX_ROOT}/sequence_dedup" "${TEX_ROOT}/pass1_precheck"

for source_name in objaverse_xl objaverse_xl_more_anim; do
  if [[ "${source_name}" == "objaverse_xl" ]]; then
    manifest="${SOURCE_DIR}/objaverse_xl_download_rebuild_manifest.jsonl"
    workers="${OXL_WORKERS}"
  else
    manifest="${SOURCE_DIR}/objaverse_xl_more_anim_download_rebuild_manifest.jsonl"
    workers="${OXL_WORKERS}"
  fi
  SRC_ROOT="${REBUILD_ROOT}/${source_name}"
  PASS1_ROOT="${SRC_ROOT}/pass1_motionfps_sequence${SEQUENCE_FRAMES}"
  mkdir -p "${PASS1_ROOT}"
  echo "[1] ${source_name}: Blender export raw dynamic NPZ from raw originals"
  "${PYTHON}" scripts/export_objaverse_xl_pass1_batch.py \
    --download-manifest-jsonl "${manifest}" \
    --out-root "${PASS1_ROOT}" \
    --manifest-jsonl "${PASS1_ROOT}/manifest.jsonl" \
    --blender "${BLENDER}" \
    --blender-threads "${BLENDER_THREADS}" \
    --frames "${SEQUENCE_FRAMES}" \
    --motion-fps-descriptor-vertices "${MOTION_FPS_DESCRIPTOR_VERTICES}" \
    --timeout-sec "${PASS1_TIMEOUT_SEC}" \
    --timeout-retries "${PASS1_TIMEOUT_RETRIES}" \
    --workers "${workers}" \
    --limit "${LIMIT}" \
    --max-joints "${MAX_JOINTS}" \
    --min-vertices "${RAW_MIN_VERTICES}" \
    --max-vertices "${MAX_VERTICES}" \
    --max-faces "${MAX_FACES}" \
    --bbox-ratio-hard-min "${RAW_BBOX_RATIO_HARD_MIN}" \
    --bbox-ratio-hard-max "${RAW_BBOX_RATIO_HARD_MAX}" \
    --min-motion-p95-bbox "${RAW_MIN_MOTION_P95_BBOX}" \
    --active-skin-threshold "${RAW_ACTIVE_SKIN_THRESHOLD}"
  run_source "${source_name}" "${PASS1_ROOT}/manifest.jsonl" "${PASS1_ROOT}" "${SRC_ROOT}/sequence_dedup" "${SRC_ROOT}/pass1_precheck"
done

echo "[4] combine Pass1 precheck manifests with cross-source dedup"
COMBINED="${REBUILD_ROOT}/combined_pass1_precheck"
"${PYTHON}" scripts/combine_dynamic_precheck_manifests.py \
  --source "texverse=${TEX_ROOT}/pass1_precheck" \
  --source "objaverse_xl=${REBUILD_ROOT}/objaverse_xl/pass1_precheck" \
  --source "objaverse_xl_more_anim=${REBUILD_ROOT}/objaverse_xl_more_anim/pass1_precheck" \
  --output-dir "${COMBINED}" \
  --check-paths

echo "[5] rootless structural rewrite NPZ"
ROOTLESS="${REBUILD_ROOT}/rootless_clean"
"${PYTHON}" scripts/build_rootless_dynamic_npz.py \
  --manifest "${COMBINED}/train_manifest.jsonl" \
  --manifest "${COMBINED}/val_manifest.jsonl" \
  --manifest "${COMBINED}/test_manifest.jsonl" \
  --output-root "${ROOTLESS}" \
  --active-skin-threshold "${ACTIVE_SKIN_THRESHOLD}" \
  --min-component-controlled-ratio "${MIN_COMPONENT_CONTROLLED_RATIO}" \
  --compression compressed \
  --verify-vertices 256 \
  --workers "${ROOTLESS_WORKERS}" \
  --allow-build-rejects

for split in train val test; do
  cp -f "${ROOTLESS}/${split}_manifest.jsonl" "${ROOTLESS}/${split}_manifest.westlake.jsonl"
done

echo "[6] first formal hard reject on rootless target fields"
ROOTLESS_STRICT="${REBUILD_ROOT}/rootless_target_strict"
"${PYTHON}" scripts/build_rootless_target_strict_dataset.py \
  --manifest "${ROOTLESS}/train_manifest.jsonl" \
  --manifest "${ROOTLESS}/val_manifest.jsonl" \
  --manifest "${ROOTLESS}/test_manifest.jsonl" \
  --output-root "${ROOTLESS_STRICT}" \
  --materialize-mode hardlink \
  --workers "${AUDIT_WORKERS}" \
  --sample-vertices 256 \
  --min-frames "${SEQUENCE_FRAMES}" \
  --min-vertices 100 \
  --min-faces 20 \
  --min-target-joints 4 \
  --min-active-target-joints 1 \
  --min-motion-coverage-score 0.15 \
  --min-motion-amount-score 0.15 \
  --min-bbox-stability-score 0.50 \
  --min-edge-stretch-stability-score 0.50 \
  --min-edge-collapse-stability-score 0.20 \
  --min-spike-cleanliness-score 0.50 \
  --max-lbs-recon-p95-bbox 0.02 \
  --quality-gate-mode hard

echo "[7] rootless final validation and final histogram screening"
AUDIT="${REBUILD_ROOT}/rootless_quality/audit"
QUALITY="${REBUILD_ROOT}/quality_distributions"
ROOTLESS_DYNAMIC_QUALITY="${QUALITY}/rootless_dynamic_scores"
ROOTLESS_BBOX_QUALITY="${QUALITY}/rootless_bbox_consistency"
"${PYTHON}" scripts/audit_rootless_final_dataset.py \
  --dataset "${ROOTLESS_STRICT}" \
  --out-dir "${AUDIT}" \
  --sample-vertices 128 \
  --workers "${AUDIT_WORKERS}" \
  --no-figures
"${PYTHON}" scripts/validate_rootless_dynamic_npz.py \
  --manifest "${ROOTLESS_STRICT}/train_manifest.jsonl" \
  --manifest "${ROOTLESS_STRICT}/val_manifest.jsonl" \
  --manifest "${ROOTLESS_STRICT}/test_manifest.jsonl" \
  --out-json "${ROOTLESS_STRICT}/rootless_validation.json" \
  --sample-vertices 256 \
  --workers "${AUDIT_WORKERS}"
"${PYTHON}" scripts/build_rootless_quality_score_distributions.py \
  --manifest "${ROOTLESS_STRICT}/train_manifest.jsonl" \
  --manifest "${ROOTLESS_STRICT}/val_manifest.jsonl" \
  --manifest "${ROOTLESS_STRICT}/test_manifest.jsonl" \
  --out-dir "${ROOTLESS_DYNAMIC_QUALITY}" \
  --workers "${AUDIT_WORKERS}" \
  --bins 80
"${PYTHON}" scripts/build_rootless_bbox_consistency_distributions.py \
  --manifest "${ROOTLESS_STRICT}/train_manifest.jsonl" \
  --manifest "${ROOTLESS_STRICT}/val_manifest.jsonl" \
  --manifest "${ROOTLESS_STRICT}/test_manifest.jsonl" \
  --out-dir "${ROOTLESS_BBOX_QUALITY}" \
  --active-skin-threshold "${ACTIVE_SKIN_THRESHOLD}" \
  --min-target-joint-mesh-bbox-consistency "${ROOTLESS_MIN_TARGET_JOINT_MESH_BBOX_CONSISTENCY}" \
  --workers "${AUDIT_WORKERS}" \
  --bins 80

cat > "${REBUILD_ROOT}/README_REBUILD_OUTPUTS.txt" <<EOF
source manifests: ${SOURCE_DIR}
pass1 raw NPZ roots: ${TEX_PASS1}, ${REBUILD_ROOT}/objaverse_xl/pass1_motionfps_sequence${SEQUENCE_FRAMES}, ${REBUILD_ROOT}/objaverse_xl_more_anim/pass1_motionfps_sequence${SEQUENCE_FRAMES}
combined Pass1 precheck manifests: ${COMBINED}
rootless derived cache: ${ROOTLESS}
rootless target strict dataset: ${ROOTLESS_STRICT}
rootless quality metrics: ${AUDIT}/rootless_final_metrics.csv
final dynamic quality histogram screening: ${ROOTLESS_DYNAMIC_QUALITY}
final skeleton/mesh bbox consistency histogram screening: ${ROOTLESS_BBOX_QUALITY}
final bbox-consistency screened manifests: ${ROOTLESS_BBOX_QUALITY}/final_manifests
final training manifest: ${ROOTLESS_BBOX_QUALITY}/final_manifests/train_manifest.jsonl
final validation manifest: ${ROOTLESS_BBOX_QUALITY}/final_manifests/valid_manifest.jsonl
final split policy: train/valid only; original test rows are merged into train
EOF

echo "[done] rebuild root: ${REBUILD_ROOT}"
