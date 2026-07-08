#!/usr/bin/env bash
set -euo pipefail

export PYTHON=/opt/conda/envs/evoweave/bin/python
export EVOWEAVE_UNIRIG_ROOT=/ssdwork/liuhaohan/evorig/evoweave_repo/external/UniRig
export PYTHONPATH=/ssdwork/liuhaohan/evorig/data_processing/src:${EVOWEAVE_UNIRIG_ROOT}/src:${PYTHONPATH:-}
export RIGWEAVE_DISABLE_OPEN3D=1

ROOT=/ssdwork/liuhaohan/evorig/evoweave_rebuild_rootless_v3_20260706
DP=/ssdwork/liuhaohan/evorig/data_processing
ROOTLESS=${ROOT}/rootless_clean
AUDIT=${ROOT}/rootless_quality/audit
QUALITY=${ROOT}/quality_distributions
ROOTLESS_DYNAMIC_QUALITY=${QUALITY}/rootless_dynamic_scores
ROOTLESS_BBOX_QUALITY=${QUALITY}/rootless_bbox_consistency
WORKERS=${EVOWEAVE_AUDIT_WORKERS:-16}
ACTIVE_SKIN_THRESHOLD=${EVOWEAVE_ACTIVE_SKIN_THRESHOLD:-1e-4}
ROOTLESS_MIN_TARGET_JOINT_MESH_BBOX_CONSISTENCY=${EVOWEAVE_ROOTLESS_MIN_TARGET_JOINT_MESH_BBOX_CONSISTENCY:-0.4}
LOG=/ssdwork/liuhaohan/evorig/logs/stage5_finish_20260706.log

mkdir -p /ssdwork/liuhaohan/evorig/logs
exec > >(tee -a "${LOG}") 2>&1

cd "${DP}"
echo "stage5_finish_start $(date -Is)"
rm -f "${ROOTLESS}/_tmp_bad_manifest.jsonl" "${ROOTLESS}/_tmp_bad_validation.json"

"${PYTHON}" scripts/audit_rootless_final_dataset.py \
  --dataset "${ROOTLESS}" \
  --out-dir "${AUDIT}" \
  --sample-vertices 128 \
  --workers "${WORKERS}" \
  --no-figures

"${PYTHON}" scripts/validate_rootless_dynamic_npz.py \
  --manifest "${ROOTLESS}/train_manifest.jsonl" \
  --manifest "${ROOTLESS}/val_manifest.jsonl" \
  --manifest "${ROOTLESS}/test_manifest.jsonl" \
  --out-json "${ROOTLESS}/rootless_validation.json" \
  --sample-vertices 256 \
  --workers "${WORKERS}"

"${PYTHON}" scripts/build_rootless_quality_score_distributions.py \
  --manifest "${ROOTLESS}/train_manifest.jsonl" \
  --manifest "${ROOTLESS}/val_manifest.jsonl" \
  --manifest "${ROOTLESS}/test_manifest.jsonl" \
  --out-dir "${ROOTLESS_DYNAMIC_QUALITY}" \
  --workers "${WORKERS}" \
  --bins 80

"${PYTHON}" scripts/build_rootless_bbox_consistency_distributions.py \
  --manifest "${ROOTLESS}/train_manifest.jsonl" \
  --manifest "${ROOTLESS}/val_manifest.jsonl" \
  --manifest "${ROOTLESS}/test_manifest.jsonl" \
  --out-dir "${ROOTLESS_BBOX_QUALITY}" \
  --active-skin-threshold "${ACTIVE_SKIN_THRESHOLD}" \
  --min-target-joint-mesh-bbox-consistency "${ROOTLESS_MIN_TARGET_JOINT_MESH_BBOX_CONSISTENCY}" \
  --workers "${WORKERS}" \
  --bins 80

cat > "${ROOT}/README_REBUILD_OUTPUTS.txt" <<EOF
source manifests: ${ROOT}/00_source_manifests
strict combined manifests: ${ROOT}/combined_strict
rootless clean dataset: ${ROOTLESS}
rootless quality metrics: ${AUDIT}/rootless_final_metrics.csv
final dynamic quality histogram screening: ${ROOTLESS_DYNAMIC_QUALITY}
final skeleton/mesh bbox consistency histogram screening: ${ROOTLESS_BBOX_QUALITY}
final bbox-consistency screened manifests: ${ROOTLESS_BBOX_QUALITY}/final_manifests
EOF

echo "stage5_finish_end $(date -Is)"
