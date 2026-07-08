#!/usr/bin/env bash
set -euo pipefail

REBUILD_ROOT="${EVOWEAVE_REBUILD_ROOT:-/ssdwork/liuhaohan/evorig/evoweave_rebuild_from_originals_20260702}"
REPO_ROOT="${EVOWEAVE_REPO_ROOT:-/ssdwork/liuhaohan/evorig/evoweave_repo/rigweave}"
PYTHON="${PYTHON:-/opt/conda/envs/evoweave/bin/python}"
ROOTLESS="${REBUILD_ROOT}/rootless_clean"
AUDIT="${REBUILD_ROOT}/rootless_quality/audit"
QUALITY="${REBUILD_ROOT}/quality_distributions"
STRICT_METRICS="${QUALITY}/strict_query_root_metrics.csv"
STRICT_QUALITY="${QUALITY}/strict_query_root_scores"
ROOTLESS_RAW_QUALITY="${QUALITY}/rootless_mesh_skeleton_raw"
LOG="${EVOWEAVE_CONTINUE_LOG:-/ssdwork/liuhaohan/evorig/logs/evoweave_rebuild_continue_20260702.log}"
MAIN_LOG="${EVOWEAVE_MAIN_REBUILD_LOG:-/ssdwork/liuhaohan/evorig/logs/evoweave_rebuild_from_originals_20260702_fixed_raw_rerun.log}"
WORKERS="${EVOWEAVE_AUDIT_WORKERS:-64}"
LOCK_FILE="${EVOWEAVE_CONTINUE_LOCK:-${REBUILD_ROOT}/.continue_20260702.lock}"

mkdir -p "$(dirname "${LOG}")"
exec > >(tee -a "${LOG}") 2>&1
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "[continue] another continuation process is active; exiting"
  exit 0
fi

echo "[continue] start $(date -Is)"
echo "[continue] rebuild root: ${REBUILD_ROOT}"
cd "${REPO_ROOT}"
export RIGWEAVE_DISABLE_OPEN3D="${RIGWEAVE_DISABLE_OPEN3D:-1}"
export EVOWEAVE_UNIRIG_ROOT="${EVOWEAVE_UNIRIG_ROOT:-${REPO_ROOT}/external/UniRig}"
export PYTHONPATH="${REPO_ROOT}/src:${EVOWEAVE_UNIRIG_ROOT}/src:${PYTHONPATH:-}"

if [[ ! -s "${ROOTLESS}/train_manifest.jsonl" || ! -s "${ROOTLESS}/val_manifest.jsonl" || ! -s "${ROOTLESS}/test_manifest.jsonl" ]]; then
  echo "[continue] missing rootless manifests under ${ROOTLESS}" >&2
  exit 2
fi

if [[ ! -s "${AUDIT}/rootless_final_metrics.csv" ]]; then
  echo "[continue] missing audit metrics: ${AUDIT}/rootless_final_metrics.csv" >&2
  exit 2
fi

while [[ ! -s "${ROOTLESS}/rootless_validation.json" && -f "${MAIN_LOG}" ]]; do
  now=$(date +%s)
  mtime=$(stat -c %Y "${MAIN_LOG}" 2>/dev/null || echo 0)
  age=$((now - mtime))
  if (( age > 180 )); then
    echo "[continue] main log stale for ${age}s; taking over"
    break
  fi
  echo "[continue] main rebuild log active (${age}s old); waiting for primary validation"
  sleep 60
done

if [[ ! -s "${ROOTLESS}/rootless_validation.json" ]]; then
  echo "[continue] validate rootless final dataset"
  "${PYTHON}" scripts/validate_rootless_dynamic_npz.py \
    --manifest "${ROOTLESS}/train_manifest.jsonl" \
    --manifest "${ROOTLESS}/val_manifest.jsonl" \
    --manifest "${ROOTLESS}/test_manifest.jsonl" \
    --out-json "${ROOTLESS}/rootless_validation.json" \
    --sample-vertices 256 \
    --workers "${WORKERS}"
else
  echo "[continue] validation exists, skip: ${ROOTLESS}/rootless_validation.json"
fi

if [[ ! -s "${STRICT_METRICS}" ]]; then
  echo "[continue] merge strict query-root metrics"
  "${PYTHON}" scripts/merge_metric_csvs.py \
    --out-csv "${STRICT_METRICS}" \
    "${REBUILD_ROOT}/texverse/strict/metrics/query_root_motion_metrics.csv" \
    "${REBUILD_ROOT}/objaverse_xl/strict/metrics/query_root_motion_metrics.csv" \
    "${REBUILD_ROOT}/objaverse_xl_more_anim/strict/metrics/query_root_motion_metrics.csv"
else
  echo "[continue] strict metrics exist, skip: ${STRICT_METRICS}"
fi

if [[ ! -s "${STRICT_QUALITY}/quality_scores.csv" || ! -s "${STRICT_QUALITY}/quality_score_report.md" || ! -s "${STRICT_QUALITY}/quality_score_distribution_grid.png" ]]; then
  echo "[continue] build strict quality score distributions"
  mkdir -p "${STRICT_QUALITY}"
  "${PYTHON}" scripts/build_quality_score_distributions.py \
    --metrics-csv "${STRICT_METRICS}" \
    --out-dir "${STRICT_QUALITY}" \
    --bins 80
else
  echo "[continue] strict distributions exist, skip: ${STRICT_QUALITY}"
fi

if [[ ! -s "${ROOTLESS_RAW_QUALITY}/raw_metrics.csv" || ! -s "${ROOTLESS_RAW_QUALITY}/raw_metric_report.md" || ! -s "${ROOTLESS_RAW_QUALITY}/raw_metric_distribution_grid.png" ]]; then
  echo "[continue] build rootless raw metric distributions"
  mkdir -p "${ROOTLESS_RAW_QUALITY}"
  "${PYTHON}" scripts/build_quality_score_distributions.py \
    --metrics-csv "${AUDIT}/rootless_final_metrics.csv" \
    --out-dir "${ROOTLESS_RAW_QUALITY}" \
    --raw-only \
    --bins 80
else
  echo "[continue] rootless raw distributions exist, skip: ${ROOTLESS_RAW_QUALITY}"
fi

cat > "${REBUILD_ROOT}/README_REBUILD_OUTPUTS.txt" <<EOF
source manifests: ${REBUILD_ROOT}/00_source_manifests
pass1 raw NPZ roots: ${REBUILD_ROOT}/texverse/pass1_motionfps_sequence40, ${REBUILD_ROOT}/objaverse_xl/pass1_motionfps_sequence40, ${REBUILD_ROOT}/objaverse_xl_more_anim/pass1_motionfps_sequence40
strict combined manifests: ${REBUILD_ROOT}/combined_strict
rootless clean dataset: ${ROOTLESS}
rootless validation: ${ROOTLESS}/rootless_validation.json
rootless quality metrics: ${AUDIT}/rootless_final_metrics.csv
strict quality metrics: ${STRICT_METRICS}
strict quality score distributions: ${STRICT_QUALITY}
rootless raw metric distributions: ${ROOTLESS_RAW_QUALITY}
EOF

echo "[continue] done $(date -Is)"
