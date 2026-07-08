#!/usr/bin/env bash
set -euo pipefail

LOG_DIR=/ssdwork/liuhaohan/evorig/logs
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/clean_module_full_rebuild_20260706.log"
exec > >(tee -a "${LOG}") 2>&1

echo "clean_module_full_rebuild_20260706_start $(date -Is)"

export PYTHON=/opt/conda/envs/evoweave/bin/python
export BLENDER=/ssdwork/liuhaohan/tools/blender/blender-3.6.23-linux-x64/blender-softwaregl
export EVOWEAVE_7Z=/usr/bin/7z

if ! command -v 7z >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y p7zip-full
fi
if ! "${BLENDER}" --version >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libsm6 \
    libxrender1 \
    libxext6 \
    libxi6 \
    libxfixes3 \
    libxrandr2 \
    libxcursor1 \
    libxinerama1 \
    libgl1
fi

export EVOWEAVE_TRANSFER_ROOT=/ssdwork/liuhaohan/evoweave_full_transfer_download/evoweave_full_transfer_20260623/evorig
export EVOWEAVE_TRANSFER_MANIFEST=/ssdwork/liuhaohan/evoweave_full_transfer_download/evoweave_full_transfer_20260623/evorig/transfer/evoweave_full_transfer_20260623/full_transfer_manifest.csv
export EVOWEAVE_REBUILD_ROOT=/ssdwork/liuhaohan/evorig/evoweave_rebuild_rootless_v3_20260706
export EVOWEAVE_UNIRIG_ROOT=/ssdwork/liuhaohan/evorig/evoweave_repo/external/UniRig

export EVOWEAVE_REBUILD_LIMIT=0
export EVOWEAVE_SEQUENCE_FRAMES=40
export EVOWEAVE_MOTION_FPS_DESCRIPTOR_VERTICES=1024
export EVOWEAVE_TEXVERSE_PASS0_WORKERS=32
export EVOWEAVE_TEXVERSE_PASS1_WORKERS=32
export EVOWEAVE_OXL_PASS1_WORKERS=32
export EVOWEAVE_STRICT_WORKERS=32
export EVOWEAVE_ROOTLESS_WORKERS=32
export EVOWEAVE_AUDIT_WORKERS=32
export EVOWEAVE_BLENDER_THREADS=1
export EVOWEAVE_PASS1_TIMEOUT_SEC=900

export EVOWEAVE_RAW_MIN_VERTICES=1
export EVOWEAVE_RAW_BBOX_RATIO_HARD_MIN=0.0
export EVOWEAVE_RAW_BBOX_RATIO_HARD_MAX=0.0
export EVOWEAVE_RAW_MIN_MOTION_P95_BBOX=0.0
export EVOWEAVE_RAW_ACTIVE_SKIN_THRESHOLD=0.0
export EVOWEAVE_ACTIVE_SKIN_THRESHOLD=1e-4
export EVOWEAVE_MIN_COMPONENT_CONTROLLED_RATIO=0.95
export EVOWEAVE_ROOTLESS_MIN_TARGET_JOINT_MESH_BBOX_CONSISTENCY=0.4

LOCK_DIR=/ssdwork/liuhaohan/evorig/locks
LOCK_FILE="${LOCK_DIR}/clean_module_full_rebuild_20260706.lock"
mkdir -p "${LOCK_DIR}"
exec 9>"${LOCK_FILE}"
flock 9

if [[ -f "${EVOWEAVE_REBUILD_ROOT}/rootless_clean/rootless_validation.json" \
   && -f "${EVOWEAVE_REBUILD_ROOT}/quality_distributions/rootless_bbox_consistency/final_manifests/accepted_manifest.jsonl" ]]; then
  echo "clean_module_full_rebuild_20260706_already_complete ${EVOWEAVE_REBUILD_ROOT}"
  exit 0
fi

cd /ssdwork/liuhaohan/evorig/data_processing
bash run_rebuild_from_originals.sh

echo "clean_module_full_rebuild_20260706_end $(date -Is)"
