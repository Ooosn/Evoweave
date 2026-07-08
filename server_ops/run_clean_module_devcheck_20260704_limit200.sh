#!/usr/bin/env bash
set -euo pipefail

export PYTHON=/opt/conda/envs/evoweave/bin/python
export BLENDER=/ssdwork/liuhaohan/tools/blender/blender-3.6.23-linux-x64/blender-softwaregl
export EVOWEAVE_7Z=/usr/bin/7z

export EVOWEAVE_TRANSFER_ROOT=/ssdwork/liuhaohan/evoweave_full_transfer_download/evoweave_full_transfer_20260623/evorig
export EVOWEAVE_TRANSFER_MANIFEST=/ssdwork/liuhaohan/evoweave_full_transfer_download/evoweave_full_transfer_20260623/evorig/transfer/evoweave_full_transfer_20260623/full_transfer_manifest.csv
export EVOWEAVE_REBUILD_ROOT=/ssdwork/liuhaohan/evorig/evoweave_rebuild_clean_devcheck_20260704_limit200
export EVOWEAVE_UNIRIG_ROOT=/ssdwork/liuhaohan/evorig/evoweave_repo/external/UniRig

export EVOWEAVE_REBUILD_LIMIT=200
export EVOWEAVE_TEXVERSE_PASS0_WORKERS=8
export EVOWEAVE_TEXVERSE_PASS1_WORKERS=8
export EVOWEAVE_OXL_PASS1_WORKERS=8
export EVOWEAVE_STRICT_WORKERS=16
export EVOWEAVE_ROOTLESS_WORKERS=16
export EVOWEAVE_AUDIT_WORKERS=16
export EVOWEAVE_BLENDER_THREADS=1
export EVOWEAVE_PASS1_TIMEOUT_SEC=900

export EVOWEAVE_RAW_MIN_VERTICES=1
export EVOWEAVE_RAW_BBOX_RATIO_HARD_MIN=0.0
export EVOWEAVE_RAW_BBOX_RATIO_HARD_MAX=0.0
export EVOWEAVE_RAW_MIN_MOTION_P95_BBOX=0.0
export EVOWEAVE_RAW_ACTIVE_SKIN_THRESHOLD=0.0
export EVOWEAVE_ACTIVE_SKIN_THRESHOLD=1e-4
export EVOWEAVE_MIN_COMPONENT_CONTROLLED_RATIO=0.95

cd /ssdwork/liuhaohan/evorig/data_processing_module_clean
exec bash run_rebuild_from_originals.sh
