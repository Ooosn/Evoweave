#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="/ssdwork/liuhaohan/evorig/data_processing/src:/ssdwork/liuhaohan/evorig/evoweave_repo/external/UniRig/src:${PYTHONPATH:-}"

PY="/opt/conda/envs/evoweave/bin/python"
MOD="/ssdwork/liuhaohan/evorig/data_processing"
ROOT="/ssdwork/liuhaohan/evorig/evoweave_rebuild_clean_full_20260704/rootless_clean"
OUT="/ssdwork/liuhaohan/evorig/evoweave_rebuild_clean_full_20260704/quality_distributions_latest_20260706"

mkdir -p "${OUT}" /ssdwork/liuhaohan/evorig/logs
cd "${MOD}"

echo "latest_hist_start $(date -Is)"

"${PY}" scripts/build_rootless_quality_score_distributions.py \
  --manifest "${ROOT}/train_manifest.jsonl" \
  --manifest "${ROOT}/val_manifest.jsonl" \
  --manifest "${ROOT}/test_manifest.jsonl" \
  --out-dir "${OUT}/rootless_dynamic_scores" \
  --workers 32 \
  --bins 80

"${PY}" scripts/build_rootless_bbox_consistency_distributions.py \
  --manifest "${ROOT}/train_manifest.jsonl" \
  --manifest "${ROOT}/val_manifest.jsonl" \
  --manifest "${ROOT}/test_manifest.jsonl" \
  --out-dir "${OUT}/rootless_bbox_consistency" \
  --active-skin-threshold 1e-4 \
  --workers 32 \
  --bins 80

echo "latest_hist_done $(date -Is)"
find "${OUT}" -maxdepth 3 -type f | sort
