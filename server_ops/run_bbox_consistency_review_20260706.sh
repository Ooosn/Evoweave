#!/usr/bin/env bash
set -euo pipefail

OUT="/ssdwork/liuhaohan/evorig/evoweave_rebuild_clean_full_20260704/bbox_consistency_review_20260706"
MET="/ssdwork/liuhaohan/evorig/evoweave_rebuild_clean_full_20260704/quality_distributions_latest_20260706/rootless_bbox_consistency/rootless_bbox_consistency_metrics.csv"

/opt/conda/envs/evoweave/bin/python /ssdwork/liuhaohan/evorig/plot_bbox_consistency_review.py \
  --metrics-csv "${MET}" \
  --out-dir "${OUT}" \
  --sample-per-bucket 24 \
  --detail-count 12

find "${OUT}" -maxdepth 2 -type f | sort
