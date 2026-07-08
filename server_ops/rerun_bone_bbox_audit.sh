#!/usr/bin/env bash
set -euo pipefail

MOD=/ssdwork/liuhaohan/evorig/data_processing_module_clean
PY=/opt/conda/envs/evoweave/bin/python
RUN=/ssdwork/liuhaohan/evorig/clean_realdata_check_20260703/formal_probe_limit200
ROOTLESS="$RUN/rootless_clean"
AUDIT="$RUN/rootless_quality/audit_bone_bbox"
HIST="$RUN/quality_distributions/rootless_mesh_skeleton_bone_raw"

cd "$MOD"
"$PY" scripts/audit_rootless_final_dataset.py \
  --dataset "$ROOTLESS" \
  --out-dir "$AUDIT" \
  --sample-vertices 128 \
  --workers 8 \
  --no-figures
"$PY" scripts/build_quality_score_distributions.py \
  --metrics-csv "$AUDIT/rootless_final_metrics.csv" \
  --out-dir "$HIST" \
  --raw-only \
  --bins 80
cat "$HIST/raw_metric_report.md"
