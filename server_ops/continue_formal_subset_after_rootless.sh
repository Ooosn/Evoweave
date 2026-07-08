#!/usr/bin/env bash
set -euo pipefail

MOD=/ssdwork/liuhaohan/evorig/data_processing_module_clean
PY=/opt/conda/envs/evoweave/bin/python
RUN=/ssdwork/liuhaohan/evorig/clean_realdata_check_20260703/subset_real_formal_pass0_v1
ROOTLESS="$RUN/rootless_clean"
AUDIT="$RUN/rootless_quality/audit"
ORACLE=/ssdwork/liuhaohan/evorig/evoweave_rebuild_from_official_pass1_npz_20260703_record_quality_v3
COMPARE_SCRIPT=${COMPARE_SCRIPT:-/tmp/compare_pass1_oracle.py}

cd "$MOD"
for split in train val test; do
  cp -f "$ROOTLESS/${split}_manifest.jsonl" "$ROOTLESS/${split}_manifest.westlake.jsonl"
done
"$PY" scripts/audit_rootless_final_dataset.py \
  --dataset "$ROOTLESS" \
  --out-dir "$AUDIT" \
  --sample-vertices 128 \
  --workers 8 \
  --no-figures
"$PY" scripts/validate_rootless_dynamic_npz.py \
  --manifest "$ROOTLESS/train_manifest.jsonl" \
  --manifest "$ROOTLESS/val_manifest.jsonl" \
  --manifest "$ROOTLESS/test_manifest.jsonl" \
  --out-json "$ROOTLESS/rootless_validation.json" \
  --sample-vertices 256 \
  --workers 8
"$PY" scripts/build_quality_score_distributions.py \
  --metrics-csv "$AUDIT/rootless_final_metrics.csv" \
  --out-dir "$RUN/rootless_mesh_skeleton_hist" \
  --raw-only \
  --bins 40
"$PY" "$COMPARE_SCRIPT" \
  --run "$RUN" \
  --oracle "$ORACLE" \
  --out "$RUN/oracle_compare"
echo continue_done
