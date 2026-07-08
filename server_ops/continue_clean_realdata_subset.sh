#!/usr/bin/env bash
set -euo pipefail

MOD=/ssdwork/liuhaohan/evorig/data_processing_module_clean
PY=/opt/conda/envs/evoweave/bin/python
SOURCE_DIR=/ssdwork/liuhaohan/evorig/clean_realdata_check_20260703/00_source_manifests
RUN=/ssdwork/liuhaohan/evorig/clean_realdata_check_20260703/subset_real_oracle_v1
ORACLE=/ssdwork/liuhaohan/evorig/evoweave_rebuild_from_official_pass1_npz_20260703_record_quality_v3
UNIRIG=/ssdwork/liuhaohan/evorig/evoweave_repo/external/UniRig
TOKENIZER_CONFIG="$UNIRIG/configs/tokenizer/tokenizer_parts_articulationxl_256.yaml"
COMPARE_SCRIPT=${COMPARE_SCRIPT:-/tmp/compare_pass1_oracle.py}

export RIGWEAVE_DISABLE_OPEN3D=1
export EVOWEAVE_UNIRIG_ROOT="$UNIRIG"
export PYTHONPATH="$MOD/src:$UNIRIG/src:${PYTHONPATH:-}"

cd "$MOD"
rm -rf \
  "$RUN/texverse/sequence_dedup" "$RUN/texverse/strict" \
  "$RUN/objaverse_xl/sequence_dedup" "$RUN/objaverse_xl/strict" \
  "$RUN/objaverse_xl_more_anim/sequence_dedup" "$RUN/objaverse_xl_more_anim/strict" \
  "$RUN/combined_strict" "$RUN/rootless_clean" "$RUN/rootless_quality" \
  "$RUN/rootless_mesh_skeleton_hist" "$RUN/oracle_compare"

run_strict() {
  local manifest="$1"
  local out="$2"
  "$PY" scripts/dedupe_dynamic_sequences.py \
    --manifest-jsonl "$manifest" \
    --fingerprint-frames 8 \
    --out-keep-jsonl "$out/../sequence_dedup/sequence_unique_manifest.jsonl" \
    --out-duplicate-jsonl "$out/../sequence_dedup/sequence_duplicate_manifest.jsonl"
  "$PY" scripts/build_strict_phase1_dataset.py \
    --manifest-jsonl "$out/../sequence_dedup/sequence_unique_manifest.jsonl" \
    --output-root "$out" \
    --target-assets 0 \
    --min-frames 40 \
    --min-vertices 100 \
    --min-faces 20 \
    --min-joints 4 \
    --motion-rate-eps 0.01 \
    --min-motion-rate 0.01 \
    --min-motion-coverage-score 0.15 \
    --min-motion-amount-score 0.15 \
    --min-bbox-stability-score 0.50 \
    --min-edge-stretch-stability-score 0.50 \
    --min-edge-collapse-stability-score 0.20 \
    --min-spike-cleanliness-score 0.50 \
    --max-lbs-recon-p95-bbox 0.02 \
    --min-active-joint-inside-padded-bbox-ratio 0.90 \
    --max-active-nonroot-edge-len-bbox 1.00 \
    --center-lbs-joint-mesh-center-offset 0.35 \
    --center-lbs-recon-p95-bbox 0.005 \
    --split-map-csv "$SOURCE_DIR/split_map.csv" \
    --workers 8 \
    --tokenizer-config "$TOKENIZER_CONFIG"
}

echo "[2] strict"
run_strict "$RUN/texverse/pass1_motionfps_sequence40/manifest.jsonl" "$RUN/texverse/strict"
run_strict "$RUN/objaverse_xl/pass1_motionfps_sequence40/manifest.jsonl" "$RUN/objaverse_xl/strict"
run_strict "$RUN/objaverse_xl_more_anim/pass1_motionfps_sequence40/manifest.jsonl" "$RUN/objaverse_xl_more_anim/strict"

echo "[3] combine/rootless/audit"
COMBINED="$RUN/combined_strict"
"$PY" scripts/combine_dynamic_strict_manifests.py \
  --source "texverse=$RUN/texverse/strict" \
  --source "objaverse_xl=$RUN/objaverse_xl/strict" \
  --source "objaverse_xl_more_anim=$RUN/objaverse_xl_more_anim/strict" \
  --output-dir "$COMBINED" \
  --check-paths

ROOTLESS="$RUN/rootless_clean"
"$PY" scripts/build_rootless_dynamic_npz.py \
  --manifest "$COMBINED/train_manifest.jsonl" \
  --manifest "$COMBINED/val_manifest.jsonl" \
  --manifest "$COMBINED/test_manifest.jsonl" \
  --output-root "$ROOTLESS" \
  --active-skin-threshold 1e-4 \
  --min-component-controlled-ratio 0.95 \
  --compression compressed \
  --verify-vertices 256 \
  --workers 8 \
  --allow-build-rejects
for split in train val test; do
  cp -f "$ROOTLESS/${split}_manifest.jsonl" "$ROOTLESS/${split}_manifest.westlake.jsonl"
done

AUDIT="$RUN/rootless_quality/audit"
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

echo "[4] compare"
"$PY" "$COMPARE_SCRIPT" \
  --run "$RUN" \
  --oracle "$ORACLE" \
  --out "$RUN/oracle_compare"

echo "[done] $RUN"
