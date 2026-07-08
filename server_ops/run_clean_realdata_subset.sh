#!/usr/bin/env bash
set -euo pipefail

MOD=/ssdwork/liuhaohan/evorig/data_processing_module_clean
PY=/opt/conda/envs/evoweave/bin/python
BLENDER=/ssdwork/liuhaohan/tools/blender/blender-3.6.23-linux-x64/blender-softwaregl
TRANSFER_ROOT=/ssdwork/liuhaohan/evoweave_full_transfer_download/evoweave_full_transfer_20260623/evorig
SOURCE_DIR=/ssdwork/liuhaohan/evorig/clean_realdata_check_20260703/00_source_manifests
RUN=/ssdwork/liuhaohan/evorig/clean_realdata_check_20260703/subset_real_oracle_v1
ORACLE=/ssdwork/liuhaohan/evorig/evoweave_rebuild_from_official_pass1_npz_20260703_record_quality_v3
UNIRIG=/ssdwork/liuhaohan/evorig/evoweave_repo/external/UniRig
TOKENIZER_CONFIG="$UNIRIG/configs/tokenizer/tokenizer_parts_articulationxl_256.yaml"
COMPARE_SCRIPT=${COMPARE_SCRIPT:-/tmp/compare_pass1_oracle.py}

export RIGWEAVE_DISABLE_OPEN3D=1
export EVOWEAVE_UNIRIG_ROOT="$UNIRIG"
export PYTHONPATH="$MOD/src:$UNIRIG/src:${PYTHONPATH:-}"

rm -rf "$RUN"
mkdir -p "$RUN/subsets"

"$PY" - <<'PY'
import json
from pathlib import Path

source_dir = Path("/ssdwork/liuhaohan/evorig/clean_realdata_check_20260703/00_source_manifests")
out = Path("/ssdwork/liuhaohan/evorig/clean_realdata_check_20260703/subset_real_oracle_v1/subsets")
out.mkdir(parents=True, exist_ok=True)

known = {
    "texverse": ["f32a94a86b114b9bbf44e93552b56761", "0aca8647928b4f3db0e20811481d98ee"],
    "objaverse_xl": ["oxl_f26219facfc262ce15f1"],
    "objaverse_xl_more_anim": [],
}

def read_jsonl(path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def choose(rows, count, known_ids):
    by_id = {str(r["asset_id"]): r for r in rows}
    selected = []
    seen = set()
    for aid in known_ids:
        if aid in by_id:
            selected.append(by_id[aid])
            seen.add(aid)
    for row in rows:
        aid = str(row["asset_id"])
        if aid not in seen:
            selected.append(row)
            seen.add(aid)
        if len(selected) >= count:
            break
    return selected

configs = [
    ("texverse", source_dir / "texverse_pass0_rebuild_audit.jsonl", out / "texverse_subset.jsonl", 12),
    ("objaverse_xl", source_dir / "objaverse_xl_download_rebuild_manifest.jsonl", out / "objaverse_xl_subset.jsonl", 8),
    ("objaverse_xl_more_anim", source_dir / "objaverse_xl_more_anim_download_rebuild_manifest.jsonl", out / "objaverse_xl_more_anim_subset.jsonl", 6),
]
summary = {}
for source, src, dst, count in configs:
    rows = list(read_jsonl(src))
    selected = choose(rows, count, known.get(source, []))
    with dst.open("w") as f:
        for row in selected:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    summary[source] = [r["asset_id"] for r in selected]
(out / "selected_asset_ids.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
PY

cd "$MOD"

echo "[1] texverse pass1 from original zip"
TEX="$RUN/texverse/pass1_motionfps_sequence40"
"$PY" scripts/export_texverse_pass1_batch.py \
  --audit-jsonl "$RUN/subsets/texverse_subset.jsonl" \
  --zip-dir "$TRANSFER_ROOT/evoweave_texverse_full/pass0_wide/zips" \
  --out-root "$TEX" \
  --extract-root "$TEX/extract" \
  --manifest-jsonl "$TEX/manifest.jsonl" \
  --blender "$BLENDER" \
  --blender-threads 1 \
  --frames 40 \
  --motion-fps-descriptor-vertices 1024 \
  --timeout-sec 900 \
  --workers 4 \
  --max-joints 256 \
  --min-vertices 1 \
  --max-vertices 300000 \
  --max-faces 600000 \
  --active-skin-threshold 0.0 \
  --shuffle \
  --seed 2026

for source in objaverse_xl objaverse_xl_more_anim; do
  echo "[1] $source pass1 from original raw"
  ROOT="$RUN/$source"
  PASS1="$ROOT/pass1_motionfps_sequence40"
  mkdir -p "$PASS1"
  "$PY" scripts/export_objaverse_xl_pass1_batch.py \
    --download-manifest-jsonl "$RUN/subsets/${source}_subset.jsonl" \
    --out-root "$PASS1" \
    --manifest-jsonl "$PASS1/manifest.jsonl" \
    --blender "$BLENDER" \
    --blender-threads 1 \
    --frames 40 \
    --motion-fps-descriptor-vertices 1024 \
    --timeout-sec 900 \
    --workers 4 \
    --max-joints 256 \
    --min-vertices 1 \
    --max-vertices 300000 \
    --max-faces 600000 \
    --active-skin-threshold 0.0
done

run_strict() {
  local source_name="$1"
  local manifest="$2"
  local out="$3"
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

echo "[2] strict/rootless/audit"
run_strict texverse "$TEX/manifest.jsonl" "$RUN/texverse/strict"
run_strict objaverse_xl "$RUN/objaverse_xl/pass1_motionfps_sequence40/manifest.jsonl" "$RUN/objaverse_xl/strict"
run_strict objaverse_xl_more_anim "$RUN/objaverse_xl_more_anim/pass1_motionfps_sequence40/manifest.jsonl" "$RUN/objaverse_xl_more_anim/strict"

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

echo "[3] compare new pass1 with oracle"
"$PY" "$COMPARE_SCRIPT" \
  --run "$RUN" \
  --oracle "$ORACLE" \
  --out "$RUN/oracle_compare"

echo "[done] $RUN"
