#!/usr/bin/env bash
set -euo pipefail

MOD=/ssdwork/liuhaohan/evorig/data_processing_module_clean
PY=/opt/conda/envs/evoweave/bin/python
BLENDER=/ssdwork/liuhaohan/tools/blender/blender-3.6.23-linux-x64/blender-softwaregl
TRANSFER_ROOT=/ssdwork/liuhaohan/evoweave_full_transfer_download/evoweave_full_transfer_20260623/evorig
TRANSFER_MANIFEST="$TRANSFER_ROOT/transfer/evoweave_full_transfer_20260623/full_transfer_manifest.csv"
SOURCE_DIR=/ssdwork/liuhaohan/evorig/clean_realdata_check_20260703/01_source_manifests_formal_pass0
RUN=/ssdwork/liuhaohan/evorig/clean_realdata_check_20260703/subset_real_formal_pass0_v1
ORACLE=/ssdwork/liuhaohan/evorig/evoweave_rebuild_from_official_pass1_npz_20260703_record_quality_v3
UNIRIG=/ssdwork/liuhaohan/evorig/evoweave_repo/external/UniRig
TOKENIZER_CONFIG="$UNIRIG/configs/tokenizer/tokenizer_parts_articulationxl_256.yaml"
COMPARE_SCRIPT=${COMPARE_SCRIPT:-/tmp/compare_pass1_oracle.py}

export RIGWEAVE_DISABLE_OPEN3D=1
export EVOWEAVE_UNIRIG_ROOT="$UNIRIG"
export PYTHONPATH="$MOD/src:$UNIRIG/src:${PYTHONPATH:-}"

rm -rf "$RUN"
mkdir -p "$RUN/subsets" "$SOURCE_DIR"

cd "$MOD"

"$PY" scripts/build_rebuild_source_manifests.py \
  --transfer-manifest-csv "$TRANSFER_MANIFEST" \
  --transfer-root "$TRANSFER_ROOT" \
  --out-dir "$SOURCE_DIR" \
  --fail-on-missing-original

"$PY" - <<'PY'
import json
from pathlib import Path

source_dir = Path("/ssdwork/liuhaohan/evorig/clean_realdata_check_20260703/01_source_manifests_formal_pass0")
out = Path("/ssdwork/liuhaohan/evorig/clean_realdata_check_20260703/subset_real_formal_pass0_v1/subsets")
out.mkdir(parents=True, exist_ok=True)

known = {
    "texverse": [
        "0005bab5f144465c9c92efaf13bf0f9d",
        "f32a94a86b114b9bbf44e93552b56761",
        "0aca8647928b4f3db0e20811481d98ee",
    ],
    "objaverse_xl": ["oxl_f26219facfc262ce15f1", "oxl_0000ec7cf83fdc0539c5"],
    "objaverse_xl_more_anim": ["oxl_047eff7af465c3c29cd1"],
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

tex_rows = choose(list(read_jsonl(source_dir / "texverse_originals.jsonl")), 14, known["texverse"])
oxl_rows = choose(list(read_jsonl(source_dir / "objaverse_xl_download_rebuild_manifest.jsonl")), 8, known["objaverse_xl"])
more_rows = choose(
    list(read_jsonl(source_dir / "objaverse_xl_more_anim_download_rebuild_manifest.jsonl")),
    6,
    known["objaverse_xl_more_anim"],
)

def write_jsonl(path, rows):
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

write_jsonl(out / "texverse_originals_subset.jsonl", tex_rows)
write_jsonl(out / "objaverse_xl_subset.jsonl", oxl_rows)
write_jsonl(out / "objaverse_xl_more_anim_subset.jsonl", more_rows)
(out / "texverse_model_paths_subset.txt").write_text(
    "\n".join(str(r["transfer_original_rel"]) for r in tex_rows) + "\n"
)
(out / "texverse_animation_ids_subset.txt").write_text(
    "\n".join(str(r["asset_id"]) for r in tex_rows) + "\n"
)
summary = {
    "texverse": [r["asset_id"] for r in tex_rows],
    "objaverse_xl": [r["asset_id"] for r in oxl_rows],
    "objaverse_xl_more_anim": [r["asset_id"] for r in more_rows],
}
(out / "selected_asset_ids.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "[1a] texverse pass0 audit from original zips"
TEX="$RUN/texverse"
TEX_PASS0="$TEX/pass0_audit"
"$PY" scripts/texverse_quality_audit.py \
  --model-paths "$RUN/subsets/texverse_model_paths_subset.txt" \
  --animation-ids "$RUN/subsets/texverse_animation_ids_subset.txt" \
  --download-dir "$TRANSFER_ROOT/evoweave_texverse_full/pass0_wide/zips" \
  --extract-root "$TEX/pass0_extract" \
  --output-jsonl "$TEX_PASS0/audit.jsonl" \
  --blender "$BLENDER" \
  --blender-threads 1 \
  --timeout-sec 240 \
  --workers 4 \
  --max-candidates-per-zip 8 \
  --resume
"$PY" scripts/summarize_texverse_audit.py \
  --input-jsonl "$TEX_PASS0/audit.jsonl" \
  --out-dir "$TEX_PASS0/summary" \
  --name texverse_pass0

echo "[1b] texverse pass1 from selected pass0 source"
TEX_PASS1="$TEX/pass1_motionfps_sequence40"
"$PY" scripts/export_texverse_pass1_batch.py \
  --audit-jsonl "$TEX_PASS0/audit.jsonl" \
  --zip-dir "$TRANSFER_ROOT/evoweave_texverse_full/pass0_wide/zips" \
  --out-root "$TEX_PASS1" \
  --extract-root "$TEX_PASS1/extract" \
  --manifest-jsonl "$TEX_PASS1/manifest.jsonl" \
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

echo "[2] strict/rootless/audit"
run_strict "$TEX_PASS1/manifest.jsonl" "$RUN/texverse/strict"
run_strict "$RUN/objaverse_xl/pass1_motionfps_sequence40/manifest.jsonl" "$RUN/objaverse_xl/strict"
run_strict "$RUN/objaverse_xl_more_anim/pass1_motionfps_sequence40/manifest.jsonl" "$RUN/objaverse_xl_more_anim/strict"

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

echo "[3] compare new pass1 with old NPZ for diagnosis only"
"$PY" "$COMPARE_SCRIPT" \
  --run "$RUN" \
  --oracle "$ORACLE" \
  --out "$RUN/oracle_compare"

echo "[done] $RUN"
