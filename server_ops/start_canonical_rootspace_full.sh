#!/usr/bin/env bash
set -euo pipefail

cd /ssdwork/liuhaohan/evorig/evoweave_repo/rigweave

LOG=/ssdwork/liuhaohan/evorig/canonical_rootspace_20260630_full.log
PIDFILE=/ssdwork/liuhaohan/evorig/canonical_rootspace_20260630_full.pid
OUT=/ssdwork/liuhaohan/evorig/evoweave_combined_dynamic_canonical_rootspace_20260630

nohup /opt/conda/envs/evoweave/bin/python scripts/build_canonical_dynamic_npz.py \
  --manifest /ssdwork/liuhaohan/evorig/evoweave_combined_dynamic_strict_20260613_more_anim/train_manifest.westlake.jsonl \
  --manifest /ssdwork/liuhaohan/evorig/evoweave_combined_dynamic_strict_20260613_more_anim/val_manifest.westlake.jsonl \
  --manifest /ssdwork/liuhaohan/evorig/evoweave_combined_dynamic_strict_20260613_more_anim/test_manifest.westlake.jsonl \
  --output-root "$OUT" \
  --verify-vertices 64 \
  --compression stored \
  --skip-existing \
  > "$LOG" 2>&1 &

echo "$!" > "$PIDFILE"
echo "PID=$(cat "$PIDFILE")"
echo "LOG=$LOG"
echo "OUT=$OUT"
