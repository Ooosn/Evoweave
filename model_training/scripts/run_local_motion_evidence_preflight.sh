#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-/opt/conda/envs/evoweave/bin/python}"
output_dir="${OUTPUT_DIR:?OUTPUT_DIR must name a persistent analysis directory}"

train_limit="${TRAIN_LIMIT:-512}"
valid_limit="${VALID_LIMIT:-128}"
max_edges_per_asset="${MAX_EDGES_PER_ASSET:-1024}"
frame_count="${FRAME_COUNT:-24}"
query_repeats="${QUERY_REPEATS:-1}"
epochs="${EPOCHS:-8}"
seed="${SEED:-20260724}"

cd "${repo_root}"
"${python_bin}" model_training/tools/agent_work_guard.py check --operation evidence_analysis

exec "${python_bin}" model_training/analysis/analyze_local_motion_evidence.py \
  --output-dir "${output_dir}" \
  --train-limit "${train_limit}" \
  --valid-limit "${valid_limit}" \
  --max-edges-per-asset "${max_edges_per_asset}" \
  --frame-count "${frame_count}" \
  --query-repeats "${query_repeats}" \
  --epochs "${epochs}" \
  --seed "${seed}" \
  --cache-features \
  --device cuda
