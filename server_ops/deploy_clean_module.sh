#!/usr/bin/env bash
set -euo pipefail

archive="${1:?archive path required}"
target="/ssdwork/liuhaohan/evorig/data_processing_module_clean"
parent="/ssdwork/liuhaohan/evorig"

case "${target}" in
  /ssdwork/liuhaohan/evorig/data_processing_module_clean) ;;
  *) echo "refusing unsafe target: ${target}" >&2; exit 9 ;;
esac

rm -rf "${target}"
mkdir -p "${parent}"
tar -xzf "${archive}" -C "${parent}"
cd "${target}"
bash -n run_rebuild_from_originals.sh
/opt/conda/envs/evoweave/bin/python -m py_compile \
  scripts/build_rebuild_source_manifests.py \
  scripts/texverse_quality_audit.py \
  scripts/export_texverse_pass1_batch.py \
  scripts/build_strict_phase1_dataset.py \
  scripts/build_rootless_dynamic_npz.py
find . -type d -name __pycache__ -prune -exec rm -rf {} +
echo uploaded_clean_module_ok
