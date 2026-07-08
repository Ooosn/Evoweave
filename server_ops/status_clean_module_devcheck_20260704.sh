#!/usr/bin/env bash
set -euo pipefail

RUN=/ssdwork/liuhaohan/evorig/evoweave_rebuild_clean_devcheck_20260704_limit200
LOG=/ssdwork/liuhaohan/evorig/logs/clean_module_devcheck_20260704_limit200.log
PIDFILE=/ssdwork/liuhaohan/evorig/logs/clean_module_devcheck_20260704_limit200.pid
PYTHON=/opt/conda/envs/evoweave/bin/python

pid="$(cat "${PIDFILE}" 2>/dev/null || true)"
if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
  echo "running:${pid}"
else
  echo "not_running:${pid}"
fi

echo "npz_counts"
for source in texverse objaverse_xl objaverse_xl_more_anim; do
  count="$(find "${RUN}/${source}/pass1_motionfps_sequence40/npz" -name "*.npz" 2>/dev/null | wc -l)"
  echo "${source} ${count}"
done

echo "strict_summaries"
for source in texverse objaverse_xl objaverse_xl_more_anim; do
  summary="${RUN}/${source}/strict/summary.json"
  if [[ -f "${summary}" ]]; then
    "${PYTHON}" - "${source}" "${summary}" <<'PY'
import json
import sys
source, path = sys.argv[1], sys.argv[2]
data = json.load(open(path, encoding="utf-8"))
print(
    source,
    "input", data.get("input_files"),
    "accepted", data.get("accepted"),
    "rejected", data.get("rejected"),
    "mode", data.get("pass1b_gate_mode"),
    "top", data.get("top_reject_reasons"),
)
PY
  fi
done

echo "combined_rootless"
for path in \
  "${RUN}/combined_strict/combine_summary.json" \
  "${RUN}/rootless_clean/rootless_summary.json" \
  "${RUN}/rootless_clean/rootless_validation.json"; do
  if [[ -f "${path}" ]]; then
    echo "${path}"
    "${PYTHON}" - "${path}" <<'PY'
import json
import sys
path = sys.argv[1]
data = json.load(open(path, encoding="utf-8"))
for key in ("combined_total", "combined_counts", "rows", "failed", "invalid", "max_target_alignment_median"):
    if key in data:
        print(key, data[key])
if "validation" in data:
    v = data["validation"]
    print("validation_rows", v.get("rows"), "validation_invalid", v.get("invalid"))
PY
  fi
done

echo "histogram_outputs"
for path in \
  "${RUN}/quality_distributions/strict_query_root_scores/quality_score_report.md" \
  "${RUN}/quality_distributions/rootless_alignment_candidates/rootless_alignment_report.md" \
  "${RUN}/quality_distributions/rootless_alignment_candidates/rootless_alignment_metrics.csv"; do
  [[ -f "${path}" ]] && echo "${path}"
done

echo "log_tail"
tail -n 120 "${LOG}" 2>/dev/null || true
