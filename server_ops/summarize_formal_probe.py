#!/usr/bin/env python3
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def status_counts(path: Path):
    return dict(Counter(row.get("status") for row in read_jsonl(path)))


def load_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(row, key):
    try:
        return float(row.get(key) or 0.0)
    except ValueError:
        return 0.0


def asset_source_from_path(path: str):
    if "/texverse/" in path:
        return "texverse"
    if "/objaverse_xl_more_anim/" in path:
        return "objaverse_xl_more_anim"
    if "/objaverse_xl/" in path:
        return "objaverse_xl"
    return "unknown"


def main() -> int:
    run = Path(sys.argv[1])
    out = {}

    pass0_rows = read_jsonl(run / "00_source_manifests/texverse_pass0_audit.jsonl")
    pass0_assets = defaultdict(list)
    for row in pass0_rows:
        pass0_assets[row.get("asset_id")].append(row)
    out["texverse_pass0"] = {
        "records": len(pass0_rows),
        "assets": len(pass0_assets),
        "usable_assets": sum(any(r.get("usable") is True for r in rows) for rows in pass0_assets.values()),
        "reject_reasons": dict(Counter(reason for row in pass0_rows if row.get("usable") is not True for reason in (row.get("reject_reasons") or ["unknown"]))),
    }

    out["pass1_status"] = {
        "texverse": status_counts(run / "texverse/pass1_motionfps_sequence40/manifest.jsonl"),
        "objaverse_xl": status_counts(run / "objaverse_xl/pass1_motionfps_sequence40/manifest.jsonl"),
        "objaverse_xl_more_anim": status_counts(run / "objaverse_xl_more_anim/pass1_motionfps_sequence40/manifest.jsonl"),
    }

    out["strict"] = {}
    for source in ("texverse", "objaverse_xl", "objaverse_xl_more_anim"):
        summary_path = run / source / "strict/summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        rejects = read_jsonl(run / source / "strict/rejected.jsonl")
        out["strict"][source] = {
            "accepted": summary.get("accepted"),
            "rejected": summary.get("rejected"),
            "top_reject_reasons": summary.get("top_reject_reasons"),
            "rejected_assets": [{"asset_id": r.get("asset_id"), "reasons": r.get("reasons")} for r in rejects[:20]],
        }

    rootless_summary = json.loads((run / "rootless_clean/rootless_summary.json").read_text())
    failures = []
    for split_summary in rootless_summary["summaries"]:
        for item in split_summary.get("failures", []):
            failures.append(item)
    by_source = Counter(asset_source_from_path(item.get("path", "")) for item in failures)
    out["rootless"] = {
        "rows": rootless_summary.get("rows"),
        "failed": rootless_summary.get("failed"),
        "failure_by_source": dict(by_source),
        "dropped_mesh_component_count": sum(s.get("dropped_mesh_component_count", 0) for s in rootless_summary["summaries"]),
        "dropped_vertex_count": sum(s.get("dropped_vertex_count", 0) for s in rootless_summary["summaries"]),
        "raw_root_disposition": dict(sum((Counter(s.get("raw_root_disposition", {})) for s in rootless_summary["summaries"]), Counter())),
        "failures": [{"asset_id": x.get("asset_id"), "source": asset_source_from_path(x.get("path", "")), "error": x.get("error")} for x in failures],
    }

    validation = json.loads((run / "rootless_clean/rootless_validation.json").read_text())
    out["validation"] = validation

    metrics = load_csv(run / "rootless_quality/audit/rootless_final_metrics.csv")
    low_active = sorted(metrics, key=lambda r: f(r, "active_joint_mesh_bbox_diag_consistency"))[:20]
    high_align = sorted(metrics, key=lambda r: f(r, "target_alignment_median"), reverse=True)[:20]
    high_lbs = sorted(metrics, key=lambda r: f(r, "lbs_recon_p95_bbox"), reverse=True)[:20]
    out["rootless_metric_outliers"] = {
        "active_joint_mesh_bbox_diag_consistency_low": [
            {
                "asset_id": r.get("asset_id"),
                "source": asset_source_from_path(r.get("path", "")),
                "value": f(r, "active_joint_mesh_bbox_diag_consistency"),
                "target_active_joint_count": f(r, "target_active_joint_count"),
                "target_joint_count": f(r, "target_joint_count"),
            }
            for r in low_active
        ],
        "target_alignment_median_high": [
            {"asset_id": r.get("asset_id"), "source": asset_source_from_path(r.get("path", "")), "value": f(r, "target_alignment_median")}
            for r in high_align
        ],
        "lbs_recon_p95_bbox_high": [
            {"asset_id": r.get("asset_id"), "source": asset_source_from_path(r.get("path", "")), "value": f(r, "lbs_recon_p95_bbox")}
            for r in high_lbs
        ],
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
