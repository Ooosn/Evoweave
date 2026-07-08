#!/usr/bin/env python3
"""Summarize TexVerse audit JSONL files into manifests and markdown."""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--name", default="texverse_audit")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_jsonl)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    passed = [row for row in rows if row.get("usable") is True]
    rejected = [row for row in rows if row.get("usable") is not True]
    reasons = collections.Counter()
    status = collections.Counter(row.get("status", "") for row in rows)
    for row in rejected:
        for reason in row.get("reject_reasons") or ["unknown"]:
            reasons[reason] += 1

    fields = [
        "asset_id",
        "usable",
        "reject_reasons",
        "status",
        "max_bones",
        "skinned_vertices",
        "skin_coverage",
        "max_action_frames",
        "max_bone_fcurves",
        "total_vertices",
        "total_faces",
        "candidate",
        "zip_path",
        "texverse_rel_path",
    ]
    with (args.out_dir / f"{args.name}_summary.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: "+".join(row.get(field) or [])
                    if field == "reject_reasons"
                    else row.get(field, "")
                    for field in fields
                }
            )

    with (args.out_dir / f"{args.name}_passed_ids.txt").open("w", encoding="utf-8") as handle:
        for row in passed:
            handle.write(f"{row['asset_id']}\n")

    with (args.out_dir / f"{args.name}_rejected_ids.tsv").open("w", encoding="utf-8") as handle:
        handle.write("asset_id\treject_reasons\tcandidate\n")
        for row in rejected:
            handle.write(
                f"{row.get('asset_id', '')}\t"
                f"{'+'.join(row.get('reject_reasons') or ['unknown'])}\t"
                f"{row.get('candidate', '')}\n"
            )

    pass_rate = 100.0 * len(passed) / max(1, len(rows))
    md = [
        f"# {args.name}",
        "",
        f"- total records: {len(rows)}",
        f"- passed: {len(passed)}",
        f"- rejected: {len(rejected)}",
        f"- pass rate: {pass_rate:.2f}%",
        "",
        "## Status",
        "",
    ]
    for key, value in status.most_common():
        md.append(f"- {key}: {value}")
    md.extend(["", "## Reject Reasons", ""])
    for key, value in reasons.most_common():
        md.append(f"- {key}: {value}")
    md.extend(["", "## Passed Examples", ""])
    for row in passed[:20]:
        md.append(
            "- "
            f"{row.get('asset_id')} | bones={row.get('max_bones')} | "
            f"skinned_v={row.get('skinned_vertices')} | "
            f"frames={row.get('max_action_frames')} | "
            f"{row.get('candidate', '')}"
        )

    (args.out_dir / f"{args.name}_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
