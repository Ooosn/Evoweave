#!/usr/bin/env python3
"""Combine per-source Pass1 precheck manifests before rootless rewriting."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source must be NAME=PRECHECK_DIR")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("source name is empty")
    return name, Path(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def row_key(row: dict[str, Any]) -> str:
    asset_id = str(row.get("asset_id", ""))
    sequence_id = str(row.get("sequence_id", "0"))
    return f"{asset_id}::{sequence_id}"


def load_source(name: str, precheck_dir: Path, *, check_paths: bool) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    issues: list[dict[str, Any]] = []
    for split in SPLITS:
        manifest = precheck_dir / f"{split}_manifest.jsonl"
        rows = read_jsonl(manifest)
        for index, row in enumerate(rows):
            out = dict(row)
            out["dataset_source"] = name
            out["source_precheck_dir"] = str(precheck_dir)
            out["source_manifest"] = str(manifest)
            out["source_manifest_index"] = int(index)
            out["split"] = split
            if check_paths:
                path = Path(str(out.get("path", "")))
                if not path.exists():
                    issues.append(
                        {
                            "source": name,
                            "split": split,
                            "asset_id": out.get("asset_id"),
                            "path": str(path),
                            "reason": "missing_npz_path",
                        }
                    )
            by_split[split].append(out)
    return by_split, issues


def combine_sources(sources: list[tuple[str, Path]], *, check_paths: bool) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    combined: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    issues: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    duplicates: list[dict[str, Any]] = []
    source_counts: dict[str, dict[str, int]] = {}

    for name, precheck_dir in sources:
        rows_by_split, source_issues = load_source(name, precheck_dir, check_paths=check_paths)
        issues.extend(source_issues)
        source_counts[name] = {split: len(rows_by_split[split]) for split in SPLITS}
        for split in SPLITS:
            for row in rows_by_split[split]:
                key = row_key(row)
                if key in seen:
                    duplicates.append(
                        {
                            "key": key,
                            "kept_source": seen[key].get("dataset_source"),
                            "dropped_source": name,
                            "kept_path": seen[key].get("path"),
                            "dropped_path": row.get("path"),
                        }
                    )
                    continue
                seen[key] = row
                combined[split].append(row)

    for split in SPLITS:
        combined[split].sort(key=lambda row: (str(row.get("dataset_source")), str(row.get("asset_id")), str(row.get("sequence_id", "0"))))

    summary = {
        "sources": [{"name": name, "precheck_dir": str(path)} for name, path in sources],
        "source_counts": source_counts,
        "combined_counts": {split: len(combined[split]) for split in SPLITS},
        "combined_total": int(sum(len(combined[split]) for split in SPLITS)),
        "duplicate_sequence_count": int(len(duplicates)),
        "duplicates_first50": duplicates[:50],
        "path_issue_count": int(len(issues)),
        "path_issues_first50": issues[:50],
        "dataset_source_counts": dict(Counter(str(row.get("dataset_source")) for split in SPLITS for row in combined[split])),
    }
    return combined, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=parse_source, required=True, help="NAME=PRECHECK_DIR. Repeatable.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-paths", action="store_true")
    args = parser.parse_args()

    combined, summary = combine_sources(args.source, check_paths=bool(args.check_paths))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        write_jsonl(args.output_dir / f"{split}_manifest.jsonl", combined[split])
    write_jsonl(args.output_dir / "accepted.jsonl", [row for split in SPLITS for row in combined[split]])
    (args.output_dir / "combine_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    if args.check_paths and summary["path_issue_count"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
