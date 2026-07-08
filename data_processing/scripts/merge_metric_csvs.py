#!/usr/bin/env python3
"""Merge compatible metric CSV files without changing rows or thresholds."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("metrics_csv", type=Path, nargs="+")
    args = parser.parse_args()

    fieldnames: list[str] | None = None
    rows: list[dict[str, str]] = []
    for path in args.metrics_csv:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            current = list(reader.fieldnames or [])
            if fieldnames is None:
                fieldnames = current
            elif current != fieldnames:
                raise SystemExit(f"CSV schema mismatch: {path}")
            rows.extend(dict(row) for row in reader)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or [])
        writer.writeheader()
        writer.writerows(rows)

    print(f"merged {len(rows)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
