#!/usr/bin/env python3
"""Create rebuild source manifests from the Baidu transfer manifest.

The transfer manifest is the source of truth after the old NPZ files are
deleted: each row maps one old derived NPZ to one original downloaded asset.
This script does not filter data. It only writes clean source manifests for the
Pass1 Blender export stage.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SOURCE_NAMES = ("texverse", "objaverse_xl", "objaverse_xl_more_anim")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer-manifest-csv", type=Path, required=True)
    parser.add_argument("--transfer-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fail-on-missing-original", action="store_true")
    args = parser.parse_args()

    transfer_root = args.transfer_root.resolve()
    out_dir = args.out_dir
    by_source: dict[str, list[dict[str, Any]]] = {name: [] for name in SOURCE_NAMES}
    texverse_model_paths: list[str] = []
    texverse_animation_ids: list[str] = []
    split_rows: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    seen_asset_ids: set[str] = set()

    with args.transfer_manifest_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source = str(row.get("dataset_source") or "").strip()
            asset_id = str(row.get("asset_id") or "").strip()
            split = str(row.get("split") or "").strip()
            rel = str(row.get("transfer_original_rel") or "").strip()
            if source not in by_source or not asset_id or not rel:
                continue
            if asset_id in seen_asset_ids:
                raise ValueError(f"duplicate asset_id in transfer manifest: {asset_id}")
            seen_asset_ids.add(asset_id)

            original = transfer_root / rel
            if not original.exists():
                missing.append({"asset_id": asset_id, "dataset_source": source, "path": str(original)})
                if args.fail_on_missing_original:
                    continue

            split_rows.append({"asset_id": asset_id, "dataset_source": source, "split": split})
            common = {
                "asset_id": asset_id,
                "dataset_source": source,
                "split": split,
                "original_path": str(original),
                "transfer_original_rel": rel,
                "status": "ok" if original.exists() else "missing_original",
            }
            if source == "texverse":
                # TexVerse zips can contain several importable FBX/GLB files.
                # This stage only records the preserved outer zip.  The real
                # source-file choice must be made by texverse_quality_audit.py
                # from the original zip contents, not by a placeholder
                # candidate and not by any old derived NPZ.
                by_source[source].append({**common, "usable": original.exists()})
                texverse_model_paths.append(rel)
                texverse_animation_ids.append(asset_id)
            else:
                by_source[source].append({**common, "path": str(original)})

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "texverse_originals.jsonl", by_source["texverse"])
    write_jsonl(out_dir / "objaverse_xl_download_rebuild_manifest.jsonl", by_source["objaverse_xl"])
    write_jsonl(
        out_dir / "objaverse_xl_more_anim_download_rebuild_manifest.jsonl",
        by_source["objaverse_xl_more_anim"],
    )
    write_jsonl(out_dir / "missing_originals.jsonl", missing)
    with (out_dir / "split_map.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["asset_id", "dataset_source", "split"])
        writer.writeheader()
        writer.writerows(split_rows)
    (out_dir / "texverse_model_paths.txt").write_text(
        "\n".join(texverse_model_paths) + ("\n" if texverse_model_paths else ""),
        encoding="utf-8",
    )
    (out_dir / "texverse_animation_ids.txt").write_text(
        "\n".join(texverse_animation_ids) + ("\n" if texverse_animation_ids else ""),
        encoding="utf-8",
    )

    summary = {
        "transfer_manifest_csv": str(args.transfer_manifest_csv),
        "transfer_root": str(transfer_root),
        "out_dir": str(out_dir),
        "counts": {name: len(rows) for name, rows in by_source.items()},
        "missing_original_count": len(missing),
        "split_map_rows": len(split_rows),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if missing and args.fail_on_missing_original:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
