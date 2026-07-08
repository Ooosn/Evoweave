#!/usr/bin/env python3
"""Prepare TexVerse zip manifests for the RigWeave data pipeline.

This script does not download assets.  It lists the Hugging Face dataset repo
and writes the two text files consumed by `texverse_quality_audit.py`:

  model_paths.txt    one relative zip path per line
  animation_ids.txt  one asset id per line, derived from zip stems
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_REPO = "YiboZhang2001/TexVerse-Skeleton-Animation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--suffix", default=".zip")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = HfApi()
    files = api.list_repo_files(args.repo_id, repo_type="dataset")
    suffix = args.suffix.lower()
    zip_paths = sorted(path for path in files if path.lower().endswith(suffix))
    if args.shuffle:
        random.Random(args.seed).shuffle(zip_paths)
    if args.limit > 0:
        zip_paths = zip_paths[: args.limit]

    asset_ids = [Path(path).stem for path in zip_paths]
    duplicate_ids = sorted(asset_id for asset_id, count in Counter(asset_ids).items() if count > 1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "model_paths.txt").write_text(
        "\n".join(zip_paths) + ("\n" if zip_paths else ""),
        encoding="utf-8",
    )
    (args.output_dir / "animation_ids.txt").write_text(
        "\n".join(asset_ids) + ("\n" if asset_ids else ""),
        encoding="utf-8",
    )
    summary = {
        "repo_id": args.repo_id,
        "zip_count": len(zip_paths),
        "unique_asset_ids": len(set(asset_ids)),
        "duplicate_asset_ids": duplicate_ids[:100],
        "duplicate_asset_id_count": len(duplicate_ids),
        "model_paths": str(args.output_dir / "model_paths.txt"),
        "animation_ids": str(args.output_dir / "animation_ids.txt"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
