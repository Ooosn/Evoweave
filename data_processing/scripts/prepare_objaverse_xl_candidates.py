#!/usr/bin/env python3
"""Prepare Objaverse-XL metadata candidates for dynamic rigging screening.

This script only reads Objaverse-XL annotations.  It does not download model
assets.  The resulting manifest is a proxy candidate list: each row is a file
whose source path/metadata suggests both rigging and animation and whose file
type can carry Blender-importable rig data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import quote

import pandas as pd


CAPABLE_TYPES = {"fbx", "blend", "glb", "gltf", "dae", "usd", "usdz"}
RIG_WORDS = (
    "rig",
    "rigged",
    "skeleton",
    "armature",
    "bone",
    "bones",
    "skinned",
    "skin",
    "character",
    "avatar",
    "humanoid",
)
ANIM_WORDS = (
    "anim",
    "animation",
    "animated",
    "motion",
    "walk",
    "run",
    "idle",
    "dance",
    "attack",
    "action",
    "pose",
)
EXTRA_ANIM_WORDS = (
    "jump",
    "turn",
    "cycle",
    "crawl",
    "fly",
    "swim",
    "climb",
    "fight",
    "hit",
    "kick",
    "punch",
    "shoot",
    "melee",
    "gesture",
    "emote",
    "moving",
    "move",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an Objaverse-XL dynamic-rig proxy manifest.")
    parser.add_argument("--annotations-dir", type=Path, default=Path("objaverse_xl_audit"))
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, default=None)
    parser.add_argument("--include-usd", action="store_true")
    parser.add_argument(
        "--expanded-animation-words",
        action="store_true",
        help="Add motion/action verbs such as jump, turn, kick, climb. Keeps rig words unchanged.",
    )
    parser.add_argument(
        "--exclude-sha-jsonl",
        type=Path,
        default=None,
        help="Optional existing candidate JSONL whose sha256 values should be excluded.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_annotations(root: Path) -> pd.DataFrame:
    paths = [
        root / "github" / "github.parquet",
        root / "thingiverse" / "thingiverse.parquet",
        root / "smithsonian" / "smithsonian.parquet",
        root / "sketchfab" / "sketchfab.parquet",
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"missing Objaverse-XL annotation parquet files: {missing}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def regex(words: tuple[str, ...]) -> str:
    return "|".join(re.escape(word) for word in words)


def github_raw_url(identifier: str) -> str | None:
    match = re.match(r"https://github.com/([^/]+/[^/]+)/blob/([^/]+)/(.*)", identifier)
    if not match:
        return None
    repo, commit, rel = match.groups()
    quoted_rel = "/".join(quote(part) for part in rel.split("/"))
    return f"https://raw.githubusercontent.com/{repo}/{commit}/{quoted_rel}"


def github_repo(identifier: str) -> str | None:
    match = re.match(r"https://github.com/([^/]+/[^/]+)/blob/([^/]+)/(.*)", identifier)
    return match.group(1).lower() if match else None


def stable_asset_id(identifier: str, sha256: str) -> str:
    token = sha256 if sha256 else identifier
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:20]
    return f"oxl_{digest}"


def main() -> None:
    args = parse_args()
    df = load_annotations(args.annotations_dir)
    file_type = df["fileType"].astype(str).str.lower()
    capable = set(CAPABLE_TYPES)
    if not args.include_usd:
        capable -= {"usd", "usdz"}
    anim_words = ANIM_WORDS + (EXTRA_ANIM_WORDS if args.expanded_animation_words else ())
    text = (df["fileIdentifier"].fillna("") + " " + df["metadata"].fillna("")).str.lower()
    mask = file_type.isin(capable) & text.str.contains(regex(RIG_WORDS), regex=True) & text.str.contains(
        regex(anim_words), regex=True
    )
    candidates = df.loc[mask, ["fileIdentifier", "source", "license", "fileType", "sha256", "metadata"]].copy()
    candidates["fileType"] = candidates["fileType"].astype(str).str.lower()
    candidates["asset_id"] = [
        stable_asset_id(identifier, sha)
        for identifier, sha in zip(candidates["fileIdentifier"].astype(str), candidates["sha256"].astype(str))
    ]
    candidates["github_repo"] = [github_repo(identifier) for identifier in candidates["fileIdentifier"].astype(str)]
    candidates["download_url"] = [github_raw_url(identifier) for identifier in candidates["fileIdentifier"].astype(str)]
    candidates = candidates[candidates["download_url"].notna()].copy()
    candidates = candidates.drop_duplicates(subset=["sha256"]).sort_values(["github_repo", "fileIdentifier"])
    excluded_sha_count = 0
    if args.exclude_sha_jsonl is not None:
        excluded_sha: set[str] = set()
        with args.exclude_sha_jsonl.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                value = row.get("sha256")
                if value:
                    excluded_sha.add(str(value))
        before = len(candidates)
        candidates = candidates[~candidates["sha256"].astype(str).isin(excluded_sha)].copy()
        excluded_sha_count = before - len(candidates)
    if args.shuffle:
        candidates = candidates.sample(frac=1.0, random_state=args.seed)
    if args.limit > 0:
        candidates = candidates.sample(n=min(args.limit, len(candidates)), random_state=args.seed).sort_values("asset_id")

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for row in candidates.to_dict(orient="records"):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "annotations_rows": int(len(df)),
        "candidate_rows": int(len(candidates)),
        "unique_sha256": int(candidates["sha256"].nunique(dropna=True)),
        "unique_github_repos": int(candidates["github_repo"].nunique(dropna=True)),
        "file_type_counts": candidates["fileType"].value_counts().to_dict(),
        "source_counts": candidates["source"].value_counts().to_dict(),
        "capable_types": sorted(capable),
        "expanded_animation_words": bool(args.expanded_animation_words),
        "excluded_sha_count": int(excluded_sha_count),
        "rig_words": list(RIG_WORDS),
        "animation_words": list(anim_words),
    }
    summary_path = args.out_summary or args.out_jsonl.with_suffix(".summary.json")
    Path(summary_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
