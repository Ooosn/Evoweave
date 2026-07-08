#!/usr/bin/env python3
"""Deduplicate sequence NPZ files by a small exact motion fingerprint.

`frame0_vertices_hash` is an identity/group key: it finds the same character or
same static mesh.  It is not enough to delete a sample, because the same mesh can
carry multiple different actions.  A sample is removed only when a fixed subset
of frames from `frame_vertices` first matches, then the candidate pair is
verified as byte-identical over the full sequence.  This keeps the common case
cheap while avoiding false deletes for different actions with the same first
frame or same sparse fingerprint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact dynamic sequence dedup for RigWeave sequence NPZ files.")
    parser.add_argument("--manifest-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--npz-dir", type=Path, action="append", default=[])
    parser.add_argument("--fingerprint-frames", type=int, default=8)
    parser.add_argument("--out-keep-jsonl", type=Path, required=True)
    parser.add_argument("--out-duplicate-jsonl", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def collect_paths(args: argparse.Namespace) -> list[tuple[Path, dict]]:
    items: list[tuple[Path, dict]] = []
    for npz_dir in args.npz_dir:
        for path in sorted(npz_dir.glob("*.npz")):
            items.append((path, {"path": str(path)}))
    for manifest in args.manifest_jsonl:
        for row in read_jsonl(manifest):
            if row.get("status") not in (None, "clean"):
                continue
            value = row.get("npz_path") or row.get("path")
            if value:
                items.append((Path(value), row))
    unique: dict[str, tuple[Path, dict]] = {}
    for path, row in items:
        if path.is_absolute():
            key_path = path
        else:
            key_path = path.resolve()
        unique[key_path.as_posix()] = (key_path, row)
    return sorted(unique.values(), key=lambda item: item[0].as_posix())


def _array_hash(array: np.ndarray) -> tuple[str, tuple[int, ...]]:
    value = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.tobytes())
    return digest.hexdigest(), tuple(int(x) for x in value.shape)


def fingerprint_indices(frame_count: int, fingerprint_frames: int) -> np.ndarray:
    if frame_count <= 0:
        raise ValueError("empty frame sequence")
    count = max(1, min(int(fingerprint_frames), int(frame_count)))
    return np.unique(np.linspace(0, frame_count - 1, count).round().astype(np.int64))


def full_sequence_hash(path: Path) -> tuple[str, tuple[int, ...]]:
    with np.load(path, allow_pickle=True) as raw:
        frames = np.asarray(raw["frame_vertices"], dtype=np.float32)
    return _array_hash(frames)


def sequence_hashes(
    path: Path,
    fingerprint_frames: int,
) -> tuple[str, tuple[int, ...], str, tuple[int, ...], str, tuple[int, ...], list[int]]:
    with np.load(path, allow_pickle=True) as raw:
        frames = np.asarray(raw["frame_vertices"], dtype=np.float32)
        frame0_hash, frame0_shape = _array_hash(frames[0])
        idx = fingerprint_indices(frames.shape[0], fingerprint_frames)
        fingerprint = frames[idx]
        fingerprint_hash, fingerprint_shape = _array_hash(fingerprint)
        motion_hash, motion_shape = _array_hash(fingerprint - frames[:1])
    return (
        frame0_hash,
        frame0_shape,
        fingerprint_hash,
        fingerprint_shape,
        motion_hash,
        motion_shape,
        [int(x) for x in idx.tolist()],
    )


def _shape_tuple(value: object) -> tuple[int, ...] | None:
    if value is None:
        return None
    try:
        shape = tuple(int(x) for x in value)  # type: ignore[arg-type]
    except Exception:
        return None
    return shape if shape else None


def manifest_sequence_hashes(
    row: dict,
    fingerprint_frames: int,
) -> tuple[str, tuple[int, ...], str, tuple[int, ...], str, tuple[int, ...], list[int]] | None:
    frame0_hash = row.get("frame0_vertices_hash")
    fingerprint_hash = row.get("sequence_fingerprint_hash")
    motion_hash = row.get("sequence_fingerprint_motion_delta_hash")
    frame0_shape = _shape_tuple(row.get("frame0_vertices_shape"))
    fingerprint_shape = _shape_tuple(row.get("sequence_fingerprint_shape"))
    motion_shape = _shape_tuple(row.get("motion_delta_shape"))
    raw_idx = row.get("sequence_fingerprint_frame_indices")
    if not (
        isinstance(frame0_hash, str)
        and isinstance(fingerprint_hash, str)
        and isinstance(motion_hash, str)
        and frame0_shape is not None
        and fingerprint_shape is not None
        and motion_shape is not None
        and isinstance(raw_idx, list)
    ):
        return None
    try:
        fingerprint_idx = [int(x) for x in raw_idx]
    except Exception:
        return None
    expected_count = max(1, int(fingerprint_frames))
    if len(fingerprint_idx) != expected_count:
        return None
    if len(fingerprint_shape) < 1 or int(fingerprint_shape[0]) != len(fingerprint_idx):
        return None
    if len(motion_shape) < 1 or int(motion_shape[0]) != len(fingerprint_idx):
        return None
    return (
        frame0_hash,
        frame0_shape,
        fingerprint_hash,
        fingerprint_shape,
        motion_hash,
        motion_shape,
        fingerprint_idx,
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    seen_fingerprint: dict[str, list[dict]] = {}
    seen_frame0: dict[str, dict] = {}
    keep = []
    dup = []
    hash_source_counts = {"manifest": 0, "npz": 0}
    for idx, (path, row) in enumerate(collect_paths(args), start=1):
        try:
            cached = manifest_sequence_hashes(row, args.fingerprint_frames)
            if cached is None:
                (
                    frame0_digest,
                    frame0_shape,
                    fingerprint_digest,
                    fingerprint_shape,
                    motion_digest,
                    motion_shape,
                    fingerprint_idx,
                ) = sequence_hashes(path, args.fingerprint_frames)
                hash_source_counts["npz"] += 1
            else:
                (
                    frame0_digest,
                    frame0_shape,
                    fingerprint_digest,
                    fingerprint_shape,
                    motion_digest,
                    motion_shape,
                    fingerprint_idx,
                ) = cached
                hash_source_counts["manifest"] += 1
            out = dict(row)
            out["path"] = str(path)
            out["frame0_vertices_hash"] = frame0_digest
            out["frame0_vertices_shape"] = list(frame0_shape)
            out["sequence_fingerprint_frame_indices"] = fingerprint_idx
            out["sequence_fingerprint_hash"] = fingerprint_digest
            out["sequence_fingerprint_shape"] = list(fingerprint_shape)
            out["sequence_fingerprint_motion_delta_hash"] = motion_digest
            out["motion_delta_shape"] = list(motion_shape)
            if frame0_digest in seen_frame0:
                out["same_identity_as"] = seen_frame0[frame0_digest].get("asset_id") or seen_frame0[frame0_digest].get("path")
            else:
                seen_frame0[frame0_digest] = out
            duplicate_of = None
            full_digest = None
            full_shape = None
            for previous in seen_fingerprint.get(fingerprint_digest, []):
                if full_digest is None:
                    full_digest, full_shape = full_sequence_hash(path)
                    out["full_sequence_verification_hash"] = full_digest
                    out["full_sequence_verification_shape"] = list(full_shape)
                previous_full = previous.get("full_sequence_verification_hash")
                if previous_full is None:
                    previous_full, previous_shape = full_sequence_hash(Path(previous["path"]))
                    previous["full_sequence_verification_hash"] = previous_full
                    previous["full_sequence_verification_shape"] = list(previous_shape)
                if full_digest == previous_full:
                    duplicate_of = previous
                    break

            if duplicate_of is not None:
                out["status"] = "duplicate_verified_full_sequence"
                out["duplicate_of"] = duplicate_of.get("asset_id") or duplicate_of.get("path")
                dup.append(out)
            else:
                out["status"] = row.get("status", "keep_sequence_unique")
                if fingerprint_digest in seen_fingerprint:
                    out["same_sequence_fingerprint_as"] = (
                        seen_fingerprint[fingerprint_digest][0].get("asset_id")
                        or seen_fingerprint[fingerprint_digest][0].get("path")
                    )
                seen_fingerprint.setdefault(fingerprint_digest, []).append(out)
                keep.append(out)
        except Exception as exc:  # noqa: BLE001
            dup.append({"path": str(path), "status": "sequence_hash_error", "error": repr(exc)})
        if args.progress_every > 0 and idx % int(args.progress_every) == 0:
            print(
                json.dumps(
                    {
                        "duplicates": len(dup),
                        "event": "dedupe_progress",
                        "hash_source_counts": hash_source_counts,
                        "keep": len(keep),
                        "processed": idx,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
    write_jsonl(args.out_keep_jsonl, keep)
    write_jsonl(args.out_duplicate_jsonl, dup)
    same_identity_different_motion = sum(1 for row in keep if "same_identity_as" in row)
    print(
        json.dumps(
            {
                "input": len(keep) + len(dup),
                "keep": len(keep),
                "duplicates": len(dup),
                "hash_source_counts": hash_source_counts,
                "same_identity_different_motion_keep": same_identity_different_motion,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
