#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_dynamic_rig import build_tokenizer  # noqa: E402
from rigweave.stack_close import StackCloseTokenizer  # noqa: E402


def _manifest_paths(path: Path) -> list[Path]:
    rows: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        value = row.get("path") or row.get("npz_path") or row.get("file")
        if not value:
            raise ValueError(f"manifest row has no path: {line[:160]}")
        rows.append(Path(value))
    return rows


def _seed(path: Path, base_seed: int, draw: int) -> int:
    digest = hashlib.blake2b(digest_size=8)
    digest.update(str(path).encode("utf-8"))
    digest.update(int(base_seed).to_bytes(8, "little", signed=False))
    digest.update(int(draw).to_bytes(4, "little", signed=False))
    return int.from_bytes(digest.digest(), "little") % (2**32)


def _edge_identity(
    parents: np.ndarray,
    original_indices: np.ndarray,
) -> set[tuple[int, int]]:
    return {
        (
            int(original_indices[parent]),
            int(original_indices[child]),
        )
        for child, parent in enumerate(parents.tolist())
        if parent >= 0
    }


def _audit_manifest(
    manifest: Path,
    tokenizer: StackCloseTokenizer,
    *,
    sibling_draws: int,
    seed: int,
    limit: int,
) -> dict[str, Any]:
    paths = _manifest_paths(manifest)
    if limit > 0:
        paths = paths[:limit]
    started = time.perf_counter()
    total_tokens = 0
    max_tokens = 0
    joint_counts: list[int] = []
    sibling_order_changed = 0
    duplicate_quantized_rows = 0
    for row_index, path in enumerate(paths):
        with np.load(path, allow_pickle=True) as raw:
            vertices = np.asarray(
                raw["frame_vertices_rootspace"][0],
                dtype=np.float32,
            )
            joints = np.asarray(
                raw["target_joints_rootspace"][0],
                dtype=np.float32,
            )
            parents = np.asarray(raw["target_parents"], dtype=np.int64)
        center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
        scale = float(np.max((vertices.max(axis=0) - vertices.min(axis=0)) * 0.5))
        if not np.isfinite(scale) or scale <= 1.0e-8:
            raise ValueError(f"{path}: degenerate frame-0 query bbox")
        joints = (joints - center) / scale
        expected_edges = {
            (int(parent), int(child))
            for child, parent in enumerate(parents.tolist())
            if parent >= 0
        }

        canonical = tokenizer.serialize_tree(
            joints,
            parents,
            cls="articulationxl",
        )
        decoded = tokenizer.detokenize(canonical.tokens)
        decoded_parents = np.asarray(
            [-1 if parent is None else parent for parent in decoded.parents],
            dtype=np.int64,
        )
        if not np.array_equal(decoded_parents, canonical.parents):
            raise AssertionError(f"{path}: canonical parent round-trip mismatch")
        if _edge_identity(
            canonical.parents,
            canonical.original_indices,
        ) != expected_edges:
            raise AssertionError(f"{path}: canonical edge identity mismatch")
        if canonical.tokens.shape[0] != 4 * joints.shape[0] + 3:
            raise AssertionError(f"{path}: stack token length formula mismatch")
        if int(np.count_nonzero(canonical.tokens == tokenizer.token_id_close)) != joints.shape[0]:
            raise AssertionError(f"{path}: CLOSE count does not equal joint count")
        if canonical.coordinate_token_positions.shape != (joints.shape[0], 3):
            raise AssertionError(f"{path}: coordinate position map shape mismatch")
        coordinate_tokens = canonical.tokens[canonical.coordinate_token_positions]
        if not np.array_equal(coordinate_tokens, tokenizer.discretize(canonical.joints)):
            raise AssertionError(f"{path}: coordinate position map values mismatch")

        quantized = tokenizer.discretize(joints)
        if np.unique(quantized, axis=0).shape[0] != quantized.shape[0]:
            duplicate_quantized_rows += 1

        canonical_order = tuple(canonical.original_indices.tolist())
        row_changed = False
        for draw in range(sibling_draws):
            randomized = tokenizer.serialize_tree(
                joints,
                parents,
                cls="articulationxl",
                sibling_rng=np.random.default_rng(_seed(path, seed, draw)),
            )
            random_decoded = tokenizer.detokenize(randomized.tokens)
            random_parents = np.asarray(
                [
                    -1 if parent is None else parent
                    for parent in random_decoded.parents
                ],
                dtype=np.int64,
            )
            if not np.array_equal(random_parents, randomized.parents):
                raise AssertionError(f"{path}: randomized parent round-trip mismatch")
            if _edge_identity(
                randomized.parents,
                randomized.original_indices,
            ) != expected_edges:
                raise AssertionError(f"{path}: randomized edge identity mismatch")
            row_changed = row_changed or (
                tuple(randomized.original_indices.tolist()) != canonical_order
            )
        sibling_order_changed += int(row_changed)

        total_tokens += int(canonical.tokens.shape[0])
        max_tokens = max(max_tokens, int(canonical.tokens.shape[0]))
        joint_counts.append(int(joints.shape[0]))
        if (row_index + 1) % 1000 == 0:
            print(
                f"[stack audit] {manifest.name} rows={row_index + 1}/{len(paths)}",
                flush=True,
            )

    return {
        "manifest": str(manifest),
        "rows": len(paths),
        "joint_total": int(sum(joint_counts)),
        "joint_p50": float(np.percentile(joint_counts, 50)) if joint_counts else 0.0,
        "joint_p95": float(np.percentile(joint_counts, 95)) if joint_counts else 0.0,
        "token_total": total_tokens,
        "token_max": max_tokens,
        "duplicate_quantized_rows": duplicate_quantized_rows,
        "sibling_draws": sibling_draws,
        "sibling_order_changed_rows": sibling_order_changed,
        "round_trip_failures": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit exact stack-close tree round-trip on frozen manifests.",
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sibling-draws", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.sibling_draws < 0:
        parser.error("--sibling-draws must be non-negative")

    legacy = build_tokenizer(args.tokenizer_config)
    tokenizer = StackCloseTokenizer(legacy)
    report = {
        "contract": "stack_close_v1",
        "close_token_id": tokenizer.token_id_close,
        "legacy_branch_token_id": int(legacy.token_id_branch),
        "vocab_size": tokenizer.vocab_size,
        "num_discrete": tokenizer.num_discrete,
        "baseline_files_modified": False,
        "splits": [
            _audit_manifest(
                manifest,
                tokenizer,
                sibling_draws=args.sibling_draws,
                seed=args.seed,
                limit=args.limit,
            )
            for manifest in (args.train_manifest, args.val_manifest)
        ],
    }
    report["rows"] = sum(split["rows"] for split in report["splits"])
    report["round_trip_failures"] = sum(
        split["round_trip_failures"] for split in report["splits"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    os.environ.setdefault("RIGWEAVE_DISABLE_OPEN3D", "1")
    main()
