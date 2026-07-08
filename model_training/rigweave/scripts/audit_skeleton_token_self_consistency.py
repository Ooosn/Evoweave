#!/usr/bin/env python3
"""Audit whether dynamic-rig skeleton targets survive UniRig token round-trip.

This checks the data/representation contract before blaming the generator:

NPZ -> DynamicRigManifestDataset target_joints/target_parents
    -> tokenizer input_ids -> tokenizer.detokenize(input_ids)
    -> parent/edge consistency.

The important failure mode is branch-parent ambiguity. UniRig branch tokens store
the parent joint coordinate, not a parent index. Detokenization recovers the
parent by nearest previous joint coordinate. If previous joints collide after
quantization, the original parent may be unrecoverable from the token stream.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from string import Template
from typing import Any

import numpy as np


def _load_env(path: Path) -> None:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    combined = os.environ.copy()
    for _ in range(4):
        changed = False
        for key, value in list(values.items()):
            expanded = Template(value).safe_substitute({**combined, **values})
            if expanded != value:
                values[key] = expanded
                changed = True
        if not changed:
            break
    os.environ.update(values)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def _load_train_helpers(root: Path):
    train_script = root / "rigweave/scripts/train_dynamic_rig.py"
    spec = importlib.util.spec_from_file_location("train_dynamic_rig", train_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import {train_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parents_from_output(output: Any) -> list[int]:
    parents = getattr(output, "parents", None)
    if parents is None:
        parents = output._get_parents()
    return [-1 if p is None else int(p) for p in parents]


def _edge_set(parents: list[int]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for child, parent in enumerate(parents):
        if parent < 0 or parent >= len(parents) or parent == child:
            continue
        edges.add((int(parent), int(child)))
    return edges


def _f1(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> tuple[float, float, float]:
    if not a and not b:
        return 1.0, 1.0, 1.0
    p = len(a & b) / max(len(a), 1)
    r = len(a & b) / max(len(b), 1)
    f1 = 0.0 if p + r <= 1.0e-12 else 2.0 * p * r / (p + r)
    return float(p), float(r), float(f1)


def _nn_edge_f1(
    pred_joints: np.ndarray,
    pred_parents: list[int],
    target_joints: np.ndarray,
    target_parents: list[int],
) -> tuple[float, float, float]:
    def overlap(source_joints, source_parents, target_joints, target_parents):
        source_edges = _edge_set(source_parents)
        target_edges = _edge_set(target_parents)
        if not source_edges:
            return 1.0 if not target_edges else 0.0
        if target_joints.size == 0:
            return 0.0
        d = np.linalg.norm(source_joints[:, None, :] - target_joints[None, :, :], axis=-1)
        mapping = d.argmin(axis=1)
        matched = 0
        for parent, child in source_edges:
            if (int(mapping[parent]), int(mapping[child])) in target_edges:
                matched += 1
        return matched / max(len(source_edges), 1)

    p = overlap(pred_joints, pred_parents, target_joints, target_parents)
    r = overlap(target_joints, target_parents, pred_joints, pred_parents)
    f1 = 0.0 if p + r <= 1.0e-12 else 2.0 * p * r / (p + r)
    return float(p), float(r), float(f1)


def _stats(values: list[float]) -> dict[str, float | int | None]:
    clean = [float(x) for x in values if math.isfinite(float(x))]
    if not clean:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    arr = np.asarray(clean, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _qkey(row: np.ndarray) -> tuple[int, int, int]:
    return (int(row[0]), int(row[1]), int(row[2]))


def audit_one(sample: Any, tokenizer: Any, discretize_fn: Any, continuous_range: tuple[float, float]) -> dict[str, Any]:
    target_joints = sample.target_joints.detach().cpu().numpy().astype(np.float32)
    target_parents = [int(x) for x in sample.target_parents.detach().cpu().tolist()]
    ids = sample.input_ids.detach().cpu().numpy().astype(np.int64)
    row: dict[str, Any] = {
        "path": sample.path,
        "name": Path(sample.path).name,
        "target_joint_count": int(target_joints.shape[0]),
        "source_joint_count": int(sample.source_joint_count),
        "active_skin_joint_count": int(sample.active_skin_joint_count),
        "target_start_index": int(sample.target_start_index),
        "alignment_root_index": int(sample.alignment_root_index),
        "selected_query_frame": int(sample.selected_frames[0]),
        "token_count": int(ids.shape[0]),
        "target_abs_max": float(np.abs(target_joints).max()) if target_joints.size else 0.0,
    }
    q = discretize_fn(target_joints, continuous_range=continuous_range, num_discrete=int(tokenizer.num_discrete))
    q = np.asarray(q, dtype=np.int64)
    keys = [_qkey(x) for x in q]
    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for idx, key in enumerate(keys):
        groups[key].append(idx)
    duplicate_groups = [v for v in groups.values() if len(v) > 1]
    row["duplicate_quantized_coord_groups"] = int(len(duplicate_groups))
    row["duplicate_quantized_joint_count"] = int(sum(len(v) for v in duplicate_groups))
    row["duplicate_quantized_max_group"] = int(max((len(v) for v in duplicate_groups), default=1))

    quant_parent_child_same = 0
    continuous_zero_edges = 0
    branch_count = 0
    branch_parent_tie_count = 0
    branch_parent_bad_tie_count = 0
    branch_parent_examples = []
    for child, parent in enumerate(target_parents):
        if parent < 0:
            continue
        if np.array_equal(q[child], q[parent]):
            quant_parent_child_same += 1
        if float(np.linalg.norm(target_joints[child] - target_joints[parent])) <= 1.0e-8:
            continuous_zero_edges += 1
        if parent != child - 1:
            branch_count += 1
            parent_key = keys[parent]
            tied_previous = [j for j in range(child) if keys[j] == parent_key]
            if len(tied_previous) > 1:
                branch_parent_tie_count += 1
                detok_tie_choice = max(tied_previous)
                bad = detok_tie_choice != parent
                branch_parent_bad_tie_count += int(bad)
                if len(branch_parent_examples) < 8:
                    branch_parent_examples.append(
                        {
                            "child": int(child),
                            "target_parent": int(parent),
                            "same_coord_previous": tied_previous,
                            "detokenizer_tie_choice": int(detok_tie_choice),
                            "bad": bool(bad),
                            "parent_q": list(parent_key),
                        }
                    )
    row["branch_count"] = int(branch_count)
    row["quant_parent_child_same_edges"] = int(quant_parent_child_same)
    row["continuous_zero_length_edges"] = int(continuous_zero_edges)
    row["branch_parent_tie_count"] = int(branch_parent_tie_count)
    row["branch_parent_bad_tie_count"] = int(branch_parent_bad_tie_count)
    row["branch_parent_tie_examples"] = branch_parent_examples

    try:
        detok = tokenizer.detokenize(ids)
        detok_joints = np.asarray(detok.joints, dtype=np.float32)
        detok_parents = _parents_from_output(detok)
        row["detokenize_ok"] = True
        row["detok_joint_count"] = int(detok_joints.shape[0])
        row["detok_parent_count"] = int(len(detok_parents))
        n = min(len(detok_parents), len(target_parents), int(detok_joints.shape[0]), int(target_joints.shape[0]))
        if n > 0:
            diff = np.abs(detok_joints[:n] - target_joints[:n])
            row["aligned_joint_linf"] = float(diff.max())
            row["aligned_joint_mae"] = float(diff.mean())
        else:
            row["aligned_joint_linf"] = None
            row["aligned_joint_mae"] = None
        parent_mismatches = [
            {
                "child": int(i),
                "target_parent": int(target_parents[i]),
                "detok_parent": int(detok_parents[i]),
                "target_parent_q": list(keys[target_parents[i]]) if target_parents[i] >= 0 else None,
                "child_q": list(keys[i]) if i < len(keys) else None,
            }
            for i in range(n)
            if int(detok_parents[i]) != int(target_parents[i])
        ]
        row["parent_mismatch_count"] = int(len(parent_mismatches))
        row["parent_mismatch_examples"] = parent_mismatches[:12]
        target_edges = _edge_set(target_parents[:n])
        detok_edges = _edge_set(detok_parents[:n])
        p, r, f1 = _f1(detok_edges, target_edges)
        row["id_edge_precision"] = p
        row["id_edge_recall"] = r
        row["id_edge_f1"] = f1
        np_, nr, nf1 = _nn_edge_f1(detok_joints[:n], detok_parents[:n], target_joints[:n], target_parents[:n])
        row["nn_edge_precision"] = np_
        row["nn_edge_recall"] = nr
        row["nn_edge_f1"] = nf1
    except Exception as exc:
        row["detokenize_ok"] = False
        row["error"] = repr(exc)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=Path("rigweave/configs/evoweave_dfs_clean.westlake.env"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seeds", default="20260529,20260530,20260531,20260532")
    parser.add_argument("--random-query", action="store_true")
    parser.add_argument("--frames", type=int, default=None)
    args = parser.parse_args()

    if args.env_file.exists():
        _load_env(args.env_file)
    root = Path(_require_env("EVOWEAVE_ROOT")).expanduser()
    unirig_root = Path(_require_env("EVOWEAVE_UNIRIG_ROOT")).expanduser()
    sys.path.insert(0, str(root / "rigweave" / "src"))
    sys.path.insert(0, str(unirig_root))
    os.environ.setdefault("RIGWEAVE_DISABLE_OPEN3D", "1")
    os.environ.setdefault("RIGWEAVE_DISABLE_LIGHTNING_IMPORT", "1")

    train = _load_train_helpers(root)
    from rigweave.dynamic_rig.data import DynamicRigManifestDataset
    from src.tokenizer.tokenizer_part import discretize

    tokenizer = train.build_tokenizer(Path(_require_env("EVOWEAVE_TOKENIZER_CONFIG")))
    continuous_range = tokenizer.continuous_range
    frames = args.frames if args.frames is not None else int(os.environ.get("RIGWEAVE_FRAMES", "24"))
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    rows = []
    for seed in seeds:
        dataset = DynamicRigManifestDataset(
            args.manifest,
            tokenizer,
            frame_count=frames,
            limit=args.limit,
            random_query=bool(args.random_query),
            seed=seed,
            motion_fps_ratio=float(os.environ.get("RIGWEAVE_MOTION_FPS_RATIO", "0.7")),
            motion_vertex_samples=int(os.environ.get("RIGWEAVE_MOTION_VERTEX_SAMPLES", "512")),
            target_start_policy=os.environ.get("RIGWEAVE_TARGET_START_POLICY", "joint0"),
            target_root_policy=os.environ.get("RIGWEAVE_TARGET_ROOT_POLICY", "legacy"),
        )
        for index in range(len(dataset)):
            try:
                row = audit_one(dataset[index], tokenizer, discretize, continuous_range)
                row["index"] = int(index)
                row["seed"] = int(seed)
            except Exception as exc:
                row = {"index": int(index), "seed": int(seed), "detokenize_ok": False, "error": repr(exc)}
            rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows_path = args.output.with_suffix(".jsonl")
    with rows_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    ok_rows = [r for r in rows if r.get("detokenize_ok")]
    summary = {
        "manifest": str(args.manifest),
        "rows_jsonl": str(rows_path),
        "row_count": len(rows),
        "ok_count": len(ok_rows),
        "seeds": seeds,
        "random_query": bool(args.random_query),
        "limit": int(args.limit),
        "parent_mismatch_rows": int(sum(1 for r in ok_rows if int(r.get("parent_mismatch_count", 0)) > 0)),
        "bad_tie_rows": int(sum(1 for r in ok_rows if int(r.get("branch_parent_bad_tie_count", 0)) > 0)),
        "duplicate_quantized_rows": int(sum(1 for r in ok_rows if int(r.get("duplicate_quantized_coord_groups", 0)) > 0)),
        "quant_parent_child_same_rows": int(sum(1 for r in ok_rows if int(r.get("quant_parent_child_same_edges", 0)) > 0)),
        "id_edge_f1": _stats([float(r["id_edge_f1"]) for r in ok_rows if "id_edge_f1" in r]),
        "nn_edge_f1": _stats([float(r["nn_edge_f1"]) for r in ok_rows if "nn_edge_f1" in r]),
        "parent_mismatch_count": _stats([float(r.get("parent_mismatch_count", 0)) for r in ok_rows]),
        "branch_parent_bad_tie_count": _stats([float(r.get("branch_parent_bad_tie_count", 0)) for r in ok_rows]),
        "duplicate_quantized_coord_groups": _stats([float(r.get("duplicate_quantized_coord_groups", 0)) for r in ok_rows]),
        "worst_id_edge_f1": sorted(
            [
                {
                    "name": r.get("name"),
                    "seed": r.get("seed"),
                    "index": r.get("index"),
                    "id_edge_f1": r.get("id_edge_f1"),
                    "nn_edge_f1": r.get("nn_edge_f1"),
                    "parent_mismatch_count": r.get("parent_mismatch_count"),
                    "branch_parent_bad_tie_count": r.get("branch_parent_bad_tie_count"),
                    "duplicate_quantized_coord_groups": r.get("duplicate_quantized_coord_groups"),
                    "quant_parent_child_same_edges": r.get("quant_parent_child_same_edges"),
                    "branch_parent_tie_examples": r.get("branch_parent_tie_examples", [])[:3],
                    "parent_mismatch_examples": r.get("parent_mismatch_examples", [])[:5],
                }
                for r in ok_rows
            ],
            key=lambda x: (float(x["id_edge_f1"]), -int(x.get("parent_mismatch_count") or 0)),
        )[:20],
    }
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
