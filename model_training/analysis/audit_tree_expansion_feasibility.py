#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_path = row.get("path")
            if not sample_path:
                raise ValueError(f"{path}:{line_number} has no path")
            rows.append(
                {
                    "path": str(sample_path),
                    "split": str(row.get("split", path.stem.replace("_manifest", ""))),
                    "dataset_source": str(row.get("dataset_source", "unknown")),
                }
            )
    return rows


def _tree_depths(parents: np.ndarray) -> np.ndarray:
    depths = np.full(parents.shape[0], -1, dtype=np.int64)

    def visit(index: int, stack: set[int]) -> int:
        if depths[index] >= 0:
            return int(depths[index])
        if index in stack:
            raise ValueError(f"cycle at joint {index}")
        parent = int(parents[index])
        if parent < 0:
            depth = 0
        else:
            if parent >= parents.shape[0]:
                raise ValueError(f"parent {parent} out of range for joint {index}")
            depth = visit(parent, stack | {index}) + 1
        depths[index] = depth
        return depth

    for joint_index in range(parents.shape[0]):
        visit(joint_index, set())
    return depths


def _stack_close_audit(parents: np.ndarray) -> dict[str, Any]:
    """Check whether the stored node order is exactly representable by DFS CLOSE tokens."""

    if parents.shape[0] <= 0 or int(parents[0]) >= 0:
        return {
            "representable": False,
            "first_invalid_joint": 0,
            "close_count": 0,
            "max_consecutive_closes": 0,
            "flat_token_count": 0,
            "stack_token_count": 0,
        }

    stack = [0]
    close_count = 0
    max_consecutive_closes = 0
    branch_jump_count = 0
    for joint_index in range(1, int(parents.shape[0])):
        parent_index = int(parents[joint_index])
        if parent_index not in stack:
            return {
                "representable": False,
                "first_invalid_joint": int(joint_index),
                "close_count": int(close_count),
                "max_consecutive_closes": int(max_consecutive_closes),
                "flat_token_count": 0,
                "stack_token_count": 0,
            }
        parent_offset = stack.index(parent_index)
        closes = len(stack) - parent_offset - 1
        close_count += closes
        max_consecutive_closes = max(max_consecutive_closes, closes)
        del stack[parent_offset + 1 :]
        if parent_index != joint_index - 1:
            branch_jump_count += 1
        if stack[-1] != parent_index:
            raise AssertionError("stack parent recovery failed")
        stack.append(joint_index)

    trailing_closes = len(stack)
    close_count += trailing_closes
    max_consecutive_closes = max(max_consecutive_closes, trailing_closes)
    joint_count = int(parents.shape[0])
    if close_count != joint_count:
        raise AssertionError(
            f"expected one CLOSE per node, got close_count={close_count} joints={joint_count}"
        )

    # BOS + class + coordinate triples + branch jumps + EOS.
    flat_token_count = 3 + 3 * joint_count + 4 * branch_jump_count
    # BOS + class + coordinate triples + one CLOSE per node + EOS.
    stack_token_count = 3 + 4 * joint_count
    return {
        "representable": True,
        "first_invalid_joint": None,
        "close_count": int(close_count),
        "max_consecutive_closes": int(max_consecutive_closes),
        "branch_jump_count": int(branch_jump_count),
        "flat_token_count": int(flat_token_count),
        "stack_token_count": int(stack_token_count),
        "stack_to_flat_token_ratio": float(stack_token_count / flat_token_count),
    }


def _analyze_row(row: dict[str, Any]) -> dict[str, Any]:
    path = Path(row["path"])
    with np.load(path, allow_pickle=True) as raw:
        parents = np.asarray(raw["target_parents"], dtype=np.int64).reshape(-1)
        has_skin = np.asarray(raw["target_has_skin"], dtype=bool).reshape(-1)
        is_connector = np.asarray(raw["target_is_connector"], dtype=bool).reshape(-1)
        is_tail_or_end = np.asarray(raw["target_is_tail_or_end"], dtype=bool).reshape(-1)
        joints = np.asarray(raw["target_joints_rootspace"], dtype=np.float32)
        vertices = np.asarray(raw["frame_vertices_rootspace"], dtype=np.float32)
        skin_weights = np.asarray(raw["target_skin_weights"], dtype=np.float32)

    joint_count = int(parents.shape[0])
    if joint_count <= 0:
        raise ValueError(f"{path} has no target joints")
    expected_vector_shape = (joint_count,)
    for name, value in (
        ("target_has_skin", has_skin),
        ("target_is_connector", is_connector),
        ("target_is_tail_or_end", is_tail_or_end),
    ):
        if value.shape != expected_vector_shape:
            raise ValueError(f"{path} {name} shape={value.shape}, expected {expected_vector_shape}")
    if joints.ndim != 3 or joints.shape[1:] != (joint_count, 3):
        raise ValueError(f"{path} target_joints_rootspace shape={joints.shape}")
    if vertices.ndim != 3 or vertices.shape[0] != joints.shape[0] or vertices.shape[-1] != 3:
        raise ValueError(f"{path} frame_vertices_rootspace shape={vertices.shape}")
    if skin_weights.shape != (vertices.shape[1], joint_count):
        raise ValueError(f"{path} target_skin_weights shape={skin_weights.shape}")

    roots = np.flatnonzero(parents < 0)
    if roots.tolist() != [0]:
        raise ValueError(f"{path} roots={roots.tolist()}, expected root joint 0")

    children: list[list[int]] = [[] for _ in range(joint_count)]
    for child_index, parent_index in enumerate(parents.tolist()):
        if parent_index >= 0:
            if parent_index >= child_index:
                raise ValueError(
                    f"{path} parent order is not topological: joint={child_index} parent={parent_index}"
                )
            children[parent_index].append(child_index)
    child_counts = np.asarray([len(value) for value in children], dtype=np.int64)
    depths = _tree_depths(parents)
    stack_close = _stack_close_audit(parents)

    query_vertices = vertices[0]
    mesh_lo = query_vertices.min(axis=0)
    mesh_hi = query_vertices.max(axis=0)
    mesh_bbox_diag = float(np.linalg.norm(mesh_hi - mesh_lo))
    if not np.isfinite(mesh_bbox_diag) or mesh_bbox_diag <= 1.0e-8:
        raise ValueError(f"{path} has degenerate query mesh bbox")

    query_joints = joints[0]
    edge_lengths: list[float] = []
    edge_records: list[tuple[float, float, int, int]] = []
    for child_index, parent_index in enumerate(parents.tolist()):
        if parent_index >= 0:
            frame_lengths = np.linalg.norm(
                joints[:, child_index] - joints[:, parent_index],
                axis=-1,
            )
            length = float(frame_lengths[0])
            normalized_length = length / mesh_bbox_diag
            edge_lengths.append(normalized_length)
            edge_records.append(
                (
                    normalized_length,
                    float(frame_lengths.max()) / mesh_bbox_diag,
                    child_index,
                    parent_index,
                )
            )

    weight_sums = skin_weights.sum(axis=0, dtype=np.float64)
    positive_weight_vertices = (skin_weights > 1.0e-4).sum(axis=0)
    hard_assignment = np.argmax(skin_weights, axis=1)
    hard_assignment_counts = np.bincount(hard_assignment, minlength=joint_count)
    no_hard_assignment = has_skin & (hard_assignment_counts == 0)

    vertex_motion = np.linalg.norm(vertices - vertices[:1], axis=-1).max(axis=0)
    weighted_motion = np.zeros(joint_count, dtype=np.float64)
    valid_weight = weight_sums > 1.0e-8
    if valid_weight.any():
        weighted_motion[valid_weight] = (
            skin_weights[:, valid_weight].T @ vertex_motion.astype(np.float64)
        ) / weight_sums[valid_weight]
    weighted_motion /= mesh_bbox_diag

    joint_motion = np.linalg.norm(joints - joints[:1], axis=-1).max(axis=0) / mesh_bbox_diag

    connector_indices = np.flatnonzero(is_connector)
    connector_child_counts = child_counts[connector_indices]
    connector_without_skin = is_connector & ~has_skin
    connector_without_skin_indices = np.flatnonzero(connector_without_skin)
    connector_descendant_skin = []
    descendant_cache: dict[int, bool] = {}

    def has_skinned_descendant(index: int) -> bool:
        if index in descendant_cache:
            return descendant_cache[index]
        value = any(
            bool(has_skin[child]) or has_skinned_descendant(child)
            for child in children[index]
        )
        descendant_cache[index] = value
        return value

    for connector_index in connector_without_skin_indices.tolist():
        connector_descendant_skin.append(has_skinned_descendant(connector_index))

    depth_hist = np.bincount(depths, minlength=int(depths.max()) + 1)
    degree_hist = np.bincount(child_counts, minlength=int(child_counts.max()) + 1)
    topology_signature = ",".join(str(int(value)) for value in parents.tolist())
    max_children_parent_index = int(np.argmax(child_counts))
    near_zero_edges = [record for record in edge_records if record[0] <= 1.0e-6]
    exact_zero_edges = [record for record in edge_records if record[0] <= 1.0e-12]
    persistent_zero_edges = [
        record for record in exact_zero_edges if record[1] <= 1.0e-6
    ]
    high_degree_nodes = np.flatnonzero(child_counts > 8)

    return {
        "path": str(path),
        "split": row["split"],
        "dataset_source": row["dataset_source"],
        "joint_count": joint_count,
        "edge_count": joint_count - 1,
        "leaf_count": int((child_counts == 0).sum()),
        "branch_node_count": int((child_counts > 1).sum()),
        "root_child_count": int(child_counts[0]),
        "max_children": int(child_counts.max()),
        "max_depth": int(depths.max()),
        "depth_hist": depth_hist.tolist(),
        "degree_hist": degree_hist.tolist(),
        "child_counts": child_counts.tolist(),
        "connector_count": int(is_connector.sum()),
        "connector_without_skin_count": int(connector_without_skin.sum()),
        "connector_chain_count": int(((connector_without_skin) & (child_counts == 1)).sum()),
        "connector_branch_count": int(((connector_without_skin) & (child_counts > 1)).sum()),
        "connector_leaf_count": int(((connector_without_skin) & (child_counts == 0)).sum()),
        "connector_without_skinned_descendant_count": int(
            sum(not value for value in connector_descendant_skin)
        ),
        "tail_or_end_count": int(is_tail_or_end.sum()),
        "skinned_joint_count": int(has_skin.sum()),
        "unskinned_joint_count": int((~has_skin).sum()),
        "skinned_no_hard_assignment_count": int(no_hard_assignment.sum()),
        "skinned_positive_vertex_counts": positive_weight_vertices[has_skin].astype(int).tolist(),
        "skinned_hard_assignment_counts": hard_assignment_counts[has_skin].astype(int).tolist(),
        "skinned_weighted_motion": weighted_motion[has_skin].astype(float).tolist(),
        "joint_motion": joint_motion.astype(float).tolist(),
        "edge_lengths_bbox": edge_lengths,
        "near_zero_edge_count": len(near_zero_edges),
        "exact_zero_edge_count": len(exact_zero_edges),
        "exact_zero_edge_child_connector_count": int(
            sum(
                bool(is_connector[child])
                for _length, _max_length, child, _parent in exact_zero_edges
            )
        ),
        "exact_zero_edge_parent_connector_count": int(
            sum(
                bool(is_connector[parent])
                for _length, _max_length, _child, parent in exact_zero_edges
            )
        ),
        "exact_zero_edge_both_skinned_count": int(
            sum(
                bool(has_skin[child] and has_skin[parent])
                for _length, _max_length, child, parent in exact_zero_edges
            )
        ),
        "exact_zero_edge_child_tail_or_end_count": int(
            sum(
                bool(is_tail_or_end[child])
                for _length, _max_length, child, _parent in exact_zero_edges
            )
        ),
        "persistent_zero_edge_count": len(persistent_zero_edges),
        "persistent_zero_edge_child_connector_count": int(
            sum(
                bool(is_connector[child])
                for _length, _max_length, child, _parent in persistent_zero_edges
            )
        ),
        "persistent_zero_edge_parent_connector_count": int(
            sum(
                bool(is_connector[parent])
                for _length, _max_length, _child, parent in persistent_zero_edges
            )
        ),
        "persistent_zero_edge_both_skinned_count": int(
            sum(
                bool(has_skin[child] and has_skin[parent])
                for _length, _max_length, child, parent in persistent_zero_edges
            )
        ),
        "near_zero_edge_child_connector_count": int(
            sum(
                bool(is_connector[child])
                for _length, _max_length, child, _parent in near_zero_edges
            )
        ),
        "near_zero_edge_parent_connector_count": int(
            sum(
                bool(is_connector[parent])
                for _length, _max_length, _child, parent in near_zero_edges
            )
        ),
        "near_zero_edge_both_skinned_count": int(
            sum(
                bool(has_skin[child] and has_skin[parent])
                for _length, _max_length, child, parent in near_zero_edges
            )
        ),
        "near_zero_edge_child_tail_or_end_count": int(
            sum(
                bool(is_tail_or_end[child])
                for _length, _max_length, child, _parent in near_zero_edges
            )
        ),
        "max_children_parent": {
            "index": max_children_parent_index,
            "is_root": bool(parents[max_children_parent_index] < 0),
            "has_skin": bool(has_skin[max_children_parent_index]),
            "is_connector": bool(is_connector[max_children_parent_index]),
            "is_tail_or_end": bool(is_tail_or_end[max_children_parent_index]),
            "child_count": int(child_counts[max_children_parent_index]),
            "skinned_child_count": int(
                sum(bool(has_skin[child]) for child in children[max_children_parent_index])
            ),
            "connector_child_count": int(
                sum(bool(is_connector[child]) for child in children[max_children_parent_index])
            ),
        },
        "high_degree_gt8_node_count": int(high_degree_nodes.size),
        "high_degree_gt8_root_count": int(np.sum(parents[high_degree_nodes] < 0)),
        "high_degree_gt8_connector_count": int(is_connector[high_degree_nodes].sum()),
        "high_degree_gt8_skinned_count": int(has_skin[high_degree_nodes].sum()),
        "topology_signature": topology_signature,
        "stack_close": stack_close,
    }


def _quantiles(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "min": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def _counter(values: Iterable[int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(values).items())}


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    topology_counts = Counter(row["topology_signature"] for row in rows)
    child_counts = [value for row in rows for value in row["child_counts"]]
    non_leaf_child_counts = [value for value in child_counts if value > 0]
    edge_lengths = [value for row in rows for value in row["edge_lengths_bbox"]]
    positive_vertex_counts = [
        value for row in rows for value in row["skinned_positive_vertex_counts"]
    ]
    hard_assignment_counts = [
        value for row in rows for value in row["skinned_hard_assignment_counts"]
    ]
    weighted_motion = [
        value for row in rows for value in row["skinned_weighted_motion"]
    ]
    joint_motion = [value for row in rows for value in row["joint_motion"]]

    connector_total = sum(row["connector_count"] for row in rows)
    connector_without_skin_total = sum(row["connector_without_skin_count"] for row in rows)
    skinned_total = sum(row["skinned_joint_count"] for row in rows)
    edge_total = sum(row["edge_count"] for row in rows)
    node_total = sum(row["joint_count"] for row in rows)
    sample_count = len(rows)
    stack_rows = [row["stack_close"] for row in rows]
    stack_representable = [
        row for row in stack_rows if bool(row["representable"])
    ]

    def sample_rate(predicate: Any) -> float:
        return float(sum(bool(predicate(row)) for row in rows) / max(sample_count, 1))

    return {
        "sample_count": sample_count,
        "node_count": int(node_total),
        "edge_count": int(edge_total),
        "unique_ordered_topology_count": int(len(topology_counts)),
        "singleton_topology_count": int(sum(value == 1 for value in topology_counts.values())),
        "largest_topology_frequency": int(max(topology_counts.values(), default=0)),
        "joint_count": _quantiles(row["joint_count"] for row in rows),
        "max_depth": _quantiles(row["max_depth"] for row in rows),
        "max_children_per_sample": _quantiles(row["max_children"] for row in rows),
        "root_child_count": _quantiles(row["root_child_count"] for row in rows),
        "leaf_count": _quantiles(row["leaf_count"] for row in rows),
        "branch_node_count": _quantiles(row["branch_node_count"] for row in rows),
        "all_node_child_count_hist": _counter(child_counts),
        "non_leaf_child_count_hist": _counter(non_leaf_child_counts),
        "edge_length_bbox": _quantiles(edge_lengths),
        "near_zero_edge_count": int(sum(row["near_zero_edge_count"] for row in rows)),
        "exact_zero_edge_count": int(sum(row["exact_zero_edge_count"] for row in rows)),
        "exact_zero_edges": {
            "count": int(sum(row["exact_zero_edge_count"] for row in rows)),
            "child_connector_count": int(
                sum(row["exact_zero_edge_child_connector_count"] for row in rows)
            ),
            "parent_connector_count": int(
                sum(row["exact_zero_edge_parent_connector_count"] for row in rows)
            ),
            "both_skinned_count": int(
                sum(row["exact_zero_edge_both_skinned_count"] for row in rows)
            ),
            "child_tail_or_end_count": int(
                sum(row["exact_zero_edge_child_tail_or_end_count"] for row in rows)
            ),
            "persistent_across_frames_count": int(
                sum(row["persistent_zero_edge_count"] for row in rows)
            ),
            "ever_separates_count": int(
                sum(
                    row["exact_zero_edge_count"]
                    - row["persistent_zero_edge_count"]
                    for row in rows
                )
            ),
            "persistent_child_connector_count": int(
                sum(row["persistent_zero_edge_child_connector_count"] for row in rows)
            ),
            "persistent_parent_connector_count": int(
                sum(row["persistent_zero_edge_parent_connector_count"] for row in rows)
            ),
            "persistent_both_skinned_count": int(
                sum(row["persistent_zero_edge_both_skinned_count"] for row in rows)
            ),
        },
        "near_zero_edges": {
            "count": int(sum(row["near_zero_edge_count"] for row in rows)),
            "child_connector_count": int(
                sum(row["near_zero_edge_child_connector_count"] for row in rows)
            ),
            "parent_connector_count": int(
                sum(row["near_zero_edge_parent_connector_count"] for row in rows)
            ),
            "both_skinned_count": int(
                sum(row["near_zero_edge_both_skinned_count"] for row in rows)
            ),
            "child_tail_or_end_count": int(
                sum(row["near_zero_edge_child_tail_or_end_count"] for row in rows)
            ),
        },
        "tail_or_end": {
            "node_count": int(sum(row["tail_or_end_count"] for row in rows)),
            "sample_fraction": sample_rate(lambda row: row["tail_or_end_count"] > 0),
        },
        "high_degree_gt8": {
            "node_count": int(sum(row["high_degree_gt8_node_count"] for row in rows)),
            "root_count": int(sum(row["high_degree_gt8_root_count"] for row in rows)),
            "connector_count": int(
                sum(row["high_degree_gt8_connector_count"] for row in rows)
            ),
            "skinned_count": int(sum(row["high_degree_gt8_skinned_count"] for row in rows)),
            "sample_fraction": sample_rate(lambda row: row["high_degree_gt8_node_count"] > 0),
            "max_parent_is_root_fraction": sample_rate(
                lambda row: row["max_children_parent"]["is_root"]
            ),
        },
        "connector": {
            "node_count": int(connector_total),
            "node_fraction": float(connector_total / max(node_total, 1)),
            "sample_fraction": sample_rate(lambda row: row["connector_count"] > 0),
            "without_skin_count": int(connector_without_skin_total),
            "without_skin_fraction": float(connector_without_skin_total / max(connector_total, 1)),
            "chain_count": int(sum(row["connector_chain_count"] for row in rows)),
            "branch_count": int(sum(row["connector_branch_count"] for row in rows)),
            "leaf_count": int(sum(row["connector_leaf_count"] for row in rows)),
            "without_skinned_descendant_count": int(
                sum(row["connector_without_skinned_descendant_count"] for row in rows)
            ),
        },
        "skin_evidence": {
            "skinned_joint_count": int(skinned_total),
            "positive_weight_vertex_count": _quantiles(positive_vertex_counts),
            "hard_assignment_vertex_count": _quantiles(hard_assignment_counts),
            "skinned_joint_without_hard_assignment_count": int(
                sum(row["skinned_no_hard_assignment_count"] for row in rows)
            ),
            "skinned_joint_without_hard_assignment_fraction": float(
                sum(row["skinned_no_hard_assignment_count"] for row in rows)
                / max(skinned_total, 1)
            ),
            "weighted_vertex_motion_bbox": _quantiles(weighted_motion),
            "weighted_motion_below_1e-3_fraction": float(
                sum(value < 1.0e-3 for value in weighted_motion) / max(len(weighted_motion), 1)
            ),
            "weighted_motion_below_1e-2_fraction": float(
                sum(value < 1.0e-2 for value in weighted_motion) / max(len(weighted_motion), 1)
            ),
        },
        "joint_motion_bbox": _quantiles(joint_motion),
        "stack_close_representation": {
            "representable_count": int(len(stack_representable)),
            "representable_fraction": float(
                len(stack_representable) / max(sample_count, 1)
            ),
            "max_consecutive_closes": _quantiles(
                row["max_consecutive_closes"] for row in stack_representable
            ),
            "flat_token_count": _quantiles(
                row["flat_token_count"] for row in stack_representable
            ),
            "stack_token_count": _quantiles(
                row["stack_token_count"] for row in stack_representable
            ),
            "stack_to_flat_token_ratio": _quantiles(
                row["stack_to_flat_token_ratio"] for row in stack_representable
            ),
            "flat_total_tokens": int(
                sum(row["flat_token_count"] for row in stack_representable)
            ),
            "stack_total_tokens": int(
                sum(row["stack_token_count"] for row in stack_representable)
            ),
            "invalid_examples": [
                {
                    "path": rows[index]["path"],
                    "first_invalid_joint": value["first_invalid_joint"],
                }
                for index, value in enumerate(stack_rows)
                if not bool(value["representable"])
            ][:20],
        },
        "samples": {
            "with_connector_fraction": sample_rate(lambda row: row["connector_count"] > 0),
            "with_unskinned_connector_fraction": sample_rate(
                lambda row: row["connector_without_skin_count"] > 0
            ),
            "with_max_children_gt_4_fraction": sample_rate(lambda row: row["max_children"] > 4),
            "max_children_gt_fraction": {
                str(threshold): sample_rate(
                    lambda row, threshold=threshold: row["max_children"] > threshold
                )
                for threshold in (1, 2, 4, 5, 8, 16, 32, 64)
            },
            "with_near_zero_edge_fraction": sample_rate(lambda row: row["near_zero_edge_count"] > 0),
            "with_skinned_joint_without_hard_assignment_fraction": sample_rate(
                lambda row: row["skinned_no_hard_assignment_count"] > 0
            ),
        },
        "largest_topologies": [
            {"signature": signature, "count": int(count)}
            for signature, count in topology_counts.most_common(20)
        ],
        "extreme_rows": {
            "max_children": sorted(
                (
                    {
                        "path": row["path"],
                        "joint_count": row["joint_count"],
                        "max_children": row["max_children"],
                        "max_children_parent": row["max_children_parent"],
                    }
                    for row in rows
                ),
                key=lambda value: (value["max_children"], value["joint_count"]),
                reverse=True,
            )[:20],
            "max_depth": sorted(
                (
                    {
                        "path": row["path"],
                        "joint_count": row["joint_count"],
                        "max_depth": row["max_depth"],
                    }
                    for row in rows
                ),
                key=lambda value: (value["max_depth"], value["joint_count"]),
                reverse=True,
            )[:20],
            "connectors": sorted(
                (
                    {
                        "path": row["path"],
                        "joint_count": row["joint_count"],
                        "connector_count": row["connector_count"],
                    }
                    for row in rows
                ),
                key=lambda value: (value["connector_count"], value["joint_count"]),
                reverse=True,
            )[:20],
            "near_zero_edges": sorted(
                (
                    {
                        "path": row["path"],
                        "joint_count": row["joint_count"],
                        "near_zero_edge_count": row["near_zero_edge_count"],
                        "exact_zero_edge_count": row["exact_zero_edge_count"],
                        "persistent_zero_edge_count": row[
                            "persistent_zero_edge_count"
                        ],
                        "child_connector_count": row[
                            "near_zero_edge_child_connector_count"
                        ],
                        "parent_connector_count": row[
                            "near_zero_edge_parent_connector_count"
                        ],
                        "both_skinned_count": row[
                            "near_zero_edge_both_skinned_count"
                        ],
                    }
                    for row in rows
                ),
                key=lambda value: (
                    value["exact_zero_edge_count"],
                    value["near_zero_edge_count"],
                    value["joint_count"],
                ),
                reverse=True,
            )[:20],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether the finalized rootless trees support parent-wise child-set expansion."
    )
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_rows: list[dict[str, Any]] = []
    for manifest in args.manifest:
        manifest_rows.extend(_read_manifest(manifest))
    if not manifest_rows:
        raise ValueError("no manifest rows")

    with ThreadPoolExecutor(max_workers=max(int(args.workers), 1)) as executor:
        rows = list(executor.map(_analyze_row, manifest_rows))

    by_split = {
        split: _aggregate([row for row in rows if row["split"] == split])
        for split in sorted({row["split"] for row in rows})
    }
    report = {
        "manifests": [str(path) for path in args.manifest],
        "row_count": len(rows),
        "aggregate": _aggregate(rows),
        "by_split": by_split,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
