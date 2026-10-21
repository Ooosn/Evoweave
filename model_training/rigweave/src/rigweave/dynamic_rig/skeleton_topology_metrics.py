"""Original nearest-joint topology metric, shared without training-runtime imports."""

from __future__ import annotations

import numpy as np


def _edge_set(parents: list[int | None]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for child, parent in enumerate(parents):
        if parent is None or parent < 0 or parent >= len(parents) or child == parent:
            continue
        edges.add((int(parent), int(child)))
    return edges


def _edge_overlap(
    source_joints: np.ndarray,
    source_parents: list[int | None],
    target_joints: np.ndarray,
    target_parents: list[int | None],
) -> dict[str, float | int]:
    source_edges = _edge_set(source_parents)
    target_edges = _edge_set(target_parents)
    if not source_edges:
        return {"edge_count": 0, "matched_edges": 0, "score": 1.0 if not target_edges else 0.0}
    if target_joints.size == 0:
        return {"edge_count": len(source_edges), "matched_edges": 0, "score": 0.0}
    d = np.linalg.norm(source_joints[:, None, :] - target_joints[None, :, :], axis=-1)
    mapping = d.argmin(axis=1)
    matched = 0
    for parent, child in source_edges:
        mapped = (int(mapping[parent]), int(mapping[child]))
        if mapped in target_edges:
            matched += 1
    return {
        "edge_count": len(source_edges),
        "matched_edges": int(matched),
        "score": float(matched / max(len(source_edges), 1)),
    }


def _topology_metrics(
    pred_joints: np.ndarray,
    pred_parents: list[int | None],
    target_joints: np.ndarray,
    target_parents: list[int | None],
) -> dict[str, float | int]:
    precision = _edge_overlap(pred_joints, pred_parents, target_joints, target_parents)
    recall = _edge_overlap(target_joints, target_parents, pred_joints, pred_parents)
    p = float(precision["score"])
    r = float(recall["score"])
    f1 = 0.0 if p + r <= 1.0e-12 else 2.0 * p * r / (p + r)
    return {
        "edge_precision": p,
        "edge_recall": r,
        "edge_f1": float(f1),
        "pred_edge_count": int(precision["edge_count"]),
        "target_edge_count": int(recall["edge_count"]),
        "pred_matched_edges": int(precision["matched_edges"]),
        "target_matched_edges": int(recall["matched_edges"]),
    }
