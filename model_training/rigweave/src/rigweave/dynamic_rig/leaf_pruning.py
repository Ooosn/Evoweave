"""CPU-only random deletion of leaves from a rooted parent-index tree."""

from __future__ import annotations

from dataclasses import dataclass
import numbers
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LeafPruningResult:
    """A pruned tree plus mappings back to the original source indices."""

    parents: np.ndarray
    kept_source_indices: np.ndarray
    source_to_kept: np.ndarray
    removed_source_indices: np.ndarray


def _is_strict_integer(value: Any) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_))


def _integer_parent_array(parents: Any) -> np.ndarray:
    if isinstance(parents, np.ndarray):
        if parents.ndim != 1:
            raise ValueError(f"parents must be one-dimensional, got shape {parents.shape}")
        if not np.issubdtype(parents.dtype, np.integer) or np.issubdtype(parents.dtype, np.bool_):
            raise ValueError(f"parents must use an integer dtype, got {parents.dtype}")
        raw_values = parents.tolist()
    elif isinstance(parents, (str, bytes)):
        raise ValueError("parents must be a one-dimensional integer sequence")
    else:
        try:
            raw_values = list(parents)
        except TypeError as exc:
            raise ValueError("parents must be a one-dimensional integer sequence") from exc

    if not raw_values:
        raise ValueError("parents must contain at least one node")

    values: list[int] = []
    for index, value in enumerate(raw_values):
        if not _is_strict_integer(value):
            raise ValueError(f"parents[{index}] must be an integer, not {type(value).__name__}")
        values.append(int(value))
    try:
        return np.asarray(values, dtype=np.int64)
    except (OverflowError, ValueError) as exc:
        raise ValueError("parents contains an integer outside the int64 range") from exc


def _validate_tree(parents: np.ndarray) -> tuple[int, np.ndarray]:
    count = int(parents.shape[0])
    if np.any(parents < -1):
        raise ValueError("parents may use only -1 for the root or an in-range parent index")
    if np.any(parents >= count):
        raise ValueError("parents contains an out-of-range parent index")

    roots = np.flatnonzero(parents == -1)
    if roots.size != 1:
        raise ValueError(f"parents must contain exactly one root, found {roots.tolist()}")
    root = int(roots[0])

    child_counts = np.zeros(count, dtype=np.int64)
    for child, parent in enumerate(parents.tolist()):
        parent = int(parent)
        if parent == child:
            raise ValueError(f"node {child} cannot parent itself")
        if parent >= 0:
            child_counts[parent] += 1

    # With exactly one root, valid parent bounds, and no cycle, every node is
    # necessarily connected to that root. Mark each parent chain once so this
    # validation remains linear in the number of nodes.
    state = np.zeros(count, dtype=np.int8)
    for start in range(count):
        if state[start] == 2:
            continue
        path: list[int] = []
        current = start
        while current >= 0 and state[current] == 0:
            state[current] = 1
            path.append(current)
            current = int(parents[current])
        if current >= 0 and state[current] == 1:
            raise ValueError(f"parent graph contains a cycle involving node {current}")
        for node in path:
            state[node] = 2

    return root, child_counts


def prune_random_leaves(
    parents: Any,
    *,
    remove_count: int,
    rng: np.random.Generator,
) -> LeafPruningResult:
    """Uniformly remove current non-root leaves from a rooted tree.

    ``parents`` uses ``-1`` for the single root.  The output parent array is
    compacted in ascending original-source-index order; all other arrays use
    the same source index space unless their name says otherwise.
    """

    source_parents = _integer_parent_array(parents)
    count = int(source_parents.shape[0])
    if not _is_strict_integer(remove_count):
        raise ValueError("remove_count must be an integer, not a boolean or float")
    remove_count = int(remove_count)
    if remove_count < 0 or remove_count > count - 1:
        raise ValueError(f"remove_count must be in [0, {count - 1}], got {remove_count}")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an instance of numpy.random.Generator")

    root, child_counts = _validate_tree(source_parents)

    # This vector contains exactly the currently removable leaves.  Deletions
    # use swap-pop, and a parent is appended only when its final child goes.
    leaves = np.empty(count, dtype=np.int64)
    leaf_count = 0
    for node, child_count in enumerate(child_counts.tolist()):
        if node != root and child_count == 0:
            leaves[leaf_count] = node
            leaf_count += 1

    removed_source_indices = np.empty(remove_count, dtype=np.int64)
    for deletion in range(remove_count):
        leaf_position = int(rng.integers(leaf_count))
        removed = int(leaves[leaf_position])
        removed_source_indices[deletion] = removed

        leaf_count -= 1
        leaves[leaf_position] = leaves[leaf_count]
        parent = int(source_parents[removed])
        child_counts[parent] -= 1
        if parent != root and child_counts[parent] == 0:
            leaves[leaf_count] = parent
            leaf_count += 1

    kept_mask = np.ones(count, dtype=bool)
    kept_mask[removed_source_indices] = False
    kept_source_indices = np.flatnonzero(kept_mask).astype(np.int64, copy=False)
    source_to_kept = np.full(count, -1, dtype=np.int64)
    source_to_kept[kept_source_indices] = np.arange(kept_source_indices.size, dtype=np.int64)

    compact_parents = source_parents[kept_source_indices].copy()
    nonroots = compact_parents >= 0
    compact_parents[nonroots] = source_to_kept[compact_parents[nonroots]]
    return LeafPruningResult(
        parents=compact_parents,
        kept_source_indices=kept_source_indices,
        source_to_kept=source_to_kept,
        removed_source_indices=removed_source_indices,
    )


__all__ = ["LeafPruningResult", "prune_random_leaves"]
