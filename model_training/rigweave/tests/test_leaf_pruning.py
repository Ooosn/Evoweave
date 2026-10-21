from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rigweave.dynamic_rig.leaf_pruning import LeafPruningResult, prune_random_leaves


def _assert_mapping(source: np.ndarray, result: LeafPruningResult) -> None:
    count = int(source.size)
    assert result.parents.dtype == np.int64
    assert result.kept_source_indices.dtype == np.int64
    assert result.source_to_kept.dtype == np.int64
    assert result.removed_source_indices.dtype == np.int64
    assert result.kept_source_indices.shape == (count - result.removed_source_indices.size,)
    assert result.source_to_kept.shape == (count,)
    assert np.array_equal(
        np.sort(np.concatenate((result.kept_source_indices, result.removed_source_indices))),
        np.arange(count, dtype=np.int64),
    )
    expected_mapping = np.full(count, -1, dtype=np.int64)
    expected_mapping[result.kept_source_indices] = np.arange(
        result.kept_source_indices.size, dtype=np.int64
    )
    np.testing.assert_array_equal(result.source_to_kept, expected_mapping)
    for compact_index, source_index in enumerate(result.kept_source_indices.tolist()):
        source_parent = int(source[source_index])
        expected_parent = -1 if source_parent < 0 else int(expected_mapping[source_parent])
        assert int(result.parents[compact_index]) == expected_parent


def _assert_deletions_are_current_leaves(source: np.ndarray, result: LeafPruningResult) -> None:
    active = set(range(int(source.size)))
    root = int(np.flatnonzero(source == -1)[0])
    for removed in result.removed_source_indices.tolist():
        assert removed in active
        assert removed != root
        assert not any(child in active and int(source[child]) == removed for child in active)
        active.remove(removed)
    assert active == set(result.kept_source_indices.tolist())


def _random_tree(rng: np.random.Generator, count: int) -> np.ndarray:
    order = rng.permutation(count)
    parents = np.full(count, -1, dtype=np.int64)
    for position in range(1, count):
        child = int(order[position])
        parents[child] = int(order[int(rng.integers(position))])
    return parents


def test_count_zero_and_single_root_are_identity_operations() -> None:
    source = np.asarray([2, 2, -1, 0], dtype=np.int64)
    result = prune_random_leaves(source, remove_count=0, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(result.parents, source)
    np.testing.assert_array_equal(result.kept_source_indices, np.arange(4, dtype=np.int64))
    np.testing.assert_array_equal(result.source_to_kept, np.arange(4, dtype=np.int64))
    assert result.removed_source_indices.shape == (0,)
    _assert_mapping(source, result)

    root_only = np.asarray([-1], dtype=np.int64)
    result = prune_random_leaves(root_only, remove_count=0, rng=np.random.default_rng(2))
    _assert_mapping(root_only, result)
    np.testing.assert_array_equal(result.parents, root_only)


def test_chain_only_has_one_current_leaf_and_exposes_ancestors() -> None:
    source = np.asarray([-1, 0, 1, 2, 3], dtype=np.int64)
    result = prune_random_leaves(source, remove_count=3, rng=np.random.default_rng(3))
    np.testing.assert_array_equal(result.removed_source_indices, np.asarray([4, 3, 2], dtype=np.int64))
    np.testing.assert_array_equal(result.kept_source_indices, np.asarray([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(result.parents, np.asarray([-1, 0], dtype=np.int64))
    _assert_mapping(source, result)


def test_star_keeps_root_and_removes_only_original_leaves() -> None:
    source = np.asarray([-1, 0, 0, 0, 0], dtype=np.int64)
    result = prune_random_leaves(source, remove_count=3, rng=np.random.default_rng(4))
    assert 0 not in result.removed_source_indices
    assert result.kept_source_indices[0] == 0
    assert result.parents[0] == -1
    assert np.all(result.parents[1:] == 0)
    _assert_mapping(source, result)


def test_branched_pruning_appends_newly_exposed_leaf() -> None:
    source = np.asarray([-1, 0, 0, 1, 1, 2, 2], dtype=np.int64)
    found: LeafPruningResult | None = None
    for seed in range(128):
        candidate = prune_random_leaves(source, remove_count=3, rng=np.random.default_rng(seed))
        if 1 in candidate.removed_source_indices:
            found = candidate
            break

    assert found is not None
    removed = found.removed_source_indices.tolist()
    exposed_position = removed.index(1)
    assert set(removed[:exposed_position]) == {3, 4}
    np.testing.assert_array_equal(found.kept_source_indices, np.asarray([0, 2, 5, 6], dtype=np.int64))
    np.testing.assert_array_equal(found.parents, np.asarray([-1, 0, 1, 1], dtype=np.int64))
    _assert_mapping(source, found)


def test_one_deletion_samples_all_current_leaves() -> None:
    source = np.asarray([-1, 0, 0, 1, 1, 2, 2], dtype=np.int64)
    counts = {leaf: 0 for leaf in (3, 4, 5, 6)}
    rng = np.random.default_rng(20261022)
    for _ in range(2048):
        result = prune_random_leaves(source, remove_count=1, rng=rng)
        counts[int(result.removed_source_indices[0])] += 1
    assert set(counts) == {3, 4, 5, 6}
    assert min(counts.values()) > 200


def test_same_seed_reproduces_all_source_identity_arrays() -> None:
    source = np.asarray([-1, 0, 0, 1, 1, 2, 2, 5], dtype=np.int64)
    first = prune_random_leaves(source, remove_count=4, rng=np.random.default_rng(17))
    second = prune_random_leaves(source, remove_count=4, rng=np.random.default_rng(17))
    for field in ("parents", "kept_source_indices", "source_to_kept", "removed_source_indices"):
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))


def test_random_small_trees_preserve_leaf_order_and_mapping_invariants() -> None:
    tree_rng = np.random.default_rng(91)
    for _ in range(100):
        count = int(tree_rng.integers(1, 65))
        source = _random_tree(tree_rng, count)
        remove_count = int(tree_rng.integers(count))
        result = prune_random_leaves(
            source,
            remove_count=remove_count,
            rng=np.random.default_rng(int(tree_rng.integers(1 << 32))),
        )
        _assert_mapping(source, result)
        _assert_deletions_are_current_leaves(source, result)


@pytest.mark.parametrize(
    "parents",
    (
        np.asarray([], dtype=np.int64),
        np.asarray([[-1, 0]], dtype=np.int64),
        np.asarray([-1.0, 0.0]),
        np.asarray([True, False]),
        np.asarray(["-1", "0"]),
        np.asarray([-1, 0], dtype=object),
        [-1, 0.5],
    ),
)
def test_invalid_parent_shapes_and_types_are_rejected(parents: object) -> None:
    with pytest.raises(ValueError):
        prune_random_leaves(parents, remove_count=0, rng=np.random.default_rng(0))


@pytest.mark.parametrize(
    "parents",
    (
        np.asarray([-1, -1], dtype=np.int64),
        np.asarray([-1, 2, 1], dtype=np.int64),
        np.asarray([-1, 0, 3], dtype=np.int64),
        np.asarray([-1, -2], dtype=np.int64),
        np.asarray([-1, 0, 1, 3], dtype=np.int64),
    ),
)
def test_invalid_parent_graphs_are_rejected(parents: np.ndarray) -> None:
    with pytest.raises(ValueError):
        prune_random_leaves(parents, remove_count=0, rng=np.random.default_rng(0))


@pytest.mark.parametrize("remove_count", (True, np.bool_(False), -1, 3, 1.0, "0"))
def test_invalid_remove_counts_are_rejected(remove_count: object) -> None:
    with pytest.raises(ValueError):
        prune_random_leaves(np.asarray([-1, 0, 0], dtype=np.int64), remove_count=remove_count, rng=np.random.default_rng(0))


def test_rng_must_be_numpy_generator() -> None:
    with pytest.raises(TypeError):
        prune_random_leaves(
            np.asarray([-1, 0], dtype=np.int64),
            remove_count=1,
            rng=np.random.RandomState(0),
        )
