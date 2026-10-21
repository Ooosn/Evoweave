from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rigweave.dynamic_rig.aux_skeleton_tokens import AuxSkeletonTokens
from rigweave.dynamic_rig.leaf_pruning import prune_random_leaves
from rigweave.dynamic_rig.structure_xyz_fill_data import (
    build_fill_inputs,
    build_structure_fill_targets,
    decode_coordinate_tokens,
    quantize_coordinates,
)


def _codec() -> AuxSkeletonTokens:
    return AuxSkeletonTokens(max_joints=32, max_sequence_tokens=256)


def _targets(parent_rows: list[np.ndarray], *, frame_count: int = 2) -> dict[str, np.ndarray]:
    max_joints = max(int(row.size) for row in parent_rows)
    xyz = np.zeros((len(parent_rows), frame_count, max_joints, 3), dtype=np.float32)
    parents = np.full((len(parent_rows), max_joints), -2, dtype=np.int64)
    joint_mask = np.zeros((len(parent_rows), max_joints), dtype=bool)
    for asset, row in enumerate(parent_rows):
        count = int(row.size)
        parents[asset, :count] = row
        joint_mask[asset, :count] = True
        for frame in range(frame_count):
            for joint in range(count):
                xyz[asset, frame, joint] = (
                    -0.8 + 0.1 * joint,
                    -0.6 + 0.15 * frame,
                    -0.4 + 0.05 * (joint + frame),
                )
    return {
        "frame_target_joints": xyz,
        "target_parents": parents,
        "joint_mask": joint_mask,
        "frame_mask": np.ones((len(parent_rows), frame_count), dtype=bool),
    }


def test_quantizer_matches_official_bankers_rounding_and_half_bin_decode() -> None:
    values = np.asarray([-1.0, -0.99609375, 0.0, 0.00390625, 1.0], dtype=np.float32)
    np.testing.assert_array_equal(
        quantize_coordinates(values),
        np.asarray([0, 0, 128, 128, 255], dtype=np.int64),
    )
    assert quantize_coordinates(values).dtype == np.int64
    np.testing.assert_array_equal(
        decode_coordinate_tokens(np.asarray([0, 255], dtype=np.int64)),
        np.asarray([-0.99609375, 0.99609375], dtype=np.float32),
    )


@pytest.mark.parametrize(
    "values",
    (
        np.asarray([np.nan], dtype=np.float32),
        np.asarray([np.inf], dtype=np.float32),
        np.asarray([-1.0001], dtype=np.float32),
        np.asarray([1.0001], dtype=np.float32),
        np.asarray([0], dtype=np.int64),
        np.asarray([True], dtype=bool),
    ),
)
def test_quantizer_rejects_nonfinite_out_of_range_and_nonfloating_values(values: np.ndarray) -> None:
    with pytest.raises(ValueError):
        quantize_coordinates(values)


@pytest.mark.parametrize("bins", (0, -1, True, 1.5))
def test_coordinate_parameters_and_decode_ids_are_validated(bins: object) -> None:
    with pytest.raises(ValueError):
        quantize_coordinates(np.asarray([0.0], dtype=np.float32), bins=bins)
    with pytest.raises(ValueError):
        decode_coordinate_tokens(np.asarray([0], dtype=np.int64), bins=bins)
    with pytest.raises(ValueError):
        quantize_coordinates(np.asarray([0.0], dtype=np.float32), continuous_range=(1.0, 1.0))

    for ids in (
        np.asarray([-1], dtype=np.int64),
        np.asarray([256], dtype=np.int64),
        np.asarray([0.0], dtype=np.float32),
        np.asarray([True], dtype=bool),
    ):
        with pytest.raises(ValueError):
            decode_coordinate_tokens(ids)


def test_chain_fill_positions_own_three_axis_masks_by_dfs_id() -> None:
    codec = _codec()
    encoding = codec.encode(np.asarray([-1, 0, 1], dtype=np.int64))
    result = build_fill_inputs([encoding.tokens], codec)

    expected_ids = np.asarray(
        [codec.bos, codec.joint_offset, codec.vocab_size, codec.vocab_size + 1, codec.vocab_size + 2,
         codec.joint_offset + 1, codec.vocab_size, codec.vocab_size + 1, codec.vocab_size + 2,
         codec.joint_offset + 2, codec.vocab_size, codec.vocab_size + 1, codec.vocab_size + 2, codec.eos],
        dtype=np.int64,
    )
    expected_owners = np.asarray(
        [-1, -1, 0, 0, 0, -1, 1, 1, 1, -1, 2, 2, 2, -1], dtype=np.int64
    )
    np.testing.assert_array_equal(result["fill_input_ids"][0].numpy(), expected_ids)
    np.testing.assert_array_equal(result["fill_joint_ids"][0].numpy(), expected_owners)
    np.testing.assert_array_equal(
        result["fill_positions"][0].numpy(),
        np.asarray([[2, 3, 4], [6, 7, 8], [10, 11, 12]], dtype=np.int64),
    )
    np.testing.assert_array_equal(result["fill_parents"][0].numpy(), np.asarray([-1, 0, 1], dtype=np.int64))
    assert result["fill_lengths"].tolist() == [14]
    assert result["fill_joint_counts"].tolist() == [3]
    assert result["fill_attention_mask"].dtype == torch.bool


def test_branch_parent_reference_gets_no_xyz_slots_and_source_order_maps_to_dfs() -> None:
    codec = _codec()
    source_parents = np.asarray([-1, 0, 0, 1, 1], dtype=np.int64)
    encoding = codec.encode(source_parents)
    result = build_fill_inputs([encoding.tokens], codec)
    fill = result["fill_input_ids"][0].tolist()
    owners = result["fill_joint_ids"][0].tolist()
    branches = [index for index, token in enumerate(fill) if token == codec.branch]
    assert len(branches) == 2
    for branch in branches:
        new_joint = fill[branch + 2] - codec.joint_offset
        assert owners[branch : branch + 3] == [-1, -1, -1]
        assert owners[branch + 3 : branch + 6] == [new_joint, new_joint, new_joint]
    assert result["fill_joint_counts"].item() == 5
    assert result["fill_lengths"].item() == encoding.tokens.size + 3 * 5
    np.testing.assert_array_equal(result["fill_parents"][0, :5].numpy(), codec.decode(encoding.tokens))


def test_padded_structure_rows_are_ignored_only_after_complete_eos() -> None:
    codec = _codec()
    first = codec.encode(np.asarray([-1, 0, 1], dtype=np.int64)).tokens
    second = codec.encode(np.asarray([-1], dtype=np.int64)).tokens
    padded = np.full((2, first.size), codec.pad, dtype=np.int64)
    padded[0, : first.size] = first
    padded[1, : second.size] = second
    result = build_fill_inputs(padded, codec)

    assert result["fill_attention_mask"][0].all()
    assert result["fill_attention_mask"][1, result["fill_lengths"][1] :].logical_not().all()
    assert torch.all(result["fill_input_ids"][1, result["fill_lengths"][1] :] == codec.pad)
    assert torch.all(result["fill_joint_ids"][1, result["fill_lengths"][1] :] == -1)
    assert torch.all(result["fill_positions"][1, 1:] == 0)
    assert torch.all(result["fill_joint_mask"][1, 1:].logical_not())
    assert torch.all(result["fill_parents"][1, 1:] == -2)


def test_fill_capacity_fails_without_truncation() -> None:
    codec = _codec()
    sequence = codec.encode(np.asarray([-1, 0, 1], dtype=np.int64)).tokens
    with pytest.raises(ValueError, match="max_fill_tokens"):
        build_fill_inputs([sequence], codec, max_fill_tokens=int(sequence.size + 8))


def test_structure_targets_prune_and_remap_all_frames_but_keep_full_structure_targets() -> None:
    codec = _codec()
    source = np.asarray([-1, 0, 0, 1, 1, 2, 2], dtype=np.int64)
    targets = _targets([source])
    full = build_structure_fill_targets(targets, codec)
    full_encoding = codec.encode(source)
    np.testing.assert_array_equal(
        full["fill_source_indices"][0, : source.size].numpy(),
        full_encoding.dfs_order,
    )
    pruned = build_structure_fill_targets(
        targets,
        codec,
        remove_counts=[3],
        prune_seeds=[35],
    )
    expected = prune_random_leaves(source, remove_count=3, rng=np.random.default_rng(35))
    encoding = codec.encode(expected.parents)
    expected_sources = expected.kept_source_indices[encoding.dfs_order]

    np.testing.assert_array_equal(pruned["fill_source_indices"][0, : expected_sources.size].numpy(), expected_sources)
    np.testing.assert_array_equal(
        pruned["fill_parents"][0, : expected_sources.size].numpy(),
        codec.decode(encoding.tokens),
    )
    assert pruned["original_joint_counts"].tolist() == [source.size]
    assert pruned["removed_joint_counts"].tolist() == [3]
    assert 0 in expected_sources
    removed = set(expected.removed_source_indices.tolist())
    for source_index in expected_sources.tolist():
        parent = int(source[source_index])
        while parent >= 0:
            assert parent not in removed
            parent = int(source[parent])

    for field in ("structure_input_ids", "structure_attention_mask", "structure_labels", "structure_lengths"):
        torch.testing.assert_close(pruned[field], full[field])
    torch.testing.assert_close(
        pruned["xyz_targets"],
        torch.from_numpy(targets["frame_target_joints"][0][:, expected_sources, :][None]),
    )
    np.testing.assert_array_equal(
        pruned["xyz_labels"][0, :, : expected_sources.size].numpy(),
        quantize_coordinates(targets["frame_target_joints"][0][:, expected_sources, :]),
    )
    assert pruned["frame_mask"].all()


def test_coordinate_labels_do_not_change_fill_inputs() -> None:
    codec = _codec()
    source = np.asarray([-1, 0, 0], dtype=np.int64)
    first_targets = _targets([source])
    second_targets = {key: value.copy() for key, value in first_targets.items()}
    second_targets["frame_target_joints"][:, :, :3, :] += np.float32(0.35)
    first = build_structure_fill_targets(first_targets, codec)
    second = build_structure_fill_targets(second_targets, codec)
    for field in (
        "fill_input_ids",
        "fill_attention_mask",
        "fill_joint_ids",
        "fill_positions",
        "fill_joint_mask",
        "fill_lengths",
        "fill_joint_counts",
        "fill_parents",
    ):
        torch.testing.assert_close(first[field], second[field])
    assert not torch.equal(first["xyz_labels"], second["xyz_labels"])
    assert not torch.equal(first["xyz_targets"], second["xyz_targets"])


def test_target_coordinate_and_mask_validation_is_fail_closed() -> None:
    codec = _codec()
    source = np.asarray([-1, 0, 0], dtype=np.int64)
    targets = _targets([source])
    for value in (np.float32(1.1), np.float32(np.nan)):
        invalid = {key: array.copy() for key, array in targets.items()}
        invalid["frame_target_joints"][0, 0, 0, 0] = value
        with pytest.raises(ValueError):
            build_structure_fill_targets(invalid, codec)

    bad_mask = {key: array.copy() for key, array in targets.items()}
    bad_mask["joint_mask"][0] = np.asarray([True, False, True])
    with pytest.raises(ValueError):
        build_structure_fill_targets(bad_mask, codec)
    bad_padding = {key: array.copy() for key, array in targets.items()}
    bad_padding["target_parents"] = np.pad(source, (0, 1), constant_values=-1)[None]
    bad_padding["joint_mask"] = np.asarray([[True, True, True, False]])
    bad_padding["frame_target_joints"] = np.pad(
        targets["frame_target_joints"], ((0, 0), (0, 0), (0, 1), (0, 0))
    )
    with pytest.raises(ValueError):
        build_structure_fill_targets(bad_padding, codec)


@pytest.mark.parametrize(
    "remove_counts, prune_seeds",
    (
        ([1], None),
        ([-1], [1]),
        ([3], [1]),
        ([True], [1]),
        ([1.0], [1]),
        ([1], [None]),
        ([1], [1.5]),
        ([1], [True]),
        ([0], ["bad"]),
    ),
)
def test_bad_pruning_counts_and_seeds_are_rejected(remove_counts: object, prune_seeds: object) -> None:
    codec = _codec()
    targets = _targets([np.asarray([-1, 0, 0], dtype=np.int64)])
    with pytest.raises(ValueError):
        build_structure_fill_targets(
            targets,
            codec,
            remove_counts=remove_counts,
            prune_seeds=prune_seeds,
        )


def test_positive_pruning_uses_one_seeded_mask_per_asset() -> None:
    codec = _codec()
    rows = [
        np.asarray([-1, 0, 0], dtype=np.int64),
        np.asarray([-1, 0, 1, 2], dtype=np.int64),
    ]
    targets = _targets(rows)
    first = build_structure_fill_targets(targets, codec, remove_counts=[1, 1], prune_seeds=[4, 5])
    second = build_structure_fill_targets(targets, codec, remove_counts=[1, 1], prune_seeds=[4, 5])
    for field in ("fill_input_ids", "fill_source_indices", "xyz_labels", "xyz_targets"):
        torch.testing.assert_close(first[field], second[field])
    assert first["removed_joint_counts"].tolist() == [1, 1]
