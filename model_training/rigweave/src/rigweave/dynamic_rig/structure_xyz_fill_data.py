"""CPU preparation for structure-conditioned XYZ fill targets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import numbers
from typing import Any

import numpy as np
import torch

from .aux_skeleton_tokens import AuxSkeletonTokens
from .leaf_pruning import prune_random_leaves
from .virtual_query_xyz_data import _validate_common_targets


def _strict_integer(value: Any) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_))


def _positive_integer(value: Any, name: str) -> int:
    if not _strict_integer(value) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _to_cpu_numpy(value: Any, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    try:
        return np.asarray(value)
    except Exception as exc:  # pragma: no cover - NumPy supplies the detail
        raise ValueError(f"{name} must be convertible to a CPU NumPy array") from exc


def _continuous_range(value: Sequence[float]) -> tuple[float, float]:
    if isinstance(value, (str, bytes)):
        raise ValueError("continuous_range must contain two finite real values")
    try:
        values = list(value)
    except TypeError as exc:
        raise ValueError("continuous_range must contain two finite real values") from exc
    if len(values) != 2:
        raise ValueError("continuous_range must contain exactly two values")
    if any(isinstance(item, (bool, np.bool_)) or not isinstance(item, numbers.Real) for item in values):
        raise ValueError("continuous_range must contain two finite real values")
    lo, hi = float(values[0]), float(values[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or not hi > lo:
        raise ValueError("continuous_range must be finite with hi > lo")
    return lo, hi


def _validate_bins(bins: Any) -> int:
    return _positive_integer(bins, "bins")


def quantize_coordinates(
    values: Any,
    *,
    bins: int = 256,
    continuous_range: Sequence[float] = (-1.0, 1.0),
) -> np.ndarray:
    """Quantize finite floating coordinates without clipping invalid GT."""

    bin_count = _validate_bins(bins)
    lo, hi = _continuous_range(continuous_range)
    array = _to_cpu_numpy(values, "values")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError("values must use a floating dtype")
    if not bool(np.isfinite(array).all()):
        raise ValueError("values must contain only finite coordinates")
    if bool(np.any(array < lo) or np.any(array > hi)):
        raise ValueError("values contain coordinates outside continuous_range")

    # Keep NumPy's input dtype through the arithmetic.  In particular, a
    # float32 input must use float32 rounding before the integer cast.
    scaled = (array - lo) / (hi - lo)
    scaled *= bin_count
    return np.clip(scaled.round(), 0, bin_count - 1).astype(np.int64)


def decode_coordinate_tokens(
    ids: Any,
    *,
    bins: int = 256,
    continuous_range: Sequence[float] = (-1.0, 1.0),
) -> np.ndarray:
    """Decode coordinate-bin IDs using the UniRig half-bin convention."""

    bin_count = _validate_bins(bins)
    lo, hi = _continuous_range(continuous_range)
    array = _to_cpu_numpy(ids, "ids")
    if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
        raise ValueError("ids must use an integer dtype")
    if bool(np.any(array < 0) or np.any(array >= bin_count)):
        raise ValueError("ids contain a value outside [0, bins)")

    decoded = array.astype(np.float32) + 0.5
    decoded /= bin_count
    decoded *= hi - lo
    decoded += lo
    return decoded.astype(np.float32, copy=False)


def _codec_integer(codec: AuxSkeletonTokens, name: str) -> int:
    value = getattr(codec, name, None)
    if not _strict_integer(value):
        raise ValueError(f"codec.{name} must be an integer")
    return int(value)


def _sequence_rows(value: Any, name: str) -> list[Any]:
    if hasattr(value, "tokens") and not isinstance(value, (np.ndarray, torch.Tensor)):
        return [value]
    if isinstance(value, torch.Tensor):
        if value.ndim == 1:
            return [value]
        if value.ndim == 2 and value.shape[0] > 0:
            return [value[index] for index in range(value.shape[0])]
        raise ValueError(f"{name} must contain one or more one-dimensional sequences")
    if isinstance(value, np.ndarray):
        if value.ndim == 1:
            return [value]
        if value.ndim == 2 and value.shape[0] > 0:
            return [value[index] for index in range(value.shape[0])]
        raise ValueError(f"{name} must contain one or more one-dimensional sequences")
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain one or more token sequences")
    try:
        raw = list(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain one or more token sequences") from exc
    if not raw:
        raise ValueError(f"{name} must contain one or more token sequences")
    first = raw[0]
    if hasattr(first, "tokens") or isinstance(first, (list, tuple, np.ndarray, torch.Tensor)):
        return raw
    try:
        iter(first)
    except TypeError:
        return [raw]
    return raw


def _integer_token_row(row: Any, name: str) -> np.ndarray:
    if hasattr(row, "tokens"):
        row = row.tokens
    array = _to_cpu_numpy(row, name)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{name} must be a one-dimensional integer token sequence")
    return array.astype(np.int64, copy=False)


def _validated_structure_row(
    row: Any,
    codec: AuxSkeletonTokens,
    *,
    row_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    tokens = _integer_token_row(row, f"structure_sequences[{row_index}]")
    pad = _codec_integer(codec, "pad")
    end = int(tokens.shape[0])
    while end > 0 and int(tokens[end - 1]) == pad:
        end -= 1
    tokens = tokens[:end]
    if tokens.size == 0:
        raise ValueError(f"structure_sequences[{row_index}] is empty after padding")
    try:
        parents = np.asarray(codec.decode(tokens), dtype=np.int64)
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"structure_sequences[{row_index}] is not a complete codec structure") from exc
    if parents.ndim != 1 or parents.size == 0:
        raise ValueError(f"structure_sequences[{row_index}] codec.decode returned invalid parents")
    return tokens, parents


def _compose_fill_row(
    tokens: np.ndarray,
    parents: np.ndarray,
    codec: AuxSkeletonTokens,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vocab_size = _codec_integer(codec, "vocab_size")
    joint_offset = _codec_integer(codec, "joint_offset")
    branch = _codec_integer(codec, "branch")
    bos = _codec_integer(codec, "bos")
    eos = _codec_integer(codec, "eos")
    mask_tokens = (vocab_size, vocab_size + 1, vocab_size + 2)

    fill_ids: list[int] = []
    fill_joint_ids: list[int] = []
    fill_positions: list[tuple[int, int, int]] = []

    def append_plain(token: int) -> None:
        fill_ids.append(token)
        fill_joint_ids.append(-1)

    def append_xyz(owner: int) -> None:
        start = len(fill_ids)
        fill_ids.extend(mask_tokens)
        fill_joint_ids.extend((owner, owner, owner))
        fill_positions.append((start, start + 1, start + 2))

    cursor = 0
    while cursor < tokens.size:
        token = int(tokens[cursor])
        cursor += 1
        append_plain(token)
        if token == branch:
            parent_token = int(tokens[cursor])
            new_token = int(tokens[cursor + 1])
            cursor += 2
            append_plain(parent_token)
            append_plain(new_token)
            append_xyz(new_token - joint_offset)
        elif token not in (bos, eos):
            append_xyz(token - joint_offset)

    if len(fill_positions) != int(parents.size):
        raise ValueError("codec structure does not contain one XYZ slot group per decoded joint")
    return (
        np.asarray(fill_ids, dtype=np.int64),
        np.asarray(fill_joint_ids, dtype=np.int64),
        np.asarray(fill_positions, dtype=np.int64),
    )


def _torch_array(array: np.ndarray, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=dtype, device=device)


def build_fill_inputs(
    structure_sequences: Any,
    codec: AuxSkeletonTokens,
    device: torch.device | str | None = None,
    *,
    max_fill_tokens: int = 3076,
) -> dict[str, torch.Tensor]:
    """Insert X/Y/Z mask slots after each newly introduced structure joint."""

    capacity = _positive_integer(max_fill_tokens, "max_fill_tokens")
    rows = _sequence_rows(structure_sequences, "structure_sequences")
    prepared: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for index, row in enumerate(rows):
        tokens, parents = _validated_structure_row(row, codec, row_index=index)
        fill_ids, fill_joint_ids, fill_positions = _compose_fill_row(tokens, parents, codec)
        if fill_ids.size > capacity:
            raise ValueError(
                f"structure_sequences[{index}] expands to {fill_ids.size} fill tokens, "
                f"exceeding max_fill_tokens={capacity}"
            )
        prepared.append((fill_ids, fill_joint_ids, fill_positions, parents))

    batch_size = len(prepared)
    max_length = max(int(item[0].size) for item in prepared)
    max_joints = max(int(item[3].size) for item in prepared)
    pad = _codec_integer(codec, "pad")
    fill_ids = np.full((batch_size, max_length), pad, dtype=np.int64)
    fill_attention = np.zeros((batch_size, max_length), dtype=bool)
    fill_joint_ids = np.full((batch_size, max_length), -1, dtype=np.int64)
    fill_positions = np.zeros((batch_size, max_joints, 3), dtype=np.int64)
    fill_joint_mask = np.zeros((batch_size, max_joints), dtype=bool)
    fill_parents = np.full((batch_size, max_joints), -2, dtype=np.int64)
    fill_lengths = np.zeros(batch_size, dtype=np.int64)
    fill_joint_counts = np.zeros(batch_size, dtype=np.int64)

    for index, (row_ids, row_joint_ids, row_positions, parents) in enumerate(prepared):
        length = int(row_ids.size)
        joint_count = int(parents.size)
        fill_ids[index, :length] = row_ids
        fill_attention[index, :length] = True
        fill_joint_ids[index, :length] = row_joint_ids
        fill_positions[index, :joint_count] = row_positions
        fill_joint_mask[index, :joint_count] = True
        fill_parents[index, :joint_count] = parents
        fill_lengths[index] = length
        fill_joint_counts[index] = joint_count

    target_device = torch.device("cpu" if device is None else device)
    return {
        "fill_input_ids": _torch_array(fill_ids, torch.long, target_device),
        "fill_attention_mask": _torch_array(fill_attention, torch.bool, target_device),
        "fill_joint_ids": _torch_array(fill_joint_ids, torch.long, target_device),
        "fill_positions": _torch_array(fill_positions, torch.long, target_device),
        "fill_joint_mask": _torch_array(fill_joint_mask, torch.bool, target_device),
        "fill_lengths": _torch_array(fill_lengths, torch.long, target_device),
        "fill_joint_counts": _torch_array(fill_joint_counts, torch.long, target_device),
        "fill_parents": _torch_array(fill_parents, torch.long, target_device),
    }


def _per_asset_values(value: Any, batch_size: int, name: str) -> list[Any]:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
        if array.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional per-asset sequence")
        values = array.tolist()
    elif isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional per-asset sequence")
        values = value.tolist()
    elif isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a one-dimensional per-asset sequence")
    else:
        try:
            values = list(value)
        except TypeError as exc:
            raise ValueError(f"{name} must be a one-dimensional per-asset sequence") from exc
    if len(values) != batch_size:
        raise ValueError(f"{name} must contain one value per asset")
    return values


def _resolve_pruning(
    remove_counts: Any,
    prune_seeds: Any,
    counts: list[int],
) -> tuple[list[int], list[int | None]]:
    batch_size = len(counts)
    raw_counts = (
        [0] * batch_size
        if remove_counts is None
        else _per_asset_values(remove_counts, batch_size, "remove_counts")
    )
    resolved_counts: list[int] = []
    for index, (value, count) in enumerate(zip(raw_counts, counts)):
        if not _strict_integer(value):
            raise ValueError(f"remove_counts[{index}] must be an integer")
        remove_count = int(value)
        if remove_count < 0 or remove_count > count - 1:
            raise ValueError(f"remove_counts[{index}] must be in [0, {count - 1}]")
        resolved_counts.append(remove_count)

    if prune_seeds is None:
        if any(value > 0 for value in resolved_counts):
            raise ValueError("positive remove_counts require one integer prune_seed per asset")
        return resolved_counts, [None] * batch_size

    raw_seeds = _per_asset_values(prune_seeds, batch_size, "prune_seeds")
    resolved_seeds: list[int | None] = []
    for index, (value, remove_count) in enumerate(zip(raw_seeds, resolved_counts)):
        if value is None and remove_count == 0:
            resolved_seeds.append(None)
            continue
        if not _strict_integer(value):
            raise ValueError(f"prune_seeds[{index}] must be an integer when supplied")
        resolved_seeds.append(int(value))
    return resolved_counts, resolved_seeds


def build_structure_fill_targets(
    targets: Mapping[str, Any],
    codec: AuxSkeletonTokens,
    device: torch.device | str | None = None,
    *,
    max_fill_tokens: int = 3076,
    bins: int = 256,
    continuous_range: Sequence[float] = (-1.0, 1.0),
    remove_counts: Any = None,
    prune_seeds: Any = None,
) -> dict[str, torch.Tensor]:
    """Build full-structure CE targets and pruned structure-to-XYZ fill targets."""

    if not isinstance(targets, Mapping):
        raise ValueError("targets must be a mapping")
    bin_count = _validate_bins(bins)
    coord_range = _continuous_range(continuous_range)
    xyz, parents, counts, frame_mask = _validate_common_targets(dict(targets))
    batch_size, frame_count = int(xyz.shape[0]), int(xyz.shape[1])
    resolved_counts, resolved_seeds = _resolve_pruning(remove_counts, prune_seeds, counts)

    source_parent_rows = [parents[index, :count].copy() for index, count in enumerate(counts)]
    structure_batch = codec.batch_teacher_forcing(source_parent_rows)

    pruned_sequences: list[np.ndarray] = []
    fill_source_rows: list[np.ndarray] = []
    removed_counts: list[int] = []
    for index, (source_row, remove_count, seed) in enumerate(
        zip(source_parent_rows, resolved_counts, resolved_seeds)
    ):
        if remove_count == 0:
            kept_source_indices = np.arange(source_row.size, dtype=np.int64)
            pruned_parents = source_row
        else:
            if seed is None:  # guarded by _resolve_pruning; retain a local invariant
                raise ValueError(f"prune_seeds[{index}] is required for positive pruning")
            try:
                random = np.random.default_rng(seed)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"prune_seeds[{index}] is not a valid NumPy generator seed") from exc
            pruned = prune_random_leaves(source_row, remove_count=remove_count, rng=random)
            kept_source_indices = pruned.kept_source_indices
            pruned_parents = pruned.parents
        encoding = codec.encode(pruned_parents)
        dfs_source_indices = kept_source_indices[encoding.dfs_order]
        pruned_sequences.append(encoding.tokens)
        fill_source_rows.append(np.asarray(dfs_source_indices, dtype=np.int64))
        removed_counts.append(int(source_row.size - kept_source_indices.size))

    fill = build_fill_inputs(
        pruned_sequences,
        codec,
        device=device,
        max_fill_tokens=max_fill_tokens,
    )
    max_fill_joints = max(int(row.size) for row in fill_source_rows)
    xyz_labels = np.full((batch_size, frame_count, max_fill_joints, 3), -100, dtype=np.int64)
    xyz_targets = np.zeros((batch_size, frame_count, max_fill_joints, 3), dtype=np.float32)
    fill_source_indices = np.full((batch_size, max_fill_joints), -1, dtype=np.int64)
    for index, source_indices in enumerate(fill_source_rows):
        joint_count = int(source_indices.size)
        fill_source_indices[index, :joint_count] = source_indices
        xyz_targets[index, :, :joint_count] = xyz[index][:, source_indices, :]
        xyz_labels[index, :, :joint_count] = quantize_coordinates(
            xyz_targets[index, :, :joint_count],
            bins=bin_count,
            continuous_range=coord_range,
        )

    target_device = torch.device("cpu" if device is None else device)
    result = dict(fill)
    result.update(
        {
            "structure_input_ids": _torch_array(structure_batch.input_ids, torch.long, target_device),
            "structure_attention_mask": _torch_array(structure_batch.attention_mask, torch.bool, target_device),
            "structure_labels": _torch_array(structure_batch.labels, torch.long, target_device),
            "structure_lengths": _torch_array(structure_batch.lengths, torch.long, target_device),
            "xyz_labels": _torch_array(xyz_labels, torch.long, target_device),
            "xyz_targets": _torch_array(xyz_targets, torch.float32, target_device),
            "fill_source_indices": _torch_array(fill_source_indices, torch.long, target_device),
            "original_joint_counts": _torch_array(np.asarray(counts, dtype=np.int64), torch.long, target_device),
            "removed_joint_counts": _torch_array(np.asarray(removed_counts, dtype=np.int64), torch.long, target_device),
            "frame_mask": _torch_array(frame_mask, torch.bool, target_device),
        }
    )
    return result


__all__ = [
    "build_fill_inputs",
    "build_structure_fill_targets",
    "decode_coordinate_tokens",
    "quantize_coordinates",
]
