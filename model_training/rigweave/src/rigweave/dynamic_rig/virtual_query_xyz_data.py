"""CPU target preparation for per-frame UniRig XYZ and per-asset aux targets.

The main stream is deliberately delegated to the existing data helpers
``_derive_tokenizer_tails`` and ``_make_token_input``.  This adapter only
validates, pads, and moves their complete token sequences.  The auxiliary
stream uses one fixed source-order parent array per asset and never consumes
frame coordinates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import numbers
from typing import Any

import numpy as np
import torch

from .data import _derive_tokenizer_tails, _make_token_input


def _to_cpu_numpy(value: Any, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    try:
        return np.asarray(value)
    except Exception as exc:  # pragma: no cover - NumPy supplies the detail
        raise ValueError(f"{name} must be convertible to a CPU NumPy array") from exc


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _integer_array(value: Any, name: str) -> np.ndarray:
    array = _to_cpu_numpy(value, name)
    if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{name} must use an integer dtype")
    return array.astype(np.int64, copy=False)


def _bool_array(value: Any, name: str) -> np.ndarray:
    array = _to_cpu_numpy(value, name)
    if not np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{name} must use boolean dtype")
    return array.astype(bool, copy=False)


def _real_float_array(value: Any, name: str) -> np.ndarray:
    array = _to_cpu_numpy(value, name)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError(f"{name} must use a real numeric dtype")
    return array


def _single_asset_frames(value: Any, name: str) -> np.ndarray:
    array = _real_float_array(value, name)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(f"raw {name} must have leading batch size one")
        array = array[0]
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"raw {name} must have shape (T,J,3) or (1,T,J,3)")
    return array


def _single_asset_vector(value: Any, name: str, dtype: str) -> np.ndarray:
    array = _bool_array(value, name) if dtype == "bool" else _integer_array(value, name)
    if array.ndim == 2:
        if array.shape[0] != 1:
            raise ValueError(f"raw {name} must have leading batch size one")
        array = array[0]
    if array.ndim != 1:
        raise ValueError(f"raw {name} must have shape (N,) or (1,N)")
    return array


def _raw_class_name(batch: Mapping[str, Any]) -> str | None:
    value = batch.get("cls", batch.get("classnames"))
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        values = list(value)
    except TypeError as exc:
        raise ValueError("raw cls/classnames metadata must contain strings") from exc
    if len(values) != 1 or not isinstance(values[0], str):
        raise ValueError("raw single-asset cls/classnames metadata must contain one string")
    return values[0]


def _collate_raw_batches(batches: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[str | None]]:
    if not batches:
        raise ValueError("raw target batches must contain at least one asset")
    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    class_names: list[str | None] = []
    frame_count: int | None = None
    max_joints = 0
    for index, batch in enumerate(batches):
        if not isinstance(batch, Mapping):
            raise ValueError(f"raw target batch {index} must be a mapping")
        required = [name for name in ("frame_target_joints", "target_parents", "joint_mask") if name not in batch]
        if required:
            raise ValueError(f"raw target batch {index} is missing {required}")
        frames = _single_asset_frames(batch["frame_target_joints"], "frame_target_joints")
        parents = _single_asset_vector(batch["target_parents"], "target_parents", "int")
        joints = _single_asset_vector(batch["joint_mask"], "joint_mask", "bool")
        if frames.shape[1] != parents.shape[0] or joints.shape != parents.shape:
            raise ValueError(f"raw target batch {index} has inconsistent joint dimensions")
        if frame_count is None:
            frame_count = int(frames.shape[0])
        elif int(frames.shape[0]) != frame_count:
            raise ValueError("all raw target batches must have the same real frame count")
        if "frame_mask" in batch:
            frame_mask = _single_asset_vector(batch["frame_mask"], "frame_mask", "bool")
            if frame_mask.shape != (frames.shape[0],):
                raise ValueError(f"raw target batch {index} frame_mask does not match frame count")
        else:
            frame_mask = np.ones((frames.shape[0],), dtype=bool)
        rows.append((frames, parents, joints, frame_mask))
        class_names.append(_raw_class_name(batch))
        max_joints = max(max_joints, int(parents.shape[0]))

    assert frame_count is not None
    asset_count = len(rows)
    frame_targets = np.zeros((asset_count, frame_count, max_joints, 3), dtype=np.float64)
    target_parents = np.full((asset_count, max_joints), -2, dtype=np.int64)
    joint_mask = np.zeros((asset_count, max_joints), dtype=bool)
    frame_mask = np.zeros((asset_count, frame_count), dtype=bool)
    for index, (frames, parents, joints, valid_frames) in enumerate(rows):
        count = int(parents.shape[0])
        frame_targets[index, :, :count] = frames
        target_parents[index, :count] = parents
        joint_mask[index, :count] = joints
        frame_mask[index] = valid_frames
    return (
        {
            "frame_target_joints": frame_targets,
            "target_parents": target_parents,
            "joint_mask": joint_mask,
            "frame_mask": frame_mask,
        },
        class_names,
    )


def _target_mapping(
    targets: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[str | None] | None]:
    if isinstance(targets, Mapping):
        return dict(targets), None
    if isinstance(targets, (str, bytes)):
        raise ValueError("targets must be a collated mapping or a list of raw mappings")
    try:
        batches = list(targets)
    except TypeError as exc:
        raise ValueError("targets must be a collated mapping or a list of raw mappings") from exc
    return _collate_raw_batches(batches)


def _normalise_classnames(
    value: str | Sequence[str], batch_size: int, metadata: list[str | None] | None,
    target_metadata: Mapping[str, Any],
) -> list[str]:
    target_value = target_metadata.get("cls", target_metadata.get("classnames"))
    if target_value is not None:
        if isinstance(target_value, str):
            target_names = [target_value] * batch_size
        else:
            try:
                target_names = list(target_value)
            except TypeError as exc:
                raise ValueError("target cls/classnames metadata must contain strings") from exc
            if len(target_names) != batch_size:
                raise ValueError("target cls/classnames metadata must match batch size")
        if not all(isinstance(name, str) and name for name in target_names):
            raise ValueError("target cls/classnames metadata must contain non-empty strings")
        return target_names
    if isinstance(value, str):
        if not value:
            raise ValueError("classnames must contain non-empty strings")
        names = [value] * batch_size
    else:
        try:
            names = list(value)
        except TypeError as exc:
            raise ValueError("classnames must be a string or sequence of strings") from exc
        if len(names) != batch_size or not all(isinstance(name, str) and name for name in names):
            raise ValueError("classnames must match batch size and contain non-empty strings")
    if metadata is not None:
        if len(metadata) != batch_size:
            raise ValueError("raw class metadata must match batch size")
        names = [str(metadata[index]) if metadata[index] is not None else names[index] for index in range(batch_size)]
    return names


def _tokenizer_int(tokenizer: Any, name: str) -> int:
    if not hasattr(tokenizer, name):
        raise ValueError(f"tokenizer must expose {name}")
    value = getattr(tokenizer, name)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"tokenizer.{name} must be an integer")
    return int(value)


def _token_array(value: Any, name: str) -> np.ndarray:
    array = _to_cpu_numpy(value, name)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{name} must be a one-dimensional integer token sequence")
    return array.astype(np.int64, copy=False)


def _batch_field(batch: Any, name: str) -> Any:
    if hasattr(batch, name):
        return getattr(batch, name)
    if isinstance(batch, Mapping) and name in batch:
        return batch[name]
    raise ValueError(f"auxiliary batch is missing {name}")


def _validate_common_targets(targets: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray]:
    required = ("frame_target_joints", "target_parents", "joint_mask")
    missing = [name for name in required if name not in targets]
    if missing:
        raise ValueError(f"targets is missing {missing}")
    raw_xyz = _real_float_array(targets["frame_target_joints"], "frame_target_joints")
    if raw_xyz.ndim != 4 or raw_xyz.shape[-1] != 3:
        raise ValueError("frame_target_joints must have shape (B,T,J,3)")
    batch_size, frame_count, joint_capacity, _ = raw_xyz.shape
    if min(batch_size, frame_count, joint_capacity) < 1:
        raise ValueError("frame_target_joints must have non-empty B, T, and J dimensions")

    parents = _integer_array(targets["target_parents"], "target_parents")
    joint_mask = _bool_array(targets["joint_mask"], "joint_mask")
    if parents.shape != (batch_size, joint_capacity) or joint_mask.shape != parents.shape:
        raise ValueError("target_parents and joint_mask must both have shape (B,J)")

    if "frame_mask" in targets:
        frame_mask = _bool_array(targets["frame_mask"], "frame_mask")
        if frame_mask.shape != (batch_size, frame_count):
            raise ValueError("frame_mask must have shape (B,T)")
        if not bool(frame_mask.all()):
            raise ValueError("padded/invalid frame_mask is unsupported; all actual frames are required")
    else:
        frame_mask = np.ones((batch_size, frame_count), dtype=bool)

    counts: list[int] = []
    for index in range(batch_size):
        count = int(joint_mask[index].sum())
        if count <= 0:
            raise ValueError(f"asset {index} has no valid joints")
        expected = np.arange(joint_capacity) < count
        if not np.array_equal(joint_mask[index], expected):
            raise ValueError("joint_mask must contain a contiguous valid prefix")
        if count < joint_capacity and bool(np.any(parents[index, count:] != -2)):
            raise ValueError("padded target_parents entries must use -2")
        actual = raw_xyz[index, :, :count]
        if not bool(np.isfinite(actual).all()):
            raise ValueError(f"asset {index} has non-finite actual frame target joints")
        if bool(np.any(actual < -1.0) or np.any(actual > 1.0)):
            raise ValueError(f"asset {index} has actual frame target joints outside [-1,1]; no clipping is allowed")
        counts.append(count)

    xyz = raw_xyz.astype(np.float32, copy=False)
    for index, count in enumerate(counts):
        if not bool(np.isfinite(xyz[index, :, :count]).all()):
            raise ValueError(f"asset {index} frame_target_joints cannot be represented as finite float32 values")
    return xyz, parents, counts, frame_mask


def _validate_main_sequence(
    sequence: Any,
    *,
    tokenizer: Any,
    class_name: str,
    bos: int,
    eos: int,
    pad: int,
    vocab_size: int,
) -> np.ndarray:
    tokens = _token_array(sequence, "tokenizer output")
    if tokens.size < 3:
        raise ValueError("tokenizer output must contain BOS, class token, and EOS")
    if bool(np.any(tokens < 0) or np.any(tokens >= vocab_size)):
        raise ValueError("tokenizer output contains a token outside tokenizer.vocab_size")
    if int(tokens[0]) != bos:
        raise ValueError("tokenizer output must start with tokenizer.bos")
    if not callable(getattr(tokenizer, "cls_name_to_token", None)):
        raise ValueError("tokenizer must expose cls_name_to_token")
    class_token = tokenizer.cls_name_to_token(class_name)
    if isinstance(class_token, (bool, np.bool_)) or not isinstance(class_token, numbers.Integral):
        raise ValueError("tokenizer.cls_name_to_token must return an integer")
    if int(tokens[1]) != int(class_token):
        raise ValueError("tokenizer output class token does not match cls_name_to_token")
    if int(tokens[-1]) != eos or bool(np.any(tokens[:-1] == eos)):
        raise ValueError("tokenizer output must terminate with exactly one EOS")
    if bool(np.any(tokens == pad)):
        raise ValueError("tokenizer output must not contain padding before EOS")
    return tokens


def _validate_aux_batch(aux_batch: Any, codec: Any, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    inputs = _integer_array(_batch_field(aux_batch, "input_ids"), "aux_input_ids")
    labels = _integer_array(_batch_field(aux_batch, "labels"), "aux_labels")
    attention = _bool_array(_batch_field(aux_batch, "attention_mask"), "aux_attention_mask")
    lengths = _integer_array(_batch_field(aux_batch, "lengths"), "aux_lengths").reshape(-1)
    orders_value = _batch_field(aux_batch, "dfs_orders")
    orders = tuple(_integer_array(order, "aux dfs order").reshape(-1) for order in orders_value)
    if inputs.ndim != 2 or labels.shape != inputs.shape or attention.shape != inputs.shape:
        raise ValueError("auxiliary teacher-forcing arrays must share shape (B,A)")
    if inputs.shape[0] != batch_size or lengths.shape != (batch_size,) or len(orders) != batch_size:
        raise ValueError("auxiliary teacher-forcing batch size mismatch")
    aux_pad = _tokenizer_int(codec, "pad")
    aux_bos = _tokenizer_int(codec, "bos")
    aux_eos = _tokenizer_int(codec, "eos")
    aux_vocab = _tokenizer_int(codec, "vocab_size")
    for index in range(batch_size):
        valid_steps = int(attention[index].sum())
        if valid_steps < 2 or not np.array_equal(attention[index], np.arange(inputs.shape[1]) < valid_steps):
            raise ValueError("aux_attention_mask must contain a valid prefix with BOS and EOS label")
        if int(inputs[index, 0]) != aux_bos:
            raise ValueError("aux_input_ids must start with BOS")
        valid_inputs = inputs[index, :valid_steps]
        valid_labels = labels[index, :valid_steps]
        if bool(np.any(valid_inputs < 0) or np.any(valid_inputs >= aux_vocab)):
            raise ValueError("aux_input_ids contains an out-of-vocabulary token")
        if bool(np.any(valid_labels < 0) or np.any(valid_labels >= aux_vocab)):
            raise ValueError("aux_labels contains an out-of-vocabulary token")
        if int(valid_labels[-1]) != aux_eos:
            raise ValueError("aux_labels must terminate with EOS")
        if bool(np.any(inputs[index, valid_steps:] != aux_pad)):
            raise ValueError("aux_input_ids padding must use codec.pad")
        if bool(np.any(labels[index, valid_steps:] != -100)):
            raise ValueError("aux_labels padding must use -100")
        if int(lengths[index]) != valid_steps + 1:
            raise ValueError("aux_lengths must include the complete BOS-through-EOS sequence")
    return inputs, labels, attention, lengths, orders


def _to_device(array: np.ndarray, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=dtype, device=device)


def build_xyz_aux_targets(
    targets: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    tokenizer: Any,
    aux_codec: Any,
    device: torch.device | str | None = None,
    *,
    classnames: str | Sequence[str] = "articulationxl",
    max_main_tokens: int = 2048,
) -> dict[str, Any]:
    """Build complete per-frame main targets and one aux target per asset.

    ``targets`` may be the dictionary from ``collate_virtual_targets`` or a
    list of raw single-asset medium batches.  Real frame masks must be all
    true; this adapter does not invent GT for padded frames.  Main tokenization
    remains in source joint order and uses the existing data helpers exactly.

    Returned ``main_lengths`` include the final EOS.  ``aux_lengths`` include
    the complete aux BOS-through-EOS sequence, while ``aux_input_ids`` and
    ``aux_labels`` are the shifted teacher-forcing arrays.
    """

    max_main_tokens = _positive_int(max_main_tokens, "max_main_tokens")
    target_mapping, raw_names = _target_mapping(targets)
    xyz, parents, joint_counts, frame_mask = _validate_common_targets(target_mapping)
    batch_size, frame_count, _, _ = xyz.shape
    names = _normalise_classnames(classnames, batch_size, raw_names, target_mapping)

    bos = _tokenizer_int(tokenizer, "bos")
    eos = _tokenizer_int(tokenizer, "eos")
    pad = _tokenizer_int(tokenizer, "pad")
    vocab_size = _tokenizer_int(tokenizer, "vocab_size")
    for name, token in (("bos", bos), ("eos", eos), ("pad", pad)):
        if not 0 <= token < vocab_size:
            raise ValueError(f"tokenizer.{name} is outside tokenizer.vocab_size")

    main_sequences: list[list[np.ndarray]] = []
    main_lengths = np.zeros((batch_size, frame_count), dtype=np.int64)
    for asset in range(batch_size):
        count = joint_counts[asset]
        source_parents = [int(value) for value in parents[asset, :count].tolist()]
        tokenizer_parents = [None if value < 0 else value for value in source_parents]
        asset_sequences: list[np.ndarray] = []
        for frame in range(frame_count):
            joints = xyz[asset, frame, :count]
            tails = _derive_tokenizer_tails(joints, tokenizer_parents)
            sequence = _make_token_input(
                joints, tokenizer_parents, names[asset], tokenizer, tails=tails
            )
            sequence = _validate_main_sequence(
                sequence, tokenizer=tokenizer, class_name=names[asset], bos=bos,
                eos=eos, pad=pad, vocab_size=vocab_size,
            )
            length = int(sequence.shape[0])
            if length > max_main_tokens:
                raise ValueError(
                    f"main token sequence length {length} exceeds max_main_tokens={max_main_tokens}"
                )
            asset_sequences.append(sequence)
            main_lengths[asset, frame] = length
        main_sequences.append(asset_sequences)

    main_token_width = int(main_lengths.max())
    main_input_ids = np.full((batch_size, frame_count, main_token_width), pad, dtype=np.int64)
    main_attention_mask = np.zeros((batch_size, frame_count, main_token_width), dtype=bool)
    for asset, sequences in enumerate(main_sequences):
        for frame, sequence in enumerate(sequences):
            length = int(sequence.shape[0])
            main_input_ids[asset, frame, :length] = sequence
            main_attention_mask[asset, frame, :length] = True

    parent_rows = [parents[index, :count].copy() for index, count in enumerate(joint_counts)]
    aux_batch = aux_codec.batch_teacher_forcing(parent_rows)
    aux_input_ids, aux_labels, aux_attention_mask, aux_lengths, dfs_orders = _validate_aux_batch(
        aux_batch, aux_codec, batch_size
    )
    max_joint_count = parents.shape[1]
    aux_dfs_order = np.full((batch_size, max_joint_count), -1, dtype=np.int64)
    aux_source_to_dfs = np.full((batch_size, max_joint_count), -1, dtype=np.int64)
    aux_dfs_mask = np.zeros((batch_size, max_joint_count), dtype=bool)
    for asset, order in enumerate(dfs_orders):
        count = joint_counts[asset]
        if order.shape != (count,) or sorted(order.tolist()) != list(range(count)):
            raise ValueError("auxiliary codec returned an invalid DFS mapping")
        aux_dfs_order[asset, :count] = order
        aux_dfs_mask[asset, :count] = True
        aux_source_to_dfs[asset, order] = np.arange(count, dtype=np.int64)

    target_device = torch.device("cpu" if device is None else device)
    return {
        "main_input_ids": _to_device(main_input_ids, torch.long, target_device),
        "main_attention_mask": _to_device(main_attention_mask, torch.bool, target_device),
        "main_lengths": _to_device(main_lengths, torch.long, target_device),
        "main_token_lengths": _to_device(main_lengths, torch.long, target_device),
        "aux_input_ids": _to_device(aux_input_ids, torch.long, target_device),
        "aux_attention_mask": _to_device(aux_attention_mask, torch.bool, target_device),
        "aux_labels": _to_device(aux_labels, torch.long, target_device),
        "aux_lengths": _to_device(aux_lengths, torch.long, target_device),
        "aux_input_lengths": _to_device(aux_attention_mask.sum(axis=1), torch.long, target_device),
        "aux_dfs_order": _to_device(aux_dfs_order, torch.long, target_device),
        "aux_source_to_dfs": _to_device(aux_source_to_dfs, torch.long, target_device),
        "aux_dfs_mask": _to_device(aux_dfs_mask, torch.bool, target_device),
        "aux_joint_counts": _to_device(np.asarray(joint_counts, dtype=np.int64), torch.long, target_device),
        "frame_mask": _to_device(frame_mask, torch.bool, target_device),
        "classnames": names,
    }


__all__ = ["build_xyz_aux_targets"]
