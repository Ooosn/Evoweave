"""CPU-safe evaluation helpers for structure-conditioned XYZ filling."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
import math
import numbers
from typing import Any

import numpy as np
import torch

from rigweave.dynamic_rig.aux_skeleton_tokens import AuxSkeletonTokens
from rigweave.dynamic_rig.structure_xyz_fill_data import (
    build_fill_inputs,
    decode_coordinate_tokens,
)
from rigweave.dynamic_rig.skeleton_topology_metrics import _topology_metrics
from rigweave.dynamic_rig.virtual_query_xyz_data import build_xyz_aux_targets


def _strict_integer(value: Any) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_))


def _cpu_numpy(value: Any, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    try:
        return np.asarray(value)
    except Exception as exc:  # pragma: no cover - NumPy supplies the detail
        raise ValueError(f"{name} must be convertible to a CPU NumPy array") from exc


def _integer_array(value: Any, name: str) -> np.ndarray:
    array = _cpu_numpy(value, name)
    if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{name} must use an integer dtype")
    return array.astype(np.int64, copy=False)


def _bool_array(value: Any, name: str) -> np.ndarray:
    array = _cpu_numpy(value, name)
    if not np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{name} must use a boolean dtype")
    return array.astype(bool, copy=False)


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _validate_frame_indices(frame_indices: Any, frame_count: int | None) -> list[int]:
    if isinstance(frame_indices, (str, bytes)):
        raise ValueError("frame_indices must be a non-empty sequence of unique integers")
    try:
        values = list(frame_indices)
    except TypeError as exc:
        raise ValueError("frame_indices must be a non-empty sequence of unique integers") from exc
    if not values:
        raise ValueError("frame_indices must be non-empty")
    if any(not _strict_integer(value) for value in values):
        raise ValueError("frame_indices must contain only non-boolean integers")
    result = [int(value) for value in values]
    if len(set(result)) != len(result):
        raise ValueError("frame_indices must not contain duplicates")
    if frame_count is not None and any(value < 0 or value >= frame_count for value in result):
        raise ValueError(f"frame_indices must be within [0, {frame_count})")
    return result


def _continuous_range(value: Sequence[float]) -> tuple[float, float]:
    if isinstance(value, (str, bytes)):
        raise ValueError("continuous_range must contain two finite real values")
    try:
        values = list(value)
    except TypeError as exc:
        raise ValueError("continuous_range must contain two finite real values") from exc
    if len(values) != 2 or any(
        isinstance(item, (bool, np.bool_)) or not isinstance(item, numbers.Real)
        for item in values
    ):
        raise ValueError("continuous_range must contain two finite real values")
    lo, hi = float(values[0]), float(values[1])
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        raise ValueError("continuous_range must be finite with hi > lo")
    return lo, hi


def _asset_id(geometry: Mapping[str, Any]) -> str:
    if "asset_id" not in geometry:
        raise ValueError("geometry is missing asset_id")
    value = geometry["asset_id"]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        value = list(value)
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("geometry asset_id must contain exactly one value")
        value = value[0]
    if value is None:
        raise ValueError("geometry asset_id must be non-null")
    return str(value)


@dataclass(frozen=True)
class _PreparedItem:
    asset_id: str
    geometry_xyz: np.ndarray
    geometry_parents: np.ndarray
    raw_frame_indices: np.ndarray
    target_xyz_dfs: np.ndarray
    target_labels: np.ndarray
    target_fill_parents: np.ndarray
    fill_source_indices: np.ndarray
    target_tokens: list[int]


def _joint_count_from_geometry(geometry: Mapping[str, Any], parents: np.ndarray) -> int:
    if "joint_mask" in geometry:
        mask = _bool_array(geometry["joint_mask"], "geometry joint_mask")
        if mask.ndim == 2:
            if mask.shape[0] != 1:
                raise ValueError("geometry joint_mask must have batch size one")
            mask = mask[0]
        if mask.ndim != 1 or not bool(mask.any()):
            raise ValueError("geometry joint_mask must contain a non-empty valid prefix")
        count = int(mask.sum())
        if not np.array_equal(mask, np.arange(mask.size) < count):
            raise ValueError("geometry joint_mask must contain a contiguous valid prefix")
    elif "joint_count" in geometry:
        counts = _integer_array(geometry["joint_count"], "geometry joint_count").reshape(-1)
        if counts.shape != (1,):
            raise ValueError("geometry joint_count must contain one value")
        count = int(counts[0])
    else:
        padded = np.flatnonzero(parents[0] == -2)
        count = int(padded[0]) if padded.size else int(parents.shape[1])
    if count <= 0 or count > parents.shape[1]:
        raise ValueError("geometry joint_count is outside its parent capacity")
    if count < parents.shape[1] and bool(np.any(parents[0, count:] != -2)):
        raise ValueError("geometry padded target_parents entries must use -2")
    return count


def _prepare_item(
    surface: Mapping[str, Any],
    targets: Mapping[str, Any],
    geometry: Mapping[str, Any],
    codec: AuxSkeletonTokens,
    *,
    bins: int,
) -> _PreparedItem:
    if not isinstance(surface, Mapping) or not isinstance(targets, Mapping) or not isinstance(geometry, Mapping):
        raise ValueError("each evaluation item must contain surface, targets, and geometry mappings")
    if "query_points" not in surface:
        raise ValueError("surface is missing query_points")
    query_points = surface["query_points"]
    if not isinstance(query_points, torch.Tensor) or query_points.ndim != 4:
        raise ValueError("surface query_points must be a tensor with shape (1,8,Q,3)")
    if query_points.shape[0] != 1 or query_points.shape[1] < 1 or query_points.shape[-1] != 3:
        raise ValueError("surface query_points must have shape (1,T,Q,3) with T > 0")
    if not query_points.is_floating_point() or not bool(torch.isfinite(query_points).all()):
        raise ValueError("surface query_points must be finite floating values")

    geometry_xyz = _cpu_numpy(geometry.get("frame_target_joints"), "geometry frame_target_joints")
    geometry_parents = _integer_array(geometry.get("target_parents"), "geometry target_parents")
    frame_count = int(query_points.shape[1])
    if geometry_xyz.ndim != 4 or geometry_xyz.shape[0] != 1 or geometry_xyz.shape[1] != frame_count:
        raise ValueError("geometry frame_target_joints must have shape (1,T,J,3)")
    if geometry_xyz.shape[-1] != 3 or geometry_parents.ndim != 2 or geometry_parents.shape[0] != 1:
        raise ValueError("geometry frame_target_joints and target_parents have incompatible shapes")
    if geometry_xyz.shape[2] != geometry_parents.shape[1]:
        raise ValueError("geometry frame_target_joints and target_parents joint capacities differ")
    if not bool(np.isfinite(geometry_xyz).all()):
        raise ValueError("geometry frame_target_joints must be finite")
    geometry_xyz = geometry_xyz.astype(np.float32, copy=False)
    count = _joint_count_from_geometry(geometry, geometry_parents)
    if bool(np.any(geometry_xyz[:, :, :count] < -1.0) or np.any(geometry_xyz[:, :, :count] > 1.0)):
        raise ValueError("geometry frame_target_joints must lie within [-1,1]")

    selected_frames = _integer_array(geometry.get("selected_frames"), "geometry selected_frames")
    if selected_frames.shape == (1, frame_count):
        raw_frame_indices = selected_frames[0]
    elif selected_frames.shape == (frame_count,):
        raw_frame_indices = selected_frames
    else:
        raise ValueError("geometry selected_frames must have shape (1,T) or (T,)")

    required_target_keys = (
        "removed_joint_counts",
        "original_joint_counts",
        "fill_joint_counts",
        "fill_source_indices",
        "fill_parents",
        "xyz_labels",
        "xyz_targets",
        "frame_mask",
    )
    missing = [key for key in required_target_keys if key not in targets]
    if missing:
        raise ValueError(f"targets is missing required evaluator fields {missing}")
    removed = _integer_array(targets["removed_joint_counts"], "removed_joint_counts").reshape(-1)
    original_counts = _integer_array(targets["original_joint_counts"], "original_joint_counts").reshape(-1)
    fill_counts = _integer_array(targets["fill_joint_counts"], "fill_joint_counts").reshape(-1)
    if removed.shape != (1,) or original_counts.shape != (1,) or fill_counts.shape != (1,):
        raise ValueError("single-asset target counts must each contain one value")
    if int(removed[0]) != 0:
        raise ValueError("evaluation requires unpruned targets; removed_joint_counts must be zero")
    if int(original_counts[0]) != count or int(fill_counts[0]) != count:
        raise ValueError("target joint counts do not match original geometry")

    target_xyz = _cpu_numpy(targets["xyz_targets"], "targets xyz_targets")
    target_labels = _integer_array(targets["xyz_labels"], "targets xyz_labels")
    target_frame_mask = _bool_array(targets["frame_mask"], "targets frame_mask")
    if target_xyz.shape[:2] != (1, frame_count) or target_xyz.shape[-1] != 3:
        raise ValueError("targets xyz_targets must have shape (1,T,J,3)")
    if target_labels.shape != target_xyz.shape or target_frame_mask.shape != (1, frame_count):
        raise ValueError("target XYZ labels and frame mask have incompatible shapes")
    if not bool(target_frame_mask.all()) or not bool(np.isfinite(target_xyz).all()):
        raise ValueError("targets frame mask must be fully valid and xyz_targets finite")
    if target_xyz.shape[2] < count:
        raise ValueError("targets xyz_targets joint capacity is too small")
    valid_labels = target_labels[:, :, :count]
    if bool(np.any(valid_labels < 0) or np.any(valid_labels >= bins)):
        raise ValueError("targets xyz_labels contain an invalid coordinate bin")
    if target_labels.shape[2] > count and bool(np.any(target_labels[:, :, count:] != -100)):
        raise ValueError("targets padded xyz_labels entries must use -100")

    fill_source = _integer_array(targets["fill_source_indices"], "fill_source_indices")
    target_fill_parents = _integer_array(targets["fill_parents"], "fill_parents")
    if fill_source.ndim != 2 or target_fill_parents.ndim != 2 or fill_source.shape[0] != 1:
        raise ValueError("fill source mappings must have batch size one")
    if fill_source.shape != target_fill_parents.shape or fill_source.shape[1] < count:
        raise ValueError("fill source mappings have incompatible capacities")

    target_encoding = codec.encode(geometry_parents[0, :count])
    expected_source = np.asarray(target_encoding.dfs_order, dtype=np.int64)
    expected_parents = np.asarray(codec.decode(target_encoding.tokens), dtype=np.int64)
    if not np.array_equal(fill_source[0, :count], expected_source):
        raise ValueError("fill_source_indices do not preserve the original DFS/source mapping")
    if not np.array_equal(target_fill_parents[0, :count], expected_parents):
        raise ValueError("fill_parents do not match codec DFS parents")
    if fill_source.shape[1] > count and bool(np.any(fill_source[0, count:] != -1)):
        raise ValueError("padded fill_source_indices entries must use -1")
    if target_fill_parents.shape[1] > count and bool(np.any(target_fill_parents[0, count:] != -2)):
        raise ValueError("padded fill_parents entries must use -2")

    target_xyz_dfs = target_xyz[0, :, :count].astype(np.float32, copy=False)
    expected_xyz_dfs = geometry_xyz[0][:, expected_source, :]
    if not np.array_equal(target_xyz_dfs, expected_xyz_dfs):
        raise ValueError("targets xyz_targets do not preserve original GT source identity")
    return _PreparedItem(
        asset_id=_asset_id(geometry),
        geometry_xyz=geometry_xyz[0, :, :count].copy(),
        geometry_parents=geometry_parents[0, :count].copy(),
        raw_frame_indices=raw_frame_indices.copy(),
        target_xyz_dfs=target_xyz_dfs.copy(),
        target_labels=target_labels[0, :, :count].copy(),
        target_fill_parents=expected_parents,
        fill_source_indices=expected_source,
        target_tokens=target_encoding.tokens.tolist(),
    )


def _autocast_context(tensor: torch.Tensor):
    if tensor.device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _require_finite_tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")
    return value


def _frame_metric(output: Mapping[str, Any], name: str, frame: int, frame_count: int) -> float:
    if name not in output:
        raise ValueError(f"model forward output is missing {name}")
    array = _cpu_numpy(output[name], name)
    if array.shape == (1, frame_count):
        value = float(array[0, frame])
    elif array.shape == (frame_count,):
        value = float(array[frame])
    else:
        raise ValueError(f"model forward output {name} must have shape (1,{frame_count})")
    if not math.isfinite(value):
        raise ValueError(f"model forward output {name} contains a non-finite value")
    return value


def _spatial_edge_f1(
    predicted_joints: np.ndarray,
    predicted_parents: Sequence[int],
    target_joints: np.ndarray,
    target_parents: Sequence[int],
) -> float:
    # This is deliberately delegated to the established nearest-joint mapping
    # implementation, rather than replacing it with Hungarian matching.
    value = float(
        _topology_metrics(
            np.asarray(predicted_joints, dtype=np.float32),
            [int(parent) for parent in predicted_parents],
            np.asarray(target_joints, dtype=np.float32),
            [int(parent) for parent in target_parents],
        )["edge_f1"]
    )
    if not math.isfinite(value):
        raise ValueError("spatial topology metric is non-finite")
    return value


def _dfs_edge_f1(predicted_parents: Sequence[int], target_parents: Sequence[int]) -> float:
    """Return directed DFS-ID edge F1 without spatial matching."""

    def edges(parents: Sequence[int]) -> set[tuple[int, int]]:
        return {
            (int(parent), int(child))
            for child, parent in enumerate(parents)
            if int(parent) >= 0
        }

    predicted = edges(predicted_parents)
    target = edges(target_parents)
    if not predicted and not target:
        return 1.0
    if not predicted or not target:
        return 0.0
    return float(2.0 * len(predicted & target) / (len(predicted) + len(target)))


def _j2j(predicted_joints: np.ndarray, target_joints: np.ndarray) -> float:
    predicted = np.asarray(predicted_joints, dtype=np.float32)
    target = np.asarray(target_joints, dtype=np.float32)
    if predicted.ndim != 2 or target.ndim != 2 or predicted.shape[1:] != (3,) or target.shape[1:] != (3,):
        raise ValueError("joints must have shape (J,3)")
    if predicted.shape[0] == 0 or target.shape[0] == 0:
        raise ValueError("j2j requires non-empty joint sets")
    distances = np.linalg.norm(predicted[:, None, :] - target[None, :, :], axis=-1)
    value = float(0.5 * (distances.min(axis=0).mean() + distances.min(axis=1).mean()))
    if not math.isfinite(value):
        raise ValueError("j2j is non-finite")
    return value


def _reference_rows(
    geometry: Mapping[str, Any],
    original_tokenizer: Any,
    codec: AuxSkeletonTokens,
    *,
    group: Any,
    asset_id: str,
    frame_indices: Sequence[int],
    raw_frame_indices: np.ndarray,
    reference_lookup: Mapping[Any, Any],
) -> dict[tuple[Any, str, int], float]:
    reference_targets = build_xyz_aux_targets(geometry, original_tokenizer, codec, device="cpu")
    input_ids = reference_targets["main_input_ids"]
    attention = reference_targets["main_attention_mask"]
    if not isinstance(input_ids, torch.Tensor) or not isinstance(attention, torch.Tensor):
        raise ValueError("reference tokenizer adapter returned non-tensor main targets")
    if input_ids.ndim != 3 or attention.shape != input_ids.shape or input_ids.shape[0] != 1:
        raise ValueError("reference tokenizer adapter returned invalid main target shapes")

    results: dict[tuple[Any, str, int], float] = {}
    for frame in frame_indices:
        key = (group, asset_id, int(frame))
        if key not in reference_lookup:
            raise ValueError(f"reference lookup is missing row {key!r}")
        reference_row = reference_lookup[key]
        if not isinstance(reference_row, Mapping):
            raise ValueError(f"reference lookup row {key!r} must be a mapping")
        valid = attention[0, frame].bool()
        tokens = input_ids[0, frame][valid].detach().cpu().tolist()
        if (
            "raw_frame_index" not in reference_row
            or int(reference_row["raw_frame_index"]) != int(raw_frame_indices[frame])
        ):
            raise ValueError(f"reference raw_frame_index mismatch for {key!r}")
        if "target_tokens" not in reference_row or list(reference_row["target_tokens"]) != tokens:
            raise ValueError(f"reference target_tokens mismatch for {key!r}")
        if "original_edge_f1" not in reference_row:
            raise ValueError(f"reference row {key!r} is missing original_edge_f1")
        score = float(reference_row["original_edge_f1"])
        if not math.isfinite(score):
            raise ValueError(f"reference row {key!r} has non-finite original_edge_f1")
        results[key] = score
    return results


def _summary(rows: list[dict[str, Any]], structure_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows or not structure_rows:
        raise ValueError("evaluation produced no rows")
    failures = sum(not bool(row["success"]) for row in rows)
    success_j2j = [float(row["j2j"]) for row in rows if row["j2j"] is not None]

    def mean(name: str) -> float:
        values = [float(row[name]) for row in rows]
        value = float(np.mean(values))
        if not math.isfinite(value):
            raise ValueError(f"summary metric {name} is non-finite")
        return value

    summary = {
        "asset_count": len(structure_rows),
        "score_frame_count": len(rows),
        "success_count": len(rows) - failures,
        "failure_count": failures,
        "success": float(np.mean([bool(row["success"]) for row in rows])),
        "mean_original_edge_f1": mean("original_edge_f1"),
        "mean_gt_structure_xyz_ce": mean("gt_structure_xyz_ce"),
        "mean_gt_structure_xyz_accuracy": mean("gt_structure_xyz_accuracy"),
        "mean_gt_structure_xyz_l1": mean("gt_structure_xyz_l1"),
        "mean_gt_structure_edge_f1": mean("gt_structure_edge_f1"),
        "mean_j2j": float(np.mean(success_j2j)) if len(success_j2j) == len(rows) else None,
        "mean_j2j_successful": float(np.mean(success_j2j)) if success_j2j else None,
        "j2j_coverage": len(success_j2j) / len(rows),
        "mean_structure_dfs_edge_f1": float(np.mean([row["dfs_edge_f1"] for row in structure_rows])),
        "exact_count": float(np.mean([bool(row["exact_count"]) for row in rows])),
        "counts": {
            "assets": len(structure_rows),
            "rows": len(rows),
            "successes": len(rows) - failures,
            "failures": failures,
        },
    }
    summary.update(
        {
            "gt_structure_xyz_ce": summary["mean_gt_structure_xyz_ce"],
            "gt_structure_xyz_accuracy": summary["mean_gt_structure_xyz_accuracy"],
            "gt_structure_xyz_l1": summary["mean_gt_structure_xyz_l1"],
            "original_edge_f1": summary["mean_original_edge_f1"],
        }
    )
    if any("reference_original_edge_f1" in row for row in rows):
        summary["mean_reference_original_edge_f1"] = float(
            np.mean([float(row["reference_original_edge_f1"]) for row in rows])
        )
    return summary


def summarize_fill_rows(
    rows: Iterable[Mapping[str, Any]],
    structure_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate evaluator rows for local or cross-rank reporting."""

    return _summary(list(rows), list(structure_rows))


def evaluate_fill_items(
    model: Any,
    items: Iterable[tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]],
    codec: AuxSkeletonTokens,
    *,
    frame_indices: Sequence[int],
    group: Any,
    original_tokenizer: Any = None,
    reference_lookup: Mapping[Any, Any] | None = None,
    structure_weight: float = 1.0,
    bins: int = 256,
    continuous_range: Sequence[float] = (-1.0, 1.0),
) -> dict[str, Any]:
    """Evaluate GT-conditioned filling and one free structure generation per asset."""

    if reference_lookup is not None and original_tokenizer is None:
        raise ValueError("original_tokenizer is required when reference_lookup is supplied")
    if reference_lookup is not None and not isinstance(reference_lookup, Mapping):
        raise ValueError("reference_lookup must be a mapping")
    if not _strict_integer(bins) or int(bins) <= 0:
        raise ValueError("bins must be a positive integer")
    bin_count = int(bins)
    coord_range = _continuous_range(continuous_range)
    weight = _finite_positive(structure_weight, "structure_weight")
    selected_frames = _validate_frame_indices(frame_indices, None)
    if model is None or not callable(getattr(model, "encode", None)):
        raise ValueError("model must expose encode")

    rows: list[dict[str, Any]] = []
    structure_rows: list[dict[str, Any]] = []
    seen_asset_ids: set[str] = set()
    used_reference_keys: set[Any] = set()
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        with torch.no_grad():
            for item_index, item in enumerate(items):
                try:
                    surface, targets, geometry = item
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"evaluation item {item_index} must be a (surface, targets, geometry) tuple"
                    ) from exc
                prepared = _prepare_item(surface, targets, geometry, codec, bins=bin_count)
                item_frame_indices = _validate_frame_indices(
                    selected_frames,
                    int(prepared.geometry_xyz.shape[0]),
                )
                if prepared.asset_id in seen_asset_ids:
                    raise ValueError(f"duplicate asset_id in evaluation items: {prepared.asset_id}")
                seen_asset_ids.add(prepared.asset_id)
                if reference_lookup is not None:
                    reference_scores = _reference_rows(
                        geometry,
                        original_tokenizer,
                        codec,
                        group=group,
                        asset_id=prepared.asset_id,
                        frame_indices=item_frame_indices,
                        raw_frame_indices=prepared.raw_frame_indices,
                        reference_lookup=reference_lookup,
                    )
                    used_reference_keys.update(reference_scores)
                else:
                    reference_scores = {}

                query_points = surface["query_points"]
                with _autocast_context(query_points):
                    encoded = model.encode(surface)
                    if not isinstance(encoded, (tuple, list)) or len(encoded) != 2:
                        raise ValueError("model.encode must return (final_frames, virtual)")
                    final_frames, virtual = encoded
                    final_frames = _require_finite_tensor(final_frames, "model final_frames")
                    virtual = _require_finite_tensor(virtual, "model virtual")
                    if (
                        final_frames.ndim < 2
                        or final_frames.shape[0] != 1
                        or final_frames.shape[1] != prepared.geometry_xyz.shape[0]
                    ):
                        raise ValueError("model final_frames must contain every item input frame")
                    if virtual.shape[0] != 1:
                        raise ValueError("model virtual must have batch size one")

                    selected_final_frames = final_frames[:, item_frame_indices]
                    selected_points = query_points[:, item_frame_indices]
                    selected_targets = dict(targets)
                    for key in ("xyz_labels", "xyz_targets", "frame_mask"):
                        selected_targets[key] = targets[key][:, item_frame_indices]
                    forward = model.forward_encoded(
                        selected_final_frames,
                        virtual,
                        selected_points,
                        selected_targets,
                        structure_weight=weight,
                        return_logits=True,
                    )
                    if not isinstance(forward, Mapping) or "xyz_logits" not in forward:
                        raise ValueError("model.forward_encoded must return xyz_logits and per-frame metrics")
                    xyz_logits = _require_finite_tensor(forward["xyz_logits"], "model xyz_logits")
                    if (
                        xyz_logits.ndim != 5
                        or xyz_logits.shape[0] != 1
                        or xyz_logits.shape[1] != len(item_frame_indices)
                        or xyz_logits.shape[2] != prepared.geometry_parents.size
                        or xyz_logits.shape[3] != 3
                        or xyz_logits.shape[4] != bin_count
                    ):
                        raise ValueError("model xyz_logits shape does not match selected fill targets")
                    generation = model.generate_structure(virtual)

                if not isinstance(generation, Mapping) or "success" not in generation or "tokens" not in generation:
                    raise ValueError("model.generate_structure returned an invalid result")
                generated_success = bool(generation["success"])
                generated_tokens = [int(value) for value in list(generation["tokens"])]
                generated_parents: list[int] | None = None
                predicted_coordinates: np.ndarray | None = None
                if generated_success:
                    generated_parents_array = np.asarray(codec.decode(generated_tokens), dtype=np.int64)
                    generated_parents = generated_parents_array.tolist()
                    if "parents" in generation and generation["parents"] is not None:
                        if [int(value) for value in list(generation["parents"])] != generated_parents:
                            raise ValueError("model generation parents disagree with codec.decode")
                    generated_fill = build_fill_inputs(
                        [np.asarray(generated_tokens, dtype=np.int64)],
                        codec,
                        device=query_points.device,
                        max_fill_tokens=model.config.fill_max_tokens,
                    )
                    with _autocast_context(query_points):
                        free_logits = model.fill_decoder(selected_final_frames, selected_points, generated_fill)
                    free_logits = _require_finite_tensor(free_logits, "free-generation xyz_logits")
                    if (
                        free_logits.ndim != 5
                        or free_logits.shape[0] != 1
                        or free_logits.shape[1] != len(item_frame_indices)
                        or free_logits.shape[2] != len(generated_parents)
                        or free_logits.shape[3] != 3
                        or free_logits.shape[4] != bin_count
                    ):
                        raise ValueError("free-generation xyz_logits shape does not match generated structure")
                    predicted_coordinates = decode_coordinate_tokens(
                        free_logits.argmax(dim=-1).detach().cpu().numpy(),
                        bins=bin_count,
                        continuous_range=coord_range,
                    )[0]

                target_labels = prepared.target_labels[item_frame_indices]
                gt_coordinates = decode_coordinate_tokens(
                    xyz_logits.argmax(dim=-1).detach().cpu().numpy(),
                    bins=bin_count,
                    continuous_range=coord_range,
                )[0]
                structure_status = str(generation.get("status", "eos" if generated_success else "generation_failed"))
                dfs_f1 = (
                    0.0
                    if not generated_success
                    else _dfs_edge_f1(generated_parents or [], prepared.target_fill_parents)
                )
                structure_row = {
                    "asset_id": prepared.asset_id,
                    "group": group,
                    "status": structure_status,
                    "success": generated_success,
                    "generated_tokens": generated_tokens,
                    "target_tokens": prepared.target_tokens,
                    "parents": generated_parents,
                    "generated_parents": generated_parents,
                    "target_parents": prepared.target_fill_parents.tolist(),
                    "dfs_edge_f1": float(dfs_f1),
                    "dfs_edge_f1_definition": "DFS-ID directed-edge diagnostic; not main spatial F1",
                }
                structure_rows.append(structure_row)

                for score_position, frame_index in enumerate(item_frame_indices):
                    original_joints = prepared.geometry_xyz[frame_index]
                    gt_joints = gt_coordinates[score_position]
                    gt_ce = _frame_metric(
                        forward,
                        "xyz_loss_per_frame",
                        score_position,
                        len(item_frame_indices),
                    )
                    gt_accuracy = _frame_metric(
                        forward,
                        "xyz_accuracy_per_frame",
                        score_position,
                        len(item_frame_indices),
                    )
                    gt_l1 = float(np.abs(gt_joints - prepared.target_xyz_dfs[frame_index]).mean())
                    if not math.isfinite(gt_l1):
                        raise ValueError("GT-conditioned XYZ L1 is non-finite")
                    if generated_success:
                        assert predicted_coordinates is not None and generated_parents is not None
                        predicted_joints = predicted_coordinates[score_position]
                        original_edge_f1 = _spatial_edge_f1(
                            predicted_joints,
                            generated_parents,
                            original_joints,
                            prepared.geometry_parents,
                        )
                        j2j = _j2j(predicted_joints, original_joints)
                        exact_count = len(generated_parents) == prepared.geometry_parents.size
                        predicted_joint_list: list[list[float]] | None = predicted_joints.tolist()
                        predicted_parent_list: list[int] | None = list(generated_parents)
                    else:
                        original_edge_f1 = 0.0
                        j2j = None
                        exact_count = False
                        predicted_joint_list = None
                        predicted_parent_list = None

                    row: dict[str, Any] = {
                        "asset_id": prepared.asset_id,
                        "group": group,
                        "frame_index": int(frame_index),
                        "raw_frame_index": int(prepared.raw_frame_indices[frame_index]),
                        "input_anchor_points": query_points[0, frame_index].float().detach().cpu().tolist(),
                        "status": structure_status,
                        "success": generated_success,
                        "original_edge_f1": float(original_edge_f1),
                        "j2j": j2j,
                        "exact_count": bool(exact_count),
                        "gt_structure_xyz_ce": gt_ce,
                        "gt_structure_xyz_accuracy": gt_accuracy,
                        "gt_structure_xyz_l1": gt_l1,
                        "gt_structure_edge_f1": _spatial_edge_f1(
                            gt_joints,
                            prepared.target_fill_parents,
                            original_joints,
                            prepared.geometry_parents,
                        ),
                        "original_target_joints": original_joints.tolist(),
                        "original_target_parents": prepared.geometry_parents.tolist(),
                        "gt_structure_predicted_joints": gt_joints.tolist(),
                        "gt_structure_predicted_parents": prepared.target_fill_parents.tolist(),
                        "predicted_joints": predicted_joint_list,
                        "predicted_parents": predicted_parent_list,
                        "fill_source_indices": prepared.fill_source_indices.tolist(),
                        "target_tokens": prepared.target_tokens,
                    }
                    key = (group, prepared.asset_id, int(frame_index))
                    if reference_lookup is not None:
                        row["reference_original_edge_f1"] = reference_scores[key]
                        used_reference_keys.add(key)
                    rows.append(row)
    finally:
        model.train(was_training)

    if not rows:
        raise ValueError("evaluation items must contain at least one asset")
    if reference_lookup is not None:
        expected_keys = {
            (group, row["asset_id"], int(row["frame_index"]))
            for row in rows
        }
        actual_keys = set(reference_lookup.keys())
        if actual_keys != expected_keys or used_reference_keys != expected_keys:
            raise ValueError("reference lookup coverage does not exactly match evaluated rows")
    return {
        "rows": rows,
        "structure_rows": structure_rows,
        "summary": summarize_fill_rows(rows, structure_rows),
    }


__all__ = ["evaluate_fill_items", "summarize_fill_rows"]
