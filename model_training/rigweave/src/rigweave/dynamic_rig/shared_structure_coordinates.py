from __future__ import annotations

import math
from numbers import Real
from typing import Any

import torch


def _require_float_tensor(value: Any, name: str, ndim: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != ndim or not torch.is_floating_point(value):
        raise ValueError(f"{name} must be a floating Tensor with ndim={ndim}")
    return value


def _require_same_device(values: dict[str, torch.Tensor]) -> torch.device:
    devices = {value.device for value in values.values()}
    if len(devices) != 1:
        details = ", ".join(f"{name}={value.device}" for name, value in values.items())
        raise ValueError(f"coordinate tensors must share a device, got {details}")
    return next(iter(devices))


def _require_padding(padding: float) -> float:
    if isinstance(padding, bool) or not isinstance(padding, Real):
        raise ValueError(f"padding must be a finite real number >= 1, got {padding!r}")
    value = float(padding)
    if not math.isfinite(value) or value < 1.0:
        raise ValueError(f"padding must be a finite real number >= 1, got {padding!r}")
    return value


def _require_batch_tensors(batch: dict[str, Any]) -> tuple[torch.Tensor | None, ...]:
    if not isinstance(batch, dict):
        raise TypeError("batch must be a dictionary")
    frame_vertices = _require_float_tensor(batch.get("frame_vertices"), "frame_vertices", 4)
    batch_size, frame_count, vertex_slots, coordinates = frame_vertices.shape
    if coordinates != 3 or batch_size < 1 or frame_count < 1 or vertex_slots < 1:
        raise ValueError("frame_vertices must have shape (B, T, V, 3) with positive B/T/V")
    frame_targets = batch.get("frame_target_joints")
    joint_mask = batch.get("joint_mask")
    if (frame_targets is None) != (joint_mask is None):
        raise ValueError("frame_target_joints and joint_mask must be provided together or both omitted")
    if frame_targets is not None:
        frame_targets = _require_float_tensor(frame_targets, "frame_target_joints", 4)
        if not isinstance(joint_mask, torch.Tensor) or joint_mask.ndim != 2 or joint_mask.dtype != torch.bool:
            raise ValueError("joint_mask must be a boolean Tensor with shape (B, J)")
        if frame_targets.shape[:2] != (batch_size, frame_count) or frame_targets.shape[-1] != 3:
            raise ValueError("frame_target_joints must align with frame_vertices as (B, T, J, 3)")
        if joint_mask.shape != (batch_size, frame_targets.shape[2]):
            raise ValueError("joint_mask must align with frame_target_joints as (B, J)")
    elif "target_joints" in batch:
        raise ValueError("target_joints cannot be transformed without frame_target_joints and joint_mask")
    vertex_count = batch.get("vertex_count")
    if not isinstance(vertex_count, torch.Tensor) or vertex_count.ndim != 1 or vertex_count.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError("vertex_count must be a one-dimensional integer Tensor")
    query_center, query_scale = batch.get("query_center"), batch.get("query_scale")
    if (query_center is None) != (query_scale is None):
        raise ValueError("query_center and query_scale must be provided together or both omitted")
    if query_center is None:
        query_center, query_scale = frame_vertices.new_zeros(batch_size, 3), frame_vertices.new_ones(batch_size)
    query_center = _require_float_tensor(query_center, "query_center", 2)
    query_scale = _require_float_tensor(query_scale, "query_scale", 1)
    tensors = {"frame_vertices": frame_vertices, "vertex_count": vertex_count,
               "query_center": query_center, "query_scale": query_scale}
    if frame_targets is not None:
        tensors.update(frame_target_joints=frame_targets, joint_mask=joint_mask)
    _require_same_device(tensors)
    if vertex_count.shape != (batch_size,):
        raise ValueError("vertex_count must have shape (B,)")
    if query_center.shape != (batch_size, 3) or query_scale.shape != (batch_size,):
        raise ValueError("query_center/query_scale must have shapes (B, 3)/(B,)")
    if bool(((vertex_count < 1) | (vertex_count > vertex_slots)).any()):
        raise ValueError("vertex_count must select at least one and at most V actual vertices")
    if not bool(torch.isfinite(query_center).all()) or not bool(torch.isfinite(query_scale).all()):
        raise ValueError("query_center and query_scale must be finite")
    if bool((query_scale <= 0).any()):
        raise ValueError("query_scale must be positive")
    return frame_vertices, frame_targets, vertex_count, joint_mask, query_center, query_scale


def normalize_target_meshes(batch: dict[str, Any], padding: float = 1.1) -> dict[str, Any]:
    """Normalize each selected mesh independently for the local position decoder.

    Targets/masks are optional. With a base query affine, the returned inverse
    maps directly to original source coordinates. With neither affine field,
    the input mesh is treated as source coordinates. Inference requires no GT.
    """

    padding_value = _require_padding(padding)
    (
        frame_vertices,
        frame_targets,
        vertex_count,
        joint_mask,
        query_center,
        query_scale,
    ) = _require_batch_tensors(batch)
    batch_size, frame_count, vertex_slots, _ = frame_vertices.shape
    vertex_indices = torch.arange(vertex_slots, device=frame_vertices.device)[None, None, :]
    valid_vertices = (vertex_indices < vertex_count[:, None, None]).expand(-1, frame_count, -1)
    actual_vertices = frame_vertices[valid_vertices]
    if not bool(torch.isfinite(actual_vertices).all()):
        raise ValueError("actual mesh vertices must be finite")

    zero_vertices = torch.zeros_like(frame_vertices)
    safe_vertices = torch.where(valid_vertices[..., None], frame_vertices, zero_vertices)
    positive_infinity = torch.full_like(safe_vertices, torch.inf)
    negative_infinity = torch.full_like(safe_vertices, -torch.inf)
    mesh_min = torch.where(valid_vertices[..., None], safe_vertices, positive_infinity).amin(dim=2)
    mesh_max = torch.where(valid_vertices[..., None], safe_vertices, negative_infinity).amax(dim=2)
    local_centers = (mesh_min + mesh_max) * 0.5
    half_extent = ((mesh_max - mesh_min) * 0.5).amax(dim=-1)
    local_scales = half_extent * frame_vertices.new_tensor(padding_value)
    if not bool(torch.isfinite(local_centers).all()) or not bool(torch.isfinite(local_scales).all()):
        raise ValueError("actual mesh bounds must be finite")
    if bool((local_scales <= 0).any()):
        raise ValueError("actual meshes must have positive spatial extent")

    normalized_vertices = (safe_vertices - local_centers[:, :, None, :]) / local_scales[:, :, None, None]
    if not bool(torch.isfinite(normalized_vertices[valid_vertices]).all()):
        raise ValueError("normalized actual mesh vertices must be finite")
    normalized_vertices = torch.where(valid_vertices[..., None], normalized_vertices, zero_vertices)

    normalized_targets = None
    if frame_targets is not None:
        valid_joints = joint_mask[:, None, :].expand(batch_size, frame_count, frame_targets.shape[2])
        actual_targets = frame_targets[valid_joints]
        if actual_targets.numel() and not bool(torch.isfinite(actual_targets).all()):
            raise ValueError("valid target joints must be finite")
        zero_targets = torch.zeros_like(frame_targets)
        safe_targets = torch.where(valid_joints[..., None], frame_targets, zero_targets)
        normalized_targets = (safe_targets - local_centers[:, :, None, :]) / local_scales[:, :, None, None]
        if actual_targets.numel() and not bool(torch.isfinite(normalized_targets[valid_joints]).all()):
            raise ValueError("normalized valid target joints must be finite")
        normalized_targets = torch.where(valid_joints[..., None], normalized_targets, zero_targets)

    source_centers = query_center[:, None, :] + query_scale[:, None, None] * local_centers
    source_scales = query_scale[:, None] * local_scales
    if not bool(torch.isfinite(source_centers).all()) or not bool(torch.isfinite(source_scales).all()):
        raise ValueError("source mesh affine must be finite")
    if bool((source_scales <= 0).any()):
        raise ValueError("source mesh scales must be positive")

    output = dict(batch)
    output["frame_vertices"] = normalized_vertices
    if normalized_targets is not None:
        output["frame_target_joints"] = normalized_targets
        output["target_joints"] = normalized_targets[:, 0]
    output["target_mesh_centers"] = local_centers
    output["target_mesh_scales"] = local_scales
    output["target_source_centers"] = source_centers
    output["target_source_scales"] = source_scales
    return output


def restore_target_coordinates(
    local_xyz: torch.Tensor,
    source_centers: torch.Tensor,
    source_scales: torch.Tensor,
) -> torch.Tensor:
    """Restore local decoder coordinates to original NPZ source coordinates."""

    local_xyz = _require_float_tensor(local_xyz, "local_xyz", 4)
    source_centers = _require_float_tensor(source_centers, "source_centers", 3)
    source_scales = _require_float_tensor(source_scales, "source_scales", 2)
    _require_same_device(
        {
            "local_xyz": local_xyz,
            "source_centers": source_centers,
            "source_scales": source_scales,
        }
    )
    batch_size, frame_count, joint_slots, coordinates = local_xyz.shape
    if coordinates != 3 or source_centers.shape != (batch_size, frame_count, 3):
        raise ValueError("source_centers must have shape (B, T, 3) for local_xyz (B, T, J, 3)")
    if source_scales.shape != (batch_size, frame_count):
        raise ValueError("source_scales must have shape (B, T)")
    if not bool(torch.isfinite(local_xyz).all()):
        raise ValueError("local_xyz must be finite")
    if not bool(torch.isfinite(source_centers).all()) or not bool(torch.isfinite(source_scales).all()):
        raise ValueError("source affine must be finite")
    if bool((source_scales <= 0).any()):
        raise ValueError("source_scales must be positive")
    del joint_slots
    return local_xyz * source_scales[:, :, None, None] + source_centers[:, :, None, :]
