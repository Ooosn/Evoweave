"""Explicit raw-frame selection for the bounded shared-structure experiment."""
from __future__ import annotations

from numbers import Integral
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .data import _face_normals, _vertex_normals
from .shared_structure_data import _require_integer_array, _validate_target_parents


FrameIds = Sequence[int] | np.ndarray | torch.Tensor
_TARGET_AFFINES = (
    "target_mesh_centers", "target_mesh_scales", "target_source_centers", "target_source_scales",
)


def _integer(value: int, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _frame_ids(value: FrameIds, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.dtype not in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}:
            raise ValueError(f"{name} must be a one-dimensional integer array")
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer) or not array.size:
        raise ValueError(f"{name} must be a non-empty one-dimensional integer array")
    if np.any(array < 0) or np.any(array > np.iinfo(np.int64).max):
        raise ValueError(f"{name} must contain nonnegative int64 frame IDs")
    if np.unique(array).size != array.size:
        raise ValueError(f"{name} must contain unique frame IDs")
    return array.astype(np.int64, copy=True)


def partition_raw_frames(
    frame_count: int, seed: int, heldout_frames: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint ordered NPZ row IDs, with at least 24 train and 8 heldout.

    The caller supplies an immutable per-asset seed. No geometry, labels, query
    selection, FPS, or global random state participates in this partition.
    """
    count = _integer(frame_count, "frame_count", 1)
    seed = _integer(seed, "seed", 0)
    heldout = _integer(heldout_frames, "heldout_frames", 8)
    if count - heldout < 24:
        raise ValueError("partition requires at least 24 train frames and 8 heldout frames")
    order = np.random.default_rng(seed).permutation(count).astype(np.int64)
    return order[:-heldout].copy(), order[-heldout:].copy()


def source_bank_indices(train_ids: FrameIds, bank: int) -> np.ndarray:
    """Return the same caller-provided eligible IDs with two different queries."""
    ids = _frame_ids(train_ids, "train_ids")
    bank = _integer(bank, "bank", 0)
    if bank not in (0, 1):
        raise ValueError("bank must be 0 or 1")
    if ids.size < 2:
        raise ValueError("two source banks require at least two eligible frames")
    return ids if bank == 0 else np.roll(ids, -(ids.size // 2))


def _float_array(value: Any, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim or not (
        np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.integer)
    ):
        raise ValueError(f"{name} must be a real numeric array with ndim={ndim}")
    with np.errstate(over="ignore", invalid="ignore"):
        array = array.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite float32 values")
    return array


def _joint_flags(value: Any, name: str, joints: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (joints,) or not (
        np.issubdtype(array.dtype, np.bool_) or np.issubdtype(array.dtype, np.integer)
    ) or not np.isin(array, (0, 1)).all():
        raise ValueError(f"{name} must be a boolean vector with one entry per target joint")
    return array.astype(bool, copy=True)


def load_medium_batch(
    path: str | Path, frame_ids: FrameIds, cls_name: str = "articulationxl",
) -> dict[str, Any]:
    """Read a complete rootless-v3 target tree into an unpadded CPU B=1 batch.

    ``frame_ids`` are original NPZ row IDs in the requested order, not source
    animation frame numbers. The first selected mesh alone defines the common
    query chart. No legacy tokenizer, joint filtering, or GT bounds are used.
    """
    ids = _frame_ids(frame_ids, "frame_ids")
    path = Path(path).expanduser()
    if not isinstance(cls_name, str) or not cls_name:
        raise ValueError("cls_name must be a non-empty string")
    required = (
        "canonical_schema_version", "frame_vertices_rootspace", "faces",
        "target_joints_rootspace", "target_parents", "target_has_skin", "target_is_synthetic_root",
    )
    # Canonical files store the scalar schema string as a NumPy object array.
    with np.load(path, allow_pickle=True) as raw:
        missing = [key for key in required if key not in raw.files]
        if missing:
            raise KeyError(f"{path} missing canonical fields: {missing}")
        schema = np.asarray(raw["canonical_schema_version"])
        if schema.size != 1 or schema.item() != "rootless_dynamic_npz_v3":
            raise ValueError(f"{path} requires canonical_schema_version='rootless_dynamic_npz_v3'")
        vertices = _float_array(raw["frame_vertices_rootspace"], "frame_vertices_rootspace", 3)
        if vertices.shape[-1] != 3 or min(vertices.shape[:2]) < 1:
            raise ValueError("frame_vertices_rootspace must have non-empty shape (F,V,3)")
        total_frames, vertex_count, _ = vertices.shape
        if np.any(ids >= total_frames):
            raise ValueError("frame_ids contains a row outside the NPZ frame range")
        faces = np.asarray(raw["faces"])
        if faces.ndim != 2 or faces.shape[1] != 3 or not faces.shape[0] or not np.issubdtype(faces.dtype, np.integer):
            raise ValueError("faces must be a non-empty integer array with shape (M,3)")
        if np.any(faces < 0) or np.any(faces >= vertex_count):
            raise ValueError("faces contains an index outside the actual vertex range")
        faces = faces.astype(np.int64, copy=True)
        targets = _float_array(raw["target_joints_rootspace"], "target_joints_rootspace", 3)
        if targets.shape[0] != total_frames or targets.shape[-1] != 3 or targets.shape[1] < 1:
            raise ValueError("target_joints_rootspace must have shape (F,J,3) aligned with all mesh frames")
        joints = targets.shape[1]
        raw_parents = np.asarray(raw["target_parents"])
        if not np.issubdtype(raw_parents.dtype, np.signedinteger):
            raise ValueError("target_parents must use signed integers, including the root parent -1")
        parents = _require_integer_array(raw_parents, "target_parents", path).copy()
        if parents.shape != (joints,):
            raise ValueError("target_parents must match the complete target joint count")
        _validate_target_parents(parents, path)
        has_skin = _joint_flags(raw["target_has_skin"], "target_has_skin", joints)
        synthetic = _joint_flags(raw["target_is_synthetic_root"], "target_is_synthetic_root", joints)
        if synthetic.any():
            raise ValueError("target_is_synthetic_root must be false for every real target joint")
        frame_numbers = None
        if "frame_numbers" in raw.files:
            frame_numbers = _require_integer_array(raw["frame_numbers"], "frame_numbers", path)
            if frame_numbers.shape != (total_frames,):
                raise ValueError("frame_numbers must align with all NPZ frame rows")
            frame_numbers = frame_numbers[ids].copy()
        selected_vertices, selected_targets = vertices[ids], targets[ids]

    query = selected_vertices[0]
    lo, hi = query.min(axis=0), query.max(axis=0)
    center = (lo + hi) * 0.5
    scale = np.max((hi - lo) * 0.5)
    if not np.isfinite(center).all() or not np.isfinite(scale) or scale < 1.0e-8:
        raise ValueError("first selected query mesh must have finite non-degenerate bounds")
    with np.errstate(over="ignore", invalid="ignore"):
        vertices_n = ((selected_vertices - center) / scale).astype(np.float32)
        targets_n = ((selected_targets - center) / scale).astype(np.float32)
    if not np.isfinite(vertices_n).all() or not np.isfinite(targets_n).all():
        raise ValueError("query normalization produced non-finite geometry or targets")
    face_normals = np.stack([_face_normals(frame, faces) for frame in vertices_n])
    vertex_normals = np.stack([
        _vertex_normals(frame, faces, normals) for frame, normals in zip(vertices_n, face_normals)
    ])
    if not np.isfinite(face_normals).all() or not np.isfinite(vertex_normals).all():
        raise ValueError("mesh normals must be finite")
    batch = {
        "path": [str(path)], "cls": [cls_name],
        "frame_vertices": torch.from_numpy(vertices_n)[None],
        "faces": torch.from_numpy(faces)[None],
        "vertex_normals": torch.from_numpy(vertex_normals)[None],
        "face_normals": torch.from_numpy(face_normals)[None],
        "vertex_count": torch.tensor([vertex_count], dtype=torch.long),
        "face_count": torch.tensor([len(faces)], dtype=torch.long),
        "selected_frames": torch.from_numpy(ids)[None],
        "query_center": torch.from_numpy(center.copy())[None],
        "query_scale": torch.tensor([float(scale)], dtype=torch.float32),
        "frame_target_joints": torch.from_numpy(targets_n)[None],
        "target_joints": torch.from_numpy(targets_n[0].copy())[None],
        "target_parents": torch.from_numpy(parents)[None],
        "joint_count": torch.tensor([joints], dtype=torch.long),
        "joint_mask": torch.ones(1, joints, dtype=torch.bool),
        "frame_mask": torch.ones(1, len(ids), dtype=torch.bool),
        "target_has_skin": torch.from_numpy(has_skin)[None],
        "target_is_synthetic_root": torch.from_numpy(synthetic)[None],
    }
    if frame_numbers is not None:
        batch["frame_numbers"] = torch.from_numpy(frame_numbers)[None]
    return batch


def _validate_source_batch(batch: dict[str, Any]) -> np.ndarray:
    if any(key in batch for key in _TARGET_AFFINES):
        raise ValueError("source view requires a common query chart, not per-mesh target normalization")
    required = (
        "frame_vertices", "faces", "vertex_normals", "face_normals", "vertex_count", "face_count",
        "selected_frames", "query_center", "query_scale", "frame_target_joints", "target_joints",
        "target_parents", "joint_count", "joint_mask", "frame_mask", "path", "cls",
    )
    missing = [key for key in required if key not in batch]
    if missing:
        raise ValueError(f"source batch missing fields: {missing}")
    vertices, targets, faces = batch["frame_vertices"], batch["frame_target_joints"], batch["faces"]
    for value, name in ((vertices, "frame_vertices"), (targets, "frame_target_joints")):
        if not isinstance(value, torch.Tensor) or value.ndim != 4 or value.shape[0] != 1 or value.shape[-1] != 3:
            raise ValueError(f"{name} must have unpadded shape (1,T,N,3)")
    _, frames, vertex_count, _ = vertices.shape
    joints = targets.shape[2]
    if min(frames, vertex_count, joints) < 1 or targets.shape[1] != frames:
        raise ValueError("source geometry and complete targets must have matching non-empty frames")
    if not isinstance(faces, torch.Tensor) or faces.ndim != 3 or faces.shape[0] != 1 or faces.shape[2] != 3 or not faces.shape[1]:
        raise ValueError("faces must have unpadded shape (1,M,3)")
    float_shapes = {
        "frame_vertices": (1, frames, vertex_count, 3), "frame_target_joints": (1, frames, joints, 3),
        "vertex_normals": (1, frames, vertex_count, 3), "face_normals": (1, frames, faces.shape[1], 3),
        "target_joints": (1, joints, 3), "query_center": (1, 3), "query_scale": (1,),
    }
    for name, shape in float_shapes.items():
        value = batch[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not torch.is_floating_point(value):
            raise ValueError(f"{name} must be a floating tensor with shape {shape}")
        if value.device != vertices.device or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite and on the geometry device")
    integer_shapes = {
        "faces": tuple(faces.shape), "selected_frames": (1, frames), "target_parents": (1, joints),
        "vertex_count": (1,), "face_count": (1,), "joint_count": (1,),
    }
    if "frame_numbers" in batch:
        integer_shapes["frame_numbers"] = (1, frames)
    for name, shape in integer_shapes.items():
        value = batch[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != torch.long:
            raise ValueError(f"{name} must be an int64 tensor with shape {shape}")
        if value.device != vertices.device:
            raise ValueError(f"{name} must be on the geometry device")
    for name, count in (("vertex_count", vertex_count), ("face_count", faces.shape[1]), ("joint_count", joints)):
        if int(batch[name][0]) != count:
            raise ValueError(f"{name} does not match the unpadded source batch")
    if bool(((faces < 0) | (faces >= vertex_count)).any()) or not bool((batch["query_scale"] > 0).all()):
        raise ValueError("source face indices or query scale are invalid")
    for name, shape in (("joint_mask", (1, joints)), ("frame_mask", (1, frames))):
        value = batch[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != torch.bool:
            raise ValueError(f"{name} must be a boolean tensor with shape {shape}")
        if value.device != vertices.device or not bool(value.all()):
            raise ValueError("source views require complete joints and unpadded frames")
    for name in ("path", "cls"):
        if not isinstance(batch[name], list) or len(batch[name]) != 1 or not isinstance(batch[name][0], str):
            raise ValueError(f"{name} must contain one string for the single asset")
    _validate_target_parents(batch["target_parents"][0].detach().cpu().numpy(), Path(batch["path"][0]))
    if not torch.allclose(batch["target_joints"], targets[:, 0], atol=1e-6, rtol=1e-5):
        raise ValueError("target_joints does not match the first selected frame")
    return _frame_ids(batch["selected_frames"][0], "selected_frames")


def select_source_view(batch: dict[str, Any], raw_frame_ids: FrameIds) -> dict[str, Any]:
    """Select NPZ row IDs and rebase a single-asset source view without mutation.

    Input must be a complete, unpadded batch in one common query chart. The
    caller owns train/heldout eligibility. Positive isotropic rebasing preserves
    selected normals; target mesh charts must be built independently afterward.
    """
    if not isinstance(batch, dict):
        raise TypeError("batch must be a dictionary")
    available = _validate_source_batch(batch)
    requested = _frame_ids(raw_frame_ids, "raw_frame_ids")
    slots = {int(raw_id): slot for slot, raw_id in enumerate(available)}
    if any(int(raw_id) not in slots for raw_id in requested):
        raise ValueError("raw_frame_ids contains a frame not present in the source batch")
    device = batch["frame_vertices"].device
    indices = torch.tensor([slots[int(raw_id)] for raw_id in requested], dtype=torch.long, device=device)
    output = dict(batch)
    for key in ("frame_vertices", "vertex_normals", "face_normals", "frame_target_joints",
                "selected_frames", "frame_mask", "frame_numbers"):
        if key in batch:
            output[key] = batch[key].index_select(1, indices)
    query = output["frame_vertices"][:, 0]
    lo, hi = query.amin(dim=1), query.amax(dim=1)
    local_center = (lo + hi) * 0.5
    local_scale = ((hi - lo) * 0.5).amax(dim=-1)
    source_center = batch["query_center"] + batch["query_scale"][:, None] * local_center
    source_scale = batch["query_scale"] * local_scale
    if not bool(torch.isfinite(source_center).all()) or not bool(torch.isfinite(source_scale).all()) or not bool((source_scale >= 1e-8).all()):
        raise ValueError("selected query mesh must have finite non-degenerate source bounds")
    for key in ("frame_vertices", "frame_target_joints"):
        output[key] = (output[key] - local_center[:, None, None, :]) / local_scale[:, None, None, None]
        if not bool(torch.isfinite(output[key]).all()):
            raise ValueError(f"rebased {key} must be finite")
    output["target_joints"] = output["frame_target_joints"][:, 0].clone()
    output["query_center"], output["query_scale"] = source_center, source_scale
    return output
