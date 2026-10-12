from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import DynamicRigManifestDataset, DynamicRigSample, dynamic_rig_collate
from .frame_batch_sampler import FrameSampleRequest


_FLOAT32_RTOL = 1.0e-5
_FLOAT32_ATOL = 1.0e-6


@dataclass(frozen=True)
class SharedStructureSample:
    base: DynamicRigSample
    frame_target_joints: torch.Tensor


def _as_numpy(value: Any, *, dtype: np.dtype | None = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _require_integer_array(value: Any, name: str, path: Path) -> np.ndarray:
    array = _as_numpy(value)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{path} {name} must be a one-dimensional integer array, got {array.shape} {array.dtype}")
    return array.astype(np.int64, copy=False)


def _validate_target_parents(parents: np.ndarray, path: Path) -> None:
    if parents.size == 0 or int(parents[0]) != -1:
        raise ValueError(f"{path} target_parents must have exactly one root at index 0")
    if np.any(parents[1:] < 0):
        raise ValueError(f"{path} target_parents must have exactly one root at index 0")
    child_indices = np.arange(parents.size, dtype=np.int64)
    if np.any(parents[1:] >= child_indices[1:]):
        raise ValueError(f"{path} target_parents must satisfy parent_index < child_index")


def _load_frame_target_joints(base: DynamicRigSample) -> torch.Tensor:
    if getattr(base, "root_rt", None) is not None:
        raise ValueError("legacy SharedStructureDataset does not support root_rt; use VirtualFrameDataset")
    path = Path(base.path)
    selected = _require_integer_array(base.selected_frames, "selected_frames", path)
    frame_vertices = _as_numpy(base.frame_vertices)
    if frame_vertices.ndim < 1:
        raise ValueError(f"{path} base frame_vertices must have a frame dimension")
    if selected.size == 0 or selected.shape[0] != frame_vertices.shape[0]:
        raise ValueError(
            f"{path} selected_frames length {selected.shape[0]} does not match sampled frames {frame_vertices.shape[0]}"
        )
    if np.any(selected < 0):
        raise ValueError(f"{path} selected_frames contains a negative frame index")
    if np.unique(selected).size != selected.size:
        raise ValueError(f"{path} selected_frames must contain unique frame indices")

    base_parents = _require_integer_array(base.target_parents, "base target_parents", path)
    base_joints = _as_numpy(base.target_joints, dtype=np.float32)
    if base_joints.ndim != 2 or base_joints.shape[-1] != 3:
        raise ValueError(f"{path} base target_joints must have shape (J, 3), got {base_joints.shape}")
    if int(base.joint_count) != int(base_joints.shape[0]):
        raise ValueError(
            f"{path} base joint_count={base.joint_count} does not match target_joints J={base_joints.shape[0]}"
        )
    if base_parents.shape != (base_joints.shape[0],):
        raise ValueError(
            f"{path} base target_parents shape {base_parents.shape} does not match target joints {base_joints.shape[0]}"
        )
    if not np.isfinite(base_joints).all():
        raise ValueError(f"{path} base target_joints contains non-finite values")

    center = _as_numpy(base.query_center, dtype=np.float32)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError(f"{path} base query_center must be finite with shape (3,), got {center.shape}")
    scale_array = _as_numpy(base.query_scale)
    if scale_array.ndim != 0 or not np.issubdtype(scale_array.dtype, np.number):
        raise ValueError(f"{path} base query_scale must be a numeric scalar, got {scale_array.shape}")
    scale = float(np.asarray(scale_array, dtype=np.float32).item())
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{path} base query_scale must be finite and positive, got {scale}")

    with np.load(path, allow_pickle=False) as raw:
        missing = [
            name for name in ("target_joints_rootspace", "target_parents") if name not in raw.files
        ]
        if missing:
            raise KeyError(f"{path} shared-structure NPZ is missing {missing}")

        raw_targets = np.asarray(raw["target_joints_rootspace"])
        if (
            raw_targets.ndim != 3
            or raw_targets.shape[-1] != 3
            or not np.issubdtype(raw_targets.dtype, np.number)
            or np.issubdtype(raw_targets.dtype, np.complexfloating)
        ):
            raise ValueError(
                f"{path} target_joints_rootspace must have numeric shape (F, J, 3), got {raw_targets.shape} {raw_targets.dtype}"
            )
        raw_targets = raw_targets.astype(np.float32, copy=False)
        if not np.isfinite(raw_targets).all():
            raise ValueError(f"{path} target_joints_rootspace contains non-finite values")

        raw_parents = _require_integer_array(raw["target_parents"], "target_parents", path)
        if raw_parents.shape != (raw_targets.shape[1],):
            raise ValueError(
                f"{path} target_parents shape {raw_parents.shape} does not match target joints {raw_targets.shape[1]}"
            )
        _validate_target_parents(raw_parents, path)
        if raw_parents.shape != base_parents.shape or not np.array_equal(raw_parents, base_parents):
            raise ValueError(f"{path} target_parents does not exactly match base target_parents")

        if np.any(selected >= raw_targets.shape[0]):
            raise ValueError(
                f"{path} selected_frames contains an index outside target frame range [0, {raw_targets.shape[0]})"
            )
        selected_targets = raw_targets[selected]

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized = ((selected_targets - center) / scale).astype(np.float32, copy=False)
    if not np.isfinite(normalized).all():
        raise ValueError(f"{path} normalized frame targets contain non-finite values")
    if not np.allclose(
        normalized[0],
        base_joints,
        rtol=_FLOAT32_RTOL,
        atol=_FLOAT32_ATOL,
    ):
        max_error = float(np.max(np.abs(normalized[0] - base_joints)))
        raise ValueError(
            f"{path} first selected target does not match base.target_joints within float32 tolerance; "
            f"max_abs_error={max_error:.6g}"
        )
    return torch.from_numpy(np.array(normalized, dtype=np.float32, copy=True))


class SharedStructureDataset(Dataset):
    def __init__(self, base_dataset: DynamicRigManifestDataset) -> None:
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int | FrameSampleRequest) -> SharedStructureSample:
        base_sample = self.base_dataset[index]
        return SharedStructureSample(
            base=base_sample,
            frame_target_joints=_load_frame_target_joints(base_sample),
        )

    @property
    def paths(self) -> Any:
        return self.base_dataset.paths

    @property
    def frame_count(self) -> int:
        return self.base_dataset.frame_count

    @property
    def random_query(self) -> bool:
        return self.base_dataset.random_query


def shared_structure_collate(
    batch: list[SharedStructureSample], pad_token: int
) -> dict[str, Any]:
    if not batch:
        raise ValueError("shared_structure_collate requires a non-empty batch")

    base_batch = [item.base for item in batch]
    frame_targets = [item.frame_target_joints for item in batch]
    frame_counts: list[int] = []
    joint_counts: list[int] = []
    for item, targets in zip(batch, frame_targets):
        if not isinstance(targets, torch.Tensor) or targets.ndim != 3 or targets.shape[-1] != 3:
            raise ValueError(
                "frame_target_joints must be a Tensor with shape (T, J, 3), "
                f"got {getattr(targets, 'shape', None)}"
            )
        base_joints = item.base.target_joints
        if base_joints.ndim != 2 or base_joints.shape[-1] != 3:
            raise ValueError("base target_joints must have shape (J, 3)")
        if int(targets.shape[0]) != int(item.base.frame_vertices.shape[0]):
            raise ValueError("frame_target_joints T does not match base sampled frame count")
        if int(targets.shape[1]) != int(base_joints.shape[0]):
            raise ValueError("frame_target_joints J does not match base target_joints")
        frame_counts.append(int(targets.shape[0]))
        joint_counts.append(int(targets.shape[1]))
    if len(set(frame_counts)) != 1:
        raise ValueError("all batch items must have the same sampled frame_count")

    collated = dynamic_rig_collate(base_batch, pad_token=pad_token)
    max_joints = max(joint_counts)
    first_targets = frame_targets[0]
    padded_targets = first_targets.new_zeros(
        (len(batch), frame_counts[0], max_joints, 3)
    )
    joint_mask = torch.zeros(
        (len(batch), max_joints), dtype=torch.bool, device=first_targets.device
    )
    for batch_index, targets in enumerate(frame_targets):
        joint_count = int(targets.shape[1])
        padded_targets[batch_index, :, :joint_count] = targets
        joint_mask[batch_index, :joint_count] = True
    frame_mask = torch.ones(
        (len(batch), frame_counts[0]), dtype=torch.bool, device=first_targets.device
    )
    collated["frame_target_joints"] = padded_targets
    collated["joint_mask"] = joint_mask
    collated["frame_mask"] = frame_mask
    return collated
