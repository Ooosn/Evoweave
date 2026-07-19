from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from rigweave.dynamic_rig.data import (
    DynamicRigManifestDataset,
    DynamicRigSample,
    dynamic_rig_collate,
)

from .tokenizer import StackCloseSerialization, StackCloseTokenizer


@dataclass(frozen=True)
class StackCloseSample:
    core: DynamicRigSample
    coordinate_token_positions: torch.LongTensor
    perturb_axes: torch.Tensor
    perturb_lengths: torch.Tensor
    perturb_valid_mask: torch.Tensor
    original_joint_indices: torch.LongTensor


def _joint_perturbation_geometry(
    joints: np.ndarray,
    parents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint_array = np.asarray(joints, dtype=np.float32)
    parent_array = np.asarray(parents, dtype=np.int64)
    count = int(joint_array.shape[0])
    children: list[list[int]] = [[] for _ in range(count)]
    for child, parent in enumerate(parent_array.tolist()):
        if parent >= 0:
            children[parent].append(child)

    edge_lengths = np.zeros((count,), dtype=np.float32)
    for child, parent in enumerate(parent_array.tolist()):
        if parent >= 0:
            edge_lengths[child] = float(
                np.linalg.norm(joint_array[child] - joint_array[parent])
            )
    nonzero_lengths = edge_lengths[edge_lengths > 1.0e-7]
    global_length = (
        float(np.median(nonzero_lengths))
        if nonzero_lengths.shape[0] > 0
        else 2.0 / 256.0
    )

    axes = np.zeros_like(joint_array)
    lengths = edge_lengths.copy()
    valid = np.ones((count,), dtype=bool)
    for node in range(count):
        parent = int(parent_array[node])
        if parent >= 0 and edge_lengths[node] > 1.0e-7:
            axes[node] = joint_array[node] - joint_array[parent]
            continue

        child_vectors = np.asarray(
            [
                joint_array[child] - joint_array[node]
                for child in children[node]
                if np.linalg.norm(joint_array[child] - joint_array[node]) > 1.0e-7
            ],
            dtype=np.float32,
        )
        if child_vectors.shape[0] > 0:
            axes[node] = child_vectors.mean(axis=0)
            lengths[node] = float(np.median(np.linalg.norm(child_vectors, axis=1)))
            continue

        if parent >= 0:
            parent_axis = axes[parent]
            if float(np.linalg.norm(parent_axis)) > 1.0e-7:
                axes[node] = parent_axis
                lengths[node] = max(float(lengths[parent]), global_length)
                valid[node] = False
                continue

        axes[node] = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        lengths[node] = global_length
        valid[node] = False

    norms = np.linalg.norm(axes, axis=1, keepdims=True)
    axes = axes / np.maximum(norms, 1.0e-8)
    lengths = np.maximum(lengths, 0.0).astype(np.float32)
    return axes.astype(np.float32), lengths, valid


class StackCloseManifestDataset(Dataset[StackCloseSample]):
    """Isolated stack-close view over the frozen rootless-v3 reader."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        legacy_tokenizer: Any,
        stack_tokenizer: StackCloseTokenizer,
        frame_count: int = 24,
        limit: int = 0,
        cls_name: str = "articulationxl",
        random_query: bool = True,
        random_sibling_order: bool = True,
        seed: int = 20260720,
        motion_fps_ratio: float = 0.7,
        motion_vertex_samples: int = 512,
    ) -> None:
        self.base = DynamicRigManifestDataset(
            manifest,
            legacy_tokenizer,
            frame_count=frame_count,
            limit=limit,
            cls_name=cls_name,
            random_query=random_query,
            seed=seed,
            motion_fps_ratio=motion_fps_ratio,
            motion_vertex_samples=motion_vertex_samples,
            target_active_skin_only=False,
            active_skin_threshold=1.0e-4,
            target_start_policy="joint0",
            target_root_policy="legacy",
            input_space_policy="mesh_query_bbox",
        )
        self.stack_tokenizer = stack_tokenizer
        self.random_sibling_order = bool(random_sibling_order)
        self.seed = int(seed)
        self._rng: np.random.Generator | None = None
        self._rng_worker_seed: int | None = None

    def __len__(self) -> int:
        return len(self.base)

    def _sibling_rng(self) -> np.random.Generator | None:
        if not self.random_sibling_order:
            return None
        worker = get_worker_info()
        worker_seed = (
            int(torch.initial_seed())
            if worker is None
            else int(worker.seed)
        )
        if self._rng is None or self._rng_worker_seed != worker_seed:
            seed = (worker_seed ^ self.seed) % (2**32)
            self._rng = np.random.default_rng(seed)
            self._rng_worker_seed = worker_seed
        return self._rng

    def __getitem__(self, index: int) -> StackCloseSample:
        base = self.base[int(index)]
        serialization: StackCloseSerialization = self.stack_tokenizer.serialize_tree(
            base.target_joints.numpy(),
            base.target_parents.numpy(),
            cls=base.cls,
            sibling_rng=self._sibling_rng(),
        )
        axes, lengths, valid = _joint_perturbation_geometry(
            serialization.joints,
            serialization.parents,
        )
        empty_xyz = torch.zeros((0, 3), dtype=torch.float32)
        empty_mask = torch.zeros((0,), dtype=torch.bool)
        core = replace(
            base,
            input_ids=torch.from_numpy(serialization.tokens),
            attention_mask=torch.ones(
                (serialization.tokens.shape[0],),
                dtype=torch.float32,
            ),
            target_joints=torch.from_numpy(serialization.joints),
            target_parents=torch.from_numpy(serialization.parents),
            branch_prior_roots=empty_xyz,
            branch_prior_children=empty_xyz,
            branch_prior_mask=empty_mask,
        )
        return StackCloseSample(
            core=core,
            coordinate_token_positions=torch.from_numpy(
                serialization.coordinate_token_positions
            ),
            perturb_axes=torch.from_numpy(axes),
            perturb_lengths=torch.from_numpy(lengths),
            perturb_valid_mask=torch.from_numpy(valid),
            original_joint_indices=torch.from_numpy(serialization.original_indices),
        )


def _pad_joint_matrix(
    values: list[torch.Tensor],
    *,
    pad_value: int | float,
) -> torch.Tensor:
    max_joints = max(int(value.shape[0]) for value in values)
    trailing = tuple(values[0].shape[1:])
    out = values[0].new_full((len(values), max_joints, *trailing), pad_value)
    for row, value in enumerate(values):
        out[row, : value.shape[0]] = value
    return out


def stack_close_collate(
    batch: list[StackCloseSample],
    *,
    pad_token: int,
) -> dict[str, Any]:
    out = dynamic_rig_collate(
        [item.core for item in batch],
        pad_token=pad_token,
    )
    out["target_ids"] = out["input_ids"].clone()
    out["coordinate_token_positions"] = _pad_joint_matrix(
        [item.coordinate_token_positions for item in batch],
        pad_value=-1,
    )
    out["perturb_axes"] = _pad_joint_matrix(
        [item.perturb_axes for item in batch],
        pad_value=0.0,
    )
    out["perturb_lengths"] = _pad_joint_matrix(
        [item.perturb_lengths for item in batch],
        pad_value=0.0,
    )
    out["perturb_valid_mask"] = _pad_joint_matrix(
        [item.perturb_valid_mask for item in batch],
        pad_value=0,
    ).to(dtype=torch.bool)
    out["original_joint_indices"] = _pad_joint_matrix(
        [item.original_joint_indices for item in batch],
        pad_value=-1,
    )
    return out
