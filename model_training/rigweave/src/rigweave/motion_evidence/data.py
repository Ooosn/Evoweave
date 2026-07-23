"""Strict rootless dataset wrapper carrying training-only skin supervision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from rigweave.dynamic_rig.data import (
    DynamicRigManifestDataset,
    DynamicRigSample,
    dynamic_rig_collate,
)


@dataclass(frozen=True)
class MotionEvidenceTrainingSample:
    dynamic: DynamicRigSample
    target_skin_weights: torch.Tensor


class MotionEvidenceManifestDataset(Dataset):
    """Add target skin weights without changing the baseline dataset contract."""

    def __init__(self, manifest: str, tokenizer: Any, **kwargs: Any) -> None:
        self.dynamic = DynamicRigManifestDataset(manifest, tokenizer, **kwargs)

    def __len__(self) -> int:
        return len(self.dynamic)

    def __getitem__(self, index: int) -> MotionEvidenceTrainingSample:
        sample = self.dynamic[index]
        with np.load(sample.path, allow_pickle=True) as raw:
            if "target_skin_weights" not in raw.files:
                raise KeyError(f"{sample.path} lacks target_skin_weights")
            skin = np.asarray(raw["target_skin_weights"], dtype=np.float32)
        if skin.shape != (sample.vertex_count, sample.joint_count):
            raise ValueError(
                f"{sample.path} target_skin_weights shape {skin.shape} != "
                f"({sample.vertex_count}, {sample.joint_count})"
            )
        return MotionEvidenceTrainingSample(
            dynamic=sample,
            target_skin_weights=torch.from_numpy(skin),
        )


def _pad_skin_weights(values: list[torch.Tensor]) -> torch.Tensor:
    max_vertices = max(int(value.shape[0]) for value in values)
    max_joints = max(int(value.shape[1]) for value in values)
    out = values[0].new_zeros((len(values), max_vertices, max_joints))
    for index, value in enumerate(values):
        out[index, : value.shape[0], : value.shape[1]] = value
    return out


def motion_evidence_collate(
    samples: list[MotionEvidenceTrainingSample],
    *,
    pad_token: int,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty motion-evidence batch")
    batch = dynamic_rig_collate(
        [sample.dynamic for sample in samples],
        pad_token=pad_token,
    )
    batch["target_skin_weights"] = _pad_skin_weights(
        [sample.target_skin_weights for sample in samples]
    )
    return batch
