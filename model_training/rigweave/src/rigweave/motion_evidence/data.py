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
    token_joint_indices: torch.LongTensor
    token_completed_joint_counts: torch.LongTensor
    token_branch_decision_mask: torch.BoolTensor


def _token_joint_indices(
    input_ids: torch.LongTensor,
    target_parents: torch.LongTensor,
    *,
    branch_token: int,
    eos_token: int,
) -> torch.LongTensor:
    """Map each causal hidden position to the joint its next token describes."""

    ids = input_ids.tolist()
    parents = target_parents.tolist()
    mapping = torch.full_like(input_ids, -1)
    cursor = 2  # BOS and class token precede the first joint.
    for joint_index, parent in enumerate(parents):
        is_branch = joint_index > 0 and int(parent) != joint_index - 1
        if is_branch:
            if cursor >= len(ids) or int(ids[cursor]) != int(branch_token):
                raise ValueError(
                    f"token sequence lacks branch marker for joint {joint_index} at {cursor}"
                )
            mapping[cursor - 1] = int(parent)
            cursor += 1
            if cursor + 6 > len(ids):
                raise ValueError("branch token sequence ends before parent/child coordinates")
            mapping[cursor - 1 : cursor + 2] = int(parent)
            cursor += 3
            mapping[cursor - 1 : cursor + 2] = joint_index
            cursor += 3
        else:
            if cursor + 3 > len(ids):
                raise ValueError("token sequence ends before joint coordinates")
            mapping[cursor - 1 : cursor + 2] = joint_index
            cursor += 3
    if cursor >= len(ids) or int(ids[cursor]) != int(eos_token):
        raise ValueError(f"expected EOS at token index {cursor}")
    if cursor != len(ids) - 1:
        raise ValueError("unexpected tokens follow EOS")
    return mapping


def _token_coverage_state(
    input_ids: torch.LongTensor,
    target_parents: torch.LongTensor,
    *,
    branch_token: int,
    eos_token: int,
) -> tuple[torch.LongTensor, torch.BoolTensor]:
    """Describe completed joints and explicit branch decisions per hidden state."""

    ids = input_ids.tolist()
    parents = target_parents.tolist()
    completed = torch.full_like(input_ids, -1)
    branch_decision = torch.zeros_like(input_ids, dtype=torch.bool)
    cursor = 2
    for joint_index, parent in enumerate(parents):
        is_branch = joint_index > 0 and int(parent) != joint_index - 1
        if is_branch:
            if cursor >= len(ids) or int(ids[cursor]) != int(branch_token):
                raise ValueError(
                    f"token sequence lacks branch marker for joint {joint_index} at {cursor}"
                )
            if cursor + 7 > len(ids):
                raise ValueError("branch token sequence ends before parent/child coordinates")
            completed[cursor - 1 : cursor + 6] = joint_index
            branch_decision[cursor - 1] = True
            cursor += 7
        else:
            if cursor + 3 > len(ids):
                raise ValueError("token sequence ends before joint coordinates")
            completed[cursor - 1 : cursor + 2] = joint_index
            cursor += 3

    if cursor >= len(ids) or int(ids[cursor]) != int(eos_token):
        raise ValueError(f"expected EOS at token index {cursor}")
    if cursor != len(ids) - 1:
        raise ValueError("unexpected tokens follow EOS")
    completed[cursor - 1] = len(parents)
    return completed, branch_decision


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
        coverage_state = _token_coverage_state(
            sample.input_ids,
            sample.target_parents,
            branch_token=int(self.dynamic.tokenizer.token_id_branch),
            eos_token=int(self.dynamic.tokenizer.eos),
        )
        return MotionEvidenceTrainingSample(
            dynamic=sample,
            target_skin_weights=torch.from_numpy(skin),
            token_joint_indices=_token_joint_indices(
                sample.input_ids,
                sample.target_parents,
                branch_token=int(self.dynamic.tokenizer.token_id_branch),
                eos_token=int(self.dynamic.tokenizer.eos),
            ),
            token_completed_joint_counts=coverage_state[0],
            token_branch_decision_mask=coverage_state[1],
        )


def _pad_skin_weights(values: list[torch.Tensor]) -> torch.Tensor:
    max_vertices = max(int(value.shape[0]) for value in values)
    max_joints = max(int(value.shape[1]) for value in values)
    out = values[0].new_zeros((len(values), max_vertices, max_joints))
    for index, value in enumerate(values):
        out[index, : value.shape[0], : value.shape[1]] = value
    return out


def _pad_token_joint_indices(values: list[torch.LongTensor]) -> torch.LongTensor:
    max_length = max(int(value.shape[0]) for value in values)
    out = values[0].new_full((len(values), max_length), -1)
    for index, value in enumerate(values):
        out[index, : value.shape[0]] = value
    return out


def _pad_token_branch_decisions(values: list[torch.BoolTensor]) -> torch.BoolTensor:
    max_length = max(int(value.shape[0]) for value in values)
    out = values[0].new_zeros((len(values), max_length))
    for index, value in enumerate(values):
        out[index, : value.shape[0]] = value
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
    batch["token_joint_indices"] = _pad_token_joint_indices(
        [sample.token_joint_indices for sample in samples]
    )
    batch["token_completed_joint_counts"] = _pad_token_joint_indices(
        [sample.token_completed_joint_counts for sample in samples]
    )
    batch["token_branch_decision_mask"] = _pad_token_branch_decisions(
        [sample.token_branch_decision_mask for sample in samples]
    )
    return batch
