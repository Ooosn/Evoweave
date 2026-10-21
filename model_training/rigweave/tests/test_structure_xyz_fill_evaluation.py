from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rigweave.dynamic_rig.aux_skeleton_tokens import AuxSkeletonTokens
from rigweave.dynamic_rig.structure_xyz_fill_data import build_fill_inputs, build_structure_fill_targets
from structure_xyz_fill_evaluation import (
    _dfs_edge_f1,
    _j2j,
    evaluate_fill_items,
    summarize_fill_rows,
)


class FakeFillModel(torch.nn.Module):
    def __init__(
        self,
        codec: AuxSkeletonTokens,
        labels: torch.Tensor,
        tokens: list[int],
        *,
        bins: int = 16,
        success: bool = True,
        fill_max_tokens: int = 3076,
    ):
        super().__init__()
        self.config = SimpleNamespace(fill_max_tokens=fill_max_tokens)
        self.codec = codec
        self.labels = labels
        self.tokens = tokens
        self.bins = bins
        self.generation_success = success
        self.last_labels = labels
        self.encode_calls = 0
        self.forward_calls = 0
        self.generate_calls = 0
        self.fill_calls = 0

    @staticmethod
    def _logits(labels: torch.Tensor, bins: int) -> torch.Tensor:
        logits = torch.full((*labels.shape, bins), -10.0, dtype=torch.float32)
        return logits.scatter(-1, labels[..., None], 10.0)

    def encode(self, surface):
        self.encode_calls += 1
        points = surface["query_points"]
        return torch.zeros((1, points.shape[1], 2, 4)), torch.zeros((1, 2, 4))

    def forward_encoded(self, final_frames, virtual, anchor_points, targets, *, structure_weight, return_logits):
        self.forward_calls += 1
        assert return_logits is True
        labels = targets["xyz_labels"]
        self.last_labels = labels
        return {
            "xyz_logits": self._logits(labels, self.bins),
            "xyz_loss_per_frame": torch.arange(labels.shape[1], dtype=torch.float32)[None] + 1.0,
            "xyz_accuracy_per_frame": torch.full((1, labels.shape[1]), 1.0),
        }

    def generate_structure(self, virtual):
        self.generate_calls += 1
        return {
            "success": self.generation_success,
            "status": "eos" if self.generation_success else "max_tokens",
            "tokens": self.tokens if self.generation_success else [self.codec.bos],
            "parents": self.codec.decode(self.tokens).tolist() if self.generation_success else None,
        }

    def fill_decoder(self, final_frames, anchor_points, inputs):
        self.fill_calls += 1
        labels = self.last_labels
        return self._logits(labels, self.bins)


def _item(
    codec: AuxSkeletonTokens,
    *,
    asset_id: str = "asset-0",
    bins: int = 16,
    frame_count: int = 8,
):
    parents = np.asarray([-1, 0, 0, 1], dtype=np.int64)
    xyz = np.zeros((1, frame_count, 4, 3), dtype=np.float32)
    for frame in range(frame_count):
        for joint in range(4):
            xyz[0, frame, joint] = (-0.7 + joint * 0.12, -0.5 + frame * 0.03, -0.2 + joint * 0.04)
    geometry = {
        "asset_id": [asset_id],
        "selected_frames": np.arange(100, 100 + frame_count, dtype=np.int64)[None],
        "frame_target_joints": xyz.copy(),
        "target_parents": parents[None],
        "joint_mask": np.ones((1, 4), dtype=bool),
        "frame_mask": np.ones((1, frame_count), dtype=bool),
    }
    targets = build_structure_fill_targets(geometry, codec, bins=bins, max_fill_tokens=3076)
    query_points = torch.zeros((1, frame_count, 2, 3), dtype=torch.float32)
    for frame in range(frame_count):
        query_points[:, frame, :, 0] = frame / 10.0
        query_points[:, frame, :, 1] = (frame + 1) / 20.0
    surface = {"query_points": query_points}
    tokens = codec.encode(parents).tokens.tolist()
    return surface, targets, geometry, tokens


def test_pure_metrics_use_directed_dfs_edges_and_symmetric_j2j() -> None:
    assert _dfs_edge_f1([-1], [-1]) == 1.0
    assert _dfs_edge_f1([-1, 0, 0], [-1, 0, 1]) == 0.5
    assert _dfs_edge_f1([-1], [-1, 0]) == 0.0
    assert _j2j(np.asarray([[0.0, 0.0, 0.0]]), np.asarray([[1.0, 0.0, 0.0]])) == 1.0


def test_evaluate_scores_selected_frames_but_encodes_all_eight_and_restores_training() -> None:
    codec = AuxSkeletonTokens(max_joints=8, max_sequence_tokens=64)
    surface, targets, geometry, tokens = _item(codec)
    model = FakeFillModel(codec, targets["xyz_labels"], tokens)
    report = evaluate_fill_items(
        model, [(surface, targets, geometry)], codec, frame_indices=[0, 2, 4, 6], group="g", bins=16
    )

    assert model.encode_calls == 1
    assert model.forward_calls == 1
    assert model.generate_calls == 1
    assert model.fill_calls == 1
    assert model.training is True
    assert len(report["rows"]) == 4
    assert len(report["structure_rows"]) == 1
    assert [row["frame_index"] for row in report["rows"]] == [0, 2, 4, 6]
    assert [row["raw_frame_index"] for row in report["rows"]] == [100, 102, 104, 106]
    for row in report["rows"]:
        expected_points = surface["query_points"][0, row["frame_index"]].float().cpu().tolist()
        assert row["input_anchor_points"] == expected_points
    assert all(row["success"] for row in report["rows"])
    assert all(row["original_edge_f1"] == 1.0 for row in report["rows"])
    assert all(row["j2j"] is not None and row["j2j"] <= 0.13 for row in report["rows"])
    assert all(row["gt_structure_xyz_l1"] <= 0.13 for row in report["rows"])
    assert report["structure_rows"][0]["dfs_edge_f1"] == 1.0
    assert report["summary"]["failure_count"] == 0
    assert report["summary"]["score_frame_count"] == 4
    assert report["summary"]["gt_structure_xyz_ce"] == report["summary"]["mean_gt_structure_xyz_ce"]
    assert report["summary"]["original_edge_f1"] == report["summary"]["mean_original_edge_f1"]
    assert summarize_fill_rows(report["rows"], report["structure_rows"]) == report["summary"]


def test_preflight_accepts_four_input_frames_and_arbitrary_group_without_reference() -> None:
    codec = AuxSkeletonTokens(max_joints=8, max_sequence_tokens=64)
    surface, targets, geometry, tokens = _item(codec, frame_count=4)
    model = FakeFillModel(codec, targets["xyz_labels"], tokens)
    report = evaluate_fill_items(
        model,
        [(surface, targets, geometry)],
        codec,
        frame_indices=[0, 1, 2, 3],
        group="preflight-custom-group",
        bins=16,
    )
    assert [row["frame_index"] for row in report["rows"]] == [0, 1, 2, 3]
    assert [row["raw_frame_index"] for row in report["rows"]] == [100, 101, 102, 103]


def test_free_fill_builder_receives_query_device_and_model_capacity() -> None:
    codec = AuxSkeletonTokens(max_joints=8, max_sequence_tokens=64)
    surface, targets, geometry, tokens = _item(codec)
    model = FakeFillModel(codec, targets["xyz_labels"], tokens, fill_max_tokens=31)
    with patch("structure_xyz_fill_evaluation.build_fill_inputs", wraps=build_fill_inputs) as builder:
        evaluate_fill_items(
            model,
            [(surface, targets, geometry)],
            codec,
            frame_indices=[0, 2, 4, 6],
            group="g",
            bins=16,
        )

    assert builder.call_count == 1
    assert builder.call_args.kwargs["device"] == surface["query_points"].device
    assert builder.call_args.kwargs["max_fill_tokens"] == 31


def test_failed_free_generation_scores_zero_without_swallowing_gt_metrics() -> None:
    codec = AuxSkeletonTokens(max_joints=8, max_sequence_tokens=64)
    surface, targets, geometry, tokens = _item(codec)
    model = FakeFillModel(codec, targets["xyz_labels"], tokens, success=False)
    report = evaluate_fill_items(
        model, [(surface, targets, geometry)], codec, frame_indices=[0, 2, 4, 6], group="g", bins=16
    )

    assert model.fill_calls == 0
    assert all(not row["success"] for row in report["rows"])
    assert all(row["original_edge_f1"] == 0.0 for row in report["rows"])
    assert all(row["j2j"] is None for row in report["rows"])
    assert all(row["exact_count"] is False for row in report["rows"])
    assert all(row["gt_structure_xyz_accuracy"] == 1.0 for row in report["rows"])
    assert report["summary"]["failure_count"] == 4
    assert report["summary"]["mean_original_edge_f1"] == 0.0
    assert report["summary"]["mean_j2j"] == 0.0


@pytest.mark.parametrize("frame_indices", ([], [0, 0], [8], [True]))
def test_invalid_frame_selection_is_rejected(frame_indices: object) -> None:
    codec = AuxSkeletonTokens(max_joints=8, max_sequence_tokens=64)
    surface, targets, geometry, tokens = _item(codec)
    model = FakeFillModel(codec, targets["xyz_labels"], tokens)
    with pytest.raises(ValueError):
        evaluate_fill_items(
            model, [(surface, targets, geometry)], codec,
            frame_indices=frame_indices, group="g", bins=16,
        )


def test_pruned_targets_and_invalid_masks_are_rejected() -> None:
    codec = AuxSkeletonTokens(max_joints=8, max_sequence_tokens=64)
    surface, targets, geometry, tokens = _item(codec)
    model = FakeFillModel(codec, targets["xyz_labels"], tokens)
    changed = dict(targets)
    changed["removed_joint_counts"] = torch.ones_like(targets["removed_joint_counts"])
    with pytest.raises(ValueError, match="unpruned"):
        evaluate_fill_items(
            model, [(surface, changed, geometry)], codec,
            frame_indices=[0, 2, 4, 6], group="g", bins=16,
        )

    changed = dict(targets)
    changed["frame_mask"] = targets["frame_mask"].clone()
    changed["frame_mask"][0, 2] = False
    with pytest.raises(ValueError):
        evaluate_fill_items(
            model, [(surface, changed, geometry)], codec,
            frame_indices=[0, 2, 4, 6], group="g", bins=16,
        )


def test_reference_lookup_checks_raw_frames_and_full_target_tokens_once_per_asset() -> None:
    codec = AuxSkeletonTokens(max_joints=8, max_sequence_tokens=64)
    surface, targets, geometry, tokens = _item(codec)
    model = FakeFillModel(codec, targets["xyz_labels"], tokens)
    reference_tokens = [11, 12, 13]
    reference_tensor = torch.tensor(reference_tokens, dtype=torch.long).reshape(1, 1, -1).repeat(1, 8, 1)
    reference_mask = torch.ones_like(reference_tensor, dtype=torch.bool)
    lookup = {
        ("g", "asset-0", frame): {
            "raw_frame_index": 100 + frame,
            "target_tokens": reference_tokens,
            "original_edge_f1": 0.75,
        }
        for frame in (0, 2, 4, 6)
    }
    with patch("structure_xyz_fill_evaluation.build_xyz_aux_targets", return_value={
        "main_input_ids": reference_tensor,
        "main_attention_mask": reference_mask,
    }) as build_reference:
        report = evaluate_fill_items(
            model,
            [(surface, targets, geometry)],
            codec,
            frame_indices=[0, 2, 4, 6],
            group="g",
            original_tokenizer=object(),
            reference_lookup=lookup,
            bins=16,
        )
    build_reference.assert_called_once()
    assert all(row["reference_original_edge_f1"] == 0.75 for row in report["rows"])


def test_reference_mismatch_raises_and_nonfinite_logits_are_not_failures() -> None:
    codec = AuxSkeletonTokens(max_joints=8, max_sequence_tokens=64)
    surface, targets, geometry, tokens = _item(codec)
    model = FakeFillModel(codec, targets["xyz_labels"], tokens)
    lookup = {
        ("g", "asset-0", frame): {
            "raw_frame_index": 999,
            "target_tokens": [1, 2, 3],
            "original_edge_f1": 0.75,
        }
        for frame in (0, 2, 4, 6)
    }
    with patch("structure_xyz_fill_evaluation.build_xyz_aux_targets", return_value={
        "main_input_ids": torch.tensor([[[11, 12, 13]] * 8]),
        "main_attention_mask": torch.ones((1, 8, 3), dtype=torch.bool),
    }):
        with pytest.raises(ValueError, match="raw_frame_index"):
            evaluate_fill_items(
                model,
                [(surface, targets, geometry)],
                codec,
                frame_indices=[0, 2, 4, 6],
                group="g",
                original_tokenizer=object(),
                reference_lookup=lookup,
                bins=16,
            )

    class NonfiniteModel(FakeFillModel):
        def forward_encoded(self, *args, **kwargs):
            output = super().forward_encoded(*args, **kwargs)
            output["xyz_logits"][0, 0, 0, 0, 0] = float("nan")
            return output

    with pytest.raises(ValueError, match="non-finite"):
        evaluate_fill_items(
            NonfiniteModel(codec, targets["xyz_labels"], tokens),
            [(surface, targets, geometry)],
            codec,
            frame_indices=[0, 2, 4, 6],
            group="g",
            bins=16,
        )
