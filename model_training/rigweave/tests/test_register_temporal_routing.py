from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rigweave.dynamic_rig.motion_encoder import (  # noqa: E402
    AnchorWiseAlternatingBlock,
    AnchorWiseAlternatingMotionEncoder,
)


class _MeanMix(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[torch.Tensor] = []

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        self.inputs.append(tokens.detach().clone())
        return tokens.mean(dim=1, keepdim=True).expand_as(tokens)


def _legacy_block_forward(block: AnchorWiseAlternatingBlock, tokens: torch.Tensor) -> torch.Tensor:
    batch_size, frame_count, slot_count, dim = tokens.shape
    z = block.pose_inner(tokens.reshape(batch_size * frame_count, slot_count, dim))
    z = z.reshape(batch_size, frame_count, slot_count, dim)
    z = z.permute(0, 2, 1, 3).reshape(batch_size * slot_count, frame_count, dim)
    z = block.anchor_temporal(z)
    return z.reshape(batch_size, slot_count, frame_count, dim).permute(0, 2, 1, 3)


class RegisterTemporalRoutingTest(unittest.TestCase):
    def test_default_and_explicit_aligned_match_legacy_forward(self) -> None:
        torch.manual_seed(101)
        settings = dict(dim=8, depth=2, heads=2, register_tokens=2, max_frames=4)
        default = AnchorWiseAlternatingMotionEncoder(**settings).eval()
        explicit = AnchorWiseAlternatingMotionEncoder(
            **settings, register_temporal_mode="aligned"
        ).eval()
        explicit.load_state_dict(default.state_dict(), strict=True)
        frame_tokens = torch.randn(2, 3, 4, 8)

        default_tokens, default_all = default(frame_tokens, return_all=True)
        explicit_tokens, explicit_all = explicit(frame_tokens, return_all=True)

        torch.testing.assert_close(default_tokens, explicit_tokens, atol=0, rtol=0)
        torch.testing.assert_close(default_all, explicit_all, atol=0, rtol=0)

        block = AnchorWiseAlternatingBlock(
            dim=8,
            heads=2,
            register_tokens=2,
            role_prefix_count=1,
            register_temporal_mode="aligned",
        ).eval()
        tokens = torch.randn(2, 3, 7, 8)
        actual = block(tokens)
        expected = _legacy_block_forward(block, tokens)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_global_mode_has_identical_state_keys_and_strict_old_load(self) -> None:
        torch.manual_seed(103)
        settings = dict(dim=8, depth=2, heads=2, register_tokens=2, max_frames=4)
        old = AnchorWiseAlternatingMotionEncoder(**settings).eval()
        experimental = AnchorWiseAlternatingMotionEncoder(
            **settings, register_temporal_mode="global"
        ).eval()
        old_keys = list(old.state_dict())
        global_keys = list(experimental.state_dict())
        self.assertEqual(global_keys, old_keys)

        loaded = experimental.load_state_dict(old.state_dict(), strict=True)
        self.assertEqual(loaded.missing_keys, [])
        self.assertEqual(loaded.unexpected_keys, [])
        before = {name: value.clone() for name, value in experimental.state_dict().items()}
        experimental.set_register_temporal_mode("global")
        self.assertEqual(list(experimental.state_dict()), old_keys)
        for name, value in experimental.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)
        self.assertEqual(experimental.register_temporal_mode, "global")
        self.assertTrue(all(block.register_temporal_mode == "global" for block in experimental.blocks))

    def test_global_registers_mix_across_slots_without_asset_leak(self) -> None:
        batch_size, frame_count, register_count, anchor_count, dim = 2, 3, 2, 2, 4
        slot_count = 1 + register_count + anchor_count
        aligned = AnchorWiseAlternatingBlock(
            dim=dim,
            heads=1,
            mlp_ratio=1.0,
            register_temporal_mode="aligned",
            register_tokens=register_count,
            role_prefix_count=1,
        ).eval()
        global_block = AnchorWiseAlternatingBlock(
            dim=dim,
            heads=1,
            mlp_ratio=1.0,
            register_temporal_mode="global",
            register_tokens=register_count,
            role_prefix_count=1,
        ).eval()
        aligned.pose_inner = nn.Identity()
        global_block.pose_inner = nn.Identity()
        aligned_mix = _MeanMix()
        global_mix = _MeanMix()
        aligned.anchor_temporal = aligned_mix
        global_block.anchor_temporal = global_mix

        register_perturbation = torch.zeros(batch_size, frame_count, slot_count, dim)
        register_perturbation[0, 1, 1, 0] = 1.0
        aligned_result = aligned(register_perturbation)
        global_result = global_block(register_perturbation)

        self.assertEqual(len(aligned_mix.inputs), 1)
        self.assertEqual(aligned_mix.inputs[0].shape, (batch_size * slot_count, frame_count, dim))
        self.assertEqual(len(global_mix.inputs), 2)
        self.assertEqual(global_mix.inputs[0].shape, (batch_size * (slot_count - register_count), frame_count, dim))
        self.assertEqual(global_mix.inputs[1].shape, (batch_size, frame_count * register_count, dim))
        self.assertEqual(float(aligned_result[0, :, 2, 0].abs().max()), 0.0)
        self.assertGreater(float(global_result[0, :, 2, 0].abs().max()), 0.0)
        self.assertTrue(torch.equal(global_result[1], torch.zeros_like(global_result[1])))
        self.assertTrue(torch.equal(global_result[0, :, 3:, :], torch.zeros_like(global_result[0, :, 3:, :])))

        anchor_perturbation = torch.zeros_like(register_perturbation)
        anchor_perturbation[0, 2, 3, 0] = 1.0
        anchor_result = global_block(anchor_perturbation)
        self.assertGreater(float(anchor_result[0, :, 3, 0].abs().max()), 0.0)
        self.assertTrue(torch.equal(anchor_result[0, :, 4, :], torch.zeros_like(anchor_result[0, :, 4, :])))
        self.assertTrue(torch.equal(anchor_result[0, :, 1:3, :], torch.zeros_like(anchor_result[0, :, 1:3, :])))
        self.assertTrue(torch.equal(anchor_result[1], torch.zeros_like(anchor_result[1])))

    def test_zero_registers_use_legacy_path_and_single_frame_is_valid(self) -> None:
        torch.manual_seed(107)
        settings = dict(dim=8, depth=1, heads=2, register_tokens=0, max_frames=2)
        aligned = AnchorWiseAlternatingMotionEncoder(
            **settings, register_temporal_mode="aligned"
        ).eval()
        global_mode = AnchorWiseAlternatingMotionEncoder(
            **settings, register_temporal_mode="global"
        ).eval()
        global_mode.load_state_dict(aligned.state_dict(), strict=True)
        frame_tokens = torch.randn(2, 1, 3, 8)
        aligned_output, aligned_all = aligned(frame_tokens, return_all=True)
        global_output, global_all = global_mode(frame_tokens, return_all=True)
        torch.testing.assert_close(global_output, aligned_output, atol=0, rtol=0)
        torch.testing.assert_close(global_all, aligned_all, atol=0, rtol=0)
        self.assertEqual(tuple(global_output.shape), (2, 3, 8))
        self.assertEqual(tuple(global_all.shape), (2, 1, 1 + 3, 8))

    def test_global_shapes_and_backward_are_finite(self) -> None:
        torch.manual_seed(109)
        encoder = AnchorWiseAlternatingMotionEncoder(
            dim=8,
            depth=2,
            heads=2,
            register_tokens=2,
            max_frames=4,
            register_temporal_mode="global",
        )
        frame_tokens = torch.randn(2, 3, 4, 8, requires_grad=True)
        output, all_tokens = encoder(frame_tokens, return_all=True)
        self.assertEqual(tuple(output.shape), (2, 4, 8))
        self.assertEqual(tuple(all_tokens.shape), (2, 3, 1 + 2 + 4, 8))
        (output.square().mean() + all_tokens.square().mean()).backward()
        self.assertIsNotNone(frame_tokens.grad)
        self.assertTrue(bool(torch.isfinite(frame_tokens.grad).all()))
        for parameter in encoder.parameters():
            if parameter.requires_grad and parameter.grad is not None:
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_invalid_modes_counts_and_slot_layout_fail(self) -> None:
        with self.assertRaises(ValueError):
            AnchorWiseAlternatingMotionEncoder(8, register_temporal_mode="invalid")
        with self.assertRaises(ValueError):
            AnchorWiseAlternatingMotionEncoder(8, register_tokens=-1)
        with self.assertRaises(ValueError):
            AnchorWiseAlternatingBlock(8, 2, register_temporal_mode="invalid")
        with self.assertRaises(ValueError):
            AnchorWiseAlternatingBlock(8, 2, register_tokens=-1)
        with self.assertRaises(ValueError):
            AnchorWiseAlternatingBlock(8, 2, role_prefix_count=-1)

        encoder = AnchorWiseAlternatingMotionEncoder(8, depth=1, heads=2, register_tokens=1)
        with self.assertRaises(ValueError):
            encoder.set_register_temporal_mode("invalid")
        encoder.set_register_temporal_mode("global")
        self.assertEqual(encoder.register_temporal_mode, "global")
        block = AnchorWiseAlternatingBlock(
            8,
            2,
            register_temporal_mode="global",
            register_tokens=2,
            role_prefix_count=1,
        )
        with self.assertRaises(ValueError):
            block(torch.zeros(1, 2, 2, 8))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
