from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rigweave.dynamic_rig.motion_encoder import AnchorWiseAlternatingMotionEncoder
from rigweave.dynamic_rig.motion_evidence import MotionEvidence, MotionEvidenceInjection, grouped_motion_evidence


def clouds():
    centers = torch.tensor([[-0.75, 0, 0], [-0.25, 0, 0], [0.25, 0, 0], [0.75, 0, 0]])
    offsets = torch.tensor([[0.02, 0.02, 0.02], [0.02, -0.02, -0.02],
                            [-0.02, 0.02, -0.02], [-0.02, -0.02, 0.02]])
    query = (centers[:, None] + offsets[None]).reshape(16, 3)
    return query, torch.tensor([[0, 4, 8, 12]])


class MotionObservationsTest(unittest.TestCase):
    def test_static_and_ambient_autocast(self):
        query, indices = clouds()
        points = query[None, None].repeat(1, 3, 1, 1).requires_grad_(True)
        expected = grouped_motion_evidence(points, indices)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = grouped_motion_evidence(points, indices)
        torch.testing.assert_close(actual.states, expected.states, atol=0, rtol=0)
        self.assertEqual(actual.states.dtype, torch.float32)
        self.assertFalse(actual.states.requires_grad)
        self.assertTrue(bool(actual.valid_groups.all()))
        torch.testing.assert_close(actual.states[..., 0], torch.ones(1, 4, 4), atol=0, rtol=0)
        self.assertEqual(float(actual.states[..., 1:].abs().max()), 0)
        expected_features = actual.anchor_features.new_tensor([0, 0, 1, 0, 0]).expand_as(actual.anchor_features)
        torch.testing.assert_close(actual.anchor_features, expected_features, atol=0, rtol=0)

    def test_exact_static_policy_does_not_threshold_real_motion(self):
        query, indices = clouds()
        moved = query.clone()
        moved[:, 0] += 1e-5
        clips = torch.stack((query[None].repeat(3, 1, 1), torch.stack((query, moved, query))))
        result = grouped_motion_evidence(clips, indices.repeat(2, 1))
        torch.testing.assert_close(result.states[0, ..., 0], torch.ones(4, 4), atol=0, rtol=0)
        self.assertEqual(float(result.states[0, ..., 1:].abs().max()), 0)
        self.assertGreater(float(result.states[1, ..., 1:].sum()), 0)

    def test_asymmetric_static_cloud_is_exactly_unknown(self):
        generator = torch.Generator().manual_seed(91)
        query = torch.randn(64, 3, generator=generator)
        result = grouped_motion_evidence(query[None, None].repeat(1, 8, 1, 1), torch.tensor([[0, 16, 32, 48]]))
        self.assertTrue(bool(result.valid_groups.any()))
        torch.testing.assert_close(result.states[..., 0], torch.ones(1, 4, 4), atol=0, rtol=0)
        self.assertEqual(float(result.states[..., 1:].abs().max()), 0)

    def test_common_rigid_motion_is_not_relative_change(self):
        query, indices = clouds()
        c, s = math.cos(0.3), math.sin(0.3)
        rotation = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        moved = query @ rotation.T + torch.tensor([0.1, 0.2, -0.1])
        result = grouped_motion_evidence(torch.stack((query, moved, moved))[None], indices)
        self.assertLess(float(result.states[..., 2].abs().max()), 5e-5)
        self.assertGreater(float(result.states[..., 1].mean()), 0.5)

    def test_relative_change_and_frame_replay(self):
        query, indices = clouds()
        moved = query.clone()
        moved[8:, 0] += 0.25
        points = torch.stack((query, moved, query))[None]
        actual = grouped_motion_evidence(points, indices)
        torch.testing.assert_close(actual.states.sum(-1), torch.ones(1, 4, 4))
        self.assertGreater(float(actual.states[0, 0, 3, 2]), 0.6)
        self.assertLess(float(actual.states[0, 2, 3, 2]), 5e-5)
        permuted = grouped_motion_evidence(points[:, [0, 2, 1]], indices)
        duplicate = grouped_motion_evidence(points[:, [0, 1, 2, 1, 2]], indices)
        torch.testing.assert_close(permuted.states, actual.states, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(duplicate.states, actual.states, atol=1e-6, rtol=1e-6)
        order = [3, 1, 0, 2]
        reordered = grouped_motion_evidence(points, indices[:, order])
        torch.testing.assert_close(reordered.states, actual.states[:, order][:, :, order], atol=1e-5, rtol=1e-5)

    def test_ill_posed_rotation_is_unknown_not_a_missing_token(self):
        query = torch.stack((torch.linspace(-1, 1, 16), torch.zeros(16), torch.zeros(16)), dim=-1)
        result = grouped_motion_evidence(query[None, None].repeat(1, 2, 1, 1), torch.tensor([[0, 5, 10, 15]]))
        self.assertEqual(result.states.shape, (1, 4, 4, 3))
        self.assertFalse(bool(result.valid_groups.any()))
        torch.testing.assert_close(result.states[..., 0], torch.ones(1, 4, 4))
        self.assertEqual(float(result.states[..., 1:].abs().max()), 0.0)


def evidence(batch=2, anchors=8):
    states = torch.rand(batch, anchors, anchors, 3, dtype=torch.float64).softmax(-1)
    features = torch.cat((torch.rand(batch, anchors, 2, dtype=torch.float64), states.mean(2)), dim=-1)
    return MotionEvidence(states, features, torch.ones(batch, anchors, dtype=torch.bool))


def reference(layer, tokens, observations, adapter):
    batch, frames, slots, dim = tokens.shape
    heads = layer.self_attn.num_heads
    anchors = observations.states.shape[1]
    prefix = slots - anchors
    if adapter.token_weight is not None:
        delta = F.linear(observations.anchor_features, adapter.token_weight)
        tokens = torch.cat((tokens[:, :, :prefix], tokens[:, :, prefix:] + delta[:, None]), dim=2)
    x = tokens.reshape(batch * frames, slots, dim)
    normalized = layer.norm1(x)
    qkv = F.linear(normalized, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias)
    qkv = qkv.reshape(batch * frames, slots, 3, heads, dim // heads).permute(2, 0, 3, 1, 4)
    query, key, value = qkv.unbind(0)
    mask = None
    if adapter.bias_weights is not None:
        selected = F.linear(observations.states, adapter.bias_weights).permute(0, 3, 1, 2)
        rest = selected.new_zeros(batch, heads - adapter.biased_heads, anchors, anchors)
        mask = F.pad(torch.cat((selected, rest), dim=1), (prefix, 0, prefix, 0))
        mask = mask[:, None].expand(batch, frames, heads, slots, slots).reshape(batch * frames, heads, slots, slots)
    attended = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
    attended = attended.transpose(1, 2).reshape(batch * frames, slots, dim)
    attended = F.linear(attended, layer.self_attn.out_proj.weight, layer.self_attn.out_proj.bias)
    x = x + attended
    x = x + layer.linear2(layer.activation(layer.linear1(layer.norm2(x))))
    return x.reshape(batch, frames, slots, dim)


class MotionFusionTest(unittest.TestCase):
    def test_zero_initialization_and_nonzero_gradient_reference(self):
        for fusion, heads in [("bias", 1), ("bias", 2), ("bias", 4), ("token", 0), ("hybrid", 2)]:
            with self.subTest(fusion=fusion, heads=heads):
                torch.manual_seed(92)
                layer = nn.TransformerEncoderLayer(32, 4, 64, dropout=0, activation="gelu", batch_first=True, norm_first=True).double()
                adapter = MotionEvidenceInjection(32, 4, fusion=fusion, biased_heads=heads).double()
                x = torch.randn(2, 3, 11, 32, dtype=torch.float64, requires_grad=True)
                obs = evidence()
                zero = adapter(layer, x, obs)
                baseline = layer(x.reshape(6, 11, 32)).reshape_as(x)
                torch.testing.assert_close(zero, baseline, atol=1e-10, rtol=1e-10)
                with torch.no_grad():
                    for parameter in adapter.parameters():
                        parameter.normal_(0, 0.1)
                actual, expected = adapter(layer, x, obs), reference(layer, x, obs, adapter)
                torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
                probe = torch.randn_like(actual)
                variables = (x, *adapter.parameters(), *layer.parameters())
                actual_grad = torch.autograd.grad((actual * probe).mean(), variables)
                expected_grad = torch.autograd.grad((expected * probe).mean(), variables)
                for actual_value, expected_value in zip(actual_grad, expected_grad):
                    torch.testing.assert_close(actual_value, expected_value, atol=1e-10, rtol=1e-10)

    def test_encoder_initialization_and_checkpointed_gradients(self):
        settings = dict(dim=32, depth=2, heads=4, register_tokens=2, max_frames=24, gradient_checkpointing=True)
        torch.manual_seed(91)
        baseline = AnchorWiseAlternatingMotionEncoder(**settings).double()
        torch.manual_seed(91)
        changed = AnchorWiseAlternatingMotionEncoder(**settings, motion_evidence_fusion="hybrid", motion_evidence_heads=2).double()
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(value, changed.state_dict()[name], atol=0, rtol=0)
        for frames in [2, 8, 24]:
            x = torch.randn(2, frames, 8, 32, dtype=torch.float64, requires_grad=True)
            obs = evidence()
            expected, actual = baseline(x), changed(x, motion_evidence=obs)
            torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
            changed.zero_grad(set_to_none=True)
            (actual * torch.randn_like(actual)).mean().backward()
            for block in changed.blocks:
                self.assertGreater(float(block.motion_injection.bias_weights.grad.abs().sum()), 0)
                self.assertGreater(float(block.motion_injection.token_weight.grad.abs().sum()), 0)
                self.assertGreater(float(block.pose_inner.self_attn.in_proj_weight.grad.abs().sum()), 0)
            self.assertGreater(float(x.grad.abs().sum()), 0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
