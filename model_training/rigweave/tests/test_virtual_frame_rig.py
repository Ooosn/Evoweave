from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from rigweave.dynamic_rig.motion_evidence import MotionEvidence
from rigweave.dynamic_rig.virtual_frame_rig import (
    ContinuousJointLocalizer, VirtualFrameMotionEncoder, VirtualFrameRig,
    VirtualFrameRigConfig, coordinate_regression_loss,
)


def settings(**overrides):
    base = VirtualFrameRigConfig(
        dim=16, heads=4, query_tokens=7, register_tokens=2, motion_depth=3,
        position_depth=2, max_joints=8, motion_evidence_fusion="off", motion_evidence_heads=0,
    )
    return replace(base, **overrides)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(16, 4, 32, dropout=0, batch_first=True, norm_first=True)

    def forward(self, *, inputs_embeds, attention_mask, use_cache, return_dict):
        assert not use_cache and return_dict
        size = inputs_embeds.shape[1]
        causal = torch.ones(size, size, dtype=torch.bool, device=inputs_embeds.device).triu(1)
        hidden = self.layer(inputs_embeds, src_mask=causal, src_key_padding_mask=~attention_mask.bool())
        return SimpleNamespace(last_hidden_state=hidden)


class TinySurface(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(6, 16)

    def forward(self, dense_points, dense_normals, query_points, query_normals):
        return self.projection(torch.cat((query_points + dense_points.mean(1, keepdim=True), query_normals), -1))


def inputs(batch=2, frames=3):
    surface = {
        "dense_points": torch.randn(batch, frames, 12, 3),
        "dense_normals": torch.randn(batch, frames, 12, 3),
        "query_points": torch.randn(batch, frames, 7, 3),
        "query_normals": torch.randn(batch, frames, 7, 3),
        "query_indices": torch.arange(7)[None].expand(batch, -1),
    }
    parents = torch.tensor([[-1, 0, 0, 2], [-1, 0, -2, -2]])[:batch]
    targets = {"target_parents": parents, "joint_mask": parents != -2,
               "frame_mask": torch.ones(batch, frames, dtype=torch.bool),
               "frame_target_joints": 9 * torch.randn(batch, frames, 4, 3)}
    return surface, targets


class VirtualFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(20261012)

    def test_virtual_frame_has_no_mesh_and_temporal_slots_are_aligned(self):
        model = VirtualFrameMotionEncoder(settings()).eval()
        spatial, temporal, hooks = [], [], []
        for block in model.blocks:
            hooks.append(block.pose_inner.register_forward_pre_hook(lambda m, a: spatial.append(tuple(a[0].shape))))
            hooks.append(block.anchor_temporal.register_forward_pre_hook(lambda m, a: temporal.append(tuple(a[0].shape))))
        anchors, shared = model(torch.randn(2, 5, 7, 16))
        self.assertEqual(anchors.shape, (2, 5, 7, 16))
        self.assertEqual(shared.shape, (2, 2, 16))
        self.assertEqual(spatial, [(10, 10, 16), (2, 3, 16)] * 3)
        self.assertEqual(temporal, [(6, 6, 16), (14, 5, 16)] * 3)
        for hook in hooks:
            hook.remove()

    def test_frame_permutation_preserves_shared_structure_and_reorders_anchors(self):
        model = VirtualFrameMotionEncoder(settings()).eval()
        features = torch.randn(2, 4, 7, 16)
        order = torch.tensor([3, 0, 2, 1])
        anchors, shared = model(features)
        permuted, other = model(features[:, order])
        torch.testing.assert_close(permuted, anchors[:, order], rtol=3e-6, atol=3e-6)
        torch.testing.assert_close(other, shared, rtol=3e-6, atol=3e-6)
        changed = model(features * -1)[1]
        self.assertGreater((shared - changed).abs().max().item(), 1e-3)

    def test_final_anchor_route_and_all_components_receive_gradients(self):
        model = VirtualFrameRig(TinySurface(), TinyBackbone(), settings()).train()
        surface, targets = inputs()
        captured = {}
        hook1 = model.motion_encoder.register_forward_hook(lambda m, a, out: captured.update(final=out[0]))
        hook2 = model.positions.register_forward_pre_hook(lambda m, a: captured.update(position_input=a[2]))
        output = model(surface, targets, xyz_weight=2)
        self.assertIs(captured["final"], captured["position_input"])
        self.assertEqual(output["joints"].shape, (2, 3, 4, 3))
        output["loss"].backward()
        for group in (model.surface_tokenizer, model.motion_encoder, model.structure, model.positions):
            gradients = [p.grad for p in group.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
        self.assertGreater(model.motion_encoder.role_token.grad[:, 0].abs().sum().item(), 0)
        self.assertGreater(model.motion_encoder.role_token.grad[:, 1].abs().sum().item(), 0)
        self.assertGreater(model.motion_encoder.register.grad.abs().sum().item(), 0)
        for block in model.motion_encoder.blocks:
            self.assertGreater(block.pose_inner.self_attn.in_proj_weight.grad.abs().sum().item(), 0)
            self.assertGreater(block.anchor_temporal.self_attn.in_proj_weight.grad.abs().sum().item(), 0)
        hook1.remove()
        hook2.remove()

    def test_checkpointing_preserves_outputs_and_gradients(self):
        config = settings()
        checked = VirtualFrameRig(TinySurface(), TinyBackbone(), config).train()
        plain = VirtualFrameRig(TinySurface(), TinyBackbone(), replace(
            config, motion_checkpointing=False, surface_checkpointing=False, position_checkpointing=False)).train()
        plain.load_state_dict(checked.state_dict())
        surface, targets = inputs()
        left, right = checked(surface, targets, xyz_weight=1), plain(surface, targets, xyz_weight=1)
        torch.testing.assert_close(left["joints"], right["joints"])
        left["loss"].backward()
        right["loss"].backward()
        for (name, p), (other_name, q) in zip(checked.named_parameters(), plain.named_parameters()):
            self.assertEqual(name, other_name)
            self.assertEqual(p.grad is None, q.grad is None)
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-6)

    def test_motion_bias_remains_real_anchor_only_and_receives_gradients(self):
        model = VirtualFrameMotionEncoder(settings(motion_evidence_fusion="bias", motion_evidence_heads=4)).train()
        features = torch.randn(2, 3, 7, 16, requires_grad=True)
        evidence = MotionEvidence(torch.rand(2, 7, 7, 3), torch.rand(2, 7, 5), torch.ones(2, 7, dtype=torch.bool))
        anchors, shared = model(features, motion_evidence=evidence)
        (anchors.square().mean() + (shared * torch.randn_like(shared)).mean()).backward()
        self.assertEqual(anchors.shape, features.shape)
        self.assertEqual(shared.shape, (2, 2, 16))
        for block in model.blocks:
            gradient = block.motion_injection.bias_weights.grad
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_depth_attention_variants_and_unbounded_output(self):
        for depth, use_self in ((1, True), (2, True), (4, True), (2, False)):
            localizer = ContinuousJointLocalizer(settings(position_depth=depth, position_self_attention=use_self))
            self.assertEqual(len(localizer.layers), depth)
            self.assertEqual(localizer.layers[0].self_attention is not None, use_self)
            with torch.no_grad():
                localizer.xyz_head.weight.zero_()
                localizer.xyz_head.bias.fill_(7)
            xyz = localizer(torch.randn(2, 4, 16), torch.ones(2, 4, dtype=torch.bool),
                            torch.randn(2, 3, 7, 16), torch.randn(2, 3, 7, 3))
            torch.testing.assert_close(xyz, torch.full_like(xyz, 7))

    def test_loss_is_per_asset_masked_mean_not_sum_over_frames(self):
        prediction = torch.zeros(2, 3, 4, 3, requires_grad=True)
        target = torch.ones_like(prediction)
        target[1] = 2
        joints = torch.tensor([[True, True, True, True], [True, True, False, False]])
        frames = torch.tensor([[True, True, True], [True, False, False]])
        valid = frames[:, :, None, None] & joints[:, None, :, None]
        target = torch.where(valid, target, float("nan"))
        loss = coordinate_regression_loss(prediction, target, joints, frames)
        torch.testing.assert_close(loss, torch.tensor([1., 2.]))
        loss.mean().backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(prediction.grad[1, 1:].abs().sum().item(), 0)
        repeated = coordinate_regression_loss(prediction[:1].repeat(1, 2, 1, 1), target[:1].repeat(1, 2, 1, 1),
                                             joints[:1], frames[:1].repeat(1, 2))
        torch.testing.assert_close(repeated, loss[:1])

    def test_head_keeps_shared_features_and_does_not_mix_assets_or_frames(self):
        model = ContinuousJointLocalizer(settings()).eval()
        shared = torch.randn(2, 4, 16, requires_grad=True)
        before = shared.detach().clone()
        anchors, points = torch.randn(2, 3, 7, 16), torch.randn(2, 3, 7, 3)
        mask = torch.ones(2, 4, dtype=torch.bool)
        result = model(shared, mask, anchors, points)
        for asset in range(2):
            for frame in range(3):
                single = model(shared[asset:asset + 1], mask[asset:asset + 1],
                               anchors[asset:asset + 1, frame:frame + 1], points[asset:asset + 1, frame:frame + 1])
                torch.testing.assert_close(single, result[asset:asset + 1, frame:frame + 1], atol=2e-6, rtol=2e-6)
        result.square().mean().backward()
        torch.testing.assert_close(shared.detach(), before)
        self.assertGreater(shared.grad.abs().sum().item(), 0)

    def test_generation_terminates_naturally_and_localizes_all_input_frames(self):
        model = VirtualFrameRig(TinySurface(), TinyBackbone(), settings()).eval()
        with torch.no_grad():
            model.structure.head.weight.zero_()
            model.structure.head.bias.zero_()
            model.structure.head.bias[model.structure.tokens.eos] = 10
        surface, _ = inputs(batch=1)
        result = model.generate(surface)
        self.assertTrue(result["success"])
        self.assertEqual(result["parents"], [-1])
        self.assertEqual(result["joints"].shape, (1, 3, 1, 3))
        reused = model.localize(result["skeleton_sequence"], surface)
        torch.testing.assert_close(reused, result["joints"])


if __name__ == "__main__":
    unittest.main()
