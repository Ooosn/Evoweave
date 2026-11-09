import copy
from dataclasses import replace

import pytest
import torch
from torch import nn

from rigweave.dynamic_rig.anchor_joint_binding import (
    AnchorJointBinding, AnchorJointConfig, SharedAnchorJointRig,
)
from rigweave.dynamic_rig.anchor_stage2 import AnchorStage2, AnchorStage2Config, AnchorStage2Encoder
from rigweave.dynamic_rig.virtual_path_ablation import VirtualPathAblationRig
from rigweave.dynamic_rig.virtual_query_xyz import VirtualQueryXYZConfig, VirtualQueryXYZRig


class Surface(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(6, 16)

    def forward(self, dense, normals, query, query_normals):
        return self.projection(torch.cat((query, query_normals), dim=-1))


class DecoderStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 16)

    def get_input_embeddings(self):
        return self.embedding


def sample():
    torch.manual_seed(5)
    dense = torch.randn(2, 3, 12, 3)
    surface = {'dense_points': dense, 'dense_normals': torch.randn_like(dense),
               'query_points': dense[:, :, :4].clone(), 'query_normals': torch.randn(2, 3, 4, 3),
               'query_indices': torch.arange(4)[None].expand(2, -1).clone()}
    geometry = {'frame_target_joints': torch.randn(2, 3, 3, 3)*.1,
                'target_parents': torch.tensor([[-1, 0, 1], [-1, 0, 0]]),
                'joint_mask': torch.ones(2, 3, dtype=torch.bool)}
    return surface, geometry, torch.tensor([0, 2])


def head():
    return AnchorJointBinding(AnchorJointConfig(input_dim=16, dim=16, heads=4, reference_depth=1,
                                                target_depth=1, gradient_checkpointing=False))


@pytest.mark.parametrize('fusion', ['off', 'bias'])
def test_real_only_path_and_loss_match_legacy(fusion):
    args = dict(dim=16, heads=4, query_tokens=4, register_tokens=2, max_frames=4,
                motion_evidence_fusion=fusion, motion_evidence_heads=4 if fusion == 'bias' else 0,
                gradient_checkpointing=False, temporal_chunk_size=7)
    old_config = VirtualQueryXYZConfig(**args, fusion_mode='real_queries_virtual',
                                       motion_depth=2, auxiliary_depth=1)
    original = VirtualQueryXYZRig(Surface(), DecoderStub(), old_config, auxiliary_vocab_size=8)
    legacy = SharedAnchorJointRig(VirtualPathAblationRig(original, virtual_enabled=False), head(),
                                 ar_enabled=False)
    encoder = AnchorStage2Encoder(copy.deepcopy(original.surface_tokenizer),
                                  AnchorStage2Config(2, 2, **args))
    encoder.blocks.load_state_dict(original.motion_encoder.blocks.state_dict(), strict=True)
    encoder.norm.load_state_dict(original.motion_encoder.norm.state_dict(), strict=True)
    with torch.no_grad():
        encoder.role_token.copy_(original.motion_encoder.role_token[:, 1:])
        encoder.register.copy_(original.motion_encoder.register)
        if fusion == 'bias':
            for old, new in zip(original.motion_encoder.blocks, encoder.blocks, strict=True):
                old.motion_injection.bias_weights.normal_()
                new.motion_injection.bias_weights.copy_(old.motion_injection.bias_weights)
    new = AnchorStage2(encoder, copy.deepcopy(legacy.head))
    surface, geometry, ref = sample()
    for training in (True, False):
        new.train(training)
        legacy.train(training)
        x, _ = new.backbone.encode(surface)
        y, _ = legacy.backbone.encode(surface)
        torch.testing.assert_close(x, y, rtol=0, atol=0)
        a = new(surface, geometry, ref, 10., .25)
        b = legacy(surface, None, geometry, None, None, ref, 10., .25)
        for k in a:
            torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)


def test_two_spatial_layers_have_gradient_and_share_common_initialization():
    args = dict(dim=16, heads=4, query_tokens=4, register_tokens=2, max_frames=4,
                motion_evidence_fusion='bias', motion_evidence_heads=4, temporal_chunk_size=7)
    torch.manual_seed(3)
    one = AnchorStage2Encoder(Surface(), AnchorStage2Config(3, 3, **args))
    torch.manual_seed(3)
    two = AnchorStage2Encoder(Surface(), AnchorStage2Config(4, 2, **args))
    for name, state in two.blocks.state_dict().items():
        torch.testing.assert_close(state, one.blocks.state_dict()[name], rtol=0, atol=0)
    assert sum(isinstance(m, nn.TransformerEncoderLayer) for m in one.modules()) == 6
    assert sum(isinstance(m, nn.TransformerEncoderLayer) for m in two.modules()) == 6
    model = AnchorStage2(two, head())
    surface, geometry, ref = sample()
    loss = model(surface, geometry, ref, 10., .25)['loss_per_asset'].mean()
    loss.backward()
    assert not any(isinstance(m, (DecoderStub, nn.Embedding)) for m in model.backbone.modules())
    for layer in two.extra_spatial:
        assert torch.isfinite(layer.layer.linear1.weight.grad).all()
        assert layer.layer.linear1.weight.grad.norm() > 0
        assert torch.isfinite(layer.injection.bias_weights.grad).all()


@pytest.mark.parametrize('schedule', [(2, 2), (4, 2)])
def test_independent_asset_recomputation_keeps_values_and_gradients(schedule):
    args = dict(dim=16, heads=4, query_tokens=4, register_tokens=2, max_frames=4,
                motion_evidence_fusion='bias', motion_evidence_heads=4, temporal_chunk_size=7)
    plain = AnchorStage2(AnchorStage2Encoder(Surface(), AnchorStage2Config(*schedule, **args)), head())
    with torch.no_grad():
        for name, weight in plain.named_parameters():
            if name.endswith('bias_weights'):
                weight.normal_()
    chunked = copy.deepcopy(plain)
    chunked.backbone.config = replace(chunked.backbone.config, checkpoint_asset_chunk_size=1)
    surface, geometry, ref = sample()
    a = plain(surface, geometry, ref, 10., .25)['loss_per_asset']
    b = chunked(surface, geometry, ref, 10., .25)['loss_per_asset']
    torch.testing.assert_close(a, b)
    a.mean().backward()
    b.mean().backward()
    for x, y in zip(plain.parameters(), chunked.parameters(), strict=True):
        assert (x.grad is None) == (y.grad is None)
        if x.grad is not None:
            torch.testing.assert_close(x.grad, y.grad, atol=1e-6, rtol=1e-4)
