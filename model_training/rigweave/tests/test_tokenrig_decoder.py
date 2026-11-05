from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from rigweave.dynamic_rig.data import _make_token_input
from rigweave.dynamic_rig.virtual_query_xyz import VirtualQueryXYZConfig, VirtualQueryXYZRig
from tokenrig_decoder import NativeTokenRigProjection, TokenRigSkeletonTokenizer, initialize_native_motion_identity
from rigweave.dynamic_rig.virtual_path_ablation import VirtualPathAblationRig


def test_native_projection_lift_preserves_pretrained_prefix():
    torch.manual_seed(29)
    native = nn.Sequential(nn.Linear(4, 6), nn.RMSNorm(6))
    projection = NativeTokenRigProjection(8, 4, 6)
    projection.load_native(native[0].weight, native[0].bias, native[1].weight)
    value = torch.randn(2, 7, 4)
    padded = torch.nn.functional.pad(value, (0, 4))
    torch.testing.assert_close(projection(padded), native(value), rtol=1e-5, atol=1e-6)
    assert torch.count_nonzero(projection.weight[:, 4:]) == 0
    full = torch.randn(2, 7, 8, requires_grad=True)
    projection(full).square().sum().backward()
    assert projection.weight.grad[:, 4:].norm() > 0
    assert projection.normalization.weight.grad.norm() > 0


def test_native_projection_accepts_released_bf16_weights_for_fp32_audit():
    native = nn.Sequential(nn.Linear(4, 6), nn.RMSNorm(6)).to(torch.bfloat16)
    projection = NativeTokenRigProjection(8, 4, 6)
    projection.load_native(native[0].weight, native[0].bias, native[1].weight)
    value = torch.randn(2, 7, 4)
    expected = torch.nn.functional.rms_norm(torch.nn.functional.linear(value,
        native[0].weight.float(), native[0].bias.float()), (6,), native[1].weight.float())
    torch.testing.assert_close(projection(torch.nn.functional.pad(value, (0, 4))),
                               expected, atol=1e-6, rtol=1e-5)


def test_native_motion_identity_preserves_affine_basis_and_trains_outputs():
    torch.manual_seed(31)
    config = VirtualQueryXYZConfig(dim=8, heads=2, query_tokens=4, motion_depth=2, fusion_depth=2,
        motion_evidence_fusion='off', motion_evidence_heads=0, fusion_mode='real_queries_virtual')
    decoder = Qwen3ForCausalLM(Qwen3Config(vocab_size=16, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=4))
    norm = nn.LayerNorm(4)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([.7, .9, 1.2, 1.3]))
        norm.bias.copy_(torch.tensor([-.2, .1, .3, -.1]))
    encoder = nn.Module()
    encoder.width = 4
    encoder.encoder = nn.Module()
    encoder.encoder.ln_post = norm
    surface = nn.Module()
    surface.michelangelo_encoder = encoder
    model = VirtualQueryXYZRig(surface, decoder, config, auxiliary_vocab_size=16)
    model.decoder_condition_projection = NativeTokenRigProjection(8, 4, 8)
    native = nn.Sequential(nn.Linear(4, 8), nn.RMSNorm(8))
    model.decoder_condition_projection.load_native(native[0].weight, native[0].bias, native[1].weight)
    initialize_native_motion_identity(model)
    wrapped = VirtualPathAblationRig(model, virtual_enabled=False)
    raw = torch.nn.functional.layer_norm(torch.randn(1, 2, 4, 4), (4,), eps=0.)
    native_features = raw * norm.weight + norm.bias
    lifted = surface.output_proj(native_features)
    frames, virtual = wrapped._encode_real_only(lifted, None)
    condition = wrapped.conditions(frames, virtual)
    actual = model.decoder_condition_projection(condition)
    torch.testing.assert_close(actual, native(native_features), atol=5e-5, rtol=5e-5)
    actual[..., 0].mean().backward()
    assert model.motion_encoder.blocks[0].anchor_temporal.self_attn.out_proj.weight.grad.norm() > 0
    assert model.frame_fusion[0].ffn[-1].weight.grad.norm() > 0
    assert surface.output_proj.weight.grad.norm() > 0


def test_native_skeleton_adapter_keeps_xyz_and_parent_identity():
    native = SimpleNamespace(tokenize=lambda item: item)
    tokenizer = TokenRigSkeletonTokenizer(native, SimpleNamespace)
    xyz = np.asarray([[0., 0., 0.], [.5, 0., 0.], [0., .5, 0.]])
    parents = [None, 0, 0]
    encoded = _make_token_input(xyz, parents, 'articulation', tokenizer, None)
    assert encoded.joints is xyz
    assert encoded.parents == parents
    assert encoded.cls == 'articulation'


@pytest.mark.parametrize('prefix_tokens', [None, 2])
def test_qwen_prefix_projection_gradients_and_cache_parity(prefix_tokens):
    decoder = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        pad_token_id=0, bos_token_id=1, eos_token_id=2))
    config = VirtualQueryXYZConfig(dim=24, heads=2, query_tokens=4, motion_depth=1, fusion_depth=1,
                                   motion_evidence_fusion='off', motion_evidence_heads=0)
    model = VirtualQueryXYZRig(nn.Identity(), decoder, config, auxiliary_vocab_size=32,
                               project_decoder_condition=True)
    model.decoder_condition_tokens = prefix_tokens
    condition, ids = torch.randn(1, 4, 24), torch.tensor([[1, 4, 8, 2]])
    projected = model.decoder_prefix(condition, ids)
    assert projected.shape == (1, (prefix_tokens or 4) + 4, 16)
    model.main_logits(condition, ids, torch.ones_like(ids)).square().mean().backward()
    assert model.decoder_condition_projection.weight.grad.norm() > 0
    model.eval()
    with torch.no_grad():
        full = model.main_logits(condition, ids, torch.ones_like(ids))
        prefix = model.decoder_prefix(condition, ids[:, :2])
        cached = decoder(inputs_embeds=prefix, use_cache=True)
        torch.testing.assert_close(cached.logits[:, -1], full[:, 1])
        for slot in range(2, ids.shape[1]):
            cached = decoder(input_ids=ids[:, slot:slot+1], past_key_values=cached.past_key_values,
                             use_cache=True)
            torch.testing.assert_close(cached.logits[:, -1], full[:, slot])
