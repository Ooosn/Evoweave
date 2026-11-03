from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from rigweave.dynamic_rig.data import _make_token_input
from rigweave.dynamic_rig.virtual_query_xyz import VirtualQueryXYZConfig, VirtualQueryXYZRig
from tokenrig_decoder import TokenRigSkeletonTokenizer


def test_native_skeleton_adapter_keeps_xyz_and_parent_identity():
    native = SimpleNamespace(tokenize=lambda item: item)
    tokenizer = TokenRigSkeletonTokenizer(native, SimpleNamespace)
    xyz = np.asarray([[0., 0., 0.], [.5, 0., 0.], [0., .5, 0.]])
    parents = [None, 0, 0]
    encoded = _make_token_input(xyz, parents, 'articulation', tokenizer, None)
    assert encoded.joints is xyz
    assert encoded.parents == parents
    assert encoded.cls == 'articulation'


def test_qwen_prefix_projection_gradients_and_cache_parity():
    decoder = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        pad_token_id=0, bos_token_id=1, eos_token_id=2))
    config = VirtualQueryXYZConfig(dim=24, heads=2, query_tokens=4, motion_depth=1, fusion_depth=1,
                                   motion_evidence_fusion='off', motion_evidence_heads=0)
    model = VirtualQueryXYZRig(nn.Identity(), decoder, config, auxiliary_vocab_size=32,
                               project_decoder_condition=True)
    condition, ids = torch.randn(1, 4, 24), torch.tensor([[1, 4, 8, 2]])
    projected = model.decoder_prefix(condition, ids)
    assert projected.shape == (1, 8, 16)
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
