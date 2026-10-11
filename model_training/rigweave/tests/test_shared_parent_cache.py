from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers import OPTConfig, OPTModel

from rigweave.dynamic_rig.shared_structure import SharedParentDecoder, SharedStructureConfig


class ScriptedHead(nn.Module):
    def __init__(self, vocabulary, tokens):
        super().__init__()
        self.vocabulary = vocabulary
        self.tokens = iter(tokens)

    def forward(self, hidden):
        result = hidden.new_full((*hidden.shape[:-1], self.vocabulary), -10)
        result[..., next(self.tokens)] = 10
        return result


class SharedParentCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self):
        config = OPTConfig(vocab_size=32, hidden_size=16, word_embed_proj_dim=16, ffn_dim=32,
                           num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=64,
                           dropout=0., attention_dropout=0., do_layer_norm_before=True)
        config._attn_implementation = "eager"
        return SharedParentDecoder(OPTModel(config), SharedStructureConfig(dim=16, heads=4, max_joints=8)).eval()

    def test_cached_post_token_features_equal_full_prefix_and_teacher_forcing(self):
        torch.manual_seed(42)
        model = self.model()
        memory = torch.randn(1, 4, 16)
        tokens = [model.tokens.root, 0, 1, model.tokens.eos]
        with patch.object(model, "head", ScriptedHead(model.tokens.vocab_size, tokens)):
            full = model.generate(memory)
        with patch.object(model, "head", ScriptedHead(model.tokens.vocab_size, tokens)):
            cached = model.generate_cached(memory)
        self.assertEqual(cached["tokens"], full["tokens"])
        self.assertEqual(cached["parents"], [-1, 0, 1])
        torch.testing.assert_close(cached["joint_features"], full["joint_features"], rtol=2e-5, atol=2e-6)
        teacher = model.teacher_force(memory, torch.tensor([[-1, 0, 1]]), torch.ones(1, 3, dtype=torch.bool))
        torch.testing.assert_close(cached["joint_features"], teacher["joint_features"], rtol=2e-5, atol=2e-6)

    def test_cache_does_not_force_eos_at_the_joint_limit(self):
        model = self.model()
        memory = torch.randn(1, 4, 16)
        tokens = [model.tokens.root, 0, 1, 1]
        with patch.object(model, "head", ScriptedHead(model.tokens.vocab_size, tokens)):
            result = model.generate_cached(memory, joint_limit=3)
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "max_joints")
        self.assertEqual(result["tokens"], tokens[:3])

    def test_random_head_greedy_tokens_match_uncached_generation(self):
        for seed in range(3):
            torch.manual_seed(seed)
            model = self.model()
            memory = torch.randn(1, 4, 16)
            full, cached = model.generate(memory), model.generate_cached(memory)
            self.assertEqual(full["tokens"], cached["tokens"])
            self.assertEqual(full["status"], cached["status"])
            if full["success"]:
                torch.testing.assert_close(full["joint_features"], cached["joint_features"], rtol=2e-5, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
