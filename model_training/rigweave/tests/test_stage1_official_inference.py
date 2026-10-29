import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from transformers import OPTConfig, OPTForCausalLM

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'model_training/rigweave/scripts'), str(ROOT/'model_training/rigweave/src')]
import stage1_official_inference as inference
from predicted_reference_protocol import validate_config, scientific_sha


class Tokenizer:
    bos, pad, eos = 0, 1, 2
    def cls_name_to_token(self, name):
        assert name == 'articulationxl'
        return 3
    def next_posible_token(self, ids):
        assert list(ids[:2]) == [0, 3]
        return [2, 4, 5, 6, 7, 8]


class OfficialInferenceTests(unittest.TestCase):
    def test_visual_expansion_keeps_recipe_and_requires_inherited_capacity(self):
        config = json.loads((ROOT/'model_training/experiments/predicted_reference_visual12_20261029.json').read_text())
        validate_config(config, 'capacity')
        self.assertEqual(config['evaluation']['assets'], 12)
        self.assertFalse(set(config['official_smoke']['case_indices']) & {2, 3, 64, 127})
        self.assertFalse(config['official_smoke']['full_beam_capacity_first'])
        config['official_smoke']['inherited_official_capacity'] = None
        with self.assertRaises(ValueError):
            validate_config(config, 'capacity')

    def test_real_beam_sampling_prompt_rng_and_capacity(self):
        torch.set_num_threads(1)
        decoder = OPTForCausalLM(OPTConfig(vocab_size=16, hidden_size=16, word_embed_proj_dim=16,
            ffn_dim=32, num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=64,
            bos_token_id=0, pad_token_id=1, eos_token_id=2, _attn_implementation='eager')).eval()
        condition = torch.randn(1, 4, 16)
        state = torch.get_rng_state().clone()
        options = dict(inference.OFFICIAL_OPTIONS, max_new_tokens=8)
        with patch.object(inference, 'OFFICIAL_OPTIONS', options):
            a = inference.generate_condition(decoder, condition, Tokenizer(), seed=7)
            b = inference.generate_condition(decoder, condition, Tokenizer(), seed=7)
            stress = inference.generate_condition(decoder, condition, Tokenizer(), seed=7, capacity_only=True)
        self.assertEqual(a, b)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(a['tokens'][:2], [0, 3])
        self.assertEqual(len(stress['tokens']), 10)
        self.assertFalse(stress['success'])
        self.assertNotIn(2, stress['tokens'][2:])
        self.assertTrue(stress['capacity_only'])

    def test_exact_recipe_and_seed_identity(self):
        self.assertEqual(inference.OFFICIAL_OPTIONS, dict(max_new_tokens=2048, num_return_sequences=1,
            num_beams=15, do_sample=True, top_k=5, top_p=.95, repetition_penalty=3., temperature=1.5))
        self.assertEqual(inference.reference_seed('a', 2), inference.reference_seed('a', 2))
        self.assertNotEqual(inference.reference_seed('a', 2), inference.reference_seed('a', 3))
        self.assertNotEqual(inference.reference_seed('a', 2), inference.reference_seed('b', 2))

    def test_smoke_config_rejects_greedy_reuse_and_static_fallback(self):
        config = json.loads((ROOT/'model_training/experiments/predicted_reference_official_smoke_20261029.json').read_text())
        validate_config(config, 'audit')
        validate_config(config, 'capacity')
        for key, value in [('static_fallback', True), ('case_indices', [0, 1, 2, 3])]:
            changed = copy.deepcopy(config)
            changed['official_smoke'][key] = value
            with self.assertRaises(ValueError):
                validate_config(changed, 'capacity')
            self.assertNotEqual(scientific_sha(config), scientific_sha(changed))
        changed = copy.deepcopy(config)
        changed['official_smoke']['generation_kwargs']['num_beams'] = 1
        with self.assertRaises(ValueError):
            validate_config(changed, 'capacity')
        changed = copy.deepcopy(config)
        changed['evaluation']['stage1_cache_policy'] = 'reuse_only_when_all_input_identities_match_else_generate_same_surface'
        with self.assertRaises(ValueError):
            validate_config(changed, 'capacity')


if __name__ == '__main__':
    unittest.main()
