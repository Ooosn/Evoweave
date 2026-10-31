import sys
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from transformers import OPTConfig, OPTForCausalLM

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'model_training/rigweave/scripts'), str(ROOT/'model_training/rigweave/src')]
import diagnostic_decoding as decoding


class Tokenizer:
    num_discrete, token_id_branch, bos, eos, pad = 256, 256, 257, 258, 259
    part_token_to_name = {261: 'body'}
    def cls_name_to_token(self, name):
        return 266
    def next_posible_token(self, ids):
        assert list(ids[:2]) == [257, 266]
        suffix = list(ids[2:])
        if not suffix:
            return list(range(256))+[258, 261]
        if suffix[-1] >= 256:
            return list(range(256))
        last_special = max([i for i, t in enumerate(suffix) if t >= 256], default=-1)
        count = len(suffix)-1-last_special
        return list(range(256)) if count % 3 else list(range(256))+[256, 258, 261]


class DiagnosticDecodingTests(unittest.TestCase):
    def test_original_policy_delegates_without_changes(self):
        with patch.object(decoding.official, 'generate_condition', return_value={'sentinel': 1}) as call:
            self.assertEqual(decoding.generate(None, None, None, seed=7, policy='official'), {'sentinel': 1})
            call.assert_called_once_with(None, None, None, seed=7)

    def test_partial_branch_cannot_end_or_start_another_branch(self):
        tokenizer = Tokenizer()
        processor = decoding.strict_processor(tokenizer, torch.tensor([257, 266]))
        for suffix in ([], [1], [1, 2], [1, 2, 3, 256], [1, 2, 3, 256, 4, 5, 6]):
            output = processor(torch.tensor([suffix], dtype=torch.long), torch.zeros(1, 267))
            self.assertTrue(torch.isneginf(output[0, tokenizer.eos]))
            if suffix:
                self.assertTrue(torch.isneginf(output[0, tokenizer.token_id_branch]))
        full = [1, 2, 3, 256, 4, 5, 6, 2, 3, 4]
        output = processor(torch.tensor([full]), torch.zeros(1, 267))
        self.assertTrue(torch.isfinite(output[0, tokenizer.eos]))

    def test_invalid_prefix_is_rejected_not_repaired(self):
        with self.assertRaises(ValueError):
            decoding.coordinate_state([1, 2, 3, 256, 4, 5, 6, 258], Tokenizer())
        with self.assertRaises(ValueError):
            decoding.coordinate_state([1, 2, 261], Tokenizer())
        self.assertEqual(decoding.coordinate_state([1, 2, 3, 258, 259], Tokenizer()), (0, True, True))

    def test_policy_changes_are_explicit_and_bounded(self):
        self.assertEqual(decoding.options('strict_sampling'), decoding.official.OFFICIAL_OPTIONS)
        deterministic = decoding.options('strict_beam')
        self.assertFalse(deterministic['do_sample'])
        self.assertEqual(deterministic['num_beams'], 15)
        self.assertEqual(deterministic['max_new_tokens'], 2048)
        self.assertNotIn('temperature', deterministic)
        with self.assertRaises(ValueError):
            decoding.options('unapproved')

    def test_small_real_decoder_preserves_rng_and_coordinate_groups(self):
        torch.set_num_threads(1)
        decoder = OPTForCausalLM(OPTConfig(vocab_size=267, hidden_size=16, word_embed_proj_dim=16,
            ffn_dim=32, num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=64,
            bos_token_id=257, eos_token_id=258, pad_token_id=259, _attn_implementation='eager')).eval()
        condition = torch.randn(1, 4, 16)
        state = torch.get_rng_state().clone()
        settings = dict(decoding.official.OFFICIAL_OPTIONS, max_new_tokens=18)
        with patch.object(decoding.official, 'OFFICIAL_OPTIONS', settings):
            for policy in ('strict_sampling', 'strict_beam'):
                a = decoding.generate(decoder, condition, Tokenizer(), seed=9, policy=policy)
                b = decoding.generate(decoder, condition, Tokenizer(), seed=9, policy=policy)
                self.assertEqual(a, b)
                self.assertTrue(torch.equal(state, torch.get_rng_state()))
                self.assertLessEqual(len(a['tokens']), 20)
                if a['success']:
                    self.assertEqual(decoding.coordinate_state(a['tokens'][2:], Tokenizer())[0], 0)

    def test_diagnosis_config_binds_policies_and_forbids_retries(self):
        from predicted_reference_protocol import validate_config, scientific_sha
        config = json.loads((ROOT/'model_training/experiments/inference_diagnosis_20261031.json').read_text())
        validate_config(config, 'audit')
        for field, value in [('retry_failures', True), ('select_using_GT', True),
                             ('require_exact_baseline_replay', False), ('optimizer_updates', 1)]:
            changed = copy.deepcopy(config)
            changed['inference_diagnosis'][field] = value
            with self.assertRaises(ValueError):
                validate_config(changed, 'capacity')
            self.assertNotEqual(scientific_sha(changed), scientific_sha(config))
        changed = copy.deepcopy(config)
        changed['inference_diagnosis']['policies'].append('unreviewed')
        with self.assertRaises(ValueError):
            validate_config(changed, 'capacity')


if __name__ == '__main__':
    unittest.main()
