"""Bounded CPU coverage of the real-frame model, OPT cache and vocabulary lift."""
from copy import deepcopy
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn
from transformers import OPTConfig, OPTForCausalLM

os.environ.setdefault('RIGWEAVE_DISABLE_OPEN3D', '1')
os.environ.setdefault('RIGWEAVE_DISABLE_LIGHTNING_IMPORT', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rigweave.dynamic_rig.id_xyz_consistency import (
    IdXYZConfig, IdXYZRig, extend_decoder_vocabulary,
)
from rigweave.dynamic_rig.id_xyz_tokens import IdXYZTokens
from rigweave.dynamic_rig.motion_encoder import AnchorWiseAlternatingMotionEncoder


DIM = 16
QUERY_TOKENS = 3
SURFACE_KEYS = ('dense_points', 'dense_normals', 'query_points', 'query_normals')


def codec():
    return IdXYZTokens(271, tuple(range(256)), bos=257, eos=258, pad=259,
                       branch=256, class_token=270, max_joints=8, max_sequence_tokens=64)


def tiny_opt(*, tied=True, bias=False):
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(921)
        config = OPTConfig(vocab_size=271, hidden_size=DIM, word_embed_proj_dim=DIM,
                           ffn_dim=32, num_hidden_layers=2, num_attention_heads=2,
                           max_position_embeddings=96, dropout=0., attention_dropout=0.,
                           layerdrop=0., tie_word_embeddings=tied,
                           bos_token_id=257, eos_token_id=258, pad_token_id=259)
        model = OPTForCausalLM(config)
        if bias:
            head = nn.Linear(DIM, 271, bias=True)
            with torch.no_grad():
                head.weight.copy_(model.get_output_embeddings().weight)
                head.bias.copy_(torch.linspace(-.2, .2, 271))
            model.set_output_embeddings(head)
    return model


class TinySurfaceEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(12, DIM), nn.Tanh(), nn.Linear(DIM, DIM))

    def forward(self, dense_points, dense_normals, query_points, query_normals):
        count = query_points.shape[1]
        context = torch.cat((dense_points.mean(1), dense_normals.mean(1)), dim=-1)
        context = context[:, None].expand(-1, count, -1)
        return self.projection(torch.cat((context, query_points, query_normals), dim=-1))


def tiny_model(*, checkpointing=False, evidence='off', decoder=None):
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(719)
        if decoder is None:
            decoder = tiny_opt()
            extend_decoder_vocabulary(decoder, codec().vocab_size, seed=123)
        config = IdXYZConfig(dim=DIM, heads=2, query_tokens=QUERY_TOKENS,
                             register_tokens=2, motion_depth=2, max_frames=4,
                             gradient_checkpointing=checkpointing,
                             motion_evidence_fusion=evidence,
                             motion_evidence_heads=2 if evidence == 'bias' else 0)
        return IdXYZRig(TinySurfaceEncoder(), decoder, config)


def surface_fixture(*, batch=2, frames=3, requires_grad=False):
    generator = torch.Generator().manual_seed(293)
    result = {}
    for key in SURFACE_KEYS:
        count = 12 if key.startswith('dense') else QUERY_TOKENS
        result[key] = torch.randn(batch, frames, count, 3, generator=generator)
        result[key].requires_grad_(requires_grad)
    result['query_indices'] = torch.tensor([0, 4, 8]).expand(batch, -1).clone()
    return result


def target_fixture(*, frames=3):
    tokens = codec()
    encoded = []
    for parents in ([-1, 0, 0], [-1, 0]):
        joints = np.linspace(-.6, .6, frames * len(parents) * 3, dtype=np.float32)
        encoded.append(tokens.encode(parents, joints.reshape(frames, len(parents), 3)))
    length = max(item['input_ids'].shape[1] for item in encoded) + 2
    ids = torch.full((2, frames, length), tokens.pad, dtype=torch.long)
    attention = torch.zeros_like(ids, dtype=torch.bool)
    structure = torch.zeros_like(attention)
    for asset, item in enumerate(encoded):
        end = item['input_ids'].shape[1]
        ids[asset, :, :end] = torch.from_numpy(item['input_ids'])
        attention[asset, :, :end] = True
        structure[asset, :, :end] = torch.from_numpy(item['structure_mask'])
    return {'input_ids': ids, 'attention_mask': attention, 'structure_mask': structure,
            'coordinate_mask': (ids < 256) & attention,
            'frame_mask': torch.ones(2, frames, dtype=torch.bool)}


class ScriptedDecoder(nn.Module):
    """High illegal scores ensure generate really applies the grammar each step."""
    def __init__(self, sequence, *, keep_cache=True, nonfinite=False):
        super().__init__()
        self.embedding = nn.Embedding(codec().vocab_size, DIM)
        with torch.no_grad():
            self.embedding.weight.zero_()
            self.embedding.weight[:, 0] = torch.arange(codec().vocab_size)
        self.config = SimpleNamespace(max_position_embeddings=96)
        self.sequence = list(sequence)
        self.keep_cache = keep_cache
        self.nonfinite = nonfinite
        self.calls = []

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, *, inputs_embeds=None, input_ids=None, attention_mask,
                past_key_values=None, use_cache, return_dict):
        if not use_cache or not return_dict:
            raise AssertionError('generation must request its cache and named output')
        if inputs_embeds is not None:
            history = inputs_embeds[0, -2:, 0].round().long().tolist()
            self.condition_length = inputs_embeds.shape[1] - 2
            length = inputs_embeds.shape[1]
            if past_key_values is not None or input_ids is not None:
                raise AssertionError('prefill must use only continuous condition and BOS/CLS')
        else:
            if input_ids.shape != (1, 1) or past_key_values is None:
                raise AssertionError('cached decoding must consume exactly one new token')
            history = list(past_key_values) + input_ids[0].tolist()
            length = 1
        if history != self.sequence[:len(history)]:
            raise AssertionError('generated history differs from scripted legal sequence')
        if attention_mask.shape != (1, self.condition_length + len(history)):
            raise AssertionError('cache attention mask lost a prefix or generated token')
        if not bool(attention_mask.all()):
            raise AssertionError('unexpected masked position in this unpadded generation')
        self.calls.append((inputs_embeds is not None, length, attention_mask.shape[1]))
        logits = self.embedding.weight.new_full((1, length, codec().vocab_size), -20.)
        logits[0, -1, codec().pad] = 100.
        logits[0, -1, self.sequence[len(history)]] = 10.
        if self.nonfinite:
            logits[0, -1, 0] = float('nan')
        return SimpleNamespace(logits=logits,
                               past_key_values=tuple(history) if self.keep_cache else None)


class IdXYZModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def assert_nonzero_finite(self, value):
        self.assertIsNotNone(value)
        self.assertTrue(bool(torch.isfinite(value).all()))
        self.assertGreater(float(value.detach().abs().sum()), 0.)

    def test_full_teacher_forcing_is_causal_without_future_leakage(self):
        model = tiny_model().eval()
        ids = target_fixture()['input_ids'][0:1, 0, :17]
        condition = torch.randn(1, QUERY_TOKENS, DIM, generator=torch.Generator().manual_seed(9))
        with torch.no_grad():
            original = model.logits(condition, ids, torch.ones_like(ids))
            for boundary in (0, 5, 11):
                with self.subTest(boundary=boundary):
                    changed = ids.clone()
                    changed[:, boundary + 1:] = (changed[:, boundary + 1:] + 71) % codec().vocab_size
                    perturbed = model.logits(condition, changed, torch.ones_like(changed))
                    torch.testing.assert_close(original[:, :boundary + 1], perturbed[:, :boundary + 1],
                                               rtol=0., atol=1e-7)
                    self.assertGreater(float((original[:, boundary + 1:] - perturbed[:, boundary + 1:]).abs().max()), 1e-5)
                    short = model.logits(condition, ids[:, :boundary + 1], torch.ones_like(ids[:, :boundary + 1]))
                    torch.testing.assert_close(original[:, :boundary + 1], short, rtol=2e-5, atol=2e-6)

    def test_opt_cached_next_token_matches_full_teacher_forcing(self):
        model = tiny_model().eval()
        ids = target_fixture()['input_ids'][:, 0, :10]
        condition = torch.randn(2, QUERY_TOKENS, DIM, generator=torch.Generator().manual_seed(21))
        decoder = model.xyz_decoder
        with torch.no_grad():
            full = model.logits(condition, ids, torch.ones_like(ids))
            prefix = torch.cat((condition, decoder.get_input_embeddings()(ids[:, :2])), dim=1)
            mask = torch.ones(2, prefix.shape[1], dtype=torch.long)
            output = decoder(inputs_embeds=prefix, attention_mask=mask, use_cache=True, return_dict=True)
            torch.testing.assert_close(output.logits[:, -1], full[:, 1], rtol=2e-5, atol=2e-6)
            for index in range(2, ids.shape[1]):
                self.assertIsNotNone(output.past_key_values)
                mask = torch.cat((mask, mask.new_ones(2, 1)), dim=1)
                output = decoder(input_ids=ids[:, index:index + 1], attention_mask=mask,
                                 past_key_values=output.past_key_values, use_cache=True, return_dict=True)
                torch.testing.assert_close(output.logits[:, -1], full[:, index], rtol=2e-5, atol=2e-6)

    def test_only_real_frame_conditions_and_no_extra_model_branches(self):
        model = tiny_model().eval()
        captured = []
        hook = model.motion_encoder.register_forward_hook(lambda _m, _a, output: captured.append(output[1]))
        try:
            with torch.no_grad():
                condition = model.encode(surface_fixture())
        finally:
            hook.remove()
        self.assertEqual(condition.shape, (2, 3, QUERY_TOKENS, DIM))
        torch.testing.assert_close(condition, captured[0][:, :, 3:], rtol=0., atol=0.)
        self.assertEqual(set(dict(model.named_children())), {'surface_tokenizer', 'xyz_decoder', 'motion_encoder'})
        for name in ('virtual_frame', 'frame_fusion', 'auxiliary_decoder'):
            self.assertIs(model.configuration()[name], False)
        self.assertEqual(model.motion_encoder.frame_role_policy, 'all_real')
        self.assertFalse(model.motion_encoder.use_time_embedding)
        self.assertFalse(model.motion_encoder.use_motion_features)
        self.assertTrue(all(block.relation_residual is None for block in model.motion_encoder.blocks))
        for name, _module in model.named_modules():
            self.assertFalse(any(part in name.lower() for part in ('virtual', 'fusion', 'auxiliary')))

    def test_frame_permutation_equivariance_and_loss_invariance(self):
        model = tiny_model().eval()
        surface, targets = surface_fixture(), target_fixture()
        permutation = torch.tensor([2, 0, 1])
        permuted_surface = {key: value[:, permutation] if key in SURFACE_KEYS else value
                            for key, value in surface.items()}
        permuted_targets = {key: value[:, permutation] for key, value in targets.items()}
        with torch.no_grad():
            condition = model.encode(surface)
            permuted_condition = model.encode(permuted_surface)
            torch.testing.assert_close(permuted_condition, condition[:, permutation], rtol=2e-5, atol=2e-6)
            original = model.forward_encoded(condition, targets, consistency_weight=1.)
            permuted = model.forward_encoded(permuted_condition, permuted_targets, consistency_weight=1.)
        for name in ('loss', 'loss_per_asset', 'main_loss', 'consistency_ce', 'consistency_kl',
                     'structure_ce', 'coordinate_ce', 'structure_accuracy', 'coordinate_accuracy'):
            torch.testing.assert_close(original[name], permuted[name], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(original['main_loss_per_frame'][:, permutation],
                                   permuted['main_loss_per_frame'], rtol=2e-5, atol=2e-6)

    def test_all_real_policy_does_not_use_query_only_role_parameters(self):
        model = tiny_model().eval()
        surface = surface_fixture()
        with torch.no_grad():
            original = model.encode(surface)
            model.motion_encoder.role_token[:, 0].add_(1000.)
            same = model.encode(surface)
        torch.testing.assert_close(original, same, rtol=0., atol=0.)
        legacy = AnchorWiseAlternatingMotionEncoder(DIM, depth=1, heads=2)
        self.assertEqual(legacy.frame_role_policy, 'query_first')
        with self.assertRaisesRegex(ValueError, 'frame_role_policy'):
            AnchorWiseAlternatingMotionEncoder(DIM, depth=1, heads=2, frame_role_policy='invalid')

    def test_full_loss_reaches_surface_motion_decoder_and_every_input_frame(self):
        for checkpointing in (False, True):
            with self.subTest(checkpointing=checkpointing):
                model = tiny_model(checkpointing=checkpointing).train()
                surface = surface_fixture(requires_grad=True)
                result = model(surface, target_fixture(), consistency_weight=1.)
                result['loss'].backward()
                self.assert_nonzero_finite(result['loss'])
                for prefix in ('surface_tokenizer', 'motion_encoder.blocks.0.pose_inner',
                               'motion_encoder.blocks.0.anchor_temporal', 'xyz_decoder'):
                    gradients = [parameter.grad for name, parameter in model.named_parameters()
                                 if name.startswith(prefix) and parameter.grad is not None]
                    self.assertTrue(gradients, prefix)
                    self.assert_nonzero_finite(torch.cat([value.flatten() for value in gradients]))
                self.assert_nonzero_finite(model.xyz_decoder.get_input_embeddings().weight.grad[271:])
                for key in SURFACE_KEYS:
                    for frame in range(3):
                        self.assert_nonzero_finite(surface[key].grad[:, frame])

    def test_one_frame_condition_depends_on_all_observed_frames(self):
        model = tiny_model().train()
        surface = surface_fixture(requires_grad=True)
        condition = model.encode(surface)
        probe = torch.randn_like(condition[:, 0])
        (condition[:, 0] * probe).sum().backward()
        for frame in range(3):
            self.assert_nonzero_finite(surface['query_points'].grad[:, frame])

    def test_bias_evidence_path_backpropagates_without_auxiliary_losses(self):
        model = tiny_model(evidence='bias').train()
        result = model(surface_fixture(), target_fixture(), consistency_weight=1.)
        result['loss'].backward()
        for block in model.motion_encoder.blocks:
            self.assert_nonzero_finite(block.motion_injection.bias_weights.grad)
        self.assertFalse(any('aux' in key for key in result))

    def test_lambda_zero_is_main_only_in_value_and_gradient(self):
        model = tiny_model().train()
        condition = torch.randn(2, 3, QUERY_TOKENS, DIM, requires_grad=True)
        result = model.forward_encoded(condition, target_fixture(), consistency_weight=0.)
        torch.testing.assert_close(result['loss'], result['main_loss'], rtol=0., atol=0.)
        self.assertGreater(float(result['consistency_ce'].detach()), 0.)
        parameters = (condition, *model.xyz_decoder.parameters())
        total_gradients = torch.autograd.grad(result['loss'], parameters, retain_graph=True)
        main_gradients = torch.autograd.grad(result['main_loss'], parameters)
        for actual, expected in zip(total_gradients, main_gradients):
            torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
        for weight in (-1., float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, 'consistency weight'):
                model.forward_encoded(condition, target_fixture(), consistency_weight=weight)

    def test_shift_padding_coordinate_metrics_and_ignored_frame_gradient(self):
        model = tiny_model()
        targets = target_fixture()
        targets['frame_mask'][1, 2] = False
        ids = targets['input_ids']
        logits = torch.randn(*ids.shape[:2], ids.shape[-1] - 1, codec().vocab_size,
                             generator=torch.Generator().manual_seed(71), requires_grad=True)
        condition = torch.zeros(2, 3, QUERY_TOKENS, DIM)
        with patch.object(model, 'logits', side_effect=[logits[:, frame] for frame in range(3)]) as calls:
            result = model.forward_encoded(condition, targets, consistency_weight=0.)
        self.assertEqual(calls.call_count, 3)
        for frame, call in enumerate(calls.call_args_list):
            torch.testing.assert_close(call.args[1], ids[:, frame, :-1])
            torch.testing.assert_close(call.args[2], targets['attention_mask'][:, frame, :-1])
        labels = ids[:, :, 1:]
        valid = targets['attention_mask'][:, :, 1:]
        frames = targets['frame_mask']
        nll = -logits.log_softmax(-1).gather(-1, labels[..., None]).squeeze(-1)
        per_frame = (nll * valid).sum(-1) / valid.sum(-1)
        per_asset = (per_frame * frames).sum(-1) / frames.sum(-1)
        torch.testing.assert_close(result['loss_per_asset'], per_asset)
        torch.testing.assert_close(result['main_loss_per_frame'], per_frame)
        for mask_name, metric in (('coordinate_mask', 'coordinate_ce'), ('structure_mask', 'structure_ce')):
            selected = targets[mask_name][:, :, 1:] & valid & frames[..., None]
            torch.testing.assert_close(result[metric], nll[selected].mean())
        result['loss'].backward()
        excluded = ~valid | ~frames[..., None]
        self.assertEqual(float(logits.grad[excluded].abs().sum()), 0.)
        for mask_name in ('structure_mask', 'coordinate_mask'):
            selected = targets[mask_name][:, :, 1:] & valid & frames[..., None]
            self.assert_nonzero_finite(logits.grad[selected])

    def test_full_generation_uses_cache_grammar_and_decodes(self):
        sequence = target_fixture()['input_ids'][0, 0, :17].tolist()
        decoder = ScriptedDecoder(sequence)
        model = tiny_model(decoder=decoder).eval()
        result = model.generate(torch.zeros(1, QUERY_TOKENS, DIM), codec(), max_new_tokens=len(sequence) - 2)
        self.assertTrue(result['success'])
        self.assertEqual(result['status'], 'eos')
        self.assertEqual(result['tokens'], sequence)
        self.assertFalse(result['eos_only_legal_choice'])
        parsed = codec().decode(result['tokens'])
        np.testing.assert_array_equal(parsed['parents'], [-1, 0, 0])
        self.assertEqual(parsed['joints'].shape, (3, 3))
        self.assertEqual(len(decoder.calls), len(sequence) - 2)
        self.assertEqual(decoder.calls[0], (True, QUERY_TOKENS + 2, QUERY_TOKENS + 2))
        self.assertTrue(all(not prefill and length == 1 for prefill, length, _ in decoder.calls[1:]))

    def test_generation_reports_budget_and_rejects_missing_cache_or_nonfinite_scores(self):
        sequence = target_fixture()['input_ids'][0, 0, :17].tolist()
        condition = torch.zeros(1, QUERY_TOKENS, DIM)
        model = tiny_model(decoder=ScriptedDecoder(sequence)).eval()
        result = model.generate(condition, codec(), max_new_tokens=4)
        self.assertFalse(result['success'])
        self.assertEqual(result['status'], 'max_tokens')
        self.assertEqual(result['tokens'], sequence[:6])
        for kwargs, message in (({'keep_cache': False}, 'KV cache'), ({'nonfinite': True}, 'nonfinite')):
            bad_model = tiny_model(decoder=ScriptedDecoder(sequence, **kwargs)).eval()
            with self.assertRaisesRegex(RuntimeError, message):
                bad_model.generate(condition, codec(), max_new_tokens=20)
        with self.assertRaisesRegex(ValueError, 'eval mode'):
            model.train().generate(condition, codec(), max_new_tokens=20)
        with self.assertRaisesRegex(ValueError, 'one asset/frame'):
            model.eval().generate(condition.expand(2, -1, -1), codec(), max_new_tokens=20)

    def test_vocabulary_extension_preserves_every_original_row_and_other_parameters(self):
        for tied, bias in ((True, False), (False, False), (False, True)):
            with self.subTest(tied=tied, bias=bias):
                decoder = tiny_opt(tied=tied, bias=bias)
                old_input = decoder.get_input_embeddings().weight.detach().clone()
                old_output = decoder.get_output_embeddings().weight.detach().clone()
                old_bias = decoder.get_output_embeddings().bias
                if old_bias is not None:
                    old_bias = old_bias.detach().clone()
                old_other = {key: value.clone() for key, value in decoder.state_dict().items()
                             if key not in {'model.decoder.embed_tokens.weight', 'lm_head.weight', 'lm_head.bias'}}
                result = extend_decoder_vocabulary(decoder, codec().vocab_size, seed=13)
                new_input, new_output = decoder.get_input_embeddings(), decoder.get_output_embeddings()
                self.assertTrue(torch.equal(new_input.weight[:271], old_input))
                self.assertTrue(torch.equal(new_output.weight[:271], old_output))
                self.assertEqual(new_input.weight.data_ptr() == new_output.weight.data_ptr(), tied)
                self.assertEqual(new_input.padding_idx, 259)
                self.assertEqual(decoder.config.vocab_size, codec().vocab_size)
                self.assertEqual(result['new_rows'], 8)
                self.assertEqual(result['tied'], tied)
                self.assert_nonzero_finite(new_input.weight[271:])
                self.assert_nonzero_finite(new_output.weight[271:])
                if old_bias is not None:
                    self.assertTrue(torch.equal(new_output.bias[:271], old_bias))
                    self.assertEqual(float(new_output.bias[271:].detach().abs().sum()), 0.)
                for key, value in old_other.items():
                    self.assertTrue(torch.equal(decoder.state_dict()[key], value), key)
                output = decoder(input_ids=torch.tensor([[257, 271, 278]]), use_cache=False)
                self.assertEqual(output.logits.shape, (1, 3, codec().vocab_size))

    def test_fresh_id_rows_are_seeded_deterministically_and_distinct(self):
        for tied in (True, False):
            first = tiny_opt(tied=tied)
            second, different = deepcopy(first), deepcopy(first)
            extend_decoder_vocabulary(first, codec().vocab_size, seed=101)
            extend_decoder_vocabulary(second, codec().vocab_size, seed=101)
            extend_decoder_vocabulary(different, codec().vocab_size, seed=102)
            for getter in ('get_input_embeddings', 'get_output_embeddings'):
                a = getattr(first, getter)().weight.detach()[271:]
                b = getattr(second, getter)().weight.detach()[271:]
                c = getattr(different, getter)().weight.detach()[271:]
                self.assertTrue(torch.equal(a, b))
                self.assertFalse(torch.equal(a, c))
                self.assertEqual(torch.unique(a, dim=0).shape[0], 8)
            if not tied:
                self.assertFalse(torch.equal(first.get_input_embeddings().weight[271:],
                                             first.get_output_embeddings().weight[271:]))

    def test_vocabulary_extension_preserves_global_cpu_rng_states(self):
        decoder = tiny_opt()
        torch_before = torch.get_rng_state().clone()
        numpy_before = np.random.get_state()
        python_before = random.getstate()
        extend_decoder_vocabulary(decoder, codec().vocab_size, seed=332)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])
        self.assertEqual(random.getstate(), python_before)

    def test_vocabulary_extension_preserves_mocked_cuda_rng_state(self):
        decoder = tiny_opt()
        # torch.manual_seed also seeds CUDA, even when the weights are on CPU.
        # Model one initialized device without initializing or using a real GPU.
        before = torch.tensor([11, 27, 89], dtype=torch.uint8)
        state = [before.clone()]

        def seed_all(seed):
            state[0] = torch.tensor([int(seed) % 256], dtype=torch.uint8)

        def restore(value, device=0):
            state[0] = value.clone()

        with patch.multiple(torch.cuda, manual_seed_all=Mock(side_effect=seed_all),
                            get_rng_state=Mock(side_effect=lambda device=0: state[0].clone()),
                            set_rng_state=Mock(side_effect=restore),
                            device_count=Mock(return_value=1),
                            is_initialized=Mock(return_value=True),
                            is_available=Mock(return_value=True)):
            extend_decoder_vocabulary(decoder, codec().vocab_size, seed=332)
        self.assertTrue(torch.equal(state[0], before),
                        'CPU vocabulary extension changed the modeled CUDA RNG without restoring it')


if __name__ == '__main__':
    unittest.main()
