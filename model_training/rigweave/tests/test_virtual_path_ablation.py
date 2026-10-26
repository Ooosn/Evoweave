"""CPU contract tests for the matched virtual-path ablation wrapper."""
from __future__ import annotations

from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from rigweave.dynamic_rig.virtual_path_ablation import VirtualPathAblationRig
from rigweave.dynamic_rig.virtual_query_xyz import VirtualQueryXYZConfig, VirtualQueryXYZRig


class TinySurface(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.projection = nn.Linear(6, dim)

    def forward(self, dense, normals, query, query_normals):
        return self.projection(torch.cat((query, query_normals), dim=-1))


class TinyDecoder(nn.Module):
    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.head = nn.Linear(dim, vocab_size)

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.head

    def forward(self, inputs_embeds=None, input_ids=None, attention_mask=None,
                use_cache=False, return_dict=True, past_key_values=None):
        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)
        # Give the fake decoder causal prefix dependence so condition gradients
        # exercise the same upstream route as the real autoregressive decoder.
        logits = self.head(torch.cumsum(inputs_embeds, dim=1))
        return SimpleNamespace(logits=logits, past_key_values=() if use_cache else None)


def make_base(seed: int = 41) -> VirtualQueryXYZRig:
    torch.manual_seed(seed)
    config = VirtualQueryXYZConfig(
        dim=8, heads=2, query_tokens=3, register_tokens=1, motion_depth=2,
        fusion_depth=2, auxiliary_depth=1, auxiliary_max_tokens=16, max_frames=4,
        motion_evidence_fusion='off', motion_evidence_heads=0,
        gradient_checkpointing=False, temporal_chunk_size=2,
        fusion_mode='real_queries_virtual')
    decoder = TinyDecoder(config.dim, 17)
    return VirtualQueryXYZRig(TinySurface(config.dim), decoder, config, auxiliary_vocab_size=7)


def make_surface(batch=2, frames=3, queries=3):
    torch.manual_seed(73)
    return {key: torch.randn(batch, frames, queries if 'query' in key else 5, 3)
            for key in ('dense_points', 'dense_normals', 'query_points', 'query_normals')}


def make_targets(batch=2, frames=3, length=6):
    torch.manual_seed(97)
    return {
        'main_input_ids': torch.randint(1, 17, (batch, frames, length)),
        'main_attention_mask': torch.ones(batch, frames, length, dtype=torch.bool),
    }


def assert_state_equal(test: unittest.TestCase, left, right):
    test.assertEqual(left.keys(), right.keys())
    for key in left:
        torch.testing.assert_close(left[key], right[key], atol=0, rtol=0, msg=key)


class VirtualPathAblationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.surface = make_surface()
        self.targets = make_targets()

    def test_vplus_is_exact_base_forward_and_condition_delegate(self):
        reference = make_base()
        ablation_base = deepcopy(reference)
        reference.eval()
        plus = VirtualPathAblationRig(ablation_base, True).eval()

        expected_frames, expected_virtual = reference.encode(self.surface)
        actual_frames, actual_virtual = plus.encode(self.surface)
        torch.testing.assert_close(actual_frames, expected_frames, atol=0, rtol=0)
        torch.testing.assert_close(actual_virtual, expected_virtual, atol=0, rtol=0)
        expected_condition = reference.conditions(expected_frames, expected_virtual)
        actual_condition = plus.conditions(actual_frames, actual_virtual)
        torch.testing.assert_close(actual_condition, expected_condition, atol=0, rtol=0)

        expected = reference(self.surface, self.targets, auxiliary_weight=0.)
        actual = plus(self.surface, self.targets, auxiliary_weight=0.)
        self.assertEqual(expected.keys(), actual.keys())
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0, msg=key)
        self.assertFalse(plus.configuration()['evaluate_auxiliary'])

    def test_common_initialized_state_is_unchanged_and_checkpoint_roundtrips(self):
        reference = make_base()
        original = {key: value.detach().clone() for key, value in reference.state_dict().items()}
        plus = VirtualPathAblationRig(deepcopy(reference), True)
        minus = VirtualPathAblationRig(deepcopy(reference), False)
        assert_state_equal(self, original, reference.state_dict())
        assert_state_equal(self, original, plus.base.state_dict())
        assert_state_equal(self, original, minus.base.state_dict())

        buffer = BytesIO()
        torch.save(minus.state_dict(), buffer)
        restored = VirtualPathAblationRig(deepcopy(reference), False)
        restored.load_state_dict(torch.load(BytesIO(buffer.getvalue()), weights_only=True))
        assert_state_equal(self, minus.state_dict(), restored.state_dict())

    def test_vminus_ignores_virtual_slots_and_frame_cross_modules(self):
        model = VirtualPathAblationRig(make_base(), False).eval()
        frames, virtual = model.encode(self.surface)
        self.assertIsNone(virtual)
        first = model.conditions(frames, virtual)
        with torch.no_grad():
            model.motion_encoder.virtual_slots.fill_(123.)
            for layer in model.frame_fusion:
                layer.cross.in_proj_weight.fill_(321.)
                layer.cross.in_proj_bias.fill_(321.)
        changed = model.conditions(frames, None)
        torch.testing.assert_close(changed, first, atol=0, rtol=0)
        self.assertIn('base.motion_encoder.virtual_slots', model.disabled_parameter_names)
        self.assertTrue(all(not parameter.requires_grad for layer in model.frame_fusion
                            for module in (layer.query_norm, layer.memory_norm, layer.cross)
                            for parameter in module.parameters()))

    def test_vminus_real_frames_remain_temporally_coupled(self):
        model = VirtualPathAblationRig(make_base(), False).eval()
        first, _ = model.encode(self.surface)
        changed_surface = {key: value.clone() for key, value in self.surface.items()}
        changed_surface['query_points'][:, 1] += 2.
        second, _ = model.encode(changed_surface)
        self.assertFalse(torch.equal(first[:, 0], second[:, 0]))

    def test_vminus_all_frame_loss_has_no_auxiliary_route(self):
        model = VirtualPathAblationRig(make_base(), False).eval()
        frames, virtual = model.encode(self.surface)
        output = model.forward_encoded(frames, virtual, self.targets, auxiliary_weight=0.)
        self.assertEqual(tuple(output['main_loss_per_frame'].shape), (2, 3))
        torch.testing.assert_close(output['loss'], output['main_loss'], atol=0, rtol=0)
        self.assertEqual(float(output['auxiliary_loss']), 0.)
        self.assertEqual(float(output['auxiliary_accuracy']), 0.)
        with self.assertRaisesRegex(ValueError, 'exactly 0'):
            model.forward_encoded(frames, virtual, self.targets, auxiliary_weight=.1)

    def test_vminus_active_gradients_and_unused_gradients(self):
        model = VirtualPathAblationRig(make_base(), False).train()
        output = model(self.surface, self.targets, auxiliary_weight=0.)
        output['loss'].backward()

        active = [
            model.surface_tokenizer.projection.weight,
            model.motion_encoder.blocks[0].pose_inner.self_attn.in_proj_weight,
            model.motion_encoder.blocks[0].anchor_temporal.self_attn.in_proj_weight,
            model.frame_fusion[0].ffn[0].weight,
            model.condition_norm.weight,
            model.xyz_decoder.embedding.weight,
            model.xyz_decoder.head.weight,
        ]
        active_names = ('surface', 'pose_inner', 'anchor_temporal', 'frame_ffn',
                        'condition_norm', 'decoder_embedding', 'decoder_head')
        for name, parameter in zip(active_names, active):
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.norm()), 0.)
        self.assertIsNone(model.motion_encoder.virtual_slots.grad)
        self.assertIsNone(model.frame_fusion[0].cross.in_proj_weight.grad)
        self.assertIsNone(model.frame_fusion[0].query_norm.weight.grad)
        self.assertIsNone(model.frame_fusion[0].memory_norm.weight.grad)
        self.assertTrue(all(parameter.grad is None for parameter in model.auxiliary.parameters()))
        self.assertTrue(torch.equal(model.motion_encoder.role_token.grad[:, 0],
                                    torch.zeros_like(model.motion_encoder.role_token.grad[:, 0])))
        self.assertGreater(float(model.motion_encoder.role_token.grad[:, 1].norm()), 0.)

    def test_generate_condition_delegates_with_synchronized_eval_mode(self):
        model = VirtualPathAblationRig(make_base(), True)
        model.train()
        self.assertTrue(model.base.training)
        model.eval()
        self.assertFalse(model.base.training)
        frames, virtual = model.encode(self.surface[:1] if isinstance(self.surface, list) else {
            key: value[:1] for key, value in self.surface.items()})
        condition = model.conditions(frames, virtual)[:, 0]
        result = model.generate_condition(condition, torch.tensor([1]), eos_token=2,
                                          max_new_tokens=1)
        self.assertIn(result['status'], {'eos', 'max_tokens'})


class VirtualTwoTargetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = VirtualPathAblationRig(make_base().double(), False).train()
        self.surface = {key: value.double() for key, value in make_surface().items()}
        self.targets = make_targets()
        lengths = torch.tensor([[6, 3, 5], [4, 6, 2]])
        self.targets['main_attention_mask'] = torch.arange(6)[None, None] < lengths[:, :, None]
        self.targets['main_input_ids'].masked_fill_(~self.targets['main_attention_mask'], 0)
        self.indices = torch.tensor([[0, 2], [2, 1]], dtype=torch.int64)

    def assert_gradients_equal(self, actual, expected):
        named = [(name, p) for name, p in self.model.named_parameters() if p.requires_grad]
        parameters = [p for _, p in named]
        actual_grad = torch.autograd.grad(actual, parameters, allow_unused=True, retain_graph=True)
        expected_grad = torch.autograd.grad(expected, parameters, allow_unused=True)
        for (name, _), actual, expected in zip(named, actual_grad, expected_grad):
            with self.subTest(parameter=name):
                if expected is None:
                    self.assertIsNone(actual)
                else:
                    self.assertIsNotNone(actual)
                    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-6, msg=name)

    def test_two_target_loss_and_gradients_equal_selected_all_frame_entries(self):
        frames, virtual = self.model.encode(self.surface)
        all_frames = self.model.forward_encoded(frames, virtual, self.targets, auxiliary_weight=0.)
        selected = self.model.forward_encoded(frames, virtual, self.targets, auxiliary_weight=0.,
                                              target_frame_indices=self.indices)
        rows = torch.arange(2)[:, None]
        self.assertEqual(selected.keys(), all_frames.keys())
        for key in ('main_loss_per_frame', 'main_accuracy_per_frame'):
            self.assertEqual(selected[key].shape, (2, 2))
            torch.testing.assert_close(selected[key], all_frames[key][rows, self.indices], atol=1e-12, rtol=1e-12)
        expected_per_asset = all_frames['main_loss_per_frame'][rows, self.indices].mean(dim=1)
        torch.testing.assert_close(selected['loss_per_asset'], expected_per_asset, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(selected['loss'], expected_per_asset.mean(), atol=1e-12, rtol=1e-12)
        self.assert_gradients_equal(selected['loss'], expected_per_asset.mean())

    def test_uniform_two_target_expectation_matches_all_frame_loss_and_gradients(self):
        frames, virtual = self.model.encode(self.surface)
        all_frames = self.model.forward_encoded(frames, virtual, self.targets, auxiliary_weight=0.)
        losses = [self.model.forward_encoded(frames, virtual, self.targets, auxiliary_weight=0.,
            target_frame_indices=torch.tensor([pair, pair]))['loss']
            for pair in ((0, 1), (0, 2), (1, 2))]
        mean = torch.stack(losses).mean()
        torch.testing.assert_close(mean, all_frames['loss'], atol=1e-12, rtol=1e-12)
        self.assert_gradients_equal(mean, all_frames['loss'])

    def test_k2_equals_all_when_t2_and_matrix_k1_equals_legacy_vector(self):
        surface = {key: value[:, :2] for key, value in self.surface.items()}
        targets = {key: value[:, :2] for key, value in self.targets.items()}
        all_frames = self.model(surface, targets, auxiliary_weight=0.)
        two = self.model(surface, targets, auxiliary_weight=0., target_frame_indices=torch.tensor([[0, 1], [0, 1]]))
        for key in all_frames:
            torch.testing.assert_close(two[key], all_frames[key], atol=0, rtol=0, msg=key)
        indices = torch.tensor([2, 0])
        vector = self.model(self.surface, self.targets, auxiliary_weight=0., target_frame_indices=indices)
        matrix = self.model(self.surface, self.targets, auxiliary_weight=0., target_frame_indices=indices[:, None])
        for key in vector:
            torch.testing.assert_close(matrix[key], vector[key], atol=0, rtol=0, msg=key)

    def test_only_selected_decoders_run_after_full_encoding(self):
        targets = deepcopy(self.targets)
        rows = torch.arange(2)[:, None]
        unselected = torch.ones(2, 3, dtype=torch.bool)
        unselected[rows, self.indices] = False
        targets['main_input_ids'][unselected] = 1000
        frames, virtual = self.model.encode(self.surface)
        expected = self.model.conditions(frames, virtual)
        with patch.object(self.model, '_encode_real_only', wraps=self.model._encode_real_only) as encoder, \
                patch.object(self.model.surface_tokenizer, 'forward', wraps=self.model.surface_tokenizer.forward) as surface, \
                patch.object(self.model, 'main_logits', wraps=self.model.main_logits) as decoder:
            self.model(self.surface, targets, auxiliary_weight=0., target_frame_indices=self.indices)
        encoder.assert_called_once()
        self.assertEqual(encoder.call_args.args[0].shape, (2, 3, 3, 8))
        self.assertEqual(surface.call_count, 3)
        self.assertEqual(decoder.call_count, 2)
        for slot, call in enumerate(decoder.call_args_list):
            condition, ids, mask = call.args
            choice = self.indices[:, slot]
            torch.testing.assert_close(condition, expected[torch.arange(2), choice], atol=0, rtol=0)
            torch.testing.assert_close(ids, self.targets['main_input_ids'][torch.arange(2), choice, :-1])
            torch.testing.assert_close(mask, self.targets['main_attention_mask'][torch.arange(2), choice, :-1])

    def test_selection_does_not_change_state_configuration_or_trainable_names(self):
        original = deepcopy(self.model.state_dict())
        configuration = deepcopy(self.model.configuration())
        trainable = self.model.trainable_parameter_names
        buffers = tuple(dict(self.model.named_buffers()))
        self.model(self.surface, self.targets, auxiliary_weight=0., target_frame_indices=self.indices)['loss'].backward()
        assert_state_equal(self, original, self.model.state_dict())
        self.assertEqual(self.model.configuration(), configuration)
        self.assertEqual(self.model.trainable_parameter_names, trainable)
        self.assertEqual(tuple(dict(self.model.named_buffers())), buffers)
        self.assertIsNone(self.model.motion_encoder.virtual_slots.grad)
        self.assertIsNone(self.model.frame_fusion[0].cross.in_proj_weight.grad)
        self.assertTrue(all(p.grad is None for p in self.model.auxiliary.parameters()))

    def test_invalid_shape_dtype_bounds_duplicates_and_meta_device_reject_before_decode(self):
        frames, virtual = self.model.encode(self.surface)
        invalid = ([[0, 1], [1, 2]], torch.tensor([[0., 1.], [1., 2.]]),
                   self.indices.to(torch.int32), self.indices.bool(), torch.tensor(0),
                   torch.tensor([[[0]], [[1]]]), torch.tensor([[0, 1]]),
                   torch.empty(2, 0, dtype=torch.int64), torch.zeros(2, 4, dtype=torch.int64),
                   torch.tensor([[0, 0], [1, 2]]), torch.tensor([[0, -1], [1, 2]]),
                   torch.tensor([[0, 3], [1, 2]]), torch.empty(2, 2, dtype=torch.int64, device='meta'))
        for indices in invalid:
            for encoded in (True, False):
                with self.subTest(indices=indices, encoded=encoded), patch.object(self.model, 'main_logits') as decoder:
                    with self.assertRaisesRegex(ValueError, 'target_frame_indices'):
                        if encoded:
                            self.model.forward_encoded(frames, virtual, self.targets, auxiliary_weight=0.,
                                                       target_frame_indices=indices)
                        else:
                            self.model(self.surface, self.targets, auxiliary_weight=0., target_frame_indices=indices)
                    decoder.assert_not_called()

    def test_two_target_mode_rejects_virtual_path_and_auxiliary_loss(self):
        plus = VirtualPathAblationRig(make_base().double(), True)
        with patch.object(plus.base, 'forward') as forward, patch.object(plus.base, 'forward_encoded') as encoded:
            with self.assertRaisesRegex(ValueError, 'virtual_enabled=False'):
                plus(self.surface, self.targets, auxiliary_weight=0., target_frame_indices=self.indices)
            with self.assertRaisesRegex(ValueError, 'virtual_enabled=False'):
                plus.forward_encoded(None, None, self.targets, auxiliary_weight=0., target_frame_indices=self.indices)
            forward.assert_not_called()
            encoded.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'exactly 0'):
            self.model(self.surface, self.targets, auxiliary_weight=.1, target_frame_indices=self.indices)


if __name__ == '__main__':
    unittest.main()
