"""CPU contracts for selecting one decoder target without pruning encoder frames."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import unittest
from unittest.mock import patch

import torch

from rigweave.dynamic_rig.virtual_path_ablation import VirtualPathAblationRig
from test_virtual_path_ablation import assert_state_equal, make_base, make_surface, make_targets


def state_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(str((name, value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class VirtualSingleTargetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = VirtualPathAblationRig(make_base().double(), False).train()
        self.surface = {key: value.double() for key, value in make_surface().items()}
        self.targets = make_targets()
        lengths = torch.tensor([[6, 3, 5], [4, 6, 2]])
        self.targets['main_attention_mask'] = torch.arange(6)[None, None] < lengths[:, :, None]
        self.targets['main_input_ids'].masked_fill_(~self.targets['main_attention_mask'], 0)
        self.indices = torch.tensor([2, 0], dtype=torch.int64)

    def assert_outputs_equal(self, actual, expected):
        self.assertEqual(actual.keys(), expected.keys())
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0, msg=key)

    def test_selected_losses_and_accuracy_match_all_frame_entries(self):
        frames, virtual = self.model.encode(self.surface)
        all_frames = self.model.forward_encoded(frames, virtual, self.targets, auxiliary_weight=0.)
        selected = self.model.forward_encoded(frames, virtual, self.targets, auxiliary_weight=0.,
                                              target_frame_indices=self.indices)
        self.assertEqual(selected.keys(), all_frames.keys())
        rows = torch.arange(2)
        for key in ('main_loss_per_frame', 'main_accuracy_per_frame'):
            self.assertEqual(tuple(selected[key].shape), (2, 1))
            torch.testing.assert_close(selected[key][:, 0], all_frames[key][rows, self.indices],
                                       atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(selected['loss_per_asset'], selected['main_loss_per_frame'][:, 0])
        torch.testing.assert_close(selected['loss'], selected['loss_per_asset'].mean())
        torch.testing.assert_close(selected['main_loss'], selected['loss'])
        torch.testing.assert_close(selected['main_accuracy'], selected['main_accuracy_per_frame'].mean())
        self.assertEqual(selected['auxiliary_loss'].item(), 0.)
        self.assertEqual(selected['auxiliary_accuracy'].item(), 0.)

    def test_uniform_selected_loss_and_all_parameter_gradients_match_all_frame_mean(self):
        frames, virtual = self.model.encode(self.surface)
        all_frames = self.model.forward_encoded(frames, virtual, self.targets, auxiliary_weight=0.)
        selections = [self.model.forward_encoded(
            frames, virtual, self.targets, auxiliary_weight=0.,
            target_frame_indices=(torch.arange(2) + frame) % frames.shape[1])
            for frame in range(frames.shape[1])]
        selected_mean = torch.stack([output['loss'] for output in selections]).mean()
        torch.testing.assert_close(selected_mean, all_frames['loss'], atol=1e-12, rtol=1e-12)
        named = [(name, parameter) for name, parameter in self.model.named_parameters()
                 if parameter.requires_grad]
        parameters = [parameter for _, parameter in named]
        all_gradients = torch.autograd.grad(all_frames['loss'], parameters,
                                            allow_unused=True, retain_graph=True)
        selected_gradients = torch.autograd.grad(selected_mean, parameters, allow_unused=True)
        for (name, _), actual, expected in zip(named, selected_gradients, all_gradients):
            with self.subTest(parameter=name):
                if expected is None:
                    self.assertIsNone(actual)
                else:
                    self.assertIsNotNone(actual)
                    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-6, msg=name)

    def test_decoder_processes_only_requested_targets_in_one_batch(self):
        frames, virtual = self.model.encode(self.surface)
        expected_condition = self.model.conditions(frames, virtual)
        rows = torch.arange(2)
        targets = deepcopy(self.targets)
        unselected = torch.ones(2, 3, dtype=torch.bool)
        unselected[rows, self.indices] = False
        targets['main_input_ids'][unselected] = 1000
        with patch.object(self.model, 'main_logits', wraps=self.model.main_logits) as logits_spy, \
                patch.object(self.model.xyz_decoder, 'forward',
                             wraps=self.model.xyz_decoder.forward) as decoder_spy:
            self.model.forward_encoded(frames, virtual, targets, auxiliary_weight=0.,
                                       target_frame_indices=self.indices)
        logits_spy.assert_called_once()
        decoder_spy.assert_called_once()
        condition, ids, mask = logits_spy.call_args.args
        torch.testing.assert_close(condition, expected_condition[rows, self.indices], atol=0, rtol=0)
        torch.testing.assert_close(ids, self.targets['main_input_ids'][rows, self.indices, :-1])
        torch.testing.assert_close(mask, self.targets['main_attention_mask'][rows, self.indices, :-1])
        self.assertEqual(tuple(decoder_spy.call_args.kwargs['inputs_embeds'].shape), (2, 8, 8))

    def test_forward_keeps_all_encoder_frames_and_dispatches_selected_targets(self):
        with patch.object(self.model, 'encode', wraps=self.model.encode) as encode_spy, \
                patch.object(self.model, '_encode_real_only',
                             wraps=self.model._encode_real_only) as motion_spy, \
                patch.object(self.model, 'forward_encoded',
                             wraps=self.model.forward_encoded) as dispatch_spy, \
                patch.object(self.model.surface_tokenizer, 'forward',
                             wraps=self.model.surface_tokenizer.forward) as surface_spy:
            actual = self.model(self.surface, self.targets, auxiliary_weight=0.,
                                target_frame_indices=self.indices)
        encode_spy.assert_called_once()
        self.assertIs(encode_spy.call_args.args[0], self.surface)
        motion_spy.assert_called_once()
        self.assertEqual(tuple(motion_spy.call_args.args[0].shape), (2, 3, 3, 8))
        self.assertEqual(surface_spy.call_count, 3)
        dispatch_spy.assert_called_once()
        self.assertIs(dispatch_spy.call_args.kwargs['target_frame_indices'], self.indices)
        frames, virtual = self.model.encode(self.surface)
        expected = self.model.forward_encoded(frames, virtual, self.targets, auxiliary_weight=0.,
                                             target_frame_indices=self.indices)
        self.assert_outputs_equal(actual, expected)
        actual['loss'].backward()
        for parameter in (self.model.surface_tokenizer.projection.weight,
                          self.model.motion_encoder.blocks[0].anchor_temporal.self_attn.in_proj_weight,
                          self.model.frame_fusion[0].ffn[0].weight,
                          self.model.condition_norm.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.norm().item(), 0.)

    def test_state_fingerprint_parameter_names_buffers_and_configuration_stay_unchanged(self):
        original = deepcopy(self.model.state_dict())
        fingerprint = state_fingerprint(self.model)
        names = self.model.trainable_parameter_names
        parameter_names = tuple(dict(self.model.named_parameters()))
        buffer_names = tuple(dict(self.model.named_buffers()))
        configuration = deepcopy(self.model.configuration())
        self.model(self.surface, self.targets, auxiliary_weight=0.)
        self.model(self.surface, self.targets, auxiliary_weight=0.,
                   target_frame_indices=self.indices)['loss'].backward()
        assert_state_equal(self, original, self.model.state_dict())
        self.assertEqual(state_fingerprint(self.model), fingerprint)
        self.assertEqual(self.model.trainable_parameter_names, names)
        self.assertEqual(tuple(dict(self.model.named_parameters())), parameter_names)
        self.assertEqual(tuple(dict(self.model.named_buffers())), buffer_names)
        self.assertEqual(self.model.configuration(), configuration)

    def test_explicit_none_is_exact_default_forward_and_encoded_for_both_arms(self):
        for virtual_enabled in (False, True):
            with self.subTest(virtual_enabled=virtual_enabled):
                model = VirtualPathAblationRig(make_base().double(), virtual_enabled).eval()
                expected = model(self.surface, self.targets, auxiliary_weight=0.)
                actual = model(self.surface, self.targets, auxiliary_weight=0., target_frame_indices=None)
                self.assert_outputs_equal(actual, expected)
                frames, virtual = model.encode(self.surface)
                encoded = model.forward_encoded(frames, virtual, self.targets, auxiliary_weight=0.,
                                                target_frame_indices=None)
                self.assert_outputs_equal(encoded, expected)
                self.assertEqual(tuple(actual['main_loss_per_frame'].shape), (2, 3))

    def test_invalid_indices_are_rejected_before_decoding(self):
        frames, virtual = self.model.encode(self.surface)
        invalid = ([0, 1], torch.tensor([0., 1.]), torch.tensor([0, 1], dtype=torch.int32),
                   torch.tensor([False, True]), torch.tensor(0), torch.tensor([[[0]], [[1]]]),
                   torch.tensor([0]), torch.tensor([0, 1, 2]), torch.empty(0, dtype=torch.int64),
                   torch.tensor([-1, 0]), torch.tensor([0, 3]))
        for indices in invalid:
            for encoded in (False, True):
                with self.subTest(indices=indices, encoded=encoded), \
                        patch.object(self.model, 'main_logits') as decoder_spy:
                    with self.assertRaisesRegex(ValueError, 'target_frame_indices'):
                        if encoded:
                            self.model.forward_encoded(frames, virtual, self.targets, auxiliary_weight=0.,
                                                       target_frame_indices=indices)
                        else:
                            self.model(self.surface, self.targets, auxiliary_weight=0.,
                                       target_frame_indices=indices)
                    decoder_spy.assert_not_called()

    def test_vplus_rejects_selection_before_base_dispatch(self):
        model = VirtualPathAblationRig(make_base().double(), True).eval()
        with patch.object(model.base, 'forward') as forward_spy, \
                patch.object(model.base, 'forward_encoded') as encoded_spy:
            with self.assertRaisesRegex(ValueError, 'virtual_enabled=False'):
                model(self.surface, self.targets, auxiliary_weight=0., target_frame_indices=self.indices)
            with self.assertRaisesRegex(ValueError, 'virtual_enabled=False'):
                model.forward_encoded(None, None, self.targets, auxiliary_weight=0.,
                                      target_frame_indices=self.indices)
        forward_spy.assert_not_called()
        encoded_spy.assert_not_called()

    def test_selected_mode_still_rejects_auxiliary_loss(self):
        with self.assertRaisesRegex(ValueError, 'exactly 0'):
            self.model(self.surface, self.targets, auxiliary_weight=.1, target_frame_indices=self.indices)
        with self.assertRaisesRegex(ValueError, 'exactly 0'):
            self.model.forward_encoded(None, None, self.targets, auxiliary_weight=.1,
                                       target_frame_indices=self.indices)


if __name__ == '__main__':
    unittest.main()
