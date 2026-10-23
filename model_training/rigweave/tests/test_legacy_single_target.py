"""CPU parity checks for the isolated original single-target route."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('RIGWEAVE_DISABLE_OPEN3D', '1')
os.environ.setdefault('RIGWEAVE_DISABLE_LIGHTNING_IMPORT', '1')
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'model_training/rigweave/src'))

import torch
from torch import nn
from transformers import OPTConfig, OPTForCausalLM

from rigweave.dynamic_rig.legacy_single_target import LegacySingleTargetRig
from rigweave.dynamic_rig.motion_encoder import AnchorWiseAlternatingMotionEncoder
from rigweave.dynamic_rig.motion_evidence import grouped_motion_evidence
from rigweave.dynamic_rig.unirig_wrapper import DynamicRigUniRigAR
from rigweave.dynamic_rig.virtual_query_xyz import VirtualQueryXYZConfig, VirtualQueryXYZRig


class Surface(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(6, 16)

    def forward(self, points, normals, queries, query_normals):
        return self.projection(torch.cat((queries + points.mean(1, keepdim=True),
                                          query_normals + normals.mean(1, keepdim=True)), dim=-1))


def fingerprint(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode())
        digest.update(value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class LegacySingleTargetTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(731)
        torch.set_num_threads(1)
        self.config = VirtualQueryXYZConfig(
            dim=16, heads=4, query_tokens=4, register_tokens=2, motion_depth=2,
            max_frames=4, auxiliary_depth=1, auxiliary_max_tokens=32,
            motion_evidence_fusion='off', motion_evidence_heads=0, gradient_checkpointing=False)
        self.surface = {
            'dense_points': torch.randn(2, 3, 24, 3),
            'dense_normals': torch.randn(2, 3, 24, 3),
            'query_indices': torch.tensor([[0, 6, 12, 18], [1, 7, 13, 19]]),
        }
        indices = self.surface['query_indices'][:, None, :, None].expand(-1, 3, -1, 3)
        self.surface['query_points'] = self.surface['dense_points'].gather(2, indices).clone()
        self.surface['query_normals'] = self.surface['dense_normals'].gather(2, indices).clone()
        self.ids = torch.tensor([[[25, 26, 5, 6, 7, 29, 8, 9, 30],
                                  [25, 26, 5, 11, 7, 30, 0, 0, 0],
                                  [25, 26, 9, 8, 7, 6, 30, 0, 0]],
                                 [[25, 26, 3, 4, 5, 30, 0, 0, 0],
                                  [25, 26, 8, 9, 10, 29, 5, 6, 30],
                                  [25, 26, 12, 13, 30, 0, 0, 0, 0]]])
        self.targets = {'main_input_ids': self.ids, 'main_attention_mask': self.ids != 0}
        self.indices = torch.tensor([1, 2])

    def base(self, *, bias=False, checkpointing=False):
        config = replace(self.config, motion_evidence_fusion='bias' if bias else 'off',
                         motion_evidence_heads=4 if bias else 0, gradient_checkpointing=checkpointing)
        decoder = OPTForCausalLM(OPTConfig(
            vocab_size=32, hidden_size=16, word_embed_proj_dim=16, ffn_dim=32,
            num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=128,
            dropout=0., attention_dropout=0., activation_dropout=0., _attn_implementation='eager'))
        base = VirtualQueryXYZRig(Surface(), decoder, config, auxiliary_vocab_size=12)
        if bias:
            with torch.no_grad():
                for block in base.motion_encoder.blocks:
                    block.motion_injection.bias_weights.copy_(torch.randn(4, 3))
        return base.eval()

    def reference_conditions(self, base, surface, indices):
        tokens = torch.stack([base.surface_tokenizer(*(surface[key][:, frame] for key in
            ('dense_points', 'dense_normals', 'query_points', 'query_normals')))
            for frame in range(3)], dim=1)
        conditions = []
        for asset, target in enumerate(indices.tolist()):
            order = [target] + [frame for frame in range(3) if frame != target]
            evidence = None
            if base.config.motion_evidence_fusion != 'off':
                evidence = grouped_motion_evidence(surface['dense_points'][asset:asset+1, order],
                                                   surface['query_indices'][asset:asset+1])
            conditions.append(AnchorWiseAlternatingMotionEncoder.forward(
                base.motion_encoder, tokens[asset:asset+1, order],
                query_points=surface['query_points'][asset:asset+1, order], motion_evidence=evidence))
        return torch.cat(conditions)

    def test_extraction_preserves_exact_filtered_state_rng_modes_and_fingerprint(self):
        base = self.base(bias=True).double()
        base.motion_encoder.blocks[0].pose_inner.linear1.weight.requires_grad_(False)
        before = {key: value.clone() for key, value in base.state_dict().items()}
        rng = torch.get_rng_state().clone()
        model = LegacySingleTargetRig.from_virtual_base(base)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertIs(model.surface_tokenizer, base.surface_tokenizer)
        self.assertIs(model.xyz_decoder, base.xyz_decoder)
        self.assertIs(type(model.motion_encoder), AnchorWiseAlternatingMotionEncoder)
        self.assertFalse(model.training)
        self.assertEqual(set(model._modules), {'surface_tokenizer', 'motion_encoder', 'xyz_decoder'})
        excluded = ('frame_fusion.', 'condition_norm.', 'auxiliary.')
        expected = {key: value for key, value in before.items()
                    if key != 'motion_encoder.virtual_slots' and not key.startswith(excluded)}
        self.assertEqual(set(model.state_dict()), set(expected))
        self.assertEqual(fingerprint(model.state_dict()), fingerprint(expected))
        for key, value in base.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.requires_grad, dict(base.named_parameters())[name].requires_grad)
        for forbidden in ('virtual_slots', 'frame_fusion', 'condition_norm', 'auxiliary'):
            self.assertFalse(any(forbidden in name for name, _ in model.named_parameters()))
        self.assertEqual(model.configuration()['family'], 'legacy_real_query_single_target_v1')

    def test_extraction_does_not_silently_discard_other_motion_state(self):
        base = self.base()
        base.motion_encoder.register_buffer('unexpected', torch.ones(1))
        with self.assertRaisesRegex(RuntimeError, 'unexpected'):
            LegacySingleTargetRig.from_virtual_base(base)

    def test_conditions_match_actual_original_encoder_with_and_without_bias(self):
        for bias in (False, True):
            with self.subTest(bias=bias):
                base = self.base(bias=bias)
                model = LegacySingleTargetRig.from_virtual_base(base)
                tokens = model.encode_surface(self.surface)
                expected = self.reference_conditions(base, self.surface, self.indices)
                actual = model.condition_from_tokens(self.surface, tokens, self.indices)
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_forward_calls_surface_T_times_motion_once_and_AR_once(self):
        model = LegacySingleTargetRig.from_virtual_base(self.base())
        calls = []
        hooks = [module.register_forward_pre_hook(lambda module, args, name=name: calls.append(name))
                 for name, module in [('surface', model.surface_tokenizer), ('motion', model.motion_encoder),
                                      ('ar', model.xyz_decoder)]]
        try:
            output = model(self.surface, self.targets, target_indices=self.indices)
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(calls, ['surface'] * 3 + ['motion', 'ar'])
        self.assertEqual(set(output), {'loss', 'loss_per_asset', 'main_loss', 'main_accuracy',
                                      'main_loss_per_asset', 'main_accuracy_per_asset'})
        self.assertEqual(output['loss_per_asset'].shape, (2,))
        torch.testing.assert_close(output['loss'], output['loss_per_asset'].mean())

    def test_selected_logits_loss_and_gradients_match_original_AR(self):
        for checkpointing in (False, True):
            with self.subTest(checkpointing=checkpointing):
                base = self.base(checkpointing=checkpointing)
                reference = deepcopy(base)
                model = LegacySingleTargetRig.from_virtual_base(base).train()
                reference.train()
                upstream = nn.Module()
                upstream.hidden_size = 16
                upstream.transformer = reference.xyz_decoder
                old = DynamicRigUniRigAR(upstream, nn.Identity(),
                    SimpleNamespace(vocab_size=32, num_discrete=24, eos=30, token_id_branch=29),
                    query_tokens=4, surface_no_grad=False, condition_fusion_heads=4)
                actual_logits = []
                hook = model.xyz_decoder.register_forward_hook(
                    lambda module, args, output: actual_logits.append(output.logits[:, 4:]))
                output = model(self.surface, self.targets, target_indices=self.indices)
                hook.remove()
                conditions = self.reference_conditions(reference, self.surface, self.indices)
                expected_logits, losses = [], []
                hook = reference.xyz_decoder.register_forward_hook(
                    lambda module, args, output: expected_logits.append(output.logits[:, 4:-1]))
                for asset, frame in enumerate(self.indices.tolist()):
                    losses.append(old._ar_losses(conditions[asset:asset+1], {
                        'input_ids': self.ids[asset:asset+1, frame],
                        'attention_mask': self.targets['main_attention_mask'][asset:asset+1, frame]},
                        include_loop_recovery=False, include_generated_prefix_recovery=False)['ce_loss'])
                hook.remove()
                torch.testing.assert_close(actual_logits[0], torch.cat(expected_logits), atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(output['loss_per_asset'], torch.stack(losses), atol=1e-6, rtol=1e-6)
                labels = self.ids[torch.arange(2), self.indices, 1:]
                valid = self.targets['main_attention_mask'][torch.arange(2), self.indices, 1:]
                expected_accuracy = ((torch.cat(expected_logits).argmax(-1) == labels) & valid).sum(-1) / valid.sum(-1)
                torch.testing.assert_close(output['main_accuracy_per_asset'], expected_accuracy)
                output['loss'].backward()
                torch.stack(losses).mean().backward()
                reference_parameters = dict(reference.named_parameters())
                for name, parameter in model.named_parameters():
                    expected_grad = reference_parameters[name].grad
                    if expected_grad is None:
                        self.assertIsNone(parameter.grad, name)
                    else:
                        torch.testing.assert_close(parameter.grad, expected_grad, atol=2e-6, rtol=2e-5, msg=name)
                for parameter in (model.surface_tokenizer.projection.weight,
                                  model.motion_encoder.blocks[0].pose_inner.self_attn.in_proj_weight,
                                  model.xyz_decoder.get_input_embeddings().weight,
                                  model.xyz_decoder.get_output_embeddings().weight):
                    self.assertGreater(float(parameter.grad.norm()), 0)

    def test_non_target_labels_and_masks_cannot_change_loss(self):
        model = LegacySingleTargetRig.from_virtual_base(self.base())
        changed = {key: value.clone() for key, value in self.targets.items()}
        for asset, target in enumerate(self.indices.tolist()):
            for frame in range(3):
                if frame != target:
                    changed['main_input_ids'][asset, frame].fill_(999)
                    changed['main_attention_mask'][asset, frame].zero_()
        first = model(self.surface, self.targets, target_indices=self.indices)
        second = model(self.surface, changed, target_indices=self.indices)
        for name in first:
            torch.testing.assert_close(first[name], second[name], atol=0, rtol=0)

    def test_target_loss_reaches_non_target_geometry(self):
        model = LegacySingleTargetRig.from_virtual_base(self.base()).train()
        surface = {key: value.clone().requires_grad_(value.is_floating_point())
                   for key, value in self.surface.items()}
        model(surface, self.targets, target_indices=self.indices)['loss'].backward()
        for asset, target in enumerate(self.indices.tolist()):
            for frame in range(3):
                if frame != target:
                    for key in ('dense_points', 'query_points'):
                        gradient = surface[key].grad[asset, frame]
                        self.assertTrue(bool(torch.isfinite(gradient).all()))
                        self.assertGreater(float(gradient.norm()), 0)

    def test_bias_recomputed_after_per_asset_reorder_and_batch_permutation(self):
        model = LegacySingleTargetRig.from_virtual_base(self.base(bias=True))
        original = {key: value.clone() for key, value in self.surface.items()}
        tokens = model.encode_surface(self.surface)
        with patch('rigweave.dynamic_rig.legacy_single_target.grouped_motion_evidence',
                   wraps=grouped_motion_evidence) as evidence:
            actual = model.condition_from_tokens(self.surface, tokens, self.indices)
        evidence.assert_called_once()
        points, indices = evidence.call_args.args
        expected_points = torch.stack([self.surface['dense_points'][0, [1, 0, 2]],
                                       self.surface['dense_points'][1, [2, 0, 1]]])
        torch.testing.assert_close(points, expected_points, atol=0, rtol=0)
        self.assertIs(indices, self.surface['query_indices'])
        permutation = torch.tensor([1, 0])
        permuted = {key: value[permutation] for key, value in self.surface.items()}
        other = model.condition_from_tokens(permuted, tokens[permutation], self.indices[permutation])
        torch.testing.assert_close(other, actual[permutation], atol=1e-6, rtol=1e-6)
        first = model(self.surface, self.targets, target_indices=self.indices)
        second = model(permuted, {key: value[permutation] for key, value in self.targets.items()},
                       target_indices=self.indices[permutation])
        torch.testing.assert_close(second['loss_per_asset'], first['loss_per_asset'][permutation], atol=1e-6, rtol=1e-6)
        for key in original:
            torch.testing.assert_close(self.surface[key], original[key], atol=0, rtol=0)

    def test_target_indices_are_required_strict_and_range_checked(self):
        model = LegacySingleTargetRig.from_virtual_base(self.base())
        with self.assertRaises(TypeError):
            model(self.surface, self.targets)
        for indices in ([0, 1], torch.tensor([0., 1.]), torch.tensor([0], dtype=torch.long),
                        torch.tensor([[0], [1]]), torch.tensor([-1, 1]), torch.tensor([0, 3])):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                model(self.surface, self.targets, target_indices=indices)
        tokens = model.encode_surface(self.surface)
        with self.assertRaises(ValueError):
            model.condition_from_tokens(self.surface, tokens[:, :1], self.indices)

    def test_generation_uses_existing_complete_AR_path(self):
        base = self.base()
        model = LegacySingleTargetRig.from_virtual_base(base)
        condition = model.condition_from_tokens(self.surface, model.encode_surface(self.surface), self.indices)[:1]
        start = torch.tensor([25, 26])
        def force_eos(ids, scores):
            forced = torch.full_like(scores, -1e9)
            forced[:, 30] = 0
            return forced
        kwargs = {'eos_token': 30, 'max_new_tokens': 4, 'logits_processor': force_eos}
        self.assertEqual(model.generate_condition(condition, start, **kwargs),
                         base.generate_condition(condition, start, **kwargs))
        self.assertEqual(model.generate_condition(condition, start, **kwargs)['tokens'], [25, 26, 30])


if __name__ == '__main__':
    unittest.main()
