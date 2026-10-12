"""Compare the actual legacy XYZ training function with the per-frame route."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from transformers import OPTConfig, OPTForCausalLM

from rigweave.dynamic_rig.unirig_wrapper import DynamicRigUniRigAR
from rigweave.dynamic_rig.virtual_query_xyz import VirtualQueryXYZConfig, VirtualQueryXYZRig, sequence_ce
from rigweave.dynamic_rig.motion_encoder import AnchorWiseAlternatingMotionEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from probe_virtual_query_reference import ReferenceConditionRig, declared_reference_run


class Surface(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(6, 16)

    def forward(self, points, normals, queries, query_normals):
        return self.projection(torch.cat((queries, query_normals), -1))


class LegacyParityTests(unittest.TestCase):
    def test_reference_authority_accepts_only_exact_single_or_declared_paired_runs(self):
        first = {'config': '/reference/a.json', 'config_sha256': 'hash-a', 'output': '/reference/out-a'}
        second = {'config': '/reference/b.json', 'config_sha256': 'hash-b', 'output': '/reference/out-b'}
        args = SimpleNamespace(config=Path(first['config']), output=Path(first['output']))
        self.assertIs(declared_reference_run(first, args, 'hash-a'), first)
        self.assertIs(declared_reference_run({'runs': [second, first]}, args, 'hash-a'), first)
        for operation, digest, output in (({'runs': [first, first]}, 'hash-a', first['output']),
                                           ({'runs': [second]}, 'hash-a', first['output']),
                                           ({'runs': [first]}, 'wrong', first['output']),
                                           ({'runs': [first, second]}, 'hash-a', second['output'])):
            with self.subTest(operation=operation, digest=digest, output=output), self.assertRaises(ValueError):
                declared_reference_run(operation, SimpleNamespace(config=args.config, output=Path(output)), digest)

    def setUp(self):
        torch.manual_seed(723)
        torch.set_num_threads(1)
        decoder = OPTForCausalLM(OPTConfig(vocab_size=32, hidden_size=16, word_embed_proj_dim=16,
            ffn_dim=32, num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=128,
            dropout=0., attention_dropout=0., activation_dropout=0., _attn_implementation='eager'))
        config = VirtualQueryXYZConfig(dim=16, heads=4, query_tokens=5, register_tokens=2,
            motion_depth=2, max_frames=4, auxiliary_depth=1, auxiliary_max_tokens=32,
            motion_evidence_fusion='off', motion_evidence_heads=0, gradient_checkpointing=False)
        self.new = VirtualQueryXYZRig(nn.Identity(), decoder, config, auxiliary_vocab_size=12)
        upstream = nn.Module()
        upstream.hidden_size = 16
        upstream.transformer = deepcopy(decoder)
        tokenizer = SimpleNamespace(vocab_size=32, num_discrete=24, eos=30, token_id_branch=29)
        self.old = DynamicRigUniRigAR(upstream, nn.Identity(), tokenizer,
            query_tokens=5, surface_no_grad=False, condition_fusion_heads=4)
        self.new.eval()
        self.old.eval()
        self.ids = torch.tensor([[[25, 26, 5, 6, 7, 29, 8, 9, 30],
                                  [25, 26, 5, 11, 7, 30, 0, 0, 0]],
                                 [[25, 26, 3, 4, 5, 30, 0, 0, 0],
                                  [25, 26, 8, 9, 10, 29, 5, 6, 30]]])
        self.mask = self.ids != 0

    def old_loss(self, condition, ids, mask):
        return self.old._ar_losses(condition, {'input_ids': ids, 'attention_mask': mask},
            include_loop_recovery=False, include_generated_prefix_recovery=False)['ce_loss']

    def test_same_condition_logits_loss_gradients_and_adam_update(self):
        new_condition = torch.randn(1, 5, 16, requires_grad=True)
        old_condition = new_condition.detach().clone().requires_grad_(True)
        ids, mask = self.ids[:1, 0], self.mask[:1, 0]
        seen = []
        hook = self.old.transformer.register_forward_hook(lambda module, args, output: seen.append(output.logits))
        old_loss = self.old_loss(old_condition, ids, mask)
        hook.remove()
        new_logits = self.new.main_logits(new_condition, ids[:, :-1], mask[:, :-1])
        old_logits = seen[0][:, 5:-1]
        torch.testing.assert_close(new_logits, old_logits, atol=1e-6, rtol=1e-6)
        labels = ids[:, 1:].masked_fill(~mask[:, 1:], -100)
        new_loss = sequence_ce(new_logits, labels)[0].mean()
        torch.testing.assert_close(new_loss, old_loss, atol=1e-7, rtol=1e-7)
        new_loss.backward()
        old_loss.backward()
        torch.testing.assert_close(new_condition.grad, old_condition.grad, atol=1e-7, rtol=1e-5)
        new_params = dict(self.new.xyz_decoder.named_parameters())
        old_params = dict(self.old.transformer.named_parameters())
        self.assertEqual(new_params.keys(), old_params.keys())
        for name in new_params:
            torch.testing.assert_close(new_params[name].grad, old_params[name].grad, atol=1e-6, rtol=1e-5)
        optimizers = [torch.optim.AdamW(params.values(), lr=1e-4, weight_decay=0.)
                      for params in (new_params, old_params)]
        for optimizer, params in zip(optimizers, (new_params, old_params)):
            torch.nn.utils.clip_grad_norm_(params.values(), 1.)
            optimizer.step()
        for name in new_params:
            torch.testing.assert_close(new_params[name], old_params[name], atol=1e-6, rtol=1e-5)

    def test_all_frame_loss_and_condition_gradients_equal_legacy_individual_calls(self):
        real = torch.randn(2, 2, 5, 16, requires_grad=True)
        virtual = torch.randn(2, 5, 16, requires_grad=True)
        targets = {'main_input_ids': self.ids, 'main_attention_mask': self.mask}
        actual = self.new.forward_encoded(real, virtual, targets, auxiliary_weight=0.)['main_loss']
        condition = self.new.conditions(real, virtual)
        references = [self.old_loss(condition[b:b+1, t], self.ids[b:b+1, t], self.mask[b:b+1, t])
                      for b in range(2) for t in range(2)]
        expected = torch.stack(references).mean()
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        variables = (real, virtual, self.new.frame_fusion[0].cross.in_proj_weight)
        ga = torch.autograd.grad(actual, variables, retain_graph=True)
        ge = torch.autograd.grad(expected, variables)
        for a, e in zip(ga, ge):
            torch.testing.assert_close(a, e, atol=1e-7, rtol=1e-5)
        self.assertTrue(all(float(g.norm()) > 0 for g in ga))

    def test_frame_batch_permutation_and_single_frame_reduction(self):
        real, virtual = torch.randn(2, 2, 5, 16), torch.randn(2, 5, 16)
        targets = {'main_input_ids': self.ids, 'main_attention_mask': self.mask}
        loss = self.new.forward_encoded(real, virtual, targets, auxiliary_weight=0.)['main_loss']
        permuted = {'main_input_ids': self.ids.flip(1), 'main_attention_mask': self.mask.flip(1)}
        other = self.new.forward_encoded(real.flip(1), virtual, permuted, auxiliary_weight=0.)['main_loss']
        torch.testing.assert_close(loss, other, atol=1e-7, rtol=1e-7)
        one = {'main_input_ids': self.ids[:1, :1], 'main_attention_mask': self.mask[:1, :1]}
        single = self.new.forward_encoded(real[:1, :1], virtual[:1], one, auxiliary_weight=0.)['main_loss']
        cond = self.new.conditions(real[:1, :1], virtual[:1])[:, 0]
        expected = self.old_loss(cond, self.ids[:1, 0], self.mask[:1, 0])
        torch.testing.assert_close(single, expected, atol=1e-7, rtol=1e-7)

    def test_reference_controls_call_actual_legacy_encoder_and_preserve_target_frame(self):
        self.new.surface_tokenizer = Surface()
        surface = {key: torch.randn(2, 2, 5, 3) for key in
                   ('dense_points', 'dense_normals', 'query_points', 'query_normals')}
        tokens = torch.stack([self.new.surface_tokenizer(*(surface[key][:, frame] for key in
            ('dense_points', 'dense_normals', 'query_points', 'query_normals'))) for frame in range(2)], 1)
        static = ReferenceConditionRig(self.new, 'static_surface').eval()
        static_conditions, _ = static.encode(surface)
        torch.testing.assert_close(static_conditions, tokens, atol=0, rtol=0)
        legacy = ReferenceConditionRig(self.new, 'legacy_query_first').eval()
        actual, _ = legacy.encode(surface)
        expected = torch.stack([AnchorWiseAlternatingMotionEncoder.forward(self.new.motion_encoder,
            tokens[:, order], query_points=surface['query_points'][:, order])
            for order in ([0, 1], [1, 0])], 1)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        flipped = {key: value.flip(1) for key, value in surface.items()}
        torch.testing.assert_close(legacy.encode(flipped)[0], actual.flip(1), atol=1e-6, rtol=1e-6)
        out = legacy(surface, {'main_input_ids': self.ids, 'main_attention_mask': self.mask}, auxiliary_weight=0.)
        out['loss'].backward()
        self.assertGreater(float(self.new.surface_tokenizer.projection.weight.grad.norm()), 0)
        self.assertGreater(float(self.new.motion_encoder.blocks[0].pose_inner.self_attn.in_proj_weight.grad.norm()), 0)
        self.assertIsNone(self.new.motion_encoder.virtual_slots.grad)
        self.assertIsNone(self.new.frame_fusion[0].cross.in_proj_weight.grad)

    def test_virtual_final_real_isolation_preserves_encoder_and_bypasses_only_fusion(self):
        self.new.surface_tokenizer = Surface()
        surface = {key: torch.randn(2, 2, 5, 3) for key in
                   ('dense_points', 'dense_normals', 'query_points', 'query_normals')}
        reference = ReferenceConditionRig(self.new, 'virtual_final_real').eval()
        expected_real, expected_virtual = self.new.encode(surface)
        real, virtual = reference.encode(surface)
        torch.testing.assert_close(real, expected_real, atol=0, rtol=0)
        torch.testing.assert_close(virtual, expected_virtual, atol=0, rtol=0)
        self.assertIs(reference.conditions(real, virtual), real)
        targets = {'main_input_ids': self.ids, 'main_attention_mask': self.mask}
        out = reference.forward_encoded(real, virtual, targets, auxiliary_weight=0.)
        expected = torch.stack([self.old_loss(expected_real[b:b+1, t], self.ids[b:b+1, t], self.mask[b:b+1, t])
                                for b in range(2) for t in range(2)]).mean()
        torch.testing.assert_close(out['main_loss'], expected, atol=1e-6, rtol=1e-6)
        out['loss'].backward()
        self.assertGreater(float(self.new.motion_encoder.virtual_slots.grad.norm()), 0)
        self.assertGreater(float(self.new.surface_tokenizer.projection.weight.grad.norm()), 0)
        self.assertGreater(float(self.new.motion_encoder.blocks[0].pose_inner.self_attn.in_proj_weight.grad.norm()), 0)
        self.assertIsNone(self.new.frame_fusion[0].cross.in_proj_weight.grad)
        self.assertIsNone(self.new.auxiliary.embedding.weight.grad)
        with self.assertRaisesRegex(ValueError, 'no auxiliary'):
            reference.forward_encoded(real, virtual, targets, auxiliary_weight=.1)


if __name__ == '__main__':
    unittest.main()
