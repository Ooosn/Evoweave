import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers import OPTConfig, OPTForCausalLM

from rigweave.dynamic_rig.virtual_query_xyz import VirtualQueryXYZConfig, VirtualQueryXYZRig, sequence_ce


class Surface(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(6, dim)

    def forward(self, dense, normals, query, query_normals):
        return self.linear(torch.cat((query, query_normals), dim=-1))


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        torch.set_num_threads(1)
        self.config = VirtualQueryXYZConfig(dim=16, heads=4, query_tokens=5, register_tokens=2,
            motion_depth=2, max_frames=4, auxiliary_depth=1, auxiliary_max_tokens=32,
            motion_evidence_fusion='off', motion_evidence_heads=0, gradient_checkpointing=False,
            temporal_chunk_size=100)
        decoder = OPTForCausalLM(OPTConfig(vocab_size=32, hidden_size=16, word_embed_proj_dim=16,
            ffn_dim=32, num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=128,
            dropout=0., attention_dropout=0., activation_dropout=0., _attn_implementation='eager'))
        self.model = VirtualQueryXYZRig(Surface(16), decoder, self.config, auxiliary_vocab_size=12)
        self.surface = {key: torch.randn(2, 3, 5, 3) for key in
                        ('dense_points', 'dense_normals', 'query_points', 'query_normals')}
        self.targets = {'main_input_ids': torch.randint(3, 32, (2, 3, 7)),
                        'main_attention_mask': torch.ones(2, 3, 7, dtype=torch.bool),
                        'aux_input_ids': torch.randint(1, 12, (2, 5)),
                        'aux_attention_mask': torch.ones(2, 5, dtype=torch.bool),
                        'aux_labels': torch.randint(1, 12, (2, 5))}

    def test_shapes_and_temporal_virtual_anchor_inclusion(self):
        calls = []
        hook = self.model.motion_encoder.blocks[0].anchor_temporal.register_forward_pre_hook(
            lambda module, args: calls.append(tuple(args[0].shape)))
        real, virtual = self.model.encode(self.surface)
        hook.remove()
        self.assertEqual(real.shape, (2, 3, 5, 16))
        self.assertEqual(virtual.shape, (2, 5, 16))
        self.assertEqual(calls, [(2*(1+2+5), 4, 16)])
        self.assertEqual(self.model.conditions(real, virtual).shape, real.shape)

    def test_aux_targets_and_head_cannot_change_main_logits(self):
        self.model.eval()
        real, virtual = self.model.encode(self.surface)
        first = self.model.forward_encoded(real, virtual, self.targets, auxiliary_weight=.1)
        changed = dict(self.targets, aux_labels=(self.targets['aux_labels']+1) % 12)
        with torch.no_grad():
            self.model.auxiliary.head.weight.add_(3.)
        second = self.model.forward_encoded(real, virtual, changed, auxiliary_weight=.1)
        self.assertTrue(torch.equal(first['main_loss'], second['main_loss']))
        self.assertTrue(torch.equal(first['main_accuracy'], second['main_accuracy']))
        with patch.object(self.model.auxiliary, 'forward', side_effect=AssertionError('auxiliary used')):
            zero = self.model.forward_encoded(real, virtual, changed, auxiliary_weight=0.)
        self.assertTrue(torch.equal(zero['main_loss'], second['main_loss']))

    def test_each_loss_reaches_virtual_and_main_xyz_head(self):
        output = self.model(self.surface, self.targets, auxiliary_weight=.1)
        virtual = self.model.motion_encoder.virtual_slots
        for name in ['main_loss', 'auxiliary_loss']:
            grad = torch.autograd.grad(output[name], virtual, retain_graph=True)[0]
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.norm()), 0)
        output['loss'].backward()
        for parameter in [self.model.surface_tokenizer.linear.weight,
                          self.model.frame_fusion[0].cross.in_proj_weight,
                          self.model.xyz_decoder.get_input_embeddings().weight,
                          self.model.xyz_decoder.get_output_embeddings().weight]:
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.norm()), 0)

    def test_final_target_frame_features_are_used(self):
        self.model.eval()
        real, virtual = self.model.encode(self.surface)
        c1 = self.model.conditions(real, virtual)
        changed = real.clone()
        changed[:, 1] = torch.randn_like(changed[:, 1])
        c2 = self.model.conditions(changed, virtual)
        self.assertTrue(torch.equal(c1[:, 0], c2[:, 0]))
        self.assertFalse(torch.allclose(c1[:, 1], c2[:, 1]))
        self.assertTrue(torch.equal(c1[:, 2], c2[:, 2]))

    def test_cached_logits_match_teacher_prefix(self):
        self.model.eval()
        condition = torch.randn(1, 5, 16)
        ids = self.targets['main_input_ids'][:1, 0, :5]
        mask = torch.ones_like(ids)
        full = self.model.main_logits(condition, ids, mask)
        first = self.model.xyz_decoder(inputs_embeds=torch.cat((condition,
            self.model.xyz_decoder.get_input_embeddings()(ids[:, :2])), dim=1),
            attention_mask=torch.ones(1, 7, dtype=torch.long), use_cache=True)
        self.assertTrue(torch.allclose(first.logits[:, -1], full[:, 1], atol=1e-6))
        for index in range(2, ids.shape[1]):
            first = self.model.xyz_decoder(input_ids=ids[:, index:index+1], past_key_values=first.past_key_values,
                attention_mask=torch.ones(1, 5+index+1, dtype=torch.long), use_cache=True)
            self.assertTrue(torch.allclose(first.logits[:, -1], full[:, index], atol=1e-6))

    def test_natural_eos_and_token_cap_are_distinct(self):
        self.model.eval()
        def choose(token):
            def processor(ids, scores):
                scores = scores.new_full(scores.shape, -1e9)
                scores[:, token] = 0
                return scores
            return processor
        condition = torch.randn(1, 5, 16)
        args = dict(eos_token=2, max_new_tokens=3)
        end = self.model.generate_condition(condition, torch.tensor([1, 4]), logits_processor=choose(2), **args)
        self.assertTrue(end['success'])
        self.assertEqual(end['tokens'], [1, 4, 2])
        cap = self.model.generate_condition(condition, torch.tensor([1, 4]), logits_processor=choose(3), **args)
        self.assertFalse(cap['success'])
        self.assertEqual(cap['status'], 'max_tokens')
        self.assertEqual(cap['tokens'], [1, 4, 3, 3, 3])

    def test_loss_ignores_padding_and_does_not_scale_with_frames(self):
        logits = torch.randn(2, 5, 12)
        labels = self.targets['aux_labels'].clone()
        labels[0, 3:] = -100
        a = sequence_ce(logits, labels)[0]
        changed = logits.clone()
        changed[0, 3:] = 1000
        b = sequence_ce(changed, labels)[0]
        self.assertTrue(torch.equal(a, b))
        self.model.eval()
        real, virtual = self.model.encode(self.surface)
        one = {**self.targets, 'main_input_ids': self.targets['main_input_ids'][:, :1],
               'main_attention_mask': self.targets['main_attention_mask'][:, :1]}
        repeat = {**one, 'main_input_ids': one['main_input_ids'].repeat(1, 3, 1),
                  'main_attention_mask': one['main_attention_mask'].repeat(1, 3, 1)}
        loss1 = self.model.forward_encoded(real[:, :1], virtual, one, auxiliary_weight=.1)['loss']
        loss2 = self.model.forward_encoded(real[:, :1].repeat(1, 3, 1, 1), virtual, repeat, auxiliary_weight=.1)['loss']
        self.assertTrue(torch.allclose(loss1, loss2, atol=1e-6))

    def test_checkpointed_path_backpropagates(self):
        self.model.config = VirtualQueryXYZConfig(**{**self.config.__dict__, 'gradient_checkpointing': True})
        self.model.motion_encoder.gradient_checkpointing = True
        loss = self.model(self.surface, self.targets, auxiliary_weight=.1)['loss']
        loss.backward()
        self.assertTrue(torch.isfinite(self.model.motion_encoder.virtual_slots.grad).all())


if __name__ == '__main__':
    unittest.main()
