"""BOS-only auxiliary generation, with parsing deliberately outside the model."""
import unittest
from unittest.mock import patch

import torch

from rigweave.dynamic_rig.aux_skeleton_tokens import AuxSkeletonTokens
from test_virtual_query_fusion import make_model


class AuxiliaryGenerationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = make_model(fusion_depth=1, auxiliary_max_tokens=8).eval()
        self.codec = AuxSkeletonTokens(max_joints=8, max_sequence_tokens=8)
        self.virtual = torch.randn(1, 3, 8)
        self.arguments = dict(bos_token=self.codec.bos, eos_token=self.codec.eos,
                              max_new_tokens=self.codec.max_tokens-1)

    def choose(self, token):
        with torch.no_grad():
            self.model.auxiliary.head.weight.zero_()
            self.model.auxiliary.head.bias.fill_(-10)
            self.model.auxiliary.head.bias[token] = 10

    def test_natural_eos_includes_bos_and_eos_without_grammar_forcing(self):
        self.assertEqual((self.codec.bos, self.codec.eos), (1, 2))
        self.choose(self.codec.eos)
        result = self.model.generate_auxiliary(self.virtual, **self.arguments)
        self.assertEqual(result, {'success': True, 'status': 'eos', 'tokens': [1, 2]})
        with self.assertRaises(ValueError):
            self.codec.decode(result['tokens'])

    def test_cap_is_bos_plus_new_tokens_and_does_not_force_or_mask_eos(self):
        self.choose(self.codec.pad)
        calls = []
        original = self.model.auxiliary.forward
        def forward(memory, inputs, attention):
            calls.append((inputs.clone(), attention.clone()))
            return original(memory, inputs, attention)
        with patch.object(self.model.auxiliary, 'forward', side_effect=forward):
            result = self.model.generate_auxiliary(self.virtual, **self.arguments)
        self.assertEqual(result, {'success': False, 'status': 'max_tokens', 'tokens': [1] + [0]*7})
        self.assertEqual([ids.shape[1] for ids, _ in calls], list(range(1, 8)))
        for ids, attention in calls:
            self.assertTrue(attention.all())
            self.assertEqual(attention.dtype, torch.bool)
            self.assertEqual(ids[0, 0].item(), self.codec.bos)
        self.assertEqual(len(result['tokens']), self.codec.max_tokens)

    def test_explicit_token_ids_and_smaller_cap_are_honored(self):
        self.choose(5)
        self.assertEqual(self.model.generate_auxiliary(self.virtual, bos_token=3, eos_token=5,
            max_new_tokens=1)['tokens'], [3, 5])
        self.choose(4)
        result = self.model.generate_auxiliary(self.virtual, **{**self.arguments, 'max_new_tokens': 2})
        self.assertEqual(result['tokens'], [1, 4, 4])
        self.assertEqual(result['status'], 'max_tokens')

    def test_fake_logits_depend_on_virtual_and_use_only_generated_prefix(self):
        calls = []
        def conditioned(memory, inputs, attention):
            self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(attention.all())
            calls.append(inputs[0].tolist())
            logits = memory.new_full((1, inputs.shape[1], 12), -10)
            token = (4 if memory[0, 0, 0] > 0 else 5) if inputs.shape[1] == 1 else self.codec.eos
            logits[:, -1, token] = 10
            return logits
        positive, negative = self.virtual.clone(), self.virtual.clone()
        positive[0, 0, 0], negative[0, 0, 0] = 1., -1.
        with (patch.object(self.model.auxiliary, 'forward', side_effect=conditioned),
              patch.object(self.model, 'conditions', side_effect=AssertionError('main fusion used')),
              patch.object(self.model.xyz_decoder, 'forward', side_effect=AssertionError('XYZ decoder used'))):
            first = self.model.generate_auxiliary(positive, **self.arguments)
            second = self.model.generate_auxiliary(negative, **self.arguments)
        self.assertEqual(first['tokens'], [1, 4, 2])
        self.assertEqual(second['tokens'], [1, 5, 2])
        self.assertEqual(calls, [[1], [1, 4], [1], [1, 5]])

    def test_parameters_gradients_rng_and_modes_are_unchanged(self):
        self.choose(self.codec.eos)
        for parameter in self.model.parameters():
            parameter.grad = torch.full_like(parameter, .125)
        parameters = {name: value.clone() for name, value in self.model.state_dict().items()}
        grads = {name: value.grad.clone() for name, value in self.model.named_parameters()}
        modes = {name: module.training for name, module in self.model.named_modules()}
        rng = torch.get_rng_state().clone()
        memory = self.virtual.clone().requires_grad_()
        original = memory.detach().clone()
        self.model.generate_auxiliary(memory, **self.arguments)
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, parameters[name], atol=0, rtol=0)
        for name, value in self.model.named_parameters():
            torch.testing.assert_close(value.grad, grads[name], atol=0, rtol=0)
        self.assertEqual({name: module.training for name, module in self.model.named_modules()}, modes)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        torch.testing.assert_close(memory, original, atol=0, rtol=0)
        self.assertIsNone(memory.grad)

    def test_training_mode_is_rejected_without_mode_mutation(self):
        self.model.train()
        with self.assertRaises(ValueError):
            self.model.generate_auxiliary(self.virtual, **self.arguments)
        self.assertTrue(self.model.training)
        self.assertTrue(self.model.auxiliary.training)
        self.model.eval()
        self.model.auxiliary.train()
        with self.assertRaises(ValueError):
            self.model.generate_auxiliary(self.virtual, **self.arguments)
        self.assertFalse(self.model.training)
        self.assertTrue(self.model.auxiliary.training)

    def test_input_shape_finiteness_tokens_and_capacity_are_checked(self):
        memories = [None, self.virtual[0], self.virtual.expand(2, -1, -1), self.virtual[:, :2],
                    self.virtual.to(torch.long), torch.full_like(self.virtual, float('nan')),
                    torch.full_like(self.virtual, float('inf'))]
        for memory in memories:
            with self.subTest(memory_type=type(memory)), self.assertRaises(ValueError):
                self.model.generate_auxiliary(memory, **self.arguments)
        for name in ('bos_token', 'eos_token'):
            for token in (-1, 12, True, 1.5, None):
                with self.subTest(name=name, token=token), self.assertRaises(ValueError):
                    self.model.generate_auxiliary(self.virtual, **{**self.arguments, name: token})
        with self.assertRaises(ValueError):
            self.model.generate_auxiliary(self.virtual, **{**self.arguments, 'eos_token': self.codec.bos})
        for cap in (0, -1, 8, 100, True, 2.5, None):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                self.model.generate_auxiliary(self.virtual, **{**self.arguments, 'max_new_tokens': cap})
        for field in ('start_tokens', 'targets', 'joint_count', 'logits_processor'):
            with self.assertRaises(TypeError):
                self.model.generate_auxiliary(self.virtual, **self.arguments, **{field: None})

    def test_memory_dtype_is_converted_without_input_mutation(self):
        self.choose(self.codec.eos)
        for dtype in (torch.float64, torch.bfloat16):
            virtual = self.virtual.to(dtype)
            result = self.model.generate_auxiliary(virtual, **self.arguments)
            self.assertEqual(result['tokens'], [1, 2])
            self.assertEqual(virtual.dtype, dtype)

    def test_auxiliary_generation_and_head_do_not_change_main_loss(self):
        real = torch.randn(1, 2, 3, 8)
        targets = {'main_input_ids': torch.randint(1, 17, (1, 2, 5)),
                   'main_attention_mask': torch.ones(1, 2, 5, dtype=torch.bool)}
        before = self.model.forward_encoded(real, self.virtual, targets, auxiliary_weight=0.)
        self.choose(self.codec.eos)
        self.model.generate_auxiliary(self.virtual, **self.arguments)
        after = self.model.forward_encoded(real, self.virtual, targets, auxiliary_weight=0.)
        torch.testing.assert_close(before['main_loss_per_frame'], after['main_loss_per_frame'], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
