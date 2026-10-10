from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest import mock

import torch


_MODULE_PATH = Path(__file__).resolve().parents[1] / "src/rigweave/dynamic_rig/motion_relation_residual.py"
_SPEC = importlib.util.spec_from_file_location("motion_relation_residual", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
MotionRelationResidual = _MODULE.MotionRelationResidual


def reference(module, tokens, states):
    hidden = module.in_proj(module.norm(tokens))
    batch, frames, anchors, _ = hidden.shape
    messages = []
    for channel in range(3):
        message = torch.stack([
            torch.stack([
                torch.stack([
                    (states[b, q, :, channel, None] * hidden[b, t]).sum(0) / anchors
                    for q in range(anchors)
                ]) for t in range(frames)
            ]) for b in range(batch)
        ])
        messages.append(message)
    mixed = module.activation(module.mix_proj(torch.cat((hidden, *messages), dim=-1)))
    return tokens + module.out_proj(mixed)


class MotionRelationResidualTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(713)
        self.module = MotionRelationResidual(12, 5).double()
        self.tokens = torch.randn(2, 3, 4, 12, dtype=torch.float64)
        self.states = torch.rand(2, 4, 4, 3, dtype=torch.float64)

    def enable_residual(self):
        with torch.no_grad():
            self.module.out_proj.weight.normal_(0, 0.2)

    def test_default_bottleneck_and_zero_output_initialization(self):
        module = MotionRelationResidual(12)
        self.assertFalse(module.reference_subtraction)
        self.assertEqual(module.bottleneck_dim, 64)
        self.assertEqual(module.in_proj.out_features, 64)
        self.assertEqual(module.mix_proj.in_features, 256)
        for parameter in self.module.out_proj.parameters():
            self.assertEqual(int(torch.count_nonzero(parameter)), 0)
        for layer in (self.module.in_proj, self.module.mix_proj):
            self.assertGreater(int(torch.count_nonzero(layer.weight)), 0)
        torch.testing.assert_close(self.module.norm.weight, torch.ones_like(self.module.norm.weight))

    def test_zero_initialization_is_exact_identity(self):
        for states in (self.states, torch.zeros_like(self.states)):
            with self.subTest(all_zero=not bool(states.any())):
                actual = self.module(self.tokens, states)
                self.assertTrue(torch.equal(actual, self.tokens))
                self.assertEqual(actual.shape, self.tokens.shape)
                self.assertEqual(actual.dtype, self.tokens.dtype)

    def test_first_step_gradient_gate_opens_after_output_update(self):
        probe = torch.randn_like(self.tokens)
        tokens = self.tokens.clone().requires_grad_(True)
        (self.module(tokens, self.states) * probe).sum().backward()
        for parameter in self.module.out_proj.parameters():
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        for layer in (self.module.norm, self.module.in_proj, self.module.mix_proj):
            for parameter in layer.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertEqual(int(torch.count_nonzero(parameter.grad)), 0)
        torch.testing.assert_close(tokens.grad, probe, atol=0, rtol=0)
        with torch.no_grad():
            for parameter in self.module.out_proj.parameters():
                parameter.add_(parameter.grad, alpha=-0.01)
        self.module.zero_grad(set_to_none=True)
        (self.module(tokens, self.states) * probe).sum().backward()
        for layer in (self.module.norm, self.module.in_proj, self.module.mix_proj):
            self.assertGreater(float(layer.weight.grad.abs().sum()), 0)

    def test_forward_and_gradients_match_explicit_pair_reference(self):
        self.enable_residual()
        tokens = self.tokens.clone().requires_grad_(True)
        states = self.states.clone().requires_grad_(True)
        actual = self.module(tokens, states)
        expected = reference(self.module, tokens, states)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        probe = torch.randn_like(tokens)
        variables = (tokens, states, *self.module.parameters())
        actual_gradients = torch.autograd.grad((actual * probe).sum(), variables)
        expected_gradients = torch.autograd.grad((expected * probe).sum(), variables)
        for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-11, rtol=1e-11)

    def test_all_three_evidence_channels_affect_output(self):
        self.enable_residual()
        baseline = self.module(self.tokens, torch.zeros_like(self.states))
        for channel in range(3):
            with self.subTest(channel=channel):
                states = torch.zeros_like(self.states)
                states[..., channel] = self.states[..., channel]
                actual = self.module(self.tokens, states)
                self.assertGreater(float((actual - baseline).detach().abs().max()), 1e-8)

    def test_zero_evidence_can_learn_query_conditioned_residual(self):
        states = torch.zeros_like(self.states)
        before = states.clone()
        optimizer = torch.optim.SGD(self.module.parameters(), lr=0.1)
        target_delta = self.tokens.sin()
        loss = (self.module(self.tokens, states) - self.tokens - target_delta).square().mean()
        loss.backward()
        optimizer.step()
        captured = []
        handle = self.module.mix_proj.register_forward_pre_hook(
            lambda _module, args: captured.append(args[0].detach())
        )
        try:
            delta = self.module(self.tokens, states) - self.tokens
        finally:
            handle.remove()
        self.assertGreater(float(delta.detach().abs().max()), 0)
        self.assertGreater(float(delta.detach().var(dim=2).sum()), 0)
        self.assertEqual(int(torch.count_nonzero(captured[0][..., self.module.bottleneck_dim:])), 0)
        self.assertTrue(torch.equal(states, before))

    def test_zero_channels_are_finite_in_forward_and_backward(self):
        self.enable_residual()
        for active_channel in (None, 0, 1, 2):
            with self.subTest(active_channel=active_channel):
                states = torch.zeros_like(self.states)
                if active_channel is not None:
                    states[..., active_channel] = self.states[..., active_channel]
                tokens = self.tokens.clone().requires_grad_(True)
                self.module.zero_grad(set_to_none=True)
                actual = self.module(tokens, states)
                actual.square().mean().backward()
                self.assertTrue(bool(torch.isfinite(actual).all()))
                self.assertTrue(bool(torch.isfinite(tokens.grad).all()))
                for parameter in self.module.parameters():
                    self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_fixed_anchor_denominator_preserves_weak_evidence(self):
        self.enable_residual()
        captured = []
        handle = self.module.mix_proj.register_forward_pre_hook(
            lambda _module, args: captured.append(args[0].detach())
        )
        try:
            self.module(self.tokens, self.states)
            self.module(self.tokens, self.states * 1e-8)
        finally:
            handle.remove()
        width = self.module.bottleneck_dim
        torch.testing.assert_close(captured[1][..., :width], captured[0][..., :width], atol=0, rtol=0)
        torch.testing.assert_close(captured[1][..., width:], captured[0][..., width:] * 1e-8,
                                   atol=1e-23, rtol=1e-12)

    def test_anchor_permutation_equivariance(self):
        self.enable_residual()
        order = torch.tensor([3, 1, 0, 2])
        expected = self.module(self.tokens, self.states)[:, :, order]
        actual = self.module(self.tokens[:, :, order], self.states[:, order][:, :, order])
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    def test_time_permutation_equivariance(self):
        self.enable_residual()
        order = torch.tensor([2, 0, 1])
        expected = self.module(self.tokens, self.states)[:, order]
        actual = self.module(self.tokens[:, order], self.states)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    def test_noncontiguous_inputs_are_not_mutated(self):
        self.enable_residual()
        tokens = self.tokens.transpose(1, 2)
        states = torch.rand(2, 3, 3, 3, dtype=torch.float64).transpose(1, 2)
        self.assertFalse(tokens.is_contiguous())
        self.assertFalse(states.is_contiguous())
        tokens_before, states_before = tokens.clone(), states.clone()
        actual = self.module(tokens, states)
        expected = reference(self.module, tokens, states)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        self.assertTrue(torch.equal(tokens, tokens_before))
        self.assertTrue(torch.equal(states, states_before))

    def test_bmm_only_uses_shared_pairs_and_bottleneck_values(self):
        with mock.patch.object(_MODULE.torch, "bmm", wraps=torch.bmm) as multiply:
            self.module(self.tokens, self.states)
        self.assertEqual(multiply.call_count, 3)
        for call in multiply.call_args_list:
            pairs, values = call.args
            self.assertEqual(pairs.shape, (2, 4, 4))
            self.assertEqual(values.shape, (2, 4, 3 * 5))
            self.assertEqual(pairs.untyped_storage().data_ptr(), self.states.untyped_storage().data_ptr())

    def test_float32_and_bfloat16_module_dtypes(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                module = MotionRelationResidual(12, 5).to(dtype=dtype)
                tokens = self.tokens.to(dtype=dtype)
                self.assertTrue(torch.equal(module(tokens, self.states), tokens))
                with torch.no_grad():
                    module.out_proj.weight.normal_(0, 0.2)
                actual = module(tokens, self.states)
                self.assertEqual(actual.dtype, dtype)
                self.assertTrue(bool(torch.isfinite(actual).all()))

    def test_cpu_autocast_keeps_residual_dtype_and_gradients(self):
        module = MotionRelationResidual(12, 5)
        tokens, states = self.tokens.float().requires_grad_(True), self.states.float()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            self.assertTrue(torch.equal(module(tokens, states), tokens))
        with torch.no_grad():
            module.out_proj.weight.normal_(0, 0.2)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = module(tokens, states)
            loss = actual.square().mean()
        loss.backward()
        self.assertEqual(actual.dtype, tokens.dtype)
        self.assertTrue(bool(torch.isfinite(actual).all()))
        self.assertTrue(bool(torch.isfinite(tokens.grad).all()))
        for layer in (module.in_proj, module.mix_proj, module.out_proj):
            self.assertTrue(bool(torch.isfinite(layer.weight.grad).all()))
            self.assertGreater(float(layer.weight.grad.abs().sum()), 0)

    def test_single_anchor_and_single_frame(self):
        self.enable_residual()
        tokens, states = self.tokens[:1, :1, :1], self.states[:1, :1, :1]
        actual = self.module(tokens, states)
        torch.testing.assert_close(actual, reference(self.module, tokens, states), atol=1e-12, rtol=1e-12)

    def test_invalid_dimensions_raise_value_error(self):
        for name in ("dim", "bottleneck_dim"):
            for value in (0, -1, 1.5, True):
                with self.subTest(name=name, value=value):
                    settings = {"dim": 12, "bottleneck_dim": 5, name: value}
                    with self.assertRaises(ValueError):
                        MotionRelationResidual(**settings)

    def test_invalid_token_shapes_raise_value_error(self):
        for shape in ((2, 4, 12), (2, 3, 4, 12, 1), (2, 3, 4, 11),
                      (0, 3, 4, 12), (2, 0, 4, 12), (2, 3, 0, 12)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.module(torch.zeros(shape, dtype=torch.float64), self.states)

    def test_invalid_evidence_shapes_raise_value_error(self):
        for shape in ((2, 4, 4), (2, 3, 4, 4, 3), (1, 4, 4, 3),
                      (2, 3, 4, 3), (2, 4, 3, 3), (2, 4, 4, 2)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.module(self.tokens, torch.zeros(shape, dtype=torch.float64))

    def test_nonfloating_inputs_and_device_mismatch_raise_value_error(self):
        with self.assertRaises(ValueError):
            self.module(self.tokens.long(), self.states)
        with self.assertRaises(ValueError):
            self.module(self.tokens, self.states.long())
        with self.assertRaises(ValueError):
            self.module(self.tokens, torch.empty(self.states.shape, device="meta"))


class MotionRelationReferenceSubtractionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(917)
        self.module = MotionRelationResidual(12, 7, reference_subtraction=True).double()
        self.tokens = torch.randn(2, 3, 7, 12, dtype=torch.float64)
        self.states = torch.rand(2, 7, 7, 3, dtype=torch.float64).softmax(-1)

    def unknown_states(self):
        states = torch.zeros_like(self.states)
        states[..., 0] = 1
        return states

    def enable_residual(self):
        with torch.no_grad():
            self.module.out_proj.weight.normal_(0, 0.2)
            self.module.out_proj.bias.normal_(0, 0.2)

    def test_unknown_is_exact_identity_with_learned_output_and_autocast(self):
        self.enable_residual()
        for dtype, autocast in ((torch.float64, False), (torch.float32, False),
                                (torch.bfloat16, False), (torch.float32, True)):
            with self.subTest(dtype=dtype, autocast=autocast):
                self.module.to(dtype=dtype)
                tokens = self.tokens.to(dtype=dtype)
                captured = []
                handle = self.module.out_proj.register_forward_hook(
                    lambda _module, _args, output: captured.append(output.detach())
                )
                try:
                    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
                        actual = self.module(tokens, self.unknown_states())
                finally:
                    handle.remove()
                self.assertEqual(len(captured), 2)
                self.assertGreater(int(torch.count_nonzero(captured[0])), 0)
                self.assertTrue(torch.equal(captured[0], captured[1]))
                self.assertTrue(torch.equal(actual, tokens))
                self.assertEqual(actual.dtype, tokens.dtype)

    def test_unknown_parameter_gradients_cancel_exactly(self):
        self.enable_residual()
        for dtype, autocast in ((torch.float64, False), (torch.float32, False),
                                (torch.bfloat16, False), (torch.float32, True)):
            with self.subTest(dtype=dtype, autocast=autocast):
                self.module.to(dtype=dtype)
                self.module.zero_grad(set_to_none=True)
                tokens = self.tokens.to(dtype=dtype).detach().requires_grad_(True)
                probe = torch.randn_like(tokens)
                with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
                    actual = self.module(tokens, self.unknown_states())
                    loss = (actual * probe).sum()
                loss.backward()
                self.assertTrue(torch.equal(tokens.grad, probe))
                for name, parameter in self.module.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertEqual(int(torch.count_nonzero(parameter.grad)), 0, name)

    def test_nonuniform_evidence_opens_gradient_gate_but_bias_cancels(self):
        probe = torch.randn_like(self.tokens)
        actual = self.module(self.tokens, self.states)
        self.assertTrue(torch.equal(actual, self.tokens))
        (actual * probe).sum().backward()
        self.assertGreater(float(self.module.out_proj.weight.grad.abs().sum()), 0)
        self.assertEqual(int(torch.count_nonzero(self.module.out_proj.bias.grad)), 0)
        for layer in (self.module.norm, self.module.in_proj, self.module.mix_proj):
            for parameter in layer.parameters():
                self.assertEqual(int(torch.count_nonzero(parameter.grad)), 0)
        with torch.no_grad():
            self.module.out_proj.weight.add_(self.module.out_proj.weight.grad, alpha=-0.01)
        self.module.zero_grad(set_to_none=True)
        (self.module(self.tokens, self.states) * probe).sum().backward()
        for layer in (self.module.norm, self.module.in_proj, self.module.mix_proj):
            self.assertGreater(float(layer.weight.grad.abs().sum()), 0)
        self.assertEqual(int(torch.count_nonzero(self.module.out_proj.bias.grad)), 0)

    def test_forward_and_gradients_match_explicit_subtraction(self):
        self.enable_residual()
        tokens = self.tokens.clone().requires_grad_(True)
        states = self.states.clone().requires_grad_(True)
        actual = self.module(tokens, states)
        expected = tokens + reference(self.module, tokens, states) - reference(self.module, tokens, self.unknown_states())
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        variables = (tokens, states, *self.module.parameters())
        probe = torch.randn_like(tokens)
        actual_gradients = torch.autograd.grad((actual * probe).sum(), variables)
        expected_gradients = torch.autograd.grad((expected * probe).sum(), variables)
        for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-11, rtol=1e-11)

    def test_state_dict_keys_and_strict_load_are_unchanged(self):
        self.enable_residual()
        plain = MotionRelationResidual(12, 7).double()
        self.assertEqual(list(plain.state_dict()), list(self.module.state_dict()))
        loaded = plain.load_state_dict(self.module.state_dict(), strict=True)
        self.assertEqual(loaded.missing_keys, [])
        self.assertEqual(loaded.unexpected_keys, [])
        loaded = self.module.load_state_dict(plain.state_dict(), strict=True)
        self.assertEqual(loaded.missing_keys, [])
        self.assertEqual(loaded.unexpected_keys, [])
        for name, parameter in self.module.state_dict().items():
            self.assertTrue(torch.equal(parameter, plain.state_dict()[name]), name)
        self.assertFalse(plain.reference_subtraction)
        self.assertTrue(self.module.reference_subtraction)
        self.assertTrue(torch.equal(self.module(self.tokens, self.unknown_states()), self.tokens))
        self.assertFalse(torch.equal(plain(self.tokens, self.unknown_states()), self.tokens))

    def test_reference_flag_is_keyword_only_and_requires_bool(self):
        with self.assertRaises(TypeError):
            MotionRelationResidual(12, 7, True)
        for value in (0, 1, None, "true", [], torch.tensor(True)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MotionRelationResidual(12, reference_subtraction=value)

    def test_readonly_evidence_and_four_compact_matching_bmms(self):
        self.enable_residual()
        states = self.states.transpose(1, 2)
        before = states.clone()
        with mock.patch.object(_MODULE.torch, "bmm", wraps=torch.bmm) as multiply:
            self.module(self.tokens, states)
        self.assertTrue(torch.equal(states, before))
        self.assertEqual(multiply.call_count, 4)
        values = multiply.call_args_list[0].args[1]
        for index, call in enumerate(multiply.call_args_list):
            pairs, current_values = call.args
            self.assertEqual(pairs.shape, (2, 7, 7))
            self.assertEqual(current_values.shape, (2, 7, 3 * 7))
            self.assertTrue(pairs.is_contiguous())
            self.assertIs(current_values, values)
            expected = states[..., index] if index < 3 else torch.ones_like(pairs)
            torch.testing.assert_close(pairs, expected, atol=0, rtol=0)

    def test_normalization_and_hidden_are_shared_between_branches(self):
        self.enable_residual()
        with mock.patch.object(self.module.norm, "forward", wraps=self.module.norm.forward) as norm:
            with mock.patch.object(self.module.in_proj, "forward", wraps=self.module.in_proj.forward) as hidden:
                with mock.patch.object(self.module.mix_proj, "forward", wraps=self.module.mix_proj.forward) as mix:
                    self.module(self.tokens, self.states)
        self.assertEqual(norm.call_count, 1)
        self.assertEqual(hidden.call_count, 1)
        self.assertEqual(mix.call_count, 2)
        first = mix.call_args_list[0].args[0][..., :self.module.bottleneck_dim]
        second = mix.call_args_list[1].args[0][..., :self.module.bottleneck_dim]
        torch.testing.assert_close(first, second, atol=0, rtol=0)

    def test_all_three_actual_evidence_channels_remain_effective(self):
        self.enable_residual()
        unknown = self.unknown_states()
        for channel in range(3):
            with self.subTest(channel=channel):
                states = unknown.clone()
                states[..., channel] = self.states[..., channel]
                delta = self.module(self.tokens, states) - self.tokens
                self.assertGreater(float(delta.detach().abs().max()), 1e-8)

    def test_anchor_permutation_equivariance(self):
        self.enable_residual()
        order = torch.tensor([6, 3, 0, 5, 1, 4, 2])
        expected = self.module(self.tokens, self.states)[:, :, order]
        actual = self.module(self.tokens[:, :, order], self.states[:, order][:, :, order])
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    def test_time_permutation_equivariance(self):
        self.enable_residual()
        order = torch.tensor([2, 0, 1])
        expected = self.module(self.tokens, self.states)[:, order]
        actual = self.module(self.tokens[:, order], self.states)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
