"""CPU checks that temporal chunking splits independent sequences, never time."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.checkpoint import set_checkpoint_early_stop

import test_id_xyz_model as fixtures
from rigweave.dynamic_rig.motion_encoder import (
    AnchorWiseAlternatingBlock, AnchorWiseAlternatingMotionEncoder,
)


def block(mode='aligned', registers=2):
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(413)
        return AnchorWiseAlternatingBlock(16, 2, mlp_ratio=2., dropout=0.,
            register_temporal_mode=mode, register_tokens=registers).train()


def values(shape, seed=97):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def objective(output):
    return (output.float() * values(output.shape, 191)).mean() + .1 * output.float().square().mean()


class TemporalBatchChunkingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.no_cuda = patch.object(torch.cuda, '_lazy_init',
                                   side_effect=AssertionError('temporal chunk tests must remain CPU-only'))
        cls.no_cuda.start()
        cls.measurements = {}

    @classmethod
    def tearDownClass(cls):
        print('TEMPORAL_CHUNK_CPU_ERRORS ' + json.dumps(cls.measurements, sort_keys=True), flush=True)
        cls.no_cuda.stop()
        torch.set_num_threads(cls.old_threads)

    def compare(self, actual, expected, *, bf16, category):
        self.assertTrue(bool(torch.isfinite(actual).all()), category)
        self.assertTrue(bool(torch.isfinite(expected).all()), category)
        actual, expected = actual.detach().float(), expected.detach().float()
        error = float((actual - expected).abs().max())
        key = ('bf16_' if bf16 else 'fp32_') + category.split(':', 1)[0]
        record = self.measurements.setdefault(key, {'tensors': 0, 'max_absolute_error': 0.})
        record['tensors'] += 1
        record['max_absolute_error'] = max(record['max_absolute_error'], error)
        if bf16:
            # BF16 epsilon is 1/128. Allow two scale-relative rounding units;
            # the small absolute floor covers nearly cancelling gradient sums.
            tolerance = max(2e-5, 2 * torch.finfo(torch.bfloat16).eps * float(expected.abs().max()))
            torch.testing.assert_close(actual, expected, rtol=0., atol=tolerance, msg=category)
        else:
            torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-7, msg=category)

    def compare_parameters(self, actual, expected, *, bf16):
        left, right = dict(actual.named_parameters()), dict(expected.named_parameters())
        self.assertEqual(left.keys(), right.keys())
        checked = 0
        for name, parameter in right.items():
            self.assertEqual(left[name].requires_grad, parameter.requires_grad)
            if not parameter.requires_grad:
                continue
            self.assertEqual(left[name].grad is None, parameter.grad is None, name)
            if parameter.grad is not None:
                self.compare(left[name].grad, parameter.grad, bf16=bf16, category='parameter:' + name)
                checked += 1
        self.assertGreater(checked, 0)

    def test_default_zero_and_setter_preserve_parameters_and_state_dict(self):
        model = block()
        self.assertEqual(model.temporal_batch_chunk_size, 0)
        identities = {name: id(parameter) for name, parameter in model.named_parameters()}
        state = deepcopy(model.state_dict())
        for size in (1, 7, 13452, 0):
            model.set_temporal_batch_chunk_size(size)
            self.assertEqual(model.temporal_batch_chunk_size, size)
            self.assertEqual({name: id(parameter) for name, parameter in model.named_parameters()}, identities)
            self.assertEqual(model.state_dict().keys(), state.keys())
            for name, value in state.items():
                torch.testing.assert_close(model.state_dict()[name], value, rtol=0., atol=0.)
            restored = block()
            restored.load_state_dict(model.state_dict(), strict=True)
            self.assertEqual(restored.temporal_batch_chunk_size, 0)

    def test_invalid_negative_bool_and_noninteger_chunk_sizes_are_rejected(self):
        for size in (-1, True, False, 1.5, '12', None, [], torch.tensor(2)):
            with self.subTest(size=repr(size)), self.assertRaises((TypeError, ValueError)):
                block().set_temporal_batch_chunk_size(size)

    def test_positive_chunk_cap_rejects_each_dropout_route(self):
        template = block()
        dropout_names = [name for name, module in template.anchor_temporal.named_modules()
                         if isinstance(module, nn.Dropout)]
        self.assertTrue(dropout_names)
        for name in [*dropout_names, 'self_attn']:
            model = block()
            module = dict(model.anchor_temporal.named_modules())[name]
            if isinstance(module, nn.Dropout):
                module.p = .1
            else:
                module.dropout = .1
            with self.subTest(module=name), self.assertRaisesRegex(ValueError, 'dropout'):
                model.set_temporal_batch_chunk_size(7)
            model.set_temporal_batch_chunk_size(0)
            self.assertEqual(model.temporal_batch_chunk_size, 0)

    def test_aligned_caps_keep_time_and_sequence_order_including_partial_tail(self):
        z = values((3, 3, 6, 16))
        expected = z.permute(0, 2, 1, 3).reshape(18, 3, 16)
        for cap, sizes in ((0, [18]), (18, [18]), (32, [18]), (7, [7, 7, 4])):
            with self.subTest(cap=cap):
                model = block()
                model.set_temporal_batch_chunk_size(cap)
                seen = []
                hook = model.anchor_temporal.register_forward_pre_hook(
                    lambda _module, args: seen.append(args[0].detach().clone()))
                try:
                    output = model._apply_temporal_attention(z, 3, 3, 6, 16)
                finally:
                    hook.remove()
                self.assertEqual(output.shape, z.shape)
                self.assertEqual([len(part) for part in seen], sizes)
                self.assertTrue(all(part.shape[1:] == (3, 16) for part in seen))
                torch.testing.assert_close(torch.cat(seen), expected, rtol=0., atol=0.)

    def test_global_register_and_anchor_routes_both_chunk_without_cutting_time(self):
        model = block('global')
        model.set_temporal_batch_chunk_size(2)
        z, seen = values((3, 3, 6, 16)), []
        hook = model.anchor_temporal.register_forward_pre_hook(
            lambda _module, args: seen.append(args[0].detach().clone()))
        try:
            output = model._apply_temporal_attention(z, 3, 3, 6, 16)
        finally:
            hook.remove()
        self.assertEqual(output.shape, z.shape)
        self.assertEqual([tuple(part.shape) for part in seen], [(2, 3, 16)] * 6 + [(2, 6, 16), (1, 6, 16)])
        anchors = torch.cat((z[:, :, :1], z[:, :, 3:]), dim=2).permute(0, 2, 1, 3).reshape(12, 3, 16)
        registers = z[:, :, 1:3].reshape(3, 6, 16)
        torch.testing.assert_close(torch.cat(seen[:6]), anchors, rtol=0., atol=0.)
        torch.testing.assert_close(torch.cat(seen[6:]), registers, rtol=0., atol=0.)

    def test_global_mode_without_registers_uses_aligned_chunk_route(self):
        model = block('global', registers=0)
        model.set_temporal_batch_chunk_size(7)
        z, seen = values((3, 3, 6, 16)), []
        hook = model.anchor_temporal.register_forward_pre_hook(lambda _module, args: seen.append(tuple(args[0].shape)))
        try:
            model._apply_temporal_attention(z, 3, 3, 6, 16)
        finally:
            hook.remove()
        self.assertEqual(seen, [(7, 3, 16), (7, 3, 16), (4, 3, 16)])

    def test_complete_block_outputs_input_and_all_parameter_gradients_match(self):
        for mode in ('aligned', 'global'):
            for bf16 in (False, True):
                for cap in (2, 7, 18, 32):
                    with self.subTest(mode=mode, bf16=bf16, cap=cap):
                        full = block(mode)
                        chunked = deepcopy(full)
                        chunked.set_temporal_batch_chunk_size(cap)
                        x = values((3, 3, 6, 16)).requires_grad_(True)
                        y = x.detach().clone().requires_grad_(True)
                        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
                            expected, actual = full(x), chunked(y)
                        objective(expected).backward()
                        objective(actual).backward()
                        self.compare(actual, expected, bf16=bf16, category='output')
                        self.compare(y.grad, x.grad, bf16=bf16, category='input')
                        self.compare_parameters(chunked, full, bf16=bf16)

    def test_checkpointed_encoder_preserves_outputs_and_gradients_in_both_modes(self):
        for mode in ('aligned', 'global'):
            for bf16 in (False, True):
                with self.subTest(mode=mode, bf16=bf16):
                    with torch.random.fork_rng(devices=[]):
                        torch.default_generator.manual_seed(271)
                        full = AnchorWiseAlternatingMotionEncoder(16, depth=2, heads=2,
                            register_tokens=2, max_frames=4, mlp_ratio=2., dropout=0.,
                            gradient_checkpointing=True, register_temporal_mode=mode,
                            frame_role_policy='all_real').train()
                    chunked = deepcopy(full)
                    calls, hooks = [], []
                    for layer in chunked.blocks:
                        layer.set_temporal_batch_chunk_size(2)
                        hooks.append(layer.anchor_temporal.register_forward_pre_hook(
                            lambda _module, args: calls.append(tuple(args[0].shape))))
                    x = values((3, 3, 3, 16)).requires_grad_(True)
                    y = x.detach().clone().requires_grad_(True)
                    try:
                        with set_checkpoint_early_stop(False), torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
                            expected = full(x, return_all=True)
                            actual = chunked(y, return_all=True)
                        forward_calls = len(calls)
                        objective(expected[1]).backward()
                        objective(actual[1]).backward()
                    finally:
                        for hook in hooks:
                            hook.remove()
                    self.assertGreater(len(calls), forward_calls)
                    self.assertTrue(all(shape[0] <= 2 for shape in calls))
                    for left, right in zip(actual, expected):
                        self.compare(left, right, bf16=bf16, category='encoder_output')
                    self.compare(y.grad, x.grad, bf16=bf16, category='encoder_input')
                    self.compare_parameters(chunked, full, bf16=bf16)
                    self.assertEqual(chunked.state_dict().keys(), full.state_dict().keys())
                    for name, value in full.state_dict().items():
                        torch.testing.assert_close(chunked.state_dict()[name], value, rtol=0., atol=0.)

    def test_checkpointed_real_idxyz_model_preserves_loss_and_all_trainable_gradients(self):
        for bf16 in (False, True):
            with self.subTest(bf16=bf16):
                full = fixtures.tiny_model(checkpointing=True, evidence='bias').train()
                full.xyz_decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
                chunked = deepcopy(full)
                for layer in chunked.motion_encoder.blocks:
                    layer.set_temporal_batch_chunk_size(5)
                surface, targets = fixtures.surface_fixture(), fixtures.target_fixture()
                with torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
                    expected = full(surface, targets, consistency_weight=1.)
                    actual = chunked(surface, targets, consistency_weight=1.)
                expected['loss'].backward()
                actual['loss'].backward()
                for name in expected:
                    self.compare(actual[name], expected[name], bf16=bf16, category='idxyz_output:' + name)
                self.compare_parameters(chunked, full, bf16=bf16)


if __name__ == '__main__':
    unittest.main()
