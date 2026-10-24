"""CPU equivalence checks for profile forwarding and local/global batch scaling."""
from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.checkpoint import set_checkpoint_early_stop

import test_id_xyz_model as fixtures
import test_id_xyz_stages as stage_fixtures

stages = stage_fixtures.stages


def profile_model(*, checkpointing=True, evidence='bias'):
    model = fixtures.tiny_model(checkpointing=checkpointing, evidence=evidence).train()
    if checkpointing:
        model.xyz_decoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})
    return model


def batch_fixture(batch):
    surface = fixtures.surface_fixture(batch=batch, frames=3)
    two = fixtures.target_fixture(frames=3)
    # Unequal target lengths and omitted frames keep the per-asset denominator relevant.
    targets = {key: value[torch.arange(batch) % 2].clone() for key, value in two.items()}
    targets['frame_mask'][1::4, -1] = False
    return surface, targets


def sliced(values, start, end):
    return {key: value[start:end] for key, value in values.items()}


def gradients(model):
    result = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            value = parameter.grad
            if value is not None:
                if not bool(torch.isfinite(value).all()):
                    raise AssertionError('nonfinite gradient: ' + name)
                value = value.detach().clone()
            result[name] = value
    if not result or not any(value is not None for value in result.values()):
        raise AssertionError('no trainable gradients were checked')
    return result


def assert_gradient_maps(actual, expected, *, rtol=0., atol=0.):
    if actual.keys() != expected.keys():
        raise AssertionError('trainable parameter coverage changed')
    for name, reference in expected.items():
        value = actual[name]
        if reference is None or value is None:
            if (reference is None) != (value is None):
                raise AssertionError('gradient presence changed: ' + name)
        else:
            torch.testing.assert_close(value, reference, rtol=rtol, atol=atol,
                                       msg=lambda message: name + ': ' + message)


def accumulate(forward, surface, targets, *, batch, local_assets=24):
    losses = []
    for start in range(0, local_assets, batch):
        end = min(start + batch, local_assets)
        sync = (forward.no_sync() if isinstance(forward, DistributedDataParallel)
                and end < local_assets else nullcontext())
        with sync:
            output = forward(sliced(surface, start, end), sliced(targets, start, end),
                             consistency_weight=1.)
            (output['loss_per_asset'].sum() / local_assets).backward()
        losses.append(output['loss_per_asset'].detach())
    return torch.cat(losses)


def ddp_profile_worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    with patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CPU test initialized CUDA')), \
            patch.object(torch.cuda, 'synchronize'):
        try:
            dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                                    timeout=timedelta(seconds=45))
        except RuntimeError as error:
            if 'makeDeviceForHostname(): unsupported gloo device' not in str(error):
                raise
            Path(directory, f'rank_{rank}.json').write_text(
                json.dumps({'rank': rank, 'skip_reason': str(error)}), encoding='utf-8')
            return
        try:
            base = profile_model()
            surface, targets = batch_fixture(48)
            reference = deepcopy(base)
            expected_output = reference(surface, targets, consistency_weight=1.)
            (expected_output['loss_per_asset'].sum() / 48).backward()
            expected_gradients = gradients(reference)
            local_surface = sliced(surface, rank * 24, (rank + 1) * 24)
            local_targets = sliced(targets, rank * 24, (rank + 1) * 24)
            checked = []
            for timed, batch in ((False, 24), (True, 24), (False, 12), (True, 12)):
                model = deepcopy(base)
                forward = stages.TimedCapacityForward(model, torch.device('cpu')) if timed else model
                wrapped = DistributedDataParallel(forward, find_unused_parameters=True,
                                                  gradient_as_bucket_view=True)
                for iteration in range(2):
                    model.zero_grad(set_to_none=True)
                    losses = accumulate(wrapped, local_surface, local_targets, batch=batch)
                    torch.testing.assert_close(losses,
                        expected_output['loss_per_asset'][rank * 24:(rank + 1) * 24],
                        rtol=2e-5, atol=2e-6)
                    actual = gradients(model)
                    assert_gradient_maps(actual, expected_gradients, rtol=3e-4, atol=3e-7)
                    groups = stages.gradient_groups(model)
                    if not all(value > 0 for value in groups.values()):
                        raise AssertionError('a model group lost its gradient')
                    checked.append({'timed': timed, 'batch': batch, 'iteration': iteration,
                                    'gradient_groups': groups})
                del wrapped, forward, model
            report = {'rank': rank, 'world_size': 2, 'global_assets': 48, 'local_assets': 24,
                      'trainable_tensors_checked': len(expected_gradients),
                      'unused_parameters': [key for key, value in expected_gradients.items() if value is None],
                      'variants': checked, 'passed': True}
            Path(directory, f'rank_{rank}.json').write_text(json.dumps(report), encoding='utf-8')
        finally:
            dist.destroy_process_group()


class IdXYZBatchProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.no_cuda = patch.object(torch.cuda, '_lazy_init',
                                   side_effect=AssertionError('CPU test initialized CUDA'))
        cls.no_cuda.start()
        cls.no_probe_run = patch.object(stages.TemporalFailureProbe, 'run',
            side_effect=AssertionError('CPU tests must not run the GPU failure probe'))
        cls.no_probe_run.start()

    @classmethod
    def tearDownClass(cls):
        cls.no_probe_run.stop()
        cls.no_cuda.stop()
        torch.set_num_threads(cls.old_threads)

    def compare_forward(self, *, checkpointing, evidence, weight, bf16=False):
        ordinary = profile_model(checkpointing=checkpointing, evidence=evidence)
        profiled = deepcopy(ordinary)
        surface, targets = batch_fixture(2)
        before = {key: value.detach().clone() for key, value in ordinary.state_dict().items()}
        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
            expected = ordinary(surface, targets, consistency_weight=weight)
        expected['loss'].backward()
        timed = stages.TimedCapacityForward(profiled, torch.device('cpu'))
        with patch.object(torch.cuda, 'synchronize') as synchronize:
            with torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
                actual = timed(surface, targets, consistency_weight=weight)
            actual['loss'].backward()
        self.assertEqual(synchronize.call_count, 2)
        self.assertEqual(set(timed.seconds), {'encode', 'decode_and_loss'})
        self.assertTrue(all(value >= 0 for value in timed.seconds.values()))
        self.assertEqual(actual.keys(), expected.keys())
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0., atol=0., msg=key)
        assert_gradient_maps(gradients(profiled), gradients(ordinary))
        self.assertEqual(stages.gradient_groups(profiled), stages.gradient_groups(ordinary))
        for name, value in profiled.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0., atol=0.)
        self.assertTrue(all(parameter.device.type == 'cpu' for parameter in profiled.parameters()))

    def test_timed_forward_matches_all_outputs_and_trainable_gradients(self):
        for checkpointing in (False, True):
            for evidence in ('off', 'bias'):
                for weight in (0., 1.):
                    with self.subTest(checkpointing=checkpointing, evidence=evidence, weight=weight):
                        self.compare_forward(checkpointing=checkpointing, evidence=evidence, weight=weight)

    def test_cpu_bfloat16_timed_forward_preserves_gradients(self):
        self.compare_forward(checkpointing=True, evidence='bias', weight=1., bf16=True)

    def test_local24_batch24_and_two_batch12_chunks_match(self):
        base = profile_model()
        surface, targets = batch_fixture(24)
        reference = deepcopy(base)
        full = reference(surface, targets, consistency_weight=1.)
        (full['loss_per_asset'].sum() / 24).backward()
        expected = gradients(reference)
        for timed, batch in ((False, 24), (True, 24), (False, 12), (True, 12)):
            with self.subTest(timed=timed, batch=batch), patch.object(torch.cuda, 'synchronize'):
                model = deepcopy(base)
                forward = stages.TimedCapacityForward(model, torch.device('cpu')) if timed else model
                actual_losses = accumulate(forward, surface, targets, batch=batch)
                torch.testing.assert_close(actual_losses, full['loss_per_asset'], rtol=2e-5, atol=2e-6)
                assert_gradient_maps(gradients(model), expected, rtol=3e-4, atol=3e-7)
                self.assertTrue(all(value > 0 for value in stages.gradient_groups(model).values()))
        print('CPU_PROFILE_GRADIENT_COVERAGE ' + json.dumps({
            'trainable_tensors_checked': len(expected),
            'unused_parameters': [name for name, value in expected.items() if value is None],
            'positive_finite_groups': list(stages.gradient_groups(reference))}), flush=True)

    def test_two_rank_local24_gradient_average_matches_global48_mean(self):
        base = profile_model()
        surface, targets = batch_fixture(48)
        reference = deepcopy(base)
        output = reference(surface, targets, consistency_weight=1.)
        (output['loss_per_asset'].sum() / 48).backward()
        expected = gradients(reference)
        for batch in (24, 12):
            with self.subTest(batch=batch), patch.object(torch.cuda, 'synchronize'):
                ranks = []
                for rank in (0, 1):
                    model = deepcopy(base)
                    accumulate(stages.TimedCapacityForward(model, torch.device('cpu')),
                        sliced(surface, rank * 24, (rank + 1) * 24),
                        sliced(targets, rank * 24, (rank + 1) * 24), batch=batch)
                    ranks.append(gradients(model))
                averaged = {name: None if value is None else (ranks[0][name] + ranks[1][name]) / 2
                            for name, value in expected.items()}
                assert_gradient_maps(averaged, expected, rtol=3e-4, atol=3e-7)

    def test_unweighted_chunk_means_would_double_local24_gradients(self):
        base = profile_model()
        surface, targets = batch_fixture(24)
        reference, incorrect = deepcopy(base), deepcopy(base)
        accumulate(reference, surface, targets, batch=12)
        for start in (0, 12):
            output = incorrect(sliced(surface, start, start + 12),
                               sliced(targets, start, start + 12), consistency_weight=1.)
            output['loss'].backward()
        expected = {name: None if value is None else value * 2 for name, value in gradients(reference).items()}
        assert_gradient_maps(gradients(incorrect), expected)
        self.assertGreater(stages.gradient_groups(reference)['surface_tokenizer'], 0.)

    def test_temporal_probe_captures_actual_checkpointed_mha_arguments_and_gradient(self):
        model = profile_model()
        module = model.motion_encoder.blocks[-1].anchor_temporal.self_attn
        probe = stages.TemporalFailureProbe(module)
        self.addCleanup(probe.close)
        originals = []

        def observe(_module, args, kwargs, output):
            if probe.enabled:
                output[0].retain_grad()
                originals.append((args, dict(kwargs), output[0]))

        observer = module.register_forward_hook(observe, with_kwargs=True)
        self.addCleanup(observer.remove)
        surface, targets = batch_fixture(2)
        with probe.armed():
            result = model(surface, targets, consistency_weight=1.)
        self.assertFalse(probe.enabled)
        self.assertEqual(len(originals), 1)
        args, kwargs, original_output = originals[0]
        self.assertEqual(len(args), 3)
        self.assertIs(args[0], args[1])
        self.assertIs(args[0], args[2])
        self.assertEqual(kwargs, {'attn_mask': None, 'key_padding_mask': None,
                                  'need_weights': False, 'is_causal': False})
        self.assertEqual(tuple(probe.input.shape), (12, 3, fixtures.DIM))
        self.assertEqual(probe.input.data_ptr(), args[0].data_ptr())
        self.assertEqual(probe.output.data_ptr(), original_output.data_ptr())
        self.assertIsNone(probe.output_gradient)
        result['loss'].backward()
        torch.testing.assert_close(probe.output_gradient, original_output.grad, rtol=0., atol=0.)
        for value in (probe.input, probe.output, probe.output_gradient):
            self.assertEqual(value.device.type, 'cpu')
            self.assertFalse(value.requires_grad)
            self.assertIsNone(value.grad_fn)
            self.assertTrue(bool(torch.isfinite(value).all()))
            self.assertGreater(float(value.abs().sum()), 0.)

    def test_temporal_probe_preserves_outputs_all_parameter_and_input_gradients(self):
        for bf16 in (False, True):
            with self.subTest(cpu_bfloat16=bf16):
                ordinary = profile_model()
                profiled = deepcopy(ordinary)
                surface, targets = batch_fixture(2)
                for key in fixtures.SURFACE_KEYS:
                    surface[key].requires_grad_(True)
                observed_surface = {key: value.detach().clone().requires_grad_(value.requires_grad)
                                    for key, value in surface.items()}
                probe = stages.TemporalFailureProbe(
                    profiled.motion_encoder.blocks[-1].anchor_temporal.self_attn)
                self.addCleanup(probe.close)
                with torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
                    expected = ordinary(surface, targets, consistency_weight=1.)
                expected['loss'].backward()
                timed = stages.TimedCapacityForward(profiled, torch.device('cpu'))
                with patch.object(torch.cuda, 'synchronize'), probe.armed(), \
                        torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
                    actual = timed(observed_surface, targets, consistency_weight=1.)
                actual['loss'].backward()
                self.assertEqual(actual.keys(), expected.keys())
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key], rtol=0., atol=0., msg=key)
                assert_gradient_maps(gradients(profiled), gradients(ordinary))
                self.assertEqual(stages.gradient_groups(profiled), stages.gradient_groups(ordinary))
                for key in fixtures.SURFACE_KEYS:
                    self.assertIsNotNone(observed_surface[key].grad)
                    self.assertTrue(bool(torch.isfinite(observed_surface[key].grad).all()))
                    torch.testing.assert_close(observed_surface[key].grad, surface[key].grad,
                                               rtol=0., atol=0.)
                for name, value in profiled.state_dict().items():
                    torch.testing.assert_close(value, ordinary.state_dict()[name], rtol=0., atol=0.)
                self.assertIsNotNone(probe.output_gradient)
                probe.close()

    def test_temporal_probe_recomputation_does_not_overwrite_original_pointers(self):
        model = profile_model()
        module = model.motion_encoder.blocks[-1].anchor_temporal.self_attn
        probe = stages.TemporalFailureProbe(module)
        self.addCleanup(probe.close)
        calls = []

        def observe(_module, args, _kwargs, output):
            calls.append((probe.enabled, args[0].detach(), output[0].detach()))

        observer = module.register_forward_hook(observe, with_kwargs=True)
        self.addCleanup(observer.remove)
        surface, targets = batch_fixture(2)
        # Force a complete real recomputation so the installed forward hook is exercised again.
        with set_checkpoint_early_stop(False), probe.armed():
            output = model(surface, targets, consistency_weight=1.)
        original_input, original_output = probe.input, probe.output
        self.assertEqual(len(calls), 1)
        output['loss'].backward()
        self.assertGreaterEqual(len(calls), 2)
        self.assertTrue(calls[0][0])
        self.assertTrue(all(not enabled for enabled, _, _ in calls[1:]))
        self.assertTrue(all(value.data_ptr() != original_input.data_ptr() for _, value, _ in calls[1:]))
        self.assertIs(probe.input, original_input)
        self.assertIs(probe.output, original_output)
        self.assertEqual(probe.input.data_ptr(), calls[0][1].data_ptr())
        self.assertEqual(probe.output.data_ptr(), calls[0][2].data_ptr())
        self.assertIsNotNone(probe.output_gradient)
        self.assertTrue(bool(torch.isfinite(probe.output_gradient).all()))
        print('CPU_TEMPORAL_PROBE ' + json.dumps({
            'original_forward_calls': 1, 'recomputation_calls': len(calls) - 1,
            'original_input_and_output_pointers_preserved': True}), flush=True)

    def test_temporal_probe_unarmed_and_closed_hook_cannot_capture(self):
        model = profile_model()
        module = model.motion_encoder.blocks[-1].anchor_temporal.self_attn
        previous = set(module._forward_hooks)
        probe = stages.TemporalFailureProbe(module)
        self.addCleanup(probe.close)
        self.assertIn(probe.handle.id, module._forward_hooks)
        surface, targets = batch_fixture(2)
        model(surface, targets, consistency_weight=1.)['loss'].backward()
        self.assertIsNone(probe.input)
        self.assertIsNone(probe.output)
        self.assertIsNone(probe.output_gradient)
        probe.close()
        probe.close()
        self.assertEqual(set(module._forward_hooks), previous)
        self.assertNotIn(probe.handle.id, module._forward_hooks_with_kwargs)
        model.zero_grad(set_to_none=True)
        with probe.armed():
            output = model(surface, targets, consistency_weight=1.)
        output['loss'].backward()
        self.assertIsNone(probe.input)
        self.assertIsNone(probe.output)
        self.assertIsNone(probe.output_gradient)

    def test_temporal_probe_armed_resets_previous_capture_and_disarms_on_exception(self):
        model = profile_model()
        probe = stages.TemporalFailureProbe(model.motion_encoder.blocks[-1].anchor_temporal.self_attn)
        self.addCleanup(probe.close)
        surface, targets = batch_fixture(2)
        with probe.armed():
            output = model(surface, targets, consistency_weight=1.)
        output['loss'].backward()
        self.assertIsNotNone(probe.output_gradient)
        with self.assertRaisesRegex(ValueError, 'test interruption'):
            with probe.armed():
                self.assertTrue(probe.enabled)
                self.assertIsNone(probe.input)
                self.assertIsNone(probe.output)
                self.assertIsNone(probe.output_gradient)
                raise ValueError('test interruption')
        self.assertFalse(probe.enabled)
        model(surface, targets, consistency_weight=1.)
        self.assertIsNone(probe.input)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'CPU Gloo is unavailable')
    def test_two_process_gloo_timed_and_ordinary_chunk_gradients(self):
        with tempfile.TemporaryDirectory(prefix='id_xyz_profile_') as directory:
            rendezvous = (Path(directory) / 'rendezvous').resolve().as_uri()
            workers = mp.spawn(ddp_profile_worker, args=(rendezvous, directory), nprocs=2, join=False)
            try:
                deadline = time.monotonic() + 120
                while not workers.join(timeout=1):
                    if time.monotonic() >= deadline:
                        self.fail('CPU Gloo test exceeded its 120-second bound')
                reports = [json.loads(Path(directory, f'rank_{rank}.json').read_text()) for rank in (0, 1)]
                limitations = [row['skip_reason'] for row in reports if 'skip_reason' in row]
                if limitations:
                    self.skipTest('CPU Gloo initialization unavailable: ' + '; '.join(sorted(set(limitations))))
                self.assertEqual([row['rank'] for row in reports], [0, 1])
                self.assertTrue(all(row['passed'] and len(row['variants']) == 8 for row in reports))
                self.assertEqual(reports[0]['trainable_tensors_checked'], reports[1]['trainable_tensors_checked'])
                print('CPU_PROFILE_DDP ' + json.dumps({
                    'ranks': 2, 'variants_per_rank': 8,
                    'trainable_tensors_checked': reports[0]['trainable_tensors_checked'],
                    'unused_parameters': reports[0]['unused_parameters']}), flush=True)
            finally:
                for process in workers.processes:
                    if process.is_alive():
                        process.terminate()
                for process in workers.processes:
                    process.join(timeout=5)


if __name__ == '__main__':
    unittest.main()
