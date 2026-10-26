from contextlib import ExitStack, nullcontext
from copy import deepcopy
import hashlib
import json
import random
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import virtual_pair_runtime as runtime
from single_query_protocol import (DENSITY_SCOPE, THREE_SELECTOR, FRACTION_SELECTOR,
    select_density_target_frames, supervised_count)
from rigweave.dynamic_rig.frame_batch_sampler import FrameSampleRequest
from rigweave.dynamic_rig.virtual_path_ablation import VirtualPathAblationRig
from test_virtual_path_ablation import make_base, make_surface, make_targets
from test_two_query_runtime import two_configuration


def configuration(algorithm=THREE_SELECTOR):
    config = two_configuration()
    config['family'] = DENSITY_SCOPE
    config['pair']['target_selection']['algorithm'] = algorithm
    config['pair']['variants'] = ['three_query' if algorithm == THREE_SELECTOR else 'fraction_query']
    return config


class DensityProtocolTests(unittest.TestCase):
    def test_selectors_match_independent_recipe_and_preserve_global_rng(self):
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        requests = [FrameSampleRequest(i, t, (1 << 63) + i) for i, t in enumerate(range(2, 25))]
        for algorithm in (THREE_SELECTOR, FRACTION_SELECTOR):
            actual = select_density_target_frames(requests, 20261026, algorithm)
            expected = []
            for request in requests:
                text = f'{algorithm}|20261026|{request.item_index}|{request.sampling_seed}|{request.frames}'
                generator = random.Random(int.from_bytes(hashlib.sha256(text.encode('ascii')).digest(), 'big'))
                k = min(3, request.frames) if algorithm == THREE_SELECTOR else max(1, int(request.frames * (.1 + .2 * generator.random())))
                expected.append(sorted(generator.sample(list(range(request.frames)), k)))
            self.assertEqual(actual, expected)
            self.assertEqual(select_density_target_frames(requests[::-1], 20261026, algorithm), actual[::-1])
            self.assertEqual(select_density_target_frames(iter(requests), np.uint64(20261026), algorithm), actual)
        self.assertEqual(before[0], random.getstate())
        np.testing.assert_equal(before[1], np.random.get_state())
        torch.testing.assert_close(before[2], torch.get_rng_state(), rtol=0, atol=0)

    def test_short_sequences_variable_counts_and_invalid_requests(self):
        requests = [FrameSampleRequest(i, 24, i + 2000) for i in range(100)]
        counts = {len(s) for s in select_density_target_frames(requests, 0, FRACTION_SELECTOR)}
        self.assertEqual(counts, {2, 3, 4, 5, 6, 7})
        self.assertEqual(select_density_target_frames([FrameSampleRequest(0, 2, 1)], 0, THREE_SELECTOR), [[0, 1]])
        self.assertEqual(len(select_density_target_frames([FrameSampleRequest(0, 2, 1)], 0, FRACTION_SELECTOR)[0]), 1)
        for request in (FrameSampleRequest(-1, 2, 0), FrameSampleRequest(0, 1, 0),
                        FrameSampleRequest(0, 25, 0), FrameSampleRequest(0, 2, True)):
            with self.assertRaises(ValueError):
                select_density_target_frames([request], 0, THREE_SELECTOR)
        for seed in (-1, True, 1.2):
            with self.assertRaises(ValueError):
                select_density_target_frames([], seed, THREE_SELECTOR)
        with self.assertRaises(ValueError):
            supervised_count(configuration(), 2, 8)

    def test_exact_full20k_prefixes_and_unchanged_input_plan(self):
        for algorithm in (THREE_SELECTOR, FRACTION_SELECTOR):
            config = configuration(algorithm)
            runtime.validate_config(config)
            audit = runtime.audit_target_selection(config)
            frames = sum(r['input_frames'] for r in audit['ranks'])
            targets = sum(r['supervised_frames'] for r in audit['ranks'])
            self.assertEqual(frames, 263608)
            self.assertEqual(runtime.supervised_exposures(config, 20000, frames), targets)
            self.assertGreater(targets, 20000)
            self.assertLess(targets, 60000)
            with self.assertRaises(RuntimeError):
                runtime.supervised_exposures(config, 20000, frames + 1)
            report = runtime.sampling_plan_report(config, runtime.make_stream(config, 0, 2))
            self.assertEqual(report['input_frame_exposures'], 1294112)
            self.assertEqual(report['plan_sha256'], runtime.make_stream(two_configuration(), 0, 2).plan_sha256)


class RaggedLossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = VirtualPathAblationRig(make_base().double(), False).train()
        self.surface = {key: value.double() for key, value in make_surface().items()}
        self.target = make_targets()
        self.indices = torch.tensor([[2, 0, 0], [0, 1, 2]])
        self.mask = torch.tensor([[True, False, False], [True, True, True]])

    def test_ragged_loss_accuracy_and_gradients_are_equal_asset_weighted(self):
        frames, virtual = self.model.encode(self.surface)
        complete = self.model.forward_encoded(frames, virtual, self.target, auxiliary_weight=0.)
        selected = self.model.forward_encoded(frames, virtual, self.target, auxiliary_weight=0.,
            target_frame_indices=self.indices, target_frame_mask=self.mask)
        expected = torch.stack((complete['main_loss_per_frame'][0, 2], complete['main_loss_per_frame'][1].mean()))
        torch.testing.assert_close(selected['loss_per_asset'], expected, atol=1e-12, rtol=1e-12)
        expected_accuracy = (complete['main_accuracy_per_frame'][0, 2] + complete['main_accuracy_per_frame'][1].mean()) / 2
        torch.testing.assert_close(selected['main_accuracy'], expected_accuracy, atol=1e-12, rtol=1e-12)
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        actual_grad = torch.autograd.grad(selected['loss'], parameters, retain_graph=True, allow_unused=True)
        expected_grad = torch.autograd.grad(expected.mean(), parameters, allow_unused=True)
        for actual, expected in zip(actual_grad, expected_grad):
            if expected is None:
                self.assertIsNone(actual)
            else:
                torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-6)
        self.assertEqual(float(selected['main_loss_per_frame'][0, 1:].detach().sum()), 0.)

    def test_padding_never_enters_decoder_and_full_input_is_kept(self):
        self.target['main_input_ids'][0, :2] = 1000
        with patch.object(self.model, 'main_logits', wraps=self.model.main_logits) as spy, \
                patch.object(self.model, 'encode', wraps=self.model.encode) as encode:
            self.model(self.surface, self.target, auxiliary_weight=0.,
                target_frame_indices=self.indices, target_frame_mask=self.mask)['loss'].backward()
        self.assertIs(encode.call_args.args[0], self.surface)
        self.assertEqual([call.args[0].shape[0] for call in spy.call_args_list], [2, 1, 1])

    def test_mask_validation_and_uniform_mask_agree_with_old_api(self):
        expected = self.model(self.surface, self.target, auxiliary_weight=0., target_frame_indices=self.indices[:, :1])
        actual = self.model(self.surface, self.target, auxiliary_weight=0.,
            target_frame_indices=self.indices[:, :1], target_frame_mask=torch.ones(2, 1, dtype=torch.bool))
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], atol=1e-12, rtol=1e-12)
        for invalid in (torch.ones(2, 3), torch.ones(2, 2, dtype=torch.bool),
                        torch.zeros(2, 3, dtype=torch.bool),
                        torch.tensor([[True, False, True], [True, True, True]]),
                        torch.tensor([[True, False, False], [True, True, False]])):
            with self.assertRaisesRegex(ValueError, 'target_frame_mask'):
                self.model(self.surface, self.target, auxiliary_weight=0.,
                    target_frame_indices=self.indices, target_frame_mask=invalid)
        with self.assertRaisesRegex(ValueError, 'target_frame_mask'):
            self.model(self.surface, self.target, auxiliary_weight=0., target_frame_mask=self.mask)


class DensityRuntimeTests(unittest.TestCase):
    def test_variable_count_checkpoint_roundtrip(self):
        config = configuration(FRACTION_SELECTOR)
        frames, count = runtime.density_exposure_plan(json.dumps(config, sort_keys=True))[5000]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'local'
            output.mkdir()
            config['artifacts']['durable_root'] = str(Path(temporary) / 'durable')
            model = torch.nn.Linear(2, 1)
            optimizer = runtime.optimizer_for(model, config)
            model(torch.ones(2, 2)).sum().backward()
            optimizer.step()
            stream = runtime.make_stream(config, 0, 1)
            state = stream.state_dict()
            state['cursor'] = 5000
            stream = runtime.make_stream(config, 0, 1, state)
            rng = {'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.tensor([], dtype=torch.uint8),
                   'numpy_rng': np.random.get_state(), 'python_rng': random.getstate()}
            with patch.object(runtime, 'random_state', return_value=rng):
                spec = runtime.save_checkpoint(model, optimizer, stream, config, output, 'fraction_query',
                    {}, 'test', 5000, frames, 105, 0, 1, torch.device('cpu'))
            self.assertEqual(spec['supervised_frame_exposures'], count)
            target = torch.nn.Linear(2, 1)
            restored = runtime.restore_checkpoint(target, runtime.optimizer_for(target, config), config,
                                                   spec, 'fraction_query', 0, 1)
            self.assertEqual(restored[3:], (5000, frames, 105))
            self.assertEqual(runtime.parameter_fingerprint(model), runtime.parameter_fingerprint(target))

    def test_update_records_only_real_selected_targets(self):
        config = configuration(FRACTION_SELECTOR)
        requests = [FrameSampleRequest(i, 24, 2000 + i) for i in range(4)]
        target = {'main_input_ids': torch.arange(4 * 24 * 4).reshape(4, 24, 4)}
        raw = [list(range(24)) for _ in requests]
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.))

            def forward(self, surface, targets, auxiliary_weight, target_frame_indices, target_frame_mask):
                selected = targets['main_input_ids'][torch.arange(4)[:, None], target_frame_indices]
                losses = self.weight * selected.float().mean(-1) * target_frame_mask
                per_asset = losses.sum(1) / target_frame_mask.sum(1)
                return {'main_loss_per_frame': losses, 'loss_per_asset': per_asset,
                        'loss': per_asset.mean(), 'main_loss': per_asset.mean(), 'main_accuracy': per_asset.mean() * 0}
        model = FakeModel()
        with ExitStack() as stack:
            stack.enter_context(patch.object(runtime, 'prepare', return_value=(None, target, None)))
            stack.enter_context(patch.object(runtime, 'record_samples', return_value={'selected_raw_frames': raw}))
            stack.enter_context(patch.object(torch, 'autocast', return_value=nullcontext()))
            for name in ('reset_peak_memory_stats', 'synchronize'):
                stack.enter_context(patch.object(torch.cuda, name))
            for name in ('max_memory_allocated', 'max_memory_reserved'):
                stack.enter_context(patch.object(torch.cuda, name, return_value=0))
            stack.enter_context(patch.object(torch.cuda, 'get_device_properties', return_value=SimpleNamespace(total_memory=100)))
            result = runtime.update(model, model, runtime.optimizer_for(model, config), None, [requests],
                                    None, None, config, 'fraction_query', 0, torch.device('cpu'), 1)
        selected = runtime.target_selection(config, requests)
        record = result['microbatches'][0]
        self.assertEqual(record['target_frame_indices'], selected)
        self.assertEqual(record['selected_target_raw_frames'], selected)
        self.assertEqual(record['decoder_targets_per_asset'], list(map(len, selected)))
        self.assertEqual(result['supervised_frames_in_update'], sum(map(len, selected)))
        values = torch.cat([target['main_input_ids'][i, slots] for i, slots in enumerate(selected)])
        self.assertEqual(record['selected_target_sha256'], hashlib.sha256(values.numpy().tobytes()).hexdigest())

    def test_inherited_evidence_has_explicit_provenance_and_rejects_capacity_drift(self):
        config = configuration()
        config['execution']['inherited_capacity'] = {'path': '/capacity', 'sha256': 'abc'}
        original = {'family': 'old'}
        audited = {'initialization': {'virtual_off': {'variant': 'virtual_off', 'weights': 'fixed'}}}
        measured = {'complete': True, 'passed': True, 'profile_weights_discarded': True,
                    'contract_sha256': 'contract', 'frame_policy': config['training']['frame_policy'],
                    'cases': config['capacity']['cases'], 'source_sha256': {'common.py': 'same'}}
        with patch.object(runtime, 'reference_control', return_value=(audited, {})), \
                patch.object(runtime, 'pinned_json', side_effect=lambda spec: measured if spec['path'] == '/capacity' else original), \
                patch.object(runtime, 'contract_sha', return_value='contract'), \
                patch.object(runtime, 'source_hashes', return_value={'common.py': 'same'}):
            accepted, capacity = runtime.training_evidence(config)
            self.assertTrue(accepted['inherited'])
            self.assertEqual(accepted['initialization']['three_query']['weights'], 'fixed')
            self.assertIs(capacity, measured)
            measured['source_sha256']['common.py'] = 'drift'
            with self.assertRaisesRegex(RuntimeError, 'common source'):
                runtime.training_evidence(config)


if __name__ == '__main__':
    unittest.main()
