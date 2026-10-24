"""Bounded CPU continuation tests; never read the production checkpoint."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import ast
import inspect
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('RIGWEAVE_DISABLE_OPEN3D', '1')
os.environ.setdefault('RIGWEAVE_DISABLE_LIGHTNING_IMPORT', '1')
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT.parent / 'analysis')]
import id_xyz_resume as resume
import id_xyz_runtime as runtime
import id_xyz_stages as stages


def recipe(target):
    return json.loads((ROOT.parent / 'experiments' / f'id_xyz_consistency_{target}_v1.json').read_text())


class TinyGroups(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ('surface_tokenizer', 'motion_encoder', 'xyz_decoder'):
            setattr(self, name, nn.Linear(2, 2))
        with torch.no_grad():
            for number, parameter in enumerate(self.parameters()):
                parameter.copy_(torch.linspace(.03, .15, parameter.numel()).reshape_as(parameter) + number * .01)

    def forward(self, value):
        return self.xyz_decoder(torch.tanh(self.motion_encoder(torch.tanh(self.surface_tokenizer(value)))))


def update(model, optimizer, samples, config):
    runtime.set_learning_rate(optimizer, samples, config)
    optimizer.zero_grad(set_to_none=True)
    value = torch.tensor([[.3, -.5], [-.7, .2]])
    target = torch.tensor([[-.1, .4], [.2, -.3]])
    (model(value) - target).square().mean().backward()
    optimizer.step()


class IdXYZResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.no_cuda = patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CPU tests may not initialize CUDA'))
        cls.no_cuda.start()
        cls.directory = tempfile.TemporaryDirectory()
        cls.parent = recipe(20000)
        cls.config = recipe(100000)
        cls.parent_path = Path(cls.directory.name) / 'parent.json'
        cls.parent_path.write_text(json.dumps(cls.parent), encoding='utf-8')
        cls.config['continuation']['parent_configuration'] = {
            'path': str(cls.parent_path), 'sha256': runtime.sha256(cls.parent_path)}
        cls.current_sources = runtime.source_hashes()
        cls.prior_sources = {key: value for key, value in cls.current_sources.items()
                             if key != 'scripts/id_xyz_resume.py'}
        for key in resume.CONTINUATION_SOURCE_CHANGES - {'scripts/id_xyz_resume.py'}:
            cls.prior_sources[key] = 'a' * 64
        cls.prior_receipt = {'passed': True, 'mode': 'audit', 'source_sha256': cls.prior_sources,
                             'paired_contract_sha256': runtime.paired_contract_hash(cls.parent)}
        cls.old_streams = [runtime.make_stream(cls.parent, rank, 2) for rank in (0, 1)]
        for stream in cls.old_streams:
            stream.cursor = 2500
        cls.frames = int(np.dot(cls.old_streams[0].update_sizes[:105].astype(np.int64),
                                cls.old_streams[0].frame_counts[:105]))
        cls.model = TinyGroups()
        cls.optimizer = runtime.optimizer_for(cls.model, cls.parent)
        starts = [0, *cls.old_streams[0].global_ends[:104].tolist()]
        for samples in starts:
            update(cls.model, cls.optimizer, samples, cls.parent)
        cls.saved = deepcopy({
            'model': cls.model.state_dict(), 'optimizer': cls.optimizer.state_dict(),
            'optimizer_parameter_names': resume.optimizer_names(cls.model, cls.optimizer),
            'asset_exposures': 5000, 'step': 105, 'input_frame_exposures': cls.frames,
            'supervised_frame_exposures': cls.frames,
            'training_contract': runtime.training_contract(cls.parent), 'configuration': cls.parent,
            'code_commit': cls.config['training']['resume_checkpoint']['producer_head'],
            'source_sha256': cls.prior_sources, 'initialization': {'initial_model_sha256': 'tiny-fixture'},
            'rank_states': [{'rank': rank, 'stream': stream.state_dict(),
                             'torch_rng': torch.get_rng_state().clone(),
                             'cuda_rng': torch.arange(16, dtype=torch.uint8),
                             'numpy_rng': np.random.RandomState(101 + rank).get_state(),
                             'python_rng': random.Random(201 + rank).getstate()}
                            for rank, stream in enumerate(cls.old_streams)]})

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()
        cls.no_cuda.stop()
        torch.set_num_threads(cls.old_threads)

    def restore(self, saved=None, *, config=None, rank=0, world=2):
        model = TinyGroups()
        optimizer = runtime.optimizer_for(model, self.config)
        with patch.object(stages, 'read_reviewed', return_value=deepcopy(self.prior_receipt)), \
                patch.object(torch, 'load', side_effect=AssertionError('use the tiny in-memory checkpoint')):
            result = resume.restore_checkpoint(model, optimizer, config or self.config,
                rank=rank, world=world, device=torch.device('cpu'),
                saved=deepcopy(self.saved if saved is None else saved))
        return model, optimizer, result

    def test_production_checkpoint_spec_matches_accepted_handoff(self):
        spec = recipe(100000)['training']['resume_checkpoint']
        self.assertEqual(spec, {
            'path': '/work/wangyy/id_xyz_consistency_20261024/outputs/consistency_20000_v1/checkpoint_005000.pt',
            'sha256': '8303f23a0b36820a1e14fa2ebee10b4325f78edf108ceca637f1804318050c90',
            'bytes': 7956024798, 'producer_head': '2fed5dc9b116a2ab2c3751d706b9ccf2e16f5f0c',
            'asset_exposures': 5000, 'step': 105})

    def test_parent_config_and_runtime_validation_accept_authorized_extension(self):
        self.assertEqual(resume.parent_configuration(self.config), self.parent)
        runtime.validate_config(self.config)

    def test_parent_configuration_is_hash_bound(self):
        changed = deepcopy(self.config)
        changed['continuation']['parent_configuration']['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            resume.parent_configuration(changed)

    def test_parent_model_data_objective_batch_and_resource_contracts_are_immutable(self):
        changes = [('architecture', 'dropout', .1), ('model_contract', 'auxiliary_loss', True),
                   ('codec', 'max_joints', 128), ('data', 'stream_seed', 7),
                   ('initialization', 'id_seed', 2), ('consistency', 'weight', 0),
                   ('optimization', 'peak_lr', 3e-5), ('evaluation', 'input_frames', 4),
                   ('final_evaluation', 'valid_assets', 64), ('surface_sampling', 'query_tokens', 512),
                   ('runtime', 'world_size', 1), ('runtime', 'gpu_memory_fraction_limit', .9),
                   ('runtime', 'tokenizer_config_sha256', '0' * 64),
                   ('training', 'effective_assets_per_update', 24),
                   ('training', 'frame_policy', {'2': 1})]
        for section, field, value in changes:
            changed = deepcopy(self.config)
            changed[section][field] = value
            with self.subTest(section=section, field=field), self.assertRaises(ValueError):
                resume.parent_configuration(changed)

    def test_only_reviewed_training_sources_can_change(self):
        allowed = {'scripts/id_xyz_runtime.py', 'scripts/id_xyz_stages.py',
                   'scripts/train_id_xyz_consistency.py', 'scripts/id_xyz_resume.py'}
        self.assertEqual(resume.CONTINUATION_SOURCE_CHANGES, allowed)
        with patch.object(runtime, 'source_hashes', return_value=self.current_sources):
            resume.validate_prior_receipt(self.config, self.prior_receipt)
        for name in self.prior_sources.keys() - allowed:
            changed = dict(self.current_sources, **{name: 'f' * 64})
            with self.subTest(source=name), patch.object(runtime, 'source_hashes', return_value=changed), \
                    self.assertRaises(ValueError):
                resume.validate_prior_receipt(self.config, self.prior_receipt)

    def test_source_additions_removals_and_wrong_parent_receipt_are_rejected(self):
        removed = dict(self.current_sources)
        removed.pop('scripts/virtual_fusion_normal_runtime.py')
        for changed in (removed, dict(self.current_sources, unreviewed='0' * 64)):
            with self.subTest(keys=list(changed)), patch.object(runtime, 'source_hashes', return_value=changed), \
                    self.assertRaises(ValueError):
                resume.validate_prior_receipt(self.config, self.prior_receipt)
        bad = dict(self.prior_receipt, paired_contract_sha256='0' * 64)
        with self.assertRaises(ValueError):
            resume.validate_prior_receipt(self.config, bad)

    def test_precision_remains_float32_parameters_with_bfloat16_training_autocast(self):
        tree = ast.parse(inspect.getsource(runtime.build_model))
        assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)]
        self.assertTrue(any(any(isinstance(target, ast.Attribute) and target.attr == 'torch_dtype'
                                for target in node.targets) and ast.unparse(node.value) == 'torch.float32'
                            for node in assignments))
        train = ast.parse(inspect.getsource(stages.train))
        contexts = [node for node in ast.walk(train) if isinstance(node, ast.Call)
                    and ast.unparse(node.func) == 'torch.autocast']
        self.assertTrue(contexts)
        for context in contexts:
            self.assertEqual(ast.literal_eval(context.args[0]), 'cuda')
            self.assertEqual({kw.arg: ast.unparse(kw.value) for kw in context.keywords}['dtype'], 'torch.bfloat16')

    def test_sampler_preserves_entire_20k_order_frames_sizes_and_next_requests(self):
        self.assertEqual(self.frames, 66928)
        for rank, old in enumerate(self.old_streams):
            restored = resume.restore_stream(self.config, old.state_dict(), samples=5000,
                frames=self.frames, step=105, rank=rank, world=2)
            self.assertEqual((len(old.update_sizes), len(restored.update_sizes)), (420, 2100))
            self.assertEqual((restored.cursor, restored.local_total), (2500, 50000))
            self.assertEqual(int(old.global_ends[104]), 5000)
            np.testing.assert_array_equal(old.order, restored.order[:20000])
            np.testing.assert_array_equal(old.frame_counts, restored.frame_counts[:420])
            np.testing.assert_array_equal(old.update_sizes, restored.update_sizes[:420])
            self.assertEqual(deepcopy(old).take(), deepcopy(restored).take())
            old_remaining, new_remaining = deepcopy(old), deepcopy(restored)
            while old_remaining.cursor < old_remaining.local_total:
                self.assertEqual(old_remaining.take(), new_remaining.take())
            self.assertEqual(new_remaining.cursor, 10000)

    def test_sampler_rejects_wrong_counter_cursor_rank_and_plan(self):
        base = dict(samples=5000, frames=self.frames, step=105, rank=0, world=2)
        for key, value in [('samples', 4992), ('samples', 5000.), ('step', 104), ('step', True),
                           ('frames', self.frames + 1), ('rank', 1), ('world', 1)]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                resume.restore_stream(self.config, self.old_streams[0].state_dict(), **dict(base, **{key: value}))
        for key, value in [('cursor', 2501), ('plan_sha256', '0' * 64)]:
            state = dict(self.old_streams[0].state_dict(), **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                resume.restore_stream(self.config, state, **base)

    def test_sampler_rejects_later_old_prefix_changes_even_when_next_request_matches(self):
        original = runtime.make_stream

        def changed_stream(config, rank, world):
            stream = original(config, rank, world)
            stream.frame_counts[300] = 2 if stream.frame_counts[300] != 2 else 3
            return stream

        with patch.object(runtime, 'make_stream', side_effect=changed_stream), self.assertRaises(ValueError):
            resume.restore_stream(self.config, self.old_streams[0].state_dict(), samples=5000,
                                  frames=self.frames, step=105, rank=0, world=2)

    def test_learning_rate_matches_old_recipe_to_handoff_and_has_continuous_cosine_tail(self):
        old_model = TinyGroups()
        old_optimizer = runtime.optimizer_for(old_model, self.parent)
        for sample in (0, 1, 1000, 1999, 2000, 4992, 4999, 5000):
            with self.subTest(sample=sample):
                self.assertEqual(resume.extended_learning_rate(sample, self.config),
                                 runtime.set_learning_rate(old_optimizer, sample, self.parent))
        initial_lr, initial_beta = resume.extended_learning_rate(5000, self.config)
        adjacent_lr, adjacent_beta = resume.extended_learning_rate(5000.001, self.config)
        self.assertAlmostEqual(initial_lr, adjacent_lr, places=14)
        self.assertAlmostEqual(initial_beta, adjacent_beta, places=14)
        self.assertGreater(initial_lr, 9e-5)
        self.assertLess(initial_lr, 1e-4)
        mid = resume.extended_learning_rate(52500, self.config)
        self.assertAlmostEqual(mid[0], (initial_lr + 2e-6) / 2, places=14)
        self.assertAlmostEqual(mid[1], (initial_beta + .95) / 2, places=14)
        finish = resume.extended_learning_rate(100000, self.config)
        self.assertAlmostEqual(finish[0], 2e-6, places=14)
        self.assertAlmostEqual(finish[1], .95, places=14)
        positions = list(range(5000, 100001, 1000))
        values = [resume.extended_learning_rate(position, self.config) for position in positions]
        self.assertTrue(all(right[0] <= left[0] and right[1] >= left[1] for left, right in zip(values, values[1:])))
        for sample in (-1, 100001, float('nan')):
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                resume.extended_learning_rate(sample, self.config)

    def test_exact_model_adam_restore_and_next_update_match_uninterrupted_cpu_run(self):
        model, optimizer, (stream, _, initial, report) = self.restore()
        self.assertEqual(resume.state_fingerprint(model.state_dict()), resume.state_fingerprint(self.model.state_dict()))
        self.assertEqual(resume.optimizer_fingerprint(model, optimizer), resume.optimizer_fingerprint(self.model, self.optimizer))
        self.assertEqual(initial, self.saved['initialization'])
        self.assertEqual((stream.cursor, report['step']), (2500, 105))
        baseline_model = deepcopy(self.model)
        baseline_optimizer = runtime.optimizer_for(baseline_model, self.parent)
        baseline_optimizer.load_state_dict(deepcopy(self.optimizer.state_dict()))
        update(model, optimizer, 5000, self.config)
        update(baseline_model, baseline_optimizer, 5000, self.config)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, baseline_model.state_dict()[name]), name)
        self.assertEqual(resume.optimizer_fingerprint(model, optimizer),
                         resume.optimizer_fingerprint(baseline_model, baseline_optimizer))

    def test_restore_report_is_json_stable_and_identical_for_both_ranks(self):
        _, _, result0 = self.restore(rank=0)
        _, _, result1 = self.restore(rank=1)
        report = result0[-1]
        self.assertEqual(report, json.loads(json.dumps(report, allow_nan=False)))
        self.assertEqual(report, result1[-1])
        self.assertIsInstance(report['next_learning_rate_and_beta1'], list)
        self.assertEqual([entry['rank'] for entry in report['rank_signatures']], [0, 1])

    def test_checkpoint_file_byte_count_and_digest_are_verified_before_load(self):
        path = Path(self.directory.name) / 'tiny.pt'
        torch.save(self.saved, path)
        config = deepcopy(self.config)
        spec = config['training']['resume_checkpoint']
        spec.update(path=str(path), bytes=path.stat().st_size, sha256=runtime.sha256(path))
        self.assertLess(spec['bytes'], 100000)
        for field, value in [('bytes', spec['bytes'] + 1), ('sha256', '0' * 64)]:
            bad = deepcopy(config)
            bad['training']['resume_checkpoint'][field] = value
            model = TinyGroups()
            with self.subTest(field=field), patch.object(torch, 'load') as loader, self.assertRaises(ValueError):
                resume.restore_checkpoint(model, runtime.optimizer_for(model, config), bad,
                    rank=0, world=2, device=torch.device('cpu'))
            loader.assert_not_called()
        model = TinyGroups()
        with patch.object(stages, 'read_reviewed', return_value=self.prior_receipt):
            result = resume.restore_checkpoint(model, runtime.optimizer_for(model, config), config,
                rank=0, world=2, device=torch.device('cpu'))
        self.assertEqual(result[-1]['asset_exposures'], 5000)

    def test_checkpoint_rejects_producer_counters_coverage_and_contract_changes(self):
        changes = [('code_commit', '0' * 40), ('source_sha256', {}), ('asset_exposures', 4992),
                   ('step', 104), ('input_frame_exposures', self.frames - 1),
                   ('supervised_frame_exposures', self.frames - 1), ('rank_states', self.saved['rank_states'][:1]),
                   ('training_contract', {})]
        for field, value in changes:
            bad = deepcopy(self.saved)
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.restore(bad)

    def test_checkpoint_rejects_swapped_rank_records_and_wrong_sampler_rank(self):
        for mutate in ('order', 'stream'):
            bad = deepcopy(self.saved)
            if mutate == 'order':
                bad['rank_states'].reverse()
            else:
                bad['rank_states'][0]['stream']['contract']['rank'] = 1
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.restore(bad)

    def test_checkpoint_rejects_invalid_requested_rank(self):
        for rank in (-1, 2, True):
            with self.subTest(rank=rank), self.assertRaises(ValueError):
                self.restore(rank=rank)

    def test_checkpoint_rejects_named_parameter_order_changes(self):
        for mutate in ('group_order', 'parameter_order', 'name'):
            bad = deepcopy(self.saved)
            names = bad['optimizer_parameter_names']
            if mutate == 'group_order':
                names.reverse()
            elif mutate == 'parameter_order':
                names[0]['parameters'].reverse()
            else:
                names[0]['parameters'][0] = 'wrong.parameter'
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.restore(bad)

    def test_checkpoint_rejects_nonfinite_model_parameters(self):
        bad = deepcopy(self.saved)
        next(iter(bad['model'].values())).flatten()[0] = float('nan')
        with self.assertRaises(ValueError):
            self.restore(bad)

    def test_checkpoint_rejects_missing_moment_wrong_step_shape_and_nonfinite_moments(self):
        for mutate in ('missing_moment', 'step', 'shape', 'nonfinite', 'empty'):
            bad = deepcopy(self.saved)
            states = bad['optimizer']['state']
            first = next(iter(states.values()))
            if mutate == 'missing_moment':
                first.pop('exp_avg')
            elif mutate == 'step':
                first['step'] = torch.tensor(104.)
            elif mutate == 'shape':
                first['exp_avg'] = torch.zeros(7)
            elif mutate == 'nonfinite':
                first['exp_avg_sq'].flatten()[0] = float('inf')
            else:
                states.clear()
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.restore(bad)

    def test_checkpoint_rejects_missing_entire_parameter_adam_state(self):
        bad = deepcopy(self.saved)
        states = bad['optimizer']['state']
        states.pop(next(iter(states)))
        with self.assertRaises(ValueError):
            self.restore(bad)

    def test_checkpoint_rejects_invalid_rng_tensor_dtype_or_rank(self):
        for key, value in [('torch_rng', torch.ones(5)), ('cuda_rng', torch.ones(2, 4, dtype=torch.uint8)), ('rank', 7)]:
            bad = deepcopy(self.saved)
            bad['rank_states'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.restore(bad)

    def test_checkpoint_rejects_malformed_rng_values_before_accepting_audit(self):
        for key, value in [('torch_rng', torch.zeros(0, dtype=torch.uint8)),
                           ('cuda_rng', torch.zeros(0, dtype=torch.uint8)),
                           ('numpy_rng', ('invalid', np.zeros(624, dtype=np.uint32), 0, 0, 0.)),
                           ('python_rng', (0, (), None))]:
            bad = deepcopy(self.saved)
            bad['rank_states'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.restore(bad)

    def test_apply_random_state_restores_cpu_numpy_python_and_routes_cuda_state_only(self):
        state = self.saved['rank_states'][0]
        original = (torch.get_rng_state(), np.random.get_state(), random.getstate())
        try:
            with patch.object(torch.cuda, 'set_rng_state') as cuda_restore:
                resume.apply_random_state(state, torch.device('cpu'))
            self.assertTrue(torch.equal(torch.get_rng_state(), state['torch_rng']))
            np.testing.assert_array_equal(np.random.get_state()[1], state['numpy_rng'][1])
            self.assertEqual(np.random.get_state()[2:], state['numpy_rng'][2:])
            self.assertEqual(random.getstate(), state['python_rng'])
            cuda_restore.assert_called_once()
            self.assertTrue(torch.equal(cuda_restore.call_args.args[0], state['cuda_rng']))
        finally:
            torch.set_rng_state(original[0])
            np.random.set_state(original[1])
            random.setstate(original[2])

    def test_handoff_deadline_is_timezone_aware_and_scoped_to_configured_run(self):
        deadline = datetime(2026, 10, 25, 8, 15, 28, tzinfo=timezone.utc)
        self.assertEqual(self.config['runtime']['stop_before_utc'], '2026-10-25T08:15:28Z')
        self.assertFalse(resume.allocation_handoff_due(self.parent, now=deadline + timedelta(days=1)))
        self.assertFalse(resume.allocation_handoff_due(self.config, now=deadline - timedelta(microseconds=1)))
        self.assertTrue(resume.allocation_handoff_due(self.config, now=deadline))
        self.assertTrue(resume.allocation_handoff_due(self.config, now=deadline + timedelta(seconds=1)))
        tokyo = timezone(timedelta(hours=9))
        self.assertTrue(resume.allocation_handoff_due(self.config, now=deadline.astimezone(tokyo)))
        config = deepcopy(self.config)
        config['runtime']['stop_before_utc'] = deadline.astimezone(tokyo).isoformat()
        self.assertTrue(resume.allocation_handoff_due(config, now=deadline))
        config['runtime']['stop_before_utc'] = deadline.replace(tzinfo=None).isoformat()
        with self.assertRaises(ValueError):
            resume.allocation_handoff_due(config, now=deadline)


if __name__ == '__main__':
    unittest.main()
