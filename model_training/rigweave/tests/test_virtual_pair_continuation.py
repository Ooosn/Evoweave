from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'model_training/rigweave/scripts'))
sys.path.insert(0, str(ROOT / 'model_training/rigweave/src'))
import virtual_pair_runtime as runtime
from virtual_pair_loading import OrderedAssetPrefetch


def configuration():
    config = json.loads((ROOT / 'model_training/experiments/virtual_query_matched_5k_20261025_v1.json').read_text())
    config['runtime'].update(cpu_slots=16, execution_cpu_budget=16, data_workers_per_rank=6,
                             prefetch_microbatches=2)
    config['execution'] = {'stop_assets_per_variant': 20000, 'evaluation_assets': [20000]}
    return config


class PrefetchTests(unittest.TestCase):
    def test_out_of_order_workers_preserve_order_and_bounded_queue(self):
        seen, lock = [], threading.Lock()
        class Dataset:
            def __getitem__(self, request):
                time.sleep(.04 if request == 0 else .001)
                with lock:
                    seen.append(request)
                return {'value': torch.tensor([request])}
        with OrderedAssetPrefetch(Dataset(), [[0, 1], [2, 3], [4, 5]], workers=3) as loader:
            time.sleep(.02)
            self.assertNotIn(4, seen)
            first, timing = loader.take([0, 1])
            self.assertLess(seen.index(1), seen.index(0))
            self.assertEqual([int(first[x]['value'][0]) for x in [0, 1]], [0, 1])
            self.assertGreaterEqual(timing['cpu_data_wait_seconds'], 0)
            with self.assertRaises(RuntimeError):
                loader.take([3, 2])
            self.assertEqual(int(loader.take([2, 3])[0][2]['value'][0]), 2)
            loader.take([4, 5])
            with self.assertRaises(RuntimeError):
                loader.take([6])

    def test_worker_failure_is_not_hidden(self):
        class Dataset:
            def __getitem__(self, request):
                raise ValueError('broken asset')
        with OrderedAssetPrefetch(Dataset(), [[0]], workers=2) as loader:
            with self.assertRaisesRegex(ValueError, 'broken asset'):
                loader.take([0])
        self.assertFalse(any(t.name.startswith('pair-data') for t in threading.enumerate()))

    def test_prefetch_never_advances_consumed_sampler_and_resume_replays(self):
        config = configuration()
        stream = runtime.make_stream(config, 0, 2)
        state = stream.state_dict()
        state['cursor'] = 2500
        stream = runtime.make_stream(config, 0, 2, state)
        original = deepcopy(stream.state_dict())
        class Dataset:
            def __getitem__(self, request):
                return tuple(request)
        with OrderedAssetPrefetch(Dataset(), runtime.planned_microbatches(config, stream, 10000), workers=3) as loader:
            self.assertEqual(stream.state_dict(), original)
            goal, batches = runtime.next_update_batches(stream, 5000, 2)
            for requests in batches:
                loaded, _ = loader.take(requests)
                self.assertEqual([loaded[r] for r in requests], [tuple(r) for r in requests])
            consumed = deepcopy(stream.state_dict())
        self.assertEqual(stream.cursor * 2, 5000 + goal)
        resumed = runtime.make_stream(config, 0, 2, consumed)
        self.assertEqual(runtime.next_update_batches(stream, 5000 + goal, 2),
                         runtime.next_update_batches(resumed, 5000 + goal, 2))


class ContinuationTests(unittest.TestCase):
    def test_CPU_audit_exercises_two_full_prefetch_queues_and_reserves_RAM(self):
        config = configuration()
        groups = {case['stress_group']: list(range(6)) for case in config['capacity']['cases']}
        class Dataset:
            def __getitem__(self, request):
                return {'request': torch.tensor(tuple(request))}
        resource = SimpleNamespace(RUSAGE_SELF=0,
            getrusage=lambda _: SimpleNamespace(ru_maxrss=4 * 1024**2))
        with patch.object(runtime, 'fixed_dataset', return_value=Dataset()), \
                patch.dict(sys.modules, {'resource': resource}):
            report = runtime.audit_loader(config, {'train': []}, groups)
            self.assertTrue(report['passed'])
            self.assertEqual(len(report['cases']), 6)
            self.assertTrue(all(case['two_rank_full_microbatch_queue_exercised'] for case in report['cases']))
            self.assertTrue(all(case['checked_assets'] == 6 for case in report['cases']))
            resource.getrusage = lambda _: SimpleNamespace(ru_maxrss=49 * 1024**2)
            with self.assertRaisesRegex(RuntimeError, 'insufficient RAM'):
                runtime.audit_loader(config, {'train': []}, groups)

    def test_evaluation_requires_hashed_rows_checkpoint_and_matched_GT(self):
        config = configuration()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            selection = output / 'selection.json'
            runtime.write_json(selection, {'valid': [{'asset_id': str(i)} for i in range(128)]})
            config['evaluation']['selection'] = {'path': str(selection), 'sha256': runtime.sha256(selection)}
            for variant in runtime.VARIANTS:
                root = output / variant / 'evaluation_020000'
                digest = ('a' if variant == 'virtual_on' else 'b') * 64
                runtime.write_json(output / variant / 'checkpoint_020000.json', {
                    'asset_exposures': 20000, 'sha256': digest})
                artifacts = []
                for rank in range(2):
                    rows = [{'asset_id': str(i), 'frame_index': f, 'raw_frame_index': f + 9,
                        'gt_joint_count': 52, 'original_GT_sha256': str(i), 'normalization_sha256': 'same',
                        'target_tokens': [1, 2], 'original_edge_f1': .6 if variant == 'virtual_on' else .4,
                        'success': True, 'exact_count': True} for i in range(rank, 128, 2) for f in range(8)]
                    path = root / 'attempt_002' / f'rank_{rank}' / 'scored_rows.json'
                    runtime.write_json(path, rows)
                    artifacts.append({'rank': rank, 'path': str(path), 'sha256': runtime.sha256(path)})
                runtime.write_json(root / 'summary.json', {'complete': True, 'samples': 20000,
                    'variant': variant, 'contract_sha256': runtime.contract_sha(config),
                    'source_sha256': runtime.source_hashes(), 'selection_sha256': runtime.sha256(selection),
                    'checkpoint_sha256': digest, 'row_artifacts': artifacts})
                self.assertTrue(runtime.evaluation_complete(output, config, variant, 20000))
            compared = runtime.compare_evaluations(output, config, 20000)
            self.assertAlmostEqual(compared['groups']['all']['Vplus_minus_Vminus_original_edge_f1'], .2)
            summary_path = output / 'virtual_off/evaluation_020000/summary.json'
            summary = json.loads(summary_path.read_text())
            row_path = Path(summary['row_artifacts'][0]['path'])
            rows = json.loads(row_path.read_text())
            rows[0]['original_GT_sha256'] = 'changed'
            runtime.write_json(row_path, rows)
            self.assertFalse(runtime.evaluation_complete(output, config, 'virtual_off', 20000))
            summary['row_artifacts'][0]['sha256'] = runtime.sha256(row_path)
            runtime.write_json(summary_path, summary)
            with self.assertRaisesRegex(RuntimeError, 'input/target differs'):
                runtime.compare_evaluations(output, config, 20000)
            summary['checkpoint_sha256'] = 'wrong'
            runtime.write_json(summary_path, summary)
            self.assertFalse(runtime.evaluation_complete(output, config, 'virtual_off', 20000))

    def test_operational_20k_preserves_contract_plan_and_lr(self):
        config = configuration()
        original = deepcopy(config)
        original.pop('execution')
        runtime.validate_config(config)
        self.assertEqual(runtime.contract_sha(config), runtime.contract_sha(original))
        self.assertEqual(runtime.make_stream(config, 0, 2).plan_sha256,
                         runtime.make_stream(original, 0, 2).plan_sha256)
        optimizer = runtime.optimizer_for(torch.nn.Linear(1, 1), config)
        for samples in (5000, 15000, 20000):
            self.assertEqual(runtime.set_lr(optimizer, samples, config), runtime.set_lr(optimizer, samples, original))
        config['runtime']['data_workers_per_rank'] = 8
        with self.assertRaises(ValueError):
            runtime.validate_config(config)

    def test_source_transition_is_exact_and_purpose_scoped(self):
        previous, current = {'a': 'old'}, {'a': 'new', 'b': 'fixed'}
        state = {'active_operation': {'source_transition': {
            'accepted': True, 'allowed_uses': ['capacity', 'checkpoint'],
            'previous_source_sha256': previous, 'current_source_sha256': current}}}
        with patch.object(runtime, 'source_hashes', return_value=current), patch.object(Path, 'read_text', return_value=json.dumps(state)):
            self.assertTrue(runtime.compatible_source(previous, 'checkpoint'))
            self.assertTrue(runtime.compatible_source(previous, 'capacity'))
            self.assertFalse(runtime.compatible_source(previous, 'CPU_audit'))
            self.assertFalse(runtime.compatible_source({'a': 'other'}, 'checkpoint'))
            state['active_operation']['source_transition']['current_source_sha256']['b'] = 'changed'
            # Separate object: an unreviewed current source cannot inherit the transition.
        with patch.object(runtime, 'source_hashes', return_value={'a': 'new', 'b': 'drift'}), patch.object(Path, 'read_text', return_value=json.dumps(state)):
            self.assertFalse(runtime.compatible_source(previous, 'checkpoint'))

    def test_training_stops_both_at_20k_without_resetting_on_arm(self):
        config = configuration()
        config['runtime']['data_workers_per_rank'] = 0
        specs = {'virtual_on': {'asset_exposures': 5000}, 'virtual_off': None}
        evaluations, updates, restored = set(), [], []
        def restore(model, optimizer, config, spec, variant, rank, world):
            restored.append((variant, spec['asset_exposures']))
            stream = runtime.make_stream(config, rank, world)
            state = stream.state_dict()
            state['cursor'] = spec['asset_exposures'] // world
            return runtime.make_stream(config, rank, world, state), {}, {}, spec['asset_exposures'], 0, spec['asset_exposures'] // 5000 * 105
        def update(model, wrapped, optimizer, dataset, batches, tokenizer, codec, config, variant, samples, device, world, **kwargs):
            size = sum(map(len, batches)) * world
            updates.append((variant, samples, size))
            return {'assets_in_update': size, 'frames_in_update': size * batches[0][0].frames}
        def save(model, optimizer, stream, config, output, variant, initial, source, samples, frames, step, rank, world, device):
            self.assertEqual(stream.cursor * world, samples)
            specs[variant] = {'asset_exposures': samples}
            return specs[variant]
        def evaluate(model, tokenizer, codec, config, entries, output, variant, samples, device, rank, world):
            evaluations.add((variant, samples))
        approved = {'initialization': dict.fromkeys(runtime.VARIANTS, {}),
                    'frame_policy': config['training']['frame_policy'], 'cases': config['capacity']['cases']}
        mocks = {'reviewed': lambda *args: approved, 'fixed_dataset': lambda *a, **k: None,
            'latest_checkpoint': lambda o, c, v: specs[v], 'shared_deadline_due': lambda *a: False,
            'build_model': lambda *a: (torch.nn.Linear(1, 1), None, None, {}),
            'wrap': lambda model, *a: model, 'restore_checkpoint': restore, 'apply_rng': lambda *a: None,
            'update': update, 'save_checkpoint': save, 'evaluate': evaluate, 'compare_evaluations': lambda *a: None,
            'evaluation_complete': lambda o, c, v, s: (v, s) in evaluations, 'synchronize': lambda *a: None}
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            for name, function in mocks.items():
                stack.enter_context(patch.object(runtime, name, side_effect=function))
            stack.enter_context(patch('builtins.print'))
            result = runtime.train_pair(config, {'train': []}, Path(temporary), 'test', torch.device('cpu'), 0, 2)
        self.assertEqual(updates[0][:2], ('virtual_off', 0))
        self.assertIn(('virtual_on', 5000), restored)
        self.assertEqual(min(s for v, s, n in updates if v == 'virtual_on'), 5000)
        self.assertEqual(result['progress'], dict.fromkeys(runtime.VARIANTS, 20000))
        self.assertEqual(evaluations, {(v, 20000) for v in runtime.VARIANTS})
        self.assertTrue(result['operation_complete'])
        self.assertFalse(result['complete'])
        self.assertFalse(result['allocation_handoff_required'])
        self.assertEqual(result['target_per_variant'], 100000)


if __name__ == '__main__':
    unittest.main()
