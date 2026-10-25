from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'model_training/rigweave/scripts'))
sys.path.insert(0, str(ROOT / 'model_training/rigweave/src'))
import virtual_pair_runtime as runtime
import virtual_pair_guard as guard


def configuration(single=True):
    name = 'real_query_single_target_5k_20261026_v1.json' if single else 'virtual_query_matched_5k_20261025_v2.json'
    return json.loads((ROOT / 'model_training/experiments' / name).read_text())


class SingleRuntimeTests(unittest.TestCase):
    def test_only_declared_single_target_policy_is_accepted(self):
        config = configuration()
        runtime.validate_config(config)
        self.assertEqual(runtime.variants(config), ('single_query',))
        changes = [('family', 'unknown'), ('pair.all_real_frames_supervised', True),
                   ('pair.variants', ['virtual_off']), ('pair.target_selection.seed', 0),
                   ('execution.stop_assets_per_variant', 25000), ('training.resume_root', '/old/all_frame')]
        for dotted, value in changes:
            wrong = deepcopy(config)
            target = wrong
            keys = dotted.split('.')
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = value
            with self.subTest(dotted=dotted), self.assertRaises(ValueError):
                runtime.validate_config(wrong)

    def test_single_target_selector_does_not_change_input_plan_or_lr(self):
        config, previous = configuration(), configuration(False)
        report = runtime.audit_target_selection(config)
        self.assertTrue(report['global_RNG_unchanged'])
        self.assertEqual(sum(r['asset_exposures'] for r in report['ranks']), 20000)
        self.assertEqual(sum(r['supervised_frames'] for r in report['ranks']), 20000)
        self.assertEqual(sum(r['input_frames'] for r in report['ranks']), 263608)
        self.assertEqual(runtime.make_stream(config, 0, 2).plan_sha256,
                         runtime.make_stream(previous, 0, 2).plan_sha256)
        optimizer = runtime.optimizer_for(torch.nn.Linear(1, 1), config)
        for count in (0, 5000, 10000, 15000, 19968):
            self.assertEqual(runtime.set_lr(optimizer, count, config), runtime.set_lr(optimizer, count, previous))
        changed = deepcopy(config)
        changed['pair']['target_selection']['seed'] += 1
        self.assertNotEqual(runtime.contract_sha(config), runtime.contract_sha(changed))

    def test_stage_family_cannot_cross_authorization(self):
        config = configuration()
        path = ROOT / 'model_training/experiments/real_query_single_target_5k_20261026_v1.json'
        args = SimpleNamespace(config=path, output=ROOT / 'unused_test', mode='audit')
        operation = {'label': runtime.SINGLE_SCOPE, 'explicit_human_authorized': True, 'complete': False,
            'required_entrypoint': guard.ENTRYPOINT, 'gpu_work_allowed': False,
            'runs': [{'mode': 'audit', 'config': str(path), 'output': str(args.output),
                      'config_sha256': runtime.sha256(path), 'approved': True}]}
        guard.validate_stage({'active_operation': operation}, config, args)
        operation['label'] = guard.SCOPE
        with self.assertRaisesRegex(RuntimeError, 'family differ'):
            guard.validate_stage({'active_operation': operation}, config, args)
        guard.scheduler_runtime(config)
        with self.assertRaises(RuntimeError):
            guard.validate_ssh_exception({'active_operation': operation}, config)

    def test_sampling_report_labels_inputs_and_targets_separately(self):
        config, previous = configuration(), configuration(False)
        single = runtime.sampling_plan_report(config, runtime.make_stream(config, 0, 2))
        all_frames = runtime.sampling_plan_report(previous, runtime.make_stream(previous, 0, 2))
        self.assertEqual(single['plan_sha256'], all_frames['plan_sha256'])
        self.assertEqual(single['input_frame_exposures'], 1294112)
        self.assertEqual(single['supervised_frame_exposures'], 100000)
        self.assertEqual(all_frames['supervised_frame_exposures'], 1294112)
        self.assertEqual(all_frames, runtime.make_stream(previous, 0, 2).plan_report())

    def test_checkpoint_restores_distinct_input_and_supervision_counts(self):
        config = configuration()
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
            state['cursor'] = 48
            stream = runtime.make_stream(config, 0, 1, state)
            rng = {'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.tensor([], dtype=torch.uint8),
                   'numpy_rng': runtime.np.random.get_state(), 'python_rng': runtime.random.getstate()}
            with patch.object(runtime, 'random_state', return_value=rng):
                spec = runtime.save_checkpoint(model, optimizer, stream, config, output, 'single_query',
                    {}, 'test', 48, 336, 1, 0, 1, torch.device('cpu'))
            self.assertEqual(spec['input_frame_exposures'], 336)
            self.assertEqual(spec['supervised_frame_exposures'], 48)
            target = torch.nn.Linear(2, 1)
            restored = runtime.restore_checkpoint(target, runtime.optimizer_for(target, config), config,
                                                 spec, 'single_query', 0, 1)
            self.assertEqual(restored[3:], (48, 336, 1))
            self.assertEqual(runtime.parameter_fingerprint(model), runtime.parameter_fingerprint(target))

    def make_reference(self, directory):
        previous, config = configuration(False), configuration()
        selection = directory / 'selection.json'
        runtime.write_json(selection, {'valid': [{'asset_id': str(i)} for i in range(128)]})
        previous['evaluation']['selection'] = config['evaluation']['selection'] = {
            'path': str(selection), 'sha256': runtime.sha256(selection)}
        sources = runtime.source_hashes()
        sources.pop('model_training/rigweave/scripts/single_query_protocol.py')
        sources['model_training/rigweave/scripts/virtual_pair_runtime.py'] = 'prior-runtime'
        initial = {'variant': 'virtual_off', 'wrapped_initial_sha256': 'same'}
        audit = {'complete': True, 'passed': True, 'cuda_initialized': False,
            'contract_sha256': runtime.contract_sha(previous), 'source_sha256': sources,
            'parallel_loader': {'passed': True}, 'initialization': {'virtual_off': initial}}
        summary = {'complete': True, 'variant': 'virtual_off', 'samples': 20000,
            'contract_sha256': audit['contract_sha256'], 'source_sha256': sources,
            'checkpoint_sha256': config['pair']['reference']['checkpoint_sha256'],
            'selection_sha256': runtime.sha256(selection), 'original_GT_scoring': True,
            'failed_views_scored_zero': True, 'oracle_not_used': True}
        for key, value in [('config', previous), ('CPU_audit', audit), ('evaluation', summary)]:
            path = directory / (key + '.json')
            runtime.write_json(path, value)
            config['pair']['reference'][key] = {'path': str(path), 'sha256': runtime.sha256(path)}
        return config, initial, summary

    def test_reference_rejects_recipe_initialization_and_data_source_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, initial, _ = self.make_reference(Path(temporary))
            initial['variant'] = 'single_query'
            runtime.reference_control(config, initial)
            for key, value in [('architecture.dropout', .1), ('optimization.peak_lr', .0002),
                               ('runtime.data_workers_per_rank', 5), ('training.frame_policy.24', 12)]:
                wrong = deepcopy(config)
                target = wrong
                parts = key.split('.')
                for part in parts[:-1]:
                    target = target[part]
                target[parts[-1]] = value
                with self.subTest(key=key), self.assertRaises(RuntimeError):
                    runtime.reference_control(wrong, initial)
            with self.assertRaisesRegex(RuntimeError, 'initialization differs'):
                runtime.reference_control(config, {**initial, 'wrapped_initial_sha256': 'different'})
            sources = runtime.source_hashes()
            sources['model_training/rigweave/src/rigweave/dynamic_rig/data.py'] = 'changed'
            with patch.object(runtime, 'source_hashes', return_value=sources), self.assertRaises(RuntimeError):
                runtime.reference_control(config, initial)

    def test_comparison_binds_reference_rows_and_reports_single_minus_all(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config, _, baseline = self.make_reference(directory)
            for name, score in [('single_query', .5), ('all_frames', .7)]:
                artifacts = []
                for rank in range(2):
                    path = directory / name / f'rows_{rank}.json'
                    rows = [{'asset_id': str(i), 'frame_index': f, 'raw_frame_index': f + 10,
                        'gt_joint_count': 52, 'original_GT_sha256': str(i), 'normalization_sha256': 'same',
                        'target_tokens': [1, 2], 'original_edge_f1': score,
                        'success': True, 'exact_count': True} for i in range(rank, 128, 2) for f in range(8)]
                    runtime.write_json(path, rows)
                    artifacts.append({'rank': rank, 'path': str(path), 'sha256': runtime.sha256(path)})
                if name == 'all_frames':
                    baseline['row_artifacts'] = artifacts
                    path = Path(config['pair']['reference']['evaluation']['path'])
                    runtime.write_json(path, baseline)
                    config['pair']['reference']['evaluation']['sha256'] = runtime.sha256(path)
                else:
                    own_artifacts = artifacts
            own = directory / 'single_query/evaluation_020000/summary.json'
            runtime.write_json(directory / 'single_query/checkpoint_020000.json', {
                'asset_exposures': 20000, 'sha256': 'single-checkpoint'})
            runtime.write_json(own, {'complete': True, 'variant': 'single_query', 'samples': 20000,
                'contract_sha256': runtime.contract_sha(config), 'source_sha256': runtime.source_hashes(),
                'selection_sha256': config['evaluation']['selection']['sha256'],
                'checkpoint_sha256': 'single-checkpoint', 'row_artifacts': own_artifacts})
            result = runtime.compare_evaluations(directory, config, 20000)
            self.assertAlmostEqual(result['groups']['all']['single_minus_all_original_edge_f1'], -.2)
            self.assertEqual(set(result['arm_source_sha256']), {'single_query', 'all_frames'})
            path = Path(baseline['row_artifacts'][0]['path'])
            rows = json.loads(path.read_text())
            rows[0]['original_GT_sha256'] = 'wrong-GT'
            runtime.write_json(path, rows)
            with self.assertRaisesRegex(RuntimeError, 'artifact binding differs'):
                runtime.compare_evaluations(directory, config, 20000)


if __name__ == '__main__':
    unittest.main()
