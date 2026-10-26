from copy import deepcopy
from contextlib import ExitStack, nullcontext
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import test_single_query_runtime as single_tests
from test_single_query_runtime import configuration, ROOT, runtime, guard
from rigweave.dynamic_rig.frame_batch_sampler import FrameSampleRequest


def two_configuration():
    return json.loads((ROOT / 'model_training/experiments/real_query_two_target_5k_20261026_v1.json').read_text())


class TwoRuntimeTests(unittest.TestCase):
    def test_update_records_two_real_targets_and_both_counters(self):
        config = two_configuration()
        requests = [FrameSampleRequest(i, 3, (1 << 60) + i) for i in range(2)]
        target = {'main_input_ids': torch.arange(24).reshape(2, 3, 4)}
        raw = [[10, 20, 30], [40, 50, 60]]
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.))

            def forward(self, surface, targets, auxiliary_weight, target_frame_indices):
                selected = targets['main_input_ids'][torch.arange(2)[:, None], target_frame_indices]
                losses = self.weight * selected.float().mean(-1)
                return {'main_loss_per_frame': losses, 'loss_per_asset': losses.mean(1),
                        'loss': losses.mean(), 'main_loss': losses.mean(), 'main_accuracy': losses.mean() * 0}
        model = FakeModel()
        with ExitStack() as stack:
            stack.enter_context(patch.object(runtime, 'prepare', return_value=(None, target, None)))
            stack.enter_context(patch.object(runtime, 'record_samples', return_value={'selected_raw_frames': raw}))
            stack.enter_context(patch.object(torch, 'autocast', return_value=nullcontext()))
            for name in ('reset_peak_memory_stats', 'synchronize'):
                stack.enter_context(patch.object(torch.cuda, name))
            for name in ('max_memory_allocated', 'max_memory_reserved'):
                stack.enter_context(patch.object(torch.cuda, name, return_value=0))
            stack.enter_context(patch.object(torch.cuda, 'get_device_properties',
                                            return_value=SimpleNamespace(total_memory=100)))
            result = runtime.update(model, model, runtime.optimizer_for(model, config), None, [requests],
                                    None, None, config, 'two_query', 0, torch.device('cpu'), 1)
        selected = runtime.target_selection(config, requests)
        recorded = result['microbatches'][0]
        self.assertEqual(recorded['target_frame_indices'], selected)
        self.assertEqual(recorded['selected_target_raw_frames'],
                         [[raw[i][j] for j in slots] for i, slots in enumerate(selected)])
        values = target['main_input_ids'][torch.arange(2)[:, None], torch.tensor(selected)]
        self.assertEqual(recorded['selected_target_sha256'], hashlib.sha256(values.numpy().tobytes()).hexdigest())
        self.assertEqual(recorded['decoder_targets_per_asset'], 2)
        self.assertEqual((result['assets_in_update'], result['frames_in_update'],
                          result['supervised_frames_in_update']), (2, 6, 4))

    def test_exact_two_policy_only_and_scheduler_authorization(self):
        config = two_configuration()
        runtime.validate_config(config)
        self.assertEqual(runtime.variants(config), ('two_query',))
        guard.scheduler_runtime(config)
        for field, value in [('target_selection', {'algorithm': runtime.SELECTOR, 'seed': 20261026}),
                             ('all_real_frames_supervised', True), ('variants', ['single_query'])]:
            wrong = deepcopy(config)
            wrong['pair'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                runtime.validate_config(wrong)
        wrong = deepcopy(config)
        wrong['training']['resume_root'] = '/old/checkpoints'
        with self.assertRaises(ValueError):
            runtime.validate_config(wrong)
        op = {'label': runtime.TWO_SCOPE, 'required_entrypoint': guard.ENTRYPOINT,
              'explicit_human_authorized': True}
        self.assertEqual(guard.operation({'active_operation': op}), op)

    def test_full20k_plan_and_separate_target_count(self):
        config = two_configuration()
        audit = runtime.audit_target_selection(config)
        self.assertTrue(audit['global_RNG_unchanged'])
        self.assertEqual(sum(r['input_frames'] for r in audit['ranks']), 263608)
        self.assertEqual(sum(r['asset_exposures'] for r in audit['ranks']), 20000)
        self.assertEqual(sum(r['supervised_frames'] for r in audit['ranks']), 40000)
        report = runtime.sampling_plan_report(config, runtime.make_stream(config, 0, 2))
        self.assertEqual(report['supervised_frame_exposures'], 200000)
        self.assertEqual(report['input_frame_exposures'], 1294112)
        self.assertEqual(report['decoder_target_policy'], runtime.TWO_SELECTOR)
        self.assertEqual(report['plan_sha256'], runtime.make_stream(configuration(), 0, 2).plan_sha256)
        self.assertNotEqual(runtime.contract_sha(config), runtime.contract_sha(configuration()))

    def test_checkpoint_restores_inputs_not_twice_assets(self):
        config = two_configuration()
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
                spec = runtime.save_checkpoint(model, optimizer, stream, config, output, 'two_query',
                    {}, 'test', 48, 336, 1, 0, 1, torch.device('cpu'))
            self.assertEqual(spec['input_frame_exposures'], 336)
            self.assertEqual(spec['supervised_frame_exposures'], 96)
            target = torch.nn.Linear(2, 1)
            restored = runtime.restore_checkpoint(target, runtime.optimizer_for(target, config), config,
                                                 spec, 'two_query', 0, 1)
            self.assertEqual(restored[3:], (48, 336, 1))
            self.assertEqual(runtime.parameter_fingerprint(model), runtime.parameter_fingerprint(target))

    def test_two_reference_and_paired_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config, initial, baseline = single_tests.SingleRuntimeTests().make_reference(directory)
            config.update(family=runtime.TWO_SCOPE)
            config['pair']['variants'] = [runtime.TWO_VARIANT]
            config['pair']['target_selection']['algorithm'] = runtime.TWO_SELECTOR
            initial['variant'] = runtime.TWO_VARIANT
            runtime.reference_control(config, initial)
            for name, score in [('two_query', .6), ('all_frames', .7)]:
                artifacts = []
                for rank in range(2):
                    path = directory / name / f'rows_{rank}.json'
                    rows = [{'asset_id': str(i), 'frame_index': f, 'raw_frame_index': f + 10,
                        'gt_joint_count': 52 if i < 45 else 34, 'original_GT_sha256': str(i),
                        'normalization_sha256': 'same', 'target_tokens': [1, 2],
                        'original_edge_f1': score, 'success': True, 'exact_count': True}
                        for i in range(rank, 128, 2) for f in range(8)]
                    runtime.write_json(path, rows)
                    artifacts.append({'rank': rank, 'path': str(path), 'sha256': runtime.sha256(path)})
                if name == 'all_frames':
                    baseline['row_artifacts'] = artifacts
                    path = Path(config['pair']['reference']['evaluation']['path'])
                    runtime.write_json(path, baseline)
                    config['pair']['reference']['evaluation']['sha256'] = runtime.sha256(path)
                else:
                    own_artifacts = artifacts
            runtime.write_json(directory / 'two_query/checkpoint_020000.json', {
                'asset_exposures': 20000, 'sha256': 'two-checkpoint'})
            runtime.write_json(directory / 'two_query/evaluation_020000/summary.json', {
                'complete': True, 'variant': 'two_query', 'samples': 20000,
                'contract_sha256': runtime.contract_sha(config), 'source_sha256': runtime.source_hashes(),
                'selection_sha256': config['evaluation']['selection']['sha256'],
                'checkpoint_sha256': 'two-checkpoint', 'row_artifacts': own_artifacts})
            result = runtime.compare_evaluations(directory, config, 20000)
            self.assertAlmostEqual(result['groups']['all']['two_minus_all_original_edge_f1'], -.1)
            self.assertEqual(set(result['arm_source_sha256']), {'two_query', 'all_frames'})
            self.assertEqual(result['groups']['non52']['assets'], 83)
            path = Path(baseline['row_artifacts'][0]['path'])
            rows = json.loads(path.read_text())
            rows[0]['normalization_sha256'] = 'drift'
            runtime.write_json(path, rows)
            baseline['row_artifacts'][0]['sha256'] = runtime.sha256(path)
            path = Path(config['pair']['reference']['evaluation']['path'])
            runtime.write_json(path, baseline)
            config['pair']['reference']['evaluation']['sha256'] = runtime.sha256(path)
            own_path = directory / 'two_query/evaluation_020000/summary.json'
            own_summary = json.loads(own_path.read_text())
            own_summary['contract_sha256'] = runtime.contract_sha(config)
            runtime.write_json(own_path, own_summary)
            with self.assertRaisesRegex(RuntimeError, 'input/target differs'):
                runtime.compare_evaluations(directory, config, 20000)


if __name__ == '__main__':
    unittest.main()
