"""CPU control-flow checks for the bounded 64-asset comparison runner."""
import ast
from contextlib import ExitStack, nullcontext
import hashlib
import json
import math
from pathlib import Path
import sys
from types import ModuleType
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch._dynamo
from torch import nn


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / 'model_training/rigweave/scripts'
NAMESPACE = {
    'Path': Path, 'hashlib': hashlib, 'json': json, 'math': math,
    'np': np, 'sys': sys, 'time': __import__('time'), 'torch': torch,
}


def extract(path, names, namespace=NAMESPACE):
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise RuntimeError('missing actual comparison helper')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), namespace)


extract(SCRIPTS / 'probe_virtual_query_lambda.py',
        ['transfer', 'training_order', 'comparison_plan', 'parameter_fingerprint',
         'evaluate_groups', 'run_comparison', 'shared_gradient_measurement'])

SCHEDULE = {'Any': object, 'math': math}
extract(SCRIPTS / 'virtual_frame_runtime.py', ['exposure_learning_rate', 'set_exposure_lr'], SCHEDULE)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n', encoding='utf-8')


def sha256(path):
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


class TinyComparisonModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.35, dtype=torch.float64))

    def forward(self, surface, targets, *, auxiliary_weight):
        value = surface['value'].to(self.weight.dtype)
        target = targets['target'].to(self.weight.dtype)
        main = (self.weight * value - target).square().mean()
        auxiliary = self.weight.square()
        return {
            'loss': main + auxiliary_weight * auxiliary,
            'main_loss': main,
            'auxiliary_loss': auxiliary,
            'main_accuracy': value.mean().detach(),
            'auxiliary_accuracy': (2 * value).mean().detach(),
            'measurement_marker': value.mean().detach(),
        }


def item(asset_id, value):
    return ({'value': torch.tensor([value], dtype=torch.float64)},
            {'target': torch.tensor([0.25 + value / 11], dtype=torch.float64)},
            {'asset_id': [asset_id]})


def groups():
    train = [item(f'a{i}', i + 1) for i in range(64)]
    return {
        'train': train,
        'train_heldout': train[:8],
        'valid': [item(f'v{i}', 100 + i) for i in range(32)],
    }


def config(updates=3, batch=None, *, measurements=(), paired=False):
    actual_batch = 1 if batch is None else batch
    comparison = {
        'frames': 2,
        'max_updates': updates,
        'order_seed': 20261013,
        'gradient_measurement_updates': list(measurements),
        'evaluation_updates': [],
        'initial_evaluation': False,
        'generation_frame_indices': [0],
        'cache': {'sha256': 'fixed-cache-hash'},
    }
    if batch is not None:
        comparison['assets_per_update'] = batch
    probe = {'auxiliary_weight': 0.1}
    if paired:
        probe['paired_output'] = 'peer-output'
    return {
        'comparison': comparison,
        'probe': probe,
        'optimization': {
            'schedule': 'onecycle',
            'peak_lr': 0.01,
            'warmup_fraction': 0.1,
            'div_factor': 5.0,
            'final_div_factor': 10.0,
            'weight_decay': 0.0,
            'schedule_samples': updates * actual_batch,
        },
    }


def gradient_norms(model):
    values = [parameter.grad.detach().float() for parameter in model.parameters()
              if parameter.grad is not None]
    return {'all': sum(float(value.square().sum()) for value in values) ** .5}


def fake_evaluation(*args, **kwargs):
    return {'groups': {'valid': {'main_accuracy': 1.0}}, 'rows': []}


def run_once(cfg, root, *, seed=91, lr_calls=None, events=None, measurement=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / 'config.json'
    cfg['_path'] = str(config_path)
    write_json(config_path, cfg)
    output = root / 'output'
    output.mkdir()
    cache = groups()
    models = []

    def build(config_value, device):
        torch.manual_seed(seed)
        model = TinyComparisonModel()
        models.append(model)
        return model, object(), None, {'initialization': 'fake'}

    def schedule(optimizer, samples, config_value):
        if lr_calls is not None:
            lr_calls.append(samples)
        return SCHEDULE['set_exposure_lr'](optimizer, samples, config_value)

    def measure(model, losses, weight):
        if measurement is not None:
            measurement.append(float(losses['measurement_marker']))
        return {'virtual_slots': {'weighted_aux_to_main_ratio': 0.125}}

    fake_fusion = ModuleType('fusion_evaluation')

    def pair_ready(output_path, peer_path, initial, items, config_value):
        if events is not None:
            events.append(('pair_ready', len(items), initial['full_initial_state_sha256'], initial['code_commit']))

    fake_fusion.pair_ready = pair_ready

    with ExitStack() as stack:
        stack.enter_context(patch.dict(NAMESPACE, {
            'load_cache': lambda config_value: (cache, {'selection_groups': {}}),
            'build_model': build,
            'sha256': sha256,
            'write_json': write_json,
            'set_exposure_lr': schedule,
            'grad_norms': gradient_norms,
            'require_gradients': lambda *args: None,
            'evaluate_groups': fake_evaluation,
            'shared_gradient_measurement': measure,
        }))
        stack.enter_context(patch.object(torch, 'autocast', lambda *args, **kwargs: nullcontext()))
        stack.enter_context(patch.object(torch.cuda, 'max_memory_allocated', lambda *args, **kwargs: 0.0))
        if events is not None:
            original_adamw = torch.optim.AdamW

            def adamw(*args, **kwargs):
                optimizer = original_adamw(*args, **kwargs)
                original_step = optimizer.step

                def step(*step_args, **step_kwargs):
                    events.append(('optimizer_step',))
                    return original_step(*step_args, **step_kwargs)

                optimizer.step = step
                return optimizer

            stack.enter_context(patch.object(torch.optim, 'AdamW', adamw))
        if cfg['probe'].get('paired_output'):
            stack.enter_context(patch.dict(sys.modules, {'fusion_evaluation': fake_fusion}))
        result = NAMESPACE['run_comparison'](cfg, output, 'cpu', 'test-commit')
    return result, models[0], output


def read_trace(output):
    return [json.loads(line) for line in (Path(output) / 'trace.jsonl').read_text().splitlines()]


class ComparisonPlanTests(unittest.TestCase):
    def test_default_batch_one_preserves_legacy_trajectory_and_asset_id(self):
        with tempfile.TemporaryDirectory() as directory:
            first_cfg = config(updates=3)
            first_result, first_model, first_output = run_once(first_cfg, Path(directory) / 'first')
            second_cfg = config(updates=3, batch=1)
            second_result, second_model, second_output = run_once(second_cfg, Path(directory) / 'second')
            for name, value in first_model.state_dict().items():
                torch.testing.assert_close(value, second_model.state_dict()[name], rtol=0., atol=0.)
            self.assertEqual(first_result, second_result)
            first_trace, second_trace = read_trace(first_output), read_trace(second_output)
            self.assertEqual(len(first_trace), 3)
            for left, right in zip(first_trace, second_trace):
                for key in ('asset_id', 'asset_ids', 'asset_exposures', 'frame_exposures',
                            'loss', 'main_loss', 'auxiliary_loss', 'main_accuracy', 'auxiliary_accuracy'):
                    self.assertEqual(left[key], right[key])
                self.assertEqual(left['asset_ids'], [left['asset_id']])
            self.assertEqual(first_result['asset_exposures'], 3)
            self.assertEqual(first_result['frame_exposures'], 6)

    def test_gradient_accumulation_matches_explicit_asset_mean_before_adamw(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(updates=3, batch=2)
            result, model, output = run_once(cfg, Path(directory) / 'run', seed=17)
            baseline = TinyComparisonModel()
            optimizer = torch.optim.AdamW(baseline.parameters(), lr=0.01 / 5., betas=(.95, .999), weight_decay=0.)
            train = groups()['train']
            order = NAMESPACE['training_order'](64, 6, cfg['comparison']['order_seed'])
            for step in range(1, 4):
                start = (step - 1) * 2
                SCHEDULE['set_exposure_lr'](optimizer, start, cfg)
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for index in order[start:start + 2]:
                    surface, targets, _ = NAMESPACE['transfer'](train[index], 'cpu')
                    losses.append(baseline(surface, targets, auxiliary_weight=.1)['loss'])
                torch.stack(losses).mean().backward()
                torch.nn.utils.clip_grad_norm_(baseline.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
            torch.testing.assert_close(model.weight, baseline.weight, rtol=0., atol=0.)
            checkpoint = torch.load(Path(output) / 'checkpoint_comparison.pt', map_location='cpu', weights_only=False)
            self.assertEqual(checkpoint['budget']['actual_asset_exposures'], 6)
            self.assertEqual(result['asset_exposures'], 6)

    def test_batch_48_uses_actual_exposure_schedule_order_hash_metrics_and_exact_budget(self):
        calls = []
        measurements = []
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(updates=5, batch=48, measurements=(1, 3))
            result, _, output = run_once(cfg, Path(directory) / 'run', lr_calls=calls, measurement=measurements)
            expected_order = NAMESPACE['training_order'](64, 5 * 48, cfg['comparison']['order_seed'])
            expected_hash = hashlib.sha256(json.dumps(expected_order, separators=(',', ':')).encode()).hexdigest()
            order_record = json.loads((Path(output) / 'training_order.json').read_text())
            trace = read_trace(output)
            self.assertEqual(calls, [0, 48, 96, 144, 192])
            self.assertEqual(order_record['indices'], expected_order)
            self.assertEqual(order_record['sha256'], expected_hash)
            self.assertEqual(result['training_order_sha256'], expected_hash)
            self.assertEqual(len(order_record['indices']), 240)
            self.assertEqual(trace[-1]['asset_exposures'], 240)
            self.assertEqual(trace[-1]['frame_exposures'], 480)
            self.assertEqual(result['asset_exposures'], 240)
            self.assertEqual(result['actual_asset_exposures'], 240)
            self.assertEqual(result['frame_exposures'], 480)
            self.assertEqual(result['budget'], {'updates': 5, 'assets_per_update': 48,
                                                'actual_asset_exposures': 240,
                                                'frame_exposures': 480, 'frames_per_asset': 2})
            checkpoint = torch.load(Path(output) / 'checkpoint_comparison.pt', map_location='cpu', weights_only=False)
            self.assertEqual(checkpoint['asset_exposures'], 240)
            self.assertEqual(checkpoint['frame_exposures'], 480)
            self.assertEqual(checkpoint['step'], 5)
            for row_number, row in enumerate(trace):
                start = row_number * 48
                self.assertEqual(row['asset_ids'], [f'a{index}' for index in expected_order[start:start + 48]])
                values = [index + 1 for index in expected_order[start:start + 48]]
                self.assertAlmostEqual(row['main_accuracy'], sum(values) / 48)
                self.assertAlmostEqual(row['auxiliary_accuracy'], 2 * sum(values) / 48)
            self.assertEqual(measurements, [expected_order[0] + 1, expected_order[96] + 1])
            self.assertEqual(trace[0]['shared_gradient_measurement_scope'], 'first_microbatch_unscaled_loss')
            self.assertEqual(trace[2]['shared_gradient_measurement_scope'], 'first_microbatch_unscaled_loss')
            self.assertIsNone(trace[1]['shared_gradient_measurement_scope'])

    def test_invalid_batch_and_schedule_are_rejected(self):
        for batch in (0, -1, True, 1.5, 65):
            with self.subTest(batch=batch):
                cfg = config(updates=5, batch=batch)
                if isinstance(batch, int) and not isinstance(batch, bool):
                    cfg['optimization']['schedule_samples'] = 5 * batch
                with self.assertRaises(ValueError):
                    NAMESPACE['comparison_plan'](cfg, 64)
        cfg = config(updates=5, batch=48)
        cfg['optimization']['schedule_samples'] = 5
        with self.assertRaisesRegex(ValueError, 'schedule_samples'):
            NAMESPACE['comparison_plan'](cfg, 64)

    def test_pair_ready_is_after_initial_fingerprint_and_before_optimizer_step(self):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(updates=1, batch=1, paired=True)
            result, _, output = run_once(cfg, Path(directory) / 'run', events=events)
            self.assertEqual(events[0][0], 'pair_ready')
            self.assertEqual(events[1], ('optimizer_step',))
            self.assertEqual(events[0][1], 104)
            self.assertTrue(events[0][2])
            self.assertEqual(events[0][3], 'test-commit')
            self.assertTrue((Path(output) / 'initialization.json').is_file())
            self.assertEqual(result['asset_exposures'], 1)


class EvaluationPropagationTests(unittest.TestCase):
    def test_auxiliary_fields_are_propagated_per_group_with_shared_frame_selection(self):
        cache = groups()
        metadata = {}
        for group in cache.values():
            for _, _, geometry in group:
                metadata[geometry['asset_id'][0]] = {
                    'asset_id': geometry['asset_id'][0], 'integration_source': 'synthetic',
                    'target_joint_count': 1, 'topology_sha256': 'topology',
                    'train_topology_frequency': 2, 'selected_train_topology_frequency': 1,
                }
        index = {'selection_groups': {key: list(metadata.values()) for key in cache}}
        cfg = config(updates=1)
        cfg['comparison']['generation_frame_indices'] = [0, 1]
        calls = []

        def evaluate(model, items, tokenizer, config_value, output, step, *, frame_indices):
            rows = list(items)
            calls.append((len(rows), list(frame_indices)))
            return {
                'summary': {'main_ce': float(len(rows)), 'main_accuracy': 1.0},
                'rows': [{'asset_id': geometry['asset_id'][0], 'main_ce': 0.5} for _, _, geometry in rows],
                'auxiliary_summary': {'exact_parents': 1.0},
                'auxiliary_rows': [{'asset_id': geometry['asset_id'][0], 'exact_parents': True}
                                   for _, _, geometry in rows],
            }

        with tempfile.TemporaryDirectory() as directory, patch.dict(NAMESPACE, {'evaluate': evaluate, 'write_json': write_json}):
            report = NAMESPACE['evaluate_groups'](object(), object(), cache, index, cfg,
                                                  Path(directory), 5, 'cpu')
        self.assertEqual(calls, [(8, [0, 1]), (8, [0, 1]), (32, [0, 1])])
        for group in ('train_seen', 'train_heldout', 'valid'):
            self.assertEqual(report['groups'][group]['auxiliary_summary'], {'exact_parents': 1.0})
            self.assertEqual(len(report['auxiliary_rows'][group]), 8 if group != 'valid' else 32)
            self.assertNotIn('auxiliary_rows', report['groups'][group])
        self.assertEqual(len(report['rows']), 48)
        self.assertTrue(all(row['integration_source'] == 'synthetic' for row in report['rows']))
        self.assertEqual(report['generation_frame_indices'], [0, 1])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
