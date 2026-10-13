"""CPU execution tests for the exact formal asset-accounting loop."""
import ast
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / 'model_training/rigweave/scripts'


def functions(filename, names, namespace):
    nodes = [node for node in ast.parse((SCRIPTS / filename).read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError('missing actual implementation')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), filename, 'exec'), namespace)


def namespace():
    result = {'torch': torch, 'np': np, 'Path': Path, 'json': json, 'hashlib': hashlib, 'time': time,
              'datetime': datetime, 'timezone': timezone, 'nullcontext': nullcontext}
    functions('train_virtual_query_xyz.py', ['validate_plan', 'update_asset_target', 'run_training',
                                           'evaluation_summary', 'load_capacity_selection', 'capacity_contract',
                                           'validate_capacity_receipt', 'run_profile', 'run_evaluation'], result)
    functions('fusion_evaluation.py', ['auxiliary_summary'], result)
    functions('virtual_query_xyz_runtime.py', ['write_json', 'sha256'], result)
    functions('virtual_frame_runtime.py', ['normalize_asset_gradients'], result)
    return result


def config():
    return {'training': {'target_assets': 80016, 'effective_assets_per_update': 48,
                         'checkpoint_assets': [10032, 40032, 80016], 'evaluation_assets': [10032, 40032],
                         'frame_budget': 48, 'batch_cap': 8},
            'optimization': {'schedule_samples': 80016, 'auxiliary_weight': .1}, 'architecture': {},
            'initialization': {'checkpoint': 'official'}, 'auxiliary_codec': {'max_joints': 256},
            'surface_sampling': {'query_tokens': 1024}, 'runtime': {'physical_gpus': [5, 7]},
            'capacity': {'selection': {'sha256': 'pinned-capacity'}},
            'data': {'frames_min': 2, 'frames_max': 24,
                     'max_main_tokens': 2048, 'frame_seed': 1, 'surface_seed': 2, 'heldout_frames': 8,
                     'manifests': {'train': {'rows': 18002}, 'valid': {'rows': 967}}}}


class TrainingAccountingTests(unittest.TestCase):
    def test_formal_evaluation_generates_and_combines_all_auxiliary_assets(self):
        for final in (False, True):
            with self.subTest(final=final), tempfile.TemporaryDirectory() as directory:
                ns, plan, calls = namespace(), config(), []
                plan['evaluation'] = {'frames': 4, 'max_new_tokens': 2048, 'generated_frame_indices': [0, 1, 2, 3]}
                heldout = [{'asset_id': 'h0'}, {'asset_id': 'h1'}]
                valid = [{'asset_id': 'v0'}, {'asset_id': 'v1'}, {'asset_id': 'v2'}]
                def evaluate(model, items, tokenizer, eval_config, output, step, frame_indices):
                    model.eval()
                    self.assertTrue(eval_config['probe']['evaluate_auxiliary'])
                    self.assertEqual(eval_config['probe']['auxiliary_weight'], .1)
                    self.assertEqual(eval_config['probe']['max_new_tokens'], 2048)
                    self.assertEqual(frame_indices, [0, 1, 2, 3])
                    assets = list(items)
                    calls.extend(assets)
                    rows = [{'asset_id': asset, 'success': True, 'quantized_edge_f1': 1.,
                             'original_edge_f1': 1., 'exact_count': True, 'main_ce': .1,
                             'main_accuracy': .99, 'quantized_j2j': 0.}
                            for asset in assets for _ in frame_indices]
                    auxiliary = [{'asset_id': asset, 'natural_eos': True, 'success': True,
                                  'exact_count': asset.endswith('0'), 'exact_parents': asset.endswith('0'),
                                  'exact_tokens': asset.endswith('0')} for asset in assets]
                    return {'rows': rows, 'auxiliary_rows': auxiliary}
                ns.update(selected_evaluation_rows=lambda *a: {'train_heldout': heldout, 'valid': valid[:2]},
                          fixed_dataset=lambda rows, *a, **k: rows,
                          prepare=lambda dataset, requests, *a: dataset[requests[0]]['asset_id'],
                          evaluate=evaluate, dist=SimpleNamespace(barrier=lambda: None))
                samples = 80016 if final else 48
                model = nn.Linear(1, 1)
                args = SimpleNamespace(output=Path(directory))
                for rank in (1, 0):
                    ns['run_evaluation'](model, None, None, {'valid': valid}, plan, args,
                                         torch.device('cpu'), samples, rank, 2)
                    self.assertTrue(model.training)
                report = json.loads((args.output / f'evaluation_{samples:06d}' / 'summary.json').read_text())
                expected_valid = 3 if final else 2
                self.assertEqual(report['auxiliary_summary']['valid']['assets'], expected_valid)
                self.assertEqual(report['summary']['valid']['views'], expected_valid * 4)
                self.assertAlmostEqual(report['auxiliary_summary']['valid']['exact_parents'], 1 / expected_valid)
                self.assertEqual(report['auxiliary_summary']['train_heldout']['assets'], 2)
                self.assertEqual(report['full_validation'], final)
                self.assertEqual(len(calls), 2 + expected_valid)

    def test_profile_exercises_accumulation_with_resident_gradients_and_token_check(self):
        ns, plan = namespace(), config()
        plan['capacity'].update(allocated_fraction_limit=.9,
                                cases=[{'frames': 2, 'batch_per_rank': 1, 'stress_group': 'longest_main'}])
        plan['optimization']['clip_grad_norm'] = 1e9
        flags, gradients_at_prepare = [], []
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(1.))
            def forward(self, surface, target, auxiliary_weight):
                return {'loss': (self.weight * surface).square().mean()}
        model = Model()
        class Wrapped:
            skip = False
            @contextmanager
            def no_sync(self):
                self.skip = True
                try:
                    yield
                finally:
                    self.skip = False
            def __call__(self, *args, **kwargs):
                flags.append(self.skip)
                return model(*args, **kwargs)
        def prepare(*args):
            gradients_at_prepare.append(None if model.weight.grad is None else float(model.weight.grad))
            return (torch.ones(1), {'main_attention_mask': torch.ones(1, 2, 7, dtype=torch.bool)},
                    {'asset_id': ['a'], 'path': ['/a.npz'], 'joint_count': torch.tensor([1])})
        ns.update(fixed_dataset=lambda *a, **k: [], optimizer_for=lambda m, c: torch.optim.SGD(m.parameters(), lr=.1),
                  wrap_model=lambda *a: Wrapped(), load_capacity_selection=lambda *a: ({'longest_main': [0]}, {('a', '/a.npz'): 7}),
                  FrameSampleRequest=lambda *a: a, prepare=prepare, grad_norms=lambda m: {}, require_gradients=lambda *a: None,
                  dist=SimpleNamespace(all_reduce=lambda *a, **k: None, ReduceOp=SimpleNamespace(MIN='min')))
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(torch, 'autocast', lambda *a, **k: nullcontext()), \
             patch.object(torch.cuda, 'reset_peak_memory_stats'), patch.object(torch.cuda, 'synchronize'), \
             patch.object(torch.cuda, 'get_device_properties', return_value=SimpleNamespace(total_memory=1000)), \
             patch.object(torch.cuda, 'max_memory_allocated', return_value=100), \
             patch.object(torch.cuda, 'max_memory_reserved', return_value=200):
            args = SimpleNamespace(output=Path(directory))
            ns['run_profile'](model, None, None, {}, {'train': []}, plan, args,
                              torch.device('cpu'), 0, 2, 'test')
            report = json.loads((Path(directory) / 'capacity_rank_0.json').read_text())
        self.assertEqual(flags, [True, False, True, False])
        self.assertEqual(gradients_at_prepare[0::2], [None, None])
        self.assertAlmostEqual(gradients_at_prepare[1], 1.)
        self.assertAlmostEqual(gradients_at_prepare[3], .8, places=6)
        self.assertAlmostEqual(float(model.weight.detach()), .64, places=6)
        self.assertTrue(report['accepted'])
        self.assertEqual([x['accumulation_microbatches'] for x in report['cases'][0]['measurements']], [2, 2])

    def test_capacity_selection_rejoins_only_exact_manifest_rows(self):
        ns, plan = namespace(), config()
        rows = [{'asset_id': f'a{i}', 'path': f'/assets/{i}.npz'} for i in range(3)]
        values = [{**row, 'manifest_row_index': i, 'main_tokens': 30 - i} for i, row in enumerate(rows)]
        selection = {'status': 'passed', 'manifest': plan['data']['manifests']['train'],
                     'groups': {'longest_main': values, 'largest_vertices': values[:1],
                                'largest_faces': values[:1], 'largest_geometry': values[:1]},
                     'geometry_by_frames': {str(t): values[0] for t in range(2, 25)}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'selection.json'
            path.write_text(json.dumps(selection))
            plan['capacity']['selection'] = {'path': str(path), 'sha256': ns['sha256'](path)}
            groups, lengths = ns['load_capacity_selection'](plan, rows)
            self.assertEqual(groups['longest_main'], [0, 1, 2])
            self.assertEqual(lengths[('a0', '/assets/0.npz')], 30)
            with self.assertRaisesRegex(ValueError, 'identity/index'):
                ns['load_capacity_selection'](plan, list(reversed(rows)))
            selection['geometry_by_frames'].pop('24')
            path.write_text(json.dumps(selection))
            plan['capacity']['selection']['sha256'] = ns['sha256'](path)
            with self.assertRaisesRegex(ValueError, 'cover all required'):
                ns['load_capacity_selection'](plan, rows)

    def test_formal_capacity_receipt_binds_runtime_and_all_frame_shapes(self):
        ns, plan = namespace(), config()
        receipt = {'capacity_contract': ns['capacity_contract'](plan), 'approved_cases': [
            {'frames': t, 'batch_per_rank': min(8, 48 // t), 'stress_group': group,
             'ranks_approved': [0, 1], 'accumulation_microbatches': 2}
            for t in range(2, 25) for group in ('longest_main', f'geometry_t{t}')]}
        ns['validate_capacity_receipt'](plan, receipt)
        receipt['approved_cases'].pop()
        with self.assertRaisesRegex(ValueError, 'lacks measured'):
            ns['validate_capacity_receipt'](plan, receipt)
        receipt['capacity_contract'] = {**receipt['capacity_contract'], 'auxiliary_weight': .3}
        with self.assertRaisesRegex(ValueError, 'differs'):
            ns['validate_capacity_receipt'](plan, receipt)

    def test_exact_formal_budget_and_boundaries(self):
        ns, plan = namespace(), config()
        milestones = ns['validate_plan'](plan, 2)
        samples, updates = 0, []
        while samples < 80016:
            amount = ns['update_asset_target'](samples, 80016, 48, milestones, 2) * 2
            samples += amount
            updates.append(samples)
        self.assertEqual(samples, 80016)
        self.assertEqual(len(updates), 1667)
        self.assertTrue(set(milestones).issubset(updates))

    def test_nonstandard_milestone_truncates_without_overshoot(self):
        ns = namespace()
        samples, counts = 0, []
        while samples < 100:
            amount = ns['update_asset_target'](samples, 100, 48, [30, 100], 2) * 2
            samples += amount
            counts.append(amount)
        self.assertEqual(counts, [30, 48, 22])

    def test_reject_contract_drift(self):
        for field, value in [('target_assets', 80015), ('target_assets', 80032),
                             ('effective_assets_per_update', 47), ('effective_assets_per_update', True)]:
            with self.subTest(field=field, value=value):
                plan = config()
                plan['training'][field] = value
                with self.assertRaises(ValueError):
                    namespace()['validate_plan'](plan, 2)
        plan = config()
        plan['architecture']['frame_residual_scale'] = 1.
        with self.assertRaises(ValueError):
            namespace()['validate_plan'](plan, 2)
        for assets, world in ((96, 2), (24, 2), (48, 1), (48, 4)):
            with self.subTest(assets=assets, world=world):
                plan = config()
                plan['training']['effective_assets_per_update'] = assets
                with self.assertRaisesRegex(ValueError, 'exactly two ranks'):
                    namespace()['validate_plan'](plan, world)

    def test_distributed_asset_scaling_equals_global_mean(self):
        normalize = namespace()['normalize_asset_gradients']
        parameter = nn.Parameter(torch.tensor(2.))
        # DDP has averaged two rank sums, with unequal microbatch sizes.
        parameter.grad = torch.tensor((3. + 7. + 11. + 13. + 17.) / 2)
        normalize([parameter], global_assets=5, world_size=2)
        self.assertAlmostEqual(parameter.grad.item(), (3 + 7 + 11 + 13 + 17) / 5, places=5)

    def test_failed_generations_remain_in_aggregate(self):
        values = []
        for success in (True, False):
            values.append({'success': success, 'quantized_edge_f1': float(success),
                           'original_edge_f1': float(success), 'exact_count': success,
                           'main_ce': 1., 'main_accuracy': .9, 'quantized_j2j': 0. if success else None})
        result = namespace()['evaluation_summary'](values)
        self.assertEqual(result['quantized_edge_f1'], .5)
        self.assertEqual(result['geometry_scored_views'], 1)
        self.assertEqual(result['views'], 2)

    def test_actual_training_loop_averages_assets_not_microbatches_or_frames(self):
        ns = namespace()
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(1.))
            def forward(self, surface, targets, auxiliary_weight):
                per_asset = (self.weight * surface).square()
                return {'loss_per_asset': per_asset, 'loss': per_asset.mean(), 'main_loss': per_asset.mean(),
                        'auxiliary_loss': per_asset.new_tensor(0.), 'main_accuracy': per_asset.new_tensor(1.)}
        class Stream:
            def __init__(self, *args, **kwargs):
                self.index = 0
            def take(self, maximum):
                size = min(2, maximum)
                frames = 2 if self.index in (0, 4) else 4
                values = [SimpleNamespace(item_index=i, frames=frames) for i in
                          range(self.index, self.index + size)]
                self.index += size
                return values
        def prepare(dataset, requests, *args):
            return torch.tensor([float(x.item_index + 1) for x in requests]), None, None
        checkpoints, evaluations = [], []
        ns.update(validate_plan=lambda c, w: [5], fixed_dataset=lambda *a, **k: [],
                  optimizer_for=lambda model, c: torch.optim.SGD(model.parameters(), lr=.1),
                  FrameBudgetStream=Stream, wrap_model=lambda model, *a: model, prepare=prepare,
                  set_exposure_lr=lambda *a: (.1, .9), grad_norms=lambda model: {},
                  require_gradients=lambda *a: None,
                  save_checkpoint=lambda *a: checkpoints.append(a[5]),
                  run_evaluation=lambda *a: evaluations.append(a[7]))
        plan = config()
        plan['training'].update(target_assets=5, effective_assets_per_update=4, frame_budget=8, batch_cap=2,
                                checkpoint_assets=[5], evaluation_assets=[])
        plan['optimization'].update(auxiliary_weight=0., clip_grad_norm=1e9)
        plan['data']['stream_seed'] = 1
        with tempfile.TemporaryDirectory() as directory:
            model = Model()
            args = SimpleNamespace(output=Path(directory))
            with patch.object(torch, 'autocast', lambda *a, **k: nullcontext()), \
                 patch.object(torch.cuda, 'reset_peak_memory_stats'), patch.object(torch.cuda, 'synchronize'), \
                 patch.object(torch.cuda, 'max_memory_allocated', return_value=0), \
                 patch.object(torch.cuda, 'max_memory_reserved', return_value=0):
                ns['run_training'](model, None, None, {}, {'train': []}, plan, args,
                                   torch.device('cpu'), 0, 1, 'test')
            report = json.loads((Path(directory) / 'rank_0.json').read_text())
            trace = [json.loads(line) for line in (Path(directory) / 'trace_rank_0.jsonl').read_text().splitlines()]
        self.assertAlmostEqual(model.weight.item(), 2., places=5)
        self.assertEqual([x['assets_in_update'] for x in trace], [4, 1])
        self.assertEqual(report['asset_exposures'], 5)
        self.assertEqual(report['supervised_frame_exposures'], 14)
        self.assertTrue(report['complete'])
        self.assertEqual(checkpoints, [5])
        self.assertEqual(evaluations, [5])


if __name__ == '__main__':
    unittest.main()
