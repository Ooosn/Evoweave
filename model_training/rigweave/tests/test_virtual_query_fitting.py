"""CPU control-flow checks for bounded fitting optimization, not model quality."""
import ast
from contextlib import nullcontext
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
# Load optimizer registrations before tests temporarily snapshot sys.modules.
import torch._dynamo
from torch import nn

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
NS = {'torch': torch, 'math': math, 'Path': Path}


def extract(path, names, namespace):
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise RuntimeError('actual fitting helper missing')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), namespace)


extract(SCRIPTS / 'probe_virtual_query_xyz.py',
        ['fitting_plan', 'fitting_update', 'fit_view_passes', 'declared_probe_run'], NS)
SCHEDULE = {'math': math, 'Any': object}
extract(SCRIPTS / 'virtual_frame_runtime.py', ['exposure_learning_rate', 'set_exposure_lr'], SCHEDULE)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([.15, -.25], dtype=torch.float64))

    def forward(self, surface, target, *, auxiliary_weight):
        main = (self.weight * surface - target).square().mean()
        auxiliary = self.weight.square().mean()
        return {'loss': main + auxiliary_weight * auxiliary, 'main_loss': main,
                'auxiliary_loss': auxiliary, 'main_accuracy': main.detach() * 0 + .5}


def configuration(batch=None, schedule='constant'):
    probe = {'max_updates': 10, 'frames': 4, 'auxiliary_weight': .1}
    if batch is not None:
        probe['assets_per_update'] = batch
    return {'probe': probe, 'optimization': {'lr': 1e-4, 'weight_decay': 0., 'schedule': schedule,
        'peak_lr': 1e-4, 'div_factor': 5., 'final_div_factor': 10., 'warmup_fraction': .1,
        'schedule_samples': 10 * (batch or 1)}}


def items():
    return [(torch.tensor([.3 + i * .05, -.2], dtype=torch.float64),
             torch.tensor([.1, .2 - i * .04], dtype=torch.float64), {'asset_id': [str(i)]})
            for i in range(5)]


def gradients(model):
    return {'all': float(model.weight.grad.norm())}


class FittingTests(unittest.TestCase):
    def setUp(self):
        module = ModuleType('virtual_frame_runtime')
        module.set_exposure_lr = SCHEDULE['set_exposure_lr']
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {'virtual_frame_runtime': module}).start()
        patch.object(torch, 'autocast', lambda *a, **k: nullcontext()).start()
        patch.dict(NS, {'grad_norms': gradients, 'require_gradients': lambda *a: None}).start()

    def test_legacy_default_is_exact_same_adamw_trajectory(self):
        cfg = configuration()
        del cfg['optimization']['schedule']
        plan = NS['fitting_plan'](cfg, 5)
        self.assertEqual(plan['max_asset_exposures'], 10)
        model, baseline = TinyModel(), TinyModel()
        left = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9, .999), weight_decay=0.)
        right = torch.optim.AdamW(baseline.parameters(), lr=1e-4, betas=(.9, .999), weight_decay=0.)
        for step in range(1, 11):
            report = NS['fitting_update'](model, left, items(), cfg, plan, step)
            right.zero_grad(set_to_none=True)
            surface, target, _ = items()[(step - 1) % 5]
            loss = baseline(surface, target, auxiliary_weight=.1)['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(baseline.parameters(), 1.)
            right.step()
            torch.testing.assert_close(model.weight, baseline.weight, rtol=0., atol=0.)
            self.assertEqual(report['loss'], float(loss.detach()))
            self.assertEqual(report['asset_indices'], [(step - 1) % 5])
            self.assertEqual(report['asset_exposures'], step)
            self.assertEqual(report['supervised_frame_exposures'], step * 4)
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(left.state[model.weight][key], right.state[baseline.weight][key], rtol=0., atol=0.)

    def test_accumulation_is_actual_asset_mean_before_clipping(self):
        cfg = configuration(2)
        plan = NS['fitting_plan'](cfg, 5)
        model, baseline = TinyModel(), TinyModel()
        left = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.)
        right = torch.optim.AdamW(baseline.parameters(), lr=1e-4, weight_decay=0.)
        seen = []
        for step in range(1, 11):
            report = NS['fitting_update'](model, left, items(), cfg, plan, step)
            indices = [((step - 1) * 2 + offset) % 5 for offset in range(2)]
            seen.extend(report['asset_indices'])
            right.zero_grad(set_to_none=True)
            loss = torch.stack([baseline(*items()[i][:2], auxiliary_weight=.1)['loss'] for i in indices]).mean()
            loss.backward()
            self.assertAlmostEqual(report['gradient_norm'], float(baseline.weight.grad.norm()), places=14)
            torch.nn.utils.clip_grad_norm_(baseline.parameters(), 1.)
            right.step()
            torch.testing.assert_close(model.weight, baseline.weight, rtol=1e-14, atol=1e-14)
            self.assertAlmostEqual(report['loss'], float(loss.detach()), places=14)
            self.assertEqual(report['asset_exposures'], step * 2)
            self.assertEqual(report['supervised_frame_exposures'], step * 8)
        self.assertEqual(seen, list(range(5)) * 4)
        self.assertEqual(plan['max_asset_exposures'], 20)

    def test_schedule_uses_actual_exposures_before_each_update(self):
        cfg = configuration(2, 'onecycle')
        plan = NS['fitting_plan'](cfg, 5)
        model = TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=7., weight_decay=0.)
        recorded = []
        for step in range(1, 11):
            recorded.append(NS['fitting_update'](model, optimizer, items(), cfg, plan, step))
        self.assertEqual(recorded[0]['lr'], 2e-5)
        self.assertEqual(recorded[0]['beta1'], .95)
        self.assertEqual(recorded[1]['lr'], 1e-4)
        self.assertAlmostEqual(recorded[1]['beta1'], .85)
        expected = 1e-4 + .5 * (1 - math.cos(math.pi * (18 / 20 - .1) / .9)) * (2e-6 - 1e-4)
        self.assertAlmostEqual(recorded[-1]['lr'], expected, places=15)
        self.assertEqual(optimizer.param_groups[0]['lr'], recorded[-1]['lr'])
        self.assertEqual(optimizer.param_groups[0]['betas'][0], recorded[-1]['beta1'])

    def test_invalid_counts_and_schedule_rejected(self):
        for batch in (0, -1, True, 1.5, 6):
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                NS['fitting_plan'](configuration(batch), 5)
        for key, value in (('schedule_samples', 19), ('peak_lr', float('nan')),
                           ('div_factor', 0), ('final_div_factor', -1), ('warmup_fraction', 1)):
            cfg = configuration(2, 'onecycle')
            cfg['optimization'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                NS['fitting_plan'](cfg, 5)

    def test_nonfinite_loss_does_not_update(self):
        cfg = configuration()
        plan = NS['fitting_plan'](cfg, 5)
        model = TinyModel()
        initial = model.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        broken = items()
        broken[0][0][0] = float('nan')
        with self.assertRaisesRegex(RuntimeError, 'nonfinite fitting loss'):
            NS['fitting_update'](model, optimizer, broken, cfg, plan, 1)
        torch.testing.assert_close(model.weight, initial, rtol=0., atol=0.)
        self.assertFalse(optimizer.state)

    def test_exact_tokens_are_stronger_than_nonreflexive_edge_matching(self):
        row = {'success': True, 'exact_count': True, 'quantized_j2j': 0.,
               'quantized_edge_f1': .5, 'generated_tokens': [1, 5, 9, 2], 'target_tokens': [1, 5, 9, 2]}
        self.assertTrue(NS['fit_view_passes'](row))
        row['generated_tokens'] = [1, 5, 8, 2]
        self.assertFalse(NS['fit_view_passes'](row))
        row.update(quantized_edge_f1=.95, quantized_j2j=4 / 256)
        self.assertTrue(NS['fit_view_passes'](row))
        row['quantized_j2j'] += 1e-7
        self.assertFalse(NS['fit_view_passes'](row))
        row.update(quantized_j2j=0., generated_tokens=[1, 5, 9, 2], success=False)
        self.assertFalse(NS['fit_view_passes'](row))
        row.update(success=True, exact_count=False)
        self.assertFalse(NS['fit_view_passes'](row))
        row.update(exact_count=True, generated_tokens=[], target_tokens=[], quantized_edge_f1=.5)
        self.assertFalse(NS['fit_view_passes'](row))

    def test_paired_operation_still_pins_exact_config_mode_and_output(self):
        args = SimpleNamespace(config=Path('/owned/first.json'), output=Path('/owned/first'), mode='fit')
        first = {'config': str(args.config), 'output': str(args.output), 'mode': 'fit', 'config_sha256': 'abc'}
        operation = {'mode': 'fit', 'runs': [first, {**first, 'config': '/owned/second.json'}]}
        self.assertIs(NS['declared_probe_run'](operation, args, 'abc'), first)
        self.assertIs(NS['declared_probe_run'](first, args, 'abc'), first)
        for key, value in [('config', Path('/unlisted.json')), ('output', Path('/wrong')), ('mode', 'preflight')]:
            changed = SimpleNamespace(**{**vars(args), key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                NS['declared_probe_run'](operation, changed, 'abc')
        with self.assertRaises(ValueError):
            NS['declared_probe_run'](operation, args, 'different')
        with self.assertRaises(ValueError):
            NS['declared_probe_run']({'mode': 'fit', 'runs': [first, first]}, args, 'abc')


if __name__ == '__main__':
    unittest.main()
