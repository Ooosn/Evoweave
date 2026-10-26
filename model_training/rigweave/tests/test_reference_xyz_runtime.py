from copy import deepcopy
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'model_training/rigweave/scripts'))
from reference_xyz_protocol import SCOPE, ENTRYPOINT, select_cohort, validate_config
from reference_xyz_runtime import pack, row_metrics
from reference_xyz_runtime import generate_reference
import virtual_pair_guard as guard
import test_virtual_pair_scheduler as scheduler_tests


class ReferenceRuntimeTest(unittest.TestCase):
    def test_natural_reference_generation_and_invalid_tree(self):
        tokenizer = SimpleNamespace(bos=100, eos=101, cls_name_to_token=lambda name: 102,
            detokenize=lambda tokens: SimpleNamespace(joints=[[0., 0., 0.], [0., .5, 0.]], parents=[None, 0]))
        calls = []
        def generate(condition, start, **kwargs):
            calls.append((condition, start, kwargs))
            return {'success': True, 'status': 'eos', 'tokens': [100, 102, 101]}
        model = SimpleNamespace(generate_condition=generate)
        with patch('reference_xyz_runtime.torch.autocast', return_value=nullcontext()):
            record = generate_reference(model, tokenizer, torch.zeros(1, 4, 8))
        self.assertTrue(record['success'])
        self.assertEqual(record['parents'].tolist(), [-1, 0])
        self.assertTrue(callable(calls[0][2]['logits_processor']))
        self.assertEqual(calls[0][1].tolist(), [100, 102])
        tokenizer.detokenize = lambda tokens: SimpleNamespace(joints=[[0., 0., 0.], [0., .5, 0.]], parents=[None, None])
        with patch('reference_xyz_runtime.torch.autocast', return_value=nullcontext()):
            failed = generate_reference(model, tokenizer, torch.zeros(1, 4, 8))
        self.assertFalse(failed['success'])
        self.assertEqual(failed['status'], 'invalid_reference_tree')

    def test_exact_config_and_bounded_contract(self):
        config = json.loads((ROOT / 'model_training/experiments/reference_skeleton_xyz_frozen_pilot_20261027_v1.json').read_text())
        validate_config(config)
        for key, value in (('updates', 10000), ('backbone_mode', 'trainable'), ('coordinate_clipping', True)):
            bad = deepcopy(config)
            bad['pilot'][key] = value
            with self.assertRaises(ValueError):
                validate_config(bad)

    def test_cohort_is_stratified_order_independent_and_split_safe(self):
        rows = [{'asset_id': f'asset_{count}_{i}', 'path': f'/npz/{count}_{i}',
                 'split': 'valid', 'canonical_metrics': {'target_joint_count': count}}
                for count in (4, 30, 52, 100) for i in range(5)]
        selected, report = select_cohort(rows, 'valid', 8, 20261027)
        reverse, _ = select_cohort(rows[::-1], 'valid', 8, 20261027)
        self.assertEqual(selected, reverse)
        self.assertEqual([r['stratum'] for r in report], ['low']*2 + ['mid']*2 + ['gt52']*2 + ['high']*2)
        with self.assertRaises(ValueError):
            select_cohort(rows, 'train', 8, 20261027)
        with self.assertRaises(ValueError):
            select_cohort(rows + rows[:1], 'valid', 8, 20261027)

    def test_pack_preserves_full_targets(self):
        rows = []
        for count in (3, 5):
            rows.append({'features': torch.ones(8, 4, 8).bfloat16(), 'points': torch.ones(8, 4, 3),
                'normals': torch.ones(8, 4, 3), 'targets': torch.full((8, count, 3), float(count)),
                'parents': torch.tensor([-1] + [0] * (count-1))})
        result = pack(rows, 'cpu')
        self.assertEqual(result['targets'].shape, (2, 8, 5, 3))
        self.assertEqual(result['parents'][0].tolist(), [-1, 0, 0, -2, -2])
        self.assertEqual(result['mask'].sum(1).tolist(), [3, 5])
        self.assertTrue((result['targets'][1] == 5).all())

    def test_position_metric_catches_shift_without_topology_change(self):
        truth = torch.tensor([[0., 0., 0.], [1., 0., 0.]])
        prediction = truth + torch.tensor([0., .2, 0.])
        metrics = row_metrics(prediction, [-1, 0], truth, [-1, 0], 2., paired_ids=True)
        self.assertEqual(metrics['edge_f1'], 1.)
        self.assertAlmostEqual(metrics['normalized_mpjpe'], .1, places=6)
        self.assertEqual(metrics['pck_005'], 0.)
        unpaired = row_metrics(prediction, [-1, 0], truth, [-1, 0], 2., paired_ids=False)
        self.assertNotIn('normalized_mpjpe', unpaired)

    def test_new_family_keeps_exact_admission_and_old_entrypoint_separate(self):
        config, state, args = scheduler_tests.fixture()
        config['family'] = SCOPE
        state['active_operation'].update(label=SCOPE, required_entrypoint=ENTRYPOINT)
        scheduler_tests.SchedulerGuardTests().run_guard(config, state, args)
        with patch.object(guard, 'file_sha256', return_value='config-digest'):
            guard.validate_stage(state, config, args)
            state['active_operation']['required_entrypoint'] = guard.ENTRYPOINT
            with self.assertRaises(RuntimeError):
                guard.validate_stage(state, config, args)

    def test_extended_lifetime_is_required_when_declared(self):
        config, _, _ = scheduler_tests.fixture()
        config['runtime']['required_h_rt_seconds'] = 172800
        grant = scheduler_tests.grant_text(config) + 'submit_cmd: qrsh -l h_rt=24:00:00\n'
        guard.verify_allocation(config, grant + 'hard_resource_list: h100=2,h_rt=172800,mem_req=4G\n')
        for missing in ('', 'hard_resource_list: h100=2,h_rt=129600,mem_req=4G\n'):
            with self.assertRaises(RuntimeError):
                guard.verify_allocation(config, grant + missing)


if __name__ == '__main__':
    unittest.main()
