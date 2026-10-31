import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'model_training/rigweave/scripts'), str(ROOT / 'model_training/rigweave/src')]
from predicted_reference_protocol import compare, reference_from_row, summarize, validate_config
from predicted_reference_runtime import head_prediction, score_output, source_rows


class PredictedReferenceTests(unittest.TestCase):
    def source(self):
        return {'success': True, 'status': 'eos', 'predicted_joints': [[0., 0., 0.], [1., 0., 0.]],
                'predicted_parents': [None, 0], 'gt_joint_count': 7}

    def test_wrong_count_is_not_fixed_or_discarded(self):
        row = self.source()
        saved = copy.deepcopy(row)
        (xyz, parents), status = reference_from_row(row)
        self.assertEqual((len(xyz), parents, status), (2, [-1, 0], 'valid'))
        self.assertEqual(row, saved)

    def test_failed_generation_stays_failed(self):
        self.assertEqual(reference_from_row({'success': False, 'status': 'token_budget'}), (None, 'generation_token_budget'))

    def test_invalid_tree_is_not_repaired(self):
        for parents in ([-1, -1], [-1, 1], [0, 0], [-1, 2], [-1], [-1, .5]):
            row = self.source()
            row['predicted_parents'] = parents
            self.assertEqual(reference_from_row(row)[1], 'invalid_reference_tree')

    def test_nonfinite_coordinates_fail(self):
        row = self.source()
        row['predicted_joints'][1][1] = float('nan')
        self.assertEqual(reference_from_row(row)[1], 'invalid_reference_coordinates')

    def test_head_joint_limit(self):
        self.assertIsNone(reference_from_row(self.source(), max_joints=1)[0])

    def test_predicted_reference_has_no_known_id_mpjpe(self):
        xyz = torch.tensor([[0., 0., 0.], [1., 0., 0.]])
        row = score_output(xyz, [-1, 0], xyz, [-1, 0], 2., known_ids=False, status='valid')
        self.assertTrue(row['success'])
        self.assertEqual(row['normalized_j2j'], 0.)
        self.assertIsNone(row['normalized_mpjpe'])
        gt = score_output(xyz, [-1, 0], xyz, [-1, 0], 2., known_ids=True, status='valid')
        self.assertEqual(gt['normalized_mpjpe'], 0.)

    def test_failure_quality_is_zero_and_geometry_null(self):
        row = score_output(None, None, None, None, 1., known_ids=False, status='generation_fail')
        self.assertEqual((row['success'], row['edge_f1'], row['normalized_j2j']), (False, 0., None))
        self.assertNotIn('predicted_joints', row)

    def test_real_head_variable_joint_count_and_no_mutation(self):
        from rigweave.dynamic_rig.anchor_joint_binding import AnchorJointBinding, AnchorJointConfig
        head = AnchorJointBinding(AnchorJointConfig(input_dim=8, dim=16, heads=2, reference_depth=1, target_depth=1,
                                                   gradient_checkpointing=False)).eval()
        state = {k: v.clone() for k, v in head.state_dict().items()}
        features = torch.randn(1, 3, 8, 8)
        surface = {'query_points': torch.randn(1, 3, 8, 3), 'query_normals': torch.randn(1, 3, 8, 3)}
        for count in (2, 5):
            xyz = torch.randn(count, 3)
            parents = [-1] + list(range(count - 1))
            result = head_prediction(head, features, surface, xyz, parents, 1)
            self.assertEqual(result.shape, (3, count, 3))
            self.assertTrue(torch.equal(result[1], xyz))
            self.assertFalse(result.requires_grad)
        self.assertTrue(all(torch.equal(v, head.state_dict()[k]) for k, v in state.items()))

    def row(self, asset='a', success=True):
        return {'asset_id': asset, 'reference_slot': 0, 'target_slot': 1, 'success': success,
            'edge_f1': .8 if success else 0., 'exact_count': success,
            'normalized_j2j': .02 if success else None, 'normalized_mpjpe': None,
            'surface_sha256': 's', 'normalization_sha256': 'n', 'raw_frame_index': 4,
            'reference_raw_frame': 2, 'original_GT_sha256': 'g', 'target_mesh_diagonal': 2.,
            'reference_kind': 'predicted', 'reference_sha256': 'r'}

    def test_asset_equal_summary_and_failure_denominator(self):
        rows = [self.row(), self.row('b', False)]
        result = summarize(rows)
        self.assertEqual((result['success'], result['edge_f1'], result['views']), (.5, .4, 2))
        self.assertEqual((result['geometry_assets'], result['geometry_views']), (1, 1))

    def test_pair_rejects_input_and_reference_mismatch(self):
        for field in ('surface_sha256', 'normalization_sha256', 'original_GT_sha256', 'reference_sha256'):
            changed = self.row()
            changed[field] = 'other'
            with self.assertRaises(ValueError):
                compare([self.row()], [changed])

    def test_paired_geometry_only_common_success(self):
        result = compare([self.row(), self.row('b', False)], [self.row(), self.row('b')])
        self.assertEqual(result['geometry_jointly_successful_views'], 1)
        self.assertEqual(result['metrics']['normalized_j2j']['assets'], 1)
        self.assertEqual(result['metrics']['success']['assets'], 2)

    def test_gt_and_predicted_reference_may_intentionally_differ(self):
        gt = self.row()
        gt.update(reference_kind='GT', reference_sha256='GT')
        self.assertEqual(compare([self.row()], [gt])['paired_views'], 1)

    def test_duplicate_rows_rejected(self):
        with self.assertRaises(ValueError):
            compare([self.row(), self.row()], [self.row(), self.row()])

    def test_full_model_visual_recipe(self):
        config = json.loads((ROOT / 'model_training/experiments/full_model_visual12_20261031.json').read_text())
        for mode in ('audit', 'capacity'):
            validate_config(config, mode)
        for field, value in [('label', 'joint_080000'), ('sha256', '0'*64)]:
            changed = copy.deepcopy(config)
            changed['checkpoints']['stage1'][field] = value
            with self.assertRaises(ValueError):
                validate_config(changed, 'capacity')
        changed = copy.deepcopy(config)
        changed['official_smoke']['full_beam_capacity_first'] = False
        with self.assertRaises(ValueError):
            validate_config(changed, 'capacity')
        with self.assertRaises(ValueError):
            validate_config(config, 'train')

    def test_full_model_inputs_are_scientifically_bound(self):
        from predicted_reference_protocol import scientific_sha
        config = json.loads((ROOT / 'model_training/experiments/full_model_visual12_20261031.json').read_text())
        changed = copy.deepcopy(config)
        changed['full_model_inputs']['rows']['sha256'] = '0'*64
        self.assertNotEqual(scientific_sha(config), scientific_sha(changed))

    def test_no_training_mode_and_exact_recipe(self):
        config = json.loads((ROOT / 'model_training/experiments/predicted_reference_20261029_v1.json').read_text())
        validate_config(config, 'audit')
        validate_config(config, 'capacity')
        with self.assertRaises(ValueError):
            validate_config(config, 'train')
        for key, value in [('optimizer_updates', 1), ('GT_reference_repair', True), ('precision', 'bf16')]:
            changed = copy.deepcopy(config)
            changed['evaluation'][key] = value
            with self.assertRaises(ValueError):
                validate_config(changed, 'capacity')


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
