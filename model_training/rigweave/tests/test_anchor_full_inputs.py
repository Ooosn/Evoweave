from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'model_training/rigweave/scripts'))
from anchor_full_evaluation import archive_inputs, canonical_input


class FakeDataset:
    def __init__(self, raw, heldout):
        self.entries = [{'asset_id': 'test', 'path': '/source/test.npz', 'root_rt': {'version': 'test'}}]
        self.batch = {'asset_id': 'test', 'frame_vertices': torch.randn(1, 8, 16, 3), 'heldout_raw_ids': heldout}
        self.geometry = {'joint_mask': torch.ones(1, 3, dtype=torch.bool),
            'frame_target_joints': torch.randn(1, 8, 3, 3), 'target_parents': torch.tensor([[-1, 0, 1]]),
            'selected_frames': torch.tensor([raw]), 'query_center': torch.zeros(1, 3), 'query_scale': torch.ones(1)}
        self.surface = {'dense_points': torch.randn(1, 8, 16, 3), 'dense_normals': torch.randn(1, 8, 16, 3),
            'query_points': torch.randn(1, 8, 4, 3), 'query_normals': torch.randn(1, 8, 4, 3),
            'query_indices': torch.zeros(1, 8, 4, dtype=torch.long)}
        self.calls = 0

    def __getitem__(self, index):
        self.calls += 1
        return self.batch


class CanonicalInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = {'data_cache': {'root': str(self.root/'cache')},
                       'artifacts': {'durable_root': str(self.root/'durable')}}
        self.base = {'data': {'frames_min': 2}, 'surface_sampling': {}}

    def tearDown(self):
        self.temp.cleanup()

    def test_seen_inputs_replay_without_resampling_and_archive_verifies(self):
        dataset = FakeDataset(list(range(8)), list(range(8, 16)))
        with patch('anchor_full_evaluation.prepare_virtual_surface', return_value=(dataset.surface, dataset.geometry)):
            first, spec = canonical_input(dataset, 0, self.base, self.config, 'seen_eval', torch.device('cpu'))
        with patch('anchor_full_evaluation.prepare_virtual_surface', side_effect=AssertionError('resampled')):
            second, repeated = canonical_input(dataset, 0, self.base, self.config, 'seen_eval', torch.device('cpu'))
        self.assertEqual(dataset.calls, 1)
        self.assertEqual(spec, repeated)
        torch.testing.assert_close(first['surface']['dense_points'], second['surface']['dense_points'], rtol=0, atol=0)
        output = self.root/'train_v26'
        output.mkdir()
        archive = archive_inputs([spec], output, self.config, final=False)
        self.assertTrue(archive['all_input_members_verified'])
        self.assertEqual(archive['canonical_assets'], 1)
        self.assertTrue(Path(archive['durable_path']).exists())

    def test_reserved_pose_diagnostic_uses_reserved_input_partition(self):
        dataset = FakeDataset(list(range(8, 16)), list(range(8, 16)))
        with patch('anchor_full_evaluation.prepare_virtual_surface', return_value=(dataset.surface, dataset.geometry)):
            result, _ = canonical_input(dataset, 0, self.base, self.config, 'heldout_eval', torch.device('cpu'))
        self.assertEqual(result['frame_partition'], 'heldout_eval')
        self.assertEqual(result['raw_frames'], list(range(8, 16)))

    def test_seen_input_cannot_use_reserved_frames(self):
        dataset = FakeDataset(list(range(8, 16)), list(range(8, 16)))
        with patch('anchor_full_evaluation.prepare_virtual_surface', return_value=(dataset.surface, dataset.geometry)):
            with self.assertRaises(RuntimeError):
                canonical_input(dataset, 0, self.base, self.config, 'seen_eval', torch.device('cpu'))


if __name__ == '__main__':
    unittest.main()
