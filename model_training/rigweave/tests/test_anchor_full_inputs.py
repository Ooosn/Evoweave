from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack
import json

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'model_training/rigweave/scripts'))
from anchor_full_evaluation import archive_inputs, canonical_input, evaluate


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


class EvaluationCoverageTests(unittest.TestCase):
    def test_complete_final_split_and_intermediate_AR_coverage(self):
        class Backbone(torch.nn.Module):
            def encode(self, surface):
                return torch.zeros(1, 8, 4, 6), None
        class Head(torch.nn.Module):
            def forward(self, features, points, normals, xyz, parents, mask, reference):
                return xyz[:, None].expand(-1, 8, -1, -1).clone()
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone, self.head = Backbone(), Head()
        class Dataset:
            def __init__(self, rows):
                self.entries = rows
        def record(dataset, index, base, config, mode, device):
            raw = list(range(8, 16)) if mode == 'heldout_eval' else list(range(8))
            asset = dataset.entries[index]['asset_id']
            value = {'asset_id': asset, 'surface': {'query_points': torch.zeros(1, 8, 4, 3),
                'query_normals': torch.zeros(1, 8, 4, 3)}, 'truth': torch.zeros(8, 3, 3),
                'parents': torch.tensor([-1, 0, 1]), 'raw_frames': raw, 'diagonals': torch.ones(8),
                'normalization_sha256': 'norm', 'surface_sha256': 'surface'}
            return value, {'path': asset+'_'+mode, 'sha256': 'test'}
        metric = {'edge_f1': 1., 'normalized_j2j': 0., 'normalized_mpjpe': 0., 'pck_005': 1.,
                  'normalized_bone_vector_error': 0.}
        ar_metric = {'success': True, 'status': 'eos', 'original_edge_f1': 1., 'exact_count': True,
                     'original_normalized_j2j': 0., 'generated_tokens': [1]}
        config = json.loads((ROOT/'model_training/experiments/anchor_full_20261030_v1.json').read_text())
        entries = {'valid': [{'asset_id': f'v{i}'} for i in range(967)],
                   'train': [{'asset_id': f't{i}'} for i in range(40)]}
        class Resolver:
            index_sha256 = 'cache'
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            for name, value in {
                'eager_backbone': lambda model, base, device: (model.backbone, None),
                'CachedFiles': lambda *args: Resolver(),
                'evaluation_rows': lambda base, rows: rows['valid'][:128],
                'fixed_dataset': lambda rows, *args, **kwargs: Dataset(rows),
                'canonical_input': record,
                'joint_metrics': lambda *args: metric,
                'shared_deadline_due': lambda *args: False,
                'gather': lambda value, world: [value],
                'mirror': lambda *args: {},
                'archive_inputs': lambda specs, *args, **kwargs: {'assets': len(specs)},
                'generate_asset': lambda *args, **kwargs: {i: {} for i in range(8)},
                'score_generation': lambda *args: ar_metric,
            }.items():
                stack.enter_context(patch('anchor_full_evaluation.'+name, value))
            stack.enter_context(patch('anchor_full_evaluation.dist.barrier', lambda **kwargs: None))
            model = Model()
            output = Path(temporary)
            common = (model, {'surface_sampling': {}}, config, entries, output)
            intermediate = evaluate(*common, 50000, torch.device('cpu'), 0, 1, include_ar=True)
            final = evaluate(*common, 300000, torch.device('cpu'), 0, 1, final=True)
            self.assertTrue(intermediate['complete'])
            self.assertEqual(intermediate['stage1']['assets'], 128)
            self.assertEqual(intermediate['stage1']['views'], 1024)
            self.assertTrue(final['complete'])
            groups = {g['split']: g for g in final['groups'] if g['method'] == 'head'}
            self.assertEqual(groups['valid']['assets'], 967)
            self.assertEqual(groups['valid']['rows'], 967*14)
            self.assertEqual(groups['train_eligible']['assets'], 32)
            self.assertEqual(groups['train_heldout']['assets'], 32)
            self.assertTrue(model.training)


if __name__ == '__main__':
    unittest.main()
