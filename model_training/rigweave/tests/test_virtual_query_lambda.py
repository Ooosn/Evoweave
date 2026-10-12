"""CPU tests for actual comparison helpers and evaluator frame routing."""
import ast
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / 'model_training/rigweave/scripts'
NAMESPACE = {'Path': Path, 'hashlib': hashlib, 'json': json, 'np': np, 'torch': torch}


def extract(path, names):
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise RuntimeError('missing actual helper under test')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), NAMESPACE)


extract(SCRIPTS / 'virtual_query_xyz_runtime.py', ['sha256', 'write_json'])
extract(SCRIPTS / 'probe_virtual_query_lambda.py',
        ['transfer', 'cache_contract', 'load_selection', 'load_cache', 'training_order',
         'parameter_fingerprint', 'shared_gradient_measurement'])
extract(SCRIPTS / 'probe_virtual_query_xyz.py', ['evaluate'])


def config():
    return {'data': {'frame_seed': 12}, 'surface_sampling': {'query_tokens': 2}, 'auxiliary_codec': {'max_joints': 8},
            'comparison': {'frames': 4, 'selection': {'path': '/fixed/selection.json', 'sha256': 'selection'}},
            'runtime': {'tokenizer_config_sha256': 'tokenizer'}, 'probe': {'auxiliary_weight': 0., 'max_new_tokens': 20}}


class ComparisonHelpersTests(unittest.TestCase):
    def test_selection_rejoins_full_rows_and_rejects_cross_split_paths(self):
        for overlap in (False, True):
            with self.subTest(overlap=overlap), tempfile.TemporaryDirectory() as temporary:
                train = [{'asset_id': f't{i}', 'path': f'/train/{i}.npz', 'root_rt': {'verified': True}}
                         for i in range(64)]
                valid = [{'asset_id': f'v{i}', 'path': f'/valid/{i}.npz', 'root_rt': {'verified': True}}
                         for i in range(32)]
                if overlap:
                    valid[0]['path'] = train[0]['path']
                groups = {'train': train, 'train_heldout': train[:8], 'valid': valid}
                manifests = {'train': 'pinned-train', 'valid': 'pinned-valid'}
                payload = {'manifests': manifests, 'groups': {key: [
                    {'asset_id': row['asset_id'], 'path': row['path']} for row in rows]
                    for key, rows in groups.items()}}
                path = Path(temporary) / 'selection.json'
                path.write_text(json.dumps(payload))
                cfg = {'comparison': {'selection': {'path': str(path), 'sha256': NAMESPACE['sha256'](path)}},
                       'data': {'manifests': manifests}}
                with patch.dict(NAMESPACE, {'read_entries': lambda c: {'train': train, 'valid': valid}}):
                    if overlap:
                        with self.assertRaisesRegex(ValueError, 'paths overlap'):
                            NAMESPACE['load_selection'](cfg)
                    else:
                        _, actual = NAMESPACE['load_selection'](cfg)
                        self.assertIs(actual['train'][0], train[0])
                        self.assertTrue(actual['train'][0]['root_rt']['verified'])

    def test_transfer_preserves_nested_types_values_and_detaches(self):
        tensor = torch.randn(2, requires_grad=True)
        value = ({'x': tensor, 'metadata': ['asset', 3]}, {'labels': torch.tensor([1, 2])})
        result = NAMESPACE['transfer'](value, 'cpu')
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0]['metadata'], list)
        self.assertFalse(result[0]['x'].requires_grad)
        self.assertTrue(tensor.requires_grad)
        torch.testing.assert_close(result[0]['x'], tensor)
        self.assertEqual(result[0]['metadata'], ['asset', 3])

    def test_cache_contract_allows_only_candidate_independent_inputs(self):
        left, right = config(), config()
        right['probe']['auxiliary_weight'] = .3
        right['runtime']['physical_gpu'] = 7
        self.assertEqual(NAMESPACE['cache_contract'](left), NAMESPACE['cache_contract'](right))
        right['data']['frame_seed'] += 1
        self.assertNotEqual(NAMESPACE['cache_contract'](left), NAMESPACE['cache_contract'](right))

    def test_training_order_exact_exposures_and_replay(self):
        function = NAMESPACE['training_order']
        order = function(64, 2048, 20261013)
        self.assertEqual(order, function(64, 2048, 20261013))
        self.assertNotEqual(order, function(64, 2048, 20261014))
        self.assertEqual(np.bincount(order).tolist(), [32] * 64)
        self.assertTrue(all(sorted(order[start:start + 64]) == list(range(64)) for start in range(0, 2048, 64)))
        self.assertEqual(function(64, 77, 20261013), order[:77])
        with self.assertRaises(ValueError):
            function(0, 77, 1)

    def test_initial_fingerprint_is_value_and_dtype_sensitive(self):
        left, right = nn.Linear(3, 2), nn.Linear(3, 2)
        right.load_state_dict(left.state_dict())
        fingerprint = NAMESPACE['parameter_fingerprint']
        self.assertEqual(fingerprint(left), fingerprint(right))
        with torch.no_grad():
            right.weight[0, 0] += 1
        self.assertNotEqual(fingerprint(left), fingerprint(right))
        right.load_state_dict(left.state_dict())
        self.assertNotEqual(fingerprint(left), fingerprint(right.to(torch.bfloat16)))

    def test_shared_gradient_ratio_and_direction_use_raw_auxiliary_loss(self):
        virtual = nn.Parameter(torch.ones(1, 2, 3))
        norm = nn.LayerNorm(3)
        model = SimpleNamespace(motion_encoder=SimpleNamespace(virtual_slots=virtual, norm=norm))
        losses = {'main_loss': 3 * virtual.sum() + 2 * norm.weight.sum(),
                  'auxiliary_loss': -4 * virtual.sum() + norm.weight.sum()}
        result = NAMESPACE['shared_gradient_measurement'](model, losses, .3)
        self.assertAlmostEqual(result['virtual_slots']['weighted_aux_to_main_ratio'], .4, places=6)
        self.assertAlmostEqual(result['virtual_slots']['cosine'], -1, places=6)
        self.assertAlmostEqual(result['motion_final_norm']['weighted_aux_to_main_ratio'], .15, places=6)
        self.assertAlmostEqual(result['motion_final_norm']['cosine'], 1, places=6)
        self.assertIsNone(virtual.grad)
        (losses['main_loss'] + .3 * losses['auxiliary_loss']).backward()
        torch.testing.assert_close(virtual.grad, torch.full_like(virtual, 1.8))

    def test_disabled_auxiliary_needs_no_auxiliary_graph(self):
        virtual = nn.Parameter(torch.ones(1, 2, 3))
        norm = nn.LayerNorm(3)
        model = SimpleNamespace(motion_encoder=SimpleNamespace(virtual_slots=virtual, norm=norm))
        result = NAMESPACE['shared_gradient_measurement'](model,
            {'main_loss': virtual.sum() + norm.weight.sum(), 'auxiliary_loss': None}, 0.)
        self.assertTrue(all(row['weighted_aux_to_main_ratio'] == 0 and row['cosine'] is None for row in result.values()))

    def make_cache(self, root):
        cfg = config()
        index = {'schema': 'virtual_query_geometry_cache_v1', 'contract': NAMESPACE['cache_contract'](cfg), 'groups': {}}
        for group, count in [('train', 64), ('train_heldout', 8), ('valid', 32)]:
            records = []
            for number in range(count):
                filename = f'{group}_{number:03d}.pt'
                asset = f'{group}_{number}'
                item = ({'dense_points': torch.zeros(1, 4, 2, 3)}, {'main_input_ids': torch.ones(1, 4, 3, dtype=torch.long)},
                        {'asset_id': [asset], 'selected_frames': torch.tensor([[1, 3, 5, 7]])})
                torch.save(item, root / filename)
                records.append({'filename': filename, 'asset_id': asset, 'selected_frames': [1, 3, 5, 7],
                                'sha256': NAMESPACE['sha256'](root / filename)})
            index['groups'][group] = records
        self.rewrite_index(cfg, index, root)
        return cfg, index

    def rewrite_index(self, cfg, index, root):
        path = root / 'cache_index.json'
        path.write_text(json.dumps(index))
        cfg['comparison']['cache'] = {'index': str(path), 'sha256': NAMESPACE['sha256'](path)}

    def test_cache_tensor_roundtrip_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, _ = self.make_cache(root)
            groups, _ = NAMESPACE['load_cache'](cfg)
            self.assertEqual({key: len(value) for key, value in groups.items()}, {'train': 64, 'train_heldout': 8, 'valid': 32})
            self.assertEqual(groups['train'][0][2]['selected_frames'].tolist(), [[1, 3, 5, 7]])
            with (root / 'train_000.pt').open('ab') as handle:
                handle.write(b'corruption')
            with self.assertRaisesRegex(ValueError, 'geometry/targets changed'):
                NAMESPACE['load_cache'](cfg)

    def test_cache_index_count_filename_and_contract_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, index = self.make_cache(root)
            index['groups']['train'][0]['filename'] = '../escape.pt'
            self.rewrite_index(cfg, index, root)
            with self.assertRaisesRegex(ValueError, 'unsafe cache entry'):
                NAMESPACE['load_cache'](cfg)
            index['groups']['train'].pop()
            self.rewrite_index(cfg, index, root)
            with self.assertRaisesRegex(ValueError, 'fixed comparison cohort'):
                NAMESPACE['load_cache'](cfg)
            cfg['data']['frame_seed'] = 99
            with self.assertRaisesRegex(ValueError, 'preprocessing/selection contract'):
                NAMESPACE['load_cache'](cfg)


class FrameEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tokens = torch.tensor([[[10, 11, 100 + frame, 200 + frame, 12] for frame in range(4)]])
        joints = np.array([[[100 + frame, 0, 0], [200 + frame, 0, 0]] for frame in range(4)], dtype=np.float32)
        self.items = [({}, {'main_input_ids': self.tokens, 'main_attention_mask': torch.ones_like(self.tokens, dtype=torch.bool)},
            {'asset_id': ['asset'], 'joint_count': torch.tensor([2]), 'selected_frames': torch.tensor([[7, 11, 13, 17]]),
             'target_parents': torch.tensor([[-1, 0]]), 'frame_target_joints': torch.from_numpy(joints)[None]})]
        tokenizer = SimpleNamespace(bos=10, eos=12, cls_name_to_token=lambda _: 11,
            detokenize=lambda ids: SimpleNamespace(joints=np.array([[ids[2], 0, 0], [ids[3], 0, 0]], dtype=np.float32)))
        self.tokenizer = tokenizer
        self.calls = []
        calls = self.calls

        class Model:
            def eval(self):
                return self

            def train(self):
                return self

            def encode(self, surface):
                return None, None

            def conditions(self, real, virtual):
                return torch.arange(4).reshape(1, 4, 1, 1).expand(-1, -1, 2, 3).float()

            def forward_encoded(self, real, virtual, targets, **kwargs):
                return {'main_loss_per_frame': torch.tensor([[.1, .2, .3, .4]]),
                        'main_accuracy_per_frame': torch.ones(1, 4)}

            def generate_condition(self, condition, start, **kwargs):
                frame = int(condition[0, 0, 0])
                calls.append(frame)
                return {'status': 'eos', 'success': True, 'tokens': [10, 11, 100 + frame, 200 + frame, 12]}

        self.model = Model()
        fake_rig = ModuleType('rigweave.dynamic_rig.virtual_query_xyz')
        fake_rig.original_xyz_vocabulary_processor = lambda *args: None
        fake_eval = ModuleType('eval_dynamic_rig_generation')
        fake_eval._parents_from_output = lambda obj: [-1, 0]
        fake_eval._topology_metrics = lambda a, ap, b, bp: {'edge_f1': float(np.array_equal(a, b))}
        self.modules = {'rigweave.dynamic_rig.virtual_query_xyz': fake_rig, 'eval_dynamic_rig_generation': fake_eval}

    def evaluate(self, directory, indices):
        with patch.dict(sys.modules, self.modules), patch.object(torch, 'autocast', lambda *a, **kw: nullcontext()):
            return NAMESPACE['evaluate'](self.model, self.items, self.tokenizer, config(), Path(directory), 0,
                                         frame_indices=indices)

    def test_selected_frames_keep_own_labels_conditions_and_teacher_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.evaluate(directory, [1, 3])
        self.assertEqual(self.calls, [1, 3])
        self.assertEqual([row['raw_frame_index'] for row in result['rows']], [11, 17])
        self.assertEqual([row['generated_tokens'] for row in result['rows']], [self.tokens[0, i].tolist() for i in (1, 3)])
        self.assertAlmostEqual(result['rows'][0]['main_ce'], .2, places=6)
        self.assertAlmostEqual(result['rows'][1]['main_ce'], .4, places=6)
        self.assertEqual(result['summary']['quantized_edge_f1'], 1.)

    def test_default_keeps_all_frames_and_invalid_selection_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(self.evaluate(directory, None)['views'], 4)
            self.assertEqual(self.calls, [0, 1, 2, 3])
            for indices in ([], [0, 0], [-1], [4], [True], [1.2]):
                with self.subTest(indices=indices), self.assertRaisesRegex(ValueError, 'generation frame selection'):
                    self.evaluate(directory, indices)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
