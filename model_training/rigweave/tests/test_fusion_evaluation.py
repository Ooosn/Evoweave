"""Auxiliary quality gates and paired-run evidence, independent of CUDA."""
import importlib.util
import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from fusion_evaluation import auxiliary_record, auxiliary_summary, tree_fingerprint, comparison_contract, pair_ready

spec = importlib.util.spec_from_file_location('isolated_aux_codec', ROOT / 'src/rigweave/dynamic_rig/aux_skeleton_tokens.py')
codec_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = codec_module
spec.loader.exec_module(codec_module)
validation = {'Path': Path}
nodes = [node for node in ast.parse((ROOT / 'scripts/probe_virtual_fusion.py').read_text()).body
         if isinstance(node, ast.FunctionDef) and node.name == 'validate_config']
exec(compile(ast.Module(body=nodes, type_ignores=[]), '<actual fusion configuration validation>', 'exec'), validation)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.codec = codec_module.AuxSkeletonTokens(max_joints=16)

    def prediction(self, parents):
        return {'success': True, 'status': 'eos', 'tokens': self.codec.encode(parents).tokens.tolist()}

    def test_parent_comparison_uses_dfs_ids_not_source_indices(self):
        parents = [-1, 0, 0, 1]
        row = auxiliary_record(self.prediction(parents), self.codec, parents, asset_id='a')
        self.assertTrue(row['exact_parents'])
        self.assertEqual(row['target_parents'], [-1, 0, 1, 0])
        self.assertEqual(auxiliary_summary([row])['exact_parents'], 1.)

    def test_same_count_does_not_imply_same_structure(self):
        row = auxiliary_record(self.prediction([-1, 0, 0]), self.codec, [-1, 0, 1], asset_id='a')
        self.assertTrue(row['success'])
        self.assertTrue(row['exact_count'])
        self.assertFalse(row['exact_parents'])

    def test_natural_eos_must_also_parse_and_hitmax_is_failure(self):
        for prediction in ({'success': True, 'status': 'eos', 'tokens': [1, 2]},
                           {'success': False, 'status': 'max_tokens', 'tokens': [1, 4, 2]}):
            row = auxiliary_record(prediction, self.codec, [-1], asset_id='a')
            self.assertFalse(row['success'])
            self.assertFalse(row['exact_parents'])

    def test_hash_covers_geometry_targets_dtype_and_order(self):
        item = {'surface': torch.tensor([1., 2.]), 'parents': np.array([-1, 0]), 'id': ['a', 'b']}
        digest = tree_fingerprint(item)
        self.assertEqual(digest, tree_fingerprint(dict(reversed(list(item.items())))))
        for key, value in [('surface', torch.tensor([1., 3.])), ('surface', torch.tensor([1., 2.], dtype=torch.float64)),
                           ('parents', np.array([-1, 1])), ('id', ['b', 'a'])]:
            self.assertNotEqual(digest, tree_fingerprint({**item, key: value}))

    def test_only_intentional_comparison_differences_are_excluded(self):
        root = ROOT.parent / 'experiments'
        a = json.loads((root / 'fusion_a_fit8_v1.json').read_text())
        b = json.loads((root / 'fusion_b_fit8_v1.json').read_text())
        self.assertEqual(comparison_contract(a), comparison_contract(b))
        b['optimization']['peak_lr'] *= 2
        self.assertNotEqual(comparison_contract(a), comparison_contract(b))

    def test_pair_ready_rejects_changed_input_and_self_peer(self):
        from types import SimpleNamespace
        def write_json(path, value):
            path.write_text(json.dumps(value))
        module = SimpleNamespace(write_json=write_json)
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, {'virtual_query_xyz_runtime': module}):
            a, b = Path(temporary) / 'a', Path(temporary) / 'b'
            a.mkdir()
            b.mkdir()
            cfg = {'architecture': {}, 'runtime': {}, 'probe': {}}
            initial = {'full_initial_state_sha256': 'weights', 'code_commit': 'source'}
            items = [({}, {}, {'asset_id': ['a']})]
            peer = {'full_initial_state_sha256': 'weights', 'code_commit': 'source',
                    'inputs_sha256': tree_fingerprint(items), 'assets': ['a'],
                    'comparison_contract_sha256': comparison_contract(cfg)}
            write_json(b / 'pair_ready.json', peer)
            self.assertEqual(pair_ready(a, b, initial, items, cfg), peer)
            with self.assertRaises(ValueError):
                pair_ready(a, a, initial, items, cfg)
            peer['inputs_sha256'] = 'wrong'
            write_json(b / 'pair_ready.json', peer)
            with self.assertRaisesRegex(RuntimeError, 'different initial weights or actual inputs'):
                pair_ready(a, b, initial, items, cfg)

    def test_fit_rejects_old_paths_resume_weights_missing_aux_and_unmatched_budget(self):
        original = json.loads((ROOT.parent / 'experiments/fusion_a_fit8_v1.json').read_text())
        validation['validate_config'](original, 'fit')
        changes = [('runtime', 'repo', '/old/code'), ('initialization', 'checkpoint', '/old/checkpoint.pt'),
                   ('probe', 'auxiliary_weight', 0.), ('probe', 'evaluate_auxiliary', False),
                   ('probe', 'assets_per_update', 1), ('probe', 'max_updates', 128),
                   ('probe', 'early_stop', True), ('probe', 'fingerprint_initial_state', False),
                   ('probe', 'paired_output', '/old/output'), ('optimization', 'schedule_samples', 1024),
                   ('optimization', 'schedule', 'constant'), ('architecture', 'frame_residual_scale', 1.)]
        for section, key, value in changes:
            changed = json.loads(json.dumps(original))
            changed[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                validation['validate_config'](changed, 'fit')


if __name__ == '__main__':
    unittest.main()
