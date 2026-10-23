"""CPU checks for the fresh single-target data, budget and launch contracts."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('RIGWEAVE_DISABLE_OPEN3D', '1')
os.environ.setdefault('RIGWEAVE_DISABLE_LIGHTNING_IMPORT', '1')
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'model_training/rigweave/src'))
sys.path.insert(0, str(ROOT / 'model_training/rigweave/scripts'))

import numpy as np
import torch
from torch import nn

import legacy_single_target_runtime as rt
from train_legacy_single_target import require_job_membership, validate_release
from virtual_fusion_normal_runtime import MatchedFrameStream
from rigweave.dynamic_rig.data import select_mesh_frame_ids


CONFIG = json.loads((ROOT / 'model_training/experiments/legacy_single_target_100000_v1.json').read_text())


def drain(stream):
    result = []
    while stream.cursor < stream.local_total:
        result.extend(stream.take())
    return result


class SingleTargetRuntimeTests(unittest.TestCase):
    def test_config_preserves_declared_contract(self):
        self.assertEqual(rt.validate_config(CONFIG), list(range(5000, 100001, 5000)))
        self.assertFalse(CONFIG['formal_release']['approved'])
        self.assertFalse(CONFIG['runtime']['require_scheduler_membership'])
        self.assertEqual(CONFIG['runtime']['execution_mode'], 'explicit_ssh_exception')

    def test_config_rejects_silent_recipe_changes(self):
        changes = [('target_policy', 'second_random_target'), ('pruning', {'enabled': True}),
                   ('global_rotation_augmentation', True)]
        for key, value in changes:
            config = deepcopy(CONFIG)
            config[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                rt.validate_config(config)
        for section, key, value in [('data', 'normalization', 'query_bbox'),
                                    ('data', 'frames_min', 1), ('optimization', 'auxiliary_weight', .1),
                                    ('training', 'target_assets', 100032), ('architecture', 'motion_depth', 8)]:
            config = deepcopy(CONFIG)
            config[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                rt.validate_config(config)

    def test_budget_is100k_assets_and2100_updates(self):
        stream = rt.make_stream(CONFIG, 0)
        report = stream.plan_report()
        self.assertEqual(len(stream.update_sizes), 2100)
        self.assertEqual(int(sum(stream.update_sizes)), 100000)
        self.assertGreater(report['supervised_frame_exposures'], 100000)
        self.assertTrue(np.all((stream.frame_counts >= 2) & (stream.frame_counts <= 24)))

    def test_original_B20k_sampling_prefix_is_preserved(self):
        for rank in (0, 1):
            old = MatchedFrameStream(range(5000), rank=rank, world_size=2, seed=20261013,
                target_assets=20000, effective_assets=48, frame_policy=CONFIG['training']['frame_policy'],
                manifest_sha256=rt.TRAIN_SHA256, update_boundaries=[5000, 10000, 15000, 20000])
            new = rt.make_stream(CONFIG, rank)
            prefix = drain(old)
            actual = []
            while len(actual) < len(prefix):
                actual.extend(new.take(len(prefix) - len(actual)))
            self.assertEqual(prefix, actual)

    def test_stream_restore_preserves_visit_seeds_and_frames(self):
        original = rt.make_stream(CONFIG, 0)
        for _ in range(37):
            original.take()
        state = original.state_dict()
        restored = MatchedFrameStream(range(5000), rank=0, world_size=2, seed=20261013,
            target_assets=100000, effective_assets=48, frame_policy=CONFIG['training']['frame_policy'],
            manifest_sha256=rt.TRAIN_SHA256, update_boundaries=CONFIG['training']['checkpoint_assets'], state=state)
        for _ in range(50):
            self.assertEqual(original.take(), restored.take())

    def test_slot0_is_a_fresh_random_raw_pose_not_fixed_frame(self):
        vertices = np.arange(32*12*3, dtype=np.float32).reshape(32, 12, 3) / 100
        chosen = set()
        for seed in range(40):
            ids = select_mesh_frame_ids(vertices, frame_count=8, path=Path('synthetic.npz'), index=0,
                random_query=True, seed=20260529, sampling_seed=seed, motion_fps_ratio=.75,
                motion_vertex_samples=12, minimum_random_frames=1)
            expected = int(np.random.default_rng(seed).integers(0, 32))
            self.assertEqual(int(ids[0]), expected)
            self.assertEqual(len(set(ids)), 8)
            chosen.add(expected)
        self.assertGreater(len(chosen), 10)
        surface = {'query_points': torch.randn(3, 8, 4, 3)}
        before = surface['query_points'].clone()
        self.assertEqual(rt.train_target_indices(surface).tolist(), [0, 0, 0])
        self.assertTrue(torch.equal(before, surface['query_points']))

    def test_learning_rate_replays_actual_B_boundaries(self):
        optimizer = torch.optim.AdamW([nn.Parameter(torch.zeros(1))], lr=2e-5, betas=(.95, .999))
        self.assertAlmostEqual(rt.set_learning_rate(optimizer, 0, CONFIG)[0], 2e-5)
        self.assertAlmostEqual(rt.set_learning_rate(optimizer, 2000, CONFIG)[0], 1e-4)
        end_lr, end_beta = rt.set_learning_rate(optimizer, 19992, CONFIG)
        self.assertLess(end_lr, 2.01e-6)
        self.assertEqual(rt.set_learning_rate(optimizer, 20000, CONFIG), (1e-4, end_beta))
        self.assertEqual(rt.set_learning_rate(optimizer, 59952, CONFIG), (1e-4, end_beta))
        self.assertEqual(rt.set_learning_rate(optimizer, 60000, CONFIG), (3e-5, end_beta))
        self.assertEqual(rt.set_learning_rate(optimizer, 99992, CONFIG), (3e-5, end_beta))
        with self.assertRaises(ValueError):
            rt.set_learning_rate(optimizer, 100000, CONFIG)

    def test_ordinary_SSH_or_wrong_job_fails_closed(self):
        require_job_membership('12:devices:/AGE/130309880.1\n7:cpuset:/AGE/130309880.1\n', 130309880)
        for text in ('12:devices:/user.slice\n7:cpuset:/\n',
                     '12:devices:/AGE/130309880.1\n7:cpuset:/\n',
                     '12:devices:/AGE/130309881.1\n7:cpuset:/AGE/130309881.1\n'):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                require_job_membership(text, 130309880)

    def test_unreleased_or_unmeasured_training_is_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_release(CONFIG, rt)
        changed = deepcopy(CONFIG)
        changed['formal_release']['approved'] = True
        with self.assertRaises(RuntimeError):
            validate_release(changed, rt)

    def test_release_requires_correct_stage_and_two_complete_rank_receipts(self):
        def receipt(mode, config):
            contract = rt.contract_sha256(config)
            ranks = []
            for rank in (0, 1):
                if mode == 'preflight':
                    cases = [{'frames': frames, 'teacher_generation_first_token_equal': True,
                              'updates': [{}, {}]} for frames in config['preflight']['frames']]
                else:
                    cases = [{**case, 'passed': True,
                        'covered_stress_groups': ['longest_main'] + [f'geometry_t{n}' for n in range(2, 25)],
                        'measurements': [{'accumulation_microbatches': 2, 'peak_allocated_bytes': 1000}]*2}
                        for case in config['capacity']['cases']]
                ranks.append({'rank': rank, 'complete': True, 'passed': True, 'source': 'HEAD',
                              'contract_sha256': contract, 'diagnostic_weights_discarded': True,
                              'cases': cases, 'gpu_total_bytes': 100000})
            return {'mode': mode, 'complete': True, 'passed': True, 'world_size': 2,
                    'contract_sha256': contract, 'code_commit': 'HEAD', 'ranks': ranks}

        def check(mutate=None):
            config = deepcopy(CONFIG)
            config['formal_release']['approved'] = True
            values = {mode: receipt(mode, config) for mode in ('preflight', 'capacity')}
            if mutate:
                mutate(values)
            with tempfile.TemporaryDirectory() as folder:
                for mode, value in values.items():
                    path = Path(folder) / f'{mode}.json'
                    path.write_text(json.dumps(value))
                    config[mode]['receipt'] = {'path': str(path), 'sha256': rt.sha256(path)}
                with patch('train_legacy_single_target.subprocess.run'):
                    validate_release(config, rt)

        check()
        mutations = [
            lambda v: v['preflight'].update(mode='capacity'),
            lambda v: v['capacity'].update(ranks=[]),
            lambda v: v['capacity']['ranks'][1].update(rank=0),
            lambda v: v['preflight']['ranks'][1].update(complete=False),
            lambda v: v['preflight']['ranks'][0]['cases'][0].update(teacher_generation_first_token_equal=False),
            lambda v: v['capacity']['ranks'][0]['cases'].pop(),
            lambda v: v['capacity']['ranks'][0]['cases'][0]['measurements'][0].update(peak_allocated_bytes=90000),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                check(mutation)

    def test_training_contract_binds_targets_but_not_receipt_publication(self):
        original = rt.contract_sha256(CONFIG)
        changed = deepcopy(CONFIG)
        changed['formal_release']['approved'] = True
        changed['capacity']['receipt'] = {'path': 'new.json', 'sha256': '0'*64}
        self.assertEqual(original, rt.contract_sha256(changed))
        changed['target_policy'] = 'all_frames'
        self.assertNotEqual(original, rt.contract_sha256(changed))

    def test_evaluation_reuses_surface_but_runs_each_real_query(self):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = object()
                self.calls = []

            def encode_surface(self, surface):
                self.calls.append('surface')
                return surface['tokens']

            def condition_from_tokens(self, surface, tokens, indices):
                self.calls.append(indices.tolist())
                return tokens[torch.arange(len(indices)), indices]

        model = Tiny()
        adapter = rt.EvaluationAdapter(model)
        tokens = torch.randn(2, 4, 3, 5)
        output, virtual = adapter.encode({'tokens': tokens})
        self.assertTrue(torch.equal(output, tokens))
        self.assertIsNone(virtual)
        self.assertEqual(model.calls, ['surface', [0, 0], [1, 1], [2, 2], [3, 3]])


if __name__ == '__main__':
    unittest.main()
