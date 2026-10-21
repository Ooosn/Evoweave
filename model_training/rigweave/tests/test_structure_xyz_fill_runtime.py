from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch import nn

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from rigweave.dynamic_rig.frame_batch_sampler import FrameSampleRequest
from structure_xyz_fill_runtime import (
    make_stream, matched_trial_fingerprint, optimizer_for, pruning_choices,
    set_learning_rate, validate_config,
)
from train_structure_xyz_fill import published_run, optimizer_parameter_names


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[2] / 'experiments/structure_xyz_fill_random_512_v1.json'
        self.config = json.loads(path.read_text())

    def test_declared_baseline_and_embedding_trial_are_matched(self):
        self.assertEqual(validate_config(self.config), [128, 256, 512])
        other = deepcopy(self.config)
        other['fill_initialization']['kind'] = 'structure_and_position'
        validate_config(other)
        self.assertEqual(matched_trial_fingerprint(self.config), matched_trial_fingerprint(other))
        other['data']['stream_seed'] += 1
        self.assertNotEqual(matched_trial_fingerprint(self.config), matched_trial_fingerprint(other))

    def test_config_refuses_membership_and_normalization_drift(self):
        changes = [lambda c: c['data']['manifests']['train'].update(rows=4999),
                   lambda c: c['data'].update(coordinate_clipping=True),
                   lambda c: c['quantization'].update(bins=128),
                   lambda c: c['fill_architecture'].update(fill_depth=1),
                   lambda c: c['training'].update(effective_assets_per_update=24),
                   lambda c: c['optimization'].update(warmup_assets=-1)]
        for change in changes:
            value = deepcopy(self.config)
            change(value)
            with self.assertRaises(ValueError):
                validate_config(value)

    def test_pruning_gate_is_global_and_visits_ignore_microbatch_partition(self):
        requests = [FrameSampleRequest(index, 8, (20261013 << 32) + index) for index in range(8)]
        counts = [52, 31, 27, 1, 100, 4, 16, 256]
        step = next(step for step in range(100) if pruning_choices(counts, requests, self.config, step, enabled=True)['batch_gate'])
        whole = pruning_choices(counts, requests, self.config, step, enabled=True)
        left = pruning_choices(counts[:3], requests[:3], self.config, step, enabled=True)
        right = pruning_choices(counts[3:], requests[3:], self.config, step, enabled=True)
        self.assertTrue(left['batch_gate'] and right['batch_gate'])
        for name in ('remove_counts', 'prune_seeds', 'fractions'):
            self.assertEqual(whole[name], left[name] + right[name])
        self.assertEqual(whole['remove_counts'][3], 0)
        self.assertTrue(all(0 <= removed <= int(.2 * (count - 1))
                            for count, removed in zip(counts, whole['remove_counts'])))
        other_visits = [FrameSampleRequest(request.item_index, 8, request.sampling_seed + 5000) for request in requests]
        different = pruning_choices(counts, other_visits, self.config, step, enabled=True)
        self.assertNotEqual(whole['prune_seeds'], different['prune_seeds'])

    def test_no_pruning_for_evaluation_and_gate_rate_is_not_a_fixed_asset_cohort(self):
        requests = [FrameSampleRequest(0, 8, 125)]
        disabled = pruning_choices([52], requests, self.config, 0, enabled=False)
        self.assertEqual(disabled['remove_counts'], [0])
        gates = [pruning_choices([52], requests, self.config, step, enabled=True)['batch_gate'] for step in range(1000)]
        self.assertGreater(sum(gates), 150)
        self.assertLess(sum(gates), 250)
        self.assertTrue(any(gates) and not all(gates))

    def test_exposure_plan_matches_ranks_and_completed_extension_keeps_prefix(self):
        first = make_stream(self.config, range(5000), rank=0, world=2)
        second = make_stream(self.config, range(5000), rank=1, world=2)
        self.assertEqual(first.plan_sha256, second.plan_sha256)
        self.assertEqual(first.global_ends[-1], 512)
        self.assertTrue(set((128, 256, 512)) <= set(first.global_ends.tolist()))
        while first.cursor < first.local_total:
            first.take()
        extended_config = deepcopy(self.config)
        extended_config['training'].update(target_assets=2048, evaluation_assets=[128, 256, 512, 1024, 2048],
                                            checkpoint_assets=[128, 256, 512, 1024, 2048])
        extended = make_stream(extended_config, range(5000), rank=0, world=2, state=first.state_dict(), extend=True)
        self.assertEqual(extended.cursor, 256)
        self.assertTrue(np.array_equal(first.order, extended.order[:512]))
        self.assertTrue(np.array_equal(first.frame_counts, extended.frame_counts[:len(first.frame_counts)]))

    def test_optimization_groups_and_continuation_do_not_restart_warmup(self):
        model = nn.Module()
        for name in ('surface_tokenizer', 'motion_encoder', 'structure_decoder', 'fill_decoder'):
            setattr(model, name, nn.Linear(2, 2))
        optimizer = optimizer_for(model, self.config)
        names = optimizer_parameter_names(model, optimizer)
        self.assertEqual([row['group'] for row in names], ['shared', 'structure', 'fill'])
        self.assertEqual(len({name for group in names for name in group['parameters']}), 8)
        first = set_learning_rate(optimizer, 0, self.config)
        self.assertAlmostEqual(first['fill'], 2e-5)
        middle = set_learning_rate(optimizer, 96, self.config)
        self.assertAlmostEqual(middle['fill'], 6e-5)
        self.assertEqual(set_learning_rate(optimizer, 512, self.config),
                         set_learning_rate(optimizer, 2048, self.config))

    def test_exact_published_mode_config_output_hash_is_required(self):
        operation = {'runs': [{'config': '/tmp/fill.json', 'output': '/tmp/fit', 'mode': 'preflight',
                               'config_sha256': 'abc', 'approved': True}]}
        self.assertEqual(published_run(operation, '/tmp/fill.json', '/tmp/fit', 'preflight', 'abc')['mode'], 'preflight')
        for mode, digest, output in [('train', 'abc', '/tmp/fit'), ('preflight', 'bad', '/tmp/fit'),
                                     ('preflight', 'abc', '/tmp/else')]:
            with self.assertRaises(ValueError):
                published_run(operation, '/tmp/fill.json', output, mode, digest)


if __name__ == '__main__':
    unittest.main()
