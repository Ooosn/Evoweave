import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'model_training/rigweave/scripts'))
from anchor_full_data import make_stream, planned_microbatches, stress_groups
from anchor_full_protocol import (TRAIN_ROWS, full_checkpoints_to_keep, scientific_sha,
    validate_config, validate_resume)
from virtual_frame_runtime import exposure_learning_rate
from virtual_pair_runtime import next_update_batches, target_selection


class FullJointTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT/'model_training/experiments/anchor_full_20261030_v1.json').read_text())
        self.base = validate_config(self.config)

    def test_declared_full_data_official_initialization_and300k_curve(self):
        self.assertEqual(self.base['data']['manifests']['train']['rows'], 18002)
        self.assertEqual(self.base['optimization']['schedule_samples'], 300000)
        self.assertEqual(self.config['training']['learning_rate_asset_offset'], 0)
        for visits, expected in [(0, 2e-5), (30000, 1e-4), (300000, 2e-6)]:
            self.assertAlmostEqual(exposure_learning_rate(visits, 300000)[0], expected)

    def test_project_checkpoint_or_old_optimizer_is_rejected(self):
        for key, value in [('learning_rate_asset_offset', 20000), ('optimizer', 'retain_pretrained_Adam'),
                           ('additional_assets', 100000), ('ar_enabled', False)]:
            bad = copy.deepcopy(self.config)
            bad['training'][key] = value
            with self.assertRaises(ValueError):
                validate_config(bad)
        self.config['backbone']['checkpoint'] = {'path': 'old.pt'}
        with self.assertRaises(ValueError):
            validate_config(self.config)

    def test_sampler_covers_all18002_in_every_full_pass(self):
        stream = make_stream(self.base, 0, 2)
        self.assertEqual(len(stream.order), 300000)
        for start in range(0, 300000, TRAIN_ROWS):
            part = stream.order[start:start+TRAIN_ROWS]
            self.assertEqual(len(set(part.tolist())), len(part))
            if len(part) == TRAIN_ROWS:
                np.testing.assert_array_equal(np.sort(part), np.arange(TRAIN_ROWS))
        self.assertEqual(len(stream.update_sizes), 6270)
        self.assertEqual(int(stream.update_sizes.sum()), 300000)

    def test_two_rank_consumption_and_exact_resume(self):
        streams = [make_stream(self.base, rank, 2) for rank in range(2)]
        samples = 0
        for _ in range(6):
            consumed = []
            for stream in streams:
                goal, batches = next_update_batches(stream, samples, 2)
                consumed += [r.item_index for batch in batches for r in batch]
            self.assertEqual(consumed, streams[0].order[samples:samples+goal].tolist())
            samples += goal
        for stream in streams:
            restored = make_stream(self.base, stream.rank, 2, stream.state_dict())
            self.assertEqual(next_update_batches(restored, samples, 2), next_update_batches(stream, samples, 2))

    def test_prefetch_planning_never_advances_consumed_cursor(self):
        stream = make_stream(self.base, 0, 2)
        before = stream.state_dict()
        planned = list(planned_microbatches(self.base, stream, 48))
        self.assertEqual(before, stream.state_dict())
        self.assertEqual(planned, next_update_batches(stream, 0, 2)[1])

    def test_full_base_uses_fractional_not_single_target_supervision(self):
        stream = make_stream(self.base, 0, 2)
        _, batches = next_update_batches(stream, 0, 2)
        for batch in batches:
            selected = target_selection(self.base, batch)
            self.assertTrue(all(len(slots) >= 2 and len(set(slots)) == len(slots) for slots in selected))

    def test_resume_rejects_nonmatching_dataset_source_or_cursor(self):
        streams = [make_stream(self.base, rank, 2) for rank in range(2)]
        for stream in streams:
            next_update_batches(stream, 0, 2)
        saved = {'scientific_sha256': scientific_sha(self.config), 'optimizer_parameter_names': ['x'],
            'source_sha256': {'test': 'digest'}, 'initialization': {'project_checkpoint_loaded': False},
            'global_backbone_asset_exposures': 48, 'additional_asset_exposures': 48,
            'rank_states': [{'rank': rank, 'stream': stream.state_dict()} for rank, stream in enumerate(streams)]}
        validate_resume(self.config, saved, ['x'], {'test': 'digest'})
        for field, value in [('scientific_sha256', 'bad'), ('source_sha256', {}),
                             ('global_backbone_asset_exposures', 20048)]:
            bad = copy.deepcopy(saved)
            bad[field] = value
            with self.assertRaises(ValueError):
                validate_resume(self.config, bad, ['x'], {'test': 'digest'})
        saved['rank_states'][0]['stream']['cursor'] = 0
        with self.assertRaises(ValueError):
            validate_resume(self.config, saved, ['x'], {'test': 'digest'})

    def test_retention_keeps_latest_two_and100k_milestones(self):
        retention = self.config['artifacts']['retention']
        self.assertEqual(full_checkpoints_to_keep([10000, 20000, 30000], retention), {20000, 30000})
        self.assertEqual(full_checkpoints_to_keep([100000, 110000, 120000, 130000], retention),
                         {100000, 120000, 130000})

    def test_stress_selection_uses_full_manifest_extremes(self):
        rows = [{'canonical_metrics': {'target_joint_count': 10, 'filtered_vertex_count': 100,
                                      'filtered_face_count': 200}} for _ in range(TRAIN_ROWS)]
        rows[-1]['canonical_metrics']['target_joint_count'] = 250
        rows[-2]['canonical_metrics']['filtered_vertex_count'] = 299563
        groups, counts = stress_groups(rows)
        self.assertEqual(groups['largest_joints'][0], TRAIN_ROWS-1)
        self.assertEqual(groups['largest_geometry'][0], TRAIN_ROWS-2)
        self.assertEqual(max(counts), 250)

    def test_fresh_joint_initialization_never_loads_an_optimizer(self):
        import anchor_full_runtime as runtime
        with patch.object(runtime, 'build_model', return_value=(nn.Linear(2, 2), None, None, {'official': True})) as build, \
                patch.object(runtime, 'VirtualPathAblationRig', side_effect=lambda model, **kwargs: model), \
                patch.object(runtime, 'AnchorJointBinding', side_effect=lambda *args: nn.Linear(2, 2)), \
                patch.object(torch.optim.AdamW, 'load_state_dict', side_effect=AssertionError('old Adam loaded')):
            model, optimizer, _, _, _, initial = runtime.initialize(self.config, torch.device('cpu'))
        self.assertFalse(optimizer.state)
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertTrue(model.ar_enabled)
        self.assertFalse(initial['project_checkpoint_loaded'])
        self.assertFalse(initial['project_optimizer_loaded'])
        self.assertEqual(initial['initial_asset_exposures'], 0)
        self.assertNotIn('load_official_weights', build.call_args.kwargs)

    def test_capacity_uses_maximum_allowed_AR_density(self):
        from anchor_full_runtime import capacity_requests
        for frames in (6, 12, 24):
            requests = capacity_requests(self.base, [10, 20, 30], 0, frames, 6, 100000)
            self.assertEqual([len(s) for s in target_selection(self.base, requests)],
                             [max(2, int(.4*frames))]*6)


if __name__ == '__main__':
    unittest.main()
