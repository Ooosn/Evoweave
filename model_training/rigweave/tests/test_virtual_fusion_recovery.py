"""CPU proofs for the bounded interrupted B60 recovery contract."""
from copy import deepcopy
import json
from pathlib import Path
import random
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'model_training/rigweave/src'))
sys.path.insert(0, str(ROOT / 'model_training/rigweave/scripts'))
from virtual_fusion_normal_runtime import MatchedFrameStream
from virtual_fusion_recovery import (
    B20_CHECKPOINT,
    B20_SOURCE_CONFIG,
    B40_CHECKPOINT,
    B40_SOURCE_CONFIG,
    BASE_INITIAL_STATE_SHA256,
    CHECKPOINT,
    INITIAL_CONFIG_SHA256,
    INITIAL_SOURCE_COMMIT,
    MILESTONES,
    ORIGINAL_PLAN_SHA256,
    PLAN_SHA256,
    PURPOSE,
    RECOVERY,
    RECOVERY_RUNTIME,
    SOURCE_COMMIT,
    SOURCE_CONFIG,
    TARGET_ASSETS,
    TARGET_FRAMES,
    TARGET_UPDATES,
    validate_recovery_config,
    validate_recovery_payload,
    restore_stream,
)


def source_config():
    path = ROOT / 'model_training/experiments/fusion_normal_b_extend60000_v1.json'
    return json.loads(path.read_text(encoding='utf-8'))


def recovery_config(source):
    config = deepcopy(source)
    config['runtime'].update(deepcopy(RECOVERY_RUNTIME))
    config['training']['checkpoint_assets'] = list(MILESTONES)
    config['training']['evaluation_assets'] = list(MILESTONES)
    config['recovery'] = deepcopy(RECOVERY)
    return config


def make_stream_state(config, rank, cursor=25000):
    data = config['data']
    training = config['training']
    stream = MatchedFrameStream(
        range(data['manifests']['train']['rows']),
        rank=rank,
        world_size=2,
        seed=data['stream_seed'],
        target_assets=training['target_assets'],
        effective_assets=training['effective_assets_per_update'],
        frame_policy=training['frame_policy'],
        manifest_sha256=data['manifests']['train']['sha256'],
        min_frames=data['frames_min'],
        max_frames=data['frames_max'],
        update_boundaries=list(range(5000, TARGET_ASSETS + 1, 5000)),
    )
    state = stream.state_dict()
    state['cursor'] = cursor
    return state


def valid_rng_state(seed):
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)
    return {
        'torch_rng': generator.get_state(),
        'cuda_rng': torch.tensor([seed % 255, 1, 2, 3], dtype=torch.uint8),
        'numpy_rng': np.random.RandomState(seed).get_state(),
        'python_rng': random.Random(seed).getstate(),
    }


def valid_optimizer():
    return {
        'state': {
            index: {
                'step': torch.tensor(float(1050), dtype=torch.float32),
                'exp_avg': torch.zeros(1, dtype=torch.float32),
                'exp_avg_sq': torch.ones(1, dtype=torch.float32),
            }
            for index in range(947)
        },
        'param_groups': [{
            'lr': 1e-4,
            'betas': (0.9499999512612207, 0.999),
            'eps': 1e-8,
            'weight_decay': 0,
        }],
    }


def valid_initialization():
    return {
        'route': 'b',
        'source': INITIAL_SOURCE_COMMIT,
        'config_sha256': INITIAL_CONFIG_SHA256,
        'full_base_initial_state_sha256': BASE_INITIAL_STATE_SHA256,
        'shared_initial_state_sha256': BASE_INITIAL_STATE_SHA256,
        'exposure_plan_sha256': 'a7bdf57c23a9edb0bcac5a79a4603003e8387d613cd0dbac7928255f612dca7b',
        'training_continuation': {
            'checkpoint': deepcopy(B40_CHECKPOINT),
            'source_config': deepcopy(B40_SOURCE_CONFIG),
            'route': 'b',
            'source': SOURCE_COMMIT,
            'config_sha256': SOURCE_CONFIG['sha256'],
            'start_asset_exposures': 40000,
            'target_asset_exposures': TARGET_ASSETS,
            'start_updates': 840,
            'target_updates': TARGET_UPDATES,
            'start_supervised_frame_exposures': 528296,
            'target_supervised_frame_exposures': TARGET_FRAMES,
            'original_exposure_plan_sha256': ORIGINAL_PLAN_SHA256,
            'extended_exposure_plan_sha256': PLAN_SHA256,
            'constant_learning_rate': 1e-4,
            'optimizer_betas': 'restore_checkpoint',
            'optimizer_state_restore_required': True,
            'fresh_official_initialization': False,
            'resumed_trained_weights_loaded': True,
            'original_20000_prefix_verified': True,
            'saved_prefix_asset_exposures_verified': 40000,
            'rng_policy': 'restore_saved_checkpoint_before_milestone_evaluation',
            'replayed_initial_evaluation': False,
            'quality_acceptance': False,
        },
        'training_continuation_history': [{
            'checkpoint': deepcopy(B20_CHECKPOINT),
            'source_config': deepcopy(B20_SOURCE_CONFIG),
            'route': 'b',
            'source': B40_CHECKPOINT['source'],
            'config_sha256': B40_SOURCE_CONFIG['sha256'],
            'start_asset_exposures': 20000,
            'target_asset_exposures': 40000,
            'start_updates': 420,
            'target_updates': 840,
            'start_supervised_frame_exposures': 263608,
            'target_supervised_frame_exposures': 528296,
            'original_exposure_plan_sha256': 'a7bdf57c23a9edb0bcac5a79a4603003e8387d613cd0dbac7928255f612dca7b',
            'extended_exposure_plan_sha256': ORIGINAL_PLAN_SHA256,
            'constant_learning_rate': 1e-4,
            'optimizer_betas': 'restore_checkpoint',
            'optimizer_state_restore_required': True,
            'fresh_official_initialization': False,
            'resumed_trained_weights_loaded': True,
            'original_20000_prefix_verified': True,
            'quality_acceptance': False,
        }],
    }


def payload_fixture():
    config = source_config()
    return {
        'schema': 'virtual_query_xyz_aux_ids_v1',
        'purpose': PURPOSE,
        'source': SOURCE_COMMIT,
        'config': deepcopy(config),
        'architecture': {'dim': 1024},
        'model': {'tiny': torch.zeros(1)},
        'optimizer': valid_optimizer(),
        'asset_exposures': 50000,
        'step': 1050,
        'supervised_frame_exposures': 657056,
        'rank_states': [
            {'stream': make_stream_state(config, rank), **valid_rng_state(100 + rank)}
            for rank in (0, 1)
        ],
        'initialization': valid_initialization(),
    }


def request_signature(requests):
    return [(request.item_index, request.frames, request.sampling_seed) for request in requests]


class RecoveryTests(unittest.TestCase):
    def test_recovery_config_allows_only_the_pinned_delta(self):
        original = source_config()
        config = recovery_config(original)
        self.assertEqual(validate_recovery_config(config, original, 2), [55000, 60000])
        for mutate in (
            lambda value: value['runtime'].__setitem__('allocation', 130309881),
            lambda value: value['runtime'].__setitem__('host', 'pcg02i'),
            lambda value: value['runtime'].__setitem__('physical_gpus', [5, 7]),
            lambda value: value['capacity'].__setitem__('allocated_fraction_limit', 0.9),
            lambda value: value['training'].__setitem__('target_assets', 60001),
            lambda value: value['evaluation'].__setitem__('initial', True),
            lambda value: value.__setitem__('purpose', 'wrong'),
        ):
            changed = recovery_config(original)
            mutate(changed)
            with self.assertRaises(ValueError):
                validate_recovery_config(changed, original, 2)

    def test_validates_full_payload_and_preserves_input_history(self):
        payload = payload_fixture()
        before_history = deepcopy(payload['initialization']['training_continuation_history'])
        streams = validate_recovery_payload(payload, source_config())
        self.assertEqual(len(streams), 2)
        self.assertEqual([stream.cursor for stream in streams], [25000, 25000])
        self.assertEqual(payload['initialization']['training_continuation_history'], before_history)

    def test_restored_stream_uses_fresh_60k_plan_without_replaying_50k_prefix(self):
        payload = payload_fixture()
        restored = restore_stream(payload, 0)
        config = source_config()
        fresh = MatchedFrameStream(
            range(5000),
            rank=0,
            world_size=2,
            seed=config['data']['stream_seed'],
            target_assets=TARGET_ASSETS,
            effective_assets=config['training']['effective_assets_per_update'],
            frame_policy=config['training']['frame_policy'],
            manifest_sha256=config['data']['manifests']['train']['sha256'],
            min_frames=2,
            max_frames=24,
            update_boundaries=list(range(5000, TARGET_ASSETS + 1, 5000)),
        )
        while fresh.cursor < 25000:
            fresh.take()
        self.assertEqual(fresh.cursor, 25000)
        np.testing.assert_array_equal(restored.order[:25000], fresh.order[:25000])
        np.testing.assert_array_equal(restored.frame_counts[:1050], fresh.frame_counts[:1050])
        np.testing.assert_array_equal(restored.update_sizes[:1050], fresh.update_sizes[:1050])
        self.assertEqual(
            request_signature(restored.take(max_assets=3)),
            request_signature(fresh.take(max_assets=3)),
        )

    def test_rejects_payload_progress_config_rng_moment_step_and_plan_tampering(self):
        mutations = (
            lambda value: value.__setitem__('asset_exposures', 49999),
            lambda value: value['config']['training'].__setitem__('target_assets', 60001),
            lambda value: value['rank_states'][0]['stream'].__setitem__('plan_sha256', '0' * 64),
            lambda value: value['rank_states'][1].pop('numpy_rng'),
            lambda value: value['optimizer']['param_groups'][0].__setitem__('lr', 3e-5),
            lambda value: value['optimizer']['state'][0].pop('exp_avg'),
            lambda value: value['optimizer']['state'][1]['step'].fill_(1049),
        )
        for mutate in mutations:
            changed = deepcopy(payload_fixture())
            mutate(changed)
            with self.assertRaises(ValueError):
                validate_recovery_payload(changed, source_config())

    def test_rejects_stream_seed_and_cursor_tampering(self):
        for field, value in (('seed', 20261014), ('target_assets', 60001)):
            changed = deepcopy(payload_fixture())
            changed['rank_states'][0]['stream']['contract'][field] = value
            with self.assertRaises(ValueError):
                validate_recovery_payload(changed, source_config())
        changed = deepcopy(payload_fixture())
        changed['rank_states'][1]['stream']['cursor'] = 24999
        with self.assertRaises(ValueError):
            validate_recovery_payload(changed, source_config())

    def test_rng_validation_does_not_mutate_global_states(self):
        payload = payload_fixture()
        torch_before = torch.get_rng_state()
        numpy_before = np.random.get_state()
        python_before = random.getstate()
        validate_recovery_payload(payload, source_config())
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_before[1])
        self.assertEqual(random.getstate(), python_before)


if __name__ == '__main__':
    unittest.main()

