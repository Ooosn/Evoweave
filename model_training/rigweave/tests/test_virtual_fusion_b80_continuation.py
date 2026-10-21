"""CPU proofs for the recovered B60-to-B80 continuation contract."""
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
import virtual_fusion_recovery as recovery
from virtual_fusion_continuation import (
    BASE_SHA256,
    B40_CHECKPOINT,
    B40_SOURCE_CONFIG,
    OLD_PLAN_SHA256,
    B60_CHECKPOINT,
    RECOVERY_INCOMING_MODEL_SHA256,
    B60_SOURCE_CONFIG,
    B80_PLAN_SHA256,
    B80_TARGET_FRAMES,
    B80_TARGET_UPDATES,
    B80_RUNTIME,
    extend_checkpoint_stream,
    continuation_contract,
    validate_checkpoint_payload,
    validate_continuation_config,
)
from virtual_fusion_recovery import B20_CHECKPOINT, B20_SOURCE_CONFIG


EXPERIMENTS = ROOT / 'model_training/experiments'
ORIGINAL_PATH = EXPERIMENTS / 'fusion_normal_b_v1.json'
SOURCE_PATH = EXPERIMENTS / 'fusion_normal_b_recover60000_20261021_v1.json'
CONFIG_PATH = EXPERIMENTS / 'fusion_normal_b_lr3e5_80000_v1.json'
ACTUAL_RECOVERY_PROVENANCE = Path(
    r'D:\evoweave\server_ops\outputs\virtual_fusion_20261013\recovery_20261021\full_terminal_v1\continuation_rank_0.json')


def load_configs():
    return tuple(json.loads(path.read_text(encoding='utf-8')) for path in
                 (ORIGINAL_PATH, SOURCE_PATH, CONFIG_PATH))


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
                'step': torch.tensor(float(1260), dtype=torch.float32),
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


def valid_stream_state(config, rank):
    stream = MatchedFrameStream(
        range(config['data']['manifests']['train']['rows']), rank=rank, world_size=2,
        seed=config['data']['stream_seed'], target_assets=60000,
        effective_assets=config['training']['effective_assets_per_update'],
        frame_policy=config['training']['frame_policy'],
        manifest_sha256=config['data']['manifests']['train']['sha256'],
        min_frames=config['data']['frames_min'], max_frames=config['data']['frames_max'],
        update_boundaries=list(range(5000, 60001, 5000)),
    )
    state = stream.state_dict()
    state['cursor'] = 30000
    return state


def actual_interrupted_recovery():
    """Use the accepted terminal provenance when this checkout has the receipt."""
    if ACTUAL_RECOVERY_PROVENANCE.exists():
        return json.loads(ACTUAL_RECOVERY_PROVENANCE.read_text(encoding='utf-8'))
    return {
        'checkpoint': deepcopy(recovery.CHECKPOINT),
        'source_config': deepcopy(recovery.SOURCE_CONFIG),
        'route': 'b',
        'source': 'da2ee8878a271efc87d1c219d88463ea574eee07',
        'config_sha256': '810af5796f7f150a7588af676b3a17173c2ab18b02ece271d47440b1cd0156d2',
        'mode': 'interrupted_resume',
        'start_asset_exposures': 50000,
        'target_asset_exposures': 60000,
        'start_updates': 1050,
        'target_updates': 1260,
        'start_supervised_frame_exposures': 657056,
        'target_supervised_frame_exposures': 788304,
        'exposure_plan_sha256': recovery.PLAN_SHA256,
        'checkpoint_model_state_sha256': RECOVERY_INCOMING_MODEL_SHA256,
        'restored_model_state_sha256': RECOVERY_INCOMING_MODEL_SHA256,
        'constant_learning_rate': 1e-4,
        'optimizer_state_restore_required': True,
        'optimizer_betas': 'restore_checkpoint',
        'rng_policy': recovery.RECOVERY['rng_policy'],
        'replayed_initial_evaluation': False,
        'sampling_prefix_preserved': True,
        'unsaved_exposures_to_replay': 4080,
        'historical_evaluations_preserved': '/home/wangyy/evorig/outputs/virtual_fusion_normal_20261014/normal_b_extend60000_v1',
        'fresh_official_initialization': False,
        'bitwise_future_trajectory_claimed': False,
        'quality_acceptance': False,
    }


def valid_initialization():
    return {
        'route': 'b',
        'source': recovery.INITIAL_SOURCE_COMMIT,
        'config_sha256': recovery.INITIAL_CONFIG_SHA256,
        'full_base_initial_state_sha256': BASE_SHA256,
        'shared_initial_state_sha256': BASE_SHA256,
        'exposure_plan_sha256': OLD_PLAN_SHA256,
        'training_continuation': {
            'checkpoint': deepcopy(B40_CHECKPOINT),
            'source_config': deepcopy(B40_SOURCE_CONFIG),
            'route': 'b',
            'source': recovery.SOURCE_COMMIT,
            'config_sha256': recovery.SOURCE_CONFIG['sha256'],
            'start_asset_exposures': 40000,
            'target_asset_exposures': 60000,
            'start_updates': 840,
            'target_updates': 1260,
            'start_supervised_frame_exposures': 528296,
            'target_supervised_frame_exposures': 788304,
            'original_exposure_plan_sha256': recovery.ORIGINAL_PLAN_SHA256,
            'extended_exposure_plan_sha256': recovery.PLAN_SHA256,
            'constant_learning_rate': 1e-4,
            'optimizer_betas': 'restore_checkpoint',
            'optimizer_state_restore_required': True,
            'fresh_official_initialization': False,
            'resumed_trained_weights_loaded': True,
            'original_20000_prefix_verified': True,
            'saved_prefix_asset_exposures_verified': 40000,
            'rng_policy': recovery.RECOVERY['rng_policy'],
            'replayed_initial_evaluation': False,
            'quality_acceptance': False,
        },
        'training_continuation_history': [{
            'checkpoint': deepcopy(B20_CHECKPOINT),
            'source_config': deepcopy(B20_SOURCE_CONFIG),
            'route': 'b',
            'source': recovery.B40_SOURCE_COMMIT,
            'config_sha256': recovery.B40_CONFIG_SHA256,
            'start_asset_exposures': 20000,
            'target_asset_exposures': 40000,
            'start_updates': 420,
            'target_updates': 840,
            'start_supervised_frame_exposures': 263608,
            'target_supervised_frame_exposures': 528296,
            'original_exposure_plan_sha256': OLD_PLAN_SHA256,
            'extended_exposure_plan_sha256': recovery.ORIGINAL_PLAN_SHA256,
            'constant_learning_rate': 1e-4,
            'optimizer_betas': 'restore_checkpoint',
            'optimizer_state_restore_required': True,
            'fresh_official_initialization': False,
            'resumed_trained_weights_loaded': True,
            'original_20000_prefix_verified': True,
            'quality_acceptance': False,
        }],
        'interrupted_recovery': actual_interrupted_recovery(),
    }


def payload_fixture():
    _, source, _ = load_configs()
    return {
        'schema': 'virtual_query_xyz_aux_ids_v1',
        'purpose': 'normal5000_continuation',
        'source': B60_CHECKPOINT['source'],
        'config': deepcopy(source),
        'architecture': {'dim': 1024},
        'model': {'tiny': torch.zeros(1)},
        'optimizer': valid_optimizer(),
        'asset_exposures': 60000,
        'step': 1260,
        'supervised_frame_exposures': 788304,
        'rank_states': [
            {'stream': valid_stream_state(source, rank), **valid_rng_state(100 + rank)}
            for rank in (0, 1)
        ],
        'initialization': valid_initialization(),
    }


def request_signature(requests):
    return [(request.item_index, request.frames, request.sampling_seed) for request in requests]


class B80ContinuationTests(unittest.TestCase):
    def test_config_is_exact_b80_delta_and_milestones(self):
        original, _, config = load_configs()
        self.assertEqual(validate_continuation_config(config, original, 2), [65000, 70000, 75000, 80000])
        contract = continuation_contract(config)
        self.assertEqual(contract['checkpoint'], B60_CHECKPOINT)
        self.assertEqual(contract['source_config'], B60_SOURCE_CONFIG)
        self.assertEqual(contract['old_plan_sha256'], recovery.PLAN_SHA256)
        self.assertEqual(contract['target_assets'], 80000)
        self.assertEqual(contract['start_updates'], 1260)
        self.assertEqual(contract['start_frames'], 788304)
        self.assertEqual(contract['learning_rate'], 3e-5)
        self.assertEqual(contract['runtime_override'], B80_RUNTIME)
        self.assertEqual(contract['predecessor_jobdir'], '/home/wangyy/jobs/virtual_fusion_recovery_20261021_v1')
        self.assertEqual(B80_PLAN_SHA256, 'f561a7d8432792920292cc1cf06f74858872b06173187d9ac3f132efdf72b7f2')
        self.assertEqual(B80_TARGET_UPDATES, 1680)
        self.assertEqual(B80_TARGET_FRAMES, 1041832)
        for mutate in (
            lambda value: value['runtime'].__setitem__('allocation', 130309881),
            lambda value: value['runtime'].__setitem__('host', 'pcg02i'),
            lambda value: value['runtime'].__setitem__('physical_gpus', [5, 7]),
            lambda value: value['runtime'].__setitem__('repo', '/wrong/checkout'),
            lambda value: value['training'].__setitem__('target_assets', 80001),
            lambda value: value['training'].__setitem__('checkpoint_assets', [65000, 70000, 75000, 80001]),
            lambda value: value['continuation'].__setitem__('learning_rate', 1e-4),
            lambda value: value['architecture'].__setitem__('fusion_depth', 2),
        ):
            changed = deepcopy(config)
            mutate(changed)
            with self.assertRaises(ValueError):
                validate_continuation_config(changed, original, 2)

    def test_valid_recovered_payload_checks_all_optimizer_entries_rng_and_streams(self):
        original, source, config = load_configs()
        payload = payload_fixture()
        before = deepcopy(payload['initialization'])
        validate_checkpoint_payload(payload, original, 2, config=config, source_config=source)
        self.assertEqual([state['stream']['cursor'] for state in payload['rank_states']], [30000, 30000])
        self.assertEqual([state['stream']['plan_sha256'] for state in payload['rank_states']],
                         [recovery.PLAN_SHA256] * 2)
        self.assertEqual(payload['initialization'], before)
        self.assertEqual(len(payload['optimizer']['state']), 947)
        self.assertTrue(all(value['step'].item() == 1260 for value in payload['optimizer']['state'].values()))

    def test_rejects_recovery_identity_provenance_optimizer_rng_and_stream_drift(self):
        original, source, config = load_configs()
        mutations = (
            lambda value: value.__setitem__('source', 'wrong-source'),
            lambda value: value['config']['recovery'].__setitem__('start_updates', 1049),
            lambda value: value['initialization']['interrupted_recovery'].__setitem__('source', 'wrong-source'),
            lambda value: value['initialization']['training_continuation_history'].pop(),
            lambda value: value['optimizer']['param_groups'][0].__setitem__('lr', 3e-5),
            lambda value: value['optimizer']['state'][0]['step'].fill_(1259),
            lambda value: value['optimizer']['state'][1]['exp_avg'].fill_(float('nan')),
            lambda value: value['rank_states'][0].pop('cuda_rng'),
            lambda value: value['rank_states'][1].__setitem__('torch_rng', torch.tensor([1], dtype=torch.int64)),
            lambda value: value['rank_states'][1]['stream'].__setitem__('plan_sha256', '0' * 64),
            lambda value: value['rank_states'][0]['stream'].__setitem__('cursor', 29999),
            lambda value: value['rank_states'][1]['stream']['contract'].__setitem__('target_assets', 50000),
        )
        for mutate in mutations:
            changed = deepcopy(payload_fixture())
            mutate(changed)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                validate_checkpoint_payload(changed, original, 2, config=config, source_config=source)

    def test_rejects_ordinary_nonrecovery_60k_source_config(self):
        original, _, config = load_configs()
        ordinary = json.loads((EXPERIMENTS / 'fusion_normal_b_extend60000_v1.json').read_text())
        with self.assertRaises(ValueError):
            validate_checkpoint_payload(payload_fixture(), original, 2, config=config, source_config=ordinary)

    def test_extends_exact_60k_prefix_and_first_eight_updates_for_both_ranks(self):
        original, _, config = load_configs()
        payload = payload_fixture()
        validate_checkpoint_payload(payload, original, 2, config=config, source_config=payload['config'])
        for rank in (0, 1):
            saved = payload['rank_states'][rank]['stream']
            old = MatchedFrameStream(
                range(5000), rank=rank, world_size=2, seed=saved['contract']['seed'],
                target_assets=60000, effective_assets=saved['contract']['effective_assets'],
                frame_policy=saved['contract']['frame_policy'],
                manifest_sha256=saved['contract']['manifest_sha256'],
                min_frames=saved['contract']['min_frames'], max_frames=saved['contract']['max_frames'],
                state=saved, update_boundaries=saved['contract']['update_boundaries'])
            stream = extend_checkpoint_stream(payload, rank, target_assets=80000, route='b')
            fresh = MatchedFrameStream(
                range(5000), rank=rank, world_size=2, seed=saved['contract']['seed'],
                target_assets=80000, effective_assets=saved['contract']['effective_assets'],
                frame_policy=saved['contract']['frame_policy'],
                manifest_sha256=saved['contract']['manifest_sha256'],
                min_frames=saved['contract']['min_frames'], max_frames=saved['contract']['max_frames'],
                update_boundaries=list(range(5000, 80001, 5000)))
            while fresh.cursor < 30000:
                fresh.take()
            self.assertEqual(stream.cursor, 30000)
            self.assertEqual(len(stream.update_sizes), 1680)
            self.assertEqual(stream.plan_sha256, B80_PLAN_SHA256)
            self.assertEqual(stream.plan_report()['supervised_frame_exposures'], B80_TARGET_FRAMES)
            self.assertEqual(stream.global_ends[1259], 60000)
            self.assertEqual(stream.global_ends[1679], 80000)
            np.testing.assert_array_equal(stream.order[:60000], old.order)
            np.testing.assert_array_equal(stream.frame_counts[:1260], old.frame_counts)
            np.testing.assert_array_equal(stream.update_sizes[:1260], old.update_sizes)
            for update in range(1260, 1268):
                remaining = int(stream.update_sizes[update]) // 2
                while remaining:
                    actual = stream.take(max_assets=remaining)
                    expected = fresh.take(max_assets=remaining)
                    self.assertEqual(request_signature(actual), request_signature(expected))
                    remaining -= len(actual)
                self.assertEqual(stream.cursor * 2, int(stream.global_ends[update]))
            self.assertEqual(stream.cursor * 2, 60384)


if __name__ == '__main__':
    unittest.main()

