"""Pinned normal5000 continuation contracts with full optimizer and RNG restoration."""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import PurePosixPath
import random

import numpy as np
import torch

import virtual_fusion_recovery as recovery
from virtual_fusion_normal_runtime import MatchedFrameStream


PURPOSE = 'normal5000_continuation'
START_ASSETS = 20000
TARGET_ASSETS = 40000
START_UPDATES = 420
START_FRAMES = 263608
BOUNDARIES = list(range(5000, TARGET_ASSETS + 1, 5000))
MILESTONES = BOUNDARIES[4:]
CHECKPOINT = {
    'path': '/home/wangyy/evorig/outputs/virtual_fusion_normal_20261014/normal_a_v1/checkpoint_asset_020000.pt',
    'sha256': 'f365379e87a2e2c946e3f1f2876c5aa3d5dadbe72468469445e1430e67bb8c26',
    'bytes': 8438105328,
    'source': '0a431201a4b65dd7c7700c85d538e703eeabbd29',
}
SOURCE_CONFIG = {
    'path': '/work/wangyy/virtual_fusion_20261013/normal_code/model_training/experiments/fusion_normal_a_v1.json',
    'sha256': '485ab26b736d1523a2fd3d2f7668d714010159610f9b67de694eb4599564bc9d',
}
OLD_PLAN_SHA256 = 'a7bdf57c23a9edb0bcac5a79a4603003e8387d613cd0dbac7928255f612dca7b'
BASE_SHA256 = 'a89b29c903abffe2eead47b5411edb8588b599fe874ce490c990ce07225d3635'
CHAINED_CHECKPOINT = {
    'path': '/home/wangyy/evorig/outputs/virtual_fusion_normal_20261014/normal_a_extend40000_v1/checkpoint_asset_040000.pt',
    'sha256': '8725e36d17733d56dbb76242f5a5ab5a657ee51a4f4be1ed8392e190579d6f13',
    'bytes': 8438108528,
    'source': '6715a2b880537ed73348d7a8320974765962af15',
}
CHAINED_SOURCE_CONFIG = {
    'path': '/work/wangyy/virtual_fusion_20261013/normal_code/model_training/experiments/fusion_normal_a_extend40000_v1.json',
    'sha256': '41faf9572d214bbe50e7378b6ab03eed83575fd7ea1895bec4061bda0b725b5a',
}
CHAINED_PLAN_SHA256 = '3321e85cb3608f51620b45ea9c54359b3b0043074ef9f4d645643d1e10a6ba09'
EXTENDED_CHECKPOINT = {
    'path': '/home/wangyy/evorig/outputs/virtual_fusion_normal_20261014/normal_a_extend80000_v1/checkpoint_asset_080000.pt',
    'sha256': 'e3ec08bd177b583874c85b33f8d921ef1e4fbd3902533320b88bae56571bb600',
    'bytes': 8438111024,
    'source': '21f3a2eba2b9b162756b6ab1684913f587c853ee',
}
EXTENDED_SOURCE_CONFIG = {
    'path': '/work/wangyy/virtual_fusion_20261013/normal_code/model_training/experiments/fusion_normal_a_extend80000_v1.json',
    'sha256': '2057ebec5370d780de393edfe026cba25b01a29335eab577f8652cc3bafc679a',
}
EXTENDED_PLAN_SHA256 = 'f561a7d8432792920292cc1cf06f74858872b06173187d9ac3f132efdf72b7f2'
A120_CHECKPOINT = {
    'path': '/home/wangyy/evorig/outputs/virtual_fusion_normal_20261014/normal_a_extend120000_v1/checkpoint_asset_120000.pt',
    'sha256': 'e3e8a8569641005c5798d9974ecf011f388987c0bfb72d47b2f6ebd736437628',
    'bytes': 8438113456,
    'source': 'ac1b0ca2abb890175a9f149ca483cc7a538dccf0',
}
A120_SOURCE_CONFIG = {
    'path': '/work/wangyy/virtual_fusion_20261013/normal_code/model_training/experiments/fusion_normal_a_extend120000_v1.json',
    'sha256': '7bec5e0c377ab3b893a54021c32506a07a8c1a61a40bb043fed5761cb4f25726',
}
A120_PLAN_SHA256 = '26ff999633011ec96e21b0015e671e18c654a92c9e0b5a7bfe1cc7cc18df4a70'
A130_CHECKPOINT = {
    'path': '/home/wangyy/evorig/outputs/virtual_fusion_normal_20261014/normal_a_lr3e5_130000_v1/checkpoint_asset_130000.pt',
    'sha256': 'd1c041e055e8be507137d0851fb137b29244a5f0bf604bc0bddc2f4f309b8676',
    'bytes': 8438115760,
    'source': 'a189013cb14efed31a534a8e801a97bb5e162edb',
}
A130_SOURCE_CONFIG = {
    'path': '/work/wangyy/virtual_fusion_20261013/normal_code/model_training/experiments/fusion_normal_a_lr3e5_130000_v1.json',
    'sha256': '584f9b709f7c9b993a5f78f38a1318bca5871883dc16add911073aa26dcdd325',
}
A130_PLAN_SHA256 = '29b3527f644f76d996ccfb3e7286d6eac43b29895d4c777ebb48d89a98a4dc41'
B_CHECKPOINT = {
    'path': '/home/wangyy/evorig/outputs/virtual_fusion_normal_20261014/normal_b_v1/checkpoint_asset_020000.pt',
    'sha256': '4c8b677f825b1f9f2ac162afa34cf45ca51a0f506a2b63f4b781c8532c23f50b',
    'bytes': 8438105328,
    'source': '2430e75d714a0e16bfa0490216e70d67409fab0d',
}
B_SOURCE_CONFIG = {
    'path': '/work/wangyy/virtual_fusion_20261013/normal_code/model_training/experiments/fusion_normal_b_v1.json',
    'sha256': 'bf66f37d722d29a18a6113fd31d86cef0f402d4cf53b1d06ad5e0aa727809068',
}
B40_CHECKPOINT = {
    'path': '/home/wangyy/evorig/outputs/virtual_fusion_normal_20261014/normal_b_extend40000_v1/checkpoint_asset_040000.pt',
    'sha256': '4e48925a44b0ec78d0b88a3a4169823d70d6bce9068e0385cef9e4d3447c0928',
    'bytes': 8438108720,
    'source': '5703f2c320841c11f0f8276570340f26dc8f11c5',
}
B40_SOURCE_CONFIG = {
    'path': '/work/wangyy/virtual_fusion_20261013/normal_code/model_training/experiments/fusion_normal_b_extend40000_v1.json',
    'sha256': '1923429a8e2974f2a06a45ed588ec034e3b21ecd9b6354e98c335376b32e18e1',
}
B60_CHECKPOINT = {
    'path': '/home/wangyy/evorig/outputs/virtual_fusion_normal_20261014/normal_b_recover60000_20261021_v1/checkpoint_asset_060000.pt',
    'sha256': '8bdf349eb0c808871dd13a9fda4ff84a9ae8ac1e815f855f6cb50c52f4608d76',
    'bytes': 8438113904,
    'source': 'da2ee8878a271efc87d1c219d88463ea574eee07',
}
B60_SOURCE_CONFIG = {
    'path': '/work/wangyy/virtual_fusion_20261013/normal_code/model_training/experiments/fusion_normal_b_recover60000_20261021_v1.json',
    'sha256': '810af5796f7f150a7588af676b3a17173c2ab18b02ece271d47440b1cd0156d2',
}
RECOVERY_INCOMING_MODEL_SHA256 = '5b73652cf5940e29bc8a236a27f65b0c7a6fd40d3a36a1563fc2c7ab8b8cdf4e'
B80_PLAN_SHA256 = 'f561a7d8432792920292cc1cf06f74858872b06173187d9ac3f132efdf72b7f2'
B80_RUNTIME = {
    'allocation': 130309880,
    'host': 'pcg01i',
    'physical_gpus': [5, 6],
}
B80_TARGET_ASSETS = 80000
B80_TARGET_UPDATES = 1680
B80_TARGET_FRAMES = 1041832


def original_references(route):
    if route == 'a':
        return CHECKPOINT, SOURCE_CONFIG
    if route == 'b':
        return B_CHECKPOINT, B_SOURCE_CONFIG
    raise ValueError('continuation route must be an explicitly approved A or B run')


def continuation_contract(config):
    target = config['training']['target_assets']
    route = config.get('route', 'a')
    original_references(route)
    if route == 'b' and target == 40000:
        result = {
            'label': 'normal_b_extend40000_v1', 'start_assets': 20000,
            'start_updates': 420, 'start_frames': 263608,
            'checkpoint': B_CHECKPOINT, 'source_config': B_SOURCE_CONFIG,
            'old_plan_sha256': OLD_PLAN_SHA256, 'source_purpose': 'normal5000_trend',
            'replacement_allocation': 130258195,
            'predecessor_label': 'normal_b_v1',
            'predecessor_result': {'execution_complete': True, 'mode': 'train', 'route': 'b',
                'purpose': 'normal5000_trend', 'code_commit': B_CHECKPOINT['source'],
                'asset_exposures': 20000, 'updates': 420, 'supervised_frame_exposures': 263608},
        }
    elif route == 'b' and target == B80_TARGET_ASSETS:
        result = {
            'label': 'normal_b_lr3e5_80000_v1', 'start_assets': 60000,
            'start_updates': 1260, 'start_frames': 788304,
            'checkpoint': B60_CHECKPOINT, 'source_config': B60_SOURCE_CONFIG,
            'old_plan_sha256': recovery.PLAN_SHA256, 'source_purpose': PURPOSE,
            'learning_rate': 3e-5, 'runtime_override': deepcopy(B80_RUNTIME),
            'predecessor_label': 'normal_b_recover60000_20261021_v1',
            'predecessor_jobdir': '/home/wangyy/jobs/virtual_fusion_recovery_20261021_v1',
            'predecessor_result': {'execution_complete': True, 'mode': 'interrupted_resume',
                'purpose': PURPOSE, 'route': 'b', 'code_commit': B60_CHECKPOINT['source'],
                'asset_exposures': 60000, 'updates': 1260, 'supervised_frame_exposures': 788304,
                'additional_asset_exposures': 10000, 'additional_updates': 210},
        }
    elif route == 'b' and target in (50000, 60000):
        result = {
            'label': 'normal_b_lr3e5_50000_v1' if target == 50000 else 'normal_b_extend60000_v1',
            'start_assets': 40000,
            'start_updates': 840, 'start_frames': 528296,
            'checkpoint': B40_CHECKPOINT, 'source_config': B40_SOURCE_CONFIG,
            'old_plan_sha256': CHAINED_PLAN_SHA256, 'source_purpose': PURPOSE,
            'learning_rate': 3e-5 if target == 50000 else 1e-4, 'replacement_allocation': 130258195,
            'predecessor_label': 'normal_b_extend40000_v1',
            'predecessor_result': {'execution_complete': True, 'mode': 'resume', 'route': 'b',
                'purpose': PURPOSE, 'code_commit': B40_CHECKPOINT['source'],
                'asset_exposures': 40000, 'updates': 840, 'supervised_frame_exposures': 528296,
                'additional_asset_exposures': 20000, 'additional_updates': 420},
        }
    elif route != 'a':
        raise ValueError('only the explicitly pinned B continuation targets are implemented')
    elif target == TARGET_ASSETS:
        result = {
            'label': 'normal_a_extend40000_v1', 'start_assets': START_ASSETS,
            'start_updates': START_UPDATES, 'start_frames': START_FRAMES,
            'checkpoint': CHECKPOINT, 'source_config': SOURCE_CONFIG,
            'old_plan_sha256': OLD_PLAN_SHA256, 'source_purpose': 'normal5000_trend',
            'predecessor_label': 'normal_c_v1',
            'predecessor_result': {'execution_complete': True, 'mode': 'train', 'route': 'c',
                'code_commit': 'cf1c1ae54cdcaef002b9c4d665f1133903e76a5d',
                'asset_exposures': 20000, 'updates': 420, 'supervised_frame_exposures': 263608},
        }
    elif target == 80000:
        result = {
            'label': 'normal_a_extend80000_v1', 'start_assets': 40000,
            'start_updates': 840, 'start_frames': 528296,
            'checkpoint': CHAINED_CHECKPOINT, 'source_config': CHAINED_SOURCE_CONFIG,
            'old_plan_sha256': CHAINED_PLAN_SHA256, 'source_purpose': PURPOSE,
            'predecessor_label': 'normal_a_extend40000_v1',
            'predecessor_result': {'execution_complete': True, 'mode': 'resume', 'route': 'a',
                'purpose': PURPOSE, 'code_commit': CHAINED_CHECKPOINT['source'],
                'asset_exposures': 40000, 'updates': 840, 'supervised_frame_exposures': 528296,
                'additional_asset_exposures': 20000, 'additional_updates': 420},
        }
    elif target == 120000:
        result = {
            'label': 'normal_a_extend120000_v1', 'start_assets': 80000,
            'start_updates': 1680, 'start_frames': 1041832,
            'checkpoint': EXTENDED_CHECKPOINT, 'source_config': EXTENDED_SOURCE_CONFIG,
            'old_plan_sha256': EXTENDED_PLAN_SHA256, 'source_purpose': PURPOSE,
            'replacement_allocation': 130258195,
            'predecessor_label': 'normal_a_extend80000_v1',
            'predecessor_result': {'execution_complete': True, 'mode': 'resume', 'route': 'a',
                'purpose': PURPOSE, 'code_commit': EXTENDED_CHECKPOINT['source'],
                'asset_exposures': 80000, 'updates': 1680, 'supervised_frame_exposures': 1041832,
                'additional_asset_exposures': 40000, 'additional_updates': 840},
        }
    elif target == 130000:
        result = {
            'label': 'normal_a_lr3e5_130000_v1', 'start_assets': 120000,
            'start_updates': 2520, 'start_frames': 1555504,
            'checkpoint': A120_CHECKPOINT, 'source_config': A120_SOURCE_CONFIG,
            'old_plan_sha256': A120_PLAN_SHA256, 'source_purpose': PURPOSE,
            'learning_rate': 3e-5, 'replacement_allocation': 130258195,
            'predecessor_label': 'normal_a_extend120000_v1',
            'predecessor_result': {'execution_complete': True, 'mode': 'resume', 'route': 'a',
                'purpose': PURPOSE, 'code_commit': A120_CHECKPOINT['source'],
                'asset_exposures': 120000, 'updates': 2520, 'supervised_frame_exposures': 1555504,
                'additional_asset_exposures': 40000, 'additional_updates': 840},
        }
    elif target == 150000:
        result = {
            'label': 'normal_a_lr3e5_150000_v1', 'start_assets': 130000,
            'start_updates': 2730, 'start_frames': 1685592,
            'checkpoint': A130_CHECKPOINT, 'source_config': A130_SOURCE_CONFIG,
            'old_plan_sha256': A130_PLAN_SHA256, 'source_purpose': PURPOSE,
            'learning_rate': 3e-5, 'replacement_allocation': 130258195,
            'predecessor_label': 'normal_a_lr3e5_130000_v1',
            'predecessor_result': {'execution_complete': True, 'mode': 'resume', 'route': 'a',
                'purpose': PURPOSE, 'code_commit': A130_CHECKPOINT['source'],
                'asset_exposures': 130000, 'updates': 2730, 'supervised_frame_exposures': 1685592,
                'additional_asset_exposures': 10000, 'additional_updates': 210},
        }
    else:
        raise ValueError('only the explicitly pinned A and B continuation targets are implemented')
    result = deepcopy(result)
    result['route'] = route
    result.setdefault('learning_rate', 1e-4)
    result['target_assets'] = target
    result['boundaries'] = list(range(5000, target + 1, 5000))
    result['source_boundaries'] = list(range(5000, result['start_assets'] + 1, 5000))
    result['milestones'] = [value for value in result['boundaries'] if value > result['start_assets']]
    return result


def validate_continuation_config(config, original, world):
    contract = continuation_contract(config)
    expected = {'checkpoint': contract['checkpoint'], 'source_config': contract['source_config'],
                'learning_rate': contract['learning_rate'], 'optimizer_betas': 'restore_checkpoint',
                'rng_policy': 'restore_saved_checkpoint_before_milestone_evaluation'}
    if world != 2 or config.get('purpose') != PURPOSE or config.get('route') != contract['route']:
        raise ValueError('continuation is only the explicitly approved two-rank route')
    if config.get('continuation') != expected:
        raise ValueError('continuation checkpoint, constant learning rate or restore policy changed')
    plan = config['training']
    if (plan['checkpoint_assets'] != contract['milestones']
            or plan['evaluation_assets'] != contract['milestones'] or config['evaluation']['initial'] is not False):
        raise ValueError('continuation must evaluate only the new exact5000-exposure milestones')
    baseline = deepcopy(config)
    baseline.pop('continuation')
    baseline['purpose'] = 'normal5000_trend'
    baseline['training'].update(target_assets=START_ASSETS, checkpoint_assets=BOUNDARIES[:4],
                                evaluation_assets=BOUNDARIES[:4])
    baseline['evaluation']['initial'] = True
    if 'runtime_override' in contract:
        for key, value in contract['runtime_override'].items():
            if baseline['runtime'].get(key) != value:
                raise ValueError(f'continuation requires its pinned replacement runtime {key}')
            baseline['runtime'][key] = original['runtime'][key]
    elif 'replacement_allocation' in contract:
        if baseline['runtime']['allocation'] != contract['replacement_allocation']:
            raise ValueError('continuation requires its explicitly granted replacement allocation')
        baseline['runtime']['allocation'] = original['runtime']['allocation']
    if baseline != original:
        raise ValueError('continuation changed the original architecture, data, optimizer or physical policy')
    return list(contract['milestones'])


def _expected_recovered_b60_config(original):
    """Build the exact recovered B60 source config without trusting a file path."""
    expected = deepcopy(original)
    expected['purpose'] = PURPOSE
    expected['route'] = 'b'
    expected['runtime'].update(deepcopy(recovery.RECOVERY_RUNTIME))
    expected['training'].update(
        target_assets=recovery.TARGET_ASSETS,
        checkpoint_assets=list(recovery.MILESTONES),
        evaluation_assets=list(recovery.MILESTONES),
    )
    expected['evaluation']['initial'] = False
    expected['continuation'] = {
        'checkpoint': deepcopy(B40_CHECKPOINT),
        'source_config': deepcopy(B40_SOURCE_CONFIG),
        'learning_rate': 1e-4,
        'optimizer_betas': 'restore_checkpoint',
        'rng_policy': recovery.RECOVERY['rng_policy'],
    }
    expected['recovery'] = deepcopy(recovery.RECOVERY)
    return expected


def _validate_b80_optimizer(optimizer):
    recovery._validate_optimizer(optimizer, expected_step=1260)
    for parameter_id, entry in optimizer['state'].items():
        for name in ('exp_avg', 'exp_avg_sq'):
            if not torch.isfinite(entry[name]).all():
                raise ValueError(f'optimizer entry {parameter_id!r} has non-finite {name}')


def _validate_b80_recovery_provenance(initialization):
    recovery._validate_continuation_metadata(initialization)
    expected = {
        'checkpoint': recovery.CHECKPOINT,
        'source_config': recovery.SOURCE_CONFIG,
        'route': 'b',
        'source': B60_CHECKPOINT['source'],
        'config_sha256': B60_SOURCE_CONFIG['sha256'],
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
        'historical_evaluations_preserved': str(PurePosixPath(recovery.CHECKPOINT['path']).parent),
        'fresh_official_initialization': False,
        'bitwise_future_trajectory_claimed': False,
        'quality_acceptance': False,
    }
    if initialization.get('interrupted_recovery') != expected:
        raise ValueError('completed B60 payload lacks the exact interrupted-recovery provenance')


def _validate_b60_stream(stream_state, rank):
    try:
        stream = MatchedFrameStream(
            range(5000), rank=rank, world_size=2,
            seed=stream_state['contract']['seed'],
            target_assets=stream_state['contract']['target_assets'],
            effective_assets=stream_state['contract']['effective_assets'],
            frame_policy=deepcopy(stream_state['contract']['frame_policy']),
            manifest_sha256=stream_state['contract']['manifest_sha256'],
            min_frames=stream_state['contract']['min_frames'],
            max_frames=stream_state['contract']['max_frames'],
            state=deepcopy(stream_state),
            update_boundaries=deepcopy(stream_state['contract']['update_boundaries']),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('completed B60 rank stream cannot be reconstructed') from exc
    if (stream.plan_sha256 != recovery.PLAN_SHA256 or stream.cursor != 30000
            or len(stream.update_sizes) != 1260
            or stream.plan_report()['supervised_frame_exposures'] != 788304):
        raise ValueError('completed B60 rank stream differs from the pinned 60k plan')


def validate_checkpoint_payload(payload, original, world, *, config=None, source_config=None):
    contract = continuation_contract(config or {'training': {'target_assets': TARGET_ASSETS}})
    initial_checkpoint, initial_config = original_references(contract['route'])
    recovered_b60 = contract['route'] == 'b' and contract['start_assets'] == 60000
    if contract['start_assets'] == START_ASSETS:
        expected_config = original
        if source_config is not None and source_config != original:
            raise ValueError('original route source configuration differs')
    elif recovered_b60:
        if source_config is None or source_config != _expected_recovered_b60_config(original):
            raise ValueError('the B80 continuation requires the exact completed recovered B60 config')
        expected_config = source_config
    else:
        if source_config is None or source_config['training']['target_assets'] != contract['start_assets']:
            raise ValueError('the chained continuation requires its completed source configuration')
        validate_continuation_config(source_config, original, world)
        expected_config = source_config
    if not isinstance(payload, dict):
        raise ValueError('checkpoint payload must be an object')
    required = {'model', 'optimizer', 'architecture', 'rank_states', 'initialization'}
    if not required <= payload.keys():
        raise ValueError('checkpoint lacks full model/optimizer/rank state')
    if (payload.get('schema') != 'virtual_query_xyz_aux_ids_v1'
            or payload.get('purpose') != contract['source_purpose'] or payload.get('config') != expected_config
            or payload.get('source') != contract['checkpoint']['source']
            or (payload.get('asset_exposures'), payload.get('step'), payload.get('supervised_frame_exposures'))
            != (contract['start_assets'], contract['start_updates'], contract['start_frames'])):
        raise ValueError('checkpoint is not the pinned completed source run')
    initial = payload['initialization']
    expected_initial_plan = OLD_PLAN_SHA256
    if (initial.get('route') != contract['route'] or initial.get('config_sha256') != initial_config['sha256']
            or initial.get('source') != initial_checkpoint['source']
            or initial.get('full_base_initial_state_sha256') != BASE_SHA256
            or initial.get('shared_initial_state_sha256') != BASE_SHA256
            or initial.get('exposure_plan_sha256') != expected_initial_plan):
        raise ValueError('checkpoint original initialization provenance changed')
    if contract['route'] == 'a' and contract['start_assets'] >= 40000:
        expected_40k = {
            'checkpoint': CHECKPOINT, 'source_config': SOURCE_CONFIG, 'route': 'a',
            'source': CHAINED_CHECKPOINT['source'], 'config_sha256': CHAINED_SOURCE_CONFIG['sha256'],
            'start_asset_exposures': START_ASSETS, 'target_asset_exposures': 40000,
            'start_updates': START_UPDATES, 'target_updates': 840,
            'start_supervised_frame_exposures': START_FRAMES, 'target_supervised_frame_exposures': 528296,
            'original_exposure_plan_sha256': OLD_PLAN_SHA256, 'extended_exposure_plan_sha256': CHAINED_PLAN_SHA256,
            'constant_learning_rate': 1e-4, 'optimizer_betas': 'restore_checkpoint',
            'optimizer_state_restore_required': True, 'fresh_official_initialization': False,
            'resumed_trained_weights_loaded': True, 'original_20000_prefix_verified': True,
        }
        prior = initial.get('training_continuation', {})
        old_prior = prior
        history = initial.get('training_continuation_history', [])
        if contract['start_assets'] == 130000:
            expected_130k = {
                'checkpoint': A120_CHECKPOINT, 'source_config': A120_SOURCE_CONFIG, 'route': 'a',
                'source': A130_CHECKPOINT['source'], 'config_sha256': A130_SOURCE_CONFIG['sha256'],
                'start_asset_exposures': 120000, 'target_asset_exposures': 130000,
                'start_updates': 2520, 'target_updates': 2730,
                'start_supervised_frame_exposures': 1555504, 'target_supervised_frame_exposures': 1685592,
                'original_exposure_plan_sha256': A120_PLAN_SHA256,
                'extended_exposure_plan_sha256': A130_PLAN_SHA256,
                'constant_learning_rate': 3e-5, 'optimizer_betas': 'restore_checkpoint',
                'optimizer_state_restore_required': True, 'fresh_official_initialization': False,
                'resumed_trained_weights_loaded': True, 'original_20000_prefix_verified': True,
                'saved_prefix_asset_exposures_verified': 120000,
            }
            if any(prior.get(key) != value for key, value in expected_130k.items()) or len(history) != 3:
                raise ValueError('completed130k checkpoint must retain its low-rate A120k continuation and full history')
            prior = history[2]
            history = history[:2]
        if contract['start_assets'] >= 120000:
            expected_120k = {
                'checkpoint': EXTENDED_CHECKPOINT, 'source_config': EXTENDED_SOURCE_CONFIG, 'route': 'a',
                'source': A120_CHECKPOINT['source'], 'config_sha256': A120_SOURCE_CONFIG['sha256'],
                'start_asset_exposures': 80000, 'target_asset_exposures': 120000,
                'start_updates': 1680, 'target_updates': 2520,
                'start_supervised_frame_exposures': 1041832, 'target_supervised_frame_exposures': 1555504,
                'original_exposure_plan_sha256': EXTENDED_PLAN_SHA256,
                'extended_exposure_plan_sha256': A120_PLAN_SHA256,
                'constant_learning_rate': 1e-4, 'optimizer_betas': 'restore_checkpoint',
                'optimizer_state_restore_required': True, 'fresh_official_initialization': False,
                'resumed_trained_weights_loaded': True, 'original_20000_prefix_verified': True,
                'saved_prefix_asset_exposures_verified': 80000,
            }
            if any(prior.get(key) != value for key, value in expected_120k.items()) or len(history) != 2:
                raise ValueError('completed120k checkpoint must retain its A80k continuation and full history')
            prior = history[1]
            history = history[:1]
        if contract['start_assets'] >= 80000:
            expected_80k = {
                'checkpoint': CHAINED_CHECKPOINT, 'source_config': CHAINED_SOURCE_CONFIG, 'route': 'a',
                'source': EXTENDED_CHECKPOINT['source'], 'config_sha256': EXTENDED_SOURCE_CONFIG['sha256'],
                'start_asset_exposures': 40000, 'target_asset_exposures': 80000,
                'start_updates': 840, 'target_updates': 1680,
                'start_supervised_frame_exposures': 528296, 'target_supervised_frame_exposures': 1041832,
                'original_exposure_plan_sha256': CHAINED_PLAN_SHA256,
                'extended_exposure_plan_sha256': EXTENDED_PLAN_SHA256,
                'constant_learning_rate': 1e-4, 'optimizer_betas': 'restore_checkpoint',
                'optimizer_state_restore_required': True, 'fresh_official_initialization': False,
                'resumed_trained_weights_loaded': True, 'original_20000_prefix_verified': True,
                'saved_prefix_asset_exposures_verified': 40000,
            }
            if any(prior.get(key) != value for key, value in expected_80k.items()):
                raise ValueError('completed80k checkpoint does not retain its A40k continuation provenance')
            if len(history) != 1:
                raise ValueError('completed80k checkpoint must retain the original A20k continuation history')
            old_prior = history[0]
        if any(old_prior.get(key) != value for key, value in expected_40k.items()):
            raise ValueError('completed40k checkpoint does not retain its original A20k continuation provenance')
    if recovered_b60:
        _validate_b80_recovery_provenance(initial)
    if contract['route'] == 'b' and contract['start_assets'] == 40000:
        expected_b40k = {
            'checkpoint': B_CHECKPOINT, 'source_config': B_SOURCE_CONFIG, 'route': 'b',
            'source': B40_CHECKPOINT['source'], 'config_sha256': B40_SOURCE_CONFIG['sha256'],
            'start_asset_exposures': 20000, 'target_asset_exposures': 40000,
            'start_updates': 420, 'target_updates': 840,
            'start_supervised_frame_exposures': 263608, 'target_supervised_frame_exposures': 528296,
            'original_exposure_plan_sha256': OLD_PLAN_SHA256, 'extended_exposure_plan_sha256': CHAINED_PLAN_SHA256,
            'constant_learning_rate': 1e-4, 'optimizer_betas': 'restore_checkpoint',
            'optimizer_state_restore_required': True, 'fresh_official_initialization': False,
            'resumed_trained_weights_loaded': True, 'original_20000_prefix_verified': True,
            'saved_prefix_asset_exposures_verified': 20000,
        }
        prior = initial.get('training_continuation', {})
        if (any(prior.get(key) != value for key, value in expected_b40k.items())
                or initial.get('training_continuation_history', [])):
            raise ValueError('completed B40k checkpoint must retain only its own B20k continuation provenance')
    states = payload['rank_states']
    if world != 2 or not isinstance(states, list) or len(states) != world:
        raise ValueError('checkpoint must contain both rank states')
    optimizer = payload['optimizer']
    if not isinstance(optimizer, dict) or not optimizer.get('state') or len(optimizer.get('param_groups', [])) != 1:
        raise ValueError('checkpoint optimizer is empty or incompatible')
    if recovered_b60:
        _validate_b80_optimizer(optimizer)
    else:
        steps = []
        for value in optimizer['state'].values():
            if not {'step', 'exp_avg', 'exp_avg_sq'} <= value.keys():
                raise ValueError('checkpoint optimizer lacks AdamW moments')
            count = float(value['step'])
            if not math.isfinite(count) or not 0 < count <= contract['start_updates'] or not count.is_integer():
                raise ValueError('checkpoint optimizer update count is invalid')
            steps.append(count)
        if max(steps) != contract['start_updates']:
            raise ValueError('checkpoint optimizer did not reach the saved source update boundary')
    if contract['start_assets'] >= 40000:
        group = optimizer['param_groups'][0]
        source_lr = continuation_contract(source_config)['learning_rate']
        if group.get('lr') != source_lr or tuple(group.get('betas', ())) != (.9499999512612207, .999):
            raise ValueError('completed continuation optimizer must retain its constant learning rate and restored betas')
    for rank, state in enumerate(states):
        if not {'stream', 'torch_rng', 'cuda_rng', 'numpy_rng', 'python_rng'} <= state.keys():
            raise ValueError('checkpoint is missing rank sampling or RNG state')
        stream = state['stream']
        if (stream.get('plan_sha256') != contract['old_plan_sha256']
                or stream.get('cursor') != contract['start_assets'] // world):
            raise ValueError('checkpoint exposure cursor is not at the completed source boundary')
        expected = {'rank': rank, 'world_size': world, 'seed': original['data']['stream_seed'],
                    'target_assets': contract['start_assets'], 'effective_assets': 48,
                    'update_boundaries': contract['source_boundaries'],
                    'dataset_rows': 5000, 'manifest_sha256': original['data']['manifests']['train']['sha256'],
                    'min_frames': 2, 'max_frames': 24, 'frame_policy': original['training']['frame_policy']}
        if stream.get('contract') != expected:
            raise ValueError('checkpoint rank sampling contract differs from original A')
        if recovered_b60:
            recovery._validate_rng_state(state, rank)
            _validate_b60_stream(stream, rank)


def extend_checkpoint_stream(payload, rank, *, target_assets=TARGET_ASSETS, route='a'):
    contract = continuation_contract({'route': route, 'training': {'target_assets': target_assets}})
    return MatchedFrameStream.extend_completed(
        range(5000), payload['rank_states'][rank]['stream'],
        target_assets=target_assets, update_boundaries=contract['boundaries'])


def restore_rank_rng(state, device):
    torch.set_rng_state(state['torch_rng'])
    np.random.set_state(state['numpy_rng'])
    random.setstate(state['python_rng'])
    if torch.device(device).type == 'cuda':
        torch.cuda.set_rng_state(state['cuda_rng'], device)


def restore_training_state(optimizer, stream, state, *, device, rank, world, target_assets):
    samples, frames, step = (state[key] for key in ('asset_exposures', 'supervised_frame_exposures', 'step'))
    if (any(type(value) is not int or value < 0 for value in (samples, frames, step))
            or not 0 < samples < target_assets or samples % world
            or stream.cursor * world != samples or stream.rank != rank or stream.world_size != world
            or stream.target_assets != target_assets):
        raise ValueError('restored training counters disagree with the extended stream')
    if step >= len(stream.update_sizes) or int(stream.global_ends[step - 1]) != samples:
        raise ValueError('restored step does not land on the saved exposure boundary')
    expected_frames = int(np.dot(stream.update_sizes[:step].astype(np.int64), stream.frame_counts[:step]))
    if frames != expected_frames or state.get('learning_rate') not in (1e-4, 3e-5):
        raise ValueError('restored supervised-frame counter or continuation learning rate differs')
    optimizer.load_state_dict(state['optimizer_state_dict'])
    if not optimizer.state:
        raise ValueError('continuation must restore nonempty optimizer moments')
    restore_rank_rng(state['rank_rng_state'], device)
    return samples, frames, step


def set_continuation_lr(optimizer, learning_rate):
    if not math.isfinite(learning_rate) or learning_rate not in (1e-4, 3e-5):
        raise ValueError('approved continuation uses only fixed1e-4 or fixed3e-5 without scheduler restart')
    betas = [group['betas'] for group in optimizer.param_groups]
    if not betas or any(value != betas[0] for value in betas):
        raise ValueError('continuation expects the original common AdamW betas')
    for group in optimizer.param_groups:
        group['lr'] = learning_rate
    return learning_rate, betas[0][0]
