"""Pinned normal5000 continuation contracts with full optimizer and RNG restoration."""
from __future__ import annotations

from copy import deepcopy
import math
import random

import numpy as np
import torch

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
    elif route == 'b' and target == 50000:
        result = {
            'label': 'normal_b_lr3e5_50000_v1', 'start_assets': 40000,
            'start_updates': 840, 'start_frames': 528296,
            'checkpoint': B40_CHECKPOINT, 'source_config': B40_SOURCE_CONFIG,
            'old_plan_sha256': CHAINED_PLAN_SHA256, 'source_purpose': PURPOSE,
            'learning_rate': 3e-5, 'replacement_allocation': 130258195,
            'predecessor_label': 'normal_b_extend40000_v1',
            'predecessor_result': {'execution_complete': True, 'mode': 'resume', 'route': 'b',
                'purpose': PURPOSE, 'code_commit': B40_CHECKPOINT['source'],
                'asset_exposures': 40000, 'updates': 840, 'supervised_frame_exposures': 528296,
                'additional_asset_exposures': 20000, 'additional_updates': 420},
        }
    elif route != 'a':
        raise ValueError('only the pinned B20k-to40k and B40k-to50k continuations are implemented for B')
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
    if 'replacement_allocation' in contract:
        if baseline['runtime']['allocation'] != contract['replacement_allocation']:
            raise ValueError('continuation requires its explicitly granted replacement allocation')
        baseline['runtime']['allocation'] = original['runtime']['allocation']
    if baseline != original:
        raise ValueError('continuation changed the original architecture, data, optimizer or physical policy')
    return list(contract['milestones'])


def validate_checkpoint_payload(payload, original, world, *, config=None, source_config=None):
    contract = continuation_contract(config or {'training': {'target_assets': TARGET_ASSETS}})
    initial_checkpoint, initial_config = original_references(contract['route'])
    if contract['start_assets'] == START_ASSETS:
        expected_config = original
        if source_config is not None and source_config != original:
            raise ValueError('original route source configuration differs')
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
    if (initial.get('route') != contract['route'] or initial.get('config_sha256') != initial_config['sha256']
            or initial.get('source') != initial_checkpoint['source']
            or initial.get('full_base_initial_state_sha256') != BASE_SHA256
            or initial.get('shared_initial_state_sha256') != BASE_SHA256
            or initial.get('exposure_plan_sha256') != OLD_PLAN_SHA256):
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
        if contract['start_assets'] == 120000:
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
        if group.get('lr') != 1e-4 or tuple(group.get('betas', ())) != (.9499999512612207, .999):
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
