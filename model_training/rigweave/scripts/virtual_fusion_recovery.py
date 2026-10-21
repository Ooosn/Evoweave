"""Bounded CPU validation and stream restoration for the interrupted B60 resume."""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import torch

from virtual_fusion_normal_runtime import MatchedFrameStream


PURPOSE = 'normal5000_continuation'
SOURCE_COMMIT = 'c643fc700da82eb66ad2307b97e973be6e9eae77'
BASE_INITIAL_STATE_SHA256 = 'a89b29c903abffe2eead47b5411edb8588b599fe874ce490c990ce07225d3635'
PLAN_SHA256 = '4fb7e07e7cc64ec0c17e78016b6899a192d320e9626a4ff8078d747843feb4bf'
ORIGINAL_PLAN_SHA256 = '3321e85cb3608f51620b45ea9c54359b3b0043074ef9f4d645643d1e10a6ba09'
INITIAL_SOURCE_COMMIT = '2430e75d714a0e16bfa0490216e70d67409fab0d'
INITIAL_CONFIG_SHA256 = 'bf66f37d722d29a18a6113fd31d86cef0f402d4cf53b1d06ad5e0aa727809068'
B40_SOURCE_COMMIT = '5703f2c320841c11f0f8276570340f26dc8f11c5'
B40_CONFIG_SHA256 = '1923429a8e2974f2a06a45ed588ec034e3b21ecd9b6354e98c335376b32e18e1'
B40_CHECKPOINT = {
    'path': '/home/wangyy/evorig/outputs/virtual_fusion_normal_20261014/normal_b_extend40000_v1/checkpoint_asset_040000.pt',
    'sha256': '4e48925a44b0ec78d0b88a3a4169823d70d6bce9068e0385cef9e4d3447c0928',
    'bytes': 8438108720,
    'source': B40_SOURCE_COMMIT,
}
B40_SOURCE_CONFIG = {
    'path': '/work/wangyy/virtual_fusion_20261013/normal_code/model_training/experiments/fusion_normal_b_extend40000_v1.json',
    'sha256': B40_CONFIG_SHA256,
}
B20_CHECKPOINT = {
    'path': '/home/wangyy/evorig/outputs/virtual_fusion_normal_20261014/normal_b_v1/checkpoint_asset_020000.pt',
    'sha256': '4c8b677f825b1f9f2ac162afa34cf45ca51a0f506a2b63f4b781c8532c23f50b',
    'bytes': 8438105328,
    'source': INITIAL_SOURCE_COMMIT,
}
B20_SOURCE_CONFIG = {
    'path': '/work/wangyy/virtual_fusion_20261013/normal_code/model_training/experiments/fusion_normal_b_v1.json',
    'sha256': INITIAL_CONFIG_SHA256,
}

CHECKPOINT = {
    'path': '/home/wangyy/evorig/outputs/virtual_fusion_normal_20261014/normal_b_extend60000_v1/checkpoint_asset_050000.pt',
    'sha256': 'dc1d598cb3179694f367a4d9cc9a069e26dceb2236b6dc235a0ed4891e36ebcf',
    'bytes': 8438111152,
    'source': SOURCE_COMMIT,
}
SOURCE_CONFIG = {
    'path': '/work/wangyy/virtual_fusion_20261013/normal_code/model_training/experiments/fusion_normal_b_extend60000_v1.json',
    'sha256': '92e59e3b73a0637785928f18b59f3bda42c86ef144d5fd832933ec5b84ce93a3',
}
RECOVERY_RUNTIME = {
    'allocation': 130309880,
    'host': 'pcg01i',
    'physical_gpus': [5, 6],
}
RECOVERY = {
    'checkpoint': CHECKPOINT,
    'source_config': SOURCE_CONFIG,
    'start_assets': 50000,
    'start_updates': 1050,
    'start_frames': 657056,
    'plan_sha256': PLAN_SHA256,
    'rng_policy': 'restore_saved_checkpoint_before_milestone_evaluation',
    'replay_initial_evaluation': False,
}

START_ASSETS = 50000
START_UPDATES = 1050
START_FRAMES = 657056
TARGET_ASSETS = 60000
TARGET_UPDATES = 1260
TARGET_FRAMES = 788304
WORLD_SIZE = 2
DATASET_ROWS = 5000
EFFECTIVE_ASSETS = 48
EXPECTED_CURSOR = START_ASSETS // WORLD_SIZE
MILESTONES = [55000, 60000]
FRAME_MIN = 2
FRAME_MAX = 24


def _require_mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f'{name} must be a mapping')
    return value


def validate_recovery_config(config, source_config, world):
    """Validate the exact B60 recovery-only config delta."""
    if world != WORLD_SIZE:
        raise ValueError('recovery requires exactly two ranks')
    source = _require_mapping(source_config, 'source_config')
    candidate = _require_mapping(config, 'config')
    expected = deepcopy(source)
    try:
        expected['runtime'].update(deepcopy(RECOVERY_RUNTIME))
        expected['training']['checkpoint_assets'] = list(MILESTONES)
        expected['training']['evaluation_assets'] = list(MILESTONES)
        expected['recovery'] = deepcopy(RECOVERY)
    except (KeyError, TypeError) as exc:
        raise ValueError('source_config lacks the B60 recovery contract') from exc
    if candidate != expected:
        raise ValueError('recovery config differs outside the approved runtime, milestones or recovery record')
    if candidate.get('purpose') != PURPOSE:
        raise ValueError('recovery purpose changed')
    training = candidate.get('training')
    evaluation = candidate.get('evaluation')
    if not isinstance(training, dict) or training.get('target_assets') != TARGET_ASSETS:
        raise ValueError('recovery target_assets changed')
    if training.get('checkpoint_assets') != MILESTONES or training.get('evaluation_assets') != MILESTONES:
        raise ValueError('recovery milestones must be exactly [55000, 60000]')
    if not isinstance(evaluation, dict) or evaluation.get('initial') is not False:
        raise ValueError('recovery must not replay the initial evaluation')
    return list(MILESTONES)


def restore_stream(payload, rank):
    """Reconstruct a 60k stream from the saved 50k state without extending it."""
    if type(rank) is not int or not 0 <= rank < WORLD_SIZE:
        raise ValueError('rank must be 0 or 1')
    payload = _require_mapping(payload, 'payload')
    states = payload.get('rank_states')
    if not isinstance(states, list) or len(states) != WORLD_SIZE:
        raise ValueError('payload must contain both rank states')
    state = _require_mapping(states[rank], f'rank_states[{rank}]')
    stream_state = _require_mapping(state.get('stream'), f'rank_states[{rank}].stream')
    contract = _require_mapping(stream_state.get('contract'), f'rank_states[{rank}].stream.contract')
    expected_keys = {
        'rank', 'world_size', 'seed', 'target_assets', 'effective_assets',
        'update_boundaries', 'dataset_rows', 'manifest_sha256', 'min_frames',
        'max_frames', 'frame_policy',
    }
    if set(contract) != expected_keys:
        raise ValueError('saved stream contract fields differ')
    if (contract['rank'] != rank or contract['world_size'] != WORLD_SIZE
            or contract['target_assets'] != TARGET_ASSETS
            or contract['effective_assets'] != EFFECTIVE_ASSETS
            or contract['dataset_rows'] != DATASET_ROWS
            or contract['min_frames'] != FRAME_MIN
            or contract['max_frames'] != FRAME_MAX):
        raise ValueError('saved stream identity or target differs')
    if stream_state.get('plan_sha256') != PLAN_SHA256:
        raise ValueError('saved stream plan differs from the 60k recovery plan')
    if stream_state.get('cursor') != EXPECTED_CURSOR:
        raise ValueError('saved stream cursor is not the completed 50k boundary')
    try:
        stream = MatchedFrameStream(
            range(DATASET_ROWS),
            rank=contract['rank'],
            world_size=contract['world_size'],
            seed=contract['seed'],
            target_assets=contract['target_assets'],
            effective_assets=contract['effective_assets'],
            frame_policy=deepcopy(contract['frame_policy']),
            manifest_sha256=contract['manifest_sha256'],
            min_frames=contract['min_frames'],
            max_frames=contract['max_frames'],
            update_boundaries=deepcopy(contract['update_boundaries']),
            state=deepcopy(stream_state),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('saved stream cannot be reconstructed') from exc
    if stream.plan_sha256 != PLAN_SHA256 or stream.cursor != EXPECTED_CURSOR:
        raise ValueError('reconstructed stream differs from the saved recovery state')
    if len(stream.update_sizes) != TARGET_UPDATES:
        raise ValueError('reconstructed stream has the wrong update count')
    if stream.plan_report()['supervised_frame_exposures'] != TARGET_FRAMES:
        raise ValueError('reconstructed stream has the wrong frame count')
    return stream


def _validate_rng_state(state, rank):
    required = {'torch_rng', 'cuda_rng', 'numpy_rng', 'python_rng'}
    if not required <= set(state):
        raise ValueError(f'rank {rank} is missing RNG state')
    for key in ('torch_rng', 'cuda_rng'):
        value = state[key]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1 or value.numel() == 0:
            raise ValueError(f'rank {rank} {key} must be a nonempty uint8 tensor')
    numpy_state = state['numpy_rng']
    if (not isinstance(numpy_state, tuple) or len(numpy_state) != 5
            or numpy_state[0] != 'MT19937'
            or not isinstance(numpy_state[1], np.ndarray)
            or numpy_state[1].dtype != np.uint32
            or numpy_state[1].shape != (624,)
            or type(numpy_state[2]) is not int
            or not 0 <= numpy_state[2] <= 624
            or type(numpy_state[3]) is not int
            or type(numpy_state[4]) not in (int, float)
            or not math.isfinite(float(numpy_state[4]))):
        raise ValueError(f'rank {rank} numpy RNG state is invalid')
    python_state = state['python_rng']
    if (not isinstance(python_state, tuple) or len(python_state) != 3
            or python_state[0] != 3
            or not isinstance(python_state[1], tuple)
            or len(python_state[1]) != 625
            or any(type(value) is not int for value in python_state[1])
            or python_state[2] is not None and type(python_state[2]) not in (int, float)):
        raise ValueError(f'rank {rank} Python RNG state is invalid')


def _validate_optimizer(optimizer, *, expected_step=START_UPDATES):
    _require_mapping(optimizer, 'optimizer')
    state = optimizer.get('state')
    groups = optimizer.get('param_groups')
    if not isinstance(state, dict) or len(state) != 947:
        raise ValueError('optimizer must contain exactly 947 entries')
    if not isinstance(groups, list) or len(groups) != 1:
        raise ValueError('optimizer must contain exactly one parameter group')
    group = groups[0]
    if (group.get('lr') != 1e-4
            or tuple(group.get('betas', ())) != (0.9499999512612207, 0.999)
            or group.get('weight_decay') != 0):
        raise ValueError('optimizer parameter group differs from the approved B60 resume')
    steps = []
    for parameter_id, entry in state.items():
        if not isinstance(entry, dict) or not {'step', 'exp_avg', 'exp_avg_sq'} <= set(entry):
            raise ValueError(f'optimizer entry {parameter_id!r} lacks AdamW moments')
        if (not isinstance(entry['exp_avg'], torch.Tensor) or not isinstance(entry['exp_avg_sq'], torch.Tensor)
                or entry['exp_avg'].numel() == 0 or entry['exp_avg_sq'].numel() == 0):
            raise ValueError(f'optimizer entry {parameter_id!r} has non-tensor AdamW moments')
        step = entry['step']
        if isinstance(step, torch.Tensor):
            if step.numel() != 1:
                raise ValueError(f'optimizer entry {parameter_id!r} has a non-scalar step')
            value = float(step.item())
        else:
            value = float(step)
        if not math.isfinite(value) or not value.is_integer() or value != expected_step:
            raise ValueError(f'optimizer entry {parameter_id!r} has an invalid step')
        steps.append(int(value))
    if len(steps) != 947 or min(steps) != expected_step or max(steps) != expected_step:
        raise ValueError('optimizer steps do not all equal the expected recovery boundary')


def _validate_continuation_metadata(initialization):
    initialization = _require_mapping(initialization, 'initialization')
    if (initialization.get('route') != 'b'
            or initialization.get('full_base_initial_state_sha256') != BASE_INITIAL_STATE_SHA256
            or initialization.get('shared_initial_state_sha256') != BASE_INITIAL_STATE_SHA256
            or initialization.get('source') != INITIAL_SOURCE_COMMIT
            or initialization.get('config_sha256') != INITIAL_CONFIG_SHA256):
        raise ValueError('initialization provenance changed')
    if initialization.get('exposure_plan_sha256') != 'a7bdf57c23a9edb0bcac5a79a4603003e8387d613cd0dbac7928255f612dca7b':
        raise ValueError('initial exposure plan changed')
    continuation = _require_mapping(initialization.get('training_continuation'), 'training_continuation')
    expected = {
        'checkpoint': B40_CHECKPOINT,
        'source_config': B40_SOURCE_CONFIG,
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
    }
    for key, value in expected.items():
        if continuation.get(key) != value:
            raise ValueError(f'training continuation field changed: {key}')
    history = initialization.get('training_continuation_history')
    if not isinstance(history, list) or len(history) != 1:
        raise ValueError('B60 checkpoint continuation history must preserve the B20-to-B40 record')
    previous = _require_mapping(history[0], 'training_continuation_history[0]')
    history_expected = {
        'checkpoint': B20_CHECKPOINT,
        'source_config': B20_SOURCE_CONFIG,
        'route': 'b',
        'source': B40_SOURCE_COMMIT,
        'config_sha256': B40_CONFIG_SHA256,
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
    }
    for key, value in history_expected.items():
        if previous.get(key) != value:
            raise ValueError(f'continuation history field changed: {key}')


def validate_recovery_payload(payload, source_config):
    """Validate the pinned 50k payload without mutating global RNG state."""
    payload = _require_mapping(payload, 'payload')
    source_config = _require_mapping(source_config, 'source_config')
    required = {'model', 'architecture', 'optimizer', 'rank_states', 'initialization'}
    if not required <= set(payload):
        raise ValueError('payload lacks full model, optimizer, architecture or rank state')
    if payload.get('config') != source_config:
        raise ValueError('payload config is not the exact B60 source config')
    if (payload.get('source') != SOURCE_COMMIT
            or payload.get('schema') != 'virtual_query_xyz_aux_ids_v1'
            or payload.get('purpose') != PURPOSE
            or (payload.get('asset_exposures'), payload.get('step'), payload.get('supervised_frame_exposures'))
            != (START_ASSETS, START_UPDATES, START_FRAMES)):
        raise ValueError('payload is not the pinned 50k B60 checkpoint')
    _validate_optimizer(payload['optimizer'])
    _validate_continuation_metadata(payload['initialization'])
    if not isinstance(payload['rank_states'], list) or len(payload['rank_states']) != WORLD_SIZE:
        raise ValueError('payload must contain exactly two rank states')
    streams = []
    for rank, state in enumerate(payload['rank_states']):
        state = _require_mapping(state, f'rank_states[{rank}]')
        _validate_rng_state(state, rank)
        streams.append(restore_stream(payload, rank))
    if (len(streams) != WORLD_SIZE
            or any(stream.cursor != EXPECTED_CURSOR for stream in streams)
            or any(stream.plan_sha256 != PLAN_SHA256 for stream in streams)):
        raise ValueError('restored rank streams differ from the pinned 50k state')
    if streams[0].contract['rank'] != 0 or streams[1].contract['rank'] != 1:
        raise ValueError('restored rank identities differ')
    frame_count = int(np.dot(streams[0].update_sizes[:START_UPDATES], streams[0].frame_counts[:START_UPDATES]))
    if frame_count != START_FRAMES:
        raise ValueError('restored stream frame count differs from the 50k payload')
    return streams

