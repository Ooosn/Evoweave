"""Strict A20k-to40k continuation contracts and state restoration, without GPU startup."""
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


def validate_continuation_config(config, original, world):
    expected = {'checkpoint': CHECKPOINT, 'source_config': SOURCE_CONFIG,
                'learning_rate': 1e-4, 'optimizer_betas': 'restore_checkpoint',
                'rng_policy': 'restore_saved_checkpoint_before_milestone_evaluation'}
    if world != 2 or config.get('purpose') != PURPOSE or config.get('route') != 'a':
        raise ValueError('continuation is only the approved two-rank A20k-to40k run')
    if config.get('continuation') != expected:
        raise ValueError('continuation checkpoint, constant learning rate or restore policy changed')
    plan = config['training']
    if (plan['target_assets'] != TARGET_ASSETS or plan['checkpoint_assets'] != MILESTONES
            or plan['evaluation_assets'] != MILESTONES or config['evaluation']['initial'] is not False):
        raise ValueError('continuation must add only the four25k/30k/35k/40k milestones')
    baseline = deepcopy(config)
    baseline.pop('continuation')
    baseline['purpose'] = 'normal5000_trend'
    baseline['training'].update(target_assets=START_ASSETS, checkpoint_assets=BOUNDARIES[:4],
                                evaluation_assets=BOUNDARIES[:4])
    baseline['evaluation']['initial'] = True
    if baseline != original:
        raise ValueError('continuation changed the original architecture, data, optimizer or physical policy')
    return list(MILESTONES)


def validate_checkpoint_payload(payload, original, world):
    if not isinstance(payload, dict):
        raise ValueError('checkpoint payload must be an object')
    required = {'model', 'optimizer', 'architecture', 'rank_states', 'initialization'}
    if not required <= payload.keys():
        raise ValueError('checkpoint lacks full model/optimizer/rank state')
    if (payload.get('schema') != 'virtual_query_xyz_aux_ids_v1'
            or payload.get('purpose') != 'normal5000_trend' or payload.get('config') != original
            or payload.get('source') != CHECKPOINT['source']
            or (payload.get('asset_exposures'), payload.get('step'), payload.get('supervised_frame_exposures'))
            != (START_ASSETS, START_UPDATES, START_FRAMES)):
        raise ValueError('checkpoint is not the pinned completed original A20k run')
    initial = payload['initialization']
    if (initial.get('route') != 'a' or initial.get('config_sha256') != SOURCE_CONFIG['sha256']
            or initial.get('source') != CHECKPOINT['source']
            or initial.get('full_base_initial_state_sha256') != BASE_SHA256
            or initial.get('shared_initial_state_sha256') != BASE_SHA256
            or initial.get('exposure_plan_sha256') != OLD_PLAN_SHA256):
        raise ValueError('checkpoint original initialization provenance changed')
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
        if not math.isfinite(count) or not 0 < count <= START_UPDATES or not count.is_integer():
            raise ValueError('checkpoint optimizer update count is invalid')
        steps.append(count)
    if max(steps) != START_UPDATES:
        raise ValueError('checkpoint optimizer did not reach the A20k update boundary')
    for rank, state in enumerate(states):
        if not {'stream', 'torch_rng', 'cuda_rng', 'numpy_rng', 'python_rng'} <= state.keys():
            raise ValueError('checkpoint is missing rank sampling or RNG state')
        stream = state['stream']
        if stream.get('plan_sha256') != OLD_PLAN_SHA256 or stream.get('cursor') != START_ASSETS // world:
            raise ValueError('checkpoint exposure cursor is not at the completed20k boundary')
        expected = {'rank': rank, 'world_size': world, 'seed': original['data']['stream_seed'],
                    'target_assets': START_ASSETS, 'effective_assets': 48, 'update_boundaries': BOUNDARIES[:4],
                    'dataset_rows': 5000, 'manifest_sha256': original['data']['manifests']['train']['sha256'],
                    'min_frames': 2, 'max_frames': 24, 'frame_policy': original['training']['frame_policy']}
        if stream.get('contract') != expected:
            raise ValueError('checkpoint rank sampling contract differs from original A')


def extend_checkpoint_stream(payload, rank):
    return MatchedFrameStream.extend_completed(
        range(5000), payload['rank_states'][rank]['stream'],
        target_assets=TARGET_ASSETS, update_boundaries=BOUNDARIES)


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
    if frames != expected_frames or state.get('learning_rate') != 1e-4:
        raise ValueError('restored supervised-frame counter or continuation learning rate differs')
    optimizer.load_state_dict(state['optimizer_state_dict'])
    if not optimizer.state:
        raise ValueError('continuation must restore nonempty optimizer moments')
    restore_rank_rng(state['rank_rng_state'], device)
    return samples, frames, step


def set_continuation_lr(optimizer, learning_rate):
    if not math.isfinite(learning_rate) or learning_rate != 1e-4:
        raise ValueError('approved continuation uses fixed1e-4 without warmup/decay')
    betas = [group['betas'] for group in optimizer.param_groups]
    if not betas or any(value != betas[0] for value in betas):
        raise ValueError('continuation expects the original common AdamW betas')
    for group in optimizer.param_groups:
        group['lr'] = learning_rate
    return learning_rate, betas[0][0]
