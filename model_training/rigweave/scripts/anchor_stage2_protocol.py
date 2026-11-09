"""Exact standalone Stage 2 experiment and capacity admission."""
import json
import math
from pathlib import Path

from reference_xyz_protocol import sha256

ROOT = Path(__file__).resolve().parents[3]
SCOPE = 'anchor_stage2_5k_20261109'
ENTRYPOINT = 'model_training/rigweave/scripts/train_anchor_stage2.py'
ARMS = {'st12': (12, 12), 'sst8': (16, 8)}
MILESTONES = list(range(10000, 100001, 10000))
SHORT_BATCH = 24
LONG_BATCHES = (24, 12)
HARD_MIB = 79872
PROCESS_MIB = 76000
ACCEPTED_WHOLE_GPU_MIB = 75 * 1024


def frame_policy(long_batch):
    if long_batch not in LONG_BATCHES:
        raise ValueError('only the two authorized long-frame batches may be used')
    return {str(t): SHORT_BATCH if t <= 12 else long_batch for t in range(2, 25)}


def capacity_stage(arm, long_batch):
    return f'capacity_{arm}_{long_batch}'


def predecessor_config(config):
    recovery = config['recovery']
    spec = recovery['config']
    path = ROOT / 'model_training/experiments' / f'{SCOPE}.json'
    if ROOT / spec['path'] != path or sha256(path) != spec['sha256']:
        raise ValueError('retained Stage 2 configuration changed')
    previous = json.loads(path.read_text())
    expected = {**previous, 'recovery': recovery, 'artifacts': config['artifacts'],
        'runtime': {**previous['runtime'], **{k: config['runtime'][k] for k in ('repo', 'output_root')}},
        'evaluation': {**previous['evaluation'], 'panel_archive': config['evaluation']['panel_archive']}}
    if (config != expected or recovery['arm'] != 'st12' or recovery['samples'] != 10000
            or len(recovery['traces']) != 2 or len(recovery['checkpoints']) != 2
            or config['runtime']['repo'] == previous['runtime']['repo']
            or config['runtime']['output_root'] == previous['runtime']['output_root']
            or config['artifacts'] == previous['artifacts']):
        raise ValueError('recovery must preserve the exact training recipe and isolate outputs')
    return previous


def validate_config(config):
    data = config['data']
    training = config['training']
    if (config['family'] != SCOPE or data['manifests']['train']['rows'] != 5000
            or data['manifests']['valid']['rows'] != 967
            or training['target_assets'] != 100000 or training['effective_assets_per_update'] != 48
            or training['checkpoint_assets'] != MILESTONES
            or training['candidate_long_batches'] != list(LONG_BATCHES)
            or config['optimization']['schedule_samples'] != training['target_assets']
            or config['evaluation']['valid_assets'] != 128 or config['evaluation']['input_frames'] != 8
            or config['evaluation']['reference_slots'] != [0, 4]
            or {k: tuple(v) for k, v in config['architecture']['schedules'].items()} != ARMS
            or config['capacity'] != {'frames': [24, 12], 'updates_per_cohort': 2,
                                      'process_limit_mib': PROCESS_MIB, 'hard_limit_mib': HARD_MIB}
            or config['architecture']['motion']['checkpoint_asset_chunk_size'] != 4
            or {'tokenrig_decoder', 'auxiliary_codec'} & config.keys()
            or 'opt_config_root' in config['runtime']):
        raise ValueError('standalone Stage 2 configuration differs from the approved scope')
    if 'recovery' in config:
        predecessor_config(config)


def read_evidence(spec):
    path = Path(spec['path'])
    if sha256(path) != spec['sha256']:
        raise RuntimeError('Stage 2 receipt changed: ' + str(path))
    value = json.loads(path.read_text())
    if not value.get('complete') or not value.get('passed'):
        raise RuntimeError('Stage 2 receipt is not accepted: ' + str(path))
    return value


def verify_evidence(op, config, mode, name, config_sha):
    validate_config(config)
    if mode == 'audit':
        if name != 'audit':
            raise ValueError('wrong Stage 2 audit output')
        return
    inherited = config.get('recovery')
    evidence_sha = inherited['config']['sha256'] if inherited else config_sha
    audit = read_evidence(op['stage_evidence']['audit'])
    if (audit['config_sha256'] != evidence_sha or audit['cuda_initialized']
            or audit['optimizer_updates'] or audit['plan']['samples'] != 100000
            or set(audit['initialization']) != ARMS.keys()):
        raise RuntimeError('Stage 2 CPU evidence does not match this run')
    if mode == 'capacity':
        if inherited:
            raise ValueError('recovery inherits the accepted capacity profiles')
        if name not in {capacity_stage(arm, size) for arm in ARMS for size in LONG_BATCHES}:
            raise ValueError('unknown Stage 2 capacity profile')
        return
    selected = op['capacity_selection']
    size = selected['long_batch_per_rank']
    if name not in ARMS or selected['frame_policy'] != frame_policy(size):
        raise ValueError('training must use the same accepted physical batches for both arms')
    if inherited:
        proof = read_evidence(op['stage_evidence']['recovery'])
        if (proof['config_sha256'] != config_sha or proof['optimizer_updates']
                or proof['cuda_initialized'] or proof['plan'] != audit['plan']
                or proof['predecessor'] != inherited['source']
                or proof['archive'] != config['evaluation']['panel_archive']
                or len(proof['inputs']) != config['evaluation']['valid_assets']):
            raise RuntimeError('standalone Stage 2 recovery audit does not match this run')
    for arm in ARMS:
        result = read_evidence(selected['results'][arm])
        resource = read_evidence(selected['resources'][arm])
        if (result['config_sha256'] != evidence_sha or result['source'] != resource['source']
                or result['arm'] != arm or result['long_batch_per_rank'] != size
                or result['initialization'] != audit['initialization'][arm]
                or len(result['profiles_by_rank']) != 2
                or any(len(rows) != 8 for rows in result['profiles_by_rank'])
                or result['optimizer_updates'] != 8 or not result['profile_weights_discarded']
                or set(resource['peak_gpu_memory_mib']) != {'3', '4'}
                or not all(0 < x <= ACCEPTED_WHOLE_GPU_MIB for x in resource['peak_gpu_memory_mib'].values())
                or not resource['payload_cleanup_complete']):
            raise RuntimeError('capacity did not bind both ranks and the selected batch profile')
        for rows in result['profiles_by_rank']:
            for row, frames in zip(rows, [24] * 4 + [12] * 4, strict=True):
                expected = frame_policy(size)[str(frames)]
                if (row['frames'] != frames or row['batch_per_rank'] != expected
                        or row['accumulation'] != 24 // expected or row['assets_in_update'] != 48
                        or row['input_frames_in_update'] != 48 * frames
                        or not all(math.isfinite(row[k]) for k in ('xyz_loss', 'bone_loss', 'gradient_norm'))
                        or row['max_reserved_bytes'] > PROCESS_MIB * 2**20):
                    raise RuntimeError('capacity proof differs from the exact admitted batch')
