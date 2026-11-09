"""Audit and restore the stopped Stage 2 run without modifying its artifacts."""
import ast
import json
from pathlib import Path
import subprocess
import time

import torch

from anchor_evaluation_inputs import archived_panel, panel_digest, read_record, verify_source
from anchor_full_data import make_stream
from anchor_stage2_protocol import ROOT, predecessor_config, sha256
from reference_flow_runtime import mirror
from reference_flow_protocol import scientific_sha
from virtual_pair_runtime import apply_rng, parameter_names


def verify_training_source(source):
    scripts = 'model_training/rigweave/scripts/'
    subprocess.run(['git', '-C', str(ROOT), 'diff', '--exit-code', source, 'HEAD', '--',
        'model_training/rigweave/src', *(scripts + name for name in (
            'anchor_full_data.py', 'id_xyz_stages.py', 'reference_flow_protocol.py',
            'reference_flow_runtime.py', 'virtual_fusion_normal_runtime.py',
            'virtual_pair_loading.py', 'virtual_pair_runtime.py', 'virtual_query_xyz_runtime.py'))],
        check=True)
    for name, functions in (
        ('train_anchor_stage2.py', ('context', 'initialize', 'build_live', 'update', 'prefetch', 'plan')),
        ('anchor_joint_runtime.py', ('clip_branch_gradients', 'reference_indices', 'head_digest')),
    ):
        path = scripts + name
        old = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{source}:{path}'], text=True)

        def definitions(text):
            return {node.name: ast.dump(node) for node in ast.parse(text).body
                    if isinstance(node, ast.FunctionDef) and node.name in functions}

        if definitions(old) != definitions((ROOT/path).read_text(encoding='utf-8')):
            raise ValueError(f'training functions changed in {name}')


def checkpoint_receipts(config):
    recovery = config['recovery']
    receipts = []
    for samples, spec in zip((48, recovery['samples']), recovery['checkpoints'], strict=True):
        if sha256(spec['path']) != spec['sha256']:
            raise ValueError('retained Stage 2 checkpoint receipt changed')
        receipt = json.loads(Path(spec['path']).read_text())
        if (not receipt['complete'] or not receipt['both_copies_verified']
                or receipt['additional_asset_exposures'] != samples
                or receipt['code_commit'] != recovery['source']
                or Path(receipt['durable_path']).stat().st_size != receipt['bytes']
                or sha256(receipt['path']) != receipt['sha256']):
            raise ValueError('retained Stage 2 checkpoint changed')
        receipts.append(receipt)
    return receipts


def checkpoint_state(config, base, proof, admitted, rank):
    recovery = config['recovery']
    arm, samples = recovery['arm'], recovery['samples']
    receipt = proof['checkpoints'][-1]
    saved = torch.load(receipt['path'], map_location='cpu', mmap=True, weights_only=False)
    previous = predecessor_config(config)
    previous['training'].update(frame_policy=base['training']['frame_policy'],
                                backbone_exposure_offset=0, arm=arm)
    previous['artifacts']['durable_root'] += '/' + arm
    frames, step = saved['input_frame_exposures'], saved['step']
    if (saved['code_commit'] != recovery['source'] or saved['configuration'] != previous
            or saved['scientific_sha256'] != scientific_sha(previous)
            or saved['initialization'] != admitted['initialization'][arm]
            or saved['additional_asset_exposures'] != samples
            or saved['global_backbone_asset_exposures'] != samples
            or admitted['plan']['milestones'][str(samples)] != {'input_frame_visits': frames, 'updates': step}
            or [r['rank'] for r in saved['rank_states']] != [0, 1]
            or not saved['optimizer']['state']):
        raise ValueError('retained Stage 2 model/optimizer/rank state does not match its admission')
    state = saved['rank_states'][rank]
    stream = make_stream(base, rank, 2, state['stream'])
    if stream.cursor * 2 != samples or stream.plan_sha256 != admitted['plan']['plan_sha256']:
        raise ValueError('retained Stage 2 sampler position differs')
    return saved, stream, state, samples, frames, step


def audit_recovery(config, base, fixed, admitted):
    started = time.monotonic()
    recovery = config['recovery']
    verify_training_source(recovery['source'])
    archive = config['evaluation']['panel_archive']
    panel = archived_panel(archive, fixed.entries, base,
        Path(config['data_cache']['root'])/'evaluation_inputs', verify_archive=True)
    references = config['evaluation']['reference_slots']
    pairs = set()
    for index, spec in enumerate(panel):
        record = read_record(spec, verify=True)
        verify_source(record, fixed[index])
        raw = record['raw_frames']
        pairs.update((record['asset_id'], ref, raw[ref], slot, frame)
                     for ref in references for slot, frame in enumerate(raw) if slot != ref)
    previous = recovery['previous_validation']
    if sha256(previous['path']) != previous['sha256']:
        raise ValueError('previous Stage 2 evaluation rows changed')
    old = [(r['asset_id'], r['reference_slot'], r['reference_raw_frame'], r['target_slot'], r['target_raw_frame'])
           for r in json.loads(Path(previous['path']).read_text()) if r['split'] == 'valid' and r['method'] == 'head']
    if set(old) != pairs or len(old) != len(pairs):
        raise ValueError('frozen panel changed the stopped run\'s evaluation frames')

    checkpoints = checkpoint_receipts(config)
    traces = []
    for rank, spec in enumerate(recovery['traces']):
        path = Path(spec['path'])
        if sha256(path) != spec['sha256']:
            raise ValueError('previous Stage 2 training trace changed')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        saved, _, state, samples, frames, step = checkpoint_state(config, base,
            {'checkpoints': checkpoints}, admitted, rank)
        if ([r['step'] for r in rows] != list(range(1, step + 1))
                or rows[-1]['samples'] != samples or rows[-1]['input_frame_visits'] != frames
                or not {'python_rng', 'numpy_rng', 'torch_rng', 'cuda_rng'} <= state.keys()):
            raise ValueError('previous rank trace does not cover the restored optimizer steps')
        traces.append(mirror(path, f'predecessor/{path.name}', config))
        adam_states = len(saved['optimizer']['state'])
        del saved
    return {'complete': True, 'passed': True, 'cuda_initialized': torch.cuda.is_initialized(),
        'optimizer_updates': 0, 'predecessor': recovery['source'], 'plan': admitted['plan'],
        'archive': archive, 'inputs': panel, 'input_panel_sha256': panel_digest(panel),
        'matched_target_rows': len(pairs), 'checkpoints': checkpoints, 'traces': traces,
        'adam_states': adam_states, 'elapsed_seconds': time.monotonic()-started}


def restore(config, base, proof, admitted, model, optimizer, rank):
    saved, stream, state, samples, frames, step = checkpoint_state(config, base, proof, admitted, rank)
    if saved['optimizer_parameter_names'] != parameter_names(model):
        raise ValueError('optimizer parameter order changed on Stage 2 restore')
    model.load_state_dict(saved['model'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    if len(optimizer.state) != proof['adam_states']:
        raise ValueError('incomplete Stage 2 Adam restoration')
    apply_rng(state)
    return stream, samples, frames, step
