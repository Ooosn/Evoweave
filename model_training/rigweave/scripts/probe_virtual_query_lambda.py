#!/usr/bin/env python3
"""Bounded paired auxiliary-weight comparison; no formal training entrypoint."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'model_training/rigweave/src'))
from virtual_query_xyz_runtime import build_model, fixed_dataset, prepare_batch, read_entries, sha256, write_json
from probe_virtual_query_xyz import evaluate, grad_norms, require_gradients
from virtual_frame_runtime import set_exposure_lr
from rigweave.dynamic_rig.aux_skeleton_tokens import AuxSkeletonTokens


def transfer(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {key: transfer(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(transfer(item, device) for item in value)
    return value


def cache_contract(config):
    return {key: config[key] for key in ('data', 'surface_sampling', 'auxiliary_codec')} | {
        'frames': config['comparison']['frames'],
        'selection': config['comparison']['selection'],
        'tokenizer_config_sha256': config['runtime']['tokenizer_config_sha256']}


def verify_operation(args):
    state = json.loads((ROOT / 'model_training/state/current.json').read_text())
    operation = state['active_operation']
    if state.get('architecture_contract', {}).get('id') != 'virtual_query_xyz_aux_ids_v1':
        raise ValueError('not the approved virtual-query architecture authority')
    if not operation.get('gpu_work_allowed') or operation.get('formal_training_allowed'):
        raise ValueError('no bounded comparison operation is authorized')
    if operation['mode'] != args.mode:
        raise ValueError('mode differs from the published bounded operation')
    runs = operation.get('runs', [operation])
    matches = [run for run in runs if Path(run['config']).resolve() == args.config.resolve()
               and Path(run['output']).resolve() == args.output.resolve()]
    if len(matches) != 1 or matches[0]['config_sha256'] != sha256(args.config):
        raise ValueError('exact config/output/hash not in the published operation')
    subprocess.run([sys.executable, str(ROOT / 'model_training/tools/agent_work_guard.py'),
        'check', '--operation', 'train', '--entrypoint', str(Path(__file__).resolve())], check=True)
    if subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
        raise ValueError('bounded comparison requires clean committed source')
    expected = operation['code_commit']
    subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', expected, 'HEAD'], check=True)
    subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet', expected, 'HEAD', '--',
                    'model_training/rigweave', 'model_training/tools'], check=True)
    return subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()


def load_selection(config):
    spec = config['comparison']['selection']
    if sha256(spec['path']) != spec['sha256']:
        raise ValueError('comparison cohort changed')
    selection = json.loads(Path(spec['path']).read_text())
    if selection['manifests'] != config['data']['manifests']:
        raise ValueError('comparison selection uses different manifests')
    entries = read_entries(config)
    lookup = {split: {row['asset_id']: row for row in rows} for split, rows in entries.items()}
    chosen = {}
    for group, count in [('train', 64), ('train_heldout', 8), ('valid', 32)]:
        metadata = selection['groups'][group]
        if len(metadata) != count or len({r['asset_id'] for r in metadata}) != count:
            raise ValueError(f'invalid fixed cohort size/identity for {group}')
        split = 'valid' if group == 'valid' else 'train'
        chosen[group] = []
        for item in metadata:
            row = lookup[split][item['asset_id']]
            if row['path'] != item['path']:
                raise ValueError('selected path differs from current immutable manifest')
            chosen[group].append(row)
    train = {row['asset_id'] for row in chosen['train']}
    if train & {row['asset_id'] for row in chosen['valid']}:
        raise ValueError('validation assets overlap training')
    if {row['path'] for row in chosen['train']} & {row['path'] for row in chosen['valid']}:
        raise ValueError('validation paths overlap training')
    if not {r['asset_id'] for r in chosen['train_heldout']} <= train:
        raise ValueError('heldout-pose assets must belong to the selected training cohort')
    return selection, chosen


def prepare_cache(config, output, device, source):
    from train_dynamic_rig import build_tokenizer
    upstream = config['runtime']['unirig_root']
    revision = subprocess.check_output(['git', '-C', upstream, 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', upstream, 'status', '--porcelain'], text=True).strip()
    if revision != config['initialization']['upstream_source_commit'] or dirty:
        raise ValueError('tokenizer source is not the pinned clean upstream revision')
    if sha256(config['runtime']['tokenizer_config']) != config['runtime']['tokenizer_config_sha256']:
        raise ValueError('tokenizer configuration changed')
    selection, groups = load_selection(config)
    tokenizer = build_tokenizer(Path(config['runtime']['tokenizer_config']))
    codec = AuxSkeletonTokens(**config['auxiliary_codec'])
    index = {'schema': 'virtual_query_geometry_cache_v1', 'contract': cache_contract(config),
             'source': source, 'learned_features_cached': False, 'groups': {},
             'selection_groups': selection['groups']}
    train_frames = {}
    for group, rows in groups.items():
        mode = 'heldout_eval' if group == 'train_heldout' else 'seen_eval'
        dataset = fixed_dataset(rows, config, config['comparison']['frames'], mode=mode)
        index['groups'][group] = []
        for number in range(len(dataset)):
            item = prepare_batch(dataset[number], config, tokenizer, codec, device)
            surface, targets, geometry = item
            asset = geometry['asset_id'][0]
            frames = geometry['selected_frames'][0].cpu().tolist()
            heldout = geometry['heldout_raw_ids'][0]
            if group == 'train' and set(frames) & set(heldout):
                raise ValueError('heldout pose leaked into training cache')
            if group == 'train':
                train_frames[asset] = frames
            if group == 'train_heldout' and (not set(frames) <= set(heldout) or set(frames) & set(train_frames[asset])):
                raise ValueError('heldout cache does not use disjoint reserved poses')
            if int(targets['main_attention_mask'].sum(-1).max()) > config['probe']['max_new_tokens'] + 2:
                raise ValueError('target exceeds the declared free-generation cap')
            filename = f'{group}_{number:03d}.pt'
            torch.save(transfer(item, 'cpu'), output / filename)
            record = {'asset_id': asset, 'filename': filename, 'sha256': sha256(output / filename),
                      'selected_frames': frames, 'heldout_raw_ids': heldout,
                      'main_token_lengths': targets['main_attention_mask'].sum(-1).cpu().tolist()}
            index['groups'][group].append(record)
            print('CACHE ' + json.dumps({'group': group, 'index': number, **record}), flush=True)
            del item, surface, targets, geometry
    write_json(output / 'cache_index.json', index)
    return {'accepted': True, 'execution_complete': True, 'stage': 'geometry_cache',
            'index': str(output / 'cache_index.json'), 'index_sha256': sha256(output / 'cache_index.json'),
            'cached_assets': sum(map(len, index['groups'].values())), 'learned_features_cached': False}


def load_cache(config):
    spec = config['comparison']['cache']
    index_path = Path(spec['index'])
    if sha256(index_path) != spec['sha256']:
        raise ValueError('common geometry cache index changed')
    index = json.loads(index_path.read_text())
    if index['schema'] != 'virtual_query_geometry_cache_v1' or index['contract'] != cache_contract(config):
        raise ValueError('cache preprocessing/selection contract differs')
    expected_counts = {'train': 64, 'train_heldout': 8, 'valid': 32}
    if set(index['groups']) != set(expected_counts) or any(
            len(index['groups'][group]) != count for group, count in expected_counts.items()):
        raise ValueError('cache groups do not match the fixed comparison cohort')
    groups = {}
    for group, records in index['groups'].items():
        groups[group] = []
        for number, record in enumerate(records):
            filename = record['filename']
            if filename != f'{group}_{number:03d}.pt':
                raise ValueError('unsafe cache entry filename')
            path = index_path.parent / filename
            if sha256(path) != record['sha256']:
                raise ValueError('cached raw geometry/targets changed')
            item = torch.load(path, map_location='cpu', weights_only=True)
            if item[2]['asset_id'] != [record['asset_id']] or item[2]['selected_frames'][0].tolist() != record['selected_frames']:
                raise ValueError('cache payload identity differs from its index')
            groups[group].append(item)
    return groups, index


def training_order(count, updates, seed):
    if (isinstance(count, bool) or not isinstance(count, int) or count < 1
            or isinstance(updates, bool) or not isinstance(updates, int) or updates < 1):
        raise ValueError('training order needs positive integer counts')
    rng = np.random.default_rng(seed)
    order = []
    while len(order) < updates:
        order.extend(rng.permutation(count).tolist())
    return order[:updates]


def parameter_fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(',', ':')).encode())
        digest.update(tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def shared_gradient_measurement(model, losses, weight):
    parameters = [model.motion_encoder.virtual_slots, model.motion_encoder.norm.weight]
    main = torch.autograd.grad(losses['main_loss'], parameters, retain_graph=True)
    auxiliary = (torch.autograd.grad(losses['auxiliary_loss'], parameters, retain_graph=True)
                 if weight else [torch.zeros_like(value) for value in main])
    report = {}
    for name, left, right in zip(('virtual_slots', 'motion_final_norm'), main, auxiliary):
        left, right = left.float(), right.float()
        a, b = float(left.norm()), float(right.norm())
        if not torch.isfinite(left).all() or not torch.isfinite(right).all() or a <= 0 or (weight and b <= 0):
            raise RuntimeError('invalid shared-gradient comparison')
        report[name] = {'main_norm': a, 'auxiliary_norm': b, 'weighted_aux_to_main_ratio': weight * b / a,
                        'cosine': float((left * right).sum()) / (a * b) if b > 0 else None}
    return report


def evaluate_groups(model, tokenizer, groups, index, config, output, step, device):
    heldout_ids = {item[2]['asset_id'][0] for item in groups['train_heldout']}
    evaluation = {'train_seen': [item for item in groups['train'] if item[2]['asset_id'][0] in heldout_ids],
                  'train_heldout': groups['train_heldout'], 'valid': groups['valid']}
    metadata = {item['asset_id']: item for group in index['selection_groups'].values() for item in group}
    report = {'step': step, 'groups': {}, 'rows': [], 'failed_generations_scored_zero': True,
              'generation_frame_indices': config['comparison']['generation_frame_indices']}
    for group, items in evaluation.items():
        directory = output / f'evaluation_{step:04d}' / group
        directory.mkdir(parents=True)
        result = evaluate(model, (transfer(item, device) for item in items), tokenizer, config, directory, step,
                          frame_indices=config['comparison']['generation_frame_indices'])
        report['groups'][group] = result['summary']
        for row in result['rows']:
            row.update(group=group, **{k: metadata[row['asset_id']][k] for k in
                ('integration_source', 'target_joint_count', 'topology_sha256',
                 'train_topology_frequency', 'selected_train_topology_frequency')})
            report['rows'].append(row)
    write_json(output / f'evaluation_{step:04d}.json', report)
    return report


def run_comparison(config, output, device, source):
    groups, index = load_cache(config)
    model, tokenizer, _, initial = build_model(config, device)
    initial.update(code_commit=source, config_sha256=sha256(config['_path']),
                   common_cache_index_sha256=config['comparison']['cache']['sha256'],
                   full_initial_state_sha256=parameter_fingerprint(model))
    write_json(output / 'initialization.json', initial)
    plan = config['comparison']
    updates, weight = plan['max_updates'], config['probe']['auxiliary_weight']
    if config['optimization']['schedule_samples'] != updates or len(groups['train']) != 64:
        raise ValueError('comparison schedule must match the declared 64-asset bounded budget')
    order = training_order(len(groups['train']), updates, plan['order_seed'])
    order_bytes = json.dumps(order, separators=(',', ':')).encode()
    write_json(output / 'training_order.json', {'indices': order, 'sha256': hashlib.sha256(order_bytes).hexdigest(),
               'assets': [item[2]['asset_id'][0] for item in groups['train']]})
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=config['optimization']['peak_lr'] / config['optimization']['div_factor'],
        betas=(.95, .999), weight_decay=config['optimization']['weight_decay'])
    if plan.get('initial_evaluation', True):
        evaluate_groups(model, tokenizer, groups, index, config, output, 0, device)
    started = time.monotonic()
    with (output / 'trace.jsonl').open('x', buffering=1) as trace:
        for step, item_index in enumerate(order, 1):
            surface, targets, geometry = transfer(groups['train'][item_index], device)
            optimizer.zero_grad(set_to_none=True)
            lr, beta1 = set_exposure_lr(optimizer, step - 1, config)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                losses = model(surface, targets, auxiliary_weight=weight)
            if not bool(torch.isfinite(losses['loss'])):
                raise RuntimeError('nonfinite comparison loss')
            balance = (shared_gradient_measurement(model, losses, weight)
                       if step in plan['gradient_measurement_updates'] else None)
            losses['loss'].backward()
            norms = grad_norms(model)
            require_gradients(model, norms, weight)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            row = {'step': step, 'asset_exposures': step, 'supervised_frame_exposures': step * plan['frames'],
                   'asset_id': geometry['asset_id'][0], 'loss': float(losses['loss']),
                   'main_loss': float(losses['main_loss']), 'auxiliary_loss': float(losses['auxiliary_loss']),
                   'main_accuracy': float(losses['main_accuracy']), 'auxiliary_accuracy': float(losses['auxiliary_accuracy']),
                   'lr': lr, 'beta1': beta1, 'gradient_norm': float(norm), 'groups': norms,
                   'shared_gradient_measurement': balance, 'elapsed_seconds': time.monotonic() - started,
                   'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30}
            trace.write(json.dumps(row) + '\n')
            if step % 32 == 0:
                print('COMPARE ' + json.dumps(row), flush=True)
            del losses, surface, targets, geometry
            if step in plan['evaluation_updates'] or step == updates:
                last = evaluate_groups(model, tokenizer, groups, index, config, output, step, device)
                write_json(output / 'progress.json', {'step': step, 'groups': last['groups'],
                           'formal_training_allowed': False})
    torch.save({'schema': 'virtual_query_xyz_aux_ids_v1', 'model': model.state_dict(),
        'optimizer': optimizer.state_dict(), 'config': config, 'step': updates,
        'initialization': initial, 'source': source, 'diagnostic_only_not_formal_initialization': True},
        output / 'checkpoint_comparison.pt')
    return {'stage': 'lambda_comparison', 'execution_complete': True, 'accepted': False,
            'accepted_meaning': 'quality_and_lambda_selection_require_primary_paired_review',
            'completed_updates': updates, 'asset_exposures': updates, 'auxiliary_weight': weight,
            'initial_evaluation_performed': plan.get('initial_evaluation', True),
            'evaluation': last['groups'], 'training_order_sha256': hashlib.sha256(order_bytes).hexdigest(),
            'common_cache_index_sha256': config['comparison']['cache']['sha256']}


def run_smoke(config, output, device, source):
    groups, _ = load_cache(config)
    model, tokenizer, _, initial = build_model(config, device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.)
    weight = config['probe']['auxiliary_weight']
    if weight <= 0:
        raise ValueError('comparison smoke must exercise both losses')
    order = sorted(range(len(groups['train'])), key=lambda i: int(groups['train'][i][1]['main_lengths'].max()))
    reports = []
    for item_index in (order[0], order[-1]):
        surface, targets, geometry = transfer(groups['train'][item_index], device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            losses = model(surface, targets, auxiliary_weight=weight)
        if not bool(torch.isfinite(losses['loss'])):
            raise RuntimeError('nonfinite comparison smoke loss')
        balance = shared_gradient_measurement(model, losses, weight)
        losses['loss'].backward()
        norms = grad_norms(model)
        require_gradients(model, norms, weight)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        reports.append({'asset_id': geometry['asset_id'][0], 'loss': float(losses['loss']),
            'main_loss': float(losses['main_loss']), 'auxiliary_loss': float(losses['auxiliary_loss']),
            'gradients': norms, 'shared_gradient_measurement': balance, 'gradient_norm': float(norm),
            'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30})
        del losses, surface, targets, geometry
    directory = output / 'evaluation'
    directory.mkdir()
    evaluation = evaluate(model, (transfer(groups['train'][i], device) for i in (order[0], order[-1])),
        tokenizer, config, directory, 2, frame_indices=[0, 1])
    return {'stage': 'comparison_execution_smoke', 'accepted': True, 'execution_complete': True,
            'model_fit_accepted': False, 'cases': reports, 'free_generation': evaluation['summary'],
            'initialization': initial, 'code_commit': source, 'weights_discarded': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['cache', 'smoke', 'compare'], required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    source = verify_operation(args)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(config['runtime']['physical_gpu']):
        raise ValueError('physical GPU differs from exact configuration')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('requires exactly one granted visible GPU')
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.95, device)
    args.output.mkdir(parents=True, exist_ok=False)
    config['_path'] = str(args.config)
    try:
        function = {'cache': prepare_cache, 'smoke': run_smoke, 'compare': run_comparison}[args.mode]
        result = function(config, args.output, device, source)
        write_json(args.output / 'result.json', {**result, 'formal_training_allowed': False,
            'finished_utc': datetime.now(timezone.utc).isoformat(), 'code_commit': source})
    except Exception:
        write_json(args.output / 'result.json', {'execution_complete': False, 'accepted': False,
                   'formal_training_allowed': False, 'traceback': traceback.format_exc(), 'code_commit': source})
        raise


if __name__ == '__main__':
    main()
