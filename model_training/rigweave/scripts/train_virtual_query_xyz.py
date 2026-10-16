#!/usr/bin/env python3
"""Measured-capacity and fresh full-data execution for virtual-query XYZ."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'model_training/rigweave/src'))
from virtual_query_xyz_runtime import build_model, fixed_dataset, read_entries, sha256, write_json
from virtual_frame_runtime import FrameBudgetStream, normalize_asset_gradients, set_exposure_lr
from probe_virtual_query_xyz import evaluate, grad_norms, require_gradients
from fusion_evaluation import auxiliary_summary
from rigweave.dynamic_rig.frame_batch_sampler import FrameSampleRequest
from rigweave.dynamic_rig.virtual_frame_data import prepare_virtual_surface
from rigweave.dynamic_rig.virtual_query_xyz_data import build_xyz_aux_targets


def verify_operation(args, config, rank):
    state = json.loads((ROOT / 'model_training/state/current.json').read_text())
    operation = state['active_operation']
    if state.get('architecture_contract', {}).get('id') != 'virtual_query_xyz_aux_ids_v1':
        raise ValueError('wrong architecture contract')
    if not operation.get('gpu_work_allowed') or operation.get('complete'):
        raise ValueError('no active GPU operation')
    if bool(operation.get('formal_training_allowed')) != (args.mode == 'train'):
        raise ValueError('formal authorization does not match this operation')
    if operation['mode'] != args.mode or Path(operation['config']).resolve() != args.config.resolve():
        raise ValueError('stage/config differs from publication')
    if operation['config_sha256'] != sha256(args.config) or Path(operation['output']).resolve() != args.output.resolve():
        raise ValueError('config hash/output differs from publication')
    if rank == 0:
        subprocess.run([sys.executable, str(ROOT / 'model_training/tools/agent_work_guard.py'),
                        'check', '--operation', 'train', '--entrypoint', str(Path(__file__).resolve())], check=True)
    if subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
        raise ValueError('GPU execution requires clean committed code')
    expected = operation['code_commit']
    subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', expected, 'HEAD'], check=True)
    subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet', expected, 'HEAD', '--',
                    'model_training/rigweave', 'model_training/tools'], check=True)
    if args.mode == 'train':
        release = config['formal_release']
        if not release.get('approved') or not all(release.get(x) for x in ('g2_receipt', 'g3_receipt', 'capacity_receipt')):
            raise ValueError('formal configuration requires reviewed G2/G3/capacity evidence')
        for name in ('g2_receipt', 'g3_receipt', 'capacity_receipt'):
            receipt = release[name]
            if sha256(receipt['path']) != receipt['sha256']:
                raise ValueError(f'{name} changed')
            if not json.loads(Path(receipt['path']).read_text()).get('accepted'):
                raise ValueError(f'{name} was not accepted')
        validate_capacity_receipt(config, json.loads(Path(release['capacity_receipt']['path']).read_text()))
    return subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()


def validate_plan(config, world):
    plan = config['training']
    target, effective = plan['target_assets'], plan['effective_assets_per_update']
    if any(type(x) is not int or x < 1 for x in (target, effective, world)):
        raise ValueError('asset budgets and world size must be positive integers')
    if effective != 48 or world != 2:
        raise ValueError('formal execution requires 48 actual assets per update across exactly two ranks')
    if target != 80016 or target != config['optimization']['schedule_samples']:
        raise ValueError('formal execution preserves the approved 80016 asset exposure budget')
    if effective % world or target % world:
        raise ValueError('global asset budgets must divide evenly across ranks')
    if (config['data']['frames_min'], config['data']['frames_max']) != (2, 24):
        raise ValueError('formal frame range must remain 2..24')
    if (type(plan['frame_budget']) is not int or plan['frame_budget'] < 24 or
            type(plan['batch_cap']) is not int or plan['batch_cap'] < 1):
        raise ValueError('formal frame budget/batch cap must cover every frame count')
    if config['data']['manifests']['train']['rows'] != 18002 or config['data']['manifests']['valid']['rows'] != 967:
        raise ValueError('formal dataset membership differs')
    milestones = sorted(set(plan['checkpoint_assets'] + plan['evaluation_assets'] + [target]))
    if any(type(x) is not int or x <= 0 or x > target or x % world for x in milestones):
        raise ValueError('milestones must be positive rank-divisible asset counts within budget')
    if config['architecture'].get('frame_residual_scale', 0.) != 0.:
        raise ValueError('diagnostic residual intervention is not the approved formal architecture')
    return milestones


def update_asset_target(samples, target, effective, milestones, world):
    boundary = min(x for x in milestones if x > samples)
    size = min(effective, target - samples, boundary - samples)
    if size <= 0 or size % world:
        raise ValueError('invalid exact update budget')
    return size // world


def prepare(dataset, requests, config, tokenizer, codec, device):
    batches = [dataset[request] for request in requests]
    surface, geometry = prepare_virtual_surface(batches, device, **config['surface_sampling'])
    target = build_xyz_aux_targets(geometry, tokenizer, codec, device=device,
                                  max_main_tokens=config['data']['max_main_tokens'])
    return surface, target, geometry


def optimizer_for(model, config):
    plan = config['optimization']
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=plan['peak_lr'] / plan['div_factor'], betas=(.95, .999),
                            weight_decay=plan['weight_decay'])


def wrap_model(model, device, world):
    return DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True,
                                   gradient_as_bucket_view=True) if world > 1 else model


def load_capacity_selection(config, rows):
    spec = config['capacity']['selection']
    if sha256(spec['path']) != spec['sha256']:
        raise ValueError('capacity stress selection changed')
    selection = json.loads(Path(spec['path']).read_text())
    if selection['status'] != 'passed' or selection['manifest'] != config['data']['manifests']['train']:
        raise ValueError('capacity selection is not bound to the active training manifest')
    lookup = {(row['asset_id'], row['path']): i for i, row in enumerate(rows)}
    groups, lengths = {}, {}
    metadata = {**selection['groups'], **{f'geometry_t{key}': [row]
                for key, row in selection['geometry_by_frames'].items()}}
    for group, values in metadata.items():
        indices = []
        for value in values:
            key = (value['asset_id'], value['path'])
            if key not in lookup or value['manifest_row_index'] != lookup[key]:
                raise ValueError('capacity stress asset identity/index differs from the manifest')
            if type(value['main_tokens']) is not int or not 3 <= value['main_tokens'] <= config['data']['max_main_tokens']:
                raise ValueError('capacity target length exceeds the approved decoder budget')
            if key in lengths and lengths[key] != value['main_tokens']:
                raise ValueError('conflicting capacity target lengths')
            lengths[key] = value['main_tokens']
            indices.append(lookup[key])
        if not indices or len(set(indices)) != len(indices):
            raise ValueError('empty or duplicated capacity stress group')
        groups[group] = indices
    required = {'longest_main', 'largest_vertices', 'largest_faces', 'largest_geometry'} | {
        f'geometry_t{frames}' for frames in range(2, 25)}
    if (not required <= set(groups) or set(groups) - required - {'largest_auxiliary'}
            or len(groups['longest_main']) != 3):
        raise ValueError('capacity selection does not cover all required token/geometry extremes')
    return groups, lengths


def capacity_contract(config):
    return {'architecture': config['architecture'], 'initialization': config['initialization'],
            'execution_route': config.get('reference_route', config['architecture'].get('fusion_mode', 'virtual_queries_real')),
            'auxiliary_codec': config['auxiliary_codec'], 'surface_sampling': config['surface_sampling'],
            'data': {key: config['data'][key] for key in
                     ('manifests', 'max_main_tokens', 'frame_seed', 'surface_seed', 'heldout_frames')},
            'auxiliary_weight': config['optimization']['auxiliary_weight'],
            'physical_gpus': config['runtime']['physical_gpus'], 'precision': 'bfloat16_autocast',
            'optimizer': 'AdamW', 'allocator_fraction': .95, 'world_size': len(config['runtime']['physical_gpus']),
            'stress_selection_sha256': config['capacity']['selection']['sha256']}


def validate_capacity_receipt(config, receipt):
    if receipt.get('capacity_contract') != capacity_contract(config):
        raise ValueError('formal runtime/model differs from the measured capacity contract')
    cases = receipt.get('approved_cases', [])
    covered = {(case['frames'], case['batch_per_rank'], case['stress_group']) for case in cases
               if case.get('ranks_approved') == [0, 1] and case.get('accumulation_microbatches') == 2}
    plan = config['training']
    for frames in range(2, 25):
        batch = min(plan['batch_cap'], plan['frame_budget'] // frames)
        for group in ('longest_main', f'geometry_t{frames}'):
            if (frames, batch, group) not in covered:
                raise ValueError('formal T/B schedule lacks measured token and geometry capacity coverage')


def run_profile(model, tokenizer, codec, initial, entries, config, args, device, rank, world, source,
                *, gradient_check=None):
    gradient_check = require_gradients if gradient_check is None else gradient_check
    plan = config['capacity']
    if not plan['cases'] or not 0 < plan['allocated_fraction_limit'] <= .90:
        raise ValueError('capacity measurement requires cases and at least ten percent allocated headroom')
    for case in plan['cases']:
        if (type(case['frames']) is not int or not 2 <= case['frames'] <= 24 or
                type(case['batch_per_rank']) is not int or case['batch_per_rank'] < 1):
            raise ValueError('invalid capacity case')
    dataset = fixed_dataset(entries['train'], config, 2, mode='train')
    optimizer = optimizer_for(model, config)
    wrapped = wrap_model(model, device, world)
    groups, lengths = load_capacity_selection(config, entries['train'])
    report = {'complete': False, 'accepted': False, 'capacity_only_no_weights_reused': True,
              'rank': rank, 'world_size': world, 'initialization': initial, 'source': source,
              'gpu_total_bytes': torch.cuda.get_device_properties(device).total_memory, 'cases': [],
              'capacity_contract': capacity_contract(config)}
    report_path = args.output / f'capacity_rank_{rank}.json'
    write_json(report_path, report)
    model.train()
    for number, case in enumerate(config['capacity']['cases']):
        frames, batch = case['frames'], case['batch_per_rank']
        indices = groups[case['stress_group']]
        requests = [FrameSampleRequest(indices[i % len(indices)], frames, 77000 + number * 100 + i)
                    for i in range(batch)]
        measurements = []
        for iteration in range(2):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.monotonic()
            losses = []
            for micro in range(2):
                surface, target, geometry = prepare(dataset, requests, config, tokenizer, codec, device)
                for index, (asset, asset_path) in enumerate(zip(geometry['asset_id'], geometry['path'])):
                    expected = lengths[(asset, asset_path)]
                    if not bool((target['main_attention_mask'][index].sum(-1) == expected).all()):
                        raise ValueError('actual XYZ tokenizer length differs from the stress certificate')
                context = wrapped.no_sync() if world > 1 and micro == 0 else nullcontext()
                with context:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        out = wrapped(surface, target, auxiliary_weight=config['optimization']['auxiliary_weight'])
                    if not bool(torch.isfinite(out['loss'])):
                        raise RuntimeError('nonfinite capacity loss')
                    (out['loss'] / 2).backward()
                losses.append(float(out['loss'].detach()))
                selected_assets = geometry['asset_id']
                maximum_tokens = int(target['main_attention_mask'].sum(-1).max())
                maximum_auxiliary_tokens = int(target['aux_lengths'].max()) if 'aux_lengths' in target else None
                maximum_joints = int(geometry['joint_count'].max())
                del surface, target, geometry, out
            norms = grad_norms(model)
            gradient_check(model, norms, config['optimization']['auxiliary_weight'])
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimization']['clip_grad_norm'],
                                                 error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize(device)
            measurements.append({'optimizer_step': iteration + 1, 'loss': float(np.mean(losses)),
                                 'accumulation_microbatches': 2,
                                 'gradient_norm': float(norm), 'seconds': time.monotonic() - started,
                                 'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                                 'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
                                 'maximum_main_tokens': maximum_tokens,
                                 'maximum_auxiliary_tokens': maximum_auxiliary_tokens,
                                 'maximum_joints': maximum_joints, 'assets': selected_assets})
        accepted = all(x['peak_allocated_bytes'] <= config['capacity']['allocated_fraction_limit'] *
                       report['gpu_total_bytes'] for x in measurements)
        consensus = torch.tensor(int(accepted), device=device)
        if world > 1:
            dist.all_reduce(consensus, op=dist.ReduceOp.MIN)
        report['cases'].append({**case, 'measurements': measurements, 'accepted': bool(consensus.item())})
        write_json(report_path, report)
        print('CAPACITY ' + json.dumps({'rank': rank, **report['cases'][-1]}), flush=True)
        if not bool(consensus.item()):
            break
    report.update(complete=True, accepted=len(report['cases']) == len(config['capacity']['cases']) and
                  all(x['accepted'] for x in report['cases']))
    write_json(report_path, report)


def selected_evaluation_rows(entries, config):
    spec = config['evaluation']['selection']
    if sha256(spec['path']) != spec['sha256']:
        raise ValueError('reviewed evaluation membership changed')
    selection = json.loads(Path(spec['path']).read_text())
    if selection['manifests'] != config['data']['manifests']:
        raise ValueError('evaluation selection uses other manifests')
    result = {}
    for group, split in [('train_heldout', 'train'), ('valid', 'valid')]:
        lookup = {(row['asset_id'], row['path']): row for row in entries[split]}
        chosen = selection['groups'][group]
        keys = [(row['asset_id'], row['path']) for row in chosen]
        if len(set(keys)) != len(keys) or any(key not in lookup for key in keys):
            raise ValueError('evaluation selection contains duplicate or foreign rows')
        result[group] = [lookup[key] for key in keys]
    if len(result['train_heldout']) != 8 or len(result['valid']) != 32:
        raise ValueError('periodic evaluation requires the reviewed 8 heldout / 32 validation assets')
    return result


def evaluation_summary(rows):
    values = {key: float(np.mean([row[key] for row in rows])) for key in
              ('success', 'quantized_edge_f1', 'original_edge_f1', 'exact_count', 'main_ce', 'main_accuracy')}
    distances = [row['quantized_j2j'] for row in rows if row['quantized_j2j'] is not None]
    values.update(views=len(rows), geometry_scored_views=len(distances),
                  quantized_j2j_successful_views=float(np.mean(distances)) if distances else None)
    return values


def run_evaluation(model, tokenizer, codec, entries, config, args, device, samples, rank, world):
    chosen = selected_evaluation_rows(entries, config)
    if samples == config['training']['target_assets']:
        chosen['valid'] = entries['valid']
    root = args.output / f'evaluation_{samples:06d}'
    root.mkdir(exist_ok=True)
    report = {'complete': False, 'samples': samples, 'rank': rank, 'groups': {}}
    for group, rows in chosen.items():
        local = rows[rank::world]
        dataset = fixed_dataset(local, config, config['evaluation']['frames'],
                                mode='heldout_eval' if group == 'train_heldout' else 'seen_eval')
        def items():
            for i in range(len(dataset)):
                yield prepare(dataset, [i], config, tokenizer, codec, device)
        directory = root / f'{group}_rank_{rank}'
        directory.mkdir(exist_ok=False)
        eval_config = {**config, 'probe': {'auxiliary_weight': config['optimization']['auxiliary_weight'],
                                         'max_new_tokens': config['evaluation']['max_new_tokens'],
                                         'evaluate_auxiliary': True}}
        result = evaluate(model, items(), tokenizer, eval_config, directory, samples,
                          frame_indices=config['evaluation']['generated_frame_indices'])
        report['groups'][group] = result
        write_json(root / f'rank_{rank}.json', report)
    report['complete'] = True
    write_json(root / f'rank_{rank}.json', report)
    if world > 1:
        dist.barrier()
    if rank == 0:
        ranks = [json.loads((root / f'rank_{r}.json').read_text()) for r in range(world)]
        combined = {group: [row for result in ranks for row in result['groups'][group]['rows']]
                    for group in chosen}
        auxiliary = {group: [row for result in ranks for row in result['groups'][group]['auxiliary_rows']]
                     for group in chosen}
        for group, rows in auxiliary.items():
            if sorted(row['asset_id'] for row in rows) != sorted(row['asset_id'] for row in chosen[group]):
                raise ValueError('auxiliary evaluation must contain every selected asset exactly once')
        write_json(root / 'summary.json', {'complete': True, 'samples': samples,
                   'summary': {group: evaluation_summary(rows) for group, rows in combined.items()},
                   'auxiliary_summary': {group: {'assets': len(rows), **auxiliary_summary(rows)}
                                         for group, rows in auxiliary.items()},
                   'failed_generations_scored_zero': True, 'data_used_for_optimization': False,
                   'generated_frame_indices': config['evaluation']['generated_frame_indices'],
                   'full_validation': samples == config['training']['target_assets']})
    if world > 1:
        dist.barrier()
    model.train()


def save_checkpoint(model, optimizer, stream, config, args, samples, frames, step, initial, source, rank, world):
    local = {'stream': stream.state_dict(), 'torch_rng': torch.get_rng_state(),
             'cuda_rng': torch.cuda.get_rng_state(), 'numpy_rng': np.random.get_state(),
             'python_rng': random.getstate()}
    ranks = [None] * world
    if world > 1:
        dist.all_gather_object(ranks, local)
    else:
        ranks[0] = local
    if rank == 0:
        path = args.output / f'checkpoint_asset_{samples:06d}.pt'
        temporary = path.with_suffix('.pt.tmp')
        torch.save({'schema': 'virtual_query_xyz_aux_ids_v1', 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'config': config, 'architecture': model.configuration(),
                    'asset_exposures': samples, 'supervised_frame_exposures': frames, 'step': step,
                    'rank_states': ranks, 'initialization': initial, 'source': source,
                    'purpose': config.get('purpose', 'formal_full'),
                    'diagnostic_only_not_formal_initialization': config.get('purpose', 'formal_full') != 'formal_full'}, temporary)
        temporary.replace(path)
        write_json(args.output / 'latest_checkpoint.json', {'path': str(path), 'bytes': path.stat().st_size,
                                                          'asset_exposures': samples, 'step': step})
    if world > 1:
        dist.barrier()


def run_training(model, tokenizer, codec, initial, entries, config, args, device, rank, world, source,
                 *, stream=None, milestones=None, evaluation_callback=None, gradient_check=None,
                 sample_recorder=None, resume_state=None):
    milestones = validate_plan(config, world) if milestones is None else milestones
    evaluation_callback = run_evaluation if evaluation_callback is None else evaluation_callback
    gradient_check = require_gradients if gradient_check is None else gradient_check
    plan = config['training']
    dataset = fixed_dataset(entries['train'], config, 2, mode='train')
    optimizer = optimizer_for(model, config)
    if resume_state is not None and stream is None:
        raise ValueError('continuation requires an explicitly restored exposure stream')
    if stream is None:
        stream = FrameBudgetStream(dataset, rank=rank, world_size=world, seed=config['data']['stream_seed'],
                                   frame_budget=plan['frame_budget'], batch_cap=plan['batch_cap'],
                                   min_frames=config['data']['frames_min'], max_frames=config['data']['frames_max'])
    wrapped = wrap_model(model, device, world)
    samples = frames_seen = step = 0
    if resume_state is not None:
        from virtual_fusion_continuation import restore_training_state, set_continuation_lr
        samples, frames_seen, step = restore_training_state(
            optimizer, stream, resume_state, device=device, rank=rank, world=world,
            target_assets=plan['target_assets'])
    starting_step = step
    report = {'complete': False, 'rank': rank, 'world_size': world, 'source': source,
              'initialization': initial, 'asset_exposures': samples, 'supervised_frame_exposures': frames_seen,
              'step': step, 'fresh_official_initialization': resume_state is None, 'evaluations': []}
    if resume_state is not None:
        report['continuation'] = resume_state['provenance']
        report['optimizer_state_restored'] = True
        report['restored_optimizer'] = {
            'state_entries': len(optimizer.state),
            'param_groups': [{'lr': group['lr'], 'betas': list(group['betas'])}
                             for group in optimizer.param_groups]}
    write_json(args.output / f'rank_{rank}.json', report)
    started = time.monotonic()
    model.train()
    with (args.output / f'trace_rank_{rank}.jsonl').open('x', buffering=1) as trace:
        while samples < plan['target_assets']:
            local_goal = update_asset_target(samples, plan['target_assets'], plan['effective_assets_per_update'],
                                             milestones, world)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(device)
            lr, beta1 = (set_exposure_lr(optimizer, samples, config) if resume_state is None else
                         set_continuation_lr(optimizer, resume_state['learning_rate']))
            totals = torch.zeros(6, dtype=torch.float64, device=device)
            local_assets, schedule = 0, []
            tick = time.monotonic()
            while local_assets < local_goal:
                requests = stream.take(local_goal - local_assets)
                size, frames = len(requests), requests[0].frames
                last = local_assets + size == local_goal
                surface, target, geometry = prepare(dataset, requests, config, tokenizer, codec, device)
                context = wrapped.no_sync() if world > 1 and not last else nullcontext()
                with context:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        out = wrapped(surface, target, auxiliary_weight=config['optimization']['auxiliary_weight'])
                    if not bool(torch.isfinite(out['loss'])):
                        raise RuntimeError(f'nonfinite training loss at update {step + 1}')
                    out['loss_per_asset'].sum().backward()
                totals += torch.stack((out['loss'].detach() * size, out['main_loss'].detach() * size,
                                       out['auxiliary_loss'].detach() * size, out['main_accuracy'].detach() * size,
                                       totals.new_tensor(size), totals.new_tensor(size * frames)))
                local_assets += size
                record = {'assets_per_rank': size, 'frames_per_asset': frames}
                if sample_recorder is not None:
                    record.update(sample_recorder(geometry, requests))
                schedule.append(record)
                del surface, target, geometry, out
            if world > 1:
                dist.all_reduce(totals)
            update_assets, update_frames = int(totals[4]), int(totals[5])
            if update_assets != local_goal * world:
                raise RuntimeError('distributed actual asset count differs from the requested update')
            normalize_asset_gradients(model.parameters(), global_assets=update_assets, world_size=world)
            if step == starting_step:
                norms = grad_norms(model)
                gradient_check(model, norms, config['optimization']['auxiliary_weight'])
                report['first_gradient_groups'] = norms
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimization']['clip_grad_norm'],
                                                 error_if_nonfinite=True)
            optimizer.step()
            samples += update_assets
            frames_seen += update_frames
            step += 1
            torch.cuda.synchronize(device)
            item = {'step': step, 'asset_exposures': samples, 'supervised_frame_exposures': frames_seen,
                    'assets_in_update': update_assets, 'microbatches': schedule,
                    'loss': float(totals[0] / update_assets), 'main_loss': float(totals[1] / update_assets),
                    'auxiliary_loss': float(totals[2] / update_assets), 'main_accuracy': float(totals[3] / update_assets),
                    'gradient_norm': float(norm), 'lr': lr, 'beta1': beta1,
                    'update_seconds': time.monotonic() - tick, 'elapsed_seconds': time.monotonic() - started,
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}
            trace.write(json.dumps(item) + '\n')
            report.update(asset_exposures=samples, supervised_frame_exposures=frames_seen, step=step, latest=item)
            write_json(args.output / f'rank_{rank}.json', report)
            print('TRAIN ' + json.dumps({'rank': rank, **item}), flush=True)
            if samples in milestones:
                save_checkpoint(model, optimizer, stream, config, args, samples, frames_seen, step,
                                initial, source, rank, world)
            if samples in plan['evaluation_assets'] or samples == plan['target_assets']:
                evaluation_callback(model, tokenizer, codec, entries, config, args, device, samples, rank, world)
                report['evaluations'].append(samples)
    report.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(args.output / f'rank_{rank}.json', report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['profile', 'train'], required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    rank, world, local = (int(os.environ.get(key, default)) for key, default in
                          [('RANK', 0), ('WORLD_SIZE', 1), ('LOCAL_RANK', 0)])
    source = verify_operation(args, config, rank)
    physical = ','.join(str(x) for x in config['runtime']['physical_gpus'])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != physical or world != len(config['runtime']['physical_gpus']):
        raise ValueError('visible devices/world size differ from the published resource contract')
    if not torch.cuda.is_available() or torch.cuda.device_count() != world:
        raise ValueError('CUDA device count differs from the granted resources')
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.cuda.set_per_process_memory_fraction(.95, device)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    try:
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=False)
        if world > 1:
            dist.barrier()
        entries = read_entries(config)
        model, tokenizer, codec, initial = build_model(config, device)
        initial.update(source=source, config_sha256=sha256(args.config), device=torch.cuda.get_device_name(device))
        write_json(args.output / f'initialization_rank_{rank}.json', initial)
        if args.mode == 'profile':
            run_profile(model, tokenizer, codec, initial, entries, config, args, device, rank, world, source)
        else:
            run_training(model, tokenizer, codec, initial, entries, config, args, device, rank, world, source)
        if world > 1:
            dist.barrier()
        if rank == 0:
            capacity = {}
            if args.mode == 'profile':
                rank_reports = [json.loads((args.output / f'capacity_rank_{r}.json').read_text()) for r in range(world)]
                accepted = all(report['accepted'] for report in rank_reports)
                cases = []
                for case in rank_reports[0]['cases']:
                    identity = {key: case[key] for key in ('frames', 'batch_per_rank', 'stress_group')}
                    matches = [[other for other in report['cases'] if all(other[key] == value for key, value in identity.items())]
                               for report in rank_reports]
                    if all(len(values) == 1 and values[0]['accepted'] for values in matches):
                        cases.append({**identity, 'ranks_approved': list(range(world)), 'accumulation_microbatches': 2,
                                      'peak_allocated_bytes': max(measurement['peak_allocated_bytes'] for values in matches
                                                                  for measurement in values[0]['measurements'])})
                capacity = {'capacity_contract': capacity_contract(config), 'approved_cases': cases}
            else:
                accepted = True
            write_json(args.output / 'result.json', {'execution_complete': True, 'accepted': accepted,
                       'mode': args.mode, 'code_commit': source,
                       'formal_training_completed': args.mode == 'train', 'quality_acceptance': False, **capacity})
    except Exception:
        if args.output.exists():
            write_json(args.output / f'failure_rank_{rank}.json', {'traceback': traceback.format_exc(),
                       'execution_complete': False, 'accepted': False, 'code_commit': source})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
