#!/usr/bin/env python3
"""Guarded CPU audit, bounded validation and matched 5k XYZ-fill experiments."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import gc
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

from rigweave.dynamic_rig.aux_skeleton_tokens import AuxSkeletonTokens
from rigweave.dynamic_rig.frame_batch_sampler import FrameSampleRequest
from rigweave.dynamic_rig.virtual_query_xyz_data import build_xyz_aux_targets
from rigweave.dynamic_rig.virtual_frame_data import collate_virtual_targets
from fusion_evaluation import tree_fingerprint
from probe_virtual_query_lambda import parameter_fingerprint
from structure_xyz_fill_runtime import (
    FAMILY, audit_new_target_contract, build_fill_model, fixed_dataset, gradient_groups,
    load_small_reference_panel, load_tokenizer, make_stream, optimizer_for, prepare_fill_batch,
    read_entries, reset_random, set_learning_rate, sha256, validate_config,
    verify_data_selection, write_json, matched_trial_fingerprint,
    restore_sampling_state, data_source_hashes, model_source_hashes,
)
from structure_xyz_fill_evaluation import evaluate_fill_items

ENTRYPOINT = 'model_training/rigweave/scripts/train_structure_xyz_fill.py'


def published_run(operation, config_path, output, mode, config_hash):
    matches = [run for run in operation.get('runs', [])
               if Path(run['config']).resolve() == Path(config_path).resolve()
               and Path(run['output']).resolve() == Path(output).resolve() and run['mode'] == mode]
    if len(matches) != 1 or matches[0]['config_sha256'] != config_hash or matches[0].get('approved') is not True:
        raise ValueError('exact mode/config/output/hash is not a uniquely approved operation')
    return matches[0]


def verify_operation(args, config):
    state = json.loads((ROOT / 'model_training/state/current.json').read_text())
    operation = state['active_operation']
    if state['architecture_contract']['id'] != FAMILY:
        raise ValueError('active architecture authority is not the XYZ token-fill model')
    run = published_run(operation, args.config, args.output, args.mode, sha256(args.config))
    gpu = args.mode not in {'audit', 'initialize'}
    if gpu and (operation.get('gpu_work_allowed') is not True or run.get('execution_allowed') is not True):
        raise ValueError('this exact GPU experiment is not released')
    subprocess.run([sys.executable, str(ROOT / 'model_training/tools/agent_work_guard.py'),
                    'check', '--operation', 'train' if gpu else 'inspect', '--entrypoint', ENTRYPOINT],
                   check=True, stdout=subprocess.DEVNULL)
    if Path(config['runtime']['repo']).resolve() != ROOT:
        raise ValueError('runtime checkout differs from its declared isolated code directory')
    if subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
        raise ValueError('published runtime requires a clean committed source tree')
    expected = operation['code_commit']
    subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', expected, 'HEAD'], check=True)
    subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet', expected, 'HEAD', '--',
                    'model_training/rigweave', 'model_training/tools'], check=True)
    if gpu and run['physical_gpus'] != config['runtime']['physical_gpus']:
        raise ValueError('GPU identities differ between operation and configuration')
    return subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()


def read_reviewed(spec, *, name):
    if not isinstance(spec, dict) or sha256(spec['path']) != spec['sha256']:
        raise ValueError(f'{name} is missing or its reviewed fingerprint changed')
    value = json.loads(Path(spec['path']).read_text())
    if value.get('passed') is not True:
        raise ValueError(f'{name} did not pass')
    for key in ('code_commit', 'config_sha256'):
        if not isinstance(spec.get(key), str) or not spec[key] or value.get(key) != spec[key]:
            raise ValueError(f'{name} producer {key} is absent or differs from its reviewed identity')
    return value


def training_contract(config):
    contract = {key: config[key] for key in (
        'purpose', 'data', 'surface_sampling', 'auxiliary_codec', 'architecture', 'fill_architecture',
        'quantization', 'initialization', 'common_initialization', 'fill_initialization', 'optimization', 'pruning',
        'evaluation', 'reference_evaluation',
    )}
    contract['execution'] = {'world_size': config['runtime']['world_size'], 'precision': 'bfloat16_autocast'}
    return contract


def initialization_contract(config):
    return {key: config[key] for key in ('initialization', 'common_initialization', 'architecture',
                                        'fill_architecture', 'fill_initialization', 'auxiliary_codec')}


def capacity_contract(config):
    return {'model': {key: config[key] for key in ('architecture', 'fill_architecture', 'auxiliary_codec')},
            'surface_sampling': config['surface_sampling'], 'quantization': config['quantization'],
            'manifests': config['data']['manifests'], 'initialization': config['initialization'],
            'common_initialization': config['common_initialization'],
            'precision': 'bfloat16_autocast', 'world_size': config['runtime']['world_size'],
            'physical_gpus': config['runtime']['physical_gpus'],
            'optimizer': {'kind': 'AdamW', 'betas': config['optimization']['betas']},
            'gradient_checkpointing': config['architecture']['gradient_checkpointing'],
            'allocated_fraction_limit': config['capacity']['allocated_fraction_limit'],
            'source_sha256': model_source_hashes()}


def validate_capacity(config, receipt):
    if receipt['capacity_contract'] != capacity_contract(config):
        raise ValueError('model/data/precision/resources differ from measured capacity')
    covered = {(row['frames'], row['batch_per_rank'], row['stress_group']) for row in receipt['cases']
               if row.get('passed') and row.get('ranks') == list(range(config['runtime']['world_size']))}
    for frames in (6, 12, 24):
        batch = config['training']['frame_policy'][str(frames)]
        for group in ('longest_fill', 'longest_structure', f'geometry_t{frames}'):
            if (frames, batch, group) not in covered:
                raise ValueError('declared microbatch schedule lacks its new-model capacity cases')
    policy = config['training']['frame_policy']
    for frames in range(2, 25):
        ceiling = 6 if frames <= 6 else 12 if frames <= 12 else 24
        if policy[str(frames)] != policy[str(ceiling)]:
            raise ValueError('unprofiled physical microbatch schedule')


def verify_reference_labels_cpu(config, entries, tokenizer, codec):
    panel, reference, manifest = load_small_reference_panel(config, entries)
    checks = []
    for group, rows in panel.items():
        dataset = fixed_dataset(rows, config, 8, mode='heldout_eval' if group == 'train_heldout' else 'seen_eval')
        for index in range(len(dataset)):
            geometry = collate_virtual_targets([dataset[index]], 'cpu')
            target = build_xyz_aux_targets(geometry, tokenizer, codec, max_main_tokens=config['data']['max_main_tokens'])
            asset_id = geometry['asset_id'][0]
            for frame in config['evaluation']['frame_indices']:
                row = reference[(group, asset_id, frame)]
                tokens = target['main_input_ids'][0, frame][target['main_attention_mask'][0, frame]].tolist()
                raw_frame = int(geometry['selected_frames'][0, frame])
                if row['raw_frame_index'] != raw_frame or row['target_tokens'] != tokens:
                    raise ValueError(f'old/new GT or frame identity differs for {group}/{asset_id}/{frame}')
                checks.append({'group': group, 'asset_id': asset_id, 'frame_index': frame,
                               'raw_frame_index': raw_frame, 'target_tokens_sha256': tree_fingerprint(tokens)})
    return {'passed': True, 'views': len(checks), 'checks': checks, 'panel': manifest}


def audit(args, config, source):
    entries = read_entries(config)
    verify_data_selection(config, entries)
    tokenizer = load_tokenizer(config)
    codec = AuxSkeletonTokens(**config['auxiliary_codec'])
    report = audit_new_target_contract(config, entries, codec, tokenizer)
    report.update(code_commit=source, config_sha256=sha256(args.config))
    write_json(args.output / 'data_audit.json', report)
    matched = verify_reference_labels_cpu(config, entries, tokenizer, codec)
    matched.update(code_commit=source, config_sha256=sha256(args.config), target_source_sha256=data_source_hashes())
    write_json(args.output / 'reference_panel_audit.json', matched)
    return {'passed': True, 'mode': 'audit', 'data_audit': str(args.output / 'data_audit.json'),
            'data_audit_sha256': sha256(args.output / 'data_audit.json'),
            'reference_panel_audit_sha256': sha256(args.output / 'reference_panel_audit.json'),
            'max_fill_tokens': report['max_fill_tokens'], 'max_joints': report['max_joints'],
            'real_loader_checks': len(report['real_loader_checks']), 'matched_reference_views': matched['views'],
            'no_gpu_or_training': True}


def gather_reports(local, world):
    if world == 1:
        return [local]
    reports = [None] * world
    dist.all_gather_object(reports, local)
    return reports


def check_gradient_routes(model, output):
    parameters = [model.motion_encoder.virtual_slots, model.surface_tokenizer.output_proj.weight]
    report = {}
    for name in ('xyz_loss', 'structure_loss'):
        gradients = torch.autograd.grad(output[name], parameters, retain_graph=True)
        norms = [float(value.float().norm()) for value in gradients]
        if any(not np.isfinite(value) or value <= 0 for value in norms):
            raise RuntimeError(f'{name} does not train the shared query and mesh feature paths')
        report[name] = {'virtual_slots': norms[0], 'surface_projection': norms[1]}
    direct = torch.autograd.grad(output['xyz_loss'], model.structure_decoder.head.weight,
                                 allow_unused=True, retain_graph=True)[0]
    if direct is not None:
        raise RuntimeError('XYZ unexpectedly has a continuous bridge into the structure decoder')
    report['xyz_to_structure_head'] = None
    return report


def prepare_requests(config, groups, case, rank, number):
    indices = groups[case['stress_group']]
    batch, frames = case['batch_per_rank'], case['frames']
    return [FrameSampleRequest(indices[(rank * batch + index) % len(indices)], frames,
                               (config['data']['stream_seed'] << 32) + 100000 + number * 1000 + rank * 100 + index)
            for index in range(batch)]


def profile(model, tokenizer, codec, initial, entries, config, args, device, rank, world):
    reviewed = read_reviewed(config['safety']['data_audit'], name='real-data target audit')
    dataset = fixed_dataset(entries['train'], config, 2, mode='train')
    optimizer = optimizer_for(model, config)
    wrapped = DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True,
                                      gradient_as_bucket_view=True) if world > 1 else model
    local = {'passed': False, 'rank': rank, 'cases': [], 'capacity_contract': capacity_contract(config),
             'profile_weights_are_never_training_initialization': True}
    total_bytes = torch.cuda.get_device_properties(device).total_memory
    model.train()
    for number, case in enumerate(config['capacity']['cases']):
        requests = prepare_requests(config, reviewed['stress_groups'], case, rank, number)
        measurements = []
        for iteration in range(2):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.monotonic()
            metrics = []
            for micro in range(2):
                surface, targets, geometry, _ = prepare_fill_batch(dataset, requests, config, codec, device)
                context = wrapped.no_sync() if world > 1 and micro == 0 else nullcontext()
                with context:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        output = wrapped(surface, targets, structure_weight=config['optimization']['structure_weight'])
                    if not bool(torch.isfinite(output['loss'])):
                        raise RuntimeError('nonfinite new-model capacity loss')
                    (output['loss'] / 2).backward()
                metrics.append(float(output['loss'].detach()))
                maximum_fill = int(targets['fill_lengths'].max())
                maximum_structure = int(targets['structure_lengths'].max())
                assets = geometry['asset_id']
                del surface, targets, geometry, output
            groups = gradient_groups(model)
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimization']['clip_grad_norm'],
                                                       error_if_nonfinite=True))
            optimizer.step()
            torch.cuda.synchronize(device)
            measurements.append({'loss': float(np.mean(metrics)), 'gradient_norm': norm, 'gradient_groups': groups,
                                 'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                                 'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
                                 'seconds': time.monotonic() - started, 'max_fill_tokens': maximum_fill,
                                 'max_structure_tokens': maximum_structure, 'assets': assets})
        passed = all(row['peak_allocated_bytes'] <= config['capacity']['allocated_fraction_limit'] * total_bytes
                     for row in measurements)
        case_report = {**case, 'passed': passed, 'measurements': measurements, 'gpu_total_bytes': total_bytes}
        local['cases'].append(case_report)
        write_json(args.output / f'capacity_rank_{rank}.json', local)
        print('CAPACITY ' + json.dumps({'rank': rank, **case_report}, allow_nan=False), flush=True)
        consensus = torch.tensor(int(passed), device=device)
        if world > 1:
            dist.all_reduce(consensus, op=dist.ReduceOp.MIN)
        if not bool(consensus.item()):
            raise RuntimeError('new model exceeds the declared memory headroom')
    local['passed'] = True
    write_json(args.output / f'capacity_rank_{rank}.json', local)
    reports = gather_reports(local, world)
    combined = {'passed': True, 'capacity_contract': capacity_contract(config), 'cases': [],
                'code_commit': initial['code_commit'], 'config_sha256': initial['config_sha256']}
    for index, case in enumerate(config['capacity']['cases']):
        combined['cases'].append({**case, 'passed': all(row['cases'][index]['passed'] for row in reports),
                                  'ranks': list(range(world)),
                                  'peak_allocated_bytes_by_rank': [max(item['peak_allocated_bytes']
                                      for item in row['cases'][index]['measurements']) for row in reports]})
    if rank == 0:
        write_json(args.output / 'capacity.json', combined)
    return {'passed': True, 'mode': 'profile', 'cases': len(combined['cases']), 'capacity': combined}


def preflight(model, tokenizer, codec, initial, entries, config, args, device, rank, world):
    reviewed = read_reviewed(config['safety']['data_audit'], name='real-data target audit')
    dataset = fixed_dataset(entries['train'], config, 2, mode='train')
    cases = [{'frames': 2, 'batch_per_rank': 1, 'stress_group': 'longest_fill'},
             {'frames': 24, 'batch_per_rank': 1, 'stress_group': 'geometry_t24'}]
    reports = []
    model.train()
    for number, case in enumerate(cases):
        requests = prepare_requests(config, reviewed['stress_groups'], case, rank, number)
        surface, targets, geometry, _ = prepare_fill_batch(dataset, requests, config, codec, device)
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(surface, targets, structure_weight=config['optimization']['structure_weight'])
        if not bool(torch.isfinite(output['loss'])):
            raise RuntimeError('nonfinite new-model contract loss')
        routes = check_gradient_routes(model, output) if number == 0 else None
        output['loss'].backward()
        norms = gradient_groups(model)
        torch.cuda.synchronize(device)
        reports.append({**case, 'assets': geometry['asset_id'], 'loss': float(output['loss'].detach()),
                        'gradient_routes': routes, 'gradient_groups': norms,
                        'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                        'seconds': time.monotonic() - started})
        del surface, targets, geometry, output
    model.zero_grad(set_to_none=True)
    # No parameter updates occurred in the contract cases; the fixed fit starts at the exact declared initialization.
    if parameter_fingerprint(model) != initial['full_initial_state_sha256']:
        raise RuntimeError('contract checks unexpectedly changed initialization weights')
    counts = reviewed['new_static_target_contract']['train']
    candidates = [row for row in counts if 20 <= row['joints'] <= 60 and row['joints'] != 52]
    if len(candidates) < world:
        raise ValueError('not enough explicitly non-template assets for the bounded fit check')
    random_order = np.random.default_rng(config['preflight']['selection_seed']).permutation(len(candidates))
    index = candidates[int(random_order[rank])]['manifest_index']
    request = FrameSampleRequest(index, config['preflight']['frames'],
                                 (config['data']['stream_seed'] << 32) + index)
    surface, targets, geometry, _ = prepare_fill_batch(dataset, [request], config, codec, device)
    write_json(args.output / f'fit_inputs_rank_{rank}.json', {
        'asset_id': geometry['asset_id'][0], 'manifest_index': index, 'sampling_seed': request.sampling_seed,
        'selected_raw_frames': geometry['selected_frames'][0].tolist(),
        'target_tokens_sha256': tree_fingerprint(targets), 'not_normal_training_initialization': True})
    optimizer = optimizer_for(model, config)
    wrapped = DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True,
                                      gradient_as_bucket_view=True) if world > 1 else model
    history = []
    for step in range(config['preflight']['fit_steps'] + 1):
        if step in config['preflight']['evaluate_steps']:
            evidence = evaluate_fill_items(model, [(surface, targets, geometry)], codec,
                frame_indices=list(range(config['preflight']['frames'])), group='fixed_fit',
                structure_weight=config['optimization']['structure_weight'], **config['quantization'])
            write_json(args.output / f'fit_eval_{step:04d}_rank_{rank}.json', evidence)
            history.append({'step': step, **evidence['summary']})
            print('FIT_EVAL ' + json.dumps({'rank': rank, **history[-1]}, allow_nan=False), flush=True)
            model.train()
        if step == config['preflight']['fit_steps']:
            break
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = wrapped(surface, targets, structure_weight=config['optimization']['structure_weight'])
        if not bool(torch.isfinite(output['loss'])):
            raise RuntimeError('nonfinite fixed-fit loss')
        output['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimization']['clip_grad_norm'], error_if_nonfinite=True)
        optimizer.step()
    local = {'rank': rank, 'contract_cases': reports, 'fit_history': history,
             'fit_asset': geometry['asset_id'][0], 'passed': True,
             'fit_weights_are_diagnostic_only': True}
    write_json(args.output / f'preflight_rank_{rank}.json', local)
    combined = gather_reports(local, world)
    result = {'passed': True, 'mode': 'preflight', 'ranks': combined,
              'quality_accepted': False, 'fit_quality_requires_primary_review': True,
              'initialization': initial, 'training_contract': training_contract(config),
              'source_sha256': model_source_hashes(), 'code_commit': initial['code_commit'],
              'config_sha256': initial['config_sha256']}
    if rank == 0:
        write_json(args.output / 'preflight.json', result)
    return result


def run_evaluation(model, tokenizer, codec, config, args, device, rank, world, samples, panel, reference, panel_sha256):
    root = args.output / f'evaluation_{samples:06d}'
    root.mkdir(parents=True, exist_ok=True)
    local = {'rank': rank, 'asset_exposures': samples, 'groups': {}, 'complete': False}
    for group, rows in panel.items():
        selected = rows[rank::world]
        dataset = fixed_dataset(selected, config, config['evaluation']['input_frames'],
                                mode='heldout_eval' if group == 'train_heldout' else 'seen_eval')
        def items():
            for index in range(len(dataset)):
                surface, targets, geometry, _ = prepare_fill_batch(dataset, [index], config, codec, device)
                actual, reserved = set(geometry['selected_frames'][0].tolist()), set(geometry['heldout_raw_ids'][0])
                if (group == 'train_heldout' and actual != reserved) or (group != 'train_heldout' and actual & reserved):
                    raise ValueError('evaluation crossed the declared heldout partition')
                yield surface, targets, geometry
        expected = {(row['asset_id'], frame) for row in selected for frame in config['evaluation']['frame_indices']}
        scoped_reference = {(group, asset_id, frame): reference[(group, asset_id, frame)]
                            for asset_id, frame in expected}
        result = evaluate_fill_items(model, items(), codec, frame_indices=config['evaluation']['frame_indices'],
            group=group, original_tokenizer=tokenizer, reference_lookup=scoped_reference,
            structure_weight=config['optimization']['structure_weight'], **config['quantization'])
        actual = [(row['asset_id'], row['frame_index']) for row in result['rows']]
        if len(actual) != len(expected) or set(actual) != expected:
            raise RuntimeError('small evaluation missed or duplicated an asset/view')
        local['groups'][group] = result
        write_json(root / f'rank_{rank}.json', local)
    local['complete'] = True
    write_json(root / f'rank_{rank}.json', local)
    if world > 1:
        dist.barrier()
    summary = None
    if rank == 0:
        from structure_xyz_fill_evaluation import summarize_fill_rows
        reports = [json.loads((root / f'rank_{number}.json').read_text()) for number in range(world)]
        if not all(row['complete'] for row in reports):
            raise RuntimeError('a rank did not complete the fixed small evaluation panel')
        groups = {}
        for group in panel:
            rows = [row for report in reports for row in report['groups'][group]['rows']]
            structures = [row for report in reports for row in report['groups'][group]['structure_rows']]
            groups[group] = summarize_fill_rows(rows, structures)
        summary = {'complete': True, 'asset_exposures': samples, 'groups': groups,
                   'panel_sha256': panel_sha256,
                   'input_frames': config['evaluation']['input_frames'],
                   'scored_frame_indices': config['evaluation']['frame_indices'],
                   'evaluation_assets': {group: len(rows) for group, rows in panel.items()},
                   'full_validation': False, 'failed_generations_scored_zero': True,
                   'quality_acceptance': False}
        write_json(root / 'summary.json', summary)
        print('EVALUATION ' + json.dumps(summary, allow_nan=False), flush=True)
    if world > 1:
        dist.barrier()
    model.train()
    return summary


def save_checkpoint(model, optimizer, stream, config, args, initial, source, samples, frames, step, rank, world):
    local = {'stream': stream.state_dict(), 'torch_rng': torch.get_rng_state(),
             'cuda_rng': torch.cuda.get_rng_state(), 'numpy_rng': np.random.get_state(),
             'python_rng': random.getstate()}
    ranks = gather_reports(local, world)
    if rank == 0:
        path = args.output / f'checkpoint_asset_{samples:06d}.pt'
        temporary = path.with_suffix('.pt.tmp')
        torch.save({'schema': FAMILY, 'mode': 'train', 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'rank_states': ranks,
                    'optimizer_parameter_names': optimizer_parameter_names(model, optimizer),
                    'config': config, 'architecture': model.configuration(),
                    'asset_exposures': samples, 'supervised_frame_exposures': frames, 'step': step,
                    'initialization': initial, 'source': source, 'training_contract': training_contract(config)}, temporary)
        temporary.replace(path)
        write_json(args.output / 'latest_checkpoint.json', {'path': str(path), 'bytes': path.stat().st_size,
                   'sha256': sha256(path), 'asset_exposures': samples, 'step': step,
                   'supervised_frame_exposures': frames})
    if world > 1:
        dist.barrier()


def restore_continuation(model, optimizer, dataset, config, rank, world, device):
    spec = config['training'].get('resume_checkpoint')
    if spec is None:
        return make_stream(config, dataset, rank=rank, world=world), 0, 0, 0, None
    if sha256(spec['path']) != spec['sha256']:
        raise ValueError('new-model continuation checkpoint changed')
    saved = torch.load(spec['path'], map_location='cpu', mmap=True, weights_only=False)
    if (saved['schema'] != FAMILY or saved['mode'] != 'train' or saved['architecture'] != model.configuration()
            or saved['training_contract'] != training_contract(config) or len(saved['rank_states']) != world):
        raise ValueError('continuation model/data/loss/evaluation/initialization contract differs')
    if saved['asset_exposures'] > config['training']['target_assets']:
        raise ValueError('a resume cannot move the asset exposure counter backwards')
    if saved['optimizer_parameter_names'] != optimizer_parameter_names(model, optimizer):
        raise ValueError('optimizer parameter identities/order differ from the saved checkpoint')
    for name, value in saved['model'].items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f'nonfinite saved model tensor {name}')
    model.load_state_dict(saved['model'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    for parameter, state in optimizer.state.items():
        if any(name not in state for name in ('step', 'exp_avg', 'exp_avg_sq')):
            raise ValueError('incomplete saved AdamW state')
        if int(state['step'].item()) != saved['step'] or float(state['step'].item()) != saved['step']:
            raise ValueError('AdamW step count differs from the saved optimizer update')
        if any(state[name].shape != parameter.shape for name in ('exp_avg', 'exp_avg_sq')):
            raise ValueError('AdamW moment shape differs from its named parameter')
        for name, value in state.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError(f'nonfinite optimizer state {name}')
            if isinstance(value, torch.Tensor) and name != 'step':
                state[name] = value.to(parameter.device)
    local = saved['rank_states'][rank]
    stream = restore_sampling_state(config, dataset, local['stream'],
        asset_exposures=saved['asset_exposures'], supervised_frame_exposures=saved['supervised_frame_exposures'],
        step=saved['step'], rank=rank, world=world)
    torch.set_rng_state(local['torch_rng'])
    torch.cuda.set_rng_state(local['cuda_rng'], device)
    np.random.set_state(local['numpy_rng'])
    random.setstate(local['python_rng'])
    result = (stream, saved['asset_exposures'], saved['supervised_frame_exposures'], saved['step'], saved['initialization'])
    return result


def optimizer_parameter_names(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return [{'group': group['name'], 'parameters': [names[id(parameter)] for parameter in group['params']]}
            for group in optimizer.param_groups]


def train(model, tokenizer, codec, initial, entries, config, args, device, rank, world, source):
    preflight_receipt = read_reviewed(config['safety']['preflight'], name='new-model preflight')
    if (preflight_receipt['training_contract'] != training_contract(config)
            or preflight_receipt['source_sha256'] != model_source_hashes()):
        raise ValueError('preflight model/data/initialization/source differs from the current training contract')
    if config['safety'].get('preflight_primary_accepted') is not True:
        raise ValueError('fixed-fit evidence has not been reviewed for trainability')
    validate_capacity(config, read_reviewed(config['safety']['capacity'], name='new-model capacity'))
    panel, reference, panel_manifest = load_small_reference_panel(config, entries)
    panel_audit = read_reviewed(config['safety']['reference_panel_audit'], name='matched reference-panel audit')
    if panel_audit['panel']['panel_sha256'] != panel_manifest['panel_sha256']:
        raise ValueError('small evaluation panel differs from its reviewed GT/reference audit')
    if rank == 0:
        write_json(args.output / 'evaluation_panel.json', panel_manifest)
    dataset = fixed_dataset(entries['train'], config, 2, mode='train')
    optimizer = optimizer_for(model, config)
    stream, samples, frames_seen, step, inherited_initial = restore_continuation(
        model, optimizer, dataset, config, rank, world, device)
    if inherited_initial is not None:
        initial = inherited_initial
    wrapped = DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True,
                                      gradient_as_bucket_view=True) if world > 1 else model
    if rank == 0:
        write_json(args.output / 'sampling_plan.json', stream.plan_report())
    report = {'complete': False, 'rank': rank, 'asset_exposures': samples, 'step': step,
              'supervised_frame_exposures': frames_seen, 'evaluations': [], 'source': source,
              'initialization': initial, 'resume': config['training'].get('resume_checkpoint')}
    write_json(args.output / f'rank_{rank}.json', report)
    previous_eval = (config['training'].get('resume_checkpoint') or {}).get('completed_evaluation')
    if previous_eval is not None:
        if sha256(previous_eval['path']) != previous_eval['sha256']:
            raise ValueError('saved incoming-checkpoint evaluation changed')
        old_eval = json.loads(Path(previous_eval['path']).read_text())
        if (old_eval.get('complete') is not True or old_eval.get('asset_exposures') != samples
                or old_eval.get('panel_sha256') != panel_manifest['panel_sha256']):
            raise ValueError('incoming evaluation is incomplete or from a different checkpoint/panel')
        report['verified_incoming_evaluation'] = previous_eval
    else:
        # A checkpoint is saved before its evaluation; after interruption that evaluation may still be missing.
        run_evaluation(model, tokenizer, codec, config, args, device, rank, world, samples, panel, reference,
                       panel_manifest['panel_sha256'])
        report['evaluations'].append(samples)
    milestones = config['training']['evaluation_assets']
    started = time.monotonic()
    with (args.output / f'trace_rank_{rank}.jsonl').open('x', buffering=1) as trace:
        while samples < config['training']['target_assets']:
            update = int(np.searchsorted(stream.global_ends, samples, side='right'))
            global_goal = int(stream.update_sizes[update])
            local_goal = global_goal // world
            optimizer.zero_grad(set_to_none=True)
            learning_rates = set_learning_rate(optimizer, samples, config)
            torch.cuda.reset_peak_memory_stats(device)
            tick = time.monotonic()
            totals = torch.zeros(7, dtype=torch.float64, device=device)
            schedule, count = [], 0
            while count < local_goal:
                requests = stream.take(local_goal - count)
                context = {'step': step + 1, 'rank': rank,
                           'requests': [as_request(request, entries['train']) for request in requests]}
                write_json(args.output / f'current_batch_rank_{rank}.json', context)
                surface, targets, geometry, pruning = prepare_fill_batch(
                    dataset, requests, config, codec, device, update_index=step, pruning=True)
                size = len(requests)
                sync = count + size == local_goal
                with (wrapped.no_sync() if world > 1 and not sync else nullcontext()):
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        output = wrapped(surface, targets, structure_weight=config['optimization']['structure_weight'])
                    if not bool(torch.isfinite(output['loss'])):
                        raise RuntimeError(f'nonfinite loss at optimizer update {step + 1}')
                    output['loss_per_asset'].sum().backward()
                totals += torch.stack((output['loss'].detach() * size, output['xyz_loss'].detach() * size,
                                       output['structure_loss'].detach() * size,
                                       output['xyz_accuracy'].detach() * size,
                                       output['structure_accuracy'].detach() * size,
                                       totals.new_tensor(size), totals.new_tensor(size * requests[0].frames)))
                selected = geometry['selected_frames'].cpu().tolist()
                for values, heldout, request in zip(selected, geometry['heldout_raw_ids'], requests):
                    if len(set(values)) != request.frames or set(values) & set(heldout):
                        raise ValueError('training frame selection duplicates or leaks heldout poses')
                schedule.append({'asset_ids': geometry['asset_id'],
                                 'manifest_indices': [request.item_index for request in requests],
                                 'sampling_seeds': [request.sampling_seed for request in requests],
                                 'selected_raw_frames': selected, 'frames': requests[0].frames,
                                 'pruning': pruning, 'fill_tokens': targets['fill_lengths'].cpu().tolist(),
                                 'target_identity_sha256': tree_fingerprint({key: targets[key] for key in
                                     ('xyz_labels', 'fill_source_indices', 'structure_labels')})})
                count += size
                del surface, targets, geometry, output
            if world > 1:
                dist.all_reduce(totals)
            actual_assets, actual_frames = int(totals[5]), int(totals[6])
            if actual_assets != global_goal:
                raise RuntimeError('actual supervised asset count differs from the sampling plan')
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(world / actual_assets)
            groups = gradient_groups(model) if step == 0 else None
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimization']['clip_grad_norm'],
                                                       error_if_nonfinite=True))
            optimizer.step()
            samples += actual_assets
            frames_seen += actual_frames
            step += 1
            torch.cuda.synchronize(device)
            item = {'step': step, 'asset_exposures': samples, 'supervised_frame_exposures': frames_seen,
                    'assets_in_update': actual_assets, 'microbatches': schedule,
                    'loss': float(totals[0] / actual_assets), 'xyz_loss': float(totals[1] / actual_assets),
                    'structure_loss': float(totals[2] / actual_assets), 'xyz_accuracy': float(totals[3] / actual_assets),
                    'structure_accuracy': float(totals[4] / actual_assets), 'gradient_norm': norm,
                    'first_gradient_groups': groups, 'learning_rates': learning_rates,
                    'update_seconds': time.monotonic() - tick, 'elapsed_seconds': time.monotonic() - started,
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(device)}
            trace.write(json.dumps(item, allow_nan=False) + '\n')
            report.update(asset_exposures=samples, step=step, supervised_frame_exposures=frames_seen,
                          latest={key: value for key, value in item.items() if key != 'microbatches'})
            write_json(args.output / f'rank_{rank}.json', report)
            print('TRAIN ' + json.dumps({'rank': rank, **report['latest']}, allow_nan=False), flush=True)
            if samples in milestones:
                save_checkpoint(model, optimizer, stream, config, args, initial, source, samples, frames_seen, step, rank, world)
                run_evaluation(model, tokenizer, codec, config, args, device, rank, world, samples, panel, reference,
                               panel_manifest['panel_sha256'])
                report['evaluations'].append(samples)
    report.update(complete=True, final_model_sha256=parameter_fingerprint(model), finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(args.output / f'rank_{rank}.json', report)
    reports = gather_reports({key: report[key] for key in
                              ('rank', 'asset_exposures', 'supervised_frame_exposures', 'step', 'evaluations', 'final_model_sha256')}, world)
    if len({row['final_model_sha256'] for row in reports}) != 1:
        raise RuntimeError('distributed final model parameters diverged')
    return {'passed': True, 'mode': 'train', 'complete': True, 'ranks': reports,
            'asset_exposures': samples, 'supervised_frame_exposures': frames_seen, 'step': step,
            'quality_accepted': False, 'next_stage_requires_primary_analysis': True}


def as_request(request, entries):
    row = entries[request.item_index]
    return {'manifest_index': request.item_index, 'asset_id': row['asset_id'], 'path': row['path'],
            'frames': request.frames, 'sampling_seed': request.sampling_seed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('audit', 'initialize', 'preflight', 'profile', 'train'), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    claimed = False
    try:
        validate_config(config)
        source = verify_operation(args, config)
        if args.mode in {'audit', 'initialize'}:
            if world != 1:
                raise ValueError('CPU audits must run once, without distributed workers')
            if os.environ.get('CUDA_VISIBLE_DEVICES'):
                raise ValueError('CPU audits require explicitly hidden CUDA devices')
            args.output.mkdir(parents=True, exist_ok=False)
            claimed = True
            if args.mode == 'audit':
                result = audit(args, config, source)
            else:
                reset_random(config['initialization']['seed'])
                model, tokenizer, codec, initial = build_fill_model(config, torch.device('cpu'))
                if any(parameter.device.type != 'cpu' for parameter in model.parameters()):
                    raise RuntimeError('initialization audit unexpectedly created GPU parameters')
                if initial['trained_experiment_weights_loaded'] or any(initial['inherited_training'].values()):
                    raise RuntimeError('fresh independent initialization inherited experiment state')
                result = {'passed': True, 'mode': 'initialize', 'no_gpu_or_training': True,
                          'initialization': initial, 'initialization_contract': initialization_contract(config),
                          'source_sha256': model_source_hashes(), 'code_commit': source,
                          'config_sha256': sha256(args.config)}
                write_json(args.output / 'initialization_audit.json', result)
        else:
            visible = ','.join(map(str, config['runtime']['physical_gpus']))
            if (os.environ.get('CUDA_VISIBLE_DEVICES') != visible or world != config['runtime']['world_size']
                    or not torch.cuda.is_available() or torch.cuda.device_count() != world):
                raise ValueError('actual CUDA devices/world differ from the exact published resource request')
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            device = torch.device('cuda', local_rank)
            torch.cuda.set_device(device)
            torch.cuda.set_per_process_memory_fraction(.95, device)
            if world > 1:
                dist.init_process_group('nccl', timeout=timedelta(minutes=30))
            if rank == 0:
                args.output.mkdir(parents=True, exist_ok=False)
            if world > 1:
                dist.barrier()
            claimed = True
            reviewed = read_reviewed(config['safety']['data_audit'], name='new target data audit')
            if (reviewed['manifests'] != config['data']['manifests'] or reviewed['architecture'] != config['fill_architecture']
                    or reviewed['quantization'] != config['quantization']
                    or reviewed['auxiliary_codec'] != config['auxiliary_codec']
                    or reviewed['target_source_sha256'] != data_source_hashes()):
                raise ValueError('CPU data audit does not cover the exact GPU target contract')
            panel_audit = read_reviewed(config['safety']['reference_panel_audit'], name='matched reference-panel audit')
            if panel_audit['target_source_sha256'] != data_source_hashes():
                raise ValueError('matched-panel audit does not cover the current target implementation')
            entries = read_entries(config)
            verify_data_selection(config, entries)
            initialization_audit = read_reviewed(config['safety']['initialization'], name='official-only initialization audit')
            if (initialization_audit['initialization_contract'] != initialization_contract(config)
                    or initialization_audit['source_sha256'] != model_source_hashes()):
                raise ValueError('initialization audit does not cover the exact current model and source')
            reset_random(config['initialization']['seed'])
            model, tokenizer, codec, initial = build_fill_model(config, device)
            if (initial['full_initial_state_sha256'] !=
                    initialization_audit['initialization']['full_initial_state_sha256']):
                raise RuntimeError('GPU starting parameters differ from the reviewed CPU initialization')
            initial.update(code_commit=source, config_sha256=sha256(args.config),
                           device=torch.cuda.get_device_name(device), torch_version=torch.__version__,
                           matched_trial_contract_sha256=matched_trial_fingerprint(config))
            fingerprints = gather_reports(initial['full_initial_state_sha256'], world)
            if len(set(fingerprints)) != 1:
                raise RuntimeError('rank initialization differs before distributed wrapping')
            write_json(args.output / f'initialization_rank_{rank}.json', initial)
            functions = {'preflight': preflight, 'profile': profile, 'train': train}
            if args.mode == 'train':
                result = train(model, tokenizer, codec, initial, entries, config, args, device, rank, world, source)
            else:
                result = functions[args.mode](model, tokenizer, codec, initial, entries, config, args, device, rank, world)
        result.update(code_commit=source, config_sha256=sha256(args.config), finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.output / f'result_rank_{rank}.json', result)
        if rank == 0:
            write_json(args.output / 'result.json', result)
        print('RESULT ' + json.dumps({key: value for key, value in result.items() if key not in ('ranks', 'capacity', 'initialization', 'training_contract')},
                                     allow_nan=False), flush=True)
    except Exception as exc:
        if claimed:
            failure = {'passed': False, 'complete': False, 'mode': args.mode, 'rank': rank,
                       'error': repr(exc), 'traceback': traceback.format_exc(),
                       'failed_utc': datetime.now(timezone.utc).isoformat()}
            write_json(args.output / f'failure_rank_{rank}.json', failure)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
