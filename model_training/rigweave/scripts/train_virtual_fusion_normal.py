#!/usr/bin/env python3
"""Matched 5k-asset online training, separate from fixed-pose and full-data runs."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import traceback

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'model_training/rigweave/src'))
import train_virtual_query_xyz as core
from fusion_evaluation import auxiliary_summary, tree_fingerprint
from probe_virtual_fusion_reference import require_reference_gradients
from probe_virtual_query_lambda import parameter_fingerprint
from probe_virtual_query_reference import ReferenceConditionRig
from virtual_fusion_normal_runtime import MatchedFrameStream
from virtual_query_xyz_runtime import build_model, fixed_dataset, read_entries, sha256, write_json

CODE_ROOT = Path('/work/wangyy/virtual_fusion_20261013/normal_code')
OUTPUT_ROOT = Path('/home/wangyy/evorig/outputs/virtual_fusion_normal_20261014')
ENTRYPOINT = 'model_training/rigweave/scripts/train_virtual_fusion_normal.py'
PURPOSE = 'normal5000_trend'
ROUTES = {'legacy': 'virtual_queries_real', 'a': 'virtual_queries_real', 'b': 'real_queries_virtual',
          'c': 'relative_position', 'd': 'token_film'}
MILESTONES = [5000, 10000, 15000, 20000]
SOURCE_HASHES = {
    'train': '653f496d6610caee5d1ba6aa8dc26b2d656665c31aad05aead93be97a07f3b27',
    'valid': 'fd2be7b5885833946e68dc186300a820757ae652c487749eec1b049d381e230a',
}
BASE_INITIAL_SHA256 = 'a89b29c903abffe2eead47b5411edb8588b599fe874ce490c990ce07225d3635'


class LegacyRig(ReferenceConditionRig):
    def configuration(self):
        return {**self.base.configuration(), 'execution_route': self.route, 'auxiliary_applicable': False}


def validate_config(config, mode, world):
    route = config.get('route')
    if config.get('purpose') != PURPOSE or route not in ROUTES:
        raise ValueError('the normal comparison needs its explicit purpose and legacy/a/b/c/d route')
    if mode not in {'profile', 'train'} or world not in {1, 2} or (mode == 'train' and world != 2):
        raise ValueError('normal training requires two ranks; only profiling may use one')
    data, plan, optim = config['data'], config['training'], config['optimization']
    if (data['frames_min'], data['frames_max'], data['heldout_frames']) != (2, 24, 8):
        raise ValueError('normal inputs must be resampled from 2..24 with eight reserved frames')
    if (data.get('normalization') != 'input_sequence_mesh_bbox' or data.get('global_rt_removal') is not True
            or data.get('coordinate_clipping') is not False):
        raise ValueError('normalization/root RT/no-clipping contract changed')
    if (data['manifests']['train']['rows'], data['manifests']['valid']['rows']) != (5000, 967):
        raise ValueError('normal training requires the fixed 5000-row subset and disjoint 967-row validation pool')
    if data['manifests']['valid']['sha256'] != SOURCE_HASHES['valid']:
        raise ValueError('validation data version changed')
    if (plan['target_assets'], plan['effective_assets_per_update'], optim['schedule_samples']) != (20000, 48, 20000):
        raise ValueError('normal comparison requires exactly 20000 asset exposures and effective batch48')
    if plan['checkpoint_assets'] != MILESTONES or plan['evaluation_assets'] != MILESTONES:
        raise ValueError('normal trend checkpoints must be at each 5000 asset exposures')
    expected_optim = {'schedule': 'onecycle', 'peak_lr': 1e-4, 'warmup_fraction': .1,
                      'div_factor': 5, 'final_div_factor': 10, 'weight_decay': 0, 'clip_grad_norm': 1,
                      'schedule_samples': 20000, 'auxiliary_weight': 0 if route == 'legacy' else .1}
    if optim != expected_optim:
        raise ValueError('matched optimizer/loss schedule changed')
    if (config['architecture']['fusion_mode'] != ROUTES[route]
            or config['architecture'].get('frame_residual_scale', 0) != 0):
        raise ValueError('the route differs from its declared fusion architecture')
    if config.get('reference_route') != ('legacy_query_first' if route == 'legacy' else None):
        raise ValueError('legacy execution must use the actual query-first reference')
    if config['normal_match']['base_initial_state_sha256'] != BASE_INITIAL_SHA256:
        raise ValueError('the shared official-initialization tensor reference changed')
    expected_eval = {'train_assets': 32, 'valid_assets': 128, 'seed': 20261014, 'frames': 8,
                     'generated_frame_indices': list(range(8)), 'max_new_tokens': 2048, 'initial': True}
    if config['evaluation'] != expected_eval:
        raise ValueError('the shared trend evaluation membership/frame contract changed')
    if any(key in config for key in ('comparison', 'probe', 'cache', 'formal_release')):
        raise ValueError('normal online training must not inherit a fixed-pose cache or formal release')
    # Construction validates every physical-batch entry and exact milestone boundary.
    make_stream(config, rank=0, world=world)
    return list(MILESTONES)


def make_stream(config, rank, world):
    return MatchedFrameStream(range(config['data']['manifests']['train']['rows']), rank=rank, world_size=world,
        seed=config['data']['stream_seed'], target_assets=config['training']['target_assets'],
        effective_assets=config['training']['effective_assets_per_update'],
        frame_policy=config['training']['frame_policy'], manifest_sha256=config['data']['manifests']['train']['sha256'],
        update_boundaries=MILESTONES)


def initialization_fingerprints(model, route):
    full = parameter_fingerprint(model)
    # C/D add fusion-only parameters; all pre-existing tensors must still match A/B.
    shared = (parameter_fingerprint(model, exclude_parts=('.relative_position_bias.', '.token_film.'))
              if route in {'c', 'd'} else full)
    return {'full_base_initial_state_sha256': full, 'shared_initial_state_sha256': shared}


def comparison_contract(config):
    architecture = deepcopy(config['architecture'])
    architecture.pop('fusion_mode')
    optimization = deepcopy(config['optimization'])
    optimization.pop('auxiliary_weight')
    return {'purpose': config['purpose'], 'data': config['data'], 'architecture': architecture,
            'initialization': config['initialization'], 'surface_sampling': config['surface_sampling'],
            'auxiliary_codec': config['auxiliary_codec'], 'optimization': optimization,
            'evaluation': config['evaluation'], 'normal_match': config['normal_match'],
            'training': {key: config['training'][key] for key in
                         ('target_assets', 'effective_assets_per_update', 'checkpoint_assets', 'evaluation_assets')}}


def verify_selection(config, entries):
    spec = config['normal_match']['selection']
    if sha256(spec['path']) != spec['sha256']:
        raise ValueError('5000-asset selection evidence changed')
    receipt = json.loads(Path(spec['path']).read_text())
    if (receipt['status'] != 'passed' or receipt['derived_manifest'] != config['data']['manifests']['train']
            or receipt['source']['train_manifest_sha256_verified'] != SOURCE_HASHES['train']
            or receipt['source']['valid_manifest_sha256_verified'] != SOURCE_HASHES['valid']
            or not receipt['rows_unchanged']['full_json_rows_preserved']
            or not receipt['rows_unchanged']['root_rt_preserved_for_selected_rows']
            or receipt['actual_frame_count']['minimum_actual_frame_count'] < 32):
        raise ValueError('normal subset is not the validated unchanged 5000-row selection')
    for field in ('asset_id', 'path'):
        train, valid = ([row[field] for row in entries[split]] for split in ('train', 'valid'))
        if len(set(train)) != 5000 or len(set(valid)) != 967 or set(train) & set(valid):
            raise ValueError('training/validation identities overlap or duplicate')


def selected_evaluation_rows(entries, config):
    evaluation = config['evaluation']
    selected = {}
    for number, (split, count) in enumerate([('train', evaluation['train_assets']), ('valid', evaluation['valid_assets'])]):
        indices = np.random.default_rng(np.random.SeedSequence([evaluation['seed'], number])).permutation(len(entries[split]))[:count]
        selected[split] = [entries[split][int(index)] for index in sorted(indices)]
    return {'train_seen': selected['train'], 'train_heldout': selected['train'], 'valid': selected['valid']}


def record_samples(geometry, requests):
    frames = geometry['selected_frames'].detach().cpu().tolist()
    heldout = geometry['heldout_raw_ids']
    if len(frames) != len(requests) or len(geometry['asset_id']) != len(requests) or len(heldout) != len(requests):
        raise ValueError('actual loaded microbatch count differs from sample requests')
    for selected, reserved, request in zip(frames, heldout, requests):
        if (len(selected) != request.frames or len(set(selected)) != request.frames or len(set(reserved)) != 8
                or set(selected) & set(reserved)):
            raise ValueError('actual training poses duplicate, leak heldout frames, or differ in count')
    return {'asset_ids': list(geometry['asset_id']), 'manifest_indices': [r.item_index for r in requests],
            'sampling_seeds': [r.sampling_seed for r in requests], 'selected_raw_frames': frames}


def validate_evaluation_rows(rows, selected, frames):
    expected = {(row['asset_id'], frame) for row in selected for frame in range(frames)}
    actual = [(row['asset_id'], row['frame_index']) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('evaluation did not generate every selected asset/frame exactly once')
    for item in selected:
        raw = [row['raw_frame_index'] for row in rows if row['asset_id'] == item['asset_id']]
        if len(set(raw)) != frames:
            raise ValueError('evaluation duplicated a raw pose')


def run_evaluation(model, tokenizer, codec, entries, config, args, device, samples, rank, world):
    chosen = selected_evaluation_rows(entries, config)
    root = args.output / f'evaluation_{samples:06d}'
    root.mkdir(exist_ok=True)
    report = {'complete': False, 'samples': samples, 'rank': rank, 'groups': {}}
    auxiliary_applicable = config['route'] != 'legacy'
    for group, rows in chosen.items():
        local = rows[rank::world]
        dataset = fixed_dataset(local, config, 8, mode='heldout_eval' if group == 'train_heldout' else 'seen_eval')
        def items():
            for index in range(len(dataset)):
                surface, target, geometry = core.prepare(dataset, [index], config, tokenizer, codec, device)
                actual = geometry['selected_frames'][0].cpu().tolist()
                reserved = geometry['heldout_raw_ids'][0]
                if group == 'train_heldout' and set(actual) != set(reserved):
                    raise ValueError('heldout evaluation must include all eight reserved poses')
                if group != 'train_heldout' and set(actual) & set(reserved):
                    raise ValueError('seen/validation sampling crossed its declared frame partition')
                yield surface, target, geometry
        directory = root / f'{group}_rank_{rank}'
        directory.mkdir(exist_ok=False)
        eval_config = {**config, 'probe': {'auxiliary_weight': config['optimization']['auxiliary_weight'],
                                         'max_new_tokens': 2048, 'evaluate_auxiliary': auxiliary_applicable}}
        result = core.evaluate(model, items(), tokenizer, eval_config, directory, samples, frame_indices=list(range(8)))
        validate_evaluation_rows(result['rows'], local, 8)
        report['groups'][group] = result
        write_json(root / f'rank_{rank}.json', report)
    report['complete'] = True
    write_json(root / f'rank_{rank}.json', report)
    if world > 1:
        dist.barrier()
    if rank == 0:
        ranks = [json.loads((root / f'rank_{r}.json').read_text()) for r in range(world)]
        combined = {group: [row for result in ranks for row in result['groups'][group]['rows']] for group in chosen}
        for group, rows in combined.items():
            validate_evaluation_rows(rows, chosen[group], 8)
        result = {'complete': True, 'samples': samples,
                  'summary': {group: core.evaluation_summary(rows) for group, rows in combined.items()},
                  'failed_generations_scored_zero': True, 'auxiliary_applicable': auxiliary_applicable,
                  'full_validation': False, 'evaluation_kind': 'fixed_random_trend_panel',
                  'evaluation_assets': {key: len(value) for key, value in chosen.items()},
                  'frames_per_asset': 8, 'quality_acceptance': False}
        if auxiliary_applicable:
            auxiliary = {group: [row for report in ranks for row in report['groups'][group]['auxiliary_rows']]
                         for group in chosen}
            for group, rows in auxiliary.items():
                if sorted(row['asset_id'] for row in rows) != sorted(row['asset_id'] for row in chosen[group]):
                    raise ValueError('auxiliary evaluation must include every selected asset exactly once')
            result['auxiliary_summary'] = {group: {'assets': len(rows), **auxiliary_summary(rows)}
                                           for group, rows in auxiliary.items()}
        write_json(root / 'summary.json', result)
    if world > 1:
        dist.barrier()
    model.train()


def validate_capacity(config, receipt):
    if (not receipt.get('accepted') or receipt.get('world_size') != 2
            or receipt.get('capacity_contract') != core.capacity_contract(config)):
        raise ValueError('training requires a matching accepted two-rank capacity receipt')
    covered = [case for case in receipt['approved_cases'] if case.get('ranks_approved') == [0, 1]
               and case.get('accumulation_microbatches') == 2]
    for frames in range(2, 25):
        batch = config['training']['frame_policy'][str(frames)]
        groups = ['longest_main', f'geometry_t{frames}']
        if config['route'] != 'legacy':
            groups.append('largest_auxiliary')
        # A bucket's longest sequence covers shorter sequences at the SAME batch.
        # Geometry aliases are recorded only for identical stress-asset membership.
        if any(not any(case['frames'] >= frames and case['batch_per_rank'] == batch
                       and group in case.get('covered_stress_groups', [case['stress_group']])
                       for case in covered) for group in groups):
            raise ValueError(f'unmeasured normal-training capacity at T={frames}, B={batch}')


def declared_operation(state, args, config):
    operation = state['active_operation']
    if (state.get('architecture_contract', {}).get('id') != 'virtual_query_xyz_aux_ids_v1'
            or operation.get('mode') != 'normal5000' or operation.get('gpu_work_allowed') is not True
            or operation.get('formal_training_allowed') is not False or operation.get('complete') is not False
            or operation.get('required_entrypoint') != ENTRYPOINT):
        raise ValueError('bounded normal5000 operation is not published')
    matches = [run for run in operation['runs'] if Path(run['config']).resolve() == args.config.resolve()]
    if len(matches) != 1:
        raise ValueError('configuration is not uniquely published')
    run = matches[0]
    if (run['config_sha256'] != sha256(args.config) or Path(run['output']).resolve() != args.output.resolve()
            or run['mode'] != args.mode or run.get('approved') is not True
            or run['physical_gpus'] != config['runtime']['physical_gpus']):
        raise ValueError('mode/config/output/devices differ from the approved run')
    if args.mode == 'train':
        spec = run['capacity_receipt']
        if sha256(spec['path']) != spec['sha256']:
            raise ValueError('reviewed capacity receipt changed')
        validate_capacity(config, json.loads(Path(spec['path']).read_text()))
    return operation


def verify_operation(args, config, rank):
    if (ROOT.resolve() != CODE_ROOT.resolve() or args.config.resolve().parent != (ROOT / 'model_training/experiments').resolve()
            or args.output.resolve().parent != OUTPUT_ROOT.resolve()):
        raise ValueError('normal work must use the separate published checkout and output root')
    state = json.loads((ROOT / 'model_training/state/current.json').read_text())
    operation = declared_operation(state, args, config)
    if rank == 0:
        subprocess.run([sys.executable, str(ROOT / 'model_training/tools/agent_work_guard.py'),
                        'check', '--operation', 'train', '--entrypoint', ENTRYPOINT], check=True)
    if subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
        raise ValueError('GPU work requires clean committed code')
    subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', operation['code_commit'], 'HEAD'], check=True)
    subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet', operation['code_commit'], 'HEAD', '--',
                    'model_training/rigweave', 'model_training/tools'], check=True)
    return subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()


def capacity_result(args, config, world):
    ranks = [json.loads((args.output / f'capacity_rank_{r}.json').read_text()) for r in range(world)]
    spec = config['capacity']['selection']
    if sha256(spec['path']) != spec['sha256']:
        raise ValueError('capacity stress selection changed during profiling')
    selection = json.loads(Path(spec['path']).read_text())
    groups = dict(selection['groups'])
    groups.update({f'geometry_t{frames}': [row] for frames, row in selection['geometry_by_frames'].items()})
    members = {name: [(row['asset_id'], row['path']) for row in values] for name, values in groups.items()}
    approved = []
    for case in ranks[0]['cases']:
        identity = {key: case[key] for key in ('frames', 'batch_per_rank', 'stress_group')}
        matches = [[other for other in report['cases'] if all(other[key] == value for key, value in identity.items())]
                   for report in ranks]
        if all(len(values) == 1 and values[0]['accepted'] for values in matches):
            approved.append({**identity, 'ranks_approved': list(range(world)), 'accumulation_microbatches': 2,
                             'covered_stress_groups': [name for name, values in members.items()
                                                       if values == members[case['stress_group']]],
                             'peak_allocated_bytes': max(m['peak_allocated_bytes'] for values in matches
                                                         for m in values[0]['measurements'])})
    return {'accepted': all(report['accepted'] for report in ranks), 'world_size': world,
            'capacity_contract': core.capacity_contract(config), 'approved_cases': approved,
            'shorter_frame_coverage': 'same physical batch, identical stress assets, at least15percent memory headroom',
            'capacity_only_no_weights_reused': True, 'single_rank_exploration': world == 1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['profile', 'train'], required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    rank, world, local = (int(os.environ.get(key, default)) for key, default in
                          [('RANK', 0), ('WORLD_SIZE', 1), ('LOCAL_RANK', 0)])
    milestones = validate_config(config, args.mode, world)
    source = verify_operation(args, config, rank)
    physical = config['runtime']['physical_gpus']
    if (socket.gethostname().split('.')[0] != 'pcg02' or physical not in ([7], [5, 7])
            or os.environ.get('CUDA_VISIBLE_DEVICES') != ','.join(map(str, physical)) or len(physical) != world
            or not torch.cuda.is_available() or torch.cuda.device_count() != world):
        raise ValueError('normal work requires its granted pcg02 physical GPU resources')
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.cuda.set_per_process_memory_fraction(.95, device)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=45))
    try:
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=False)
        if world > 1:
            dist.barrier()
        entries = read_entries(config)
        verify_selection(config, entries)
        stream = make_stream(config, rank, world)
        model, tokenizer, codec, initial = build_model(config, device)
        initial.update(**initialization_fingerprints(model, config['route']),
                       comparison_contract_sha256=tree_fingerprint(comparison_contract(config)),
                       exposure_plan_sha256=stream.plan_sha256, source=source, config_sha256=sha256(args.config),
                       route=config['route'], auxiliary_applicable=config['route'] != 'legacy',
                       old_method_trained_weights_loaded=False, optimizer_updates=0)
        if initial['shared_initial_state_sha256'] != BASE_INITIAL_SHA256:
            raise ValueError('actual shared initialization differs before optimizer creation')
        write_json(args.output / f'initialization_rank_{rank}.json', initial)
        write_json(args.output / f'exposure_plan_rank_{rank}.json', stream.plan_report())
        if config['route'] == 'legacy':
            model = LegacyRig(model, 'legacy_query_first')
        gradient_check = require_reference_gradients if config['route'] == 'legacy' else core.require_gradients
        if args.mode == 'profile':
            core.run_profile(model, tokenizer, codec, initial, entries, config, args, device, rank, world, source,
                             gradient_check=gradient_check)
        else:
            if rank == 0:
                write_json(args.output / 'evaluation_selection.json', selected_evaluation_rows(entries, config))
            run_evaluation(model, tokenizer, codec, entries, config, args, device, 0, rank, world)
            core.run_training(model, tokenizer, codec, initial, entries, config, args, device, rank, world, source,
                              stream=stream, milestones=milestones, evaluation_callback=run_evaluation,
                              gradient_check=gradient_check, sample_recorder=record_samples)
        if world > 1:
            dist.barrier()
        if rank == 0:
            extra = capacity_result(args, config, world) if args.mode == 'profile' else {
                'accepted': False, 'acceptance_pending_primary_review': True,
                'asset_exposures': 20000, 'updates': len(stream.update_sizes),
                'supervised_frame_exposures': stream.plan_report()['supervised_frame_exposures']}
            write_json(args.output / 'result.json', {'execution_complete': True, 'mode': args.mode,
                'purpose': PURPOSE, 'route': config['route'], 'code_commit': source,
                'formal_training_completed': False, 'quality_acceptance': False, **extra})
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
