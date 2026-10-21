#!/usr/bin/env python3
"""Recover the interrupted B60 stream on its replacement H100 allocation."""
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
from types import SimpleNamespace

import torch
import torch.distributed as dist

import train_virtual_fusion_normal as normal
from virtual_fusion_recovery import (
    CHECKPOINT, SOURCE_CONFIG, PLAN_SHA256, restore_stream,
    validate_recovery_config, validate_recovery_payload,
)

ROOT = Path(__file__).resolve().parents[3]
ENTRYPOINT = 'model_training/rigweave/scripts/recover_virtual_fusion_normal.py'
LABELS = {'profile': 'normal_b_recovery_capacity_20261021_v1',
          'train': 'normal_b_recover60000_20261021_v1'}


def declared_operation(state, args, config):
    operation = state['active_operation']
    if (operation.get('mode') != 'normal5000_interrupted_recovery'
            or operation.get('gpu_work_allowed') is not True
            or operation.get('formal_training_allowed') is not False
            or operation.get('complete') is not False
            or operation.get('required_entrypoint') != ENTRYPOINT):
        raise ValueError('interrupted recovery requires its separately published operation')
    runs = [run for run in operation['runs'] if run.get('mode') == args.mode
            and Path(run['config']).resolve() == args.config.resolve()]
    if len(runs) != 1:
        raise ValueError('recovery configuration/mode is not uniquely published')
    run = runs[0]
    if (run.get('approved') is not True or run.get('execution_allowed') is not True
            or run.get('config_sha256') != normal.sha256(args.config)
            or Path(run['output']).resolve() != args.output.resolve()
            or any(run.get(key) != config['runtime'][key] for key in ('allocation', 'host', 'physical_gpus'))
            or run.get('checkpoint') != CHECKPOINT):
        raise ValueError('recovery runtime/output/checkpoint differs from its reviewed declaration')
    if args.mode == 'train':
        spec = run['capacity_receipt']
        if normal.sha256(spec['path']) != spec['sha256']:
            raise ValueError('new-device capacity receipt changed')
        normal.validate_capacity(config, json.loads(Path(spec['path']).read_text()))
        if run.get('training_allowed') is not True:
            raise ValueError('recovery training has not passed the capacity gate')
    return operation


def verify_operation(args, config, rank):
    if (ROOT.resolve() != normal.CODE_ROOT.resolve()
            or args.config.resolve().parent != (ROOT / 'model_training/experiments').resolve()
            or args.output.resolve().parent != normal.OUTPUT_ROOT.resolve()
            or args.output.name != LABELS[args.mode]):
        raise ValueError('recovery must use its new exclusive output and published checkout')
    state = json.loads((ROOT / 'model_training/state/current.json').read_text())
    operation = declared_operation(state, args, config)
    if rank == 0:
        subprocess.run([sys.executable, str(ROOT / 'model_training/tools/agent_work_guard.py'),
                        'check', '--operation', 'preflight' if args.mode == 'profile' else 'train',
                        '--entrypoint', ENTRYPOINT], check=True)
    if subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
        raise ValueError('recovery requires clean committed source')
    subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', operation['code_commit'], 'HEAD'], check=True)
    subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet', operation['code_commit'], 'HEAD', '--',
                    'model_training/rigweave', 'model_training/tools'], check=True)
    return subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['profile', 'train'], required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if normal.sha256(SOURCE_CONFIG['path']) != SOURCE_CONFIG['sha256']:
        raise ValueError('the frozen interrupted-run configuration changed')
    source_config = json.loads(Path(SOURCE_CONFIG['path']).read_text())
    rank, world, local = (int(os.environ.get(key, default)) for key, default in
                         [('RANK', 0), ('WORLD_SIZE', 1), ('LOCAL_RANK', 0)])
    milestones = validate_recovery_config(config, source_config, world)
    source = verify_operation(args, config, rank)
    if (socket.gethostname().split('.')[0] != 'pcg01'
            or os.environ.get('CUDA_VISIBLE_DEVICES') != '5,6'
            or not torch.cuda.is_available() or torch.cuda.device_count() != 2 or local != rank):
        raise ValueError('recovery requires the approved pcg01 physical H100 devices5,6')
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.cuda.set_per_process_memory_fraction(.95, device)
    dist.init_process_group('nccl', timeout=timedelta(minutes=45))
    try:
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        entries = normal.read_entries(config)
        normal.verify_selection(config, entries)
        if rank == 0:
            normal.write_json(args.output / 'evaluation_selection.json', normal.selected_evaluation_rows(entries, config))
        model, tokenizer, codec, initialization = normal.build_model(config, device)
        if args.mode == 'profile':
            initialization.update(**normal.initialization_fingerprints(model, 'b'), source=source,
                                  config_sha256=normal.sha256(args.config), route='b', optimizer_updates=0)
            if initialization['shared_initial_state_sha256'] != normal.BASE_INITIAL_SHA256:
                raise ValueError('capacity initialization differs from the original shared tensors')
            normal.write_json(args.output / f'initialization_rank_{rank}.json', initialization)
            normal.core.run_profile(model, tokenizer, codec, initialization, entries, config, args,
                                    device, rank, world, source, gradient_check=normal.core.require_gradients)
            dist.barrier()
            if rank == 0:
                reports = [json.loads((args.output / f'capacity_rank_{index}.json').read_text()) for index in range(world)]
                if not all(report['complete'] and report['accepted'] for report in reports):
                    raise RuntimeError('new-device capacity did not pass on both ranks')
                normal.write_json(args.output / 'result.json', {'execution_complete': True, 'mode': 'profile',
                    'acceptance_pending_primary_review': True, 'code_commit': source, 'route': 'b',
                    'quality_acceptance': False, **normal.capacity_result(args, config, world)})
            return
        path = Path(CHECKPOINT['path'])
        if path.stat().st_size != CHECKPOINT['bytes'] or normal.sha256(path) != CHECKPOINT['sha256']:
            raise ValueError('the pinned interrupted full checkpoint changed')
        payload = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        validate_recovery_payload(payload, source_config)
        stream = restore_stream(payload, rank)
        if model.configuration() != payload['architecture']:
            raise ValueError('recovery architecture differs from the saved checkpoint')
        model.load_state_dict(payload['model'], strict=True)
        expected = normal.parameter_fingerprint(SimpleNamespace(state_dict=lambda: payload['model']))
        restored = normal.parameter_fingerprint(model)
        if expected != restored:
            raise ValueError('loaded model tensors differ from the pinned B50 checkpoint')
        provenance = {'checkpoint': dict(CHECKPOINT), 'source_config': dict(SOURCE_CONFIG), 'route': 'b',
            'source': source, 'config_sha256': normal.sha256(args.config), 'mode': 'interrupted_resume',
            'start_asset_exposures': 50000, 'target_asset_exposures': 60000,
            'start_updates': 1050, 'target_updates': 1260,
            'start_supervised_frame_exposures': 657056, 'target_supervised_frame_exposures': 788304,
            'exposure_plan_sha256': PLAN_SHA256, 'checkpoint_model_state_sha256': expected,
            'restored_model_state_sha256': restored, 'constant_learning_rate': 1e-4,
            'optimizer_state_restore_required': True, 'optimizer_betas': 'restore_checkpoint',
            'rng_policy': config['recovery']['rng_policy'], 'replayed_initial_evaluation': False,
            'sampling_prefix_preserved': True, 'unsaved_exposures_to_replay': 4080,
            'historical_evaluations_preserved': str(path.parent), 'fresh_official_initialization': False,
            'bitwise_future_trajectory_claimed': False, 'quality_acceptance': False}
        initial = deepcopy(payload['initialization'])
        initial['interrupted_recovery'] = provenance
        normal.write_json(args.output / f'continuation_rank_{rank}.json', provenance)
        normal.write_json(args.output / f'exposure_plan_rank_{rank}.json', stream.plan_report())
        resume = {'optimizer_state_dict': payload['optimizer'], 'rank_rng_state': payload['rank_states'][rank],
                  'asset_exposures': 50000, 'supervised_frame_exposures': 657056, 'step': 1050,
                  'learning_rate': 1e-4, 'provenance': provenance}
        del payload
        normal.core.run_training(model, tokenizer, codec, initial, entries, config, args, device, rank, world, source,
            stream=stream, milestones=milestones, evaluation_callback=normal.run_evaluation,
            gradient_check=normal.core.require_gradients, sample_recorder=normal.record_samples, resume_state=resume)
        dist.barrier()
        if rank == 0:
            normal.write_json(args.output / 'result.json', {'execution_complete': True, 'mode': 'interrupted_resume',
                'purpose': 'normal5000_continuation', 'route': 'b', 'code_commit': source,
                'asset_exposures': 60000, 'additional_asset_exposures': 10000,
                'updates': 1260, 'additional_updates': 210, 'supervised_frame_exposures': 788304,
                'continuation': provenance, 'formal_training_completed': False, 'quality_acceptance': False,
                'accepted': False, 'acceptance_pending_primary_review': True})
    except Exception:
        if args.output.exists():
            normal.write_json(args.output / f'failure_rank_{rank}.json', {
                'traceback': traceback.format_exc(), 'execution_complete': False, 'accepted': False,
                'code_commit': source, 'mode': args.mode})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
