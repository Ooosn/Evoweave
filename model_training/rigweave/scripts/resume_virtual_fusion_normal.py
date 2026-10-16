#!/usr/bin/env python3
"""Resume the pinned A20k checkpoint to40k after the current C run completes."""
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
from virtual_fusion_continuation import (
    BASE_SHA256, CHECKPOINT, MILESTONES, OLD_PLAN_SHA256, PURPOSE, SOURCE_CONFIG,
    START_ASSETS, START_FRAMES, START_UPDATES, TARGET_ASSETS,
    extend_checkpoint_stream, validate_checkpoint_payload, validate_continuation_config,
)

ROOT = Path(__file__).resolve().parents[3]
ENTRYPOINT = 'model_training/rigweave/scripts/resume_virtual_fusion_normal.py'
C_SOURCE = 'cf1c1ae54cdcaef002b9c4d665f1133903e76a5d'


def declared_operation(state, args, config):
    operation = state['active_operation']
    if (state.get('architecture_contract', {}).get('id') != 'virtual_query_xyz_aux_ids_v1'
            or operation.get('mode') != PURPOSE or operation.get('gpu_work_allowed') is not True
            or operation.get('formal_training_allowed') is not False or operation.get('complete') is not False
            or operation.get('required_entrypoint') != ENTRYPOINT):
        raise ValueError('the explicit A continuation operation is not published')
    matches = [run for run in operation['runs'] if Path(run['config']).resolve() == args.config.resolve()]
    if len(matches) != 1:
        raise ValueError('continuation configuration is not uniquely published')
    run = matches[0]
    if (run.get('mode') != 'resume' or run.get('approved') is not True
            or run.get('training_allowed') is not True
            or run['config_sha256'] != normal.sha256(args.config)
            or Path(run['output']).resolve() != args.output.resolve()
            or run['physical_gpus'] != config['runtime']['physical_gpus']
            or run.get('resume_checkpoint') != CHECKPOINT):
        raise ValueError('continuation config/output/devices/checkpoint differ from the approved run')
    capacity = run['capacity_receipt']
    if normal.sha256(capacity['path']) != capacity['sha256']:
        raise ValueError('reviewed A capacity receipt changed')
    normal.validate_capacity(config, json.loads(Path(capacity['path']).read_text()))
    predecessor = operation['completed_predecessor']
    expected_result = normal.OUTPUT_ROOT / 'normal_c_v1/result.json'
    expected_exit = Path('/home/wangyy/jobs/virtual_fusion_normal_20261014_v1/normal_c_v1.exit')
    if (Path(predecessor['result']).resolve() != expected_result.resolve()
            or Path(predecessor['exit']).resolve() != expected_exit.resolve()
            or expected_exit.read_text().strip() != '0'
            or normal.sha256(expected_result) != predecessor['result_sha256']):
        raise ValueError('C completion evidence is absent or changed')
    result = json.loads(expected_result.read_text())
    expected = {'execution_complete': True, 'mode': 'train', 'route': 'c', 'code_commit': C_SOURCE,
                'asset_exposures': 20000, 'updates': 420, 'supervised_frame_exposures': 263608}
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError('C has not completed its original approved budget')
    return operation


def verify_operation(args, config, rank):
    if (ROOT.resolve() != normal.CODE_ROOT.resolve()
            or args.config.resolve().parent != (ROOT / 'model_training/experiments').resolve()
            or args.output.resolve().parent != normal.OUTPUT_ROOT.resolve()
            or args.output.name != 'normal_a_extend40000_v1'):
        raise ValueError('continuation must use its separate new output and published checkout')
    state = json.loads((ROOT / 'model_training/state/current.json').read_text())
    operation = declared_operation(state, args, config)
    if rank == 0:
        subprocess.run([sys.executable, str(ROOT / 'model_training/tools/agent_work_guard.py'),
                        'check', '--operation', 'train', '--entrypoint', ENTRYPOINT], check=True)
    if subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
        raise ValueError('continuation requires clean committed source')
    subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', operation['code_commit'], 'HEAD'], check=True)
    subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet', operation['code_commit'], 'HEAD', '--',
                    'model_training/rigweave', 'model_training/tools'], check=True)
    return subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if normal.sha256(SOURCE_CONFIG['path']) != SOURCE_CONFIG['sha256']:
        raise ValueError('original A configuration changed')
    original = json.loads(Path(SOURCE_CONFIG['path']).read_text())
    rank, world, local = (int(os.environ.get(key, default)) for key, default in
                          [('RANK', 0), ('WORLD_SIZE', 1), ('LOCAL_RANK', 0)])
    normal.validate_config(original, 'train', world)
    validate_continuation_config(config, original, world)
    source = verify_operation(args, config, rank)
    checkpoint_path = Path(CHECKPOINT['path'])
    if checkpoint_path.stat().st_size != CHECKPOINT['bytes'] or normal.sha256(checkpoint_path) != CHECKPOINT['sha256']:
        raise ValueError('the pinned A20k full checkpoint changed')
    if (socket.gethostname().split('.')[0] != 'pcg02' or config['runtime']['physical_gpus'] != [5, 7]
            or os.environ.get('CUDA_VISIBLE_DEVICES') != '5,7' or not torch.cuda.is_available()
            or torch.cuda.device_count() != world or local != rank):
        raise ValueError('continuation requires the approved two physical H100 devices')
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
        payload = torch.load(checkpoint_path, map_location='cpu', mmap=True, weights_only=False)
        validate_checkpoint_payload(payload, original, world)
        stream = extend_checkpoint_stream(payload, rank)
        model, tokenizer, codec, _ = normal.build_model(config, device)
        if model.configuration() != payload['architecture']:
            raise ValueError('restored model architecture differs from the saved checkpoint')
        model.load_state_dict(payload['model'], strict=True)
        expected_weights = normal.parameter_fingerprint(SimpleNamespace(state_dict=lambda: payload['model']))
        restored_weights = normal.parameter_fingerprint(model)
        if expected_weights != restored_weights:
            raise ValueError('loaded model tensors differ from the pinned A20k checkpoint')
        provenance = {
            'checkpoint': dict(CHECKPOINT), 'source_config': dict(SOURCE_CONFIG), 'route': 'a',
            'source': source, 'config_sha256': normal.sha256(args.config),
            'start_asset_exposures': START_ASSETS, 'target_asset_exposures': TARGET_ASSETS,
            'start_updates': START_UPDATES, 'target_updates': len(stream.update_sizes),
            'start_supervised_frame_exposures': START_FRAMES,
            'target_supervised_frame_exposures': stream.plan_report()['supervised_frame_exposures'],
            'original_exposure_plan_sha256': OLD_PLAN_SHA256, 'extended_exposure_plan_sha256': stream.plan_sha256,
            'original_shared_initial_state_sha256': BASE_SHA256,
            'checkpoint_model_state_sha256': expected_weights, 'restored_model_state_sha256': restored_weights,
            'constant_learning_rate': config['continuation']['learning_rate'],
            'optimizer_betas': 'restore_checkpoint', 'optimizer_state_restore_required': True,
            'rng_policy': config['continuation']['rng_policy'], 'replayed_initial_evaluation': False,
            'historical_20000_evaluation_preserved': str(checkpoint_path.parent / 'evaluation_020000/summary.json'),
            'fresh_official_initialization': False, 'resumed_trained_weights_loaded': True,
            'original_20000_prefix_verified': True, 'quality_acceptance': False,
        }
        initial = deepcopy(payload['initialization'])
        initial['training_continuation'] = provenance
        normal.write_json(args.output / f'continuation_rank_{rank}.json', provenance)
        normal.write_json(args.output / f'exposure_plan_rank_{rank}.json', stream.plan_report())
        resume = {'optimizer_state_dict': payload['optimizer'], 'rank_rng_state': payload['rank_states'][rank],
                  'asset_exposures': START_ASSETS, 'supervised_frame_exposures': START_FRAMES, 'step': START_UPDATES,
                  'learning_rate': config['continuation']['learning_rate'], 'provenance': provenance}
        del payload
        normal.core.run_training(model, tokenizer, codec, initial, entries, config, args, device, rank, world, source,
            stream=stream, milestones=MILESTONES, evaluation_callback=normal.run_evaluation,
            gradient_check=normal.core.require_gradients, sample_recorder=normal.record_samples, resume_state=resume)
        dist.barrier()
        if rank == 0:
            normal.write_json(args.output / 'result.json', {
                'execution_complete': True, 'mode': 'resume', 'purpose': PURPOSE, 'route': 'a', 'code_commit': source,
                'asset_exposures': TARGET_ASSETS, 'additional_asset_exposures': TARGET_ASSETS - START_ASSETS,
                'updates': len(stream.update_sizes), 'additional_updates': len(stream.update_sizes) - START_UPDATES,
                'supervised_frame_exposures': stream.plan_report()['supervised_frame_exposures'],
                'continuation': provenance, 'formal_training_completed': False, 'quality_acceptance': False,
                'accepted': False, 'acceptance_pending_primary_review': True})
    except Exception:
        if args.output.exists():
            normal.write_json(args.output / f'failure_rank_{rank}.json', {
                'traceback': traceback.format_exc(), 'execution_complete': False, 'accepted': False,
                'code_commit': source, 'mode': 'resume'})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
