#!/usr/bin/env python3
"""Resume a pinned completed route checkpoint under an explicit continuation gate."""
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
    BASE_SHA256, CHECKPOINT, PURPOSE, SOURCE_CONFIG, continuation_contract, original_references,
    extend_checkpoint_stream, validate_checkpoint_payload, validate_continuation_config,
)

ROOT = Path(__file__).resolve().parents[3]
ENTRYPOINT = 'model_training/rigweave/scripts/resume_virtual_fusion_normal.py'
C_SOURCE = 'cf1c1ae54cdcaef002b9c4d665f1133903e76a5d'


def declared_operation(state, args, config):
    contract = continuation_contract(config)
    operation = state['active_operation']
    if (state.get('architecture_contract', {}).get('id') != 'virtual_query_xyz_aux_ids_v1'
            or operation.get('mode') != PURPOSE or operation.get('gpu_work_allowed') is not True
            or operation.get('formal_training_allowed') is not False or operation.get('complete') is not False
            or operation.get('required_entrypoint') != ENTRYPOINT):
        raise ValueError('the explicit continuation operation is not published')
    matches = [run for run in operation['runs'] if Path(run['config']).resolve() == args.config.resolve()]
    if len(matches) != 1:
        raise ValueError('continuation configuration is not uniquely published')
    run = matches[0]
    if (run.get('mode') != 'resume' or run.get('approved') is not True
            or run.get('training_allowed') is not True
            or run['config_sha256'] != normal.sha256(args.config)
            or Path(run['output']).resolve() != args.output.resolve()
            or run['physical_gpus'] != config['runtime']['physical_gpus']
            or run.get('allocation') != config['runtime']['allocation']
            or run.get('host') != config['runtime']['host']
            or run.get('resume_checkpoint') != contract['checkpoint']):
        raise ValueError('continuation config/output/devices/checkpoint differ from the approved run')
    capacity = run['capacity_receipt']
    if normal.sha256(capacity['path']) != capacity['sha256']:
        raise ValueError('reviewed route capacity receipt changed')
    normal.validate_capacity(config, json.loads(Path(capacity['path']).read_text()))
    predecessor = operation['completed_predecessor']
    predecessor_label = contract['predecessor_label']
    expected_result = normal.OUTPUT_ROOT / predecessor_label / 'result.json'
    expected_exit = Path(contract.get('predecessor_jobdir', '/home/wangyy/jobs/virtual_fusion_normal_20261014_v1')) / (predecessor_label + '.exit')
    if (Path(predecessor['result']).resolve() != expected_result.resolve()
            or Path(predecessor['exit']).resolve() != expected_exit.resolve()
            or expected_exit.read_text().strip() != '0'
            or normal.sha256(expected_result) != predecessor['result_sha256']):
        raise ValueError('required predecessor completion evidence is absent or changed')
    result = json.loads(expected_result.read_text())
    expected = contract['predecessor_result']
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError('required predecessor has not completed its approved budget')
    return operation


def verify_operation(args, config, rank):
    contract = continuation_contract(config)
    if (ROOT.resolve() != normal.CODE_ROOT.resolve()
            or args.config.resolve().parent != (ROOT / 'model_training/experiments').resolve()
            or args.output.resolve().parent != normal.OUTPUT_ROOT.resolve()
            or args.output.name != contract['label']):
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
    contract = continuation_contract(config)
    original_checkpoint, original_config_ref = original_references(contract['route'])
    if normal.sha256(original_config_ref['path']) != original_config_ref['sha256']:
        raise ValueError('original route configuration changed')
    original = json.loads(Path(original_config_ref['path']).read_text())
    source_config_ref = contract['source_config']
    if normal.sha256(source_config_ref['path']) != source_config_ref['sha256']:
        raise ValueError('completed source configuration changed')
    source_config = json.loads(Path(source_config_ref['path']).read_text())
    rank, world, local = (int(os.environ.get(key, default)) for key, default in
                          [('RANK', 0), ('WORLD_SIZE', 1), ('LOCAL_RANK', 0)])
    normal.validate_config(original, 'train', world)
    milestones = validate_continuation_config(config, original, world)
    source = verify_operation(args, config, rank)
    checkpoint_spec = contract['checkpoint']
    checkpoint_path = Path(checkpoint_spec['path'])
    if (checkpoint_path.stat().st_size != checkpoint_spec['bytes']
            or normal.sha256(checkpoint_path) != checkpoint_spec['sha256']):
        raise ValueError('the pinned completed full checkpoint changed')
    physical_host = config['runtime']['host'][:-1] if config['runtime']['host'].endswith('i') else config['runtime']['host'].split('.')[0]
    physical_gpus = config['runtime']['physical_gpus']
    if (socket.gethostname().split('.')[0] != physical_host
            or os.environ.get('CUDA_VISIBLE_DEVICES') != ','.join(map(str, physical_gpus))
            or not torch.cuda.is_available()
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
        validate_checkpoint_payload(payload, original, world, config=config, source_config=source_config)
        stream = extend_checkpoint_stream(payload, rank, target_assets=contract['target_assets'], route=contract['route'])
        model, tokenizer, codec, _ = normal.build_model(config, device)
        if model.configuration() != payload['architecture']:
            raise ValueError('restored model architecture differs from the saved checkpoint')
        model.load_state_dict(payload['model'], strict=True)
        expected_weights = normal.parameter_fingerprint(SimpleNamespace(state_dict=lambda: payload['model']))
        restored_weights = normal.parameter_fingerprint(model)
        if expected_weights != restored_weights:
            raise ValueError('loaded model tensors differ from the pinned completed checkpoint')
        provenance = {
            'checkpoint': dict(checkpoint_spec), 'source_config': dict(source_config_ref), 'route': contract['route'],
            'source': source, 'config_sha256': normal.sha256(args.config),
            'start_asset_exposures': contract['start_assets'], 'target_asset_exposures': contract['target_assets'],
            'start_updates': contract['start_updates'], 'target_updates': len(stream.update_sizes),
            'start_supervised_frame_exposures': contract['start_frames'],
            'target_supervised_frame_exposures': stream.plan_report()['supervised_frame_exposures'],
            'original_exposure_plan_sha256': contract['old_plan_sha256'],
            'extended_exposure_plan_sha256': stream.plan_sha256,
            'original_shared_initial_state_sha256': BASE_SHA256,
            'checkpoint_model_state_sha256': expected_weights, 'restored_model_state_sha256': restored_weights,
            'constant_learning_rate': config['continuation']['learning_rate'],
            'optimizer_betas': 'restore_checkpoint', 'optimizer_state_restore_required': True,
            'rng_policy': config['continuation']['rng_policy'], 'replayed_initial_evaluation': False,
            'historical_20000_evaluation_preserved': str(Path(original_checkpoint['path']).parent / 'evaluation_020000/summary.json'),
            'source_evaluation_preserved': str(checkpoint_path.parent / f"evaluation_{contract['start_assets']:06d}/summary.json"),
            'saved_prefix_asset_exposures_verified': contract['start_assets'],
            'fresh_official_initialization': False, 'resumed_trained_weights_loaded': True,
            'original_20000_prefix_verified': True, 'quality_acceptance': False,
        }
        initial = deepcopy(payload['initialization'])
        if 'training_continuation' in initial:
            initial.setdefault('training_continuation_history', []).append(deepcopy(initial['training_continuation']))
        initial['training_continuation'] = provenance
        normal.write_json(args.output / f'continuation_rank_{rank}.json', provenance)
        normal.write_json(args.output / f'exposure_plan_rank_{rank}.json', stream.plan_report())
        resume = {'optimizer_state_dict': payload['optimizer'], 'rank_rng_state': payload['rank_states'][rank],
                  'asset_exposures': contract['start_assets'], 'supervised_frame_exposures': contract['start_frames'],
                  'step': contract['start_updates'],
                  'learning_rate': config['continuation']['learning_rate'], 'provenance': provenance}
        del payload
        normal.core.run_training(model, tokenizer, codec, initial, entries, config, args, device, rank, world, source,
            stream=stream, milestones=milestones, evaluation_callback=normal.run_evaluation,
            gradient_check=normal.core.require_gradients, sample_recorder=normal.record_samples, resume_state=resume)
        dist.barrier()
        if rank == 0:
            normal.write_json(args.output / 'result.json', {
                'execution_complete': True, 'mode': 'resume', 'purpose': PURPOSE, 'route': contract['route'], 'code_commit': source,
                'asset_exposures': contract['target_assets'],
                'additional_asset_exposures': contract['target_assets'] - contract['start_assets'],
                'updates': len(stream.update_sizes),
                'additional_updates': len(stream.update_sizes) - contract['start_updates'],
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
