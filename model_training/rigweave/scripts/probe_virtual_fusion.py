#!/usr/bin/env python3
"""Published, isolated fusion trials and read-only auxiliary checkpoint audit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'model_training/rigweave/src'))
from virtual_query_xyz_runtime import build_model, read_entries, fixed_dataset, prepare_batch, sha256, write_json
from probe_virtual_query_xyz import declared_probe_run, run as run_fit
from fit_cohort_selection import choose_fit_rows
from fusion_evaluation import auxiliary_record, auxiliary_summary, tree_fingerprint


def verify_operation(args):
    state = json.loads((ROOT / 'model_training/state/current.json').read_text())
    operation = state['active_operation']
    if state.get('architecture_contract', {}).get('id') != 'virtual_query_xyz_aux_ids_v1':
        raise ValueError('wrong architecture authority')
    if not operation.get('gpu_work_allowed') or operation.get('formal_training_allowed') or operation.get('complete'):
        raise ValueError('only explicitly published bounded fusion operations are allowed')
    run = declared_probe_run(operation, args, sha256(args.config))
    expected_parent = (Path('/work/wangyy/virtual_fusion_20261013/cache') if args.mode == 'cache'
                       else Path('/home/wangyy/evorig/outputs/virtual_fusion_20261013'))
    if args.output.resolve().parent != expected_parent.resolve():
        raise ValueError('fusion outputs/cache must use their isolated roots')
    subprocess.run([sys.executable, str(ROOT / 'model_training/tools/agent_work_guard.py'),
        'check', '--operation', 'train', '--entrypoint', str(Path(__file__).resolve())], check=True)
    if subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
        raise ValueError('GPU work requires clean committed source')
    subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', operation['code_commit'], 'HEAD'], check=True)
    subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet', operation['code_commit'], 'HEAD', '--',
        'model_training/rigweave', 'model_training/tools'], check=True)
    return subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(), run


def validate_config(config, mode):
    runtime = config['runtime']
    if runtime['repo'] != '/work/wangyy/virtual_fusion_20261013/code':
        raise ValueError('fusion code must use its isolated checkout')
    if config['initialization']['checkpoint'] != config['initialization']['official_checkpoint']:
        raise ValueError('new trials cannot initialize from experimental checkpoints')
    if config['probe']['auxiliary_weight'] != .1 or config['architecture'].get('frame_residual_scale', 0) != 0:
        raise ValueError('these comparisons require auxiliary weight .1 and no extra frame residual')
    if not config['probe'].get('evaluate_auxiliary'):
        raise ValueError('full auxiliary generation evaluation is mandatory')
    if mode not in {'aux_audit', 'fit', 'cache', 'compare'}:
        raise ValueError('unsupported bounded fusion stage')
    if mode == 'fit':
        probe = config['probe']
        if (probe['assets'], probe['frames'], probe['assets_per_update'], probe['max_updates']) != (8, 4, 8, 1024):
            raise ValueError('fixed8/T4 fitting requires matched 1024 updates and effective batch8')
        if probe.get('early_stop', True) or not probe.get('paired_output') or not probe.get('fingerprint_initial_state'):
            raise ValueError('paired full-budget fits require input/init verification, without selective early stop')
        if Path(probe['paired_output']).parent != Path('/home/wangyy/evorig/outputs/virtual_fusion_20261013'):
            raise ValueError('paired trial must also use the isolated output root')
        if config['optimization']['schedule'] != 'onecycle' or config['optimization']['schedule_samples'] != 8192:
            raise ValueError('fitting schedule must span 8192 actual asset exposures')
    if mode in {'cache', 'compare'}:
        comparison = config['comparison']
        if (comparison['frames'], comparison['assets_per_update'], comparison['max_updates']) != (4, 48, 512):
            raise ValueError('medium comparison requires T4/effective48/512updates')
        if config['optimization']['schedule'] != 'onecycle' or config['optimization']['schedule_samples'] != 24576:
            raise ValueError('medium comparison schedule must span 24576 actual asset exposures')
        if comparison['generation_frame_indices'] != [0, 1, 2, 3]:
            raise ValueError('medium fusion evaluation must cover every input frame')
        if not comparison.get('initial_evaluation', True):
            raise ValueError('different fusion modes require their own initial-generation evaluation')
        if mode == 'compare':
            cache = Path(comparison['cache']['index'])
            if not cache.resolve().is_relative_to(Path('/work/wangyy/virtual_fusion_20261013/cache').resolve()):
                raise ValueError('raw geometry cache must remain inside the verified isolated SSD root')


@torch.no_grad()
def run_aux_audit(config, output, device, source):
    from rigweave.dynamic_rig.virtual_query_xyz import sequence_ce

    spec = config['audit_checkpoint']
    if sha256(spec['path']) != spec['sha256']:
        raise ValueError('read-only audit checkpoint fingerprint changed')
    model, tokenizer, codec, initial = build_model(config, device)
    payload = torch.load(spec['path'], map_location='cpu', mmap=True, weights_only=False)
    if payload['config']['probe']['auxiliary_weight'] != .1 or payload['step'] != 2048:
        raise ValueError('audit is restricted to the recorded lambda.1 step2048 diagnostic')
    for key in ('data', 'surface_sampling', 'auxiliary_codec'):
        if payload['config'][key] != config[key]:
            raise ValueError(f'audit input contract changed: {key}')
    model.load_state_dict(payload['model'], strict=True)
    checkpoint_step = payload['step']
    del payload
    model.eval()
    before = tree_fingerprint(model.state_dict())
    selected, selection = choose_fit_rows(read_entries(config)['train'], config, return_report=True)
    write_json(output / 'cohort_selection.json', selection)
    dataset = fixed_dataset(selected, config, config['probe']['frames'])
    items = [prepare_batch(dataset[i], config, tokenizer, codec, device) for i in range(len(dataset))]
    virtuals = []
    for surface, _, _ in items:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, virtual = model.encode(surface)
        virtuals.append(virtual)
    rows, swapped = [], []
    for index, (_, targets, geometry) in enumerate(items):
        size = int(geometry['joint_count'][0])
        parents = geometry['target_parents'][0, :size].cpu().tolist()
        asset_id = geometry['asset_id'][0]
        donor = (index + 1) % len(items)
        for kind, condition, destination in [('correct', virtuals[index], rows), ('swapped', virtuals[donor], swapped)]:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model.generate_auxiliary(condition, bos_token=codec.bos,
                    eos_token=codec.eos, max_new_tokens=codec.max_tokens-1)
                logits = model.auxiliary(condition, targets['aux_input_ids'], targets['aux_attention_mask'])
                loss, accuracy = sequence_ce(logits, targets['aux_labels'])
            record = auxiliary_record(prediction, codec, parents, asset_id=asset_id)
            record.update(condition=kind, condition_asset_id=items[donor if kind == 'swapped' else index][2]['asset_id'][0],
                          teacher_ce=float(loss.mean()), teacher_accuracy=float(accuracy.mean()))
            destination.append(record)
    after = tree_fingerprint(model.state_dict())
    if before != after:
        raise RuntimeError('read-only auxiliary audit mutated model state')
    report = {'checkpoint': spec, 'checkpoint_step': checkpoint_step, 'read_only': True,
              'model_state_unchanged': True, 'model_state_sha256': before,
              'summary': auxiliary_summary(rows), 'swapped_summary_against_recipient': auxiliary_summary(swapped),
              'unique_correct_predictions': len({tuple(row['generated_tokens']) for row in rows}),
              'changed_after_swap': sum(a['generated_tokens'] != b['generated_tokens'] for a, b in zip(rows, swapped)),
              'rows': rows, 'swapped_rows': swapped, 'input_sha256': tree_fingerprint(items),
              'code_commit': source, 'main_task_not_reevaluated': True}
    write_json(output / 'auxiliary_audit.json', report)
    return {'stage': 'aux_audit', 'accepted': False, 'accepted_meaning': 'diagnosis_only_not_candidate_selection',
            'optimizer_updates': 0, 'auxiliary_summary': report['summary'],
            'unique_correct_predictions': report['unique_correct_predictions'],
            'changed_after_swap': report['changed_after_swap'], 'formal_training_allowed': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['aux_audit', 'fit', 'cache', 'compare'], required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config, args.mode)
    source, declared = verify_operation(args)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(config['runtime']['physical_gpu']):
        raise ValueError('wrong physical GPU visibility')
    if declared['physical_gpu'] != config['runtime']['physical_gpu']:
        raise ValueError('GPU resource differs from published operation')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('this process requires one granted CUDA device')
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.95, 0)
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        if args.mode == 'aux_audit':
            result = run_aux_audit(config, args.output, torch.device('cuda:0'), source)
        elif args.mode == 'fit':
            result = run_fit(args, config, source)
        else:
            from probe_virtual_query_lambda import prepare_cache, run_comparison
            config['_path'] = str(args.config)
            function = prepare_cache if args.mode == 'cache' else run_comparison
            result = function(config, args.output, torch.device('cuda:0'), source)
            if args.mode == 'compare':
                result.update(stage='fusion_comparison',
                    accepted_meaning='fusion_quality_requires_primary_matched_review; auxiliary_weight_fixed_at_.1')
        write_json(args.output / 'result.json', {**result, 'execution_complete': True,
                   'finished_utc': datetime.now(timezone.utc).isoformat(), 'code_commit': source})
    except Exception:
        write_json(args.output / 'result.json', {'accepted': False, 'execution_complete': False,
            'formal_training_allowed': False, 'traceback': traceback.format_exc(), 'code_commit': source})
        raise


if __name__ == '__main__':
    main()
