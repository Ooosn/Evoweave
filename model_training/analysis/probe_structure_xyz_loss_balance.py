#!/usr/bin/env python3
"""Measure task gradient balance on fixed training cases without model updates."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'model_training/rigweave/src'))
sys.path.insert(0, str(ROOT / 'model_training/rigweave/scripts'))

from structure_condition_metrics import build_prefix_layout, summarize_logits
from structure_xyz_fill_evaluation import _prepare_item, _reference_rows
from structure_xyz_fill_runtime import (
    FAMILY, build_fill_model, data_source_hashes, fixed_dataset, load_small_reference_panel,
    model_source_hashes, prepare_fill_batch, read_entries, reset_random, sha256,
    validate_config, verify_data_selection, write_json,
)
from train_structure_xyz_fill import (
    parameter_fingerprint, published_run, read_reviewed, training_contract,
    validate_initialization_audit,
)

ENTRYPOINT = 'model_training/analysis/probe_structure_xyz_loss_balance.py'


def checked_json(spec):
    if sha256(spec['path']) != spec['sha256']:
        raise ValueError(f"reviewed input changed: {spec['path']}")
    return json.loads(Path(spec['path']).read_text())


def validate_resource_runtime(base, runtime, run):
    # A scheduler handoff may change the grant, never the checkpoint's data/model environment.
    grant_fields = {'allocation', 'host', 'physical_gpus'}
    if ({key: value for key, value in runtime.items() if key not in grant_fields} !=
            {key: value for key, value in base['runtime'].items() if key not in grant_fields}
            or any(run.get(key) != runtime[key] for key in grant_fields)
            or len(runtime['physical_gpus']) != runtime['world_size']
            or len(set(runtime['physical_gpus'])) != runtime['world_size']):
        raise ValueError('diagnostic runtime differs beyond the exact published scheduler grant')


def gradient_statistics(left, right):
    if len(left) != len(right):
        raise ValueError('gradient lists differ')
    aa = bb = ab = 0.
    for a, b in zip(left, right):
        if a is not None:
            a = a.detach().float()
            aa += float(a.square().sum())
        if b is not None:
            b = b.detach().float()
            bb += float(b.square().sum())
        if a is not None and b is not None:
            ab += float((a * b).sum())
    values = np.asarray([aa, bb, ab])
    if not np.isfinite(values).all() or min(aa, bb) < 0:
        raise FloatingPointError('nonfinite task gradients')
    return {'structure_norm': aa ** .5, 'xyz_norm': bb ** .5,
            'structure_over_xyz': (aa / bb) ** .5 if bb else None,
            'cosine': ab / (aa * bb) ** .5 if aa and bb else None,
            'dot': ab}


def summarize(rows):
    result = {}
    for name in ('all', 'joints52', 'non52'):
        chosen = [r for r in rows if name == 'all' or
                  ((r['target_joint_count'] == 52) == (name == 'joints52'))]
        summary = {'cases': len(chosen), 'gradients': {}}
        for key in ('structure_loss', 'xyz_loss', 'structure_accuracy', 'xyz_accuracy'):
            summary[key] = float(np.mean([r[key] for r in chosen])) if chosen else None
        for group in ('surface_tokenizer', 'motion_encoder', 'shared', 'structure_decoder', 'fill_decoder'):
            metrics = {}
            for key in ('structure_norm', 'xyz_norm', 'structure_over_xyz', 'cosine'):
                values = [r['gradients'][group][key] for r in chosen
                          if r['gradients'][group][key] is not None]
                metrics[key] = {'mean': float(np.mean(values)) if values else None,
                                'median': float(np.median(values)) if values else None}
            summary['gradients'][group] = metrics
        summary['structure_categories'] = {}
        for category in ('all', 'choice', 'new_joint_id', 'branch', 'parent_reference', 'eos'):
            records = [r['structure_metrics'][category] if category in ('all', 'choice')
                       else r['structure_metrics']['by_category'][category] for r in chosen]
            summary['structure_categories'][category] = {
                'tokens': sum(r['tokens'] for r in records),
                **{key: float(np.mean([r[key] for r in records if r[key] is not None]))
                   if any(r[key] is not None for r in records) else None
                   for key in ('raw_ce', 'legal_ce', 'raw_accuracy', 'legal_accuracy')}}
        result[name] = summary
    return result


def run_probe(model, tokenizer, codec, base, config, args, rank, world, device):
    entries = read_entries(base)
    verify_data_selection(base, entries)
    expanded = deepcopy(base)
    expanded['evaluation']['train_assets'] = config['train_assets']
    panel, reference, manifest = load_small_reference_panel(expanded, entries)
    selected = panel['train_seen'][rank::world]
    dataset = fixed_dataset(selected, base, base['evaluation']['input_frames'], mode='seen_eval')
    model.train()
    before = parameter_fingerprint(model)
    if before != config['checkpoint']['model_sha256']:
        raise ValueError('loaded model differs from the accepted own80k checkpoint')
    names = ('surface_tokenizer', 'motion_encoder', 'structure_decoder', 'fill_decoder')
    groups = {name: [p for p in getattr(model, name).parameters() if p.requires_grad] for name in names}
    params = [p for name in names for p in groups[name]]
    if len({id(p) for p in params}) != len(params):
        raise ValueError('parameter groups overlap')
    rows = []
    torch.cuda.reset_peak_memory_stats(device)
    for index, entry in enumerate(selected):
        reset_random(config['diagnostic_seed'] + rank * 1000 + index)
        surface, targets, geometry, _ = prepare_fill_batch(dataset, [index], base, codec, device)
        item = _prepare_item(surface, targets, geometry, codec, bins=base['quantization']['bins'])
        if item.asset_id != entry['asset_id'] or set(item.raw_frame_indices.tolist()) & set(geometry['heldout_raw_ids'][0]):
            raise ValueError('diagnostic changed the training asset or used heldout frames')
        _reference_rows(geometry, tokenizer, codec, group='train_seen', asset_id=item.asset_id,
                        frame_indices=base['evaluation']['frame_indices'],
                        raw_frame_indices=item.raw_frame_indices, reference_lookup=reference)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(surface, targets, structure_weight=1., return_logits=True)
        structure = torch.autograd.grad(output['structure_loss'], params, retain_graph=True, allow_unused=True)
        xyz = torch.autograd.grad(output['xyz_loss'], params, allow_unused=True)
        gradients, offset = {}, 0
        for name in names:
            count = len(groups[name])
            gradients[name] = gradient_statistics(structure[offset:offset + count], xyz[offset:offset + count])
            offset += count
        shared_count = len(groups['surface_tokenizer']) + len(groups['motion_encoder'])
        gradients['shared'] = gradient_statistics(structure[:shared_count], xyz[:shared_count])
        if gradients['structure_decoder']['xyz_norm'] != 0 or gradients['fill_decoder']['structure_norm'] != 0:
            raise ValueError('unexpected cross-head gradient route')
        tokens = [codec.bos] + targets['structure_labels'][0].tolist()
        if tokens != item.target_tokens:
            raise ValueError('diagnostic target differs from original geometry')
        layout = build_prefix_layout(tokens, codec)
        scores = summarize_logits(output['structure_logits'][0].detach().float().cpu().numpy(), layout)
        scores.pop('positions', None)
        row = {'asset_id': entry['asset_id'], 'rank': rank,
               'target_joint_count': int(targets['fill_joint_mask'][0].sum()),
               **{key: float(output[key].detach()) for key in
                  ('structure_loss', 'xyz_loss', 'structure_accuracy', 'xyz_accuracy')},
               'gradients': gradients, 'structure_metrics': scores,
               'raw_frame_indices': item.raw_frame_indices.tolist(), 'pruning_applied': False}
        rows.append(row)
        del output, structure, xyz, surface, targets, geometry, item
        peak = torch.cuda.max_memory_allocated(device) / torch.cuda.get_device_properties(device).total_memory
        if peak >= config['allocated_fraction_limit']:
            raise RuntimeError('diagnostic memory limit exceeded')
        print(json.dumps({'event': 'gradient_probe', 'rank': rank, 'asset_id': row['asset_id'],
                          'shared': gradients['shared']}), flush=True)
    after = parameter_fingerprint(model)
    if before != after or any(p.grad is not None for p in params):
        raise ValueError('read-only diagnostic changed model or accumulated parameter gradients')
    local = {'complete': True, 'passed': True, 'rows': rows, 'panel_sha256': manifest['panel_sha256'],
             'model_sha256_unchanged': after, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(device)}
    write_json(args.output / f'rank_{rank}.json', local)
    if config.get('large_evaluation', {}).get('enabled'):
        from structure_xyz_large_evaluation import run_large_evaluation
        large_base = deepcopy(base)
        large_base['large_evaluation'] = config['large_evaluation']
        small_panel, reference, small_manifest = load_small_reference_panel(base, entries)
        run_large_evaluation(model, tokenizer, codec, large_base, args, device, rank, world,
                             config['checkpoint']['asset_exposures'], small_panel, reference,
                             small_manifest['panel_sha256'])
        if parameter_fingerprint(model) != before:
            raise ValueError('evaluation changed model state')
    return local


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('loss_balance_audit',), required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    claimed = False
    try:
        config = json.loads(args.config.read_text())
        base = checked_json(config['base_config'])
        validate_config(base)
        state = json.loads((ROOT / 'model_training/state/current.json').read_text())
        operation = state['active_operation']
        run = published_run(operation, args.config, args.output, args.mode, sha256(args.config))
        validate_resource_runtime(base, config['runtime'], run)
        if (config['purpose'] != 'structure_xyz_loss_balance_audit_v1'
                or state['architecture_contract']['id'] != FAMILY
                or operation.get('gpu_work_allowed') is not True
                or operation.get('formal_training_allowed') is not False
                or run.get('execution_allowed') is not True
                or Path(base['runtime']['repo']).resolve() != ROOT):
            raise ValueError('read-only diagnostic authority/runtime differs')
        subprocess.run([sys.executable, str(ROOT / 'model_training/tools/agent_work_guard.py'),
                        'check', '--operation', 'preflight', '--entrypoint', ENTRYPOINT], check=True,
                       stdout=subprocess.DEVNULL)
        if subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
            raise ValueError('diagnostic requires clean published source')
        subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', operation['code_commit'], 'HEAD'], check=True)
        for relative, digest in config['diagnostic_source_sha256'].items():
            if sha256(ROOT / relative) != digest:
                raise ValueError(f'diagnostic source changed: {relative}')
        source = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        if (world != base['runtime']['world_size'] or
                os.environ.get('CUDA_VISIBLE_DEVICES') != ','.join(map(str, config['runtime']['physical_gpus']))
                or not torch.cuda.is_available() or torch.cuda.device_count() != world):
            raise ValueError('CUDA visibility differs')
        device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(config['allocated_fraction_limit'], device)
        if world > 1:
            dist.init_process_group('nccl', timeout=timedelta(minutes=30))
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=False)
        if world > 1:
            dist.barrier()
        claimed = True
        if read_reviewed(base['safety']['preflight'], name='model preflight')['source_sha256'] != model_source_hashes():
            raise ValueError('model implementation changed since accepted preflight')
        if read_reviewed(base['safety']['data_audit'], name='data audit')['target_source_sha256'] != data_source_hashes():
            raise ValueError('data implementation changed')
        initial_audit = read_reviewed(base['safety']['initialization'], name='initialization audit')
        validate_initialization_audit(base, initial_audit)
        reset_random(base['initialization']['seed'])
        model, tokenizer, codec, initial = build_fill_model(base, device)
        if initial['full_initial_state_sha256'] != initial_audit['initialization']['full_initial_state_sha256']:
            raise ValueError('official-only initialization changed')
        spec = config['checkpoint']
        if sha256(spec['path']) != spec['sha256']:
            raise ValueError('own80k checkpoint changed')
        saved = torch.load(spec['path'], map_location='cpu', mmap=True, weights_only=False)
        if (saved['schema'] != FAMILY or saved['mode'] != 'train' or saved['config'] != base
                or saved['architecture'] != model.configuration() or saved['training_contract'] != training_contract(base)
                or saved['asset_exposures'] != spec['asset_exposures'] or saved['step'] != spec['step']
                or saved['source'] != config['base_source_commit']):
            raise ValueError('checkpoint contract differs')
        model.load_state_dict(saved['model'], strict=True)
        del saved
        run_probe(model, tokenizer, codec, base, config, args, rank, world, device)
        if world > 1:
            dist.barrier()
        if rank == 0:
            reports = [json.loads((args.output / f'rank_{r}.json').read_text()) for r in range(world)]
            rows = [row for report in reports for row in report['rows']]
            if len(rows) != config['train_assets'] or len({row['asset_id'] for row in rows}) != len(rows):
                raise ValueError('diagnostic train cases missing or duplicated')
            result = {'complete': True, 'passed': True, 'quality_accepted': False,
                      'code_commit': source, 'checkpoint': spec, 'config_sha256': sha256(args.config),
                      'optimizer_created': False, 'optimizer_updates': 0,
                      'model_sha256_unchanged': spec['model_sha256'], 'groups': summarize(rows),
                      'interpretation': 'Per-asset raw task gradient norms in train mode, not Adam-preconditioned updates. '
                      'Fixed training cases only; no validation-based weight fitting. No optimizer or weight update.',
                      'finished_utc': datetime.now(timezone.utc).isoformat()}
            write_json(args.output / 'result.json', result)
            print('RESULT ' + json.dumps(result['groups']), flush=True)
        if world > 1:
            dist.barrier()
    except Exception as exc:
        if claimed:
            write_json(args.output / f'failure_rank_{rank}.json', {'complete': False, 'passed': False,
                       'rank': rank, 'error': repr(exc), 'traceback': traceback.format_exc()})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
