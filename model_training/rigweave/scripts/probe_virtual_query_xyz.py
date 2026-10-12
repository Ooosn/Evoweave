#!/usr/bin/env python3
"""Bounded full-size preflight/fitting; never starts a following stage itself."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
sys.path.insert(0, str(ROOT/'model_training/rigweave/src'))
from virtual_query_xyz_runtime import build_model, read_entries, fixed_dataset, prepare_batch, sha256, write_json


def verify_operation(args):
    state = json.loads((ROOT/'model_training/state/current.json').read_text())
    operation = state['active_operation']
    if state.get('architecture_contract', {}).get('id') != 'virtual_query_xyz_aux_ids_v1':
        raise ValueError('active state is not this model route')
    if operation['mode'] != args.mode or Path(operation['config']).resolve() != args.config.resolve():
        raise ValueError('CLI does not match the exact recorded stage/config')
    if operation['config_sha256'] != sha256(args.config):
        raise ValueError('config differs from the published operation')
    if Path(operation['output']).resolve() != args.output.resolve():
        raise ValueError('output differs from the published operation')
    if not operation['gpu_work_allowed'] or operation['formal_training_allowed']:
        raise ValueError('this bounded probe cannot execute under a formal-training operation')
    subprocess.run([sys.executable, str(ROOT/'model_training/tools/agent_work_guard.py'),
                    'check', '--operation', 'train'], check=True)
    dirty = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip()
    if dirty:
        raise ValueError('GPU probe requires a clean committed code checkout')
    return subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()


def choose_rows(rows, config):
    rng = np.random.default_rng(config['probe']['selection_seed'])
    def count(row):
        return row.get('canonical_metrics', {}).get('target_joint_count',
            row.get('rootless_target_strict_metrics', {}).get('target_joint_count', 0))
    pool = [r for r in rows if 20 <= count(r) <= 60 and r.get('integration_source') == 'base']
    if not pool:
        raise ValueError('fixed fit cohort is empty')
    selected = [pool[int(i)] for i in rng.permutation(len(pool))[:config['probe']['assets']]]
    if len(selected) != config['probe']['assets']:
        raise ValueError('not enough fixed assets')
    return selected


def grad_norms(model):
    result = {}
    for name in ('surface_tokenizer', 'motion_encoder', 'frame_fusion', 'xyz_decoder', 'auxiliary'):
        gradients = [p.grad.detach().float() for p in getattr(model, name).parameters() if p.grad is not None]
        if any(not bool(torch.isfinite(g).all()) for g in gradients):
            raise RuntimeError(f'nonfinite gradients in {name}')
        result[name] = sum(float(g.square().sum()) for g in gradients) ** .5
    return result


@torch.no_grad()
def evaluate(model, items, tokenizer, config, output, step):
    from src.model.unirig_ar import VocabSwitchingLogitsProcessor
    from eval_dynamic_rig_generation import _topology_metrics
    model.eval()
    records = []
    for asset_index, (surface, targets, geometry) in enumerate(items):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            real, virtual = model.encode(surface)
            conditions = model.conditions(real, virtual)
            teacher = model.forward_encoded(real, virtual, targets, auxiliary_weight=config['probe']['auxiliary_weight'])
        for frame in range(conditions.shape[1]):
            start = torch.tensor([tokenizer.bos, tokenizer.cls_name_to_token('articulationxl')], device=conditions.device)
            processor = VocabSwitchingLogitsProcessor(tokenizer, start)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                predicted = model.generate_condition(conditions[:, frame], start, eos_token=tokenizer.eos,
                    max_new_tokens=config['probe']['max_new_tokens'], logits_processor=processor)
            ids = targets['main_input_ids'][0, frame][targets['main_attention_mask'][0, frame]].cpu().numpy()
            truth = tokenizer.detokenize(ids.astype(np.int64))
            record = {'asset_index': asset_index, 'asset_id': geometry['asset_id'][0], 'frame_index': frame,
                      'raw_frame_index': int(geometry['selected_frames'][0, frame]), 'status': predicted['status'],
                      'success': False, 'quantized_edge_f1': 0., 'original_edge_f1': 0., 'exact_count': False,
                      'main_ce': float(teacher['main_loss']), 'main_accuracy': float(teacher['main_accuracy']),
                      'generated_tokens': predicted['tokens'], 'target_tokens': ids.tolist(),
                      'target_quantized_joints': truth.joints.tolist(), 'target_quantized_parents': truth.parents}
            size = int(geometry['joint_count'][0])
            original = geometry['frame_target_joints'][0, frame, :size].cpu().numpy()
            original_parents = geometry['target_parents'][0, :size].cpu().tolist()
            record['quantization_label_ceiling_f1'] = _topology_metrics(truth.joints, truth.parents,
                original, original_parents)['edge_f1']
            if predicted['success']:
                try:
                    parsed = tokenizer.detokenize(np.asarray(predicted['tokens'], dtype=np.int64))
                    if parsed.joints.shape[0] < 1 or not np.isfinite(parsed.joints).all():
                        raise ValueError('nonfinite or empty skeleton')
                    record.update(success=True, predicted_joints=parsed.joints.tolist(), predicted_parents=parsed.parents,
                        quantized_edge_f1=_topology_metrics(parsed.joints, parsed.parents, truth.joints, truth.parents)['edge_f1'],
                        original_edge_f1=_topology_metrics(parsed.joints, parsed.parents, original, original_parents)['edge_f1'],
                        exact_count=len(parsed.joints)==len(truth.joints))
                except Exception as exc:
                    record.update(status='invalid_eos_sequence', error=repr(exc))
            records.append(record)
    summary = {key: float(np.mean([r[key] for r in records])) for key in
               ('success', 'quantized_edge_f1', 'original_edge_f1', 'exact_count', 'main_ce', 'main_accuracy')}
    fit_acceptance = (summary['main_accuracy'] >= .99 and all(r['success'] for r in records)
        and np.mean([r['exact_count'] and r['quantized_edge_f1'] >= .95 for r in records]) >= .9)
    report = {'step': step, 'views': len(records), 'summary': summary, 'fit_acceptance': bool(fit_acceptance),
              'rows': records, 'failed_generations_scored_zero': True}
    write_json(output/f'evaluation_{step:04d}.json', report)
    print('EVALUATION '+json.dumps({k:v for k,v in report.items() if k!='rows'}), flush=True)
    model.train()
    return report


def run(args, config, source):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(config['runtime']['physical_gpu']):
        raise ValueError('visible physical GPU differs from the exact recorded probe resource')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('bounded probe needs exactly one granted CUDA device')
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.95, device)
    rows = choose_rows(read_entries(config)['train'], config)
    write_json(args.output/'selected_assets.json', [{'asset_id': r['asset_id'], 'path': r['path']} for r in rows])
    model, tokenizer, codec, initial = build_model(config, device)
    initial.update(code_commit=source, config_sha256=sha256(args.config), device=torch.cuda.get_device_name(),
                   torch_version=torch.__version__)
    write_json(args.output/'initialization.json', initial)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=config['optimization']['lr'], betas=(.9, .999), weight_decay=config['optimization']['weight_decay'])
    trace_path = args.output/'trace.jsonl'
    if args.mode == 'preflight':
        reports = []
        for frames in config['probe']['preflight_frames']:
            dataset = fixed_dataset(rows[:1], config, frames)
            item = prepare_batch(dataset[0], config, tokenizer, codec, device)
            surface, target, geometry = item
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(surface, target, auxiliary_weight=.1)
            if not bool(torch.isfinite(out['loss'])):
                raise RuntimeError('nonfinite preflight loss')
            if frames == config['probe']['preflight_frames'][0]:
                gradients = {}
                for key in ('main_loss', 'auxiliary_loss'):
                    g = torch.autograd.grad(out[key], model.motion_encoder.virtual_slots, retain_graph=True)[0]
                    gradients[key] = float(g.float().norm())
                    if not bool(torch.isfinite(g).all()) or gradients[key] <= 0:
                        raise RuntimeError('missing/nonfinite virtual-slot learning signal')
                write_json(args.output/'shared_gradient_check.json', gradients)
            out['loss'].backward()
            norms = grad_norms(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            torch.cuda.synchronize()
            report = {'frames': frames, 'batch_size': 1, 'seconds': time.monotonic()-started,
                      'main_loss': float(out['main_loss']), 'auxiliary_loss': float(out['auxiliary_loss']),
                      'gradient_norms': norms, 'max_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
                      'max_reserved_gib': torch.cuda.max_memory_reserved()/2**30,
                      'selected_frames': geometry['selected_frames'].cpu().tolist()}
            reports.append(report)
            print('PREFLIGHT '+json.dumps(report), flush=True)
            write_json(args.output/'preflight.json', {'complete': False, 'cases': reports})
            del item, surface, target, geometry, out
        write_json(args.output/'preflight.json', {'complete': True, 'execution_accepted': True,
                   'model_fit_accepted': False, 'cases': reports})
        return {'stage': 'preflight', 'accepted': True, 'formal_training_allowed': False}
    dataset = fixed_dataset(rows, config, config['probe']['frames'])
    items = [prepare_batch(dataset[i], config, tokenizer, codec, device) for i in range(len(dataset))]
    write_json(args.output/'fixed_frames.json', [{'asset_id': x[2]['asset_id'][0],
        'selected_frames': x[2]['selected_frames'].cpu().tolist(),
        'heldout_raw_ids': x[2]['heldout_raw_ids']} for x in items])
    last = evaluate(model, items, tokenizer, config, args.output, 0)
    started = time.monotonic()
    for step in range(1, config['probe']['max_updates']+1):
        surface, target, _ = items[(step-1) % len(items)]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(surface, target, auxiliary_weight=config['probe']['auxiliary_weight'])
        if not bool(torch.isfinite(out['loss'])):
            raise RuntimeError('nonfinite fitting loss; further training stopped')
        out['loss'].backward()
        norms = grad_norms(model)
        clipped = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
        optimizer.step()
        torch.cuda.synchronize()
        record = {'step': step, 'asset_exposures': step, 'supervised_frame_exposures': step*config['probe']['frames'],
            'loss': float(out['loss']), 'main_loss': float(out['main_loss']), 'auxiliary_loss': float(out['auxiliary_loss']),
            'main_accuracy': float(out['main_accuracy']), 'gradient_norm': clipped, 'groups': norms,
            'elapsed_seconds': time.monotonic()-started, 'peak_gib': torch.cuda.max_memory_allocated()/2**30}
        with trace_path.open('a') as f:
            f.write(json.dumps(record)+'\n')
        if step % 8 == 0:
            print('FIT '+json.dumps(record), flush=True)
        if step in config['probe']['evaluation_updates'] or step == config['probe']['max_updates']:
            last = evaluate(model, items, tokenizer, config, args.output, step)
            write_json(args.output/'progress.json', {'step': step, 'evaluation': last['summary'],
                       'fit_acceptance': last['fit_acceptance'], 'formal_training_allowed': False})
            if last['fit_acceptance']:
                break
    torch.save({'schema': 'virtual_query_xyz_aux_ids_v1', 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'config': config, 'step': step, 'initialization': initial, 'source': source,
        'diagnostic_only_not_formal_initialization': True}, args.output/'checkpoint_fit.pt')
    return {'stage': 'fit', 'completed_updates': step, 'accepted': last['fit_acceptance'],
            'evaluation': last['summary'], 'formal_training_allowed': False,
            'next_action': 'primary_review_required_no_automatic_continuation'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['preflight', 'fit'], required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    source = verify_operation(args)
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result = run(args, config, source)
        write_json(args.output/'result.json', {**result, 'execution_complete': True,
            'finished_utc': datetime.now(timezone.utc).isoformat(), 'code_commit': source})
    except Exception:
        write_json(args.output/'result.json', {'accepted': False, 'execution_complete': False,
                   'formal_training_allowed': False, 'traceback': traceback.format_exc(), 'code_commit': source})
        raise


if __name__ == '__main__':
    main()
