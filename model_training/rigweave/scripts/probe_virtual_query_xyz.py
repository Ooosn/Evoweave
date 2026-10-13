#!/usr/bin/env python3
"""Bounded full-size preflight/fitting; never starts a following stage itself."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
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
from fit_cohort_selection import choose_fit_rows


def declared_probe_run(operation, args, config_digest):
    matches = [run for run in operation.get('runs', [operation])
               if Path(run['config']).resolve() == args.config.resolve()]
    if len(matches) != 1:
        raise ValueError('probe configuration is not uniquely declared')
    run = matches[0]
    if operation['mode'] != args.mode or run['mode'] != args.mode:
        raise ValueError('CLI does not match the recorded stage')
    if run['config_sha256'] != config_digest:
        raise ValueError('config differs from the published operation')
    if Path(run['output']).resolve() != args.output.resolve():
        raise ValueError('output differs from the published operation')
    return run


def verify_operation(args):
    state = json.loads((ROOT/'model_training/state/current.json').read_text())
    operation = state['active_operation']
    if state.get('architecture_contract', {}).get('id') != 'virtual_query_xyz_aux_ids_v1':
        raise ValueError('active state is not this model route')
    if not operation.get('gpu_work_allowed') or operation.get('formal_training_allowed'):
        raise ValueError('no bounded GPU operation has been published for this route')
    declared_probe_run(operation, args, sha256(args.config))
    subprocess.run([sys.executable, str(ROOT/'model_training/tools/agent_work_guard.py'),
                    'check', '--operation', 'train', '--entrypoint', str(Path(__file__).resolve())], check=True)
    dirty = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip()
    if dirty:
        raise ValueError('GPU probe requires a clean committed code checkout')
    expected = operation['code_commit']
    subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', expected, 'HEAD'], check=True)
    subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet', expected, 'HEAD', '--',
                    'model_training/rigweave', 'model_training/tools'], check=True)
    return subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()


def choose_rows(rows, config):
    """Historical single-case selector, retained for exact diagnostic replay."""
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
    for name in ('surface_tokenizer', 'motion_encoder', 'frame_fusion', 'condition_norm', 'xyz_decoder', 'auxiliary'):
        gradients = [p.grad.detach().float() for p in getattr(model, name).parameters() if p.grad is not None]
        if any(not bool(torch.isfinite(g).all()) for g in gradients):
            raise RuntimeError(f'nonfinite gradients in {name}')
        result[name] = sum(float(g.square().sum()) for g in gradients) ** .5
    return result


def require_gradients(model, norms, auxiliary_weight):
    required = set(norms) - ({'auxiliary'} if auxiliary_weight == 0 else set())
    if any(norms[name] <= 0 for name in required):
        raise RuntimeError(f'disconnected required parameter group: {norms}')
    for name, parameter in [('virtual_slots', model.motion_encoder.virtual_slots),
                            ('xyz_embedding', model.xyz_decoder.get_input_embeddings().weight),
                            ('xyz_output_head', model.xyz_decoder.get_output_embeddings().weight)]:
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()) or float(parameter.grad.float().norm()) <= 0:
            raise RuntimeError(f'missing/nonfinite learning signal for {name}')


def fitting_plan(config, asset_count):
    plan = config['probe']
    updates = plan['max_updates']
    batch = plan.get('assets_per_update', 1)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
           for value in (updates, batch, asset_count)) or batch > asset_count:
        raise ValueError('fit update count and assets per update must be positive, without duplicate assets')
    optimization = config['optimization']
    schedule = optimization.get('schedule', 'constant')
    if schedule not in ('constant', 'onecycle'):
        raise ValueError('unsupported fitting schedule')
    exposures = updates * batch
    if schedule == 'onecycle':
        if optimization['schedule_samples'] != exposures:
            raise ValueError('OneCycle schedule must match actual declared asset exposures')
        for key in ('peak_lr', 'div_factor', 'final_div_factor'):
            if not math.isfinite(optimization[key]) or optimization[key] <= 0:
                raise ValueError('invalid positive fitting schedule parameter')
        if not 0 < optimization['warmup_fraction'] < 1:
            raise ValueError('invalid fitting warmup fraction')
    elif not math.isfinite(optimization['lr']) or optimization['lr'] <= 0:
        raise ValueError('invalid constant fitting learning rate')
    if not math.isfinite(optimization['weight_decay']) or optimization['weight_decay'] < 0:
        raise ValueError('invalid fitting weight decay')
    return {'schedule': schedule, 'assets_per_update': batch, 'max_updates': updates,
            'max_asset_exposures': exposures, 'order': 'fixed_cyclic', 'gradient_reduction': 'asset_mean'}


def fitting_update(model, optimizer, items, config, plan, step):
    batch = plan['assets_per_update']
    seen = (step - 1) * batch
    if plan['schedule'] == 'onecycle':
        from virtual_frame_runtime import set_exposure_lr
        lr, beta1 = set_exposure_lr(optimizer, seen, config)
    else:
        lr, beta1 = optimizer.param_groups[0]['lr'], optimizer.param_groups[0]['betas'][0]
    optimizer.zero_grad(set_to_none=True)
    keys = ('loss', 'main_loss', 'auxiliary_loss', 'main_accuracy')
    metrics = {key: 0. for key in keys}
    indices = [(seen + offset) % len(items) for offset in range(batch)]
    shared_signal = None
    for index in indices:
        surface, target, _ = items[index]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(surface, target, auxiliary_weight=config['probe']['auxiliary_weight'])
        if not bool(torch.isfinite(out['loss'])):
            raise RuntimeError('nonfinite fitting loss; further training stopped')
        if step == 1 and shared_signal is None and config['probe'].get('verify_shared_gradients', False):
            shared_signal = {}
            for name in ('main_loss', 'auxiliary_loss'):
                gradient = torch.autograd.grad(out[name], model.motion_encoder.virtual_slots, retain_graph=True)[0]
                norm = float(gradient.float().norm())
                if not bool(torch.isfinite(gradient).all()) or norm <= 0:
                    raise RuntimeError(f'{name} has no finite learning signal into virtual slots')
                shared_signal[name] = norm
        # Every item is one asset; retain the established per-frame/token mean.
        (out['loss'] / batch).backward()
        for key in keys:
            metrics[key] += float(out[key].detach()) / batch
        del out
    norms = grad_norms(model)
    require_gradients(model, norms, config['probe']['auxiliary_weight'])
    clipped = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
    optimizer.step()
    return {**metrics, 'gradient_norm': clipped, 'groups': norms, 'lr': lr, 'beta1': beta1,
            'first_update_separate_virtual_gradients': shared_signal,
            'asset_indices': indices, 'asset_exposures': seen + batch,
            'supervised_frame_exposures': (seen + batch) * config['probe']['frames']}


def fit_view_passes(row):
    # Nearest-neighbor edge F1 is not reflexive at coincident joint positions.
    # Exact canonical target tokens are stronger evidence than that metric.
    exact = bool(row.get('target_tokens')) and row.get('generated_tokens') == row['target_tokens']
    return (row['success'] and row['exact_count']
            and row['quantized_j2j'] is not None and row['quantized_j2j'] <= 4 / 256
            and (row['quantized_edge_f1'] >= .95 or exact))


@torch.no_grad()
def evaluate(model, items, tokenizer, config, output, step, *, frame_indices=None):
    from rigweave.dynamic_rig.virtual_query_xyz import original_xyz_vocabulary_processor
    from eval_dynamic_rig_generation import _topology_metrics, _parents_from_output
    model.eval()
    records = []
    auxiliary_rows = []
    check_auxiliary = config['probe'].get('evaluate_auxiliary', False)
    if check_auxiliary:
        from rigweave.dynamic_rig.aux_skeleton_tokens import AuxSkeletonTokens
        from fusion_evaluation import auxiliary_record, auxiliary_summary
        codec = AuxSkeletonTokens(**config['auxiliary_codec'])
    for asset_index, (surface, targets, geometry) in enumerate(items):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            real, virtual = model.encode(surface)
            position_args = ({'query_points': surface['query_points']} if
                getattr(getattr(model, 'config', None), 'fusion_mode', None) == 'relative_position' else {})
            conditions = model.conditions(real, virtual, **position_args)
            teacher = model.forward_encoded(real, virtual, targets,
                auxiliary_weight=config['probe']['auxiliary_weight'], **position_args)
            if check_auxiliary:
                prediction = model.generate_auxiliary(virtual, bos_token=codec.bos,
                    eos_token=codec.eos, max_new_tokens=codec.max_tokens-1)
                size = int(geometry['joint_count'][0])
                auxiliary_rows.append(auxiliary_record(prediction, codec,
                    geometry['target_parents'][0, :size].cpu().tolist(), asset_id=geometry['asset_id'][0]))
        selected = list(range(conditions.shape[1])) if frame_indices is None else list(frame_indices)
        if not selected or len(set(selected)) != len(selected) or any(
                isinstance(frame, bool) or not isinstance(frame, int) or frame < 0 or frame >= conditions.shape[1]
                for frame in selected):
            raise ValueError('generation frame selection is empty, duplicated or out of range')
        for frame in selected:
            start = torch.tensor([tokenizer.bos, tokenizer.cls_name_to_token('articulationxl')], device=conditions.device)
            processor = original_xyz_vocabulary_processor(tokenizer, start)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                predicted = model.generate_condition(conditions[:, frame], start, eos_token=tokenizer.eos,
                    max_new_tokens=config['probe']['max_new_tokens'], logits_processor=processor)
            ids = targets['main_input_ids'][0, frame][targets['main_attention_mask'][0, frame]].cpu().numpy()
            truth = tokenizer.detokenize(ids.astype(np.int64))
            truth_parents = _parents_from_output(truth)
            record = {'asset_index': asset_index, 'asset_id': geometry['asset_id'][0], 'frame_index': frame,
                      'raw_frame_index': int(geometry['selected_frames'][0, frame]), 'status': predicted['status'],
                      'success': False, 'quantized_edge_f1': 0., 'original_edge_f1': 0., 'exact_count': False,
                      'main_ce': float(teacher['main_loss_per_frame'][0, frame]),
                      'main_accuracy': float(teacher['main_accuracy_per_frame'][0, frame]),
                      'generated_tokens': predicted['tokens'], 'target_tokens': ids.tolist(),
                      'exact_target_tokens': predicted['tokens'] == ids.tolist(),
                      'target_quantized_joints': truth.joints.tolist(), 'target_quantized_parents': truth_parents,
                      'quantized_j2j': None}
            record['target_self_edge_f1'] = _topology_metrics(truth.joints, truth_parents,
                truth.joints, truth_parents)['edge_f1']
            size = int(geometry['joint_count'][0])
            original = geometry['frame_target_joints'][0, frame, :size].cpu().numpy()
            original_parents = geometry['target_parents'][0, :size].cpu().tolist()
            record['quantization_label_ceiling_f1'] = _topology_metrics(truth.joints, truth_parents,
                original, original_parents)['edge_f1']
            if predicted['success']:
                try:
                    parsed = tokenizer.detokenize(np.asarray(predicted['tokens'], dtype=np.int64))
                    if parsed.joints.shape[0] < 1 or not np.isfinite(parsed.joints).all():
                        raise ValueError('nonfinite or empty skeleton')
                    parsed_parents = _parents_from_output(parsed)
                    distances = np.linalg.norm(parsed.joints[:, None]-truth.joints[None], axis=-1)
                    record.update(success=True, predicted_joints=parsed.joints.tolist(), predicted_parents=parsed_parents,
                        quantized_j2j=float(.5*(distances.min(0).mean()+distances.min(1).mean())),
                        quantized_edge_f1=_topology_metrics(parsed.joints, parsed_parents, truth.joints, truth_parents)['edge_f1'],
                        original_edge_f1=_topology_metrics(parsed.joints, parsed_parents, original, original_parents)['edge_f1'],
                        exact_count=len(parsed.joints)==len(truth.joints))
                except Exception as exc:
                    record.update(status='invalid_eos_sequence', error=repr(exc))
            records.append(record)
    summary = {key: float(np.mean([r[key] for r in records])) for key in
               ('success', 'quantized_edge_f1', 'original_edge_f1', 'exact_count', 'main_ce', 'main_accuracy')}
    fit_acceptance = (summary['main_accuracy'] >= .99 and all(r['success'] for r in records)
        and np.mean([fit_view_passes(r) for r in records]) >= .9)
    distances = [r['quantized_j2j'] for r in records if r['quantized_j2j'] is not None]
    summary['quantized_j2j_successful_views'] = float(np.mean(distances)) if distances else None
    summary['geometry_scored_views'] = len(distances)
    report = {'step': step, 'views': len(records), 'summary': summary, 'fit_acceptance': bool(fit_acceptance),
              'fit_gate': 'exact_tokens_or_raw_f1_v2', 'passing_fit_views': sum(fit_view_passes(r) for r in records),
              'rows': records, 'failed_generations_scored_zero': True, 'teacher_metrics_scope': 'per_frame'}
    if check_auxiliary:
        report['main_fit_acceptance'] = report['fit_acceptance']
        report['auxiliary_summary'] = auxiliary_summary(auxiliary_rows)
        report['auxiliary_rows'] = auxiliary_rows
        report['auxiliary_fit_acceptance'] = all(r['success'] and r['exact_parents'] for r in auxiliary_rows)
        report['fit_acceptance'] = report['main_fit_acceptance'] and report['auxiliary_fit_acceptance']
    write_json(output/f'evaluation_{step:04d}.json', report)
    print('EVALUATION '+json.dumps({k:v for k,v in report.items() if k not in ('rows', 'auxiliary_rows')}), flush=True)
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
    rows, selection = choose_fit_rows(read_entries(config)['train'], config, return_report=True)
    selection.update(manifest=config['data']['manifests']['train'], config_sha256=sha256(args.config))
    write_json(args.output/'cohort_selection.json', selection)
    write_json(args.output/'selected_assets.json', [{'asset_id': r['asset_id'], 'path': r['path']} for r in rows])
    model, tokenizer, codec, initial = build_model(config, device)
    if config['probe'].get('fingerprint_initial_state', False):
        from probe_virtual_query_lambda import parameter_fingerprint
        initial['full_initial_state_sha256'] = parameter_fingerprint(model)
    initial.update(code_commit=source, config_sha256=sha256(args.config), device=torch.cuda.get_device_name(),
                   torch_version=torch.__version__)
    write_json(args.output/'initialization.json', initial)
    plan = fitting_plan(config, len(rows)) if args.mode == 'fit' else None
    if plan is not None:
        write_json(args.output/'fitting_protocol.json', plan)
    scheduled = plan is not None and plan['schedule'] == 'onecycle'
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=(config['optimization']['peak_lr'] / config['optimization']['div_factor']
            if scheduled else config['optimization']['lr']),
        betas=(.95 if scheduled else .9, .999), weight_decay=config['optimization']['weight_decay'])
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
            require_gradients(model, norms, .1)
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
    if any(len(item[2]['asset_id']) != 1 for item in items):
        raise ValueError('fixed fitting items must each contain exactly one asset')
    if any(int(x[1]['main_attention_mask'].sum(-1).max()) > config['probe']['max_new_tokens'] + 2 for x in items):
        raise ValueError('fixed target exceeds declared free-generation budget; review config before fitting')
    write_json(args.output/'fixed_frames.json', [{'asset_id': x[2]['asset_id'][0],
        'selected_frames': x[2]['selected_frames'].cpu().tolist(),
        'heldout_raw_ids': x[2]['heldout_raw_ids'],
        'main_token_lengths': x[1]['main_attention_mask'].sum(-1).cpu().tolist()} for x in items])
    if config['probe'].get('paired_output'):
        from fusion_evaluation import pair_ready
        pair_ready(args.output, config['probe']['paired_output'], initial, items, config)
    last = evaluate(model, items, tokenizer, config, args.output, 0)
    first_accepted_step = None
    started = time.monotonic()
    for step in range(1, config['probe']['max_updates']+1):
        metrics = fitting_update(model, optimizer, items, config, plan, step)
        torch.cuda.synchronize()
        record = {'step': step, **metrics,
            'elapsed_seconds': time.monotonic()-started, 'peak_gib': torch.cuda.max_memory_allocated()/2**30}
        with trace_path.open('a') as f:
            f.write(json.dumps(record)+'\n')
        if step % 8 == 0:
            print('FIT '+json.dumps(record), flush=True)
        if step in config['probe']['evaluation_updates'] or step == config['probe']['max_updates']:
            last = evaluate(model, items, tokenizer, config, args.output, step)
            write_json(args.output/'progress.json', {'step': step, 'evaluation': last['summary'],
                       'auxiliary_summary': last.get('auxiliary_summary'),
                       'fit_acceptance': last['fit_acceptance'], 'formal_training_allowed': False})
            if last['fit_acceptance']:
                if first_accepted_step is None:
                    first_accepted_step = step
                    torch.save({'schema': 'virtual_query_xyz_aux_ids_v1', 'model': model.state_dict(),
                        'config': config, 'step': step, 'initialization': initial, 'source': source,
                        'diagnostic_only_not_formal_initialization': True}, args.output/'checkpoint_first_pass.pt')
                if config['probe'].get('early_stop', True):
                    break
    torch.save({'schema': 'virtual_query_xyz_aux_ids_v1', 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'config': config, 'step': step, 'initialization': initial, 'source': source,
        'diagnostic_only_not_formal_initialization': True}, args.output/'checkpoint_fit.pt')
    return {'stage': 'fit', 'completed_updates': step, 'accepted': first_accepted_step is not None,
            'first_accepted_step': first_accepted_step, 'final_fit_acceptance': last['fit_acceptance'],
            'asset_exposures': step * plan['assets_per_update'], 'fitting_protocol': plan,
            'evaluation': last['summary'], 'auxiliary_summary': last.get('auxiliary_summary'),
            'formal_training_allowed': False,
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
