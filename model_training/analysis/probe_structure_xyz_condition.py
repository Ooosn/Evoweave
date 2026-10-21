#!/usr/bin/env python3
"""Read-only paired condition audit of an already trained XYZ-fill checkpoint."""

from __future__ import annotations

import argparse
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

from structure_condition_metrics import (
    build_prefix_layout, choose_partner_indices, contrast_logits, summarize_logits,
)
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

ENTRYPOINT = 'model_training/analysis/probe_structure_xyz_condition.py'


def checked_json(spec):
    if sha256(spec['path']) != spec['sha256']:
        raise ValueError(f"reviewed input changed: {spec['path']}")
    return json.loads(Path(spec['path']).read_text())


def verify_operation(args, config, base):
    if config.get('purpose') != 'structure_xyz_condition_audit_v1':
        raise ValueError('configuration is not the declared read-only structure diagnostic')
    state = json.loads((ROOT / 'model_training/state/current.json').read_text())
    operation = state['active_operation']
    run = published_run(operation, args.config, args.output, args.mode, sha256(args.config))
    if (state['architecture_contract']['id'] != FAMILY
            or operation.get('gpu_work_allowed') is not True
            or run.get('execution_allowed') is not True
            or operation.get('formal_training_allowed') is not False
            or config['runtime'] != base['runtime']
            or run['physical_gpus'] != config['runtime']['physical_gpus']):
        raise ValueError('audit authority, resource identity or base runtime differs')
    subprocess.run([sys.executable, str(ROOT / 'model_training/tools/agent_work_guard.py'),
                    'check', '--operation', 'preflight', '--entrypoint', ENTRYPOINT],
                   check=True, stdout=subprocess.DEVNULL)
    if Path(base['runtime']['repo']).resolve() != ROOT:
        raise ValueError('audit is not running in the declared checkout')
    if subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
        raise ValueError('audit requires clean published source')
    subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor',
                    operation['code_commit'], 'HEAD'], check=True)
    subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet', operation['code_commit'],
                    'HEAD', '--', 'model_training/rigweave', 'model_training/tools'], check=True)
    for relative, digest in config['diagnostic_source_sha256'].items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f'audit implementation changed: {relative}')
    return subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()


def mean_or_none(values):
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values)) if values else None


def summarize_cases(rows):
    result = {}
    for group in ('train_seen', 'train_heldout', 'valid'):
        grouped = [row for row in rows if row['group'] == group]
        result[group] = {}
        for subset in ('all', 'joints52', 'non52'):
            selected = [row for row in grouped if subset == 'all' or
                        ((row['target_joint_count'] == 52) == (subset == 'joints52'))]
            summary = {'cases': len(selected), 'prefixes': {}}
            for prefix in ('teacher', 'self'):
                modes = {}
                for mode in ('correct', 'different_asset', 'zero'):
                    categories = {}
                    for category in ('all', 'choice', 'new_joint_id', 'branch', 'parent_reference', 'eos'):
                        values = [row[prefix]['modes'][mode][category] if category in ('all', 'choice')
                                  else row[prefix]['modes'][mode]['by_category'][category]
                                  for row in selected]
                        categories[category] = {
                            'contributing_cases': sum(value['tokens'] > 0 for value in values),
                            'total_tokens': sum(value['tokens'] for value in values),
                            **{name: mean_or_none([value[name] for value in values]) for name in
                               ('raw_ce', 'raw_accuracy', 'legal_ce', 'legal_accuracy')},
                        }
                    modes[mode] = categories
                contrasts = {}
                for mode in ('different_asset', 'zero'):
                    values = [row[prefix]['contrasts'][mode]['choice'] for row in selected]
                    deltas = [row[prefix]['modes'][mode]['choice']['legal_ce'] -
                              row[prefix]['modes']['correct']['choice']['legal_ce'] for row in selected]
                    contrasts[mode] = {
                        'choice_legal_ce_counter_minus_correct': mean_or_none(deltas),
                        'median_choice_legal_ce_counter_minus_correct':
                            float(np.median(deltas)) if deltas else None,
                        'correct_lower_ce_cases': sum(value > 1e-6 for value in deltas),
                        'counter_lower_ce_cases': sum(value < -1e-6 for value in deltas),
                        **{name: mean_or_none([value[name] for value in values]) for name in
                           ('raw_js', 'legal_js', 'raw_argmax_change_fraction', 'legal_argmax_change_fraction')},
                    }
                summary['prefixes'][prefix] = {'modes': modes, 'contrasts': contrasts}
            result[group][subset] = summary
    return result


def infer_logits(model, memory, tokens, device):
    inputs = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model.structure_decoder(memory.to(device), inputs, torch.ones_like(inputs, dtype=torch.bool))
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError('nonfinite structure logits')
    return logits[0].float().cpu().numpy()


def mismatch_position(left, right):
    for index, (a, b) in enumerate(zip(left[1:], right[1:])):
        if a != b:
            return index
    return None if left == right else min(len(left), len(right)) - 1


def contrast_segments(contrast, mismatch):
    positions = contrast['positions']
    output = {}
    for name, start, stop in (
        ('before_first_mismatch', 0, len(positions) if mismatch is None else mismatch),
        ('mismatch_through_next7', mismatch, None if mismatch is None else mismatch + 8),
        ('next8_through31', None if mismatch is None else mismatch + 8,
         None if mismatch is None else mismatch + 32),
        ('after32', None if mismatch is None else mismatch + 32, len(positions)),
    ):
        selected = [] if start is None else positions[start:stop]
        output[name] = {'positions': len(selected),
                        **{key: mean_or_none([row[key] for row in selected])
                           for key in ('raw_js', 'legal_js')}}
    return output


@torch.no_grad()
def run_audit(model, tokenizer, codec, base, config, args, rank, world, device):
    entries = read_entries(base)
    verify_data_selection(base, entries)
    panel, reference, manifest = load_small_reference_panel(base, entries)
    if manifest['panel_sha256'] != config['panel_sha256']:
        raise ValueError('diagnostic panel identity changed')
    baseline_summary = checked_json(config['reference_evaluation']['summary'])
    if (not baseline_summary['complete'] or baseline_summary['panel_sha256'] != manifest['panel_sha256']
            or baseline_summary['asset_exposures'] != config['checkpoint']['asset_exposures']):
        raise ValueError('baseline summary differs from the exact completed checkpoint/panel')
    previous = checked_json(config['reference_evaluation'][f'rank_{rank}'])
    if not previous['complete'] or previous['asset_exposures'] != config['checkpoint']['asset_exposures']:
        raise ValueError('baseline evaluation is incomplete or from a different exposure count')
    rows, arrays = [], {}
    model.eval()
    model.requires_grad_(False)
    before = parameter_fingerprint(model)
    if before != config['checkpoint']['model_sha256']:
        raise ValueError('loaded model tensors differ from the completed training result')
    torch.cuda.reset_peak_memory_stats(device)
    for group, panel_rows in panel.items():
        selected = panel_rows[rank::world]
        dataset = fixed_dataset(selected, base, base['evaluation']['input_frames'],
                                mode='heldout_eval' if group == 'train_heldout' else 'seen_eval')
        prior_structures = {row['asset_id']: row for row in previous['groups'][group]['structure_rows']}
        prior_views = {(row['asset_id'], row['frame_index']): row for row in previous['groups'][group]['rows']}
        cache = []
        for index, panel_row in enumerate(selected):
            surface, targets, geometry, _ = prepare_fill_batch(dataset, [index], base, codec, device)
            item = _prepare_item(surface, targets, geometry, codec, bins=base['quantization']['bins'])
            actual, reserved = set(item.raw_frame_indices.tolist()), set(geometry['heldout_raw_ids'][0])
            if ((group == 'train_heldout' and actual != reserved) or
                    (group != 'train_heldout' and actual & reserved) or item.asset_id != panel_row['asset_id']):
                raise ValueError('diagnostic changed the asset or heldout frame partition')
            _reference_rows(geometry, tokenizer, codec, group=group, asset_id=item.asset_id,
                            frame_indices=base['evaluation']['frame_indices'],
                            raw_frame_indices=item.raw_frame_indices, reference_lookup=reference)
            prior = prior_structures[item.asset_id]
            tokens = [codec.bos] + targets['structure_labels'][0].tolist()
            if tokens != item.target_tokens or tokens != prior['target_tokens']:
                raise ValueError('teacher targets differ from the original completed evaluation')
            anchor_differences = []
            for frame in base['evaluation']['frame_indices']:
                view = prior_views[(item.asset_id, frame)]
                if (int(item.raw_frame_indices[frame]) != view['raw_frame_index']
                        or item.geometry_parents.tolist() != view['original_target_parents']
                        or item.geometry_xyz[frame].tolist() != view['original_target_joints']):
                    raise ValueError('raw frames or original continuous GT changed')
                points = surface['query_points'][0, frame].float().cpu().numpy()
                anchor_differences.append(float(np.abs(points - np.asarray(view['input_anchor_points'])).max()))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                final_frames, virtual = model.encode(surface)
                if not bool(torch.isfinite(virtual).all()):
                    raise FloatingPointError('nonfinite virtual condition')
                generated = model.generate_structure(virtual)
            if (not generated['success'] or generated['tokens'] != prior['generated_tokens']
                    or generated['parents'] != prior['parents']):
                raise ValueError('correct-condition generation did not replay the completed baseline')
            cache.append({'asset_id': item.asset_id, 'virtual': virtual.cpu(), 'target_tokens': tokens,
                          'target_parents': prior['target_parents'], 'generation': generated,
                          'raw_frames': item.raw_frame_indices.tolist(),
                          'max_saved_anchor_absolute_difference': max(anchor_differences)})
            del final_frames, virtual, surface, targets, geometry
            peak = torch.cuda.max_memory_allocated(device) / torch.cuda.get_device_properties(device).total_memory
            if peak >= config['allocated_fraction_limit']:
                raise RuntimeError('diagnostic exceeded the published GPU memory limit')
            print(json.dumps({'rank': rank, 'event': 'encoded', 'group': group,
                              'asset_id': item.asset_id, 'baseline_replayed': True}), flush=True)
        partners = choose_partner_indices([row['target_parents'] for row in cache])
        for index, item in enumerate(cache):
            partner = cache[partners[index]]
            own, other = item['virtual'].float(), partner['virtual'].float()
            mismatch = mismatch_position(item['generation']['tokens'], item['target_tokens'])
            row = {key: value for key, value in item.items() if key != 'virtual'}
            row.update(group=group, rank=rank, target_joint_count=len(item['target_parents']),
                       partner_asset_id=partner['asset_id'], partner_target_parents=partner['target_parents'],
                       first_generated_target_mismatch_position=mismatch,
                       virtual_relative_l2=float((own - other).norm() / own.norm().clamp_min(1e-12)),
                       virtual_cosine=float(torch.nn.functional.cosine_similarity(
                           own.reshape(1, -1), other.reshape(1, -1))[0]))
            for prefix, tokens in (('teacher', item['target_tokens']), ('self', item['generation']['tokens'])):
                layout = build_prefix_layout(tokens, codec)
                logits, scores, contrasts = {}, {}, {}
                for mode, memory in (('correct', item['virtual']), ('different_asset', partner['virtual']),
                                     ('zero', torch.zeros_like(item['virtual']))):
                    logits[mode] = infer_logits(model, memory, tokens, device)
                    scores[mode] = summarize_logits(logits[mode], layout)
                    arrays[f'case_{len(rows):03d}_{prefix}_{mode}'] = logits[mode]
                for mode in ('different_asset', 'zero'):
                    contrasts[mode] = contrast_logits(logits['correct'], logits[mode], layout)
                    if prefix == 'self':
                        contrasts[mode]['mismatch_segments'] = contrast_segments(contrasts[mode], mismatch)
                row[prefix] = {'tokens': tokens, 'modes': scores, 'contrasts': contrasts}
            rows.append(row)
            write_json(args.output / f'rank_{rank}.json', {'complete': False, 'rank': rank, 'rows': rows})
        del cache
    after = parameter_fingerprint(model)
    if before != after or any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError('read-only diagnostic changed model state or populated gradients')
    expected = {(group, row['asset_id']) for group, selected in panel.items() for row in selected[rank::world]}
    actual = [(row['group'], row['asset_id']) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise RuntimeError('diagnostic missed or duplicated panel cases')
    peak = torch.cuda.max_memory_allocated(device) / torch.cuda.get_device_properties(device).total_memory
    if peak >= config['allocated_fraction_limit']:
        raise RuntimeError('diagnostic exceeded the published GPU memory limit')
    logits_path = args.output / f'logits_rank_{rank}.npz'
    with logits_path.with_suffix('.npz.tmp').open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    logits_path.with_suffix('.npz.tmp').replace(logits_path)
    result = {'passed': True, 'complete': True, 'rank': rank, 'rows': rows,
              'checkpoint_sha256': config['checkpoint']['sha256'], 'panel_sha256': manifest['panel_sha256'],
              'model_sha256_before': before, 'model_sha256_after': after,
              'optimizer_created': False, 'optimizer_updates': 0, 'baseline_generations_replayed': len(rows),
              'original_GT_and_raw_frames_verified': True,
              'max_saved_anchor_absolute_difference': max(row['max_saved_anchor_absolute_difference'] for row in rows),
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
              'peak_allocated_fraction': peak,
              'allocated_fraction_limit': config['allocated_fraction_limit'],
              'logits': {'path': str(logits_path), 'sha256': sha256(logits_path)}}
    write_json(args.output / f'rank_{rank}.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('condition_audit',), required=True)
    args = parser.parse_args()
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    claimed = False
    try:
        config = json.loads(args.config.read_text())
        base = checked_json(config['base_config'])
        validate_config(base)
        source = verify_operation(args, config, base)
        visible = ','.join(map(str, config['runtime']['physical_gpus']))
        if (world != config['runtime']['world_size'] or os.environ.get('CUDA_VISIBLE_DEVICES') != visible
                or not torch.cuda.is_available() or torch.cuda.device_count() != world):
            raise ValueError('visible CUDA devices differ from the published diagnostic')
        device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(config['allocated_fraction_limit'], device)
        if world > 1:
            dist.init_process_group('nccl', timeout=timedelta(minutes=15))
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=False)
        if world > 1:
            dist.barrier()
        claimed = True
        preflight = read_reviewed(base['safety']['preflight'], name='model preflight')
        if preflight['source_sha256'] != model_source_hashes():
            raise ValueError('neural or data implementation changed after reviewed preflight')
        data_audit = read_reviewed(base['safety']['data_audit'], name='data audit')
        if data_audit['target_source_sha256'] != data_source_hashes():
            raise ValueError('target implementation changed after data audit')
        initial_audit = read_reviewed(base['safety']['initialization'], name='initialization audit')
        validate_initialization_audit(base, initial_audit)
        reset_random(base['initialization']['seed'])
        model, tokenizer, codec, initial = build_fill_model(base, device)
        if initial['full_initial_state_sha256'] != initial_audit['initialization']['full_initial_state_sha256']:
            raise ValueError('new-model construction differs from the accepted official-only initialization')
        spec = config['checkpoint']
        if sha256(spec['path']) != spec['sha256']:
            raise ValueError('own new-model diagnostic checkpoint changed')
        saved = torch.load(spec['path'], map_location='cpu', mmap=True, weights_only=False)
        if (saved['schema'] != FAMILY or saved['mode'] != 'train' or saved['config'] != base
                or saved['architecture'] != model.configuration() or saved['training_contract'] != training_contract(base)
                or saved['asset_exposures'] != spec['asset_exposures'] or saved['step'] != spec['step']
                or saved['source'] != config['base_source_commit']):
            raise ValueError('own checkpoint contract differs from the completed trial')
        model.load_state_dict(saved['model'], strict=True)
        del saved
        local = run_audit(model, tokenizer, codec, base, config, args, rank, world, device)
        if world > 1:
            dist.barrier()
        if rank == 0:
            reports = [json.loads((args.output / f'rank_{number}.json').read_text()) for number in range(world)]
            if not all(report['complete'] and report['passed'] for report in reports):
                raise RuntimeError('a diagnostic rank failed to complete')
            rows = [row for report in reports for row in report['rows']]
            if len(rows) != 32 or len({(row['group'], row['asset_id']) for row in rows}) != 32:
                raise RuntimeError('combined diagnostic must cover all 32 fixed asset/frame contexts')
            result = {'complete': True, 'passed': True, 'quality_accepted': False,
                      'mode': args.mode, 'code_commit': source, 'config_sha256': sha256(args.config),
                      'checkpoint': spec, 'panel_sha256': config['panel_sha256'],
                      'baseline_generations_replayed': 32, 'optimizer_created': False, 'optimizer_updates': 0,
                      'model_sha256_unchanged': config['checkpoint']['model_sha256'],
                      'max_saved_anchor_absolute_difference': max(row['max_saved_anchor_absolute_difference'] for row in rows),
                      'all_saved_anchors_bitwise_equal': all(row['max_saved_anchor_absolute_difference'] == 0 for row in rows),
                      'groups': summarize_cases(rows),
                      'interpretation': [
                          'Paired counterfactual diagnostics, not candidate inference pipelines.',
                          'The different-asset condition has a different GT parent array within the same group/rank.',
                          'Zero condition is out of distribution; use real swaps as the principal input-use evidence.',
                          'Metrics are equal-weight per-case means; token totals are reported separately.',
                          'Self-prefix accuracy is agreement with generated tokens, not accuracy against GT.',
                          'JS uses natural logarithms; memory cosine alone is not a condition-use test.',
                      ],
                      'ranks': [{key: value for key, value in report.items() if key != 'rows'} for report in reports],
                      'finished_utc': datetime.now(timezone.utc).isoformat()}
            write_json(args.output / 'result.json', result)
            print('RESULT ' + json.dumps({'passed': True, 'cases': 32,
                'valid_teacher_contrasts': result['groups']['valid']['all']['prefixes']['teacher']['contrasts']}), flush=True)
        if world > 1:
            dist.barrier()
    except Exception as exc:
        if claimed:
            write_json(args.output / f'failure_rank_{rank}.json', {'complete': False, 'passed': False,
                'error': repr(exc), 'traceback': traceback.format_exc(), 'rank': rank})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
