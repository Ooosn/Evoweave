#!/usr/bin/env python3
"""Locate a validation identity change without loading model weights."""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist

from fusion_evaluation import tree_fingerprint
from train_anchor_stage2 import context
from virtual_pair_guard import ROOT, verify_launch
from virtual_pair_runtime import evaluation_rows, gather
from virtual_query_xyz_runtime import fixed_dataset, sha256, write_json
from rigweave.dynamic_rig import virtual_frame_data


def differences(a, b):
    changed = a != b
    count = int(changed.sum())
    return {'different_values': count, 'total_values': a.numel(),
            'max_abs': float((a[changed] - b[changed]).abs().max()) if count else 0.}


def replay(batch, device, sampling):
    original_sample = virtual_frame_data.sample_trackable_surface
    original_multinomial = torch.multinomial
    reference, draw = {}, {}

    def sample(*args, **kwargs):
        result = original_sample(*args, **kwargs)
        reference.update(vars(result))
        return result

    def multinomial(weights, count, *args, **kwargs):
        generator = kwargs['generator']
        state = generator.get_state()
        result = original_multinomial(weights, count, *args, **kwargs)
        if draw:
            raise ValueError('expected one face draw per asset')
        draw.update(weights=weights, state=state, faces=result)
        return result

    with patch.object(virtual_frame_data, 'sample_trackable_surface', sample), patch.object(
            torch, 'multinomial', multinomial):
        surface, geometry = virtual_frame_data.prepare_virtual_surface([batch], device, **sampling)
    target, parents = geometry['frame_target_joints'][0], geometry['target_parents'][0]
    vertices = batch['frame_vertices'][0]
    diagonals = torch.linalg.vector_norm(vertices.amax(1) - vertices.amin(1), dim=-1)
    values = {**surface, **{f'ref_{k}': v for k, v in reference.items()},
              'face_areas': draw['weights'], 'rng_before_face_draw': draw['state']}
    cpu = {k: v.detach().cpu() for k, v in values.items()}
    if not torch.equal(draw['faces'].cpu(), cpu['ref_face_indices'][0]):
        raise ValueError('face samples do not match the intercepted draw')
    identity = {'asset_id': batch['asset_id'], 'raw_frames': geometry['selected_frames'][0].tolist(),
                'surface_sha256': tree_fingerprint({k: cpu[k] for k in surface}),
                'gt_sha256': tree_fingerprint({'joints': target, 'parents': parents, 'diagonals': diagonals})}
    return identity, {k: tree_fingerprint(v) for k, v in cpu.items()}, cpu


def panel_hash(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def compare_panel(rows, targets):
    rows = sorted(rows, key=lambda x: x['asset_id'])
    if len(rows) != 128 or len({r['asset_id'] for r in rows}) != 128:
        raise ValueError('fixed validation panel coverage differs')
    rounds = [panel_hash([r['rounds'][i] for r in rows]) for i in range(len(rows[0]['rounds']))]
    variants = [list({json.dumps(v, sort_keys=True): v for v in r['rounds']}.values()) for r in rows]
    combinations = math.prod(map(len, variants))
    matches = {}
    if combinations <= 65536:
        encoded = [[(json.dumps(v, sort_keys=True).encode(), v) for v in group] for group in variants]
        for combination in itertools.product(*encoded):
            fingerprint = hashlib.sha256(b'[' + b', '.join(value[0] for value in combination) + b']').hexdigest()
            if fingerprint in targets.values():
                matches[fingerprint] = {v['asset_id']: v['surface_sha256']
                                        for (_, v), group in zip(combination, variants, strict=True)
                                        if len(group) > 1}
    different = [r for r in rows if r['changes']]
    return {'whole_panel_sha256_by_round': rounds, 'historical_sha256': targets,
            'number_of_possible_panels': combinations,
            'exact_historical_matches': {k: matches[v] for k, v in targets.items() if v in matches},
            'assets_with_replayed_differences': len(different), 'differences': different}


@torch.no_grad()
def run(config, output, source, device, rank):
    probe = config['replay']
    base_path = ROOT/probe['failed_config']['relative_path']
    if sha256(base_path) != probe['failed_config']['sha256']:
        raise ValueError('failed training data/sampling configuration changed')
    base, entries, _, resolver = context(json.loads(base_path.read_text()), 24)
    selected = evaluation_rows(base, entries)
    dataset = fixed_dataset(selected, base, 8, mode='seen_eval', file_resolver=resolver)
    historical = {}
    for mark, spec in probe['summaries'].items():
        if sha256(spec['path']) != spec['sha256']:
            raise ValueError('retained validation evidence changed')
        historical[mark] = json.loads(Path(spec['path']).read_text())['input_sha256']
    expected_frames = {}
    if sha256(probe['rows']['path']) != probe['rows']['sha256']:
        raise ValueError('retained validation rows changed')
    for row in json.loads(Path(probe['rows']['path']).read_text()):
        if row['method'] != 'head':
            continue
        values = expected_frames.setdefault(row['asset_id'], {})
        values[row['reference_slot']] = row['reference_raw_frame']
        values[row['target_slot']] = row['target_raw_frame']
    started = time.monotonic()
    records = []
    for index in range(rank, len(selected), 2):
        if time.monotonic() - started > probe['max_seconds']:
            raise RuntimeError('bounded replay timed out')
        batch = dataset[index]
        identity, fields, first = replay(batch, device, base['surface_sampling'])
        asset = identity['asset_id']
        if identity['raw_frames'] != [expected_frames[asset][i] for i in range(8)]:
            raise ValueError('replay raw-frame identity differs from historical rows')
        row = {'asset_id': asset, 'rounds': [identity], 'initial_field_sha256': fields, 'changes': []}
        for repeat in range(1, probe['repeats']):
            current, new_fields, values = replay(batch, device, base['surface_sampling'])
            if current['gt_sha256'] != identity['gt_sha256'] or current['raw_frames'] != identity['raw_frames']:
                raise ValueError('input replay changed GT or raw frames')
            row['rounds'].append(current)
            if fields != new_fields and not any(change['field_sha256'] == new_fields for change in row['changes']):
                row['changes'].append({'repeat': repeat, 'field_sha256': new_fields,
                    'delta': {k: differences(first[k], values[k]) for k in fields if fields[k] != new_fields},
                    'same_face_areas': fields['face_areas'] == new_fields['face_areas'],
                    'same_face_rng_state': fields['rng_before_face_draw'] == new_fields['rng_before_face_draw']})
            del values
        records.append(row)
        write_json(output/f'progress_rank_{rank}.json', {'assets': len(records),
            'changed_assets': sum(bool(r['changes']) for r in records), 'last_asset': asset})
    write_json(output/f'replay_rank_{rank}.json', records)
    joined = sum(gather(records, 2), [])
    if rank == 0:
        result = compare_panel(joined, historical)
        write_json(output/'result.json', {**result, 'complete': True, 'passed': True,
            'source': source, 'assets': len(joined), 'repeats': probe['repeats'],
            'torch_version': str(torch.__version__), 'cuda_version': torch.version.cuda,
            'model_loaded': False, 'optimizer_updates': 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['capacity'], required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    source = verify_launch(args, config)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rank = int(os.environ['RANK'])
    if int(os.environ['WORLD_SIZE']) != 2 or torch.cuda.device_count() != 2:
        raise ValueError('two ranks must replay the historical GPU split')
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', timeout=timedelta(minutes=15))
    try:
        if rank == 0:
            args.output.mkdir()
        dist.barrier(device_ids=[device.index])
        run(config, args.output, source, device, rank)
        dist.barrier(device_ids=[device.index])
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
