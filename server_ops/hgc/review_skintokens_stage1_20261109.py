"""Review finished single-reference skeleton runs without loading a model."""
from collections import Counter
import hashlib
from itertools import accumulate
import json
import math
from pathlib import Path
import random
import statistics


POINTS = (0, 10000, 20000, 30000)
IDENTITY = ('asset_id', 'frame_index', 'raw_frame_index', 'normalization_sha256',
            'surface_sha256', 'original_GT_sha256', 'target_mesh_diagonal', 'gt_joint_count')


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(8 * 1024**2), b''):
            value.update(data)
    return value.hexdigest()


def stage_receipt(config_path, config, mode, source):
    root = Path(config['runtime']['output_root'])
    result_path, resource_path = root / mode / 'result.json', root / f'{mode}_resources.json'
    result, resources = read(result_path), read(resource_path)
    if (result['config_sha256'] != digest(config_path)
            or result['source'] != source or resources['source'] != source
            or result['stage2_enabled'] is not False
            or result.get('arm') != config.get('arm')
            or resources['mode'] != mode or resources['allocation'] != config['runtime']['allocation']
            or resources['physical_gpus'] != config['runtime']['physical_gpus']
            or resources['cpu_affinity'] != config['runtime']['cpu_affinity']
            or resources['exit_code'] != 0 or not resources['kernel_scheduler_membership']
            or not resources['payload_cleanup_complete']
            or not all(x['complete'] and x['passed'] for x in (result, resources))):
        raise ValueError('stage provenance or final resource receipt failed: '+str(result_path))
    peaks = resources['peak_gpu_memory_mib']
    if mode != 'audit' and (set(peaks) != {'3', '4'} or any(not 0 < x < 79872 for x in peaks.values())):
        raise ValueError('two-GPU sampled memory peaks differ from the bound')
    return result, {'path': str(result_path), 'result_sha256': digest(result_path),
        'resources_path': str(resource_path), 'resources_sha256': digest(resource_path),
        'source': source, 'output_name': mode, 'peak_gpu_memory_mib': peaks,
        'payload_cleanup_complete': True}


def panel(path, samples):
    summary, rows = read(path/'summary.json'), read(path/'stage1_rows.json')
    by_asset = {row['asset_id']: row for row in rows}
    if (len(rows) != len(by_asset) or len(rows) != 128 or summary['samples'] != samples
            or summary['stage2_enabled'] is not False or summary['reference_slot'] != 0
            or summary['policy'] != 'greedy_fp32_native_skeleton' or not summary['complete']
            or any(row['frame_index'] != 0 for row in rows)):
        raise ValueError('fixed one-reference validation is incomplete')
    quality = summary['stage1']
    if (quality['assets'] != 128 or quality['views'] != 128
            or any(not 0 <= r['original_edge_f1'] <= 1 for r in rows)
            or any(r['original_edge_f1'] != 0 for r in rows if not r['success'])
            or not math.isclose(statistics.fmean(r['original_edge_f1'] for r in rows),
                                quality['original_edge_f1'], abs_tol=1e-12)):
        raise ValueError('summary did not score every asset including unsuccessful generation')
    return summary, by_asset


def identities(rows):
    return {key: tuple(row[field] for field in IDENTITY) for key, row in rows.items()}


def compare(left, right):
    if identities(left) != identities(right):
        raise ValueError('reference frame, normalization, surface or GT changed')
    keys = sorted(left)
    delta = [right[k]['original_edge_f1'] - left[k]['original_edge_f1'] for k in keys]
    rng = random.Random(20261109)
    draws = sorted(statistics.fmean(rng.choices(delta, k=len(delta))) for _ in range(4096))
    common = [k for k in keys if left[k]['success'] and right[k]['success']]
    paired = {}
    if common:
        for metric in ('original_edge_f1', 'original_normalized_j2j'):
            paired[metric] = [statistics.fmean(rows[k][metric] for k in common) for rows in (left, right)]
    return {'assets': len(keys), 'edge_f1_delta': statistics.fmean(delta),
        'paired_asset_bootstrap_95': [draws[102], draws[3993]],
        'common_successes': len(common), 'common_success_pair': paired,
        'generation_status_left': dict(Counter(r['status'] for r in left.values())),
        'generation_status_right': dict(Counter(r['status'] for r in right.values()))}


def check_updates(out, durable, arm):
    sizes = [48] * 208 + [16]
    order, input_ids = hashlib.sha256(), hashlib.sha256()
    paths = [out / f'updates_rank_{rank}.jsonl' for rank in (0, 1)]
    for path in paths:
        if digest(path) != digest(durable/'receipts'/path.name):
            raise ValueError('training trace durable mirror differs')
    traces = [[json.loads(line) for line in path.read_text().splitlines()] for path in paths]
    frames = targets = 0
    if any(len(trace) != 627 for trace in traces):
        raise ValueError('627 completed updates required on both ranks')
    for i, (size, end) in enumerate(zip(sizes * 3, accumulate(sizes * 3))):
        local = []
        for trace in traces:
            row = trace[i]
            if ((row['step'], row['samples'], row['assets_in_update']) != (i+1, end, size)
                    or row['stage2_enabled'] is not False
                    or row.get('arm') != arm
                    or any(not math.isfinite(row[k]) for k in ('ar_loss', 'gradient_norm', 'lr', 'beta1'))):
                raise ValueError('nonfinite loss/gradient or incorrect update boundary')
            if i == 0 and (not row['gradient_groups']
                    or any(not math.isfinite(x) or x <= 0 for x in row['gradient_groups'].values())):
                raise ValueError('missing initial finite nonzero gradients')
            if arm is None:
                batch = {**row, 'raw_frames': [[x] for x in row['raw_frames']],
                         'target_slots': [[0] for _ in row['assets']]}
                parts = [batch]
            else:
                parts = row['microbatches']
            count = 0
            for part in parts:
                for asset, raw, seed, slots in zip(part['assets'], part['raw_frames'],
                                                   part['sampling_seeds'], part['target_slots'], strict=True):
                    if (len(raw) < 1 or not slots or not set(slots) <= set(range(len(raw)))
                            or len(set(slots)) != len(slots)
                            or (arm in (None, 'temporal_query') and slots != [0])):
                        raise ValueError('supervision is not the declared real-frame selection')
                    local.append((asset, seed))
                    input_ids.update(json.dumps([asset, seed, raw], separators=(',', ':')).encode())
                    count += 1
                    frames += len(raw)
                    targets += len(slots)
            if count != size // 2:
                raise ValueError('microbatch trace did not account for each local visit')
        order.update(json.dumps(local, separators=(',', ':')).encode())
    return {'ordered_assets_sha256': order.hexdigest(), 'ordered_inputs_sha256': input_ids.hexdigest(),
            'input_frame_visits': frames, 'supervised_frame_visits': targets}


def review_training(config_path, config, source):
    result, receipt = stage_receipt(config_path, config, 'train', source)
    root = Path(config['runtime']['output_root']) / 'train'
    durable = Path(config['artifacts']['durable_root'])
    if ((result['samples'], result['updates']) != (30000, 627) or result['working_model_changed']
            or [x['samples'] for x in result['checkpoints']] != list(POINTS[1:])
            or [x['samples'] for x in result['validation']] != list(POINTS)):
        raise ValueError('training did not reach the exact planned endpoint')
    for rank in (0, 1):
        rank_result = read(root/f'result_rank_{rank}.json')
        if {k:v for k,v in rank_result.items() if k != 'finished_utc'} != {
                k:v for k,v in result.items() if k != 'finished_utc'}:
            raise ValueError('rank completion results disagree')
    trace = check_updates(root, durable, config.get('arm'))
    if any(trace[k] != result[k] for k in ('input_frame_visits', 'supervised_frame_visits') if k in result):
        raise ValueError('observed frame visits differ from the endpoint receipt')
    for n, ckpt in enumerate(result['checkpoints'], start=1):
        name = f'checkpoint_{ckpt["samples"]:06d}.pt'
        if (not all(ckpt[k] for k in ('complete', 'both_copies_verified', 'optimizer_and_rank_RNG_retained'))
                or ckpt['step'] != n*209 or Path(ckpt['path']) != root/name
                or Path(ckpt['durable_path']) != durable/name):
            raise ValueError('checkpoint recovery contract differs')
        for path in (root/name, durable/name):
            if (read(path.with_suffix('.json')) != ckpt or path.stat().st_size != ckpt['bytes']
                    or digest(path) != ckpt['sha256']):
                raise ValueError('full checkpoint or durable mirror changed: '+str(path))
    curve = {}
    first = previous = None
    for samples, expected in zip(POINTS, result['validation'], strict=True):
        name = f'evaluation_{samples:06d}'
        summary, rows = panel(root/name, samples)
        if summary != expected or any(digest(root/name/f) != digest(durable/name/f)
                                      for f in ('summary.json', 'stage1_rows.json')):
            raise ValueError('validation receipt or durable mirror differs')
        if first is None:
            first = previous = rows
        curve[str(samples)] = {'metrics': summary['stage1'],
            'against_initial': compare(first, rows), 'against_previous': compare(previous, rows)}
        previous = rows
    return {'complete': True, 'passed': True, 'stage2_enabled': False, 'arm': config.get('arm', 'static'),
        'samples': result['samples'], 'updates': result['updates'], 'plan_sha256': result['plan_sha256'],
        **trace, 'checkpoints': result['checkpoints'], 'curve': curve,
        'retained_panel_root': str(durable), 'resources': receipt, 'working_model_changed': False}
