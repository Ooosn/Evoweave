"""Strict cached stage1 references and zero-update stage2 inference."""
from collections import Counter
import gc
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import torch.distributed as dist

from ar_only_runtime import load_evaluation_model
from fusion_evaluation import tree_fingerprint
from predicted_reference_protocol import (METHODS, ROOT, bound_config, compare, reference_from_row,
    scientific_sha, sha256, summarize)
from reference_flow_runtime import mirror, read_all
from reference_xyz_runtime import row_metrics
from stage1_feedback import generate_asset, score_generation, summarize_rows
from stage1_surface_repair_runtime import canonical_record, canonical_identity, validate_canonical
from virtual_pair_runtime import evaluation_rows, gather
from virtual_query_xyz_runtime import fixed_dataset, write_json
from rigweave.dynamic_rig.anchor_joint_binding import AnchorJointBinding, AnchorJointConfig
from rigweave.dynamic_rig.virtual_frame_data import prepare_virtual_surface


def source_rows(config):
    spec = config['stage1_source']
    for item in (spec['summary'], *spec['rows']):
        if sha256(item['path']) != item['sha256']:
            raise ValueError('accepted stage1 evidence changed')
    summary = json.loads(Path(spec['summary']['path']).read_text())
    rows = sum([[json.loads(line) for line in Path(x['path']).read_text().splitlines()] for x in spec['rows']], [])
    identity = summary['identity']
    if (summary['summary'] != summarize_rows(rows) or not summary['no_GT_reference_or_teacher_prefix']
            or identity['checkpoint_sha256'] != config['checkpoints']['stage1']['sha256']
            or identity['decoding']['precision'] != 'fp32'):
        raise ValueError('stage1 decoder/checkpoint evidence differs')
    values = {(r['asset_id'], r['frame_index']): r for r in rows}
    if len(values) != len(rows) or len(rows) != 1024:
        raise ValueError('stage1 panel coverage differs')
    return rows, values


def audit(config, output, source):
    original = bound_config(config['parent_config'])
    base, entries = read_all(original)
    selected = evaluation_rows(base, entries)
    rows, keyed = source_rows(config)
    ids = [r['asset_id'] for r in selected]
    if len(ids) != 128 or set(keyed) != {(a, f) for a in ids for f in range(8)}:
        raise ValueError('fixed panel changed')
    for spec in config['checkpoints'].values():
        if sha256(spec['path']) != spec['sha256']:
            raise ValueError('pipeline checkpoint changed')
    canonical = config['canonical_source']
    if sha256(canonical['path']) != canonical['sha256']:
        raise ValueError('retained surface manifest changed')
    archives = json.loads(Path(canonical['path']).read_text())
    for item in archives:
        if sha256(item['path']) != item['sha256']:
            raise ValueError('retained canonical archive changed')
    inherited = config['capacity']['inherited_generation']
    capacity = json.loads(Path(inherited['path']).read_text())
    resources = json.loads(Path(inherited['resources_path']).read_text())
    if sha256(inherited['path']) != inherited['sha256'] or not capacity['passed'] or not resources['passed'] or not resources['payload_cleanup_complete']:
        raise ValueError('generation capacity not accepted')
    if any((x['full_decode']['precision'], x['full_decode']['input_frames'], x['full_decode']['new_tokens']) != ('fp32', 24, 2048) for x in capacity['ranks']):
        raise ValueError('generation stress shape differs')
    paths = ['model_training/rigweave/src', 'model_training/rigweave/scripts/stage1_feedback.py',
             'model_training/rigweave/scripts/ar_only_runtime.py', 'model_training/rigweave/scripts/virtual_query_xyz_runtime.py']
    subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet', config['prior_source'], source, '--', *paths], check=True)
    statuses, joint_counts = Counter(), []
    for asset in ids:
        for ref in (0, 4):
            parsed, status = reference_from_row(keyed[asset, ref])
            statuses[status] += 1
            if parsed is not None:
                joint_counts.append(len(parsed[0]))
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU audit initialized CUDA')
    result = {'complete': True, 'passed': True, 'source': source, 'scientific_sha256': scientific_sha(config),
        'optimizer_updates': 0, 'cuda_initialized': False, 'assets': 128, 'references': 256,
        'views_per_method': 1792, 'reference_statuses': dict(statuses), 'maximum_predicted_joints': max(joint_counts),
        'canonical_archives_available': len(archives), 'unchanged_inference_source': True}
    write_json(output / 'result.json', result)
    return result


def score_output(prediction, parents, truth, truth_parents, diagonal, *, known_ids, status):
    record = {'success': False, 'status': status, 'edge_f1': 0., 'exact_count': False,
              'normalized_j2j': None, 'normalized_mpjpe': None, 'predicted_joint_count': None}
    if prediction is None:
        return record
    if not bool(torch.isfinite(prediction).all()):
        record['status'] = 'nonfinite_prediction'
        return record
    metrics = row_metrics(prediction, parents, truth, truth_parents, diagonal, paired_ids=known_ids)
    record.update(metrics, success=True, status='success', exact_count=len(parents) == len(truth_parents),
                  predicted_joint_count=len(parents), predicted_joints=prediction.float().cpu().tolist(), predicted_parents=parents)
    return record


@torch.inference_mode()
def head_prediction(head, features, surface, xyz, parents, ref):
    count = len(parents)
    mask = torch.ones((1, count), dtype=torch.bool, device=features.device)
    return head(features, surface['query_points'], surface['query_normals'], xyz[None],
                torch.tensor([parents], device=features.device), mask,
                torch.tensor([ref], device=features.device))[0]


@torch.inference_mode()
def run(config, output, source, device, rank, world):
    spec = config['audit_result']
    if not spec or sha256(spec['path']) != spec['sha256']:
        raise ValueError('CPU audit not bound')
    checked = json.loads(Path(spec['path']).read_text())
    if not checked['passed'] or checked['scientific_sha256'] != scientific_sha(config):
        raise ValueError('CPU admission differs')
    started = time.monotonic()
    def boundary():
        from virtual_pair_guard import deadline_due
        if time.monotonic() - started > config['evaluation']['payload_limit_seconds'] or deadline_due(config):
            raise RuntimeError('bounded evaluation deadline reached')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    original = bound_config(config['parent_config'])
    base, entries = read_all(original)
    selected = evaluation_rows(base, entries)
    dataset = fixed_dataset(selected, base, 8, mode='seen_eval')
    previous, keyed = source_rows(config)
    retained = {x['asset_id']: x for x in json.loads(Path(config['canonical_source']['path']).read_text())}
    records, references, reused, pending = [], {}, [], []
    for index in range(rank, len(selected), world):
        boundary()
        asset = selected[index]['asset_id']
        if asset in retained:
            item = retained[asset]
            if sha256(item['path']) != item['sha256']:
                raise ValueError('canonical cache changed')
            record = torch.load(item['path'], map_location='cpu', weights_only=True)
        else:
            batch = dataset[index]
            surface, geometry = prepare_virtual_surface([batch], device, **base['surface_sampling'])
            record = canonical_record(batch, surface, geometry)
            del batch, surface, geometry
        validate_canonical(record, previous)
        records.append(record)
        if all(record['surface_sha256'] == keyed[asset, ref]['surface_sha256'] for ref in (0, 4)):
            references[asset] = {ref: dict(keyed[asset, ref]) for ref in (0, 4)}
            reused.append(asset)
        else:
            pending.append(asset)
    cached = output / f'canonical_rank_{rank}.pt'
    torch.save(records, cached)
    saved = torch.load(cached, map_location='cpu', weights_only=True)
    for record in saved:
        validate_canonical(record, previous)
    del saved
    cache_artifact = mirror(cached, cached.name, config)
    if pending:
        model, tokenizer = load_evaluation_model(original, device, config['checkpoints']['stage1'])
        for record in records:
            boundary()
            asset = record['asset_id']
            if asset not in pending:
                continue
            surface = {k: v.to(device) for k, v in record['surface'].items()}
            if tree_fingerprint(surface) != record['surface_sha256']:
                raise ValueError('canonical device input changed')
            generated = generate_asset(model, surface, tokenizer, frame_indices=[0, 4], precision='fp32')
            references[asset] = {}
            for ref in (0, 4):
                row = score_generation(generated[ref], tokenizer, record['truth'][ref].numpy(), record['parents'].tolist(), float(record['diagonals'][ref]))
                row.update(canonical_identity(record, ref))
                references[asset][ref] = row
            del surface, generated
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    reference_path = output / f'references_rank_{rank}.json'
    write_json(reference_path, {'reused_assets': reused, 'regenerated_assets': pending, 'references': references})
    reference_artifact = mirror(reference_path, reference_path.name, config)
    backbone, _ = load_evaluation_model(original, device, config['checkpoints']['stage2'])
    architecture = bound_config(config['stage2_config'])['architecture']
    head = AnchorJointBinding(AnchorJointConfig(**architecture), 'reference_transport').to(device).eval()
    payload = torch.load(config['checkpoints']['stage2']['path'], map_location='cpu', mmap=True, weights_only=False)
    head.load_state_dict({k.removeprefix('head.'): v for k, v in payload['model'].items() if k.startswith('head.')}, strict=True)
    del payload
    before_head = tree_fingerprint(head.state_dict())
    profile = []
    def joint_size(record):
        sizes = [len(record['parents'])]
        for row in references[record['asset_id']].values():
            parsed, _ = reference_from_row(row)
            if parsed is not None:
                sizes.append(len(parsed[0]))
        return max(sizes)
    for record in sorted(records, key=joint_size, reverse=True)[:3]:
        boundary()
        surface = {k: v.to(device) for k, v in record['surface'].items()}
        torch.cuda.reset_peak_memory_stats(device)
        features, _ = backbone.encode(surface)
        largest = (record['truth'][0].numpy(), record['parents'].tolist())
        for row in references[record['asset_id']].values():
            parsed, _ = reference_from_row(row)
            if parsed is not None and len(parsed[0]) > len(largest[0]):
                largest = parsed
        prediction = head_prediction(head, features, surface, torch.as_tensor(largest[0], device=device), largest[1], 0)
        if not bool(torch.isfinite(prediction).all()):
            raise RuntimeError('stage2 capacity produced nonfinite coordinates')
        profile.append({'asset_id': record['asset_id'], 'joints': len(largest[0]),
                        'max_reserved_bytes': torch.cuda.max_memory_reserved(device)})
        del features, prediction, surface
    write_json(output / f'profile_rank_{rank}.json', {'passed': True, 'optimizer_updates': 0, 'cases': profile})
    rows, examples = [], []
    torch.cuda.reset_peak_memory_stats(device)
    for index, record in enumerate(records):
        boundary()
        asset = record['asset_id']
        surface = {k: v.to(device) for k, v in record['surface'].items()}
        if tree_fingerprint(surface) != record['surface_sha256']:
            raise ValueError('stage2 surface differs from stage1 input')
        features, _ = backbone.encode(surface)
        truth = record['truth'].to(device)
        truth_parents = record['parents'].tolist()
        asset_rows = []
        for ref in (0, 4):
            for kind in ('GT', 'predicted'):
                parsed, status = ((record['truth'][ref].numpy(), truth_parents), 'valid') if kind == 'GT' else reference_from_row(references[asset][ref])
                ref_sha = tree_fingerprint({'kind': kind, 'row': references[asset][ref]}) if kind == 'predicted' else tree_fingerprint({'xyz': record['truth'][ref], 'parents': record['parents']})
                if parsed is not None:
                    xyz, parents = torch.as_tensor(parsed[0], device=device), parsed[1]
                    prediction = head_prediction(head, features, surface, xyz, parents, ref)
                else:
                    xyz, parents, prediction = None, None, None
                for method in ('head', 'copy'):
                    for frame in range(8):
                        if frame == ref:
                            continue
                        output_xyz = (prediction[frame] if method == 'head' else xyz) if parsed is not None else None
                        row = score_output(output_xyz, parents, truth[frame], truth_parents, float(record['diagonals'][frame]), known_ids=kind == 'GT', status=status)
                        row.update(canonical_identity(record, frame), method=kind + '_' + method, reference_kind=kind,
                            reference_slot=ref, reference_raw_frame=record['raw_frames'][ref], target_slot=frame,
                            reference_sha256=ref_sha, stage1_cache_reused=asset in reused)
                        rows.append(row)
                        asset_rows.append(row)
                del prediction
        examples.append({'asset_id': asset, 'raw_frames': record['raw_frames'], 'points': record['surface']['query_points'][0],
                         'truth': record['truth'], 'parents': record['parents'], 'rows': asset_rows,
                         'references': references[asset]})
        write_json(output / f'progress_rank_{rank}.json', {'processed_local_assets': index + 1, 'total_local_assets': len(records),
            'reused_stage1_assets': len(reused), 'regenerated_stage1_assets': len(pending), 'seconds': time.monotonic() - started})
        del features, surface, truth
    if tree_fingerprint(head.state_dict()) != before_head:
        raise RuntimeError('inference changed stage2 parameters')
    row_path = output / f'rows_rank_{rank}.json'
    write_json(row_path, rows)
    example_path = output / f'examples_rank_{rank}.pt'
    torch.save(examples, example_path)
    receipts = gather({'rank': rank, 'cache': cache_artifact, 'references': reference_artifact,
        'rows': mirror(row_path, row_path.name, config), 'examples': mirror(example_path, example_path.name, config),
        'profile': profile, 'reused_stage1_assets': len(reused), 'regenerated_stage1_assets': len(pending),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}, world)
    result = None
    if rank == 0:
        all_rows = sum([json.loads(Path(r['rows']['path']).read_text()) for r in receipts], [])
        groups = {method: [r for r in all_rows if r['method'] == method] for method in METHODS}
        expected = {(a['asset_id'], ref, frame) for a in selected for ref in (0, 4) for frame in range(8) if frame != ref}
        for method, group in groups.items():
            if len(group) != 1792 or {(r['asset_id'], r['reference_slot'], r['target_slot']) for r in group} != expected:
                raise RuntimeError('final evaluation denominator changed')
        pairs = [('predicted_head', 'GT_head'), ('predicted_head', 'predicted_copy'), ('GT_head', 'GT_copy')]
        result = {'complete': True, 'passed': True, 'source': source, 'scientific_sha256': scientific_sha(config),
            'optimizer_updates': 0, 'checkpoints': config['checkpoints'], 'input_contract': config['evaluation'],
            'summaries': {m: summarize(v) for m, v in groups.items()},
            'comparisons': {a + '_minus_' + b: compare(groups[a], groups[b]) for a, b in pairs},
            'artifacts': receipts, 'quality_accepted': False}
        write_json(output / 'result.json', result)
        mirror(output / 'result.json', 'result.json', config)
    return gather(result, world)[0]
