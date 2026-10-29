"""Preselected visual examples, one official dynamic reference each, no training."""
import gc
import json
from pathlib import Path
import subprocess
import time

import torch

from ar_only_runtime import load_evaluation_model
from fusion_evaluation import tree_fingerprint
from predicted_reference_protocol import (METHODS, ROOT, bound_config, reference_from_row,
    scientific_sha, sha256, summarize)
from predicted_reference_runtime import head_prediction, score_output, source_rows
from reference_flow_runtime import mirror, read_all
from stage1_feedback import score_generation
from stage1_official_inference import generate_condition, reference_seed, OFFICIAL_OPTIONS, POLICY
from stage1_surface_repair_runtime import canonical_identity, validate_canonical
from virtual_pair_runtime import evaluation_rows, gather
from virtual_query_xyz_runtime import write_json
from rigweave.dynamic_rig.anchor_joint_binding import AnchorJointBinding, AnchorJointConfig


def load_records(config, rank=None):
    spec = config['official_smoke']['input_run']
    for item in (spec['result'], spec['resources']):
        if sha256(item['path']) != item['sha256']:
            raise ValueError('retained input provenance changed')
    result = json.loads(Path(spec['result']['path']).read_text())
    resource = json.loads(Path(spec['resources']['path']).read_text())
    if not (result['complete'] and result['passed'] and resource['passed']
            and resource['exit_code'] == 0 and resource['payload_cleanup_complete']):
        raise ValueError('retained input producer did not complete cleanly')
    original = bound_config(config['parent_config'])
    base, entries = read_all(original)
    panel = evaluation_rows(base, entries)
    selected = {panel[i]['asset_id']: i for i in config['official_smoke']['case_indices']}
    previous, _ = source_rows(config)
    records = []
    for item in result['artifacts']:
        if rank is not None and item['rank'] != rank:
            continue
        cache = item['cache']
        if sha256(cache['path']) != cache['sha256']:
            raise ValueError('retained canonical tensor archive changed')
        for record in torch.load(cache['path'], map_location='cpu', mmap=True, weights_only=True):
            if record['asset_id'] in selected:
                validate_canonical(record, previous)
                records.append(dict(record, panel_index=selected[record['asset_id']]))
    expected = {asset for asset, i in selected.items() if rank is None or i % 2 == rank}
    if {r['asset_id'] for r in records} != expected or len(records) != len(expected):
        raise ValueError('fixed smoke cases missing or duplicated')
    return sorted(records, key=lambda r: selected[r['asset_id']])


def inherited_capacity(config):
    spec = config['official_smoke'].get('inherited_official_capacity')
    if spec is None:
        return None
    for item in spec.values():
        if sha256(item['path']) != item['sha256']:
            raise ValueError('official beam capacity provenance changed')
    result = json.loads(Path(spec['result']['path']).read_text())
    resources = json.loads(Path(spec['resources']['path']).read_text())
    if (not result['passed'] or not result['complete'] or not resources['passed']
            or resources['exit_code'] != 0 or not resources['payload_cleanup_complete']
            or result['checkpoints'] != config['checkpoints']
            or result['policy']['generation_kwargs'] != config['official_smoke']['generation_kwargs']):
        raise ValueError('official capacity contract differs')
    profiles = {x['rank']: dict(x['beam_capacity'], inherited_from=spec['result']) for x in result['artifacts']}
    if set(profiles) != {0, 1} or any(not p['passed'] or (p['new_tokens'], p['num_beams'], p['input_frames']) != (2048, 15, 8) for p in profiles.values()):
        raise ValueError('inherited full beam stress missing')
    return profiles


def audit(config, output, source):
    records = load_records(config)
    capacity = inherited_capacity(config)
    for spec in config['checkpoints'].values():
        if sha256(spec['path']) != spec['sha256']:
            raise ValueError('pipeline checkpoint changed')
    subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet', config['prior_source'], source, '--',
        'model_training/rigweave/src', 'model_training/rigweave/scripts/ar_only_runtime.py',
        'model_training/rigweave/scripts/stage1_feedback.py',
        'model_training/rigweave/scripts/stage1_official_inference.py'], check=True)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU audit initialized CUDA')
    result = {'complete': True, 'passed': True, 'source': source,
        'scientific_sha256': scientific_sha(config), 'optimizer_updates': 0,
        'assets': [r['asset_id'] for r in records], 'references': len(records), 'views_per_method': len(records)*7,
        'inherited_official_capacity': capacity,
        'cuda_initialized': False, 'stage1_greedy_outputs_reused': 0,
        'policy': config['official_smoke'], 'quality_accepted': False}
    write_json(output/'result.json', result)
    return result


@torch.inference_mode()
def run(config, output, source, device, rank, world):
    spec = config['audit_result']
    if not spec or sha256(spec['path']) != spec['sha256']:
        raise ValueError('CPU audit not bound')
    audit_result = json.loads(Path(spec['path']).read_text())
    if not audit_result['passed'] or audit_result['scientific_sha256'] != scientific_sha(config):
        raise ValueError('CPU admission differs')
    started = time.monotonic()
    def boundary():
        from virtual_pair_guard import deadline_due
        if time.monotonic()-started > config['evaluation']['payload_limit_seconds'] or deadline_due(config):
            raise RuntimeError('smoke evaluation time limit reached')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    original = bound_config(config['parent_config'])
    records = load_records(config, rank)
    model, tokenizer = load_evaluation_model(original, device, config['checkpoints']['stage1'])
    references = {}
    inherited = inherited_capacity(config)
    profile = inherited[rank] if inherited else None
    for record in records:
        boundary()
        asset = record['asset_id']
        surface = {k: v.to(device) for k, v in record['surface'].items()}
        if tree_fingerprint(surface) != record['surface_sha256']:
            raise ValueError('device surface differs')
        real, virtual = model.encode(surface)
        condition = model.conditions(real, virtual)[:, 0]
        seed = reference_seed(asset, record['raw_frames'][0], config['official_smoke']['base_seed'])
        if profile is None:
            torch.cuda.reset_peak_memory_stats(device)
            tick = time.monotonic()
            stress = generate_condition(model.xyz_decoder, condition, tokenizer, seed=seed, capacity_only=True)
            torch.cuda.synchronize(device)
            profile = {'passed': True, 'new_tokens': len(stress['tokens'])-2, 'num_beams': 15,
                'input_frames': 8, 'seconds': time.monotonic()-tick, 'capacity_only': True,
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}
            write_json(output/f'beam_capacity_rank_{rank}.json', profile)
            del stress
        boundary()
        tick = time.monotonic()
        generated = generate_condition(model.xyz_decoder, condition, tokenizer, seed=seed)
        row = score_generation(generated, tokenizer, record['truth'][0].numpy(),
                               record['parents'].tolist(), float(record['diagonals'][0]))
        row.update(canonical_identity(record, 0), policy=POLICY, seed=seed,
            generation_kwargs=dict(OFFICIAL_OPTIONS), generation_seconds=time.monotonic()-tick)
        references[asset] = row
        write_json(output/f'references_rank_{rank}.json', references)
        write_json(output/f'progress_rank_{rank}.json', {'generated_references': len(references),
            'total_references': len(records), 'phase': 'official_generation', 'seconds': time.monotonic()-started})
        del real, virtual, condition, surface, generated
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    backbone, _ = load_evaluation_model(original, device, config['checkpoints']['stage2'])
    architecture = bound_config(config['stage2_config'])['architecture']
    head = AnchorJointBinding(AnchorJointConfig(**architecture), 'reference_transport').to(device).eval()
    payload = torch.load(config['checkpoints']['stage2']['path'], map_location='cpu', mmap=True, weights_only=False)
    head.load_state_dict({k.removeprefix('head.'): v for k, v in payload['model'].items() if k.startswith('head.')}, strict=True)
    del payload
    examples, rows = [], []
    for record in records:
        boundary()
        asset = record['asset_id']
        surface = {k: v.to(device) for k, v in record['surface'].items()}
        if tree_fingerprint(surface) != references[asset]['surface_sha256']:
            raise ValueError('stage2 input differs from stage1')
        features, _ = backbone.encode(surface)
        truth, gt_parents = record['truth'].to(device), record['parents'].tolist()
        asset_rows = []
        for kind in ('GT', 'predicted'):
            parsed, status = ((record['truth'][0].numpy(), gt_parents), 'valid') if kind == 'GT' else reference_from_row(references[asset])
            if parsed is None:
                xyz, parents, prediction = None, None, None
            else:
                xyz, parents = torch.as_tensor(parsed[0], device=device), parsed[1]
                prediction = head_prediction(head, features, surface, xyz, parents, 0)
            for method in ('head', 'copy'):
                for frame in range(1, 8):
                    output_xyz = (prediction[frame] if method == 'head' else xyz) if parsed is not None else None
                    row = score_output(output_xyz, parents, truth[frame], gt_parents,
                        float(record['diagonals'][frame]), known_ids=kind == 'GT', status=status)
                    row.update(canonical_identity(record, frame), method=kind+'_'+method,
                        reference_slot=0, target_slot=frame, reference_kind=kind,
                        reference_raw_frame=record['raw_frames'][0], stage1_cache_reused=False)
                    rows.append(row)
                    asset_rows.append(row)
        example = {'asset_id': asset, 'raw_frames': record['raw_frames'], 'panel_index': record['panel_index'],
            'points': record['surface']['query_points'][0], 'truth': record['truth'],
            'parents': record['parents'], 'rows': asset_rows, 'reference': references[asset]}
        if config['official_smoke'].get('visual_point_limit'):
            dense = record['surface']['dense_points'][0]
            if dense.ndim != 3 or dense.shape[0] != 8:
                raise ValueError('visual surface frame layout differs')
            limit = config['official_smoke']['visual_point_limit']
            step = max(1, (dense.shape[1]+limit-1)//limit)
            example['visual_points'] = dense[:, ::step].clone()
        examples.append(example)
        del features, surface, truth, prediction
    row_path, example_path = output/f'rows_rank_{rank}.json', output/f'examples_rank_{rank}.pt'
    write_json(row_path, rows)
    torch.save(examples, example_path)
    receipts = gather({'rank': rank, 'beam_capacity': profile, 'rows': mirror(row_path, row_path.name, config),
        'examples': mirror(example_path, example_path.name, config),
        'references': mirror(output/f'references_rank_{rank}.json', f'references_rank_{rank}.json', config)}, world)
    result = None
    if rank == 0:
        all_rows = sum([json.loads(Path(x['rows']['path']).read_text()) for x in receipts], [])
        groups = {m: [r for r in all_rows if r['method'] == m] for m in METHODS}
        expected = {(a, t) for a in audit_result['assets'] for t in range(1, 8)}
        for group in groups.values():
            if len(group) != config['evaluation']['assets']*7 or {(r['asset_id'], r['target_slot']) for r in group} != expected:
                raise ValueError('smoke denominator changed')
        result = {'complete': True, 'passed': True, 'source': source, 'scientific_sha256': scientific_sha(config),
            'optimizer_updates': 0, 'checkpoints': config['checkpoints'], 'policy': config['official_smoke'],
            'summaries': {m: summarize(v) for m, v in groups.items()}, 'artifacts': receipts,
            'scope': f"{config['evaluation']['assets']} preselected assets, one reference each; descriptive visual inspection, not validation-set quality.",
            'seconds': time.monotonic()-started, 'quality_accepted': False}
        write_json(output/'result.json', result)
        mirror(output/'result.json', 'result.json', config)
    return gather(result, world)[0]
