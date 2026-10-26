"""Bounded frozen-backbone readout pilot, with GT and natural-predicted references."""
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from reference_xyz_protocol import load_backbone_config, select_cohort, sha256, validate_config
from virtual_pair_guard import deadline_due
from virtual_query_xyz_runtime import build_model, fixed_dataset, read_entries, write_json
from rigweave.dynamic_rig.reference_skeleton_xyz import (
    ReferenceSkeletonXYZ, ReferenceXYZConfig, localization_loss, local_rigid_transfer,
    reference_slots, validate_reference,
)
from rigweave.dynamic_rig.virtual_frame_data import prepare_virtual_surface
from rigweave.dynamic_rig.virtual_path_ablation import VirtualPathAblationRig
from rigweave.dynamic_rig.skeleton_topology_metrics import _topology_metrics


def gather(value, world):
    values = [None] * world
    dist.all_gather_object(values, value)
    return values


def check_deadline(config, device):
    due = torch.tensor(int(deadline_due(config)), device=device)
    dist.all_reduce(due, op=dist.ReduceOp.MAX)
    if bool(due):
        raise RuntimeError('pilot reached its declared resource deadline')


def source_binding(config, source):
    return {'code_commit': source, 'config_sha256': config['_config_sha256'],
            'backbone_checkpoint': config['backbone']['checkpoint'],
            'backbone_configuration': config['backbone']['config']}


def load_frozen_backbone(config, device, rank):
    spec = config['backbone']['checkpoint']
    if rank == 0 and sha256(spec['path']) != spec['sha256']:
        raise ValueError('retained backbone checkpoint digest changed')
    dist.barrier(device_ids=[device.index])
    base_config = load_backbone_config(config)
    base, tokenizer, _, official = build_model(base_config, device)
    model = VirtualPathAblationRig(base, virtual_enabled=False)
    payload = torch.load(spec['path'], map_location='cpu', mmap=True, weights_only=False)
    if (payload['variant'] != 'fraction20_40_query' or payload['asset_exposures'] != 20000
            or payload['code_commit'] != 'a0bda62bb8fba70320310b063da8c7898ddaf441'):
        raise ValueError('wrong retained backbone provenance')
    model.load_state_dict(payload['model'], strict=True)
    report = {'actual_initialization': spec, 'backbone_frozen': True,
              'donor_model_sha256': payload['model_sha256'],
              'donor_producer_commit': payload['code_commit'],
              'strict_complete_state_load': True,
              'constructor_official_checkpoint_sha256': official['checkpoint_sha256']}
    del payload
    model.requires_grad_(False)
    model.eval()
    return model, tokenizer, base_config, report


def cohort(config, base_config):
    entries = read_entries(base_config)
    selected, records = {}, {}
    for split in ('train', 'valid'):
        selected[split], records[split] = select_cohort(entries[split], split,
            config['pilot'][f'{split}_assets'], config['pilot']['seed'])
    if ({r['asset_id'] for r in selected['train']} & {r['asset_id'] for r in selected['valid']}
            or {r['path'] for r in selected['train']} & {r['path'] for r in selected['valid']}):
        raise ValueError('training and validation asset leakage')
    return selected, records


@torch.no_grad()
def generate_reference(model, tokenizer, condition):
    from rigweave.dynamic_rig.virtual_query_xyz import original_xyz_vocabulary_processor
    start = torch.tensor([tokenizer.bos, tokenizer.cls_name_to_token('articulationxl')], device=condition.device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        generated = model.generate_condition(condition, start, eos_token=tokenizer.eos, max_new_tokens=2048,
            logits_processor=original_xyz_vocabulary_processor(tokenizer, start))
    record = {'success': False, 'status': generated['status'], 'tokens': generated['tokens']}
    if generated['success']:
        try:
            parsed = tokenizer.detokenize(np.asarray(generated['tokens'], dtype=np.int64))
            raw_parents = parsed.parents if getattr(parsed, 'parents', None) is not None else parsed._get_parents()
            parents = [(-1 if value is None else int(value)) for value in raw_parents]
            xyz = torch.as_tensor(np.asarray(parsed.joints), dtype=torch.float32)
            parent = torch.tensor(parents, dtype=torch.long)[None]
            validate_reference(xyz[None], parent, torch.ones_like(parent, dtype=torch.bool))
            record.update(success=True, xyz=xyz, parents=parent[0])
        except Exception as error:
            record.update(status='invalid_reference_tree', error=repr(error))
    return record


@torch.no_grad()
def extract_asset(model, tokenizer, batch, base_config, config, *, predicted_references):
    device = next(model.parameters()).device
    surface, geometry = prepare_virtual_surface([batch], device, **base_config['surface_sampling'])
    with torch.autocast('cuda', dtype=torch.bfloat16):
        features, virtual = model.encode(surface)
        conditions = model.conditions(features, virtual) if predicted_references else None
    vertices = batch['frame_vertices'][0]
    diagonal = torch.linalg.vector_norm(vertices.amax(1) - vertices.amin(1), dim=-1)
    if not bool((diagonal > 0).all()):
        raise ValueError('degenerate input mesh bounding box')
    record = {
        'asset_id': batch['asset_id'], 'features': features[0].cpu().to(torch.bfloat16),
        'points': surface['query_points'][0].cpu().float(),
        'normals': surface['query_normals'][0].cpu().float(),
        'targets': geometry['frame_target_joints'][0].cpu().float(),
        'parents': geometry['target_parents'][0].cpu(), 'diagonal': diagonal.cpu().float(),
        'selected_frames': geometry['selected_frames'][0].cpu(),
        'normalization_center': geometry['query_center'][0].cpu(),
        'normalization_scale': geometry['query_scale'][0].cpu(),
        'heldout_raw_ids': batch['heldout_raw_ids'], 'predicted_references': {},
    }
    if set(record['selected_frames'].tolist()) & set(record['heldout_raw_ids']):
        raise ValueError('heldout raw frames entered the pilot cache')
    if predicted_references:
        for frame in config['pilot']['evaluation_reference_slots']:
            record['predicted_references'][frame] = generate_reference(model, tokenizer, conditions[:, frame])
    return record


def pack(records, device):
    count = max(record['parents'].numel() for record in records)
    batch, frames = len(records), records[0]['targets'].shape[0]
    targets = torch.zeros(batch, frames, count, 3, device=device)
    parents = torch.full((batch, count), -2, dtype=torch.long, device=device)
    mask = torch.zeros_like(parents, dtype=torch.bool)
    for index, record in enumerate(records):
        size = record['parents'].numel()
        targets[index, :, :size] = record['targets'].to(device)
        parents[index, :size] = record['parents'].to(device)
        mask[index, :size] = True
    return {'features': torch.stack([r['features'] for r in records]).to(device),
            'points': torch.stack([r['points'] for r in records]).to(device),
            'normals': torch.stack([r['normals'] for r in records]).to(device),
            'targets': targets, 'parents': parents, 'mask': mask}


def forward_head(head, packed, reference):
    return head(packed['features'], packed['points'], packed['normals'],
                reference_slots(packed['targets'], reference), packed['parents'], packed['mask'], reference)


def capacity(config, output, source, device, rank, world):
    backbone, tokenizer, base_config, initial = load_frozen_backbone(config, device, rank)
    selected, selection = cohort(config, base_config)
    # The largest-joint pilot asset bounds this readout's J dimension. Each rank
    # uses its complete eight-pose geometry, with the inherited dense sampler.
    combined = list(zip(selected['train'], selection['train'])) + list(zip(selected['valid'], selection['valid']))
    ordered = sorted(combined, key=lambda x: x[1]['joint_count'], reverse=True)
    row, meta = ordered[rank]
    dataset = fixed_dataset([row], base_config, config['pilot']['frames'], mode='train')
    record = extract_asset(backbone, tokenizer, dataset[0], base_config, config, predicted_references=False)
    packed = pack([record] * config['pilot']['batch_per_rank'], device)
    torch.manual_seed(config['pilot']['seed'])
    head = ReferenceSkeletonXYZ(ReferenceXYZConfig(**config['architecture'])).to(device)
    wrapped = DistributedDataParallel(head, device_ids=[device.index])
    optimizer = torch.optim.AdamW(head.parameters(), lr=config['pilot']['learning_rate'])
    reference = torch.zeros(config['pilot']['batch_per_rank'], dtype=torch.long, device=device)
    losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = forward_head(wrapped, packed, reference)
            loss = localization_loss(prediction, packed['targets'], packed['mask'], reference).mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError('nonfinite capacity loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    gradients = {name: any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in module.parameters())
                 for name, module in (('reference', head.reference_blocks), ('target', head.target_blocks),
                                      ('geometry', head.target_geometry), ('output', head.output))}
    if not all(gradients.values()) or any(p.grad is not None for p in backbone.parameters()):
        raise ValueError('readout gradient or frozen-backbone contract failed')
    record = {'rank': rank, 'asset_id': row['asset_id'], 'joints': meta['joint_count'],
              'frames': config['pilot']['frames'], 'batch': config['pilot']['batch_per_rank'],
              'losses': losses, 'gradient_groups': gradients, 'last_gradient_norm': float(norm),
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}
    reports = gather(record, world)
    return {'complete': True, 'passed': True, 'mode': 'capacity', 'ranks': reports,
            'initialization': initial, 'selection': selection, **source_binding(config, source)}


def row_metrics(prediction, predicted_parents, truth, truth_parents, diagonal, *, paired_ids):
    prediction = prediction.detach().float().cpu().numpy()
    truth = truth.detach().float().cpu().numpy()
    if not np.isfinite(prediction).all():
        raise ValueError('nonfinite predicted joint coordinates')
    metrics = _topology_metrics(prediction, predicted_parents, truth, truth_parents)
    distances = np.linalg.norm(prediction[:, None] - truth[None], axis=-1)
    result = {'edge_f1': float(metrics['edge_f1']),
              'normalized_j2j': float(.5 * (distances.min(0).mean() + distances.min(1).mean()) / diagonal)}
    if paired_ids:
        errors = np.linalg.norm(prediction - truth, axis=-1) / diagonal
        result.update(normalized_mpjpe=float(errors.mean()), pck_005=float((errors <= .05).mean()))
    return result


@torch.no_grad()
def evaluate(head, caches, config, device, rank, world, step, output):
    head.eval()
    rows = []
    examples = []
    for split, records in caches.items():
        for record in records:
            packed = pack([record], device)
            for frame in config['pilot']['evaluation_reference_slots']:
                reference = torch.tensor([frame], dtype=torch.long, device=device)
                for reference_kind in (('GT', 'predicted') if split == 'valid' else ('GT',)):
                    incoming = (record['predicted_references'][frame] if reference_kind == 'predicted' else
                                {'success': True, 'xyz': record['targets'][frame], 'parents': record['parents']})
                    common = {'split': split, 'asset_id': record['asset_id'], 'reference_kind': reference_kind,
                              'reference_slot': frame, 'gt_joint_count': record['parents'].numel()}
                    if not incoming['success']:
                        for target in range(config['pilot']['frames']):
                            if target != frame:
                                for method in ('learned', 'copy', 'local_rigid'):
                                    rows.append({**common, 'target_slot': target, 'method': method,
                                        'reference_success': False, 'edge_f1': 0., 'normalized_j2j': None,
                                        'reference_failure': incoming['status']})
                        continue
                    xyz = incoming['xyz'][None].to(device)
                    parents = incoming['parents'][None].to(device)
                    mask = torch.ones_like(parents, dtype=torch.bool)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        predicted = head(packed['features'], packed['points'], packed['normals'], xyz, parents, mask, reference)
                    rigid, degenerate = local_rigid_transfer(xyz, packed['points'], reference,
                        neighbors=config['pilot']['local_rigid_neighbors'])
                    methods = {'learned': predicted[0], 'copy': xyz[0, None].expand(config['pilot']['frames'], -1, -1),
                               'local_rigid': rigid[0]}
                    if step == 800 and record is records[0]:
                        examples.append({**common, 'anchor_points': record['points'].tolist(),
                            'truth_xyz': record['targets'].tolist(), 'truth_parents': record['parents'].tolist(),
                            'reference_xyz': incoming['xyz'].tolist(), 'reference_parents': incoming['parents'].tolist(),
                            'predicted_xyz': predicted[0].float().cpu().tolist(), 'rigid_xyz': rigid[0].cpu().tolist()})
                    for target in range(config['pilot']['frames']):
                        if target == frame:
                            continue
                        for method, positions in methods.items():
                            scores = row_metrics(positions[target], incoming['parents'].tolist(),
                                record['targets'][target], record['parents'].tolist(), float(record['diagonal'][target]),
                                paired_ids=reference_kind == 'GT')
                            rows.append({**common, 'target_slot': target, 'method': method, 'reference_success': True,
                                'predicted_joint_count': parents.shape[1],
                                'rigid_degenerate_joints': int(degenerate[0, target].sum()) if method == 'local_rigid' else None,
                                **scores})
    path = output / f'evaluation_{step:04d}_rank_{rank}.json'
    write_json(path, rows)
    if examples:
        write_json(output / f'examples_{step:04d}_rank_{rank}.json', examples)
    all_rows = [row for part in gather(rows, world) for row in part]
    if rank == 0:
        groups = {}
        for split, reference in (('train', 'GT'), ('valid', 'GT'), ('valid', 'predicted')):
            for subgroup in ('all', 'non52', 'gt52'):
                group = [r for r in all_rows if r['split'] == split and r['reference_kind'] == reference
                         and (subgroup == 'all' or (r['gt_joint_count'] == 52) == (subgroup == 'gt52'))]
                for method in ('learned', 'copy', 'local_rigid'):
                    selected = [r for r in group if r['method'] == method]
                    summary = {'views': len(selected), 'assets': len({r['asset_id'] for r in selected}),
                               'failed_reference_views': sum(not r['reference_success'] for r in selected)}
                    for key in ('edge_f1', 'normalized_j2j', 'normalized_mpjpe', 'pck_005'):
                        values = [r[key] for r in selected if r.get(key) is not None]
                        summary[key] = float(np.mean(values)) if values else None
                    groups[f'{split}/{reference}/{subgroup}/{method}'] = summary
        write_json(output / f'evaluation_{step:04d}_summary.json', {
            'complete': True, 'updates': step, 'groups': groups, 'GT_reference_is_oracle': True,
            'predicted_reference_failures_F1_zero': True, 'j2j_scope': 'successful references only',
            'reference_pose_excluded': True, 'position_unit': 'fraction of target mesh bounding-box diagonal',
            'pck_threshold': .05, 'selection_distribution': 'equal low<=20,mid21..51,exact52,high>=53 strata',
            'frozen_readout_pilot_not_joint_training': True,
            'row_artifacts': [{'path': str(output / f'evaluation_{step:04d}_rank_{r}.json'),
                               'sha256': sha256(output / f'evaluation_{step:04d}_rank_{r}.json')} for r in range(world)]})
        print(json.dumps({'event': 'evaluation', 'updates': step,
            'valid_GT_learned': groups['valid/GT/all/learned'],
            'valid_GT_rigid': groups['valid/GT/all/local_rigid'],
            'valid_predicted_learned': groups['valid/predicted/all/learned']}), flush=True)
    head.train()
    return groups if rank == 0 else None


def save_head(head, optimizer, generator, config, selection, step, source, output, rank, world):
    ranks = gather({'rank': rank, 'generator_state': generator.get_state(),
                    'torch_state': torch.get_rng_state(), 'cuda_state': torch.cuda.get_rng_state()}, world)
    if rank:
        return
    path = output / f'head_{step:04d}.pt'
    payload = {'head': head.state_dict(), 'optimizer': optimizer.state_dict(), 'rank_states': ranks,
               'updates': step, 'configuration': config, 'selection': selection, **source_binding(config, source)}
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)
    digest = sha256(path)
    durable = Path(config['artifacts']['durable_root']) / path.name
    durable.parent.mkdir(parents=True, exist_ok=True)
    if durable.exists():
        raise RuntimeError('refuse to overwrite retained head checkpoint')
    temporary = durable.with_suffix('.tmp')
    shutil.copyfile(path, temporary)
    if sha256(temporary) != digest:
        raise RuntimeError('head checkpoint durable digest mismatch')
    temporary.replace(durable)
    write_json(path.with_suffix('.json'), {'complete': True, 'path': str(path), 'durable_path': str(durable),
        'sha256': digest, 'bytes': path.stat().st_size, 'both_copies_verified': True, 'updates': step,
        **source_binding(config, source)})


def train(config, output, source, device, rank, world):
    validate_config(config)
    backbone, tokenizer, base_config, initial = load_frozen_backbone(config, device, rank)
    selected, selection = cohort(config, base_config)
    if rank == 0:
        write_json(output / 'selection.json', selection)
        write_json(output / 'initialization.json', initial)
    cache_dir = output / f'cache_rank_{rank}'
    cache_dir.mkdir(exist_ok=False)
    caches = {'train': [], 'valid': []}
    cache_manifest = []
    for split in ('train', 'valid'):
        local = selected[split][rank::world]
        dataset = fixed_dataset(local, base_config, config['pilot']['frames'], mode='train' if split == 'train' else 'seen_eval')
        for index in range(len(dataset)):
            check_deadline(config, device)
            record = extract_asset(backbone, tokenizer, dataset[index], base_config, config,
                                   predicted_references=split == 'valid')
            record['source_binding'] = source_binding(config, source)
            path = cache_dir / f'{split}_{index:03d}.pt'
            torch.save(record, path)
            cache_manifest.append({'path': str(path), 'sha256': sha256(path), 'split': split,
                'asset_id': record['asset_id'], 'frames': record['selected_frames'].tolist(),
                'joints': record['parents'].numel(), 'features_dtype': str(record['features'].dtype)})
            caches[split].append(record)
            write_json(output / f'progress_rank_{rank}.json', {'phase': 'cache', 'split': split,
                'completed': index + 1, 'total': len(dataset), 'observed_utc': datetime.now(timezone.utc).isoformat()})
    write_json(output / f'cache_manifest_rank_{rank}.json', {'complete': True, 'records': cache_manifest,
        **source_binding(config, source)})
    del backbone, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(config['pilot']['seed'])
    head = ReferenceSkeletonXYZ(ReferenceXYZConfig(**config['architecture'])).to(device)
    wrapped = DistributedDataParallel(head, device_ids=[device.index])
    optimizer = torch.optim.AdamW(head.parameters(), lr=config['pilot']['learning_rate'],
                                  weight_decay=config['pilot']['weight_decay'])
    generator = torch.Generator().manual_seed(config['pilot']['seed'] + rank)
    steps = config['pilot']['updates']
    trace = output / f'trace_rank_{rank}.jsonl'
    evaluate(head, caches, config, device, rank, world, 0, output)
    started = time.monotonic()
    for step in range(1, steps + 1):
        check_deadline(config, device)
        indices = torch.randint(len(caches['train']), (config['pilot']['batch_per_rank'],), generator=generator).tolist()
        records = [caches['train'][i] for i in indices]
        packed = pack(records, device)
        reference = torch.randint(config['pilot']['frames'], (len(records),), generator=generator).to(device)
        lr = config['pilot']['learning_rate'] * (step / 20 if step <= 20 else
             .1 + .9 * .5 * (1 + math.cos(math.pi * (step - 20) / (steps - 20))))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = forward_head(wrapped, packed, reference)
            loss = localization_loss(prediction, packed['targets'], packed['mask'], reference).mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError('nonfinite pilot loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        row = {'step': step, 'loss': float(loss.detach()), 'lr': lr, 'gradient_norm': float(norm),
               'asset_ids': [r['asset_id'] for r in records], 'reference_slots': reference.tolist(),
               'local_asset_visits': step * len(records), 'elapsed_seconds': time.monotonic() - started}
        with trace.open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        if step <= 3 or step % 50 == 0:
            write_json(output / f'progress_rank_{rank}.json', {'phase': 'head_training', **row,
                'observed_utc': datetime.now(timezone.utc).isoformat()})
            if rank == 0:
                print(json.dumps({'event': 'update', **row}), flush=True)
        if step in (400, 800):
            save_head(head, optimizer, generator, config, selection, step, source, output, rank, world)
            evaluate(head, caches, config, device, rank, world, step, output)
    result = {'complete': True, 'passed': True, 'mode': 'train', 'updates': steps,
              'asset_visits': steps * config['pilot']['batch_per_rank'] * world,
              'known_reference_target_visits': steps * config['pilot']['batch_per_rank'] * world * (config['pilot']['frames'] - 1),
              'backbone_frozen': True, 'quality_accepted': False, **source_binding(config, source)}
    if rank == 0:
        durable = Path(config['artifacts']['durable_root'])
        for path in output.glob('*.json*'):
            target = durable / path.name
            if target.exists():
                raise RuntimeError('refuse to overwrite durable pilot artifacts')
            shutil.copyfile(path, target)
            if sha256(target) != sha256(path):
                raise RuntimeError('pilot artifact mirror digest mismatch')
    return result
