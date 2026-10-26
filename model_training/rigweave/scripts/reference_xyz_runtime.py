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


def compose_prediction(raw, xyz, rigid, mask, reference, baseline):
    if baseline == 'copy':
        return raw
    if baseline != 'local_rigid' or rigid is None or rigid.shape != raw.shape:
        raise ValueError('residual prediction requires a matching local-rigid baseline')
    prediction = rigid.detach().float() + (raw.float() - xyz[:, None].float())
    is_reference = torch.arange(raw.shape[1], device=raw.device)[None] == reference[:, None]
    prediction = torch.where(is_reference[:, :, None, None], xyz[:, None].float(), prediction)
    return prediction.masked_fill(~mask[:, None, :, None], 0.)


def forward_head(head, packed, reference, config):
    xyz = reference_slots(packed['targets'], reference)
    raw = head(packed['features'], packed['points'], packed['normals'],
               xyz, packed['parents'], packed['mask'], reference)
    baseline = config['pilot'].get('output_baseline', 'copy')
    rigid = None
    if baseline == 'local_rigid':
        rigid, _ = local_rigid_transfer(xyz, packed['points'], reference,
            neighbors=config['pilot']['local_rigid_neighbors'])
    return compose_prediction(raw, xyz, rigid, packed['mask'], reference, baseline)


def create_head(config, device):
    if 'generator' in config:
        from rigweave.dynamic_rig.reference_coordinate_generator import ReferenceCoordinateGenerator, CoordinateGeneratorConfig
        return ReferenceCoordinateGenerator(ReferenceXYZConfig(**config['architecture']),
                    CoordinateGeneratorConfig(**config['generator'])).to(device)
    return ReferenceSkeletonXYZ(ReferenceXYZConfig(**config['architecture'])).to(device)


def training_loss(head, packed, reference, config):
    if 'generator' in config:
        result = head(packed['features'], packed['points'], packed['normals'],
            reference_slots(packed['targets'], reference), packed['parents'], packed['mask'], reference,
            targets=packed['targets'])
        return result['loss_per_asset'].mean(), {'teacher_forced_accuracy': float(result['teacher_forced_accuracy'].mean().detach())}
    prediction = forward_head(head, packed, reference, config)
    return localization_loss(prediction, packed['targets'], packed['mask'], reference).mean(), {}


def select_generator_cohort(caches, selection, config, rank, world):
    selected, filtered = {}, {}
    for split in ('train', 'valid'):
        count = config['pilot'][f'{split}_assets']
        if count % 4:
            raise ValueError('generator cohort needs four equal count strata')
        selected[split] = [row for group in ('low', 'mid', 'gt52', 'high')
                           for row in [r for r in selection[split] if r['stratum'] == group][:count // 4]]
        if len(selected[split]) != count:
            raise ValueError('insufficient cached generator assets')
        by_id = {r['asset_id']: r for r in caches[split]}
        filtered[split] = [by_id[row['asset_id']] for row in selected[split][rank::world]]
    return filtered, selected


def load_reused_cache(config, output, source, rank, world):
    spec = config['reuse_cache']
    root = Path(spec['root']).resolve(strict=True)
    selection_path = root / 'selection.json'
    if sha256(selection_path) != spec['selection_sha256']:
        raise ValueError('reused cache selection digest changed')
    selection = json.loads(selection_path.read_text())
    manifest_path = root / f'cache_manifest_rank_{rank}.json'
    if sha256(manifest_path) != spec['manifest_sha256'][rank]:
        raise ValueError('reused cache manifest digest changed')
    manifest = json.loads(manifest_path.read_text())
    binding = {'code_commit': spec['code_commit'], 'config_sha256': spec['config_sha256'],
        'backbone_checkpoint': config['backbone']['checkpoint'], 'backbone_configuration': config['backbone']['config']}
    if not manifest['complete'] or any(manifest.get(k) != v for k, v in binding.items()):
        raise ValueError('reused cache producer binding differs')
    expected = [(split, i, row) for split in ('train', 'valid')
                for i, row in enumerate(selection[split][rank::world])]
    if len(expected) != len(manifest['records']):
        raise ValueError('reused cache coverage differs')
    all_ids, all_paths = set(), set()
    for split in ('train', 'valid'):
        expected_count = {'train': 64, 'valid': 32}[split] if 'generator' in config else config['pilot'][f'{split}_assets']
        if len(selection[split]) != expected_count:
            raise ValueError('reused cache cohort count differs')
        for item in selection[split]:
            if item['asset_id'] in all_ids or item['path'] in all_paths:
                raise ValueError('reused cache asset leakage')
            all_ids.add(item['asset_id'])
            all_paths.add(item['path'])
    caches = {'train': [], 'valid': []}
    for meta, (split, index, selected) in zip(manifest['records'], expected):
        path = Path(meta['path']).resolve(strict=True)
        wanted = root / f'cache_rank_{rank}' / f'{split}_{index:03d}.pt'
        if path != wanted or not path.is_relative_to(root) or sha256(path) != meta['sha256']:
            raise ValueError('reused cache payload path or digest changed')
        record = torch.load(path, map_location='cpu', weights_only=False)
        frames, joints = config['pilot']['frames'], selected['joint_count']
        features, points = record['features'], record['points']
        if (record['source_binding'] != binding or record['asset_id'] != selected['asset_id']
                or meta['asset_id'] != selected['asset_id'] or meta['split'] != split
                or meta['joints'] != joints or record['selected_frames'].tolist() != meta['frames']
                or len(meta['frames']) != frames or len(set(meta['frames'])) != frames
                or set(meta['frames']) & set(record['heldout_raw_ids'])
                or features.ndim != 3 or features.shape[0] != frames
                or features.shape[-1] != config['architecture']['input_dim']
                or features.dtype != torch.bfloat16 or features.requires_grad
                or points.shape != (*features.shape[:2], 3) or record['normals'].shape != points.shape
                or record['targets'].shape != (frames, joints, 3) or record['parents'].shape != (joints,)
                or record['diagonal'].shape != (frames,) or not bool((record['diagonal'] > 0).all())):
            raise ValueError('reused cache tensor/geometry contract differs')
        for key in ('points', 'normals', 'targets', 'diagonal', 'normalization_center', 'normalization_scale'):
            if not bool(torch.isfinite(record[key]).all()):
                raise ValueError('reused cache contains nonfinite geometry')
        validate_reference(record['targets'][0:1], record['parents'][None],
                           torch.ones(1, joints, dtype=torch.bool), max_joints=config['architecture']['max_joints'])
        if split == 'valid' and set(record['predicted_references']) != set(config['pilot']['evaluation_reference_slots']):
            raise ValueError('reused natural reference coverage differs')
        caches[split].append(record)
    initial = {'actual_initialization': config['backbone']['checkpoint'], 'backbone_frozen': True,
        'backbone_not_reloaded': True, 'frozen_cache_reuse': spec, 'all_local_payload_digests_verified': True}
    write_json(output / f'cache_manifest_rank_{rank}.json', {**manifest, 'reused_by': source_binding(config, source),
        'source_manifest_sha256': spec['manifest_sha256'][rank], 'read_only_reuse': True})
    if rank == 0:
        write_json(output / 'selection.json', selection)
        write_json(output / 'initialization.json', initial)
    return caches, selection, initial


def capacity(config, output, source, device, rank, world):
    # The largest-joint pilot asset bounds this readout's J dimension. Each rank
    # uses its complete eight-pose geometry, with the inherited dense sampler.
    backbone = None
    if 'reuse_cache' in config:
        caches, selection, initial = load_reused_cache(config, output, source, rank, world)
        record = max(caches['train'] + caches['valid'], key=lambda r: r['parents'].numel())
        row, meta = record, {'joint_count': record['parents'].numel()}
    else:
        backbone, tokenizer, base_config, initial = load_frozen_backbone(config, device, rank)
        selected, selection = cohort(config, base_config)
        combined = list(zip(selected['train'], selection['train'])) + list(zip(selected['valid'], selection['valid']))
        ordered = sorted(combined, key=lambda x: x[1]['joint_count'], reverse=True)
        row, meta = ordered[rank]
        dataset = fixed_dataset([row], base_config, config['pilot']['frames'], mode='train')
        record = extract_asset(backbone, tokenizer, dataset[0], base_config, config, predicted_references=False)
    packed = pack([record] * config['pilot']['batch_per_rank'], device)
    torch.manual_seed(config['pilot']['seed'])
    head = create_head(config, device)
    wrapped = DistributedDataParallel(head, device_ids=[device.index])
    optimizer = torch.optim.AdamW(head.parameters(), lr=config['pilot']['learning_rate'])
    reference = torch.zeros(config['pilot']['batch_per_rank'], dtype=torch.long, device=device)
    losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, _ = training_loss(wrapped, packed, reference, config)
        if not bool(torch.isfinite(loss)):
            raise ValueError('nonfinite capacity loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    condition = head.conditioner if 'generator' in config else head
    modules = [('reference', condition.reference_blocks), ('target', condition.target_blocks),
               ('geometry', condition.target_geometry), ('output', head.decoder if 'generator' in config else head.output)]
    gradients = {name: any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in module.parameters())
                 for name, module in modules}
    if not all(gradients.values()) or (backbone is not None and any(p.grad is not None for p in backbone.parameters())):
        raise ValueError('readout gradient or frozen-backbone contract failed')
    free_generation = None
    if 'generator' in config:
        head.eval()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            generated = head.generate(packed['features'], packed['points'], packed['normals'],
                reference_slots(packed['targets'], reference), packed['parents'], packed['mask'], reference)
        if generated.shape != packed['targets'].shape or not bool(torch.isfinite(generated).all()):
            raise ValueError('capacity free generation failed')
        free_generation = {'passed': True, 'tokens_per_target': 3 * meta['joint_count'], 'KV_cache': True,
                           'shape': list(generated.shape), 'no_geometric_baseline': True}
    record = {'rank': rank, 'asset_id': row['asset_id'], 'joints': meta['joint_count'],
              'frames': config['pilot']['frames'], 'batch': config['pilot']['batch_per_rank'],
              'losses': losses, 'gradient_groups': gradients, 'free_generation': free_generation, 'last_gradient_norm': float(norm),
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
    teacher_rows = []
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
                        if 'generator' in config:
                            predicted = head.generate(packed['features'], packed['points'], packed['normals'], xyz, parents, mask, reference)
                            if reference_kind == 'GT':
                                teacher = head(packed['features'], packed['points'], packed['normals'], xyz, parents, mask, reference,
                                               targets=packed['targets'], return_predictions=True)
                                selected = torch.arange(config['pilot']['frames']) != frame
                                errors = torch.linalg.vector_norm(teacher['teacher_prediction'][0].float().cpu() - record['targets'], dim=-1)
                                normalized = errors / record['diagonal'][:, None]
                                from rigweave.dynamic_rig.reference_coordinate_generator import quantize_xyz, dequantize_xyz
                                quantized = dequantize_xyz(quantize_xyz(record['targets'], config['generator']['bins']), config['generator']['bins'])
                                floor = torch.linalg.vector_norm(quantized - record['targets'], dim=-1) / record['diagonal'][:, None]
                                teacher_rows.append({**common, 'teacher_forced_CE': float(teacher['loss_per_asset'][0]),
                                    'teacher_forced_accuracy': float(teacher['teacher_forced_accuracy'][0]),
                                    'teacher_forced_normalized_mpjpe': float(normalized[selected].mean()),
                                    'quantized_GT_normalized_mpjpe': float(floor[selected].mean())})
                        else:
                            predicted = head(packed['features'], packed['points'], packed['normals'], xyz, parents, mask, reference)
                    rigid, degenerate = local_rigid_transfer(xyz, packed['points'], reference,
                        neighbors=config['pilot']['local_rigid_neighbors'])
                    if 'generator' not in config:
                        predicted = compose_prediction(predicted, xyz, rigid, mask, reference,
                            config['pilot'].get('output_baseline', 'copy'))
                    methods = {'learned': predicted[0], 'copy': xyz[0, None].expand(config['pilot']['frames'], -1, -1),
                               'local_rigid': rigid[0]}
                    if step == config['pilot'].get('updates', 800) and record is records[0]:
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
    if teacher_rows:
        write_json(output / f'evaluation_{step:04d}_teacher_rank_{rank}.json', teacher_rows)
    if examples:
        write_json(output / f'examples_{step:04d}_rank_{rank}.json', examples)
    all_rows = [row for part in gather(rows, world) for row in part]
    all_teacher = [row for part in gather(teacher_rows, world) for row in part]
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
            'output_baseline': config['pilot'].get('output_baseline', 'copy'),
            'teacher_forced': {split: {metric: float(np.mean([r[metric] for r in all_teacher if r['split'] == split]))
                for metric in ('teacher_forced_CE', 'teacher_forced_accuracy', 'teacher_forced_normalized_mpjpe', 'quantized_GT_normalized_mpjpe')}
                for split in ('train', 'valid')} if all_teacher else None,
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


def extract_caches(config, output, source, device, rank, world):
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
    return caches, selection, initial


def train(config, output, source, device, rank, world):
    validate_config(config)
    if 'reuse_cache' in config:
        caches, selection, initial = load_reused_cache(config, output, source, rank, world)
    else:
        caches, selection, initial = extract_caches(config, output, source, device, rank, world)
    if 'generator' in config:
        caches, selection = select_generator_cohort(caches, selection, config, rank, world)
        if rank == 0:
            write_json(output / 'selection.json', selection)
    torch.manual_seed(config['pilot']['seed'])
    head = create_head(config, device)
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
            loss, training_metrics = training_loss(wrapped, packed, reference, config)
        if not bool(torch.isfinite(loss)):
            raise ValueError('nonfinite pilot loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        row = {'step': step, 'loss': float(loss.detach()), 'lr': lr, 'gradient_norm': float(norm),
               'asset_ids': [r['asset_id'] for r in records], 'reference_slots': reference.tolist(),
               'local_asset_visits': step * len(records), 'elapsed_seconds': time.monotonic() - started, **training_metrics}
        with trace.open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        if step <= 3 or step % 50 == 0:
            write_json(output / f'progress_rank_{rank}.json', {'phase': 'head_training', **row,
                'observed_utc': datetime.now(timezone.utc).isoformat()})
            if rank == 0:
                print(json.dumps({'event': 'update', **row}), flush=True)
        if step in (steps // 2, steps):
            save_head(head, optimizer, generator, config, selection, step, source, output, rank, world)
            evaluate(head, caches, config, device, rank, world, step, output)
    result = {'complete': True, 'passed': True, 'mode': 'train', 'updates': steps,
              'asset_visits': steps * config['pilot']['batch_per_rank'] * world,
              'known_reference_target_visits': steps * config['pilot']['batch_per_rank'] * world * (config['pilot']['frames'] - 1),
              'backbone_frozen': True, 'output_baseline': config['pilot'].get('output_baseline', 'copy'),
              'quality_accepted': False, **source_binding(config, source)}
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
