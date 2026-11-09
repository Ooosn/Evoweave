#!/usr/bin/env python3
"""Train matched standalone Stage 2 schedules without a skeleton decoder."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import gc
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch import nn
import torch.distributed as dist

from anchor_full_data import CachedFiles, make_stream, planned_microbatches, stress_groups, verify_ssd
from anchor_joint_runtime import clip_branch_gradients, evaluate, head_digest, reference_indices
from anchor_stage2_protocol import (ARMS, LONG_BATCHES, MILESTONES, PROCESS_MIB, ROOT, capacity_stage,
                                    frame_policy, read_evidence, sha256, validate_config)
from id_xyz_stages import preserve_random
from reference_flow_runtime import mirror, save
from virtual_pair_guard import verify_launch
from virtual_pair_loading import OrderedAssetPrefetch
from virtual_pair_runtime import (evaluation_rows, gather, next_update_batches, optimizer_for,
                                  set_lr, shared_deadline_due, wrap)
from virtual_query_xyz_runtime import fixed_dataset, read_entries, write_json
from rigweave.dynamic_rig.anchor_joint_binding import AnchorJointBinding, AnchorJointConfig
from rigweave.dynamic_rig.anchor_stage2 import AnchorStage2, AnchorStage2Config, AnchorStage2Encoder
from rigweave.dynamic_rig.frame_batch_sampler import FrameSampleRequest
from rigweave.dynamic_rig.surface_tokenizer import FixedQuerySurfaceTokenizer
from rigweave.dynamic_rig.virtual_frame_data import prepare_virtual_surface


def context(config, long_batch):
    base = {**config, 'training': {**config['training'], 'frame_policy': frame_policy(long_batch)}}
    entries = read_entries(base)
    train_ids = {row['asset_id'] for row in entries['train']}
    valid_ids = {row['asset_id'] for row in entries['valid']}
    if len(train_ids) != 5000 or len(valid_ids) != 967 or train_ids & valid_ids:
        raise ValueError('fixed train/validation identity differs')
    verify_ssd(config['data_cache']['root'])
    resolver = CachedFiles(config, {'data': {'manifests': config['data_cache']['full_manifests']}})
    if resolver.index_sha256 != config['data_cache']['index_sha256']:
        raise ValueError('verified SSD cache changed')
    dataset = fixed_dataset(entries['train'], base, 2, mode='train', file_resolver=resolver)
    evaluation_rows(base, entries)
    return base, entries, dataset, resolver


def export_mesh(config, output):
    spec = config['backbone']['mesh_checkpoint']
    if sha256(spec['path']) != spec['sha256']:
        raise ValueError('official mesh source checkpoint changed')
    source = {k.removeprefix('model.'): v for k, v in torch.load(
        spec['path'], map_location='cpu', mmap=True, weights_only=False)['state_dict'].items()}
    mesh = {}
    for prefix in ('mesh_encoder', 'output_proj'):
        start = prefix + '.'
        mesh[prefix] = {k[len(start):]: v for k, v in source.items() if k.startswith(start)}
        if not mesh[prefix]:
            raise ValueError('official mesh/projection tensors not found')
    path = output/'mesh_initialization.pt'
    torch.save(mesh, path)
    return mirror(path, 'audit/mesh_initialization.pt', config)


def initialize(config, arm, mesh_spec):
    import yaml
    from src.model.michelangelo.get_model import get_encoder_simplified

    spec = config['backbone']
    upstream = Path(config['runtime']['unirig_root'])
    if (subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'],
                               text=True).strip() != spec['upstream_source_commit']
            or subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain']).strip()):
        raise ValueError('UniRig mesh source changed')
    model_config = ROOT/spec['model_config']['relative_path']
    if sha256(model_config) != spec['model_config']['sha256'] or sha256(mesh_spec['path']) != mesh_spec['sha256']:
        raise ValueError('mesh-only configuration or tensor digest differs')
    surface_args = dict(yaml.safe_load(model_config.read_text())['mesh_encoder'])
    if surface_args.pop('__target__') != 'michelangelo_encoder' or surface_args.get('pretrained_path'):
        raise ValueError('unexpected source for surface encoder')
    training = config['training']
    torch.manual_seed(training['initialization_seed'])
    surface = get_encoder_simplified(**surface_args)
    architecture = AnchorStage2Config(*ARMS[arm], **config['architecture']['motion'])
    projection = nn.Linear(surface.width, architecture.dim)
    mesh = torch.load(mesh_spec['path'], map_location='cpu', mmap=True, weights_only=True)
    if set(mesh) != {'mesh_encoder', 'output_proj'}:
        raise ValueError('stage2 mesh pack contains non-mesh weights')
    surface.load_state_dict(mesh['mesh_encoder'], strict=True)
    projection.load_state_dict(mesh['output_proj'], strict=True)
    torch.manual_seed(training['initialization_seed'])
    encoder = AnchorStage2Encoder(FixedQuerySurfaceTokenizer(surface, projection), architecture)
    torch.manual_seed(training['head_seed'])
    head = AnchorJointBinding(AnchorJointConfig(**config['architecture']['head']), 'reference_transport')
    model = AnchorStage2(encoder, head)
    initial = {'arm': arm, 'spatial_layers': architecture.spatial_layers,
        'temporal_layers': architecture.temporal_layers, 'mesh_only_sha256': mesh_spec['sha256'],
        'shared_pairs_sha256': head_digest(encoder.blocks[:8]), 'head_sha256': head_digest(head),
        'parameters': sum(p.numel() for p in model.parameters()), 'skeleton_decoder_parameters': 0}
    return model, initial


def plan(base):
    stream = make_stream(base, 0, 2)
    if any(len(set(stream.order[i:i+5000])) != 5000 for i in range(0, 100000, 5000)):
        raise ValueError('each epoch must visit all fixed 5k assets exactly once')
    samples = frames = 0
    milestones = {}
    for step, (size, count) in enumerate(zip(stream.update_sizes, stream.frame_counts, strict=True), 1):
        samples += int(size)
        frames += int(size) * int(count)
        if samples in MILESTONES:
            milestones[str(samples)] = {'input_frame_visits': frames, 'updates': step}
    if samples != 100000 or step != 2090 or len(milestones) != 10:
        raise ValueError('visit, update or evaluation plan differs')
    return {'samples': samples, 'updates': step, 'input_frame_visits': frames,
            'plan_sha256': stream.plan_sha256, 'milestones': milestones}


def audit(config, output):
    base, entries, dataset, _ = context(config, 24)
    groups, _ = stress_groups(entries['train'], expected_rows=5000)
    checked = []
    for group, indices in groups.items():
        sample = dataset[FrameSampleRequest(indices[0], 24, config['training']['head_seed'])]
        if sample['frame_vertices'].shape[1] != 24 or not torch.isfinite(sample['frame_vertices']).all():
            raise ValueError('capacity geometry is invalid')
        checked.append({'group': group, 'asset_id': sample['asset_id'], 'frames': 24})
    mesh = export_mesh(config, output)
    initials = {}
    for arm in ARMS:
        model, initials[arm] = initialize(config, arm, mesh)
        del model
        gc.collect()
    for key in ('mesh_only_sha256', 'shared_pairs_sha256', 'head_sha256'):
        if len({a[key] for a in initials.values()}) != 1:
            raise ValueError('common initial tensor fingerprint differs')
    expected = plan(base)
    if expected != plan({**base, 'training': {**base['training'], 'frame_policy': frame_policy(12)}}):
        raise ValueError('physical batch changed the visit plan')
    return {'complete': True, 'passed': True, 'cuda_initialized': torch.cuda.is_initialized(),
            'optimizer_updates': 0, 'initialization': initials, 'mesh': mesh, 'plan': expected,
            'train_assets': 5000, 'validation_assets': 128, 'capacity_geometry': checked}


def build_live(config, arm, admitted, device):
    model, initial = initialize(config, arm, admitted['mesh'])
    if initial != admitted['initialization'][arm]:
        raise ValueError('actual initialization differs from the CPU audit')
    model.to(device)
    optimizer = optimizer_for(model.backbone, config)
    optimizer.add_param_group({'params': list(model.head.parameters())})
    return model, optimizer, wrap(model, device, 2), initial


def update(model, optimizer, wrapped, batches, loader, base, samples, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    lr, beta = set_lr(optimizer, samples, base)
    training = base['training']
    optimizer.param_groups[-1]['lr'] = lr * training['head_lr_multiplier']
    goal = 2 * sum(len(requests) for requests in batches)
    weight = training['coordinate_weight'] * min(1., (samples + goal) / training['warmup_assets'])
    torch.cuda.reset_peak_memory_stats(device)
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    started = time.monotonic()
    trace = []
    for i, requests in enumerate(batches):
        loaded, _ = loader.take(requests)
        surface, geometry = prepare_virtual_surface([loaded[r] for r in requests], device,
                                                     **base['surface_sampling'])
        del loaded
        ref = reference_indices(requests, base, device)
        with wrapped.no_sync() if i + 1 < len(batches) else nullcontext():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                losses = wrapped(surface, geometry, ref, weight, training['bone_weight'])
            if not bool(torch.isfinite(losses['loss_per_asset']).all()):
                raise RuntimeError('nonfinite Stage 2 loss')
            (losses['loss_per_asset'].sum() / (goal // 2)).backward()
        totals += torch.stack((losses['xyz_loss'].detach().sum(), losses['bone_loss'].detach().sum(),
            totals.new_tensor(len(requests)), totals.new_tensor(sum(r.frames for r in requests))))
        trace.append({'assets': geometry['asset_id'], 'raw_frames': geometry['selected_frames'].tolist(),
            'sampling_seeds': [r.sampling_seed for r in requests], 'reference_slots': ref.tolist()})
        del surface, geometry, losses
    clipping = clip_branch_gradients(model)
    if min(clipping['gradient_norms_before_clip'].values()) <= 0:
        raise RuntimeError('Stage 2 encoder or readout received no gradient')
    optimizer.step()
    dist.all_reduce(totals)
    if int(totals[2]) != goal:
        raise ValueError('global effective update size differs')
    return {'assets_in_update': goal, 'input_frames_in_update': int(totals[3]),
            'xyz_loss': float(totals[0] / goal), 'bone_loss': float(totals[1] / goal),
            'lr': lr, 'beta1': beta, **clipping, 'seconds': time.monotonic() - started,
            'max_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'max_reserved_bytes': torch.cuda.max_memory_reserved(device), 'microbatches': trace}


def prefetch(dataset, batches, config):
    return OrderedAssetPrefetch(dataset, batches, workers=config['runtime']['data_workers_per_rank'],
                               depth=config['runtime']['prefetch_microbatches'])


def capacity(config, output, arm, long_batch, admitted, device, rank):
    base, entries, dataset, _ = context(config, long_batch)
    model, optimizer, wrapped, initial = build_live(config, arm, admitted, device)
    groups, _ = stress_groups(entries['train'], expected_rows=5000)
    profiles = []
    for frames in config['capacity']['frames']:
        size = base['training']['frame_policy'][str(frames)]
        for label, indices in groups.items():
            steps = []
            for step in range(config['capacity']['updates_per_cohort']):
                steps.append([[FrameSampleRequest(indices[(rank + i) % len(indices)], frames,
                    20261109 + step * 1000 + part * 100 + i + rank * 50) for i in range(size)]
                    for part in range(24 // size)])
            with prefetch(dataset, (b for s in steps for b in s), config) as loader:
                for i, batches in enumerate(steps):
                    if shared_deadline_due(config, device, 2):
                        raise RuntimeError('save-stop line reached during capacity')
                    value = update(model, optimizer, wrapped, batches, loader, base, i * 48, device)
                    value.pop('microbatches')
                    profiles.append({**value, 'frames': frames, 'batch_per_rank': size,
                                     'accumulation': len(batches), 'cohort': label})
                    write_json(output/f'progress_rank_{rank}.json', {'arm': arm, 'profiles': profiles})
    return {'complete': True, 'passed': True, 'arm': arm, 'long_batch_per_rank': long_batch,
            'initialization': initial, 'profiles_by_rank': gather(profiles, 2),
            'optimizer_updates': len(profiles), 'profile_weights_discarded': True}


def train(config, output, arm, long_batch, admitted, source, device, rank):
    base, entries, dataset, resolver = context(config, long_batch)
    run = {**base, 'training': {**base['training'], 'backbone_exposure_offset': 0, 'arm': arm},
           'artifacts': {'durable_root': str(Path(config['artifacts']['durable_root'])/arm)}}
    model, optimizer, wrapped, initial = build_live(config, arm, admitted, device)
    stream = make_stream(base, rank, 2)
    torch.manual_seed(config['training']['initialization_seed'] + rank)
    checkpoints, curve = [], []
    samples = step = frames = 0
    started = time.monotonic()
    phase = output/f'progress_rank_{rank}.json'

    def validation():
        with preserve_random():
            write_json(phase, {'phase': 'evaluation', 'samples': samples, 'step': step, 'arm': arm})
            return evaluate(model, base, run, entries, output, samples, device, rank, 2,
                            file_resolver=resolver, record_input_identity=True)

    curve.append(validation())
    with (output/f'updates_rank_{rank}.jsonl').open('x', encoding='utf-8') as trace:
        for boundary in MILESTONES:
            with prefetch(dataset, planned_microbatches(base, stream, boundary), config) as loader:
                while samples < boundary:
                    if shared_deadline_due(config, device, 2):
                        if samples and (not checkpoints or checkpoints[-1]['additional_asset_exposures'] != samples):
                            save(model, optimizer, stream, run, output, initial, source,
                                 samples, frames, step, device, rank, 2)
                        raise RuntimeError('save-stop line reached before completing 100k')
                    goal, batches = next_update_batches(stream, samples, 2)
                    value = update(model, optimizer, wrapped, batches, loader, base, samples, device)
                    samples += goal
                    step += 1
                    frames += value['input_frames_in_update']
                    if value['assets_in_update'] != goal or stream.cursor * 2 != samples:
                        raise ValueError('actual visits differ from the admitted plan')
                    record = {**value, 'phase': 'train', 'samples': samples, 'step': step,
                              'input_frame_visits': frames, 'arm': arm,
                              'elapsed_seconds': time.monotonic() - started}
                    trace.write(json.dumps(record, allow_nan=False)+'\n')
                    trace.flush()
                    write_json(phase, {k: v for k, v in record.items() if k != 'microbatches'})
                    if samples == 48:
                        checkpoints.append(save(model, optimizer, stream, run, output, initial, source,
                                                samples, frames, step, device, rank, 2))
            if admitted['plan']['milestones'][str(samples)] != {'input_frame_visits': frames, 'updates': step}:
                raise ValueError('milestone visits or update counts differ from the audit')
            optimizer.zero_grad(set_to_none=True)
            checkpoints.append(save(model, optimizer, stream, run, output, initial, source,
                                    samples, frames, step, device, rank, 2))
            gc.collect()
            torch.cuda.empty_cache()
            summary = validation()
            if summary['input_sha256'] != curve[0]['input_sha256']:
                raise ValueError('validation surface/frame/GT identity changed')
            curve.append(summary)
    export = None
    if rank == 0:
        path = output/'weights_100000.pt'
        torch.save({'model': model.state_dict(), 'arm': arm, 'configuration': run,
                    'samples': samples, 'code_commit': source}, path)
        export = mirror(path, path.name, run)
        for r in range(2):
            name = f'updates_rank_{r}.jsonl'
            mirror(output/name, name, run)
    export = gather(export, 2)[0]
    return {'complete': True, 'passed': True, 'arm': arm, 'samples': samples, 'updates': step,
            'plan_sha256': stream.plan_sha256, 'input_frame_visits': frames, 'initialization': initial,
            'long_batch_per_rank': long_batch, 'checkpoints': checkpoints, 'validation': curve,
            'export': export, 'elapsed_seconds': time.monotonic() - started,
            'working_model_changed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('audit', 'capacity', 'train'))
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    source = verify_launch(args, config)
    config_sha = sha256(args.config)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.mode == 'audit':
        args.output.mkdir()
        write_json(args.output/'result.json', {**audit(config, args.output), 'source': source,
                                               'config_sha256': config_sha})
        return
    op = json.loads((ROOT/'model_training/state/current.json').read_text())['active_operation']
    admitted = read_evidence(op['stage_evidence']['audit'])
    rank = int(os.environ['RANK'])
    if int(os.environ['WORLD_SIZE']) != 2 or torch.cuda.device_count() != 2:
        raise RuntimeError('exactly two H100 ranks are required')
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(
        PROCESS_MIB * 2**20 / torch.cuda.get_device_properties(device).total_memory, device)
    dist.init_process_group('nccl', timeout=timedelta(minutes=10))
    try:
        if rank == 0:
            args.output.mkdir()
        dist.barrier(device_ids=[device.index])
        if args.mode == 'capacity':
            arm, size = next((arm, size) for arm in ARMS for size in LONG_BATCHES
                             if args.output.name == capacity_stage(arm, size))
            result = capacity(config, args.output, arm, size, admitted, device, rank)
        else:
            result = train(config, args.output, args.output.name, op['capacity_selection']['long_batch_per_rank'],
                           admitted, source, device, rank)
        result.update(source=source, config_sha256=config_sha, rank=rank,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.output/f'result_rank_{rank}.json', result)
        dist.barrier(device_ids=[device.index])
        if rank == 0:
            write_json(args.output/'result.json', result)
    except BaseException as exc:
        if args.output.is_dir():
            write_json(args.output/f'failure_rank_{rank}.json', {
                'type': type(exc).__name__, 'reason': str(exc),
                'cuda_oom': isinstance(exc, torch.OutOfMemoryError), 'rank': rank})
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
