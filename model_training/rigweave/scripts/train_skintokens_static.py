#!/usr/bin/env python3
"""Native SkinTokens skeleton-only fine-tuning on one pose per asset visit."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import random
import tarfile

import numpy as np
import torch
from torch import nn
import torch.distributed as dist

from anchor_full_data import CachedFiles, stress_groups, verify_ssd
from id_xyz_stages import preserve_random, random_state
from reference_flow_protocol import sample_seed
from reference_flow_runtime import mirror
from skintokens_static_protocol import MILESTONES, sha256, validate_config
from stage1_feedback import score_generation, summarize_rows
from stage1_surface_repair_runtime import canonical_identity
from tokenrig_decoder import build_tokenrig_conditioning, build_tokenrig_decoder
from virtual_fusion_normal_runtime import MatchedFrameStream
from virtual_pair_guard import ROOT, verify_launch
from virtual_pair_loading import OrderedAssetPrefetch
from virtual_pair_runtime import evaluation_rows, gather, optimizer_for, set_lr, shared_deadline_due, wrap
from virtual_query_xyz_runtime import fixed_dataset, read_entries, write_json
from rigweave.dynamic_rig.frame_batch_sampler import FrameSampleRequest
from rigweave.dynamic_rig.virtual_frame_data import prepare_virtual_surface, read_virtual_manifest
from rigweave.dynamic_rig.virtual_query_xyz import (
    VirtualQueryXYZRig, original_xyz_vocabulary_processor, sequence_ce,
)
from rigweave.dynamic_rig.virtual_query_xyz_data import _validate_main_sequence


class SinglePoseSkeleton(nn.Module):
    """Released geometry, projection and skeleton AR, without a temporal or XYZ head."""
    decoder_condition_length = VirtualQueryXYZRig.decoder_condition_length
    decoder_prefix = VirtualQueryXYZRig.decoder_prefix
    main_logits = VirtualQueryXYZRig.main_logits
    generate_condition = VirtualQueryXYZRig.generate_condition
    decoder_condition_tokens = None

    def __init__(self, encoder, projection, decoder):
        super().__init__()
        self.mesh_encoder = encoder
        self.decoder_condition_projection = projection
        self.xyz_decoder = decoder

    def encode(self, points, normals):
        if points.ndim != 3 or points.shape[-1] != 3 or points.shape != normals.shape:
            raise ValueError('native encoder accepts one point cloud per asset')
        _, latents, _, _ = self.mesh_encoder.encode_latents(pc=points, feats=normals)
        if latents.shape != (points.shape[0], 512, 512):
            raise ValueError('released geometry prefix changed')
        return latents

    def forward(self, points, normals, ids, mask):
        condition = self.encode(points, normals)
        logits = self.main_logits(condition, ids[:, :-1], mask[:, :-1])
        return sequence_ce(logits, ids[:, 1:].masked_fill(~mask[:, 1:], -100))


def initialize(config, device, *, eager=False, load_weights=True):
    spec = {**config['tokenrig_decoder'], 'eager': eager}
    encoder, projection, mesh = build_tokenrig_conditioning(spec, 512, load_weights=load_weights)
    decoder, tokenizer, ar = build_tokenrig_decoder(spec, load_weights=load_weights)
    if not eager:
        decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    ar['activation_recomputation'] = not eager
    model = SinglePoseSkeleton(encoder, projection, decoder).to(device)
    return model, tokenizer, {'mesh': mesh, 'decoder': ar, 'project_checkpoint_loaded': False,
        'stage2_enabled': False, 'trainable_parameters': sum(
            p.numel() for p in model.parameters() if p.requires_grad)}


def make_stream(config, rank):
    spec, train = config['train_manifest'], config['training']
    return MatchedFrameStream(range(spec['rows']), rank=rank, world_size=2,
        seed=train['stream_seed'], target_assets=train['additional_assets'],
        effective_assets=train['effective_assets_per_update'], frame_policy=train['frame_policy'],
        manifest_sha256=spec['sha256'], min_frames=1, max_frames=1,
        update_boundaries=train['checkpoint_assets'])


def data_context(config):
    recipe = config['base_recipe']
    path = ROOT / recipe['path']
    if sha256(path) != recipe['sha256'] or sha256(config['data_cache']['index']) != config['data_cache']['index_sha256']:
        raise ValueError('base recipe or verified SSD cache index changed')
    base = json.loads(path.read_text())
    entries = read_entries(base)
    spec = config['train_manifest']
    train = read_virtual_manifest(spec['path'], expected_count=spec['rows'], expected_sha256=spec['sha256'])
    full = {row['asset_id']: row for row in entries['train']}
    if len({row['asset_id'] for row in train}) != 5000 or any(full.get(row['asset_id']) != row for row in train):
        raise ValueError('fixed 5k rows differ from the pinned RT dataset')
    if {r['asset_id'] for r in train} & {r['asset_id'] for r in entries['valid']}:
        raise ValueError('training and validation overlap')
    resolver = CachedFiles(config, base)
    dataset = fixed_dataset(train, base, 1, mode='train', file_resolver=resolver)
    base['optimization']['schedule_samples'] = config['training']['additional_assets']
    return base, dataset, evaluation_rows(base, entries)


def panel_specs(config, base, selected, *, verify_archive=False):
    archive = config['evaluation']['panel_archive']
    if verify_archive and sha256(archive['path']) != archive['sha256']:
        raise ValueError('retained 128-asset input archive changed')
    specs = {}
    with tarfile.open(archive['path'], 'r:') as tar:
        for member in tar:
            if not member.name.endswith('.json'):
                continue
            if not member.isfile():
                raise ValueError('canonical archive metadata must be regular files')
            spec = json.load(tar.extractfile(member))
            key = spec['identity']['asset_id']
            if key in specs:
                raise ValueError('duplicate canonical asset')
            specs[key] = spec
    if set(specs) != {row['asset_id'] for row in selected} or len(selected) != config['evaluation']['assets']:
        raise ValueError('canonical archive and fixed panel differ')
    root = Path(config['data_cache']['root']) / 'evaluation_inputs'
    panel = []
    for row in selected:
        spec = specs[row['asset_id']]
        identity = {**{key: row[key] for key in ('asset_id', 'path', 'root_rt')},
                    'data': base['data'], 'surface_sampling': base['surface_sampling'],
                    'mode': 'seen_eval', 'frames': 8}
        path = root / Path(spec['path']).name
        if spec['identity'] != identity or path.is_symlink() or not path.is_file():
            raise ValueError('canonical frame input identity or cache changed')
        panel.append((path, spec['sha256'], row['asset_id']))
    return panel


def panel_record(spec):
    path, expected, asset_id = spec
    if sha256(path) != expected:
        raise ValueError('cached validation geometry differs from the retained archive')
    record = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    if record['asset_id'] != asset_id or record['raw_frames'][0] in record['heldout_raw_ids']:
        raise ValueError('canonical reference frame changed')
    return record


def tokenize(geometry, tokenizer, max_tokens, device):
    sequences = []
    if geometry['frame_target_joints'].shape[1] != 1:
        raise ValueError('static supervision requires exactly one frame')
    for xyz, parents, count in zip(geometry['frame_target_joints'].cpu().numpy(),
                                   geometry['target_parents'].cpu().numpy(),
                                   geometry['joint_count'].tolist()):
        joints = xyz[0, :count]
        parents = [None if p < 0 else int(p) for p in parents[:count]]
        cls = tokenizer.skeleton_class
        tokens = _validate_main_sequence(tokenizer.tokenize_skeleton(joints, parents, cls),
            tokenizer=tokenizer, class_name=cls, bos=tokenizer.bos, eos=tokenizer.eos,
            pad=tokenizer.pad, vocab_size=tokenizer.vocab_size)
        if len(tokens) > max_tokens:
            raise ValueError('complete skeleton exceeds the token budget; no truncation')
        sequences.append(tokens)
    ids = np.full((len(sequences), max(map(len, sequences))), tokenizer.pad, dtype=np.int64)
    mask = np.zeros(ids.shape, dtype=bool)
    for i, tokens in enumerate(sequences):
        ids[i, :len(tokens)], mask[i, :len(tokens)] = tokens, True
    return torch.from_numpy(ids).to(device), torch.from_numpy(mask).to(device)


def prepare(batches, base, tokenizer, device):
    # The released encoder chooses its own 512 anchors from the dense points.
    surface, geometry = prepare_virtual_surface(
        batches, device, **{**base['surface_sampling'], 'query_tokens': 1})
    ids, mask = tokenize(geometry, tokenizer, base['data']['max_main_tokens'], device)
    return (surface['dense_points'][:, 0], surface['dense_normals'][:, 0], ids, mask), geometry


def audit(config):
    if torch.cuda.is_initialized():
        raise RuntimeError('audit cannot initialize CUDA')
    base, dataset, selected = data_context(config)
    panel = panel_specs(config, base, selected, verify_archive=True)
    identities = [canonical_identity(panel_record(spec), 0) for spec in panel]
    stream = make_stream(config, 0)
    if (len(stream.global_ends) != 627 or int(stream.global_ends[-1]) != 30000
            or any(len(set(stream.order[i:i+5000])) != 5000 for i in range(0, 30000, 5000))):
        raise ValueError('single-pose training exposure plan differs')
    _, tokenizer, initial = initialize(config, torch.device('cpu'), eager=True)
    groups, _ = stress_groups(dataset.entries, expected_rows=5000)
    checked = []
    for index in sorted(set(groups['largest_joints'][:2] + groups['largest_geometry'][:2] + [0])):
        batch = dataset[FrameSampleRequest(index, 1, 20261030 + index)]
        inputs, geometry = prepare([batch], base, tokenizer, torch.device('cpu'))
        checked.append({'asset_id': geometry['asset_id'][0], 'raw_frame': int(geometry['selected_frames'][0, 0]),
                        'token_count': int(inputs[-1].sum())})
    return {'complete': True, 'passed': True, 'cuda_initialized': torch.cuda.is_initialized(),
        'optimizer_updates': 0, 'train_assets': len(dataset), 'planned_updates': len(stream.global_ends),
        'planned_asset_visits': 30000, 'plan_sha256': stream.plan_sha256, 'initialization': initial,
        'checked_loader': checked, 'fixed_validation_inputs': identities,
        'SSD': verify_ssd(config['data_cache']['root'])}


def capacity(config, device, rank):
    base, dataset, _ = data_context(config)
    model, tokenizer, initial = initialize(config, device)
    groups, _ = stress_groups(dataset.entries, expected_rows=5000)
    profiles = []
    # Reserve space for Adam moments, DDP buckets, and optimizer temporaries.
    optimizer_reserve = 4 * sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    ceiling = min(76000 * 2**20, config['runtime']['gpu_memory_fraction_limit']
                  * torch.cuda.get_device_properties(device).total_memory)
    for profile, indices in groups.items():
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        batch = dataset[FrameSampleRequest(indices[rank], 1, 20261030 + rank)]
        inputs, _ = prepare([batch] * config['training']['frame_policy']['1'], base, tokenizer, device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            losses, _ = model(*inputs)
        if not bool(torch.isfinite(losses).all()):
            raise RuntimeError('nonfinite stress loss')
        losses.mean().backward()
        norms = gradient_groups(model)
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_reserved(device)
        if peak + optimizer_reserve >= ceiling:
            raise RuntimeError('capacity lacks Adam/DDP memory headroom')
        profiles.append({'profile': profile, 'assets_per_rank': len(losses), 'gradient_groups': norms,
                         'peak_reserved_bytes': peak, 'Adam_DDP_reserve_bytes': optimizer_reserve})
        del inputs, losses, batch
    return {'complete': True, 'passed': True, 'optimizer_updates': 0, 'initialization': initial,
            'profiles_by_rank': gather(profiles, 2)}


def gradient_groups(model):
    groups = {}
    for name in ('mesh_encoder', 'decoder_condition_projection', 'xyz_decoder'):
        values = [p.grad.detach().float().norm() for p in getattr(model, name).parameters() if p.grad is not None]
        groups[name] = float(torch.stack(values).norm()) if values else 0.
    if not all(np.isfinite(value) and value > 0 for value in groups.values()):
        raise RuntimeError('released mesh, projection and decoder must all receive finite gradients')
    return groups


@torch.inference_mode()
def evaluate(model, panel, base, config, output, samples, device, rank):
    rows = []
    tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    with preserve_random():
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        try:
            eager, tokenizer, _ = initialize(config, device, eager=True, load_weights=False)
            eager.load_state_dict(model.state_dict(), strict=True)
            eager.eval()
            start = torch.tensor([tokenizer.bos, tokenizer.cls_name_to_token(tokenizer.skeleton_class)], device=device)
            processor = original_xyz_vocabulary_processor(tokenizer, start)
            for spec in panel[rank::2]:
                if shared_deadline_due(config, device, 2):
                    raise RuntimeError('save-stop line reached during fixed reference evaluation')
                record = panel_record(spec)
                surface = record['surface']
                seed = sample_seed(base['evaluation']['seed'], record['asset_id'], 'native-static-512')
                torch.manual_seed(seed)
                condition = eager.encode(surface['dense_points'][:, 0].to(device), surface['dense_normals'][:, 0].to(device))
                prediction = eager.generate_condition(condition, start, eos_token=tokenizer.eos,
                    max_new_tokens=config['evaluation']['max_new_tokens'], logits_processor=processor)
                identity = canonical_identity(record, 0)
                rows.append({**identity, 'seed': seed, **score_generation(prediction, tokenizer,
                    record['truth'][0].numpy(), record['parents'].tolist(), identity['target_mesh_diagonal'])})
                del record, surface, condition, prediction
                write_json(output / f'eval_progress_rank_{rank}.json', {'samples': samples, 'completed_local': len(rows)})
            del eager
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
    rows = sum(gather(rows, 2), [])
    if len(rows) != len(panel) or {r['asset_id'] for r in rows} != {spec[2] for spec in panel}:
        raise RuntimeError('not exactly one reference result per fixed asset')
    summary = {'complete': True, 'samples': samples, 'stage1': summarize_rows(rows),
        'stage2_enabled': False, 'reference_slot': 0, 'policy': 'greedy_fp32_native_skeleton'}
    if rank == 0:
        directory = output / f'evaluation_{samples:06d}'
        directory.mkdir(exist_ok=False)
        for name, value in [('stage1_rows.json', rows), ('summary.json', summary)]:
            path = directory / name
            write_json(path, value)
            mirror(path, str(path.relative_to(output)), config)
        print('VALIDATION ' + json.dumps(summary), flush=True)
    dist.barrier(device_ids=[device.index])
    return summary


def checkpoint(model, optimizer, stream, config, output, initial, source, samples, step, rank):
    states = gather({'rank': rank, 'stream': stream.state_dict(), **random_state()}, 2)
    spec = None
    if rank == 0:
        path = output / f'checkpoint_{samples:06d}.pt'
        temp = path.with_suffix('.pt.tmp')
        if path.exists() or temp.exists() or any(s['stream']['cursor'] * 2 != samples for s in states):
            raise RuntimeError('refuse to overwrite or save an inconsistent training cursor')
        payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rank_states': states,
            'initialization': initial, 'configuration': config, 'code_commit': source,
            'samples': samples, 'step': step, 'input_frame_visits': samples, 'stage2_enabled': False}
        torch.save(payload, temp)
        saved = torch.load(temp, map_location='cpu', mmap=True, weights_only=False)
        if (saved['samples'] != samples or saved['step'] != step or len(saved['rank_states']) != 2
                or not saved['optimizer']['state'] or saved['model'].keys() != payload['model'].keys()):
            raise RuntimeError('full checkpoint is not recoverable')
        del saved, payload
        temp.replace(path)
        spec = {**mirror(path, path.name, config), 'samples': samples, 'step': step,
                'complete': True, 'both_copies_verified': True, 'optimizer_and_rank_RNG_retained': True}
        write_json(path.with_suffix('.json'), spec)
        write_json(Path(spec['durable_path']).with_suffix('.json'), spec)
    return gather(spec, 2)[0]


def train(config, output, source, device, rank):
    base, dataset, selected = data_context(config)
    panel = panel_specs(config, base, selected)
    seed = config['training']['initialization_seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model, tokenizer, initial = initialize(config, device)
    wrapped = wrap(model, device, 2)
    optimizer = optimizer_for(model, base)
    write_json(output / f'initialization_rank_{rank}.json', initial)
    curve = [evaluate(model, panel, base, config, output, 0, device, rank)]
    torch.manual_seed(seed + rank)
    stream = make_stream(config, rank)
    plan = make_stream(config, rank)
    samples = step = 0
    checkpoints = []
    log = output / f'updates_rank_{rank}.jsonl'
    with log.open('x', encoding='utf-8') as trace, OrderedAssetPrefetch(
            dataset, (plan.take() for _ in plan.global_ends),
            workers=config['runtime']['data_workers_per_rank'], depth=config['runtime']['prefetch_microbatches']) as loader:
        for end in stream.global_ends:
            if shared_deadline_due(config, device, 2):
                raise RuntimeError('save-stop line reached before completing 30000 visits')
            requests = stream.take()
            loaded, _ = loader.take(requests)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            lr, beta = set_lr(optimizer, samples, base)
            inputs, geometry = prepare([loaded[r] for r in requests], base, tokenizer, device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                losses, accuracy = wrapped(*inputs)
            if len(losses) * 2 != int(end) - samples or not bool(torch.isfinite(losses).all()):
                raise RuntimeError('nonfinite loss or incorrect effective batch')
            losses.mean().backward()
            groups = gradient_groups(model) if step == 0 else None
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
            optimizer.step()
            metrics = torch.stack((losses.detach().mean(), accuracy.detach().mean()))
            dist.all_reduce(metrics, op=dist.ReduceOp.AVG)
            samples, step = int(end), step + 1
            progress = {'samples': samples, 'step': step, 'ar_loss': float(metrics[0]),
                'ar_accuracy': float(metrics[1]), 'gradient_norm': norm, 'lr': lr, 'beta1': beta,
                'assets_in_update': len(losses)*2, 'input_frames_in_update': len(losses)*2,
                'stage2_enabled': False}
            if groups is not None:
                progress['gradient_groups'] = groups
            trace.write(json.dumps({**progress, 'assets': geometry['asset_id'],
                'raw_frames': geometry['selected_frames'].flatten().tolist(),
                'sampling_seeds': [request.sampling_seed for request in requests]}) + '\n')
            trace.flush()
            write_json(output / f'progress_rank_{rank}.json', progress)
            del inputs, geometry, losses, accuracy, loaded, metrics
            if samples in MILESTONES:
                optimizer.zero_grad(set_to_none=True)
                checkpoints.append(checkpoint(model, optimizer, stream, config, output, initial, source, samples, step, rank))
                curve.append(evaluate(model, panel, base, config, output, samples, device, rank))
    if samples != 30000 or step != 627 or [r['samples'] for r in curve] != config['evaluation']['checkpoints']:
        raise RuntimeError('incomplete exposure or fixed evaluation schedule')
    if rank == 0:
        for other in range(2):
            mirror(output / f'updates_rank_{other}.jsonl', f'receipts/updates_rank_{other}.jsonl', config)
    dist.barrier(device_ids=[device.index])
    return {'complete': True, 'passed': True, 'samples': samples, 'updates': step,
            'input_frame_visits': samples, 'plan_sha256': stream.plan_sha256,
            'checkpoints': checkpoints, 'validation': curve, 'working_model_changed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('audit', 'capacity', 'train'), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    source = verify_launch(args, config)
    config_sha = sha256(args.config)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.mode == 'audit':
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / 'result.json', {**audit(config), 'source': source, 'config_sha256': config_sha,
                                               'stage2_enabled': False})
        return
    rank = int(os.environ['RANK'])
    if int(os.environ['WORLD_SIZE']) != 2 or torch.cuda.device_count() != 2:
        raise RuntimeError('exactly two admitted H100s are required')
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    fraction = config['runtime']['gpu_memory_fraction_limit']
    torch.cuda.set_per_process_memory_fraction(
        min(fraction, 76000 * 2**20 / torch.cuda.get_device_properties(device).total_memory), device)
    dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    try:
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=False)
        dist.barrier(device_ids=[device.index])
        value = (capacity(config, device, rank) if args.mode == 'capacity'
                 else train(config, args.output, source, device, rank))
        value.update(source=source, config_sha256=config_sha, stage2_enabled=False,
                     finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.output / f'result_rank_{rank}.json', value)
        if rank == 0:
            write_json(args.output / 'result.json', value)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
