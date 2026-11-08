#!/usr/bin/env python3
"""Matched frame-local, temporal multi-target and temporal single-query Stage 1."""
from contextlib import nullcontext
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
import torch.distributed as dist

from anchor_full_data import stress_groups, verify_ssd
from anchor_full_runtime import capacity_requests
from probe_virtual_query_lambda import parameter_fingerprint
from reference_flow_runtime import mirror
from skintokens_stage1_protocol import ROOT, validate_config
from skintokens_static_protocol import MILESTONES
from stage1_surface_repair_runtime import canonical_identity
from tokenrig_decoder import (build_tokenrig_conditioning, build_tokenrig_decoder,
                              initialize_native_motion_identity)
from train_skintokens_static import (checkpoint, data_context, evaluate, main, panel_record,
                                     panel_specs, tokenize)
from virtual_fusion_normal_runtime import MatchedFrameStream
from virtual_pair_loading import OrderedAssetPrefetch
from virtual_pair_runtime import (gather, next_update_batches, optimizer_for, set_lr,
                                  shared_deadline_due, target_selection, wrap)
from virtual_query_xyz_runtime import write_json
from rigweave.dynamic_rig.aux_skeleton_tokens import AuxSkeletonTokens
from rigweave.dynamic_rig.frame_batch_sampler import FrameSampleRequest
from rigweave.dynamic_rig.surface_tokenizer import FixedQuerySurfaceTokenizer
from rigweave.dynamic_rig.virtual_frame_data import prepare_virtual_surface
from rigweave.dynamic_rig.virtual_path_ablation import VirtualPathAblationRig
from rigweave.dynamic_rig.virtual_query_xyz import VirtualQueryXYZConfig, VirtualQueryXYZRig, sequence_ce


class Stage1Skeleton(VirtualPathAblationRig):
    def __init__(self, backbone, arm):
        super().__init__(backbone, virtual_enabled=False, frame_local=arm == 'frame_local')
        self.single_query = arm == 'temporal_query'

    def _real_roles(self, batch, frames):
        if not self.single_query:
            return super()._real_roles(batch, frames)
        role = self.motion_encoder.role_token
        return torch.cat((role[:, :1].expand(batch, -1, -1, -1),
                          role[:, 1:].expand(batch, frames - 1, -1, -1)), dim=1)

    def _real_readout(self, real):
        return real[:, :1] if self.single_query else real

    def forward(self, surface, ids, mask, slots, valid):
        frames, virtual = self.encode(surface)
        if self.single_query and (slots.shape[1] != 1 or bool(slots.any())):
            raise ValueError('only the first real query may be supervised')
        counts = valid.sum(dim=1)
        loss = frames.new_zeros(len(counts), dtype=torch.float32)
        accuracy = loss.clone()
        for target in range(slots.shape[1]):
            rows = valid[:, target].nonzero(as_tuple=True)[0]
            # The condition tail is frame-local; do not run it on unlabelled slots.
            feature = frames[rows, slots[rows, target]][:, None]
            condition = self.conditions(feature, virtual)[:, 0]
            ids_at_target, mask_at_target = ids[rows, target], mask[rows, target]
            logits = self.main_logits(condition, ids_at_target[:, :-1], mask_at_target[:, :-1])
            ce, acc = sequence_ce(logits, ids_at_target[:, 1:].masked_fill(~mask_at_target[:, 1:], -100))
            loss = loss.index_add(0, rows, ce.float())
            accuracy = accuracy.index_add(0, rows, acc.float())
        return loss / counts, accuracy / counts


def initialize(config, device, *, eager=False, load_weights=True):
    base = json.loads((ROOT / config['base_recipe']['path']).read_text())
    seed = config['training']['initialization_seed']
    torch.manual_seed(seed)
    spec = {**config['tokenrig_decoder'], 'eager': eager}
    encoder, projection, mesh = build_tokenrig_conditioning(spec, 1024, load_weights=load_weights)
    mesh.pop('motion_and_stage2_architecture_unchanged')
    decoder, tokenizer, ar = build_tokenrig_decoder(spec, load_weights=load_weights)
    if not eager:
        decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    architecture = VirtualQueryXYZConfig(**base['architecture'])
    codec = AuxSkeletonTokens(**base['auxiliary_codec'])
    torch.manual_seed(seed)
    original = VirtualQueryXYZRig(FixedQuerySurfaceTokenizer(encoder, nn.Identity()), decoder,
        architecture, auxiliary_vocab_size=codec.vocab_size, auxiliary_pad=codec.pad,
        project_decoder_condition=True)
    original.decoder_condition_projection = projection
    original.decoder_condition_tokens = 512
    identity = initialize_native_motion_identity(original)
    model = Stage1Skeleton(original, config['arm'])
    initial = {'mesh': mesh, 'decoder': ar, 'motion_initialization': identity,
        'project_checkpoint_loaded': False, 'stage2_enabled': False,
        'initial_model_sha256': parameter_fingerprint(model) if load_weights else None,
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}
    return model.to(device), tokenizer, initial


def stream_for(config, rank):
    spec, train = config['train_manifest'], config['training']
    return MatchedFrameStream(range(spec['rows']), rank=rank, world_size=2,
        seed=train['stream_seed'], target_assets=train['additional_assets'],
        effective_assets=train['effective_assets_per_update'], frame_policy=train['frame_policy'],
        manifest_sha256=spec['sha256'], update_boundaries=train['checkpoint_assets'])


def selected_slots(arm, base, requests):
    if arm == 'temporal_query':
        return [[0] for _ in requests]
    return target_selection(base, requests)


def plan_for(config, base):
    streams = [stream_for(config, rank) for rank in (0, 1)]
    stream = streams[0]
    samples = frames = supervised = 0
    rows = config['train_manifest']['rows']
    if any(len(set(stream.order[i:i+rows])) != rows for i in range(0, len(stream.order), rows)):
        raise ValueError('each exposure pass must cover the fixed 5k exactly')
    milestones = {}
    for end in stream.global_ends:
        observed = []
        for part in streams:
            _, batches = next_update_batches(part, samples, 2)
            for requests in batches:
                slots = selected_slots(config['arm'], base, requests)
                if any(not 0 <= slot < req.frames for req, chosen in zip(requests, slots) for slot in chosen):
                    raise ValueError('target lies outside the sampled clip')
                observed.extend(r.item_index for r in requests)
                frames += sum(r.frames for r in requests)
                supervised += sum(map(len, slots))
        if observed != stream.order[samples:end].tolist():
            raise ValueError('rank exposure order differs')
        samples = int(end)
        if samples in MILESTONES:
            milestones[str(samples)] = {'input_frame_visits': frames, 'supervised_frame_visits': supervised}
    return {'samples': samples, 'updates': len(stream.global_ends), 'plan_sha256': stream.plan_sha256,
            'milestones': milestones, **milestones[str(samples)]}


def prepare(loaded, selected, base, tokenizer, device):
    surface, geometry = prepare_virtual_surface(loaded, device, **base['surface_sampling'])
    rows = torch.tensor([i for i, slots in enumerate(selected) for _ in slots], device=device)
    frames = torch.tensor([slot for slots in selected for slot in slots], device=device)
    # Tokenize only the supervised trees, never the context-only poses.
    chosen = {'frame_target_joints': geometry['frame_target_joints'][rows, frames][:, None],
              'target_parents': geometry['target_parents'][rows], 'joint_count': geometry['joint_count'][rows]}
    tokens, masks = tokenize(chosen, tokenizer, base['data']['max_main_tokens'], device)
    counts = [len(slots) for slots in selected]
    width = max(counts)
    slots = torch.tensor([s + [0] * (width-len(s)) for s in selected], device=device)
    valid = torch.arange(width, device=device) < torch.tensor(counts, device=device)[:, None]
    ids = torch.full((*slots.shape, tokens.shape[-1]), tokenizer.pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    ids[valid], mask[valid] = tokens, masks
    return (surface, ids, mask, slots, valid), geometry


def reference_condition(model, surface, device):
    if surface['query_points'].shape[1] != 8:
        raise ValueError('expected eight context/target inputs on the canonical panel')
    frames, virtual = model.encode({name: value.to(device) for name, value in surface.items()})
    return model.conditions(frames[:, :1], virtual)[:, 0]


def audit(config):
    device = torch.device('cpu')
    base, dataset, selected = data_context(config, frames=2)
    panel = panel_specs(config, base, selected, verify_archive=True)
    identities = [canonical_identity(panel_record(spec), 0) for spec in panel]
    plan = plan_for(config, base)
    if (plan['samples'], plan['updates']) != (30000, 627):
        raise ValueError('expected 30k exact visits with the original batch-48 plan')
    _, tokenizer, initial = initialize(config, device, eager=True)
    groups, _ = stress_groups(dataset.entries, expected_rows=5000)
    checked = []
    for frames in (2, 12, 24):
        for indices in groups.values():
            req = FrameSampleRequest(indices[0], frames, 20261030 + frames)
            slots = selected_slots(config['arm'], base, [req])
            inputs, geometry = prepare([dataset[req]], slots, base, tokenizer, device)
            checked.append({'asset_id': geometry['asset_id'][0],
                'raw_frames': geometry['selected_frames'][0].tolist(), 'target_slots': slots[0],
                'target_tokens': int(inputs[2].sum())})
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU audit used CUDA')
    return {'complete': True, 'passed': True, 'arm': config['arm'], 'cuda_initialized': False,
        'optimizer_updates': 0, 'train_assets': len(dataset), 'plan': plan, 'initialization': initial,
        'checked_loader': checked, 'fixed_validation_inputs': identities,
        'SSD': verify_ssd(config['data_cache']['root'])}


def grad_groups(model):
    groups = {'mesh': model.surface_tokenizer, 'interaction': model.motion_encoder,
              'projection': model.base.decoder_condition_projection, 'decoder': model.xyz_decoder}
    result = {}
    for name, module in groups.items():
        values = [p.grad.detach().float().norm() for p in module.parameters() if p.grad is not None]
        result[name] = float(torch.stack(values).norm()) if values else 0.
    if (not all(np.isfinite(x) and x > 0 for x in result.values())
            or any(p.grad is not None for p in model.parameters() if not p.requires_grad)):
        raise RuntimeError('invalid skeleton-only gradients')
    return result


def check_initialization(config, initial):
    path = Path(config['runtime']['output_root']) / 'audit/result.json'
    if json.loads(path.read_text())['initialization'] != initial:
        raise RuntimeError('actual weights differ from the strict CPU audit')


def capacity(config, device, rank):
    base, dataset, _ = data_context(config, frames=2)
    model, tokenizer, initial = initialize(config, device)
    check_initialization(config, initial)
    groups, _ = stress_groups(dataset.entries, expected_rows=5000)
    profiles = []
    reserve = 4 * sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    for frames in (6, 12, 24):
        for label, indices in groups.items():
            if shared_deadline_due(config, device, 2):
                raise RuntimeError('save-stop line reached before completing capacity')
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            size = config['training']['frame_policy'][str(frames)]
            requests = capacity_requests(base, indices, rank, frames, size, 20261030)
            selected = selected_slots(config['arm'], base, requests)
            inputs, _ = prepare([dataset[r] for r in requests], selected, base, tokenizer, device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, _ = model(*inputs)
            if not bool(torch.isfinite(loss).all()):
                raise RuntimeError('capacity found nonfinite loss')
            loss.mean().backward()
            gradients = grad_groups(model)
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
            reserved = torch.cuda.max_memory_reserved(device)
            # Caching allocator blocks can be reused by Adam and DDP.
            if max(reserved, peak + reserve) >= 76000 * 2**20:
                raise RuntimeError('capacity lacks Adam/DDP memory headroom')
            profiles.append({'case': label, 'frames': frames, 'assets_per_rank': size,
                'targets_per_rank': sum(map(len, selected)), 'peak_allocated_bytes': peak,
                'peak_reserved_bytes': reserved, 'Adam_DDP_reserve_bytes': reserve,
                'gradient_groups': gradients})
            del inputs, loss
    return {'complete': True, 'passed': True, 'arm': config['arm'], 'optimizer_updates': 0,
            'initialization': initial, 'profiles_by_rank': gather(profiles, 2)}


def train(config, output, source, device, rank):
    base, dataset, selected = data_context(config, frames=2)
    panel = panel_specs(config, base, selected)
    seed = config['training']['initialization_seed']
    random.seed(seed)
    np.random.seed(seed)
    model, tokenizer, initial = initialize(config, device)
    check_initialization(config, initial)
    optimizer = optimizer_for(model, base)
    wrapped = wrap(model, device, 2)
    write_json(output / f'initialization_rank_{rank}.json', initial)
    stream, prefetch_plan = stream_for(config, rank), stream_for(config, rank)
    plan = plan_for(config, base)

    def evaluate_reference(samples):
        value = evaluate(model, panel, base, config, output, samples, device, rank,
                         initialize_model=initialize, reference_condition=reference_condition)
        torch.cuda.empty_cache()
        return value

    def planned_batches():
        last = 0
        for end in prefetch_plan.global_ends:
            _, batches = next_update_batches(prefetch_plan, last, 2)
            yield from batches
            last = int(end)

    curve = [evaluate_reference(0)]
    torch.manual_seed(seed + rank)
    checkpoints = []
    samples = step = frames = supervised = 0
    with (output / f'updates_rank_{rank}.jsonl').open('x', encoding='utf-8') as trace, OrderedAssetPrefetch(
            dataset, planned_batches(), workers=config['runtime']['data_workers_per_rank'],
            depth=config['runtime']['prefetch_microbatches']) as loader:
        while samples < plan['samples']:
            if shared_deadline_due(config, device, 2):
                raise RuntimeError('save-stop line reached before completing 30k')
            goal, batches = next_update_batches(stream, samples, 2)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            lr, beta = set_lr(optimizer, samples, base)
            totals = torch.zeros(5, device=device, dtype=torch.float64)
            schedule = []
            for i, requests in enumerate(batches):
                loaded, _ = loader.take(requests)
                chosen = selected_slots(config['arm'], base, requests)
                inputs, geometry = prepare([loaded[r] for r in requests], chosen, base, tokenizer, device)
                with wrapped.no_sync() if i+1 < len(batches) else nullcontext():
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        losses, accuracy = wrapped(*inputs)
                    if len(losses) != len(requests) or not bool(torch.isfinite(losses).all()):
                        raise RuntimeError('nonfinite or misaligned per-asset loss')
                    (losses.sum() / (goal // 2)).backward()
                totals += torch.stack((losses.detach().sum(), accuracy.detach().sum(),
                    totals.new_tensor(len(requests)), totals.new_tensor(sum(r.frames for r in requests)),
                    totals.new_tensor(sum(map(len, chosen)))))
                schedule.append({'assets': geometry['asset_id'], 'raw_frames': geometry['selected_frames'].tolist(),
                    'sampling_seeds': [r.sampling_seed for r in requests], 'target_slots': chosen})
                del loaded, inputs, geometry, losses, accuracy
            groups = grad_groups(model) if step == 0 else None
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
            optimizer.step()
            dist.all_reduce(totals)
            if int(totals[2]) != goal or stream.cursor * 2 != samples + goal:
                raise RuntimeError('effective global asset count differs')
            samples, step = samples + goal, step + 1
            frames, supervised = frames + int(totals[3]), supervised + int(totals[4])
            exposure = {'input_frame_visits': frames, 'supervised_frame_visits': supervised}
            progress = {'samples': samples, 'step': step, 'assets_in_update': goal, **exposure,
                'ar_loss': float(totals[0] / goal), 'ar_accuracy': float(totals[1] / goal),
                'gradient_norm': norm, 'lr': lr, 'beta1': beta,
                'stage2_enabled': False, 'arm': config['arm']}
            if groups is not None:
                progress['gradient_groups'] = groups
            trace.write(json.dumps({**progress, 'microbatches': schedule}) + '\n')
            trace.flush()
            write_json(output / f'progress_rank_{rank}.json', progress)
            if samples in MILESTONES:
                if exposure != plan['milestones'][str(samples)]:
                    raise RuntimeError('actual input or supervised frame count differs')
                optimizer.zero_grad(set_to_none=True)
                checkpoints.append(checkpoint(model, optimizer, stream, config, output, initial,
                    source, samples, step, rank, exposure=exposure))
                curve.append(evaluate_reference(samples))
    if samples != 30000 or step != 627 or [r['samples'] for r in curve] != config['evaluation']['checkpoints']:
        raise RuntimeError('incomplete Stage 1 budget or validation')
    if rank == 0:
        for other in range(2):
            name = f'updates_rank_{other}.jsonl'
            mirror(output / name, f'receipts/{name}', config)
    dist.barrier(device_ids=[device.index])
    return {'complete': True, 'passed': True, 'samples': samples, 'updates': step, **exposure,
            'arm': config['arm'], 'plan_sha256': plan['plan_sha256'], 'checkpoints': checkpoints,
            'validation': curve, 'initialization': initial, 'working_model_changed': False}


if __name__ == '__main__':
    main(validate=validate_config, audit_model=audit, capacity_model=capacity,
         train_model=train, description=__doc__)
