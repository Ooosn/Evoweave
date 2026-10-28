"""Full-dataset shared-flow training, resource acceptance, and paired evaluation."""
from contextlib import nullcontext
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist

from fit_cohort_selection import _joint_count
from id_xyz_stages import preserve_random, random_state
from reference_flow_protocol import load_backbone_config, sample_seed, scientific_sha, sha256
from reference_xyz_runtime import generate_reference, row_metrics
from train_virtual_fusion_normal import record_samples
from train_virtual_query_xyz import prepare, load_capacity_selection
from virtual_pair_loading import OrderedAssetPrefetch
from virtual_pair_runtime import (apply_rng, audit_loader, check_gradient_groups, evaluation_rows,
    gather, make_stream, next_update_batches, optimizer_for, parameter_names, planned_microbatches,
    set_lr, shared_deadline_due, wrap)
from virtual_query_xyz_runtime import build_model, fixed_dataset, read_entries, write_json
from rigweave.dynamic_rig.frame_batch_sampler import FrameSampleRequest
from rigweave.dynamic_rig.reference_flow import ReferenceFlow, ReferenceFlowConfig, SharedReferenceFlowRig
from rigweave.dynamic_rig.reference_skeleton_xyz import reference_slots
from rigweave.dynamic_rig.virtual_frame_data import prepare_virtual_surface
from rigweave.dynamic_rig.virtual_path_ablation import VirtualPathAblationRig
from virtual_pair_runtime import target_selection


def noise_for(shape, seed, identity, domain, device):
    generator = torch.Generator().manual_seed(sample_seed(seed, identity, domain))
    return torch.randn(shape, generator=generator).to(device)


def flow_inputs(geometry, requests, config, device):
    shape = geometry['frame_target_joints'].shape
    references, noise, tau = [], [], []
    for index, request in enumerate(requests):
        generator = torch.Generator().manual_seed(sample_seed(config['training']['flow_seed'], request.sampling_seed, 'train'))
        references.append(int(torch.randint(request.frames, (1,), generator=generator)))
        tau.append(float(torch.rand((), generator=generator)))
        # Draw only actual joints: noise is independent of other batch members' padding.
        count = int(geometry['joint_mask'][index].sum())
        value = torch.zeros(shape[1:])
        value[:, :count] = torch.randn((shape[1], count, 3), generator=generator)
        noise.append(value)
    return (torch.tensor(references, device=device), torch.stack(noise).to(device),
            torch.tensor(tau, device=device))


def ar_indices(base, requests, device):
    selected = target_selection(base, requests)
    width = max(map(len, selected))
    indices = torch.tensor([slots + [0] * (width - len(slots)) for slots in selected], device=device)
    mask = torch.arange(width, device=device)[None] < torch.tensor(list(map(len, selected)), device=device)[:, None]
    return selected, indices, mask


def read_all(config):
    base = load_backbone_config(config)
    entries = read_entries(base)
    ids, paths = set(), set()
    for rows in entries.values():
        for row in rows:
            if row['asset_id'] in ids or row['path'] in paths:
                raise ValueError('duplicate or leaking asset identity')
            ids.add(row['asset_id'])
            paths.add(row['path'])
    return base, entries


def audit(config, output, source):
    base, entries = read_all(config)
    groups, _ = load_capacity_selection(base, entries['train'])
    counts = [_joint_count(row)[0] for row in entries['train']]
    if min(counts) < 1 or max(counts) > config['architecture']['max_joints']:
        raise ValueError('complete source trees exceed flow capacity')
    groups['largest_joints'] = sorted(range(5000), key=lambda i: (-counts[i], i))[:3]
    streams = [make_stream(base, rank, 2) for rank in range(2)]
    samples, segment, passes, frames = 0, [], [], 0
    while samples < config['training']['additional_assets']:
        for stream in streams:
            goal, batches = next_update_batches(stream, samples, 2)
            for batch in batches:
                segment.extend(r.item_index for r in batch)
                frames += sum(r.frames for r in batch)
                selected = target_selection(base, batch)
                if any(len(slots) < 2 or len(set(slots)) != len(slots) for slots in selected):
                    raise ValueError('AR supervision contract changed')
        samples += goal
        if samples % 5000 == 0:
            if len(segment) != 5000 or sorted(segment) != list(range(5000)):
                raise ValueError('each training pass must visit all5000assets exactly once')
            passes.append({'additional_asset_exposures': samples, 'unique_assets': 5000})
            segment = []
    if sha256(config['backbone']['checkpoint']['path']) != config['backbone']['checkpoint']['sha256']:
        raise ValueError('donor checkpoint changed')
    loader = audit_loader(base, entries, groups)
    result = {'complete': True, 'passed': True, 'source': source, 'scientific_sha256': scientific_sha(config),
        'train_assets': 5000, 'valid_assets': 967, 'passes': passes, 'input_frame_exposures': frames,
        'capacity_groups': groups, 'maximum_train_joints': max(counts), 'loader': loader,
        'gpu_work': False, 'not_model_quality': True}
    write_json(output / 'result.json', result)
    return result


def load_model(config, device, rank, world):
    base = load_backbone_config(config)
    spec = config['backbone']['checkpoint']
    if rank == 0 and sha256(spec['path']) != spec['sha256']:
        raise ValueError('donor digest changed')
    dist.barrier(device_ids=[device.index])
    original, tokenizer, codec, official = build_model(base, device)
    backbone = VirtualPathAblationRig(original, virtual_enabled=False)
    payload = torch.load(spec['path'], map_location='cpu', mmap=True, weights_only=False)
    if (payload['variant'] != 'fraction20_40_query' or payload['asset_exposures'] != 20000
            or payload['code_commit'] != 'a0bda62bb8fba70320310b063da8c7898ddaf441'
            or payload['optimizer_parameter_names'] != parameter_names(backbone)):
        raise ValueError('donor weights/optimizer identity changed')
    backbone.load_state_dict(payload['model'], strict=True)
    optimizer = optimizer_for(backbone, base)
    optimizer.load_state_dict(payload['optimizer'])
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(config['training']['flow_seed'])
        flow = ReferenceFlow(ReferenceFlowConfig(**config['architecture'])).to(device)
    model = SharedReferenceFlowRig(backbone, flow)
    optimizer.add_param_group({'params': list(flow.parameters())})
    apply_rng(payload['rank_states'][rank])
    initial = {'donor': spec, 'strict_complete_backbone_load': True, 'backbone_frozen': False,
        'backbone_optimizer_retained': True, 'flow_optimizer_new': True,
        'donor_model_sha256': payload['model_sha256'],
        'constructor_official_sha256': official['checkpoint_sha256']}
    del payload
    stream = make_stream(base, rank, world)
    samples, frames, step = 0, 0, 0
    resume = config['training'].get('resume')
    if resume:
        if sha256(resume['path']) != resume['sha256']:
            raise ValueError('resume checkpoint digest changed')
        saved = torch.load(resume['path'], map_location='cpu', mmap=True, weights_only=False)
        if (saved['scientific_sha256'] != scientific_sha(config)
                or saved['optimizer_parameter_names'] != parameter_names(model)
                or len(saved['rank_states']) != world):
            raise ValueError('resume scientific/parameter/rank contract changed')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        rank_state = saved['rank_states'][rank]
        stream = make_stream(base, rank, world, rank_state['stream'])
        apply_rng(rank_state)
        samples, frames, step = (saved[k] for k in ('additional_asset_exposures', 'input_frame_exposures', 'step'))
        if stream.cursor * world != samples:
            raise ValueError('resume sampler cursor differs')
        initial = saved['initialization']
        del saved
    return model, optimizer, tokenizer, codec, base, initial, stream, samples, frames, step


def norm_of(module):
    gradients = [p.grad.detach().float() for p in module.parameters() if p.requires_grad and p.grad is not None]
    if not gradients or any(not bool(torch.isfinite(g).all()) for g in gradients):
        raise RuntimeError('missing/nonfinite learning signal')
    norm = sum(float(g.square().sum()) for g in gradients) ** .5
    if norm <= 0:
        raise RuntimeError('zero learning signal')
    return norm


def update(model, wrapped, optimizer, dataset, batches, tokenizer, codec, base, config,
           samples, device, world, *, prefetch=None, check_groups=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    lr, beta = set_lr(optimizer, 20000 + samples, base)
    assets_goal = sum(map(len, batches)) * world
    weight = config['training']['flow_weight'] * min(1., (samples + assets_goal) / config['training']['flow_warmup_assets'])
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    totals = torch.zeros(6, device=device, dtype=torch.float64)
    records = []
    for i, requests in enumerate(batches):
        loaded = prefetch.take(requests)[0] if prefetch else dataset
        surface, targets, geometry = prepare(loaded, requests, base, tokenizer, codec, device)
        del loaded
        selected, indices, index_mask = ar_indices(base, requests, device)
        reference, noise, tau = flow_inputs(geometry, requests, config, device)
        with wrapped.no_sync() if world > 1 and i < len(batches) - 1 else nullcontext():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                losses = wrapped(surface, targets, geometry, indices, index_mask, reference, noise, tau, weight)
            if not bool(torch.isfinite(losses['loss_per_asset']).all()):
                raise RuntimeError('nonfinite joint objective')
            losses['loss_per_asset'].sum().backward()
        size = len(requests)
        totals += torch.stack((losses['ar_loss'].detach().sum(), losses['flow_loss'].detach().sum(),
            losses['main_accuracy'].detach() * size, totals.new_tensor(size),
            totals.new_tensor(size * requests[0].frames), totals.new_tensor(sum(map(len, selected)))))
        records.append({**record_samples(geometry, requests), 'frames': requests[0].frames,
            'ar_target_slots': selected, 'reference_slots': reference.tolist(), 'flow_times': tau.tolist(),
            'joint_counts': geometry['joint_mask'].sum(1).tolist()})
        del surface, targets, geometry, losses, noise
    dist.all_reduce(totals)
    assets = int(totals[3])
    if assets != assets_goal:
        raise RuntimeError('global effective batch differs')
    for p in model.parameters():
        if p.grad is not None:
            p.grad.mul_(world / assets)
    groups = None
    if check_groups:
        groups = check_gradient_groups(model.backbone, 'fraction20_40_query')
        groups['flow'] = norm_of(model.flow)
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
    optimizer.step()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_reserved(device)
    if peak > config['runtime']['gpu_memory_fraction_limit'] * torch.cuda.get_device_properties(device).total_memory:
        raise RuntimeError('shared flow exceeded approved GPU memory fraction')
    return {'assets_in_update': assets, 'frames_in_update': int(totals[4]), 'ar_targets': int(totals[5]),
        'ar_loss': float(totals[0] / assets), 'flow_loss': float(totals[1] / assets),
        'ar_accuracy': float(totals[2] / assets), 'flow_weight': weight, 'lr': lr, 'beta1': beta,
        'gradient_norm': norm, 'gradient_groups': groups, 'seconds': time.monotonic() - start,
        'max_allocated_bytes': torch.cuda.max_memory_allocated(device), 'max_reserved_bytes': peak,
        'microbatches': records}


def capacity(config, output, source, device, rank, world):
    base, entries = read_all(config)
    groups, _ = load_capacity_selection(base, entries['train'])
    counts = [_joint_count(r)[0] for r in entries['train']]
    groups['largest_joints'] = sorted(range(5000), key=lambda i: (-counts[i], i))[:3]
    model, optimizer, tokenizer, codec, _, initial, *_ = load_model(config, device, rank, world)
    dataset = fixed_dataset(entries['train'], base, 2, mode='train')
    model.train()
    requests = [FrameSampleRequest(groups['largest_joints'][rank], 2, 90000 + rank)]
    surface, _, geometry = prepare(dataset, requests, base, tokenizer, codec, device)
    reference, noise, tau = flow_inputs(geometry, requests, config, device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        features, _ = model.backbone.encode(surface)
        loss = model.flow(features, surface['query_points'], surface['query_normals'],
            reference_slots(geometry['frame_target_joints'], reference), geometry['target_parents'], geometry['joint_mask'],
            reference, targets=geometry['frame_target_joints'], noise=noise, tau=tau)['loss_per_asset'].sum()
    loss.backward()
    flow_gradient = {'surface': norm_of(model.backbone.surface_tokenizer),
        'motion': norm_of(model.backbone.motion_encoder), 'flow': norm_of(model.flow)}
    if any(p.grad is not None for p in model.backbone.xyz_decoder.parameters()):
        raise RuntimeError('flow-only objective unexpectedly traversed the AR decoder')
    optimizer.zero_grad(set_to_none=True)
    del features, loss, surface, geometry, noise
    wrapped = wrap(model, device, world)
    cases = list(base['capacity']['cases']) + [{'frames': t,
        'batch_per_rank': base['training']['frame_policy'][str(t)], 'stress_group': 'largest_joints'} for t in (6, 12, 24)]
    results = []
    for number, case in enumerate(cases):
        for step in range(config['capacity']['optimizer_steps_per_case']):
            if shared_deadline_due(config, device, world):
                raise RuntimeError('capacity deadline reached')
            indices = groups[case['stress_group']]
            batches = [[FrameSampleRequest(indices[j % len(indices)], case['frames'],
                1000000 + number * 10000 + step * 1000 + rank * 100 + k * 30 + j)
                for j in range(case['batch_per_rank'])] for k in range(config['capacity']['accumulation_microbatches'])]
            value = update(model, wrapped, optimizer, dataset, batches, tokenizer, codec, base, config,
                1000, device, world, check_groups=True)
            value.update(case=case, optimizer_step=step + 1)
            results.append(value)
            write_json(output / f'progress_rank_{rank}.json', {'complete': False, 'cases': results})
            print('CAPACITY ' + json.dumps({k: v for k, v in value.items() if k != 'microbatches'}), flush=True)
    model.eval()
    request = [FrameSampleRequest(groups['largest_joints'][rank], 24, 80000 + rank)]
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        surface, _, geometry = prepare(dataset, request, base, tokenizer, codec, device)
        features, _ = model.backbone.encode(surface)
        reference, noise, _ = flow_inputs(geometry, request, config, device)
        prediction = model.flow.generate(features, surface['query_points'], surface['query_normals'],
            reference_slots(geometry['frame_target_joints'], reference), geometry['target_parents'],
            geometry['joint_mask'], reference, noise=noise, steps=config['evaluation']['steps'])
    free = {'frames': 24, 'joint_count': int(geometry['joint_mask'].sum()),
        'steps': config['evaluation']['steps'], 'finite': bool(torch.isfinite(prediction).all())}
    ranks = gather({'rank': rank, 'flow_only_gradients': flow_gradient, 'cases': results, 'free_generation': free}, world)
    return {'complete': True, 'passed': True, 'source': source, 'scientific_sha256': scientific_sha(config),
        'initialization': initial, 'ranks': ranks, 'profile_weights_discarded': True, 'not_model_quality': True}


def mirror(path, relative, config):
    target = Path(config['artifacts']['durable_root']) / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise RuntimeError('refuse to overwrite retained durable artifact')
    temporary = target.with_name(target.name + '.tmp')
    shutil.copyfile(path, temporary)
    digest = sha256(path)
    if sha256(temporary) != digest:
        raise RuntimeError('durable mirror digest mismatch')
    temporary.replace(target)
    return {'path': str(path), 'durable_path': str(target), 'sha256': digest, 'bytes': path.stat().st_size}


def save(model, optimizer, stream, config, output, initial, source, samples, frames, step, device, rank, world):
    ranks = gather({'rank': rank, 'stream': stream.state_dict(), **random_state()}, world)
    spec = None
    if rank == 0:
        path = output / f'checkpoint_{samples:06d}.pt'
        if path.exists():
            raise RuntimeError('refuse to overwrite full checkpoint')
        payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rank_states': ranks,
            'optimizer_parameter_names': parameter_names(model), 'initialization': initial,
            'additional_asset_exposures': samples,
            'global_backbone_asset_exposures': config['training'].get('backbone_exposure_offset', 20000) + samples,
            'input_frame_exposures': frames, 'step': step, 'configuration': config,
            'scientific_sha256': scientific_sha(config), 'code_commit': source}
        temp = path.with_suffix('.pt.tmp')
        torch.save(payload, temp)
        temp.replace(path)
        spec = {**mirror(path, path.name, config), 'additional_asset_exposures': samples,
            'global_backbone_asset_exposures': config['training'].get('backbone_exposure_offset', 20000) + samples, 'complete': True,
            'both_copies_verified': True, 'code_commit': source}
        write_json(path.with_suffix('.json'), spec)
        write_json(Path(spec['durable_path']).with_suffix('.json'), spec)
    return gather(spec, world)[0]


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row['split'], row['reference_kind'], row['method'])
        groups.setdefault(key, []).append(row)
    result = []
    for key, values in sorted(groups.items()):
        assets = sorted({v['asset_id'] for v in values})
        item = dict(zip(('split', 'reference_kind', 'method'), key))
        item.update(assets=len(assets), rows=len(values), successful_rows=sum(v['reference_success'] for v in values))
        for metric in ('edge_f1', 'normalized_j2j', 'normalized_mpjpe', 'pck_005'):
            per_asset = [np.mean([v[metric] for v in values if v['asset_id'] == asset and v.get(metric) is not None])
                         for asset in assets if any(v['asset_id'] == asset and v.get(metric) is not None for v in values)]
            if per_asset:
                item[metric] = float(np.mean(per_asset))
                item[metric + '_scored_assets'] = len(per_asset)
        result.append(item)
    return result


@torch.no_grad()
def evaluate(model, tokenizer, base, config, entries, output, samples, device, rank, world):
    model.eval()
    final = samples == config['training']['additional_assets']
    panel = evaluation_rows(base, entries)
    natural_ids = {r['asset_id'] for r in panel}
    valid = entries['valid'] if final else panel
    train = sorted(entries['train'], key=lambda r: sample_seed(config['evaluation']['seed'], r['asset_id'], 'train-eval'))[:32]
    splits = [('valid', valid, 'seen_eval')]
    if final:
        splits += [('train_seen', train, 'seen_eval'), ('train_heldout', train, 'heldout_eval')]
    directory = output / f'evaluation_{samples:06d}'
    directory.mkdir(parents=True, exist_ok=True)
    rows, references, examples = [], [], []
    for split, selected, mode in splits:
        dataset = fixed_dataset(selected, base, config['evaluation']['input_frames'], mode=mode)
        for index in range(rank, len(selected), world):
            batch = dataset[index]
            surface, geometry = prepare_virtual_surface([batch], device, **base['surface_sampling'])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                features, virtual = model.backbone.encode(surface)
                natural = split == 'valid' and batch['asset_id'] in natural_ids and (samples == 0 or final)
                conditions = model.backbone.conditions(features, virtual) if natural else None
            target = geometry['frame_target_joints'][0]
            truth_parents = geometry['target_parents'][0].tolist()
            vertices = batch['frame_vertices'][0]
            diagonals = torch.linalg.vector_norm(vertices.amax(1) - vertices.amin(1), dim=-1)
            raw = geometry['selected_frames'][0].tolist()
            heldout = batch['heldout_raw_ids']
            if (mode == 'heldout_eval' and not set(raw) <= set(heldout)) or (mode != 'heldout_eval' and set(raw) & set(heldout)):
                raise RuntimeError('evaluation raw-frame partition leakage')
            for ref in config['evaluation']['reference_slots']:
                reference = torch.tensor([ref], device=device)
                incoming = {'GT': {'success': True, 'xyz': target[ref], 'parents': geometry['target_parents'][0]}}
                if natural:
                    incoming['predicted'] = generate_reference(model.backbone, tokenizer, conditions[:, ref])
                for kind, record in incoming.items():
                    common = {'split': split, 'asset_id': batch['asset_id'], 'reference_kind': kind,
                        'reference_slot': ref, 'reference_raw_frame': raw[ref], 'gt_joint_count': len(truth_parents)}
                    if kind == 'predicted':
                        meta = {k: v for k, v in record.items() if k not in ('xyz', 'parents')}
                        if record['success']:
                            meta.update(joint_count=len(record['parents']),
                                **row_metrics(record['xyz'], record['parents'].tolist(), target[ref], truth_parents,
                                              float(diagonals[ref]), paired_ids=False))
                        else:
                            meta.update(edge_f1=0.)
                        references.append({**common, **meta})
                    if not record['success']:
                        for frame in range(target.shape[0]):
                            if frame != ref:
                                for method in ('flow', 'copy'):
                                    rows.append({**common, 'target_slot': frame, 'method': method,
                                        'reference_success': False, 'edge_f1': 0., 'normalized_j2j': None})
                        continue
                    xyz = record['xyz'][None].to(device)
                    parents = record['parents'][None].to(device)
                    mask = torch.ones_like(parents, dtype=torch.bool)
                    noise = noise_for((1, target.shape[0], xyz.shape[1], 3), config['evaluation']['seed'],
                        batch['asset_id'], f'{split}:{kind}:{ref}', device)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        prediction = model.flow.generate(features, surface['query_points'], surface['query_normals'],
                            xyz, parents, mask, reference, noise=noise, steps=config['evaluation']['steps'])[0]
                    if len(examples) < 8:
                        examples.append({**common, 'prediction': prediction.cpu(), 'reference_xyz': xyz[0].cpu(),
                            'parents': parents[0].cpu(), 'truth': target.cpu(), 'truth_parents': truth_parents,
                            'points': surface['query_points'][0].cpu(), 'raw_frames': raw})
                    for frame in range(target.shape[0]):
                        if frame == ref:
                            continue
                        for method, value in (('flow', prediction[frame]), ('copy', xyz[0])):
                            rows.append({**common, 'target_slot': frame, 'target_raw_frame': raw[frame], 'method': method,
                                'predicted_joint_count': xyz.shape[1], 'reference_success': True,
                                **row_metrics(value, parents[0].tolist(), target[frame], truth_parents,
                                    float(diagonals[frame]), paired_ids=kind == 'GT')})
            del features, surface, geometry, conditions
            if (index - rank) // world % 16 == 0:
                write_json(directory / f'progress_rank_{rank}.json', {'split': split, 'processed_local': (index - rank) // world + 1,
                    'total_split': len(selected), 'rows': len(rows)})
    write_json(directory / f'rows_rank_{rank}.json', rows)
    torch.save(examples, directory / f'examples_rank_{rank}.pt')
    all_rows = sum(gather(rows, world), [])
    all_references = sum(gather(references, world), [])
    summary = {'complete': True, 'additional_asset_exposures': samples, 'final': final,
        'groups': summarize(all_rows), 'first_stage_reference_groups': summarize([
            {**r, 'method': 'AR', 'reference_success': r['success']} for r in all_references]),
        'scoring': 'asset_macro_mean; reference_pose_excluded; no_GT_alignment; one_seeded_sample',
        'failed_reference_edge_f1': 0, 'distance_metrics_conditional_on_success': True,
        'scientific_sha256': scientific_sha(config)}
    if rank == 0:
        expected_valid = 967 if final else 128
        gt = [r for r in all_rows if r['split'] == 'valid' and r['reference_kind'] == 'GT']
        if len(gt) != expected_valid * 2 * 7 * 2 or len({r['asset_id'] for r in gt}) != expected_valid:
            raise RuntimeError('validation evaluation coverage differs')
        write_json(directory / 'rows.json', all_rows)
        write_json(directory / 'references.json', all_references)
        write_json(directory / 'summary.json', summary)
        for name in ('rows.json', 'references.json', 'summary.json'):
            mirror(directory / name, Path(directory.name) / name, config)
    dist.barrier(device_ids=[device.index])
    return summary


def train(config, output, source, device, rank, world):
    base, entries = read_all(config)
    model, optimizer, tokenizer, codec, base, initial, stream, samples, frames, step = load_model(config, device, rank, world)
    checkpoints, evaluations = [], []
    with preserve_random():
        evaluations.append(evaluate(model, tokenizer, base, config, entries, output, samples, device, rank, world))
    wrapped = wrap(model, device, world)
    dataset = fixed_dataset(entries['train'], base, 2, mode='train')
    end = config['training']['additional_assets']
    stopped = False
    trace_path = output / f'updates_rank_{rank}.jsonl'
    while samples < end:
        boundary = next(v for v in config['training']['checkpoint_assets'] if v > samples)
        with OrderedAssetPrefetch(dataset, planned_microbatches(base, stream, boundary),
                workers=config['runtime']['data_workers_per_rank'], depth=config['runtime']['prefetch_microbatches']) as loader:
            while samples < boundary:
                if shared_deadline_due(config, device, world):
                    stopped = True
                    break
                goal, batches = next_update_batches(stream, samples, world)
                value = update(model, wrapped, optimizer, dataset, batches, tokenizer, codec, base, config,
                    samples, device, world, prefetch=loader, check_groups=step == 0)
                samples += goal
                frames += value['frames_in_update']
                step += 1
                value.update(step=step, additional_asset_exposures=samples, global_backbone_asset_exposures=20000 + samples,
                    input_frame_exposures=frames, code_commit=source)
                with trace_path.open('a') as handle:
                    handle.write(json.dumps(value) + '\n')
                progress = {k: v for k, v in value.items() if k != 'microbatches'}
                write_json(output / f'progress_rank_{rank}.json', progress)
                if rank == 0:
                    print('UPDATE ' + json.dumps(progress), flush=True)
            # Save consumed progress before waiting for any prefetched work to close.
            checkpoints.append(save(model, optimizer, stream, config, output, initial, source,
                samples, frames, step, device, rank, world))
        if stopped:
            break
        with preserve_random():
            evaluations.append(evaluate(model, tokenizer, base, config, entries, output, samples, device, rank, world))
        gc.collect()
        torch.cuda.empty_cache()
    return {'complete': not stopped and samples == end, 'operation_complete': not stopped and samples == end,
        'stopped_for_deadline': stopped, 'additional_asset_exposures': samples,
        'global_backbone_asset_exposures': 20000 + samples, 'train_assets': 5000,
        'source': source, 'scientific_sha256': scientific_sha(config), 'checkpoints': checkpoints,
        'evaluations': evaluations, 'quality_accepted': False,
        'interpretation': 'full5k first joint-flow trial; no matched AR-only continuation control'}
