"""Controlled GT-reference association experiment with matched arms and diagnostics."""
from contextlib import nullcontext
import copy
import gc
import hashlib
import json
import time

import numpy as np
import torch
import torch.distributed as dist

from anchor_joint_protocol import VARIANTS, load_backbone_config, sample_seed, scientific_sha, sha256, should_extend
from fit_cohort_selection import _joint_count
from id_xyz_stages import preserve_random
from reference_flow_runtime import ar_indices, audit, mirror, norm_of, read_all, save, summarize
from reference_xyz_runtime import row_metrics
from train_virtual_fusion_normal import record_samples
from train_virtual_query_xyz import prepare, load_capacity_selection
from virtual_pair_loading import OrderedAssetPrefetch
from virtual_pair_runtime import (apply_rng, check_gradient_groups, evaluation_rows, gather, make_stream,
    next_update_batches, optimizer_for, parameter_names, planned_microbatches, set_lr, shared_deadline_due, wrap)
from virtual_query_xyz_runtime import build_model, fixed_dataset, write_json
from rigweave.dynamic_rig.anchor_joint_binding import AnchorJointBinding, AnchorJointConfig, SharedAnchorJointRig, coordinate_objective
from rigweave.dynamic_rig.frame_batch_sampler import FrameSampleRequest
from rigweave.dynamic_rig.reference_skeleton_xyz import reference_slots
from rigweave.dynamic_rig.virtual_frame_data import prepare_virtual_surface
from rigweave.dynamic_rig.virtual_path_ablation import VirtualPathAblationRig


def reference_indices(requests, config, device):
    return torch.tensor([sample_seed(config['training']['head_seed'], r.sampling_seed, 'reference') % r.frames
        for r in requests], device=device)


def head_digest(head):
    value = hashlib.sha256()
    for name, parameter in head.state_dict().items():
        value.update(name.encode())
        value.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def load_model(config, device, rank, world, resume=None):
    base, spec = load_backbone_config(config), config['backbone']['checkpoint']
    if rank == 0 and sha256(spec['path']) != spec['sha256']:
        raise ValueError('donor digest changed')
    dist.barrier(device_ids=[device.index])
    original, tokenizer, codec, official = build_model(base, device)
    backbone = VirtualPathAblationRig(original, virtual_enabled=False)
    payload = torch.load(spec['path'], map_location='cpu', mmap=True, weights_only=False)
    if (payload['variant'] != 'fraction20_40_query' or payload['asset_exposures'] != 20000
            or payload['code_commit'] != 'a0bda62bb8fba70320310b063da8c7898ddaf441'
            or payload['optimizer_parameter_names'] != parameter_names(backbone)):
        raise ValueError('donor identity changed')
    backbone.load_state_dict(payload['model'], strict=True)
    optimizer = optimizer_for(backbone, base)
    optimizer.load_state_dict(payload['optimizer'])
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(config['training']['head_seed'])
        head = AnchorJointBinding(AnchorJointConfig(**config['architecture']), config['training']['variant']).to(device)
    model = SharedAnchorJointRig(backbone, head)
    optimizer.add_param_group({'params': list(head.parameters())})
    apply_rng(payload['rank_states'][rank])
    initial = {'donor': spec, 'strict_complete_backbone_load': True, 'backbone_optimizer_retained': True,
        'head_optimizer_new': True, 'head_initial_sha256': head_digest(head),
        'head_parameters': sum(p.numel() for p in head.parameters()),
        'constructor_official_sha256': official['checkpoint_sha256']}
    del payload
    stream, samples, frames, step = make_stream(base, rank, world), 0, 0, 0
    if resume:
        if rank == 0 and sha256(resume['path']) != resume['sha256']:
            raise ValueError('resume digest changed')
        dist.barrier(device_ids=[device.index])
        saved = torch.load(resume['path'], map_location='cpu', mmap=True, weights_only=False)
        if (saved['scientific_sha256'] != scientific_sha(config)
                or saved['optimizer_parameter_names'] != parameter_names(model)
                or len(saved['rank_states']) != world):
            raise ValueError('resume science/parameters/ranks changed')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        stream = make_stream(base, rank, world, saved['rank_states'][rank]['stream'])
        apply_rng(saved['rank_states'][rank])
        samples, frames, step = (saved[k] for k in ('additional_asset_exposures', 'input_frame_exposures', 'step'))
        if stream.cursor * world != samples:
            raise ValueError('resume consumed stream differs')
        initial = saved['initialization']
        del saved
    return model, optimizer, tokenizer, codec, base, initial, stream, samples, frames, step


def update(model, wrapped, optimizer, dataset, batches, tokenizer, codec, base, config,
           samples, device, world, *, prefetch=None, check_groups=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    lr, beta = set_lr(optimizer, 20000 + samples, base)
    optimizer.param_groups[-1]['lr'] = lr * config['training']['head_lr_multiplier']
    goal = sum(map(len, batches)) * world
    weight = config['training']['coordinate_weight'] * min(1., (samples + goal) / config['training']['warmup_assets'])
    torch.cuda.reset_peak_memory_stats(device)
    start, records = time.monotonic(), []
    totals = torch.zeros(7, device=device, dtype=torch.float64)
    for i, requests in enumerate(batches):
        loaded = prefetch.take(requests)[0] if prefetch else dataset
        surface, targets, geometry = prepare(loaded, requests, base, tokenizer, codec, device)
        del loaded
        selected, indices, index_mask = ar_indices(base, requests, device)
        reference = reference_indices(requests, config, device)
        with wrapped.no_sync() if world > 1 and i < len(batches) - 1 else nullcontext():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                losses = wrapped(surface, targets, geometry, indices, index_mask, reference,
                    weight, config['training']['bone_weight'])
            if not bool(torch.isfinite(losses['loss_per_asset']).all()):
                raise RuntimeError('nonfinite combined objective')
            losses['loss_per_asset'].sum().backward()
        n = len(requests)
        totals += torch.stack((losses['ar_loss'].detach().sum(), losses['xyz_loss'].detach().sum(),
            losses['bone_loss'].detach().sum(), losses['main_accuracy'].detach() * n,
            totals.new_tensor(n), totals.new_tensor(n * requests[0].frames), totals.new_tensor(sum(map(len, selected)))))
        records.append({**record_samples(geometry, requests), 'frames': requests[0].frames,
            'ar_target_slots': selected, 'reference_slots': reference.tolist(),
            'joint_counts': geometry['joint_mask'].sum(1).tolist()})
        del surface, targets, geometry, losses
    dist.all_reduce(totals)
    assets = int(totals[4])
    if assets != goal:
        raise RuntimeError('global effective batch differs')
    for p in model.parameters():
        if p.grad is not None:
            p.grad.mul_(world / assets)
    groups = None
    if check_groups:
        groups = check_gradient_groups(model.backbone, 'fraction20_40_query')
        groups['head'] = norm_of(model.head)
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
    optimizer.step()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_reserved(device)
    if peak > config['runtime']['gpu_memory_fraction_limit'] * torch.cuda.get_device_properties(device).total_memory:
        raise RuntimeError('GPU memory limit exceeded')
    return {'assets_in_update': assets, 'frames_in_update': int(totals[5]), 'ar_targets': int(totals[6]),
        'ar_loss': float(totals[0] / assets), 'xyz_loss': float(totals[1] / assets),
        'bone_loss': float(totals[2] / assets), 'ar_accuracy': float(totals[3] / assets),
        'coordinate_weight': weight, 'lr': lr, 'head_lr': optimizer.param_groups[-1]['lr'], 'beta1': beta,
        'gradient_norm': norm, 'gradient_groups': groups, 'seconds': time.monotonic() - start,
        'max_allocated_bytes': torch.cuda.max_memory_allocated(device), 'max_reserved_bytes': peak,
        'microbatches': records}


def variant_config(config, variant):
    result = copy.deepcopy(config)
    result['training']['variant'] = variant
    result['artifacts']['durable_root'] += '/' + variant
    return result


def capacity(config, output, source, device, rank, world):
    base, entries = read_all(config)
    groups, _ = load_capacity_selection(base, entries['train'])
    counts = [_joint_count(r)[0] for r in entries['train']]
    groups['largest_joints'] = sorted(range(5000), key=lambda i: (-counts[i], i))[:3]
    run = variant_config(config, 'target_content')
    model, optimizer, tokenizer, codec, _, initial, *_ = load_model(run, device, rank, world)
    dataset = fixed_dataset(entries['train'], base, 2, mode='train')
    request = [FrameSampleRequest(groups['largest_joints'][rank], 2, 90000 + rank)]
    surface, _, geometry = prepare(dataset, request, base, tokenizer, codec, device)
    reference = reference_indices(request, config, device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        features, _ = model.backbone.encode(surface)
        prediction = model.head(features, surface['query_points'], surface['query_normals'],
            reference_slots(geometry['frame_target_joints'], reference), geometry['target_parents'], geometry['joint_mask'], reference)
        loss = coordinate_objective(prediction, geometry['frame_target_joints'], geometry['target_parents'],
            geometry['joint_mask'], reference)['loss_per_asset'].sum()
    loss.backward()
    gradients = {'surface': norm_of(model.backbone.surface_tokenizer),
        'motion': norm_of(model.backbone.motion_encoder), 'head': norm_of(model.head)}
    if any(p.grad is not None for p in model.backbone.xyz_decoder.parameters()):
        raise RuntimeError('coordinate-only loss unexpectedly traversed AR decoder')
    del surface, geometry, features, prediction, loss
    optimizer.zero_grad(set_to_none=True)
    wrapped = wrap(model, device, world)
    cases = copy.deepcopy(base['capacity']['cases']) + [dict(stress_group='largest_joints', frames=t,
        batch_per_rank=base['training']['frame_policy'][str(t)]) for t in (6, 12, 24)]
    records = []
    for mode in VARIANTS:
        model.head.mode = mode
        for case in cases:
            for step in range(2):
                if shared_deadline_due(config, device, world):
                    raise RuntimeError('capacity deadline reached')
                selected = groups[case['stress_group']]
                batches = [[FrameSampleRequest(selected[(rank + i) % len(selected)], case['frames'],
                    10000 + step * 100 + k * 10 + i) for i in range(case['batch_per_rank'])] for k in range(2)]
                value = update(model, wrapped, optimizer, dataset, batches, tokenizer, codec, base,
                    run, 1000, device, world, check_groups=True)
                value.pop('microbatches')
                value.update(mode=mode, case=case, optimizer_step=step + 1)
                records.append(value)
                write_json(output / f'progress_rank_{rank}.json', {'complete': False, 'cases': records})
    model.eval()
    request = [FrameSampleRequest(groups['largest_joints'][rank], 24, 80000 + rank)]
    free = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        surface, _, geometry = prepare(dataset, request, base, tokenizer, codec, device)
        features, _ = model.backbone.encode(surface)
        reference = reference_indices(request, config, device)
        xyz = reference_slots(geometry['frame_target_joints'], reference)
        for mode in VARIANTS:
            model.head.mode = mode
            prediction = model.head(features, surface['query_points'], surface['query_normals'], xyz,
                geometry['target_parents'], geometry['joint_mask'], reference)
            if not bool(torch.isfinite(prediction).all()) or not torch.equal(reference_slots(prediction, reference), xyz.float()):
                raise RuntimeError('free inference/reference clamp failed')
            free.append({'mode': mode, 'frames': 24, 'joint_count': int(geometry['joint_mask'].sum()), 'finite': True})
    ranks = gather({'rank': rank, 'coordinate_only_gradients': gradients, 'cases': records, 'free_inference': free}, world)
    return {'complete': True, 'passed': True, 'source': source, 'initialization': initial,
        'scientific_sha256': scientific_sha(config), 'ranks': ranks,
        'profile_weights_discarded': True, 'not_model_quality': True}


def joint_metrics(prediction, truth, parents, diagonal):
    metrics = row_metrics(prediction, parents, truth, parents, diagonal, paired_ids=True)
    p = torch.tensor(parents, device=prediction.device)
    valid = p >= 0
    if bool(valid.any()):
        pb = prediction[valid] - prediction[p[valid]]
        tb = truth[valid] - truth[p[valid]]
        metrics['normalized_bone_vector_error'] = float(torch.linalg.vector_norm(pb - tb, dim=-1).mean()) / diagonal
    else:
        metrics['normalized_bone_vector_error'] = 0.
    return metrics


@torch.no_grad()
def evaluate(model, base, config, entries, output, samples, device, rank, world, *, final=False):
    model.eval()
    panel = evaluation_rows(base, entries)
    panel_ids = {r['asset_id'] for r in panel}
    train = sorted(entries['train'], key=lambda r: sample_seed(config['evaluation']['seed'], r['asset_id'], 'train-eval'))[:32]
    splits = [('valid', entries['valid'] if final else panel, 'seen_eval')]
    if final:
        splits += [('train_eligible', train, 'seen_eval'), ('train_heldout', train, 'heldout_eval')]
    directory = output / (f'final_{samples:06d}' if final else f'evaluation_{samples:06d}')
    directory.mkdir(parents=True, exist_ok=True)
    rows, examples, consistency = [], [], []
    for split, selected, mode in splits:
        dataset = fixed_dataset(selected, base, 8, mode=mode)
        for index in range(rank, len(selected), world):
            batch = dataset[index]
            surface, geometry = prepare_virtual_surface([batch], device, **base['surface_sampling'])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                features, _ = model.backbone.encode(surface)
            target, parents = geometry['frame_target_joints'][0], geometry['target_parents'][0]
            mask, raw = geometry['joint_mask'], geometry['selected_frames'][0].tolist()
            vertices = batch['frame_vertices'][0]
            diagonals = torch.linalg.vector_norm(vertices.amax(1) - vertices.amin(1), dim=-1)
            heldout = set(batch['heldout_raw_ids'])
            if (mode == 'heldout_eval' and not set(raw) <= heldout) or (mode != 'heldout_eval' and set(raw) & heldout):
                raise RuntimeError('raw-frame partition leakage')
            predictions = []
            for ref in config['evaluation']['reference_slots']:
                reference = torch.tensor([ref], device=device)
                args = [features, surface['query_points'], surface['query_normals'], target[ref:ref+1],
                    parents[None], mask, reference]
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    prediction = model.head(*args)[0]
                predictions.append(prediction)
                methods = {'head': prediction, 'copy': target[ref:ref+1].expand_as(target)}
                if final and split == 'valid' and batch['asset_id'] in panel_ids:
                    gen = torch.Generator().manual_seed(sample_seed(config['evaluation']['seed'], batch['asset_id'], f'permute:{ref}'))
                    permutation = torch.randperm(features.shape[2], generator=gen).to(device)
                    changed = [v.clone() for v in args[:3]]
                    for tensor in changed:
                        for slot in range(8):
                            if slot != ref:
                                tensor[:, slot] = tensor[:, slot, permutation]
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        methods['broken_anchor_correspondence'] = model.head(*changed, *args[3:])[0]
                for method, values in methods.items():
                    for frame in range(8):
                        if frame == ref:
                            continue
                        rows.append({'split': split, 'asset_id': batch['asset_id'], 'selection_panel': batch['asset_id'] in panel_ids,
                            'reference_kind': 'GT', 'reference_slot': ref, 'reference_raw_frame': raw[ref],
                            'target_slot': frame, 'target_raw_frame': raw[frame], 'method': method,
                            'reference_success': True, 'joint_count': len(parents),
                            **joint_metrics(values[frame], target[frame], parents.tolist(), float(diagonals[frame]))})
                if final and split == 'valid' and len(examples) < 8:
                    examples.append({'asset_id': batch['asset_id'], 'reference_slot': ref,
                        'reference_kind': 'GT', 'prediction': prediction.cpu(), 'reference_xyz': target[ref].cpu(),
                        'truth': target.cpu(), 'parents': parents.cpu(), 'points': surface['query_points'][0].cpu(), 'raw_frames': raw})
            slots = [i for i in range(8) if i not in config['evaluation']['reference_slots']]
            discrepancy = torch.linalg.vector_norm(predictions[0][slots] - predictions[1][slots], dim=-1).mean(1)
            consistency.append({'split': split, 'asset_id': batch['asset_id'],
                'normalized_cross_reference_difference': float((discrepancy / diagonals[slots].to(device)).mean())})
            del features, surface, geometry, predictions
            if (index - rank) // world % 16 == 0:
                write_json(directory / f'progress_rank_{rank}.json', {'split': split, 'processed_local': (index-rank)//world+1,
                    'total_split': len(selected), 'rows': len(rows)})
    all_rows, all_consistency = sum(gather(rows, world), []), sum(gather(consistency, world), [])
    summary = {'complete': True, 'samples': samples, 'final': final, 'variant': model.head.mode,
        'groups': summarize(all_rows), 'reference': 'GT_only', 'scientific_sha256': scientific_sha(config),
        'diagnostic_scope': 'anchor permutation at stage2 interface only; backbone features kept, not a whole-model causal ablation'}
    if final:
        summary['nonselection839_groups'] = summarize([v for v in all_rows
            if v['split'] == 'valid' and not v['selection_panel']])
        if any(g['assets'] != 839 for g in summary['nonselection839_groups']):
            raise RuntimeError('nonselection validation coverage differs')
    for group in summary['groups']:
        selected = [v for v in all_rows if all(v[k] == group[k] for k in ('split', 'reference_kind', 'method'))]
        ids = sorted({v['asset_id'] for v in selected})
        group['normalized_bone_vector_error'] = float(np.mean([np.mean([v['normalized_bone_vector_error']
            for v in selected if v['asset_id'] == identity]) for identity in ids]))
    expected = 967 if final else 128
    valid = [v for v in all_rows if v['split'] == 'valid' and v['method'] == 'head']
    if len(valid) != expected * 14 or len({v['asset_id'] for v in valid}) != expected:
        raise RuntimeError('evaluation coverage differs')
    if final:
        torch.save(examples, directory / f'examples_rank_{rank}.pt')
    if rank == 0:
        for name, data in (('rows.json', all_rows), ('reference_consistency.json', all_consistency), ('summary.json', summary)):
            write_json(directory / name, data)
            mirror(directory / name, directory.name + '/' + name, config)
    dist.barrier(device_ids=[device.index])
    return summary


def train_segment(config, output, source, device, rank, world, end, resume):
    base, entries = read_all(config)
    model, optimizer, tokenizer, codec, base, initial, stream, samples, frames, step = load_model(config, device, rank, world, resume)
    output.mkdir(parents=True, exist_ok=True)
    evaluations, checkpoints = {}, []
    if samples == 0:
        with preserve_random():
            evaluations[0] = evaluate(model, base, config, entries, output, samples, device, rank, world)
    wrapped = wrap(model, device, world)
    dataset = fixed_dataset(entries['train'], base, 2, mode='train')
    stopped = False
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
                value.update(step=step, additional_asset_exposures=samples, input_frame_exposures=frames,
                    variant=model.head.mode, code_commit=source)
                with (output / f'updates_rank_{rank}.jsonl').open('a') as handle:
                    handle.write(json.dumps(value) + '\n')
                progress = {k: v for k, v in value.items() if k != 'microbatches'}
                write_json(output / f'progress_rank_{rank}.json', progress)
                if rank == 0:
                    print('UPDATE ' + json.dumps(progress), flush=True)
            checkpoints.append(save(model, optimizer, stream, config, output, initial, source,
                samples, frames, step, device, rank, world))
        if stopped:
            break
        with preserve_random():
            evaluations[samples] = evaluate(model, base, config, entries, output, samples, device, rank, world)
        gc.collect()
        torch.cuda.empty_cache()
    result = {'complete': not stopped, 'samples': samples, 'initialization': initial,
        'checkpoints': checkpoints, 'evaluations': evaluations, 'stopped_for_deadline': stopped}
    del model, wrapped, optimizer, dataset
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier(device_ids=[device.index])
    return result


def train(config, output, source, device, rank, world):
    results, resumes = {v: [] for v in VARIANTS}, {v: None for v in VARIANTS}
    decision, horizon = None, config['training']['minimum_assets']
    for end in range(config['training']['comparison_interval_assets'], 60001, config['training']['comparison_interval_assets']):
        if end > 40000 and not decision['extend_both_to']:
            break
        for variant in VARIANTS:
            run = variant_config(config, variant)
            result = train_segment(run, output / variant, source, device, rank, world, end, resumes[variant])
            results[variant].append(result)
            resumes[variant] = result['checkpoints'][-1]
            if rank == 0:
                write_json(output / 'pair_progress.json', {'complete': False, 'results': results, 'decision': decision})
            if not result['complete']:
                return {'complete': False, 'operation_complete': False, 'stopped_for_deadline': True, 'results': results}
        initial = [results[v][0]['initialization'] for v in VARIANTS]
        if initial[0] != initial[1]:
            raise RuntimeError('matched arms did not start from identical parameters/optimizer source')
        horizon = end
        if end == 40000:
            curves = {v: {n: next(g for segment in results[v] if n in segment['evaluations']
                for g in segment['evaluations'][n]['groups']
                if g['split'] == 'valid' and g['method'] == 'head') for n in (30000, 40000)} for v in VARIANTS}
            decision = should_extend(curves)
            if rank == 0:
                write_json(output / 'extension_decision.json', decision)
    finals = {}
    for variant in VARIANTS:
        run = variant_config(config, variant)
        model, optimizer, _, _, base, *_ = load_model(run, device, rank, world, resumes[variant])
        _, entries = read_all(run)
        with preserve_random():
            finals[variant] = evaluate(model, base, run, entries, output / variant, horizon, device, rank, world, final=True)
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    return {'complete': True, 'operation_complete': True, 'source': source, 'results': results,
        'decision': decision, 'common_final_samples_per_arm': horizon, 'final_evaluations': finals,
        'scientific_sha256': scientific_sha(config), 'quality_accepted': False,
        'interpretation': 'matched reference-association transport vs target-content read; GT references only; no AR-only causal claim'}
