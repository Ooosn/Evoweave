"""One data/optimization contract for both fixed-5k virtual-path arms."""
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from fusion_evaluation import tree_fingerprint
from id_xyz_stages import preserve_random, random_state
from probe_virtual_query_lambda import parameter_fingerprint
from structure_xyz_fill_runtime import verify_data_selection, write_json
from train_virtual_query_xyz import load_capacity_selection, prepare
from train_virtual_fusion_normal import record_samples
from virtual_frame_runtime import exposure_learning_rate
from virtual_fusion_normal_runtime import MatchedFrameStream
from virtual_pair_guard import ROOT, SCOPE, deadline_due
from virtual_query_xyz_runtime import build_model as build_base, fixed_dataset, read_entries, sha256


VARIANTS = ('virtual_on', 'virtual_off')
TRAIN_SHA = 'a17b6be96fb33a32a1bee7117b07789da6261c6b5693104f3d2cce07c09bab88'
VALID_SHA = 'fd2be7b5885833946e68dc186300a820757ae652c487749eec1b049d381e230a'


def validate_config(config):
    data, plan, optim = config['data'], config['training'], config['optimization']
    if config.get('family') != SCOPE or config['pair']['variants'] != list(VARIANTS):
        raise ValueError('not the declared matched pair')
    if [(data['manifests'][key]['rows'], data['manifests'][key]['sha256']) for key in ('train', 'valid')] != [
            (5000, TRAIN_SHA), (967, VALID_SHA)]:
        raise ValueError('both arms require the exact immutable 5000-asset training set')
    if (data['frames_min'], data['frames_max'], data['heldout_frames']) != (2, 24, 8):
        raise ValueError('dynamic input/heldout partition changed')
    if (data['normalization'] != 'input_sequence_mesh_bbox' or data['global_rt_removal'] is not True
            or data['coordinate_clipping'] is not False):
        raise ValueError('data normalization/RT/no-clipping changed')
    if (plan['target_assets'], plan['effective_assets_per_update'], plan['segment_assets']) != (100000, 48, 5000):
        raise ValueError('each arm needs100k exposures with batch48 and5k scheduling segments')
    if (plan['checkpoint_assets'] != list(range(5000, 100001, 5000))
            or plan['evaluation_assets'] != [25000, 50000, 75000, 100000]):
        raise ValueError('checkpoint/evaluation milestones changed')
    if optim != {'schedule': 'onecycle', 'peak_lr': .0001, 'warmup_fraction': .1, 'div_factor': 5,
                 'final_div_factor': 10, 'weight_decay': 0, 'schedule_samples': 100000,
                 'clip_grad_norm': 1, 'auxiliary_weight': 0}:
        raise ValueError('matched optimizer schedule changed')
    if (config['pair']['auxiliary_weight'] != 0 or config['pair']['consistency_weight'] != 0
            or config['pair']['legacy_codec'] is not True
            or config['pair']['same_asset_pose_sampling_plan'] is not True
            or config['pair']['all_real_frames_supervised'] is not True
            or config['pair']['common_post_attention_ffn'] is not True
            or config['architecture']['fusion_mode'] != 'real_queries_virtual'):
        raise ValueError('this pair changes only virtual interaction/readout, not codec/losses')
    if plan['frame_policy'] != {str(t): 24 if t <= 6 else 12 if t <= 12 else 6 for t in range(2, 25)}:
        raise ValueError('training batches exceed the declared capacity buckets')
    if (config['evaluation']['valid_assets'], config['evaluation']['input_frames'],
            config['evaluation']['frame_indices']) != (128, 8, list(range(8))):
        raise ValueError('shared128-asset validation panel changed')
    if not config['artifacts']['preserve_all_checkpoints']:
        raise ValueError('checkpoint preservation is mandatory')
    make_stream(config, 0, 2)


def scientific_contract(config):
    value = deepcopy({key: config[key] for key in ('family', 'initialization', 'architecture',
        'auxiliary_codec', 'surface_sampling', 'data', 'normal_match', 'optimization', 'training',
        'evaluation', 'capacity', 'pair')})
    value['training'].pop('resume_root', None)
    return value


def contract_sha(config):
    return hashlib.sha256(json.dumps(scientific_contract(config), sort_keys=True).encode()).hexdigest()


def source_hashes():
    paths = ['model_training/rigweave/scripts/' + name for name in (
        'virtual_pair_runtime.py', 'train_virtual_pair.py', 'virtual_query_xyz_runtime.py',
        'virtual_fusion_normal_runtime.py', 'virtual_frame_runtime.py', 'probe_virtual_query_xyz.py',
        'id_xyz_stages.py', 'train_virtual_query_xyz.py', 'train_virtual_fusion_normal.py')]
    paths += ['model_training/rigweave/src/rigweave/dynamic_rig/' + name for name in (
        'virtual_path_ablation.py', 'virtual_query_xyz.py', 'motion_encoder.py', 'motion_evidence.py',
        'surface_tokenizer.py', 'virtual_frame_data.py', 'virtual_query_xyz_data.py', 'root_rt.py',
        'aux_skeleton_tokens.py', 'sampling.py', 'skeleton_topology_metrics.py')]
    return {path: sha256(ROOT / path) for path in paths}


def make_stream(config, rank, world, state=None):
    return MatchedFrameStream(range(5000), rank=rank, world_size=world,
        seed=config['data']['stream_seed'], target_assets=config['training']['target_assets'],
        effective_assets=48, frame_policy=config['training']['frame_policy'], manifest_sha256=TRAIN_SHA,
        state=state, update_boundaries=config['training']['checkpoint_assets'])


def gather(value, world):
    result = [None] * world
    if world > 1:
        dist.all_gather_object(result, value)
    else:
        result[0] = value
    return result


def synchronize(device, world):
    if world > 1:
        dist.barrier(device_ids=[device.index])


def shared_deadline_due(config, device, world):
    due = torch.tensor(int(deadline_due(config)), device=device)
    if world > 1:
        dist.all_reduce(due, op=dist.ReduceOp.MAX)
    return bool(due)


class EvaluationHandoff(RuntimeError):
    pass


def build_model(config, variant, device):
    from rigweave.dynamic_rig.virtual_path_ablation import VirtualPathAblationRig
    if variant not in VARIANTS:
        raise ValueError('unknown pair arm')
    base, tokenizer, codec, initial = build_base(config, device)
    base_fp = parameter_fingerprint(base)
    if base_fp != config['pair']['official_base_initial_sha256']:
        raise ValueError('full base initialization differs from the accepted official-initialized reference')
    model = VirtualPathAblationRig(base, virtual_enabled=variant == 'virtual_on')
    initial.update(variant=variant, base_initial_sha256=base_fp,
        wrapped_initial_sha256=parameter_fingerprint(model),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        dormant_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
        disabled_parameter_names=[name for name, p in model.named_parameters() if not p.requires_grad],
        configuration=model.configuration())
    return model, tokenizer, codec, initial


def optimizer_for(model, config):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=config['optimization']['peak_lr'] / config['optimization']['div_factor'],
        betas=(.95, .999), weight_decay=config['optimization']['weight_decay'])


def parameter_names(model):
    return [name for name, p in model.named_parameters() if p.requires_grad]


def set_lr(optimizer, samples, config):
    p = config['optimization']
    lr, beta = exposure_learning_rate(samples, p['schedule_samples'], peak=p['peak_lr'],
        warmup_fraction=p['warmup_fraction'], div_factor=p['div_factor'], final_div_factor=p['final_div_factor'])
    for group in optimizer.param_groups:
        group['lr'], group['betas'] = lr, (beta, .999)
    return lr, beta


def reviewed(config, name):
    state = json.loads((ROOT / 'model_training/state/current.json').read_text())
    spec = state['active_operation'][name]
    if spec.get('accepted') is not True or sha256(spec['path']) != spec['sha256']:
        raise RuntimeError(f'{name} is not the exact primary-accepted receipt')
    report = json.loads(Path(spec['path']).read_text())
    if (not report.get('passed') or not report.get('complete')
            or report['contract_sha256'] != contract_sha(config) or report['source_sha256'] != source_hashes()):
        raise RuntimeError(f'{name} computation or data contract differs')
    return report


def evaluation_rows(config, entries):
    spec = config['evaluation']['selection']
    if sha256(spec['path']) != spec['sha256']:
        raise ValueError('matched validation selection changed')
    selected = json.loads(Path(spec['path']).read_text())['valid']
    lookup = {row['asset_id']: row for row in entries['valid']}
    if len(selected) != 128 or len({r['asset_id'] for r in selected}) != 128:
        raise ValueError('validation membership differs')
    for row in selected:
        actual = lookup[row['asset_id']]
        if actual['path'] != row['path'] or actual['root_rt'] != row['root_rt']:
            raise ValueError('validation asset identity/RT changed')
    return [lookup[row['asset_id']] for row in selected]


def audit(config, entries, output, source):
    verify_data_selection(config, entries)
    groups, lengths = load_capacity_selection(config, entries['train'])
    selected = evaluation_rows(config, entries)
    initial = {}
    for variant in VARIANTS:
        model, _, _, report = build_model(config, variant, 'cpu')
        initial[variant] = report
        del model
        gc.collect()
    if initial['virtual_on']['wrapped_initial_sha256'] != initial['virtual_off']['wrapped_initial_sha256']:
        raise RuntimeError('pair does not start with identical common state')
    plans = [make_stream(config, r, 2).plan_report() for r in range(2)]
    if plans[0]['plan_sha256'] != plans[1]['plan_sha256']:
        raise RuntimeError('distributed sampling plans disagree')
    return {'complete': True, 'passed': True, 'mode': 'audit', 'training_assets': 5000,
        'validation_pool': 967, 'validation_panel': len(selected), 'initialization': initial,
        'capacity_groups': groups, 'capacity_length_records': len(lengths), 'sampling_plan': plans[0],
        'source': source, 'source_sha256': source_hashes(), 'contract_sha256': contract_sha(config),
        'cuda_initialized': torch.cuda.is_initialized(), 'optimizer_created': False}


def check_gradient_groups(model, variant):
    groups = {'surface': model.surface_tokenizer, 'motion': model.motion_encoder,
              'condition_ffn': model.frame_fusion, 'condition_norm': model.condition_norm,
              'decoder': model.xyz_decoder}
    report = {}
    for name, module in groups.items():
        gradients = [p.grad.detach().float() for p in module.parameters() if p.requires_grad and p.grad is not None]
        if not gradients or any(not bool(torch.isfinite(value).all()) for value in gradients):
            raise RuntimeError(f'missing/nonfinite {name} gradients')
        value = sum(float(g.square().sum()) for g in gradients) ** .5
        if value <= 0:
            raise RuntimeError(f'zero {name} gradients')
        report[name] = value
    virtual_grad = model.motion_encoder.virtual_slots.grad
    if variant == 'virtual_on' and (virtual_grad is None or not bool(torch.isfinite(virtual_grad).all())
            or float(virtual_grad.float().norm()) <= 0):
        raise RuntimeError('enabled virtual slots lack a finite learning signal')
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise RuntimeError('disabled branch received gradients')
    report['virtual_slots'] = None if virtual_grad is None else float(virtual_grad.float().norm())
    return report


def wrap(model, device, world):
    return DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True,
        gradient_as_bucket_view=True) if world > 1 else model


def update(model, wrapped, optimizer, dataset, batches, tokenizer, codec, config, variant,
           samples, device, world, *, check_groups=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    lr, beta = set_lr(optimizer, samples, config)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    schedule = []
    for i, requests in enumerate(batches):
        surface, target, geometry = prepare(dataset, requests, config, tokenizer, codec, device)
        with wrapped.no_sync() if world > 1 and i < len(batches) - 1 else nullcontext():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                losses = wrapped(surface, target, auxiliary_weight=0.)
            if not bool(torch.isfinite(losses['loss'])):
                raise RuntimeError('nonfinite pair objective')
            losses['loss_per_asset'].sum().backward()
        size, frames = len(requests), requests[0].frames
        totals += torch.stack((losses['main_loss'].detach() * size, losses['main_accuracy'].detach() * size,
            totals.new_tensor(size), totals.new_tensor(size * frames)))
        schedule.append({**record_samples(geometry, requests), 'frames': frames,
            'target_sha256': hashlib.sha256(target['main_input_ids'].cpu().numpy().tobytes()).hexdigest()})
        del surface, target, geometry, losses
    if world > 1:
        dist.all_reduce(totals)
    assets = int(totals[2])
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(world / assets)
    gradients = check_gradient_groups(model, variant) if check_groups else None
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
    optimizer.step()
    torch.cuda.synchronize(device)
    memory = {'max_allocated_bytes': torch.cuda.max_memory_allocated(device),
              'max_reserved_bytes': torch.cuda.max_memory_reserved(device)}
    if memory['max_reserved_bytes'] > .85 * torch.cuda.get_device_properties(device).total_memory:
        raise RuntimeError('pair update exceeded approved memory fraction')
    return {'assets_in_update': assets, 'frames_in_update': int(totals[3]),
        'main_loss': float(totals[0] / assets), 'main_accuracy': float(totals[1] / assets),
        'learning_rate': lr, 'beta1': beta, 'gradient_norm': norm, 'gradient_groups': gradients,
        'seconds': time.monotonic() - started, 'microbatches': schedule, **memory}


def capacity(config, entries, output, source, device, rank, world):
    from rigweave.dynamic_rig.frame_batch_sampler import FrameSampleRequest
    approved = reviewed(config, 'CPU_audit')
    dataset = fixed_dataset(entries['train'], config, 2, mode='train')
    results = []
    for variant in VARIANTS:
        model, tokenizer, codec, initial = build_model(config, variant, device)
        if initial != approved['initialization'][variant]:
            raise RuntimeError('GPU initialization differs from CPU audit')
        optimizer = optimizer_for(model, config)
        wrapped = wrap(model, device, world)
        for number, case in enumerate(config['capacity']['cases']):
            if shared_deadline_due(config, device, world):
                raise RuntimeError('capacity reached allocation handoff deadline')
            indices = approved['capacity_groups'][case['stress_group']]
            for step in range(config['capacity']['optimizer_steps_per_case']):
                batches = [[FrameSampleRequest(indices[j % len(indices)], case['frames'],
                    (config['data']['stream_seed'] << 32) + 1000000 + number * 10000 + step * 1000 + rank * 100 + k * 30 + j)
                    for j in range(case['batch_per_rank'])] for k in range(config['capacity']['accumulation_microbatches'])]
                value = update(model, wrapped, optimizer, dataset, batches, tokenizer, codec, config,
                    variant, 0, device, world, check_groups=True)
                value.update(variant=variant, case=case, optimizer_step=step + 1, rank=rank,
                    capacity_only_not_training=True)
                results.append(value)
                write_json(output / f'progress_rank_{rank}.json', {'complete': False, 'cases': results})
                print('CAPACITY ' + json.dumps({k: v for k, v in value.items() if k != 'microbatches'}), flush=True)
        del wrapped, optimizer, model, tokenizer, codec
        gc.collect()
        torch.cuda.empty_cache()
        synchronize(device, world)
    ranks = gather({'rank': rank, 'cases': results}, world)
    return {'complete': True, 'passed': True, 'mode': 'capacity', 'ranks': ranks,
        'cases': config['capacity']['cases'], 'frame_policy': config['training']['frame_policy'],
        'source': source, 'source_sha256': source_hashes(), 'contract_sha256': contract_sha(config),
        'not_model_quality': True, 'profile_weights_discarded': True}


def checkpoint_candidates(root, variant):
    directory = Path(root) / variant
    return [json.loads(path.read_text()) for path in sorted(directory.glob('checkpoint_*.json'))]


def latest_checkpoint(output, config, variant):
    rows = checkpoint_candidates(output, variant)
    if config['training'].get('resume_root'):
        rows += checkpoint_candidates(config['training']['resume_root'], variant)
    return max(rows, key=lambda row: row['asset_exposures']) if rows else None


def evaluation_complete(output, config, variant, samples):
    roots = [output]
    if config['training'].get('resume_root'):
        roots.append(Path(config['training']['resume_root']))
    for root in roots:
        path = root / variant / f'evaluation_{samples:06d}' / 'summary.json'
        if path.exists():
            result = json.loads(path.read_text())
            if (result.get('complete') is True and result.get('samples') == samples
                    and result.get('variant') == variant and result.get('contract_sha256') == contract_sha(config)):
                return True
    return False


def pending_variants(progress, output, config):
    return [variant for variant in VARIANTS if progress[variant] < config['training']['target_assets']
        or not evaluation_complete(output, config, variant, config['training']['target_assets'])]


def restore_checkpoint(model, optimizer, config, spec, variant, rank, world):
    if sha256(spec['path']) != spec['sha256']:
        raise RuntimeError('checkpoint file digest differs')
    payload = torch.load(spec['path'], map_location='cpu', mmap=True, weights_only=False)
    if (payload['variant'] != variant or payload['contract_sha256'] != contract_sha(config)
            or payload['source_sha256'] != source_hashes()
            or payload['optimizer_parameter_names'] != parameter_names(model)
            or len(payload['rank_states']) != world):
        raise RuntimeError('checkpoint scientific/optimizer/rank contract differs')
    model.load_state_dict(payload['model'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    if (parameter_fingerprint(model) != payload['model_sha256']
            or tree_fingerprint(optimizer.state_dict()) != payload['optimizer_sha256']):
        raise RuntimeError('model or Adam restoration is not exact')
    saved = payload['rank_states'][rank]
    if saved['rank'] != rank:
        raise RuntimeError('rank-state identity differs')
    stream = make_stream(config, rank, world, saved['stream'])
    samples, frames, step = (payload[key] for key in ('asset_exposures', 'supervised_frame_exposures', 'step'))
    if stream.cursor * world != samples:
        raise RuntimeError('saved sampler cursor differs from optimizer progress')
    return stream, saved, payload['initialization'], samples, frames, step


def apply_rng(saved):
    torch.set_rng_state(saved['torch_rng'])
    torch.cuda.set_rng_state(saved['cuda_rng'])
    np.random.set_state(saved['numpy_rng'])
    random.setstate(saved['python_rng'])


def save_checkpoint(model, optimizer, stream, config, output, variant, initial, source,
                    samples, frames, step, rank, world, device):
    ranks = gather({'rank': rank, 'stream': stream.state_dict(), **random_state()}, world)
    spec = None
    if rank == 0:
        directory = output / variant
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f'checkpoint_{samples:06d}.pt'
        if path.exists() or path.with_suffix('.json').exists():
            raise RuntimeError('refuse to overwrite a retained checkpoint')
        model_fp, optimizer_fp = parameter_fingerprint(model), tree_fingerprint(optimizer.state_dict())
        payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rank_states': ranks,
            'optimizer_parameter_names': parameter_names(model), 'variant': variant,
            'asset_exposures': samples, 'input_frame_exposures': frames, 'supervised_frame_exposures': frames,
            'step': step, 'initialization': initial, 'configuration': config,
            'contract_sha256': contract_sha(config), 'source_sha256': source_hashes(), 'code_commit': source,
            'model_sha256': model_fp, 'optimizer_sha256': optimizer_fp}
        temporary = path.with_suffix('.pt.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)
        digest = sha256(path)
        durable = Path(config['artifacts']['durable_root']) / variant / path.name
        durable.parent.mkdir(parents=True, exist_ok=True)
        if durable.exists():
            raise RuntimeError('refuse to overwrite durable retained checkpoint')
        temporary = durable.with_suffix('.pt.tmp')
        shutil.copyfile(path, temporary)
        if sha256(temporary) != digest:
            raise RuntimeError('durable checkpoint copy digest mismatch')
        temporary.replace(durable)
        spec = {'complete': True, 'path': str(path), 'durable_path': str(durable), 'sha256': digest,
            'bytes': path.stat().st_size, 'asset_exposures': samples, 'step': step,
            'supervised_frame_exposures': frames, 'model_sha256': model_fp, 'optimizer_sha256': optimizer_fp,
            'contract_sha256': contract_sha(config), 'variant': variant, 'both_copies_verified': True}
        write_json(path.with_suffix('.json'), spec)
        write_json(durable.with_suffix('.json'), spec)
    values = gather(spec, world)
    synchronize(device, world)
    return values[0]


def evaluate(model, tokenizer, codec, config, entries, output, variant, samples, device, rank, world):
    from probe_virtual_query_xyz import evaluate as original_evaluate
    selected = evaluation_rows(config, entries)
    local = selected[rank::world]
    dataset = fixed_dataset(local, config, 8, mode='seen_eval')
    root = output / variant / f'evaluation_{samples:06d}'
    directory = root / f'rank_{rank}'
    directory.mkdir(parents=True, exist_ok=False)
    def items():
        for i in range(len(dataset)):
            if shared_deadline_due(config, device, world):
                raise EvaluationHandoff('saved training state retained; evaluation reached deadline')
            yield prepare(dataset, [i], config, tokenizer, codec, device)
    result = original_evaluate(model, items(), tokenizer,
        {**config, 'probe': {'auxiliary_weight': 0., 'evaluate_auxiliary': False, 'max_new_tokens': 2048}},
        directory, samples, frame_indices=config['evaluation']['frame_indices'])
    for row in result['rows']:
        row['gt_joint_count'] = len(row['target_quantized_joints'])
    all_rows = [row for values in gather(result['rows'], world) for row in values]
    expected = {(row['asset_id'], f) for row in selected for f in range(8)}
    if len(all_rows) != 1024 or {(r['asset_id'], r['frame_index']) for r in all_rows} != expected:
        raise RuntimeError('paired validation coverage differs')
    if rank == 0:
        summary = {}
        for name, rows in [('all', all_rows), ('non52', [r for r in all_rows if r['gt_joint_count'] != 52]),
                           ('gt52', [r for r in all_rows if r['gt_joint_count'] == 52])]:
            summary[name] = {'views': len(rows), 'assets': len({r['asset_id'] for r in rows}),
                'original_edge_f1': float(np.mean([r['original_edge_f1'] for r in rows])) if rows else None,
                'failures': sum(not r['success'] for r in rows),
                'exact_count_fraction': float(np.mean([r['exact_count'] for r in rows])) if rows else None}
        write_json(root / 'summary.json', {'complete': True, 'variant': variant, 'samples': samples,
            'groups': summary, 'quality_accepted': False, 'failed_views_scored_zero': True,
            'original_GT_scoring': True, 'oracle_not_used': True, 'contract_sha256': contract_sha(config)})
    synchronize(device, world)


def train_pair(config, entries, output, source, device, rank, world):
    approved = reviewed(config, 'CPU_audit')
    measured = reviewed(config, 'capacity')
    if measured['frame_policy'] != config['training']['frame_policy'] or measured['cases'] != config['capacity']['cases']:
        raise RuntimeError('training physical batches were not measured')
    dataset = fixed_dataset(entries['train'], config, 2, mode='train')
    checkpoints = {variant: latest_checkpoint(output, config, variant) for variant in VARIANTS}
    progress = {variant: checkpoints[variant]['asset_exposures'] if checkpoints[variant] else 0 for variant in VARIANTS}
    finished = False
    with (output / f'trace_rank_{rank}.jsonl').open('x', buffering=1) as trace:
        while pending_variants(progress, output, config):
            if shared_deadline_due(config, device, world):
                break
            variant = min(pending_variants(progress, output, config), key=lambda key: progress[key])
            model, tokenizer, codec, initial = build_model(config, variant, device)
            if initial != approved['initialization'][variant]:
                raise RuntimeError('formal initial weights differ from the CPU audit')
            optimizer = optimizer_for(model, config)
            stream, saved = make_stream(config, rank, world), None
            samples, frames, step = 0, 0, 0
            if checkpoints[variant]:
                stream, saved, initial, samples, frames, step = restore_checkpoint(
                    model, optimizer, config, checkpoints[variant], variant, rank, world)
                write_json(output / f'restore_{variant}_{samples:06d}_rank_{rank}.json',
                    {'model_and_Adam_exact': True, 'asset_exposures': samples, 'step': step,
                     'checkpoint': checkpoints[variant], 'sampler_plan_sha256': stream.plan_sha256})
            wrapped = wrap(model, device, world)
            if saved is not None:
                apply_rng(saved)
            evaluation_interrupted = False
            if (samples in config['training']['evaluation_assets']
                    and not evaluation_complete(output, config, variant, samples)):
                try:
                    with preserve_random():
                        evaluate(model, tokenizer, codec, config, entries, output, variant, samples, device, rank, world)
                except EvaluationHandoff:
                    evaluation_interrupted = True
            limit = min(((samples // 5000) + 1) * 5000, config['training']['target_assets'])
            first_step, last_saved = step, samples if checkpoints[variant] else None
            write_json(output / f'initialization_{variant}_rank_{rank}.json', initial)
            while samples < limit and not evaluation_interrupted:
                if shared_deadline_due(config, device, world):
                    break
                index = int(np.searchsorted(stream.global_ends, samples, side='right'))
                goal = int(stream.update_sizes[index])
                batches, count = [], 0
                while count < goal // world:
                    requests = stream.take(goal // world - count)
                    batches.append(requests)
                    count += len(requests)
                value = update(model, wrapped, optimizer, dataset, batches, tokenizer, codec,
                    config, variant, samples, device, world, check_groups=step == first_step)
                if value['assets_in_update'] != goal:
                    raise RuntimeError('actual update differs from the matched exposure plan')
                samples += goal
                frames += value['frames_in_update']
                step += 1
                value.update(variant=variant, rank=rank, step=step, asset_exposures=samples,
                    input_frame_exposures=frames, supervised_frame_exposures=frames,
                    sampling_plan_sha256=stream.plan_sha256)
                trace.write(json.dumps(value, allow_nan=False) + '\n')
                progress[variant] = samples
                write_json(output / f'progress_rank_{rank}.json', {'complete': False, 'progress': progress,
                    'latest': {k: v for k, v in value.items() if k != 'microbatches'}})
                print('TRAIN ' + json.dumps({k: v for k, v in value.items() if k != 'microbatches'}), flush=True)
            if last_saved != samples:
                checkpoints[variant] = save_checkpoint(model, optimizer, stream, config, output, variant,
                    initial, source, samples, frames, step, rank, world, device)
            if (samples in config['training']['evaluation_assets']
                    and not evaluation_complete(output, config, variant, samples)
                    and not shared_deadline_due(config, device, world)):
                try:
                    with preserve_random():
                        evaluate(model, tokenizer, codec, config, entries, output, variant, samples, device, rank, world)
                except EvaluationHandoff:
                    pass
            del wrapped, optimizer, model, tokenizer, codec, stream, saved
            gc.collect()
            torch.cuda.empty_cache()
            synchronize(device, world)
        finished = not pending_variants(progress, output, config)
    return {'passed': True, 'complete': finished, 'stage_complete': True, 'mode': 'train',
        'allocation_handoff_required': not finished, 'progress': progress, 'checkpoints': checkpoints,
        'target_per_variant': config['training']['target_assets'], 'training_assets': 5000,
        'quality_accepted': False, 'contract_sha256': contract_sha(config), 'source_sha256': source_hashes()}
