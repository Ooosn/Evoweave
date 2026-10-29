"""Audited full-data joint training with fresh Adam, exact resume and bounded retention."""
import gc
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist

from anchor_full_data import CachedFiles, audit_plan, make_stream, planned_microbatches, stage_cache, stress_groups
from anchor_full_cleanup import execute_legacy
from anchor_full_evaluation import canonical_input, eager_backbone, evaluate
from anchor_full_protocol import (full_checkpoints_to_keep, scientific_sha, sha256,
    source_hashes, validate_config, validate_resume)
from anchor_joint_runtime import head_digest, reference_indices, update
from fusion_evaluation import tree_fingerprint
from id_xyz_stages import preserve_random, random_state
from reference_flow_runtime import mirror, read_all
from stage1_feedback import generate_asset
from train_virtual_query_xyz import prepare
from virtual_pair_loading import OrderedAssetPrefetch
from virtual_pair_runtime import (apply_rng, audit_loader, evaluation_rows, gather, next_update_batches,
    optimizer_for, parameter_names, shared_deadline_due, target_selection, wrap)
from virtual_query_xyz_runtime import build_model, fixed_dataset, write_json
from rigweave.dynamic_rig.anchor_joint_binding import AnchorJointBinding, AnchorJointConfig, SharedAnchorJointRig
from rigweave.dynamic_rig.frame_batch_sampler import FrameSampleRequest
from rigweave.dynamic_rig.reference_skeleton_xyz import reference_slots
from rigweave.dynamic_rig.virtual_path_ablation import VirtualPathAblationRig


def release():
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def check_teacher_prefix(backbone, features, targets):
    condition = backbone.conditions(features, None)[:, 0]
    ids = targets['main_input_ids'][:1, 0, :5]
    full = backbone.main_logits(condition, ids, torch.ones_like(ids))
    prefix = torch.cat((condition, backbone.xyz_decoder.get_input_embeddings()(ids[:, :2])), dim=1)
    cached = backbone.xyz_decoder(inputs_embeds=prefix,
        attention_mask=torch.ones(prefix.shape[:2], dtype=torch.long, device=ids.device), use_cache=True)
    errors = []
    for slot in range(1, ids.shape[1]):
        if slot > 1:
            cached = backbone.xyz_decoder(input_ids=ids[:, slot:slot+1], past_key_values=cached.past_key_values,
                attention_mask=torch.ones((1, condition.shape[1]+slot+1), dtype=torch.long, device=ids.device), use_cache=True)
        error = float((cached.logits[:, -1]-full[:, slot]).abs().max())
        errors.append(error)
        if not torch.allclose(cached.logits[:, -1], full[:, slot], atol=1e-4, rtol=1e-4):
            raise RuntimeError('teacher-forcing and cached-generation prefix logits differ')
    return {'passed': True, 'checked_positions': len(errors), 'max_absolute_logit_difference': max(errors),
        'atol': 1e-4, 'rtol': 1e-4, 'precision': 'fp32_noTF32', 'batch_per_rank': 1}


def capacity_requests(base, indices, rank, frames, size, seed):
    maximum = max(2, int(.4 * frames))
    result = []
    for index in range(size):
        for attempt in range(1000):
            request = FrameSampleRequest(indices[(rank+index) % len(indices)], frames,
                seed + index*1000 + attempt)
            if len(target_selection(base, [request])[0]) == maximum:
                result.append(request)
                break
        else:
            raise RuntimeError('could not construct maximum-AR-density capacity request')
    return result


def initialize(config, device):
    base = validate_config(config)
    original, tokenizer, codec, official = build_model(base, device)
    backbone = VirtualPathAblationRig(original, virtual_enabled=False)
    devices = [device.index] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config['training']['head_seed'])
        head = AnchorJointBinding(AnchorJointConfig(**config['architecture']), 'reference_transport').to(device)
    model = SharedAnchorJointRig(backbone, head, ar_enabled=True)
    optimizer = optimizer_for(backbone, base)
    optimizer.add_param_group({'params': list(head.parameters())})
    if optimizer.state or len(optimizer.param_groups) != 2:
        raise RuntimeError('new full-data optimizer is not empty')
    initial = {'official': official, 'project_checkpoint_loaded': False,
        'project_optimizer_loaded': False, 'optimizer_initial_state_entries': 0,
        'initial_asset_exposures': 0, 'learning_rate_asset_offset': 0,
        'initial_model_sha256': head_digest(model), 'head_initial_sha256': head_digest(head),
        'optimizer_parameter_names': parameter_names(model), 'AR_and_XYZ_active_from_start': True}
    return model, optimizer, tokenizer, codec, base, initial


def load_model(config, device, rank, world, resume=None):
    model, optimizer, tokenizer, codec, base, initial = initialize(config, device)
    stream = make_stream(base, rank, world)
    samples, frames, step, best = 0, 0, 0, {}
    seed = config['training']['stream_seed'] + rank
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if resume:
        if rank == 0 and sha256(resume['path']) != resume['sha256']:
            raise RuntimeError('full-run resume digest differs')
        dist.barrier(device_ids=[device.index])
        saved = torch.load(resume['path'], map_location='cpu', mmap=True, weights_only=False)
        validate_resume(config, saved, parameter_names(model), source_hashes(), world)
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        stream = make_stream(base, rank, world, saved['rank_states'][rank]['stream'])
        apply_rng(saved['rank_states'][rank])
        samples, frames, step = (saved[k] for k in ('additional_asset_exposures', 'input_frame_exposures', 'step'))
        initial, best = saved['initialization'], saved.get('best', {})
        if stream.cursor * world != samples or step != int(np.searchsorted(stream.global_ends, samples)) + 1:
            raise RuntimeError('restored full-data update count differs')
        best_path = Path(resume['path']).parent/'best_state.json'
        if best_path.exists():
            latest = json.loads(best_path.read_text())
            if latest['scientific_sha256'] == scientific_sha(config) and latest['samples'] <= samples:
                best = latest['best']
        del saved
    return model, optimizer, tokenizer, codec, base, initial, stream, samples, frames, step, best


def audit(config, output, source):
    base, entries = read_all(config)
    groups, counts = stress_groups(entries['train'])
    panel = evaluation_rows(base, entries)
    cache = stage_cache(config, base, entries, output)
    resolver = CachedFiles(config, base)
    plan = audit_plan(base)
    model, optimizer, _, _, _, initial = initialize(config, torch.device('cpu'))
    del model, optimizer
    release()
    loader = audit_loader(base, entries, groups, file_resolver=resolver)
    if torch.cuda.is_initialized():
        raise RuntimeError('full-data CPU audit initialized CUDA')
    cleanup = execute_legacy(config['execution']['cleanup_plan'], output)
    result = {'complete': True, 'passed': True, 'source': source, 'source_sha256': source_hashes(),
        'scientific_sha256': scientific_sha(config), 'train_assets': len(entries['train']),
        'valid_assets': len(entries['valid']), 'validation_panel': len(panel),
        'maximum_train_joints': max(counts), 'capacity_groups': groups, 'plan': plan,
        'initialization': initial, 'cache': cache, 'loader': loader, 'legacy_cleanup': cleanup, 'cuda_initialized': False}
    write_json(output/'result.json', result)
    return result


def checked_evidence(config, key):
    spec = config['execution'][key]
    if not spec or sha256(spec['path']) != spec['sha256']:
        raise RuntimeError('full-data execution evidence is missing or changed: '+key)
    result = json.loads(Path(spec['path']).read_text())
    resources = json.loads(Path(spec['resources_path']).read_text())
    if (not result.get('complete') or not result.get('passed')
            or result['scientific_sha256'] != scientific_sha(config)
            or result['source_sha256'] != source_hashes()
            or not resources.get('complete') or not resources.get('passed')
            or resources.get('exit_code') != 0 or not resources.get('payload_cleanup_complete')):
        raise RuntimeError('full-data primary evidence is not accepted: '+key)
    return result


def save_checkpoint(model, optimizer, stream, config, output, initial, source,
                    samples, frames, step, best, device, rank, world):
    ranks = gather({'rank': rank, 'stream': stream.state_dict(), **random_state()}, world)
    spec = None
    if rank == 0:
        path = output/f'checkpoint_{samples:06d}.pt'
        if path.exists():
            raise RuntimeError('refuse to overwrite a full-run checkpoint')
        payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rank_states': ranks,
            'optimizer_parameter_names': parameter_names(model), 'initialization': initial,
            'additional_asset_exposures': samples, 'global_backbone_asset_exposures': samples,
            'input_frame_exposures': frames, 'step': step, 'best': best,
            'configuration': config, 'scientific_sha256': scientific_sha(config),
            'code_commit': source, 'source_sha256': source_hashes()}
        temp = path.with_suffix('.pt.tmp')
        torch.save(payload, temp)
        temp.replace(path)
        verified = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        validate_resume(config, verified, parameter_names(model), source_hashes(), world)
        if verified['step'] != step or not verified['optimizer']['state']:
            raise RuntimeError('saved full-run checkpoint is incomplete')
        del verified, payload
        spec = {**mirror(path, path.name, config), 'samples': samples, 'step': step,
            'scientific_sha256': scientific_sha(config), 'complete': True, 'both_copies_verified': True,
            'optimizer_and_rank_RNG_retained': True, 'code_commit': source}
        write_json(path.with_suffix('.json'), spec)
        write_json(Path(spec['durable_path']).with_suffix('.json'), spec)
    return gather(spec, world)[0]


def export_weights(model, config, output, source, checkpoint):
    path = output/f"weights_{checkpoint['samples']:06d}.pt"
    if path.exists():
        return json.loads(path.with_suffix('.json').read_text())
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    payload = {'model': state, 'configuration': config, 'initialization': 'official_UniRig_then_full_joint',
        'samples': checkpoint['samples'], 'scientific_sha256': scientific_sha(config),
        'code_commit': source, 'source_sha256': source_hashes(), 'model_only': True,
        'source_checkpoint_sha256': checkpoint['sha256']}
    temporary = path.with_suffix('.pt.tmp')
    torch.save(payload, temporary)
    checked = torch.load(temporary, map_location='cpu', mmap=True, weights_only=False)
    if (set(checked['model']) != set(state)
            or any(not torch.equal(value, checked['model'][name]) for name, value in state.items())):
        raise RuntimeError('model-only export differs from the checkpoint weights')
    del checked, payload, state
    temporary.replace(path)
    spec = {**mirror(path, path.name, config), 'samples': checkpoint['samples'],
        'scientific_sha256': scientific_sha(config), 'complete': True, 'model_only': True,
        'source_checkpoint_sha256': checkpoint['sha256'], 'tensor_roundtrip_verified': True}
    write_json(path.with_suffix('.json'), spec)
    write_json(Path(spec['durable_path']).with_suffix('.json'), spec)
    return spec


def prune_checkpoints(config, output, best):
    retention = config['artifacts']['retention']
    durable = Path(config['artifacts']['durable_root']).resolve()
    full = {int(p.stem.split('_')[-1]): p for p in output.glob('checkpoint_*.pt')}
    keep_full = full_checkpoints_to_keep(full, retention)
    keep_weights = set(retention['weight_milestones']) | {v['samples'] for v in best.values()}
    removals = [path for n, path in full.items() if n not in keep_full]
    removals += [p for p in output.glob('weights_*.pt') if int(p.stem.split('_')[-1]) not in keep_weights]
    records = []
    for path in removals:
        spec = json.loads(path.with_suffix('.json').read_text())
        backup = Path(spec['durable_path'])
        if (path.resolve().parent != output.resolve() or backup.resolve().parent != durable
                or spec['scientific_sha256'] != scientific_sha(config) or not spec['complete']
                or path.stat().st_size != spec['bytes'] or backup.stat().st_size != spec['bytes']):
            raise RuntimeError('checkpoint pruning scope or verification failed')
        # Newer verified full checkpoints and selected weight exports already exist.
        path.unlink()
        backup.unlink()
        records.append({**spec, 'removed_under_declared_retention': True})
    if records:
        with (output/'retention.jsonl').open('a') as handle:
            for record in records:
                handle.write(json.dumps(record)+'\n')
        shutil_target = durable/'retention.jsonl'
        shutil_target.write_text((output/'retention.jsonl').read_text())
    return records


def capacity(config, output, source, device, rank, world):
    audited = checked_evidence(config, 'CPU_audit')
    base, entries = read_all(config)
    resolver = CachedFiles(config, base)
    if resolver.index_sha256 != audited['cache']['sha256']:
        raise RuntimeError('full SSD cache index changed after CPU audit')
    groups = audited['capacity_groups']
    dataset = fixed_dataset(entries['train'], base, 2, mode='train', file_resolver=resolver)
    model, optimizer, tokenizer, codec, base, initial, *_ = load_model(config, device, rank, world)
    if initial != audited['initialization']:
        raise RuntimeError('GPU official initialization differs from the CPU audit')
    wrapped = wrap(model, device, world)
    records = []
    one = [[FrameSampleRequest(groups['largest_joints'][rank], 2, 100000+rank)]]
    value = update(model, wrapped, optimizer, dataset, one, tokenizer, codec, base, config,
        1000, device, world, check_groups=True)
    value.pop('microbatches')
    value.update(profile='per_rank_batch1_joint_contract')
    records.append(value)
    for case in base['capacity']['cases']:
        indices, t, size = groups[case['stress_group']], case['frames'], case['batch_per_rank']
        for step in range(config['capacity']['optimizer_steps_per_case']):
            if shared_deadline_due(config, device, world):
                raise RuntimeError('capacity deadline reached')
            batches = [capacity_requests(base, indices, rank, t, size,
                200000+t*1000000+step*100000+k*30000) for k in range(24//size)]
            value = update(model, wrapped, optimizer, dataset, batches, tokenizer, codec, base, config,
                1000, device, world, check_groups=True)
            value.pop('microbatches')
            value.update(**case, profile='full_manifest_extreme', optimizer_step=step+1,
                max_AR_targets_per_asset=max(2, int(.4*t)))
            records.append(value)
            write_json(output/f'progress_rank_{rank}.json', {'cases': records, 'complete': False})
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    eager, _ = eager_backbone(model, base, device)
    request = [FrameSampleRequest(groups['largest_joints'][rank], 24, 900000+rank)]
    surface, targets, geometry = prepare(dataset, request, base, tokenizer, codec, device)
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        features, _ = eager.encode(surface)
        prefix_contract = check_teacher_prefix(eager, features, targets)
        reference = reference_indices(request, config, device)
        xyz = reference_slots(geometry['frame_target_joints'], reference)
        prediction = model.head(features, surface['query_points'], surface['query_normals'], xyz,
            geometry['target_parents'], geometry['joint_mask'], reference)
        if not torch.isfinite(prediction).all() or not torch.equal(reference_slots(prediction, reference), xyz.float()):
            raise RuntimeError('full24-frame FP32 XYZ capacity failed')
        generated = generate_asset(eager, surface, tokenizer, frame_indices=[0], max_new_tokens=2048, precision='fp32')
    inference = {'frames': 24, 'joints': int(geometry['joint_mask'].sum()),
        'precision': 'fp32_noTF32', 'generation_status': generated[0]['status'],
        'generated_tokens': len(generated[0]['tokens']), 'XYZ_finite': True, 'teacher_prefix': prefix_contract}
    validation_inputs = []
    for rows, mode in [(evaluation_rows(base, entries), 'seen_eval'),
                       ([entries['train'][i] for i in groups['largest_joints']], 'heldout_eval')]:
        fixed = fixed_dataset(rows, base, 8, mode=mode, file_resolver=resolver)
        record, input_spec = canonical_input(fixed, rank, base, config, mode, device)
        repeated, repeated_spec = canonical_input(fixed, rank, base, config, mode, device)
        if tree_fingerprint(record) != tree_fingerprint(repeated) or input_spec != repeated_spec:
            raise RuntimeError('canonical validation input replay changed')
        with torch.no_grad():
            fixed_surface = {k: v.to(device) for k, v in record['surface'].items()}
            fixed_features, _ = eager.encode(fixed_surface)
            fixed_parents = record['parents'].to(device)[None]
            fixed_prediction = model.head(fixed_features, fixed_surface['query_points'], fixed_surface['query_normals'],
                record['truth'][0:1].to(device), fixed_parents, torch.ones_like(fixed_parents, dtype=torch.bool),
                torch.tensor([0], device=device))
            if not torch.isfinite(fixed_prediction).all():
                raise RuntimeError('canonical FP32 validation forward is nonfinite')
        validation_inputs.append({'mode': mode, 'asset_id': record['asset_id'], 'input_sha256': input_spec['sha256'],
            'raw_frames': record['raw_frames'], 'bitwise_replay': True})
        del record, repeated, fixed_surface, fixed_features, fixed_prediction
    inference['canonical_validation_inputs'] = validation_inputs
    del model, wrapped, optimizer, eager, surface, targets, geometry, features, xyz, prediction, generated
    release()
    model, optimizer, tokenizer, codec, base, initial, stream, samples, frames, step, best = load_model(config, device, rank, world)
    wrapped = wrap(model, device, world)
    representative = []
    limit = int(stream.global_ends[config['capacity']['representative_updates']-1])
    with OrderedAssetPrefetch(dataset, planned_microbatches(base, stream, limit),
            workers=config['runtime']['data_workers_per_rank'], depth=config['runtime']['prefetch_microbatches']) as loader:
        while samples < limit:
            goal, batches = next_update_batches(stream, samples, world)
            value = update(model, wrapped, optimizer, dataset, batches, tokenizer, codec, base, config,
                samples, device, world, prefetch=loader, check_groups=step == 0)
            samples += goal
            frames += value['frames_in_update']
            step += 1
            value.pop('microbatches')
            representative.append(value)
    optimizer.zero_grad(set_to_none=True)
    before = {'model': head_digest(model), 'optimizer': tree_fingerprint(optimizer.state_dict()),
        'RNG': tree_fingerprint(random_state()), 'stream': stream.state_dict()}
    profile_config = json.loads(json.dumps(config))
    profile_config['artifacts']['durable_root'] += '/capacity'
    spec = save_checkpoint(model, optimizer, stream, profile_config, output, initial, source,
        samples, frames, step, {}, device, rank, world)
    del model, wrapped, optimizer
    release()
    model, optimizer, _, _, _, _, stream, restored_samples, restored_frames, restored_step, _ = load_model(
        config, device, rank, world, spec)
    after = {'model': head_digest(model), 'optimizer': tree_fingerprint(optimizer.state_dict()),
        'RNG': tree_fingerprint(random_state()), 'stream': stream.state_dict()}
    if before != after or (samples, frames, step) != (restored_samples, restored_frames, restored_step):
        raise RuntimeError('full model/Adam/RNG/data resume roundtrip failed')
    ranks = gather({'rank': rank, 'cases': records, 'FP32_inference': inference,
        'representative_updates': representative, 'resume_roundtrip_exact': True}, world)
    result = {'complete': True, 'passed': True, 'source': source, 'source_sha256': source_hashes(),
        'scientific_sha256': scientific_sha(config), 'initialization': initial, 'ranks': ranks,
        'profile_weights_not_training_initialization': True, 'not_model_quality': True,
        'representative_mean_update_seconds': float(np.mean([v['seconds'] for v in representative])),
        'resume_checkpoint': spec}
    del model, optimizer
    release()
    return result


def train(config, output, source, device, rank, world):
    audited = checked_evidence(config, 'CPU_audit')
    checked_evidence(config, 'capacity')
    base, entries = read_all(config)
    resolver = CachedFiles(config, base)
    if resolver.index_sha256 != audited['cache']['sha256']:
        raise RuntimeError('full-data cache differs from admission')
    model, optimizer, tokenizer, codec, base, initial, stream, samples, frames, step, best = load_model(
        config, device, rank, world, config['training']['resume'])
    if initial != audited['initialization']:
        raise RuntimeError('formal model was not freshly initialized from official UniRig')
    write_json(output/f'initialization_rank_{rank}.json', initial)
    wrapped = wrap(model, device, world)
    dataset = fixed_dataset(entries['train'], base, 2, mode='train', file_resolver=resolver)
    stopped, last_checkpoint = False, None
    while samples < config['training']['additional_assets']:
        boundary = next(n for n in config['training']['checkpoint_assets'] if n > samples)
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
                value.update(step=step, additional_asset_exposures=samples,
                    global_backbone_asset_exposures=samples, input_frame_exposures=frames,
                    learning_rate_asset_position=samples-goal, regime='joint_official', code_commit=source)
                with (output/f'updates_rank_{rank}.jsonl').open('a') as handle:
                    handle.write(json.dumps(value)+'\n')
                progress = {k: v for k, v in value.items() if k != 'microbatches'}
                write_json(output/f'progress_rank_{rank}.json', progress)
                if rank == 0:
                    print('UPDATE '+json.dumps(progress), flush=True)
        if samples == 0:
            raise RuntimeError('deadline arrived before any training update')
        optimizer.zero_grad(set_to_none=True)
        if last_checkpoint is None or last_checkpoint.get('samples') != samples:
            last_checkpoint = save_checkpoint(model, optimizer, stream, config, output, initial, source,
                samples, frames, step, best, device, rank, world)
        if stopped:
            break
        with preserve_random():
            result = evaluate(model, base, config, entries, output, samples, device, rank, world,
                include_ar=samples % config['evaluation']['ar_interval_assets'] == 0)
        if not result['complete']:
            stopped = True
            break
        score = next(g['normalized_mpjpe'] for g in result['groups']
            if g['split'] == 'valid' and g['method'] == 'head')
        changed = False
        if 'stage2' not in best or score < best['stage2']['score']:
            best['stage2'] = {'score': score, 'samples': samples, 'metric': 'normalized_MPJPE_lower_better'}
            changed = True
        if result['stage1'] is not None:
            score = result['stage1']['original_edge_f1']
            if 'stage1' not in best or score > best['stage1']['score']:
                best['stage1'] = {'score': score, 'samples': samples, 'metric': 'edge_F1_higher_better'}
                changed = True
        if rank == 0:
            if changed or samples in config['artifacts']['retention']['weight_milestones']:
                weights = export_weights(model, config, output, source, last_checkpoint)
                for value in best.values():
                    if value['samples'] == samples:
                        value['weights'] = weights
            best_state = {'samples': samples, 'best': best, 'scientific_sha256': scientific_sha(config)}
            write_json(output/'best_state.json', best_state)
            write_json(Path(config['artifacts']['durable_root'])/'best_state.json', best_state)
            prune_checkpoints(config, output, best)
            print('VALIDATION '+json.dumps({'samples': samples, 'groups': result['groups'],
                'stage1': result['stage1'], 'best': best}), flush=True)
        best = gather(best if rank == 0 else None, world)[0]
        release()
    final = None
    if not stopped:
        with preserve_random():
            final = evaluate(model, base, config, entries, output, samples, device, rank, world, final=True)
        stopped = not final['complete']
    return {'complete': not stopped, 'passed': not stopped, 'stopped_for_deadline': stopped,
        'samples': samples, 'input_frame_visits': frames, 'optimizer_updates': step,
        'scientific_sha256': scientific_sha(config), 'source_sha256': source_hashes(),
        'source': source, 'initialization': initial, 'last_checkpoint': last_checkpoint,
        'best': best, 'final_evaluation': final}
