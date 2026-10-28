"""AR-only continuation with inherited Adam and independent free-generation scoring."""
import gc
import json
import math
from pathlib import Path
import time

import torch
import torch.distributed as dist

from ar_only_protocol import (VARIANT, load_backbone_config, scientific_sha, sha256,
                              validate_donor, validate_resume)
from fusion_evaluation import tree_fingerprint
from id_xyz_stages import preserve_random
from reference_flow_runtime import audit as audit_data, read_all, save
from stage1_feedback import backbone_state, generate_asset, score_generation, summarize_rows
from stage1_feedback_analysis import compare
from train_virtual_query_xyz import prepare
from virtual_pair_loading import OrderedAssetPrefetch
from virtual_pair_runtime import (apply_rng, evaluation_rows, gather, make_stream, next_update_batches,
    optimizer_for, parameter_names, planned_microbatches, shared_deadline_due, target_selection,
    update as ar_update, wrap)
from virtual_query_xyz_runtime import build_model, fixed_dataset, write_json
from rigweave.dynamic_rig.frame_batch_sampler import FrameSampleRequest
from rigweave.dynamic_rig.virtual_frame_data import prepare_virtual_surface
from rigweave.dynamic_rig.virtual_path_ablation import VirtualPathAblationRig
from rigweave.dynamic_rig.virtual_query_xyz import original_xyz_vocabulary_processor


def joint_trace(config, rank):
    path = Path(config['execution']['matched_joint_trace_root'])/f'updates_rank_{rank}.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows = [row for row in rows if row['additional_asset_exposures'] <= 40000]
    if len(rows) != 840 or [r['step'] for r in rows] != list(range(1, 841)):
        raise ValueError('joint prefix is not the complete first840updates')
    if (rows[-1]['additional_asset_exposures'], rows[-1]['input_frame_exposures']) != (40000, 527856):
        raise ValueError('joint prefix exposure counters differ')
    return rows


def verify_actual_trace(value, expected):
    for key in ('step', 'additional_asset_exposures', 'input_frame_exposures',
                'assets_in_update', 'frames_in_update', 'beta1'):
        if value[key] != expected[key]:
            raise ValueError('actual AR/joint trace differs: '+key)
    if not math.isclose(value['learning_rate'], expected['lr'], rel_tol=0, abs_tol=1e-16):
        raise ValueError('AR/joint learning rate differs')
    if value['supervised_frames_in_update'] != expected['ar_targets']:
        raise ValueError('AR supervised frame count differs')
    left, right = value['microbatches'], expected['microbatches']
    if len(left) != len(right):
        raise ValueError('AR/joint microbatch count differs')
    for a, b in zip(left, right):
        for key in ('asset_ids', 'manifest_indices', 'sampling_seeds', 'selected_raw_frames', 'frames'):
            if a[key] != b[key]:
                raise ValueError('actual AR/joint data differs: '+key)
        if a['target_frame_indices'] != b['ar_target_slots']:
            raise ValueError('actual AR target choices differ')


def audit(config, output, source):
    result = audit_data(config, output, source)
    base = load_backbone_config(config)
    for rank in range(2):
        trace, stream, samples = joint_trace(config, rank), make_stream(base, rank, 2), 0
        for row in trace:
            goal, batches = next_update_batches(stream, samples, 2)
            if len(batches) != len(row['microbatches']):
                raise ValueError('planned/joint microbatch count differs')
            for batch, actual in zip(batches, row['microbatches']):
                if ([r.item_index for r in batch] != actual['manifest_indices']
                        or [r.sampling_seed for r in batch] != actual['sampling_seeds']
                        or any(r.frames != actual['frames'] for r in batch)
                        or target_selection(base, batch) != actual['ar_target_slots']):
                    raise ValueError('planned stream differs from actual joint prefix')
            samples += goal
            if samples != row['additional_asset_exposures']:
                raise ValueError('planned/joint asset count differs')
    if result['input_frame_exposures'] != 527856 or len(result['passes']) != 8:
        raise ValueError('AR continuation budget differs')
    for spec in config['comparators'].values():
        if sha256(spec['path']) != spec['sha256']:
            raise ValueError('comparison checkpoint changed')
    result.update(actual_joint_prefix_plans_verified=True, matched_updates_per_rank=840,
        cumulative_end=60000, first_stage_quality_measured=False)
    write_json(output/'result.json', result)
    return result


def load_model(config, device, rank, world, resume=None):
    base, spec = load_backbone_config(config), config['backbone']['checkpoint']
    if rank == 0 and sha256(spec['path']) != spec['sha256']:
        raise ValueError('donor digest changed')
    dist.barrier(device_ids=[device.index])
    original, tokenizer, codec, official = build_model(base, device)
    model = VirtualPathAblationRig(original, virtual_enabled=False)
    payload = torch.load(spec['path'], map_location='cpu', mmap=True, weights_only=False)
    validate_donor(payload, parameter_names(model), world)
    model.load_state_dict(payload['model'], strict=True)
    optimizer = optimizer_for(model, base)
    optimizer.load_state_dict(payload['optimizer'])
    optimizer_sha = tree_fingerprint(optimizer.state_dict())
    if optimizer_sha != tree_fingerprint(payload['optimizer']):
        raise ValueError('restored donor Adam differs')
    stream = make_stream(base, rank, world)
    apply_rng(payload['rank_states'][rank])
    initial = {'donor': spec, 'strict_complete_backbone_load': True,
        'backbone_optimizer_retained': True, 'optimizer_sha256': optimizer_sha,
        'donor_rank_RNG_restored': True, 'new_phase_stream_seed': 20261028,
        'donor_stream_not_resumed': True, 'stage2_parameters': 0,
        'constructor_official_sha256': official['checkpoint_sha256']}
    samples, frames, step = 0, 0, 0
    del payload
    if resume:
        if rank == 0 and sha256(resume['path']) != resume['sha256']:
            raise ValueError('AR resume digest changed')
        dist.barrier(device_ids=[device.index])
        saved = torch.load(resume['path'], map_location='cpu', mmap=True, weights_only=False)
        validate_resume(config, saved, parameter_names(model), world)
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        stream = make_stream(base, rank, world, saved['rank_states'][rank]['stream'])
        apply_rng(saved['rank_states'][rank])
        samples, frames, step = (saved[k] for k in ('additional_asset_exposures', 'input_frame_exposures', 'step'))
        if stream.cursor*world != samples:
            raise ValueError('resumed AR stream differs')
        initial = saved['initialization']
        del saved
    return model, optimizer, tokenizer, codec, base, initial, stream, samples, frames, step


def update(model, wrapped, optimizer, dataset, batches, tokenizer, codec, base, config,
           samples, device, world, **kwargs):
    if len(optimizer.param_groups) != 1 or hasattr(model, 'head'):
        raise ValueError('AR-only must not create or optimize a stage2 head')
    value = ar_update(model, wrapped, optimizer, dataset, batches, tokenizer, codec,
        base, VARIANT, samples+config['training']['learning_rate_asset_offset'], device, world, **kwargs)
    if value['assets_in_update'] != sum(map(len, batches))*world:
        raise ValueError('AR effective batch differs')
    value.update(ar_supervision_applied=True, stage2_forward_applied=False,
        learning_rate_asset_position=20000+samples)
    return value


def evaluation_identity(config, base, spec):
    return {'checkpoint_sha256': spec['sha256'], 'checkpoint_layout': spec['layout'],
        'decoding': config['evaluation'], 'selection_sha256': base['evaluation']['selection']['sha256'],
        'tokenizer_sha256': config['runtime']['tokenizer_config_sha256'],
        'data': base['data'], 'surface_sampling': base['surface_sampling'],
        'evaluation_sources': {name: sha256(Path(__file__).with_name(name))
            for name in ('ar_only_runtime.py', 'stage1_feedback.py', 'stage1_feedback_analysis.py')}}


def analyze_evaluations(evaluations):
    rows = {}
    for label, value in evaluations.items():
        rows[label] = []
        for spec in value['row_artifacts']:
            if sha256(spec['path']) != spec['sha256']:
                raise ValueError('evaluation rows changed before comparison')
            rows[label].extend(json.loads(line) for line in Path(spec['path']).read_text().splitlines())
    pairs = [('stage2_080000', 'donor'), ('joint_080000', 'donor'), ('joint_080000', 'stage2_080000')]
    pairs += [(f'joint_{n:06d}', f'ar_{n:06d}') for n in (10000, 20000, 30000, 40000)]
    return {a+'_minus_'+b: compare(rows[a], rows[b], evaluations[a]['identity'], evaluations[b]['identity'])
            for a, b in pairs}


def load_evaluation_model(config, device, spec):
    if sha256(spec['path']) != spec['sha256']:
        raise ValueError('evaluation checkpoint changed')
    base = load_backbone_config(config)
    # The established CPU constructor selects eager OPT attention, with identical weights/architecture.
    original, tokenizer, _, _ = build_model(base, torch.device('cpu'))
    model = VirtualPathAblationRig(original, virtual_enabled=False)
    if model.xyz_decoder.config._attn_implementation != config['evaluation']['decoder_attention']:
        raise ValueError('FP32 decoder backend differs')
    payload = torch.load(spec['path'], map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(backbone_state(payload, model.state_dict(), layout=spec['layout']), strict=True)
    del payload
    return model.to(device).eval(), tokenizer


@torch.inference_mode()
def evaluate(model, tokenizer, base, config, entries, output, label, spec, device, rank, world):
    selected = evaluation_rows(base, entries)
    dataset = fixed_dataset(selected, base, 8, mode='seen_eval')
    directory = output/'stage1'/label
    if rank == 0:
        directory.mkdir(parents=True, exist_ok=False)
    dist.barrier(device_ids=[device.index])
    rows, examples = [], []
    started = time.monotonic()
    for index in range(rank, len(selected), world):
        if shared_deadline_due(config, device, world):
            raise RuntimeError('stage1 evaluation reached allocation deadline; checkpoints retained')
        batch = dataset[index]
        surface, geometry = prepare_virtual_surface([batch], device, **base['surface_sampling'])
        raw = geometry['selected_frames'][0].tolist()
        if set(raw) & set(batch['heldout_raw_ids']):
            raise ValueError('stage1 panel entered reserved raw frames')
        inputs_sha = tree_fingerprint(surface)
        normalization_sha = tree_fingerprint({k: geometry[k] for k in ('query_center', 'query_scale')})
        count = int(geometry['joint_mask'][0].sum())
        truth = geometry['frame_target_joints'][0, :, :count]
        parents = geometry['target_parents'][0, :count]
        vertices = batch['frame_vertices'][0]
        diagonals = torch.linalg.vector_norm(vertices.amax(1)-vertices.amin(1), dim=-1)
        generated = generate_asset(model, surface, tokenizer, frame_indices=config['evaluation']['frame_indices'],
            max_new_tokens=config['evaluation']['max_new_tokens'], precision=config['evaluation']['precision'])
        asset_rows = []
        for frame in config['evaluation']['frame_indices']:
            row = score_generation(generated[frame], tokenizer, truth[frame].cpu().numpy(),
                parents.tolist(), float(diagonals[frame]))
            row.update(asset_id=batch['asset_id'], frame_index=frame, raw_frame_index=raw[frame],
                surface_sha256=inputs_sha, normalization_sha256=normalization_sha,
                original_GT_sha256=tree_fingerprint({'joints': truth[frame], 'parents': parents}))
            rows.append(row)
            asset_rows.append(row)
        if len(examples) < 8 or batch['asset_id'] in ('oxl_2aa3c8aa648c6cba21fb', 'oxl_24e8e5621802f809e815'):
            examples.append({'asset_id': batch['asset_id'], 'raw_frames': raw,
                'points': surface['query_points'][0].cpu(), 'truth': truth.cpu(),
                'parents': parents.cpu(), 'rows': asset_rows})
        with (directory/f'rows_rank_{rank}.jsonl').open('a') as handle:
            for row in asset_rows:
                handle.write(json.dumps(row)+'\n')
        write_json(directory/f'progress_rank_{rank}.json', {'label': label,
            'processed_local': (index-rank)//world+1, 'total_local': len(selected)//world,
            'seconds': time.monotonic()-started})
        del surface, geometry, generated, truth, parents
    all_rows = sum(gather(rows, world), [])
    expected = {(r['asset_id'], f) for r in selected for f in range(8)}
    if len(all_rows) != 1024 or {(r['asset_id'], r['frame_index']) for r in all_rows} != expected:
        raise ValueError('stage1 panel coverage differs')
    torch.save(examples, directory/f'examples_rank_{rank}.pt')
    row_path = directory/f'rows_rank_{rank}.jsonl'
    artifacts = gather({'rank': rank, 'path': str(row_path), 'sha256': sha256(row_path)}, world)
    result = {'complete': True, 'identity': evaluation_identity(config, base, spec),
        'label': label, 'summary': summarize_rows(all_rows), 'row_artifacts': artifacts,
        'seconds': time.monotonic()-started, 'no_GT_reference_or_teacher_prefix': True}
    if rank == 0:
        write_json(directory/'summary.json', result)
    return result


@torch.inference_mode()
def full_decode_capacity(model, surface, tokenizer, tokens):
    """Force maximum KV length for resource testing only, never score this output."""
    model.eval()
    old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.autocast('cuda', enabled=False):
            real, virtual = model.encode(surface)
            condition = model.conditions(real, virtual)[:, 0]
            start = torch.tensor([tokenizer.bos, tokenizer.cls_name_to_token('articulationxl')],
                                 device=condition.device, dtype=torch.long)
            original_processor = original_xyz_vocabulary_processor(tokenizer, start)
            def without_eos(ids, logits):
                values = original_processor(ids, logits)
                if not torch.isfinite(values).any():
                    raise RuntimeError('no finite decoder logits')
                values[:, tokenizer.eos] = -torch.inf
                if not torch.isfinite(values).any():
                    raise RuntimeError('no finite non-EOS decoder logits')
                return values
            value = model.generate_condition(condition, start, eos_token=tokenizer.eos,
                max_new_tokens=tokens, logits_processor=without_eos)
        if value['status'] != 'max_tokens' or len(value['tokens']) != tokens+len(start):
            raise ValueError('maximum-length decoder profile incomplete')
        return {'new_tokens': tokens, 'precision': 'fp32', 'tf32': False,
                'EOS_suppressed_for_capacity_only': True, 'not_model_quality': True}
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


def reviewed(config, name):
    spec = config['execution'][name]
    if not spec or sha256(spec['path']) != spec['sha256']:
        raise ValueError('missing or changed reviewed '+name)
    value = json.loads(Path(spec['path']).read_text())
    if not value['complete'] or not value['passed'] or value['scientific_sha256'] != scientific_sha(config):
        raise ValueError('unaccepted '+name)
    return value


def capacity(config, output, source, device, rank, world):
    accepted = reviewed(config, 'CPU_audit')
    base, entries = read_all(config)
    model, optimizer, tokenizer, codec, _, initial, *_ = load_model(config, device, rank, world)
    wrapped = wrap(model, device, world)
    dataset = fixed_dataset(entries['train'], base, 2, mode='train')
    cases = list(base['capacity']['cases']) + [{'frames': n,
        'batch_per_rank': base['training']['frame_policy'][str(n)], 'stress_group': 'largest_joints'} for n in (6, 12, 24)]
    results = []
    for number, case in enumerate(cases):
        for step in range(config['capacity']['optimizer_steps_per_case']):
            if shared_deadline_due(config, device, world):
                raise RuntimeError('capacity deadline reached')
            selected = accepted['capacity_groups'][case['stress_group']]
            batches = [[FrameSampleRequest(selected[j % len(selected)], case['frames'],
                1000000+number*10000+step*1000+rank*100+k*30+j)
                for j in range(case['batch_per_rank'])] for k in range(2)]
            value = update(model, wrapped, optimizer, dataset, batches, tokenizer, codec, base, config,
                0, device, world, check_groups=True)
            value.pop('microbatches')
            value.update(case=case, optimizer_step=step+1)
            results.append(value)
            write_json(output/f'progress_rank_{rank}.json', {'complete': False, 'cases': results})
    del model, wrapped, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    model, tokenizer = load_evaluation_model(config, device, config['comparators']['donor'])
    selected = accepted['capacity_groups']['largest_joints']
    surface, _, geometry = prepare(dataset, [FrameSampleRequest(selected[rank], 24, 90000+rank)],
                                   base, tokenizer, codec, device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    free = full_decode_capacity(model, surface, tokenizer, config['capacity']['full_decoder_tokens'])
    torch.cuda.synchronize(device)
    free.update(seconds=time.monotonic()-started, input_frames=24,
        max_reserved_bytes=torch.cuda.max_memory_reserved(device))
    if free['max_reserved_bytes'] > config['runtime']['gpu_memory_fraction_limit']*torch.cuda.get_device_properties(device).total_memory:
        raise RuntimeError('FP32 maximum decode exceeded memory budget')
    ranks = gather({'rank': rank, 'initialization': initial, 'cases': results, 'full_decode': free}, world)
    return {'complete': True, 'passed': True, 'source': source, 'scientific_sha256': scientific_sha(config),
        'ranks': ranks, 'profile_weights_discarded': True, 'not_model_quality': True}


def train(config, output, source, device, rank, world):
    reviewed(config, 'CPU_audit')
    reviewed(config, 'capacity')
    base, entries = read_all(config)
    model, optimizer, tokenizer, codec, _, initial, stream, samples, frames, step = load_model(
        config, device, rank, world, config['execution']['resume'])
    wrapped = wrap(model, device, world)
    dataset = fixed_dataset(entries['train'], base, 2, mode='train')
    trace = joint_trace(config, rank)
    checkpoints, evaluations, stopped = [], {}, False
    resumed_samples = samples
    while samples < 40000:
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
                value.update(step=step, additional_asset_exposures=samples, input_frame_exposures=frames,
                    global_backbone_asset_exposures=20000+samples, code_commit=source)
                verify_actual_trace(value, trace[step-1])
                value['actual_joint_trace_matched'] = True
                with (output/f'updates_rank_{rank}.jsonl').open('a') as handle:
                    handle.write(json.dumps(value)+'\n')
                write_json(output/f'progress_rank_{rank}.json', {k:v for k,v in value.items() if k != 'microbatches'})
            spec = save(model, optimizer, stream, config, output, initial, source,
                samples, frames, step, device, rank, world)
            checkpoints.append(spec)
        if stopped:
            break
        gc.collect()
        torch.cuda.empty_cache()
    del model, wrapped, optimizer, dataset
    gc.collect()
    torch.cuda.empty_cache()
    if not stopped:
        model, tokenizer = load_evaluation_model(config, device, config['comparators']['donor'])
        comparisons = dict(config['comparators'])
        for n in config['training']['comparison_assets']:
            folder = Path(config['execution']['resume']['path']).parent if n <= resumed_samples else output
            path = folder/f'checkpoint_{n:06d}.json'
            comparisons[f'ar_{n:06d}'] = {**json.loads(path.read_text()), 'layout':'density'}
        for label, spec in comparisons.items():
            if rank == 0 and sha256(spec['path']) != spec['sha256']:
                raise ValueError('comparison checkpoint changed')
            dist.barrier(device_ids=[device.index])
            payload = torch.load(spec['path'], map_location='cpu', mmap=True, weights_only=False)
            model.load_state_dict(backbone_state(payload, model.state_dict(), layout=spec['layout']), strict=True)
            del payload
            with preserve_random():
                evaluations[label] = evaluate(model, tokenizer, base, config, entries, output,
                    label, spec, device, rank, world)
            gc.collect()
            torch.cuda.empty_cache()
        analysis = analyze_evaluations(evaluations) if rank == 0 else None
        analysis = gather(analysis, world)[0]
        if rank == 0:
            write_json(output/'stage1_comparisons.json', analysis)
    else:
        analysis = None
    return {'complete': not stopped, 'operation_complete': not stopped, 'passed': not stopped,
        'stopped_for_deadline': stopped, 'source': source, 'scientific_sha256': scientific_sha(config),
        'additional_asset_exposures': samples, 'cumulative_asset_exposures': 20000+samples,
        'input_frame_exposures': frames, 'updates': step, 'initialization': initial,
        'checkpoints': checkpoints, 'evaluations': evaluations, 'paired_analysis': analysis,
        'quality_accepted': False}
