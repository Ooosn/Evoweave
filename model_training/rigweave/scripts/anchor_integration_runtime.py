"""Three matched initialization/objective regimes using the same binding head."""
import copy
import gc

import torch
import torch.distributed as dist

from anchor_integration_protocol import REGIMES, regime_config, scientific_sha
from anchor_joint_runtime import (evaluate, head_digest, load_model, read_all, reference_indices,
    train_segment, update)
from fit_cohort_selection import _joint_count
from reference_flow_runtime import audit as audit_data
from reference_flow_runtime import gather
from train_virtual_query_xyz import prepare
from virtual_fusion_normal_runtime import MatchedFrameStream
from virtual_pair_runtime import make_stream, shared_deadline_due, wrap
from virtual_query_xyz_runtime import fixed_dataset, write_json
from rigweave.dynamic_rig.frame_batch_sampler import FrameSampleRequest
from rigweave.dynamic_rig.reference_skeleton_xyz import reference_slots


def audit(config, output, source):
    result = audit_data(config, output, source)
    base, _ = read_all(config)
    old = copy.deepcopy(base)
    old['training']['target_assets'] = 60000
    old['training']['checkpoint_assets'] = list(range(5000, 60001, 5000))
    for rank in range(2):
        previous = make_stream(old, rank, 2)
        previous.cursor = previous.local_total
        extended = MatchedFrameStream.extend_completed(range(5000), previous.state_dict(),
            target_assets=80000, update_boundaries=config['training']['checkpoint_assets'])
        direct = make_stream(base, rank, 2)
        direct.cursor = previous.local_total
        if extended.state_dict() != direct.state_dict():
            raise RuntimeError('completed60k extension differs from the direct80k plan')
    result['completed60k_prefix_and_extension_verified'] = True
    result['regimes'] = list(REGIMES)
    result['AR_target_choices_are_planned_not_applied_in_stage2_only_arms'] = True
    write_json(output/'result.json', result)
    return result


def release():
    gc.collect()
    torch.cuda.empty_cache()


def capacity(config, output, source, device, rank, world):
    base, entries = read_all(config)
    counts = [_joint_count(r)[0] for r in entries['train']]
    selected = sorted(range(5000), key=lambda i: (-counts[i], i))[:3]
    dataset = fixed_dataset(entries['train'], base, 2, mode='train')
    initializations, records, free = {}, [], []
    for regime in REGIMES:
        run = regime_config(config, regime)
        model, optimizer, tokenizer, codec, _, initial, *_ = load_model(run, device, rank, world)
        initializations[regime] = initial
        wrapped = wrap(model, device, world)
        for frames in (6, 12, 24):
            size = base['training']['frame_policy'][str(frames)]
            for step in range(2):
                if shared_deadline_due(config, device, world):
                    raise RuntimeError('capacity deadline reached')
                batches = [[FrameSampleRequest(selected[(rank+i) % len(selected)], frames,
                    10000 + step*100 + k*10 + i) for i in range(size)] for k in range(2)]
                value = update(model, wrapped, optimizer, dataset, batches, tokenizer, codec, base,
                    run, 1000, device, world, check_groups=True)
                value.pop('microbatches')
                value.update(regime=regime, frames=frames, batch_per_rank=size, optimizer_step=step+1)
                records.append(value)
                write_json(output/f'progress_rank_{rank}.json', {'complete': False, 'cases': records})
        model.eval()
        requests = [FrameSampleRequest(selected[rank], 24, 80000+rank)]
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            surface, _, geometry = prepare(dataset, requests, base, tokenizer, codec, device)
            features, _ = model.backbone.encode(surface)
            reference = reference_indices(requests, config, device)
            xyz = reference_slots(geometry['frame_target_joints'], reference)
            prediction = model.head(features, surface['query_points'], surface['query_normals'], xyz,
                geometry['target_parents'], geometry['joint_mask'], reference)
            if not bool(torch.isfinite(prediction).all()) or not torch.equal(reference_slots(prediction, reference), xyz.float()):
                raise RuntimeError('24pose free inference or exact reference failed')
            free.append({'regime': regime, 'frames': 24, 'joints': int(geometry['joint_mask'].sum()), 'finite': True})
        del model, wrapped, optimizer, surface, geometry, features, prediction, xyz
        release()
    a, b, c = (initializations[r] for r in REGIMES)
    if (a['active_encoder_initial_sha256'] != b['active_encoder_initial_sha256']
            or any(a['active_encoder_initial_sha256'][k] == c['active_encoder_initial_sha256'][k]
                   for k in ('surface_tokenizer', 'motion_encoder'))
            or len({v['head_initial_sha256'] for v in initializations.values()}) != 1
            or c['backbone_optimizer_retained'] or c['constructor_official_weights_loaded']):
        raise RuntimeError('initialization factorization failed')
    parent = config['training']['parent_continuation']
    model, optimizer, _, _, _, initial, stream, samples, frames, step = load_model(
        regime_config(config, 'joint_pretrained'), device, rank, world, parent)
    if (samples, frames, step, stream.cursor*world) != (60000, 787504, 1260, 60000):
        raise RuntimeError('completed parent counters changed')
    parent_check = {'model_optimizer_RNG_loaded': initial['parent_full_model_optimizer_RNG_loaded'],
        'sampler_prefix_verified': initial['parent_sampler_prefix_verified'], 'samples': samples,
        'input_frames': frames, 'step': step, 'head_sha256': head_digest(model.head)}
    del model, optimizer
    release()
    ranks = gather({'rank': rank, 'initializations': initializations, 'cases': records,
        'free_inference': free, 'parent_resume': parent_check}, world)
    if len({r['parent_resume']['head_sha256'] for r in ranks}) != 1:
        raise RuntimeError('parent head weights differ between ranks')
    return {'complete': True, 'passed': True, 'source': source, 'scientific_sha256': scientific_sha(config),
        'ranks': ranks, 'profile_weights_discarded': True, 'not_model_quality': True}


def train(config, output, source, device, rank, world):
    phase = config['execution']['phase']
    if phase == 'controls_to60k':
        regimes, ends = REGIMES[1:], range(10000, 60001, 10000)
        resumes = {r: None for r in regimes}
    elif phase == 'common80k':
        regimes, ends = REGIMES, (70000, 80000)
        resumes = {'joint_pretrained': config['training']['parent_continuation'],
            **config['execution']['control_resumes']}
        if set(resumes) != set(REGIMES):
            raise ValueError('common80k requires both completed control checkpoints')
    else:
        raise ValueError('unknown declared execution phase')
    results = {r: [] for r in regimes}
    for end in ends:
        for regime in regimes:
            run = regime_config(config, regime)
            result = train_segment(run, output/regime, source, device, rank, world, end, resumes[regime])
            results[regime].append(result)
            if not result['complete']:
                return {'complete': False, 'operation_complete': False, 'phase': phase,
                    'stopped_for_deadline': True, 'results': results}
            if result['samples'] != end or not result['checkpoints']:
                raise RuntimeError('completed segment has invalid counters/checkpoints')
            resumes[regime] = result['checkpoints'][-1]
            if rank == 0:
                write_json(output/'regime_progress.json', {'phase': phase, 'results': results, 'resumes': resumes})
        heads = {results[r][0]['initialization']['head_initial_sha256'] for r in regimes}
        if len(heads) != 1:
            raise RuntimeError('regimes have different initial head parameters')
    finals = {}
    if phase == 'common80k':
        for regime in regimes:
            run = regime_config(config, regime)
            model, optimizer, _, _, base, *_ = load_model(run, device, rank, world, resumes[regime])
            _, entries = read_all(run)
            with torch.no_grad():
                finals[regime] = evaluate(model, base, run, entries, output/regime, 80000,
                    device, rank, world, final=True)
            del model, optimizer
            release()
            dist.barrier(device_ids=[device.index])
    return {'complete': True, 'phase_complete': True, 'operation_complete': phase == 'common80k',
        'phase': phase, 'source': source, 'scientific_sha256': scientific_sha(config),
        'results': results, 'resumes': resumes, 'final_evaluations': finals, 'quality_accepted': False,
        'interpretation': 'Joint objective vs pretrained stage2; warm-start package vs scratch stage2. GT references only. One matched recipe/seed, not a tuned best standalone model.'}
