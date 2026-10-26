"""One data/optimization contract for both fixed-5k virtual-path arms."""
from contextlib import ExitStack, nullcontext
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
import gc
import hashlib
import json
from pathlib import Path, PurePosixPath
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
from virtual_pair_loading import OrderedAssetPrefetch
from single_query_protocol import (SINGLE_SCOPE, SINGLE_VARIANT, SELECTOR, TWO_SCOPE,
    TWO_VARIANT, TWO_SELECTOR, DENSITY_SCOPE, THREE_VARIANT, FRACTION_VARIANT,
    THREE_SELECTOR, FRACTION_SELECTOR, FOUR_VARIANT, FRACTION2_VARIANT,
    FOUR_SELECTOR, FRACTION2_SELECTOR, select_density_target_frames,
    select_target_frames, select_two_target_frames, supervised_count)
from virtual_query_xyz_runtime import build_model as build_base, fixed_dataset, read_entries, sha256


VARIANTS = ('virtual_on', 'virtual_off')
TRAIN_SHA = 'a17b6be96fb33a32a1bee7117b07789da6261c6b5693104f3d2cce07c09bab88'
VALID_SHA = 'fd2be7b5885833946e68dc186300a820757ae652c487749eec1b049d381e230a'


def variants(config):
    if config.get('family') == DENSITY_SCOPE:
        algorithm = config['pair'].get('target_selection', {}).get('algorithm')
        return {THREE_SELECTOR: (THREE_VARIANT,), FRACTION_SELECTOR: (FRACTION_VARIANT,),
                FOUR_SELECTOR: (FOUR_VARIANT,), FRACTION2_SELECTOR: (FRACTION2_VARIANT,)}.get(algorithm, ())
    return {SINGLE_SCOPE: (SINGLE_VARIANT,), TWO_SCOPE: (TWO_VARIANT,)}.get(config.get('family'), VARIANTS)


def selected_target_count(config):
    return {SINGLE_SCOPE: 1, TWO_SCOPE: 2}.get(config.get('family'))


def selected_supervision(config):
    return config.get('family') in (SINGLE_SCOPE, TWO_SCOPE, DENSITY_SCOPE)


def target_selection(config, requests):
    if config.get('family') == DENSITY_SCOPE:
        spec = config['pair']['target_selection']
        return select_density_target_frames(requests, spec['seed'], spec['algorithm'])
    selector = select_two_target_frames if config.get('family') == TWO_SCOPE else select_target_frames
    return selector(requests, config['pair']['target_selection']['seed'])


def validate_config(config):
    data, plan, optim = config['data'], config['training'], config['optimization']
    single = selected_supervision(config)
    if (config.get('family') not in (SCOPE, SINGLE_SCOPE, TWO_SCOPE, DENSITY_SCOPE)
            or not variants(config) or config['pair']['variants'] != list(variants(config))):
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
            or config['pair']['all_real_frames_supervised'] is not (not single)
            or config['pair']['common_post_attention_ffn'] is not True
            or config['architecture']['fusion_mode'] != 'real_queries_virtual'):
        raise ValueError('this pair changes only virtual interaction/readout, not codec/losses')
    if single:
        selector = TWO_SELECTOR if config.get('family') == TWO_SCOPE else SELECTOR
        if config.get('family') == DENSITY_SCOPE:
            selector = {THREE_VARIANT: THREE_SELECTOR, FRACTION_VARIANT: FRACTION_SELECTOR,
                        FOUR_VARIANT: FOUR_SELECTOR, FRACTION2_VARIANT: FRACTION2_SELECTOR}[variants(config)[0]]
        if (config['pair'].get('target_selection') != {'algorithm': selector, 'seed': 20261026}
                or config['runtime'].get('execution_mode') != 'scheduler_job'
                or operation_target(config) != 20000
                or config['training'].get('resume_root')):
            raise ValueError('selected-target control must start fresh and stop at20k under the scheduler')
        reference = config['pair'].get('reference', {})
        if set(reference) != {'config', 'CPU_audit', 'evaluation', 'checkpoint_sha256'}:
            raise ValueError('single-target control needs pinned all-frame references')
        for name in ('config', 'CPU_audit', 'evaluation'):
            if (set(reference[name]) != {'path', 'sha256'}
                    or not PurePosixPath(reference[name]['path']).is_absolute()
                    or len(reference[name]['sha256']) != 64):
                raise ValueError('invalid all-frame reference receipt')
    elif 'target_selection' in config['pair'] or 'reference' in config['pair']:
        raise ValueError('target selection belongs only to the single-target experiment')
    if plan['frame_policy'] != {str(t): 24 if t <= 6 else 12 if t <= 12 else 6 for t in range(2, 25)}:
        raise ValueError('training batches exceed the declared capacity buckets')
    if (config['evaluation']['valid_assets'], config['evaluation']['input_frames'],
            config['evaluation']['frame_indices']) != (128, 8, list(range(8))):
        raise ValueError('shared128-asset validation panel changed')
    if not config['artifacts']['preserve_all_checkpoints']:
        raise ValueError('checkpoint preservation is mandatory')
    target = operation_target(config)
    extra = config.get('execution', {}).get('evaluation_assets', [])
    if (target not in plan['checkpoint_assets'] or target > plan['target_assets']
            or any(type(x) is not int or x not in plan['checkpoint_assets'] or x > target for x in extra)
            or target not in evaluation_milestones(config)):
        raise ValueError('operation must end at a retained and evaluated checkpoint')
    workers = config['runtime'].get('data_workers_per_rank', 0)
    if (type(workers) is not int or workers < 0
            or (workers and (workers + 1) * config['runtime']['world_size'] + 1
                > config['runtime']['execution_cpu_budget'])):
        raise ValueError('data workers exceed the total CPU budget')
    if config['runtime'].get('prefetch_microbatches', 2) not in (1, 2):
        raise ValueError('prefetch must remain bounded to one or two microbatches')
    make_stream(config, 0, 2)


def operation_target(config):
    return config.get('execution', {}).get('stop_assets_per_variant', config['training']['target_assets'])


def evaluation_milestones(config):
    return sorted(set(config['training']['evaluation_assets'])
        | set(config.get('execution', {}).get('evaluation_assets', [])))


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
        'id_xyz_stages.py', 'train_virtual_query_xyz.py', 'train_virtual_fusion_normal.py',
        'virtual_pair_loading.py', 'single_query_protocol.py')]
    paths += ['model_training/rigweave/src/rigweave/dynamic_rig/' + name for name in (
        'virtual_path_ablation.py', 'virtual_query_xyz.py', 'motion_encoder.py', 'motion_evidence.py',
        'surface_tokenizer.py', 'virtual_frame_data.py', 'virtual_query_xyz_data.py', 'root_rt.py',
        'aux_skeleton_tokens.py', 'sampling.py', 'skeleton_topology_metrics.py',
        'data.py', 'shared_structure_medium_data.py')]
    return {path: sha256(ROOT / path) for path in paths}


def make_stream(config, rank, world, state=None):
    return MatchedFrameStream(range(5000), rank=rank, world_size=world,
        seed=config['data']['stream_seed'], target_assets=config['training']['target_assets'],
        effective_assets=48, frame_policy=config['training']['frame_policy'], manifest_sha256=TRAIN_SHA,
        state=state, update_boundaries=config['training']['checkpoint_assets'])


def next_update_batches(stream, samples, world):
    index = int(np.searchsorted(stream.global_ends, samples, side='right'))
    goal = int(stream.update_sizes[index])
    batches, count = [], 0
    while count < goal // world:
        requests = stream.take(goal // world - count)
        batches.append(requests)
        count += len(requests)
    return goal, batches


def planned_microbatches(config, stream, limit):
    planner = make_stream(config, stream.rank, stream.world_size, stream.state_dict())
    samples = planner.cursor * planner.world_size
    while samples < limit:
        goal, batches = next_update_batches(planner, samples, planner.world_size)
        yield from batches
        samples += goal


@lru_cache(maxsize=4)
def density_exposure_plan(encoded_config):
    """Exact prefix accounting for per-asset K, including checkpoint restarts."""
    config = json.loads(encoded_config)
    streams = [make_stream(config, rank, 2) for rank in range(2)]
    samples, frames, supervised = 0, 0, 0
    result = {0: (0, 0)}
    while samples < config['training']['target_assets']:
        for stream in streams:
            goal, batches = next_update_batches(stream, samples, 2)
            for requests in batches:
                frames += sum(request.frames for request in requests)
                supervised += sum(map(len, target_selection(config, requests)))
        samples += goal
        result[samples] = (frames, supervised)
    return result


def supervised_exposures(config, samples, frames):
    if config.get('family') != DENSITY_SCOPE:
        return supervised_count(config, samples, frames)
    expected_frames, targets = density_exposure_plan(json.dumps(config, sort_keys=True))[samples]
    if frames != expected_frames:
        raise RuntimeError('variable target accounting differs from the input plan')
    return targets


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
    if variant not in variants(config):
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


def compatible_source(incoming, purpose):
    current = source_hashes()
    if incoming == current:
        return True
    state = json.loads((ROOT / 'model_training/state/current.json').read_text())
    transition = state['active_operation'].get('source_transition', {})
    return (transition.get('accepted') is True and purpose in transition.get('allowed_uses', [])
        and incoming == transition.get('previous_source_sha256')
        and current == transition.get('current_source_sha256'))


def reviewed(config, name):
    state = json.loads((ROOT / 'model_training/state/current.json').read_text())
    spec = state['active_operation'][name]
    if spec.get('accepted') is not True or sha256(spec['path']) != spec['sha256']:
        raise RuntimeError(f'{name} is not the exact primary-accepted receipt')
    report = json.loads(Path(spec['path']).read_text())
    if (not report.get('passed') or not report.get('complete')
            or report['contract_sha256'] != contract_sha(config)
            or not compatible_source(report['source_sha256'], name)):
        raise RuntimeError(f'{name} computation or data contract differs')
    return report


def training_evidence(config):
    if config.get('family') != DENSITY_SCOPE:
        return reviewed(config, 'CPU_audit'), reviewed(config, 'capacity')
    # Reuse unchanged initialization/data and the larger all-frame envelope.
    audited, _ = reference_control(config)
    spec = config['execution']['inherited_capacity']
    measured = pinned_json(spec)
    previous = pinned_json(config['pair']['reference']['config'])
    if (not measured.get('complete') or not measured.get('passed')
            or not measured.get('profile_weights_discarded')
            or measured['contract_sha256'] != contract_sha(previous)
            or measured['frame_policy'] != config['training']['frame_policy']
            or measured['cases'] != config['capacity']['cases']):
        raise RuntimeError('inherited all-frame capacity contract differs')
    current = source_hashes()
    changed = {'model_training/rigweave/scripts/virtual_pair_runtime.py',
               'model_training/rigweave/scripts/train_virtual_pair.py',
               'model_training/rigweave/src/rigweave/dynamic_rig/virtual_path_ablation.py'}
    for path, digest in measured['source_sha256'].items():
        if path not in changed and current.get(path) != digest:
            raise RuntimeError(f'inherited capacity common source differs: {path}')
    expected = deepcopy(audited['initialization']['virtual_off'])
    expected['variant'] = variants(config)[0]
    return {'initialization': {variants(config)[0]: expected},
            'inherited': True, 'CPU_audit': config['pair']['reference']['CPU_audit'],
            'capacity': spec}, measured


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


def pinned_json(spec):
    if sha256(spec['path']) != spec['sha256']:
        raise RuntimeError('pinned reference digest differs')
    return json.loads(Path(spec['path']).read_text())


def reference_control(config, initial=None):
    reference = config['pair']['reference']
    previous = pinned_json(reference['config'])
    audited = pinned_json(reference['CPU_audit'])
    evaluated = pinned_json(reference['evaluation'])
    if previous.get('family') != SCOPE:
        raise RuntimeError('reference is not the completed virtual-path pair')
    for key in ('initialization', 'architecture', 'auxiliary_codec', 'surface_sampling', 'data',
                'normal_match', 'optimization', 'evaluation', 'capacity'):
        if config[key] != previous[key]:
            raise RuntimeError(f'single-target control changed common field: {key}')
    old_training, new_training = deepcopy(previous['training']), deepcopy(config['training'])
    old_training.pop('resume_root', None)
    new_training.pop('resume_root', None)
    if old_training != new_training:
        raise RuntimeError('single-target control changed the sampling or LR horizon')
    excluded = {'variants', 'all_real_frames_supervised', 'target_selection', 'reference'}
    if ({k: v for k, v in previous['pair'].items() if k not in excluded}
            != {k: v for k, v in config['pair'].items() if k not in excluded}):
        raise RuntimeError('single-target control changed another pair setting')
    if (not audited.get('complete') or not audited.get('passed') or audited.get('cuda_initialized')
            or audited['contract_sha256'] != contract_sha(previous)
            or not audited.get('parallel_loader', {}).get('passed')):
        raise RuntimeError('reference CPU/loader audit is incomplete')
    current = source_hashes()
    permitted_changes = {'model_training/rigweave/scripts/virtual_pair_runtime.py',
                         'model_training/rigweave/src/rigweave/dynamic_rig/virtual_path_ablation.py'}
    for path, digest in audited['source_sha256'].items():
        if path not in permitted_changes and current.get(path) != digest:
            raise RuntimeError(f'unchanged reference computation differs: {path}')
    for key in ('data_workers_per_rank', 'prefetch_microbatches', 'world_size', 'execution_cpu_budget', 'rss_limit_bytes'):
        if config['runtime'][key] != previous['runtime'][key]:
            raise RuntimeError('accepted loader/RAM envelope changed')
    expected = deepcopy(audited['initialization']['virtual_off'])
    expected['variant'] = variants(config)[0]
    if initial is not None and initial != expected:
        raise RuntimeError('single-target initialization differs from the all-frame control')
    if (evaluated.get('complete') is not True or evaluated.get('variant') != 'virtual_off'
            or evaluated.get('samples') != operation_target(config)
            or evaluated.get('contract_sha256') != audited['contract_sha256']
            or evaluated.get('source_sha256') != audited['source_sha256']
            or evaluated.get('checkpoint_sha256') != reference['checkpoint_sha256']
            or evaluated.get('selection_sha256') != config['evaluation']['selection']['sha256']
            or evaluated.get('original_GT_scoring') is not True
            or evaluated.get('failed_views_scored_zero') is not True
            or evaluated.get('oracle_not_used') is not True):
        raise RuntimeError('reference evaluation/checkpoint binding differs')
    return audited, evaluated


def audit_target_selection(config):
    records, histograms = [], {}
    before = tree_fingerprint({'torch': torch.get_rng_state(), 'numpy': np.random.get_state(),
                              'python': random.getstate()})
    for rank in range(2):
        stream, samples, count, inputs, targets, digest = make_stream(config, rank, 2), 0, 0, 0, 0, hashlib.sha256()
        while samples < operation_target(config):
            goal, batches = next_update_batches(stream, samples, 2)
            for requests in batches:
                selected = target_selection(config, requests)
                for request, frame in zip(requests, selected):
                    digest.update(json.dumps([request.item_index, request.frames, request.sampling_seed, frame]).encode())
                    histogram = histograms.setdefault(str(request.frames), [0] * request.frames)
                    for slot in frame if isinstance(frame, list) else [frame]:
                        histogram[slot] += 1
                        targets += 1
                    count += 1
                    inputs += request.frames
            samples += goal
        records.append({'rank': rank, 'asset_exposures': count,
                        'supervised_frames': targets,
                        'input_frames': inputs, 'selected_plan_sha256': digest.hexdigest(),
                        'input_plan_sha256': stream.plan_sha256})
    after = tree_fingerprint({'torch': torch.get_rng_state(), 'numpy': np.random.get_state(),
                             'python': random.getstate()})
    if before != after or sum(row['asset_exposures'] for row in records) != operation_target(config):
        raise RuntimeError('target selection changed RNG or exposure accounting')
    return {'passed': True, 'global_RNG_unchanged': True, 'ranks': records,
            'selection_histograms_diagnostic_only': histograms}


def sampling_plan_report(config, stream):
    report = stream.plan_report()
    if selected_supervision(config):
        # The shared sampler's legacy field counts every input frame.
        report['input_frame_exposures'] = report['supervised_frame_exposures']
        report['supervised_frame_exposures'] = supervised_exposures(
            config, config['training']['target_assets'], report['input_frame_exposures'])
        report['decoder_target_policy'] = config['pair']['target_selection']['algorithm']
    return report


def audit(config, entries, output, source):
    verify_data_selection(config, entries)
    groups, lengths = load_capacity_selection(config, entries['train'])
    selected = evaluation_rows(config, entries)
    initial = {}
    for variant in variants(config):
        model, _, _, report = build_model(config, variant, 'cpu')
        initial[variant] = report
        del model
        gc.collect()
    if len({report['wrapped_initial_sha256'] for report in initial.values()}) != 1:
        raise RuntimeError('pair does not start with identical common state')
    plans = [sampling_plan_report(config, make_stream(config, r, 2)) for r in range(2)]
    if plans[0]['plan_sha256'] != plans[1]['plan_sha256']:
        raise RuntimeError('distributed sampling plans disagree')
    single = selected_supervision(config)
    if single:
        previous, _ = reference_control(config, initial[variants(config)[0]])
        loader = previous['parallel_loader']
    else:
        loader = audit_loader(config, entries, groups) if config['runtime'].get('data_workers_per_rank') else None
    return {'complete': True, 'passed': True, 'mode': 'audit', 'training_assets': 5000,
        'validation_pool': 967, 'validation_panel': len(selected), 'initialization': initial,
        'capacity_groups': groups, 'capacity_length_records': len(lengths), 'sampling_plan': plans[0],
        'source': source, 'source_sha256': source_hashes(), 'contract_sha256': contract_sha(config),
        'cuda_initialized': torch.cuda.is_initialized(), 'optimizer_created': False,
        'parallel_loader': loader, 'parallel_loader_inherited_from_pinned_reference': single,
        'target_selection': audit_target_selection(config) if single else None,
        'all_frame_reference_initialization_exact': single}


def audit_loader(config, entries, groups):
    import resource
    from rigweave.dynamic_rig.frame_batch_sampler import FrameSampleRequest
    dataset = fixed_dataset(entries['train'], config, 2, mode='train')
    workers = config['runtime']['data_workers_per_rank']
    before = tree_fingerprint({'torch': torch.get_rng_state(), 'numpy': np.random.get_state(),
        'python': random.getstate()})
    results = []
    for number, case in enumerate(config['capacity']['cases']):
        indices = groups[case['stress_group']]
        requests = [FrameSampleRequest(indices[i % len(indices)], case['frames'], 830000 + number * 100 + i)
                    for i in range(min(6, case['batch_per_rank']))]
        started = time.monotonic()
        expected = [tree_fingerprint(dataset[request]) for request in requests]
        serial_seconds = time.monotonic() - started
        started = time.monotonic()
        with OrderedAssetPrefetch(dataset, [requests], workers=workers) as loader:
            loaded, timing = loader.take(requests)
            actual = [tree_fingerprint(loaded[request]) for request in requests]
        parallel_seconds = time.monotonic() - started
        if actual != expected:
            raise RuntimeError('parallel CPU data changed geometry, normalization, frames or GT')
        del loaded
        with ExitStack() as stack:
            pipelines, first_batches, held = [], [], []
            for rank in range(2):
                microbatches = [[FrameSampleRequest(indices[i % len(indices)], case['frames'],
                    840000 + number * 1000 + rank * 100 + micro * 30 + i)
                    for i in range(case['batch_per_rank'])] for micro in range(3)]
                first_batches.append(microbatches[0])
                pipelines.append(stack.enter_context(OrderedAssetPrefetch(dataset, microbatches,
                    workers=workers, depth=config['runtime'].get('prefetch_microbatches', 2))))
            for pipeline, requests in zip(pipelines, first_batches):
                held.append(pipeline.take(requests)[0])
            for pipeline in pipelines:
                for _, futures in pipeline.pending:
                    for future in futures:
                        future.result()
            peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            if peak_rss > config['runtime']['rss_limit_bytes'] - 16 * 1024**3:
                raise RuntimeError('two-rank CPU prefetch leaves insufficient RAM for the measured model/runtime')
        del held, pipelines, pipeline, futures, future
        results.append({**case, 'checked_assets': len(expected), 'workers': workers,
            'all_CPU_fields_bitwise_equal': True, 'serial_seconds': serial_seconds,
            'parallel_seconds': parallel_seconds, 'comparison_order': 'serial_then_parallel_cache_bias_possible',
            'input_sha256': actual, 'two_rank_full_microbatch_queue_exercised': True,
            'resident_microbatches_per_rank': 1 + config['runtime'].get('prefetch_microbatches', 2),
            'cumulative_process_peak_rss_bytes': peak_rss, 'reserved_model_runtime_headroom_bytes': 16 * 1024**3,
            **timing})
    after = tree_fingerprint({'torch': torch.get_rng_state(), 'numpy': np.random.get_state(),
        'python': random.getstate()})
    if before != after or torch.cuda.is_initialized():
        raise RuntimeError('CPU loading changed global RNG or initialized CUDA')
    return {'passed': True, 'cases': results, 'global_RNG_unchanged': True,
            'cuda_initialized': False, 'training_or_optimizer_updates': 0}


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
           samples, device, world, *, check_groups=False, prefetch=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    lr, beta = set_lr(optimizer, samples, config)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    totals = torch.zeros(5, dtype=torch.float64, device=device)
    schedule = []
    loading = {'cpu_data_wait_seconds': 0., 'cpu_data_task_seconds': 0.}
    for i, requests in enumerate(batches):
        prepared_dataset = dataset
        if prefetch is not None:
            prepared_dataset, timing = prefetch.take(requests)
            for key in loading:
                loading[key] += timing[key]
        surface, target, geometry = prepare(prepared_dataset, requests, config, tokenizer, codec, device)
        del prepared_dataset
        selected, arguments = None, {}
        if selected_supervision(config):
            selected = target_selection(config, requests)
            if config.get('family') == DENSITY_SCOPE:
                counts = [len(slots) for slots in selected]
                width = max(counts)
                padded = [slots + [0] * (width - len(slots)) for slots in selected]
                arguments['target_frame_indices'] = torch.tensor(padded, dtype=torch.int64, device=device)
                arguments['target_frame_mask'] = (torch.arange(width, device=device)[None]
                                                  < torch.tensor(counts, device=device)[:, None])
            else:
                arguments['target_frame_indices'] = torch.tensor(selected, dtype=torch.int64, device=device)
        with wrapped.no_sync() if world > 1 and i < len(batches) - 1 else nullcontext():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                losses = wrapped(surface, target, auxiliary_weight=0., **arguments)
            expected_targets = ((arguments['target_frame_indices'].shape[1]
                                 if arguments['target_frame_indices'].ndim == 2 else 1)
                                if selected is not None else requests[0].frames)
            if losses['main_loss_per_frame'].shape != (len(requests), expected_targets):
                raise RuntimeError('decoder did not supervise the declared target count')
            if not bool(torch.isfinite(losses['loss'])):
                raise RuntimeError('nonfinite pair objective')
            losses['loss_per_asset'].sum().backward()
        size, frames = len(requests), requests[0].frames
        target_count = sum(len(slots) if isinstance(slots, list) else 1 for slots in selected) if selected is not None else size * frames
        totals += torch.stack((losses['main_loss'].detach() * size, losses['main_accuracy'].detach() * size,
            totals.new_tensor(size), totals.new_tensor(size * frames), totals.new_tensor(target_count)))
        recorded = {**record_samples(geometry, requests), 'frames': frames,
            'target_sha256': hashlib.sha256(target['main_input_ids'].cpu().numpy().tobytes()).hexdigest()}
        if selected is not None:
            rows = torch.arange(size, device=device)
            if arguments['target_frame_indices'].ndim == 2:
                rows = rows[:, None]
            selected_ids = target['main_input_ids'][rows, arguments['target_frame_indices']]
            if 'target_frame_mask' in arguments:
                selected_ids = selected_ids[arguments['target_frame_mask']]
            recorded.update(target_frame_indices=selected,
                selected_target_raw_frames=[([raw[slot] for slot in index] if isinstance(index, list) else raw[index])
                    for raw, index in zip(recorded['selected_raw_frames'], selected)],
                selected_target_sha256=hashlib.sha256(selected_ids.cpu().numpy().tobytes()).hexdigest(),
                decoder_targets_per_asset=([len(slots) for slots in selected]
                    if config.get('family') == DENSITY_SCOPE else selected_target_count(config)))
        schedule.append(recorded)
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
        'supervised_frames_in_update': int(totals[4]),
        'main_loss': float(totals[0] / assets), 'main_accuracy': float(totals[1] / assets),
        'learning_rate': lr, 'beta1': beta, 'gradient_norm': norm, 'gradient_groups': gradients,
        'data_workers_per_rank': config['runtime'].get('data_workers_per_rank', 0), **loading,
        'seconds': time.monotonic() - started, 'microbatches': schedule, **memory}


def capacity(config, entries, output, source, device, rank, world):
    from rigweave.dynamic_rig.frame_batch_sampler import FrameSampleRequest
    approved = reviewed(config, 'CPU_audit')
    dataset = fixed_dataset(entries['train'], config, 2, mode='train')
    results = []
    for variant in variants(config):
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
        root = Path(config['training']['resume_root'])
        resumed = checkpoint_candidates(root, variant)
        if root.resolve() == Path(config['artifacts']['durable_root']).resolve():
            for row in resumed:
                durable = Path(row['durable_path'])
                if durable.parent.resolve() != (root / variant).resolve():
                    raise RuntimeError('durable checkpoint escaped the explicit resume root')
                row['original_primary_path'] = row['path']
                row['path'] = str(durable)
                row['restoration_source'] = 'explicit_durable_resume_root'
        rows += resumed
    return max(rows, key=lambda row: row['asset_exposures']) if rows else None


def evaluation_summary_path(output, config, variant, samples):
    roots = [output]
    if config['training'].get('resume_root'):
        roots.append(Path(config['training']['resume_root']))
    for root in roots:
        path = root / variant / f'evaluation_{samples:06d}' / 'summary.json'
        if path.exists():
            result = json.loads(path.read_text())
            checkpoint = latest_checkpoint(output, config, variant)
            artifacts = result.get('row_artifacts', [])
            if (result.get('complete') is True and result.get('samples') == samples
                    and result.get('variant') == variant and result.get('contract_sha256') == contract_sha(config)
                    and result.get('source_sha256') == source_hashes()
                    and result.get('selection_sha256') == config['evaluation']['selection']['sha256']
                    and checkpoint and checkpoint['asset_exposures'] == samples
                    and result.get('checkpoint_sha256') == checkpoint['sha256']
                    and len(artifacts) == config['runtime']['world_size']
                    and all(Path(r['path']).is_file() and sha256(r['path']) == r['sha256'] for r in artifacts)):
                rows = [row for artifact in artifacts for row in json.loads(Path(artifact['path']).read_text())]
                selection = json.loads(Path(config['evaluation']['selection']['path']).read_text())['valid']
                expected = {(r['asset_id'], f) for r in selection for f in range(8)}
                if len(rows) == 1024 and {(r['asset_id'], r['frame_index']) for r in rows} == expected:
                    return path
    return None


def evaluation_complete(output, config, variant, samples):
    return evaluation_summary_path(output, config, variant, samples) is not None


def pending_variants(progress, output, config):
    target = operation_target(config)
    if any(value > target for value in progress.values()):
        raise RuntimeError('saved progress exceeds the authorized operation target')
    return [variant for variant in variants(config) if progress[variant] < target
        or not evaluation_complete(output, config, variant, target)]


def restore_checkpoint(model, optimizer, config, spec, variant, rank, world):
    if sha256(spec['path']) != spec['sha256']:
        raise RuntimeError('checkpoint file digest differs')
    payload = torch.load(spec['path'], map_location='cpu', mmap=True, weights_only=False)
    if (payload['variant'] != variant or payload['contract_sha256'] != contract_sha(config)
            or not compatible_source(payload['source_sha256'], 'checkpoint')
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
    samples, frames, step = (payload[key] for key in ('asset_exposures', 'input_frame_exposures', 'step'))
    if payload['supervised_frame_exposures'] != supervised_exposures(config, samples, frames):
        raise RuntimeError('checkpoint input/supervision accounting differs')
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
            'asset_exposures': samples, 'input_frame_exposures': frames,
            'supervised_frame_exposures': supervised_exposures(config, samples, frames),
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
            'input_frame_exposures': frames, 'supervised_frame_exposures': supervised_exposures(config, samples, frames),
            'model_sha256': model_fp, 'optimizer_sha256': optimizer_fp,
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
    checkpoint = latest_checkpoint(output, config, variant)
    if not checkpoint or checkpoint['asset_exposures'] != samples:
        raise RuntimeError('evaluation must bind to its retained checkpoint')
    attempt = max([int(p.name.split('_')[-1]) for p in root.glob('attempt_*')] + [0]) + 1 if rank == 0 else None
    attempt = gather(attempt, world)[0]
    directory = root / f'attempt_{attempt:03d}' / f'rank_{rank}'
    directory.mkdir(parents=True, exist_ok=False)
    original_targets = {}
    def items():
        for i in range(len(dataset)):
            if shared_deadline_due(config, device, world):
                raise EvaluationHandoff('saved training state retained; evaluation reached deadline')
            prepared = prepare(dataset, [i], config, tokenizer, codec, device)
            geometry = prepared[2]
            count = int(geometry['joint_count'][0])
            for frame in range(8):
                original_targets[(geometry['asset_id'][0], frame)] = {
                    'gt_joint_count': count,
                    'original_GT_sha256': tree_fingerprint({
                        'joints': geometry['frame_target_joints'][0, frame, :count],
                        'parents': geometry['target_parents'][0, :count]}),
                    'normalization_sha256': tree_fingerprint({key: geometry[key] for key in ('query_center', 'query_scale')})}
            yield prepared
    result = original_evaluate(model, items(), tokenizer,
        {**config, 'probe': {'auxiliary_weight': 0., 'evaluate_auxiliary': False, 'max_new_tokens': 2048}},
        directory, samples, frame_indices=config['evaluation']['frame_indices'])
    for row in result['rows']:
        row.update(original_targets[(row['asset_id'], row['frame_index'])])
        row['predicted_joint_count'] = len(row['predicted_joints']) if row['success'] else None
        row['absolute_joint_count_error'] = abs(row['predicted_joint_count'] - row['gt_joint_count']) if row['success'] else None
    row_path = directory / 'scored_rows.json'
    write_json(row_path, result['rows'])
    artifacts = gather({'rank': rank, 'path': str(row_path), 'sha256': sha256(row_path)}, world)
    all_rows = [row for values in gather(result['rows'], world) for row in values]
    expected = {(row['asset_id'], f) for row in selected for f in range(8)}
    if len(all_rows) != 1024 or {(r['asset_id'], r['frame_index']) for r in all_rows} != expected:
        raise RuntimeError('paired validation coverage differs')
    if rank == 0:
        summary = {}
        for name, rows in [('all', all_rows), ('non52', [r for r in all_rows if r['gt_joint_count'] != 52]),
                           ('gt52', [r for r in all_rows if r['gt_joint_count'] == 52])]:
            errors = [r['absolute_joint_count_error'] for r in rows if r['success']]
            summary[name] = {'views': len(rows), 'assets': len({r['asset_id'] for r in rows}),
                'original_edge_f1': float(np.mean([r['original_edge_f1'] for r in rows])) if rows else None,
                'failures': sum(not r['success'] for r in rows),
                'exact_count_fraction': float(np.mean([r['exact_count'] for r in rows])) if rows else None,
                'joint_count_MAE_successful_views': float(np.mean(errors)) if errors else None}
        write_json(root / 'summary.json', {'complete': True, 'variant': variant, 'samples': samples,
            'groups': summary, 'quality_accepted': False, 'failed_views_scored_zero': True,
            'original_GT_scoring': True, 'oracle_not_used': True, 'contract_sha256': contract_sha(config),
            'checkpoint_sha256': checkpoint['sha256'], 'source_sha256': source_hashes(),
            'selection_sha256': config['evaluation']['selection']['sha256'], 'row_artifacts': artifacts})
    synchronize(device, world)


def compare_evaluations(output, config, samples):
    rows_by_arm, checkpoints, sources, contracts = {}, {}, {}, {}
    single = selected_supervision(config)
    names = (variants(config)[0], 'all_frames') if single else VARIANTS
    def read_rows(variant, summary):
        artifacts = summary['row_artifacts']
        if (len(artifacts) != 2 or {r['rank'] for r in artifacts} != {0, 1}
                or any(sha256(row['path']) != row['sha256'] for row in artifacts)):
            raise RuntimeError('comparison row artifact binding differs')
        rows = [row for artifact in artifacts for row in json.loads(Path(artifact['path']).read_text())]
        table = {(r['asset_id'], r['frame_index']): r for r in rows}
        selection = json.loads(Path(config['evaluation']['selection']['path']).read_text())['valid']
        if len(rows) != 1024 or set(table) != {(r['asset_id'], f) for r in selection for f in range(8)}:
            raise RuntimeError('comparison coverage differs')
        rows_by_arm[variant] = table
        checkpoints[variant] = summary['checkpoint_sha256']
        sources[variant], contracts[variant] = summary['source_sha256'], summary['contract_sha256']
    for variant in variants(config):
        summary_path = evaluation_summary_path(output, config, variant, samples)
        if summary_path is None:
            raise RuntimeError('paired comparison requires both checkpoint-bound evaluations')
        summary = json.loads(summary_path.read_text())
        read_rows(variant, summary)
    if single:
        _, reference = reference_control(config)
        read_rows('all_frames', reference)
    on, off = (rows_by_arm[v] for v in names)
    difference_name = ('two_minus_all_original_edge_f1' if config.get('family') == TWO_SCOPE
        else 'single_minus_all_original_edge_f1' if single else 'Vplus_minus_Vminus_original_edge_f1')
    if config.get('family') == DENSITY_SCOPE:
        difference_name = f'{variants(config)[0]}_minus_all_original_edge_f1'
    for key, row in on.items():
        for field in ('raw_frame_index', 'gt_joint_count', 'original_GT_sha256', 'normalization_sha256', 'target_tokens'):
            if row[field] != off[key][field]:
                raise RuntimeError(f'paired validation input/target differs: {key} {field}')
    groups = {}
    for name, predicate in [('all', lambda r: True), ('non52', lambda r: r['gt_joint_count'] != 52),
                            ('gt52', lambda r: r['gt_joint_count'] == 52)]:
        keys = [key for key in on if predicate(on[key])]
        assets = sorted({key[0] for key in keys})
        differences = np.asarray([np.mean([on[k]['original_edge_f1'] - off[k]['original_edge_f1']
            for k in keys if k[0] == asset]) for asset in assets])
        rng = np.random.default_rng(20261025)
        interval = np.quantile(rng.choice(differences, size=(2000, len(differences)), replace=True).mean(1),
                               [.025, .975]).tolist() if len(differences) else None
        groups[name] = {'assets': len(assets), 'views': len(keys),
            difference_name: float(differences.mean()) if len(differences) else None,
            'paired_asset_bootstrap_95pct_interval': interval,
            'arms': {variant: {'original_edge_f1': float(np.mean([table[k]['original_edge_f1'] for k in keys])) if keys else None,
                              'failures': sum(not table[k]['success'] for k in keys),
                              'exact_count_fraction': float(np.mean([table[k]['exact_count'] for k in keys])) if keys else None}
                     for variant, table in rows_by_arm.items()}}
    result = {'complete': True, 'quality_accepted': False, 'samples_per_variant': samples,
        'validation_assets': 128, 'views_per_asset': 8, 'matched_raw_frames_GT_normalization_tokens': True,
        'checkpoint_sha256': checkpoints, 'source_sha256': source_hashes(), 'contract_sha256': contract_sha(config),
        'arm_source_sha256': sources, 'arm_contract_sha256': contracts,
        'comparison': ('two_target_vs_pinned_all_frame_control' if config.get('family') == TWO_SCOPE
            else f'{variants(config)[0]}_vs_pinned_all_frame_control' if config.get('family') == DENSITY_SCOPE
            else 'single_target_vs_pinned_all_frame_control' if single else 'virtual_on_vs_off'),
        'groups': groups, 'bootstrap': {'unit': 'asset', 'replicates': 2000, 'seed': 20261025,
            'limitation': 'validation-asset uncertainty for one training seed; not training-seed variability'}}
    write_json(output / f'paired_evaluation_{samples:06d}.json', result)
    return result


def train_pair(config, entries, output, source, device, rank, world):
    approved, measured = training_evidence(config)
    if measured['frame_policy'] != config['training']['frame_policy'] or measured['cases'] != config['capacity']['cases']:
        raise RuntimeError('training physical batches were not measured')
    dataset = fixed_dataset(entries['train'], config, 2, mode='train')
    checkpoints = {variant: latest_checkpoint(output, config, variant) for variant in variants(config)}
    progress = {variant: checkpoints[variant]['asset_exposures'] if checkpoints[variant] else 0 for variant in variants(config)}
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
            if (samples in evaluation_milestones(config)
                    and not evaluation_complete(output, config, variant, samples)):
                try:
                    with preserve_random():
                        evaluate(model, tokenizer, codec, config, entries, output, variant, samples, device, rank, world)
                except EvaluationHandoff:
                    evaluation_interrupted = True
            limit = min(((samples // 5000) + 1) * 5000, operation_target(config))
            first_step, last_saved = step, samples if checkpoints[variant] else None
            write_json(output / f'initialization_{variant}_rank_{rank}.json', initial)
            workers = config['runtime'].get('data_workers_per_rank', 0)
            loading = OrderedAssetPrefetch(dataset, planned_microbatches(config, stream, limit),
                workers=workers, depth=config['runtime'].get('prefetch_microbatches', 2)) if workers else nullcontext(None)
            with loading as prefetch:
                while samples < limit and not evaluation_interrupted:
                    if shared_deadline_due(config, device, world):
                        break
                    goal, batches = next_update_batches(stream, samples, world)
                    supervised_before = supervised_exposures(config, samples, frames)
                    value = update(model, wrapped, optimizer, dataset, batches, tokenizer, codec,
                        config, variant, samples, device, world, check_groups=step == first_step, prefetch=prefetch)
                    if value['assets_in_update'] != goal:
                        raise RuntimeError('actual update differs from the matched exposure plan')
                    samples += goal
                    frames += value['frames_in_update']
                    if (config.get('family') == DENSITY_SCOPE
                            and value['supervised_frames_in_update'] != supervised_exposures(config, samples, frames) - supervised_before):
                        raise RuntimeError('actual supervised targets differ from the deterministic plan')
                    step += 1
                    value.update(variant=variant, rank=rank, step=step, asset_exposures=samples,
                        input_frame_exposures=frames, supervised_frame_exposures=supervised_exposures(config, samples, frames),
                        sampling_plan_sha256=stream.plan_sha256)
                    trace.write(json.dumps(value, allow_nan=False) + '\n')
                    progress[variant] = samples
                    write_json(output / f'progress_rank_{rank}.json', {'complete': False, 'progress': progress,
                        'latest': {k: v for k, v in value.items() if k != 'microbatches'}})
                    print('TRAIN ' + json.dumps({k: v for k, v in value.items() if k != 'microbatches'}), flush=True)
                # Save consumed progress before waiting on any unused prefetch reads.
                if last_saved != samples:
                    checkpoints[variant] = save_checkpoint(model, optimizer, stream, config, output, variant,
                        initial, source, samples, frames, step, rank, world, device)
            if (samples in evaluation_milestones(config)
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
    if finished and rank == 0:
        compare_evaluations(output, config, operation_target(config))
    synchronize(device, world)
    return {'passed': True, 'complete': finished and operation_target(config) == config['training']['target_assets'],
        'operation_complete': finished, 'operation_target_per_variant': operation_target(config),
        'stage_complete': True, 'mode': 'train',
        'allocation_handoff_required': not finished, 'progress': progress, 'checkpoints': checkpoints,
        'target_per_variant': config['training']['target_assets'], 'training_assets': 5000,
        'quality_accepted': False, 'contract_sha256': contract_sha(config), 'source_sha256': source_hashes()}
