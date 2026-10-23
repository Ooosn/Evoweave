#!/usr/bin/env python3
"""Guarded fresh single-target training; workload imports follow launch checks."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[3]
ENTRYPOINT = 'model_training/rigweave/scripts/train_legacy_single_target.py'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require_job_membership(text, job):
    membership = {}
    for line in text.splitlines():
        _, controllers, path = line.split(':', 2)
        for controller in controllers.split(','):
            membership[controller] = path
    expected = f'/AGE/{job}.1'
    if any(membership.get(controller) != expected for controller in ('devices', 'cpuset')):
        raise RuntimeError('launcher is outside the granted scheduler job; GPU IDs alone are insufficient')


def verify_launch(args, config):
    runtime = config['runtime']
    job = runtime['allocation']
    require_job_membership(Path('/proc/self/cgroup').read_text(), job)
    if (os.environ.get('JOB_ID') != str(job) or socket.gethostname().split('.')[0] != 'pcg01'
            or os.environ.get('CUDA_VISIBLE_DEVICES') != '5,6'
            or runtime['physical_gpus'] != [5, 6] or runtime['world_size'] != 2):
        raise RuntimeError('runtime differs from the existing two-H100 allocation')
    slots = int(os.environ.get('NSLOTS', '0'))
    affinity = os.sched_getaffinity(0)
    if slots != runtime['cpu_slots'] or slots < 1 or len(affinity) != slots:
        raise RuntimeError('apply and verify the granted CPU binding before workload imports')
    import resource
    if resource.getrlimit(resource.RLIMIT_AS)[0] != resource.RLIM_INFINITY:
        raise RuntimeError('CUDA requires unlimited virtual address space')
    grant = subprocess.check_output(['qstat', '-j', str(job)], text=True, timeout=30)
    lines = [line for line in grant.splitlines() if line.startswith('granted_devices')]
    if (len(lines) != 1 or runtime['host'] + ':' not in lines[0]
            or sorted(map(int, re.findall(r'/dev/nvidia(\d+)', lines[0]))) != [5, 6]
            or not re.search(r'job_state\s+\d+:\s+r\b', grant)):
        raise RuntimeError('the scheduler no longer grants these two devices')
    if (ROOT.resolve() != Path(runtime['repo']).resolve()
            or args.config.resolve().parent != (ROOT / 'model_training/experiments').resolve()
            or args.output.resolve().parent != Path(runtime['output_root']).resolve()):
        raise RuntimeError('use only the isolated published source/config/output directories')
    state = json.loads((ROOT / 'model_training/state/current.json').read_text())
    op = state['active_operation']
    if (state.get('architecture_contract', {}).get('id') != 'legacy_real_query_single_target_v1'
            or op.get('gpu_work_allowed') is not True or op.get('complete') is not False
            or op.get('required_entrypoint') != ENTRYPOINT
            or bool(op.get('formal_training_allowed')) != (args.mode == 'train')):
        raise RuntimeError('this exact GPU stage is not published')
    matches = [run for run in op.get('runs', []) if run.get('mode') == args.mode
               and Path(run['config']).resolve() == args.config.resolve()]
    if len(matches) != 1:
        raise RuntimeError('stage/config is not uniquely approved')
    run = matches[0]
    if (run.get('approved') is not True or run['config_sha256'] != file_sha256(args.config)
            or Path(run['output']).resolve() != args.output.resolve()
            or run['allocation'] != job or run['physical_gpus'] != [5, 6]):
        raise RuntimeError('published config/output/resource binding differs')
    operation = 'train' if args.mode == 'train' else 'preflight'
    if int(os.environ.get('RANK', '0')) == 0:
        subprocess.run([sys.executable, str(ROOT / 'model_training/tools/agent_work_guard.py'),
                        'check', '--operation', operation, '--entrypoint', ENTRYPOINT], check=True)
    if subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('GPU execution requires clean committed source')
    source = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', op['code_commit'], source], check=True)
    subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet', op['code_commit'], source, '--',
                    'model_training/rigweave', 'model_training/jobs', 'model_training/tools'], check=True)
    return source


def gradients(model):
    import torch
    report = {}
    for name in ('surface_tokenizer', 'motion_encoder', 'xyz_decoder'):
        values = [p.grad.detach().float() for p in getattr(model, name).parameters() if p.grad is not None]
        if not values or any(not bool(torch.isfinite(g).all()) for g in values):
            raise RuntimeError(f'missing/nonfinite {name} gradients')
        norm = sum(float(g.square().sum()) for g in values) ** .5
        if norm <= 0:
            raise RuntimeError(f'zero {name} learning signal')
        report[name] = norm
    for module in (model.xyz_decoder.get_input_embeddings(), model.xyz_decoder.get_output_embeddings()):
        g = module.weight.grad
        if g is None or not bool(torch.isfinite(g).all()) or float(g.float().norm()) <= 0:
            raise RuntimeError('the official XYZ embedding/head must remain trainable')
    return report


def optimizer_for(model, config):
    import torch
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=2e-5, betas=(.95, .999), weight_decay=config['optimization']['weight_decay'])


def wrapped_model(model, device):
    from torch.nn.parallel import DistributedDataParallel
    return DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=False,
                                   gradient_as_bucket_view=True)


def save_checkpoint(model, optimizer, stream, config, args, initial, source, counts, rank, world, device):
    import random
    import numpy as np
    import torch
    import torch.distributed as dist
    from legacy_single_target_runtime import FAMILY, write_json
    local = {'stream': stream.state_dict(), 'torch_rng': torch.get_rng_state(),
             'cuda_rng': torch.cuda.get_rng_state(device), 'numpy_rng': np.random.get_state(),
             'python_rng': random.getstate()}
    states = [None] * world
    dist.all_gather_object(states, local)
    if rank == 0:
        path = args.output / f"checkpoint_asset_{counts['asset_exposures']:06d}.pt"
        temporary = path.with_suffix('.pt.tmp')
        torch.save({'schema': FAMILY, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'optimizer_parameter_names': [name for name, p in model.named_parameters() if p.requires_grad],
                    'config': config, 'architecture': model.configuration(), 'initialization': initial,
                    'rank_states': states, 'source': source, **counts}, temporary)
        temporary.replace(path)
        write_json(args.output / 'latest_checkpoint.json', {'path': str(path), 'bytes': path.stat().st_size, **counts})
    dist.barrier()


def run_training(model, tokenizer, codec, entries, config, args, initial, source, rank, world, device, reference):
    import torch
    import torch.distributed as dist
    import train_virtual_query_xyz as core
    from train_virtual_fusion_normal import record_samples
    from virtual_frame_runtime import normalize_asset_gradients
    import legacy_single_target_runtime as rt
    stream = rt.make_stream(config, rank, world)
    report_plan = stream.plan_report()
    report_plan['input_frame_exposures'] = report_plan.pop('supervised_frame_exposures')
    report_plan['supervised_frame_exposures'] = 100000
    rt.write_json(args.output / f'exposure_plan_rank_{rank}.json', report_plan)
    dataset = rt.fixed_dataset(entries['train'], config, 2, mode='train')
    optimizer, wrapped = optimizer_for(model, config), wrapped_model(model, device)
    counts = {'asset_exposures': 0, 'input_frame_exposures': 0, 'supervised_frame_exposures': 0, 'step': 0}
    report = {'complete': False, 'rank': rank, 'source': source, 'initialization': initial, **counts}
    rt.write_json(args.output / f'rank_{rank}.json', report)
    if config['evaluation']['initial']:
        rt.run_evaluation(model, tokenizer, codec, entries, config, args, device, 0, rank, world, reference)
    started = time.monotonic()
    model.train()
    with (args.output / f'trace_rank_{rank}.jsonl').open('x', buffering=1) as trace:
        while counts['asset_exposures'] < 100000:
            local_goal = core.update_asset_target(counts['asset_exposures'], 100000, 48,
                                                  config['training']['checkpoint_assets'], world)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(device)
            lr, beta1 = rt.set_learning_rate(optimizer, counts['asset_exposures'], config)
            totals = torch.zeros(5, dtype=torch.float64, device=device)
            seen, schedule, tick = 0, [], time.monotonic()
            while seen < local_goal:
                requests = stream.take(local_goal - seen)
                size, frames = len(requests), requests[0].frames
                surface, target, geometry = core.prepare(dataset, requests, config, tokenizer, codec, device)
                indices = rt.train_target_indices(surface)
                context = wrapped.no_sync() if seen + size < local_goal else nullcontext()
                with context:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        out = wrapped(surface, target, target_indices=indices)
                    if not bool(torch.isfinite(out['loss'])):
                        raise RuntimeError('nonfinite single-target training loss')
                    out['loss_per_asset'].sum().backward()
                totals += torch.stack((out['loss'].detach() * size, out['main_accuracy'].detach() * size,
                                       totals.new_tensor(size), totals.new_tensor(size * frames),
                                       totals.new_tensor(size)))
                record = record_samples(geometry, requests)
                record.update(assets_per_rank=size, input_frames_per_asset=frames,
                              target_frame_indices=indices.cpu().tolist(),
                              target_raw_frames=geometry['selected_frames'][:, 0].cpu().tolist())
                schedule.append(record)
                seen += size
                del surface, target, geometry, out
            dist.all_reduce(totals)
            assets, inputs, supervised = map(int, totals[2:].tolist())
            if assets != local_goal * world or supervised != assets:
                raise RuntimeError('actual training counts violate one-target-per-asset supervision')
            normalize_asset_gradients(model.parameters(), global_assets=assets, world_size=world)
            if counts['step'] == 0:
                report['first_gradient_groups'] = gradients(model)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            counts['asset_exposures'] += assets
            counts['input_frame_exposures'] += inputs
            counts['supervised_frame_exposures'] += supervised
            counts['step'] += 1
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
            if peak > .85 * torch.cuda.get_device_properties(device).total_memory:
                raise RuntimeError('training exceeded its measured85percent memory limit')
            item = {**counts, 'assets_in_update': assets, 'input_frames_in_update': inputs,
                    'supervised_frames_in_update': supervised, 'microbatches': schedule,
                    'loss': float(totals[0] / assets), 'main_accuracy': float(totals[1] / assets),
                    'gradient_norm': float(norm), 'lr': lr, 'beta1': beta1,
                    'update_seconds': time.monotonic() - tick, 'elapsed_seconds': time.monotonic() - started,
                    'peak_allocated_bytes': peak, 'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}
            trace.write(json.dumps(item) + '\n')
            report.update(**counts, latest=item)
            rt.write_json(args.output / f'rank_{rank}.json', report)
            print('TRAIN ' + json.dumps({'rank': rank, **item}), flush=True)
            if counts['asset_exposures'] in config['training']['checkpoint_assets']:
                save_checkpoint(model, optimizer, stream, config, args, initial, source, counts, rank, world, device)
            if counts['asset_exposures'] in config['training']['evaluation_assets']:
                rt.run_evaluation(model, tokenizer, codec, entries, config, args, device,
                                  counts['asset_exposures'], rank, world, reference)
    if (counts['asset_exposures'] != 100000 or counts['supervised_frame_exposures'] != 100000
            or counts['step'] != len(stream.update_sizes)
            or counts['input_frame_exposures'] != report_plan['input_frame_exposures']):
        raise RuntimeError('terminal training counts disagree with the planned exposure stream')
    report.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat(),
                  final_model_sha256=rt.parameter_fingerprint(model))
    rt.write_json(args.output / f'rank_{rank}.json', report)
    return report


def run_preflight(model, tokenizer, codec, entries, config, args, initial, source, rank, world, device):
    import torch
    import torch.distributed as dist
    import train_virtual_query_xyz as core
    import legacy_single_target_runtime as rt
    from rigweave.dynamic_rig.virtual_query_xyz import original_xyz_vocabulary_processor
    optimizer = optimizer_for(model, config)
    wrapped = wrapped_model(model, device)
    report = {'complete': False, 'passed': False, 'cases': [], 'rank': rank, 'source': source,
              'contract_sha256': rt.contract_sha256(config), 'initialization': initial,
              'diagnostic_weights_discarded': True}
    for frames in config['preflight']['frames']:
        dataset = rt.fixed_dataset(entries['train'][rank:rank+1], config, frames, mode='seen_eval')
        surface, target, geometry = core.prepare(dataset, [0], config, tokenizer, codec, device)
        case = {'frames': frames, 'asset_id': geometry['asset_id'][0],
                'selected_raw_frames': geometry['selected_frames'][0].cpu().tolist(), 'updates': []}
        for step in range(config['preflight']['updates_per_case']):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = wrapped(surface, target, target_indices=rt.train_target_indices(surface))
            if not bool(torch.isfinite(out['loss'])):
                raise RuntimeError('nonfinite preflight loss')
            out['loss'].backward()
            group_norms = gradients(model)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            case['updates'].append({'step': step+1, 'loss': float(out['loss'].detach()),
                'gradient_norm': float(norm), 'gradient_groups': group_norms,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(device)})
        model.eval()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            tokens = model.encode_surface(surface)
            condition = model.condition_from_tokens(surface, tokens, rt.train_target_indices(surface))
            prefix = target['main_input_ids'][:, 0, :2]
            logits = model.main_logits(condition, prefix, torch.ones_like(prefix, dtype=torch.bool))[:, -1]
            processor = original_xyz_vocabulary_processor(tokenizer, prefix[0])
            expected = int(processor(prefix[:, :0], logits).argmax(-1)[0])
            generated = model.generate_condition(condition, prefix[0], eos_token=tokenizer.eos,
                                                  max_new_tokens=1, logits_processor=processor)
        if generated['tokens'][2] != expected:
            raise RuntimeError('teacher and cached free-generation first-token predictions disagree')
        case['teacher_generation_first_token_equal'] = True
        report['cases'].append(case)
        rt.write_json(args.output / f'preflight_rank_{rank}.json', report)
        del surface, target, geometry, tokens, condition, out
    passed = torch.tensor(1, device=device)
    dist.all_reduce(passed, op=dist.ReduceOp.MIN)
    report.update(complete=True, passed=bool(passed.item()))
    rt.write_json(args.output / f'preflight_rank_{rank}.json', report)
    return report


def run_capacity(model, tokenizer, codec, entries, config, args, initial, source, rank, world, device):
    import torch
    import torch.distributed as dist
    import train_virtual_query_xyz as core
    import legacy_single_target_runtime as rt
    from rigweave.dynamic_rig.frame_batch_sampler import FrameSampleRequest
    groups, lengths = core.load_capacity_selection(config, entries['train'])
    dataset = rt.fixed_dataset(entries['train'], config, 2, mode='train')
    optimizer, wrapped = optimizer_for(model, config), wrapped_model(model, device)
    total_memory = torch.cuda.get_device_properties(device).total_memory
    report = {'complete': False, 'passed': False, 'cases': [], 'rank': rank, 'source': source,
              'contract_sha256': rt.contract_sha256(config), 'initialization': initial,
              'gpu_total_bytes': total_memory, 'diagnostic_weights_discarded': True}
    model.train()
    for number, case in enumerate(config['capacity']['cases']):
        indices, batch, frames = groups[case['stress_group']], case['batch_per_rank'], case['frames']
        requests = [FrameSampleRequest(indices[i % len(indices)], frames, 77000 + number*100 + i)
                    for i in range(batch)]
        measurements = []
        for iteration in range(2):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(device)
            tick = time.monotonic()
            for micro in range(2):
                surface, target, geometry = core.prepare(dataset, requests, config, tokenizer, codec, device)
                for i, (asset, path) in enumerate(zip(geometry['asset_id'], geometry['path'])):
                    if int(target['main_attention_mask'][i, 0].sum()) != lengths[asset, path]:
                        raise RuntimeError('capacity target differs from the certified longest-token cases')
                with (wrapped.no_sync() if micro == 0 else nullcontext()):
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        out = wrapped(surface, target, target_indices=rt.train_target_indices(surface))
                    if not bool(torch.isfinite(out['loss'])):
                        raise RuntimeError('nonfinite capacity loss')
                    (out['loss']/2).backward()
                del surface, target, geometry, out
            group_norms = gradients(model)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
            accepted = torch.tensor(int(peak <= .85 * total_memory), device=device)
            dist.all_reduce(accepted, op=dist.ReduceOp.MIN)
            measurements.append({'iteration': iteration+1, 'peak_allocated_bytes': peak,
                                 'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
                                 'seconds': time.monotonic()-tick, 'gradient_norm': float(norm),
                                 'gradient_groups': group_norms, 'accumulation_microbatches': 2})
            if not bool(accepted.item()):
                raise RuntimeError('dual-H100 capacity exceeded85percent')
        aliases = [name for name, ids in groups.items() if ids == indices]
        report['cases'].append({**case, 'passed': True, 'measurements': measurements,
                                'covered_stress_groups': aliases})
        rt.write_json(args.output / f'capacity_rank_{rank}.json', report)
        print('CAPACITY '+json.dumps({'rank': rank, **report['cases'][-1]}), flush=True)
    report.update(complete=True, passed=True)
    rt.write_json(args.output / f'capacity_rank_{rank}.json', report)
    return report


def validate_release(config, rt):
    if config['formal_release'].get('approved') is not True:
        raise RuntimeError('formal training is not released')
    for mode in ('preflight', 'capacity'):
        spec = config[mode]['receipt']
        if not spec or rt.sha256(spec['path']) != spec['sha256']:
            raise RuntimeError(f'{mode} receipt is missing or changed')
        result = json.loads(Path(spec['path']).read_text())
        if (result.get('mode') != mode or result.get('complete') is not True or result.get('passed') is not True
                or result.get('world_size') != 2 or result.get('contract_sha256') != rt.contract_sha256(config)):
            raise RuntimeError(f'{mode} did not validate this exact training contract')
        reports = result.get('ranks')
        if (not isinstance(reports, list) or len(reports) != 2
                or any(type(r.get('rank')) is not int for r in reports)
                or {r['rank'] for r in reports} != {0, 1}):
            raise RuntimeError(f'{mode} requires exactly two distinct rank receipts')
        for report in reports:
            if (report.get('complete') is not True or report.get('passed') is not True
                    or report.get('source') != result.get('code_commit')
                    or report.get('contract_sha256') != result['contract_sha256']
                    or report.get('diagnostic_weights_discarded') is not True):
                raise RuntimeError(f'{mode} contains incomplete or mismatched rank evidence')
            cases = report.get('cases', [])
            if mode == 'preflight':
                if (len(cases) != len(config['preflight']['frames'])
                        or sorted(c.get('frames', -1) for c in cases) != sorted(config['preflight']['frames'])
                        or any(c.get('teacher_generation_first_token_equal') is not True
                               or len(c.get('updates', [])) != config['preflight']['updates_per_case'] for c in cases)):
                    raise RuntimeError('preflight must prove both backward updates and generation parity')
            else:
                identity = lambda c: (c['frames'], c['batch_per_rank'], c['stress_group'])
                if (len(cases) != len(config['capacity']['cases'])
                        or {identity(c) for c in cases} != {identity(c) for c in config['capacity']['cases']}):
                    raise RuntimeError('capacity receipt omitted a declared stress case')
                total = report.get('gpu_total_bytes', 0)
                if type(total) is not int or total <= 0:
                    raise RuntimeError('capacity must report the measured GPU memory size')
                for case in cases:
                    measurements = case.get('measurements', [])
                    if case.get('passed') is not True or len(measurements) != 2:
                        raise RuntimeError('capacity needs two successful optimizer updates per case')
                    for measurement in measurements:
                        peak = measurement.get('peak_allocated_bytes', float('nan'))
                        if (measurement.get('accumulation_microbatches') != 2
                                or not isinstance(peak, (int, float)) or not math.isfinite(peak)
                                or not 0 < peak <= .85 * total):
                            raise RuntimeError('capacity measurement lacks accumulation or memory headroom')
        subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet', result['code_commit'], 'HEAD', '--',
                        'model_training/rigweave', 'model_training/jobs', 'model_training/tools'], check=True)
        if mode == 'capacity':
            for frames in range(2, 25):
                batch = config['training']['frame_policy'][str(frames)]
                for group in ('longest_main', f'geometry_t{frames}'):
                    for report in reports:
                        if not any(case['frames'] >= frames and case['batch_per_rank'] == batch
                                   and group in case['covered_stress_groups'] and case['passed']
                                   for case in report['cases']):
                            raise RuntimeError(f'unmeasured capacity at T={frames}, batch={batch}, group={group}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('preflight', 'capacity', 'train'), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    source = verify_launch(args, config)
    # Scheduler provenance and CPU binding have been checked before torch imports.
    sys.path.insert(0, str(ROOT / 'model_training/rigweave/src'))
    import torch
    import torch.distributed as dist
    import legacy_single_target_runtime as rt
    rt.validate_config(config)
    rank, world, local = (int(os.environ[key]) for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    if world != 2 or local != rank or not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError('exactly two local H100 ranks are required')
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    if 'H100' not in torch.cuda.get_device_name(device):
        raise RuntimeError('the authorized hardware is H100')
    torch.cuda.set_per_process_memory_fraction(.85, device)
    if args.mode == 'train':
        validate_release(config, rt)
    dist.init_process_group('nccl', timeout=timedelta(minutes=45))
    try:
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        entries = rt.read_entries(config)
        reference = rt.verify_reference(config, entries)
        model, tokenizer, codec, initial = rt.build_model(config, device)
        initial.update(source=source, config_sha256=rt.sha256(args.config),
                       contract_sha256=rt.contract_sha256(config))
        rt.write_json(args.output / f'initialization_rank_{rank}.json', initial)
        functions = {'preflight': run_preflight, 'capacity': run_capacity, 'train': run_training}
        arguments = (model, tokenizer, codec, entries, config, args, initial, source, rank, world, device)
        result = functions[args.mode](*arguments, reference) if args.mode == 'train' else functions[args.mode](*arguments)
        results = [None] * world
        dist.all_gather_object(results, result)
        if rank == 0:
            if args.mode == 'train' and len({r['final_model_sha256'] for r in results}) != 1:
                raise RuntimeError('terminal rank model tensors disagree')
            rt.write_json(args.output / 'result.json', {'complete': True,
                'passed': all(r.get('passed', r.get('complete')) is True for r in results),
                'world_size': world, 'mode': args.mode, 'ranks': results,
                'contract_sha256': rt.contract_sha256(config), 'code_commit': source,
                'config_sha256': rt.sha256(args.config), 'quality_accepted': False,
                'finished_utc': datetime.now(timezone.utc).isoformat()})
    except BaseException:
        if args.output.exists():
            rt.write_json(args.output / f'failure_rank_{rank}.json',
                          {'complete': False, 'source': source, 'traceback': traceback.format_exc()})
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
