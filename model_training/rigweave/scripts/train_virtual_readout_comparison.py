#!/usr/bin/env python3
"""Matched full-data continuation: unchanged localizer versus coordinate readout."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch

from motion_experiment_runtime import load_config, source_metadata, write_json
from train_virtual_frame import dataset_for, evaluate, load_entries, optimizer_for, prepare
from virtual_frame_runtime import (FrameBudgetStream, build_virtual_model, file_sha256,
                                   normalize_asset_gradients, set_exposure_lr)
from rigweave.dynamic_rig.virtual_frame_rig import AnchorCoordinateReadout, VirtualFrameRigConfig


def paired_requests(streams, remaining):
    if len(streams) != 2 or remaining < 2 or remaining % 2:
        raise ValueError("continuation must replay two original data ranks with an even asset boundary")
    batches = [stream.take(remaining // 2) for stream in streams]
    if len(batches[0]) != len(batches[1]) or batches[0][0].frames != batches[1][0].frames:
        raise ValueError("original data ranks no longer share their frame/batch schedule")
    return batches


def enable_coordinate_readout(model, optimizer, *, seed, attention_dim):
    if model.positions.coordinate_readout is not None:
        raise ValueError("coordinate readout already attached")
    device = next(model.parameters()).device
    with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        head = AnchorCoordinateReadout(model.config.dim, attention_dim).to(device)
    model.positions.coordinate_readout = head
    model.config = replace(model.config, position_coordinate_readout=True, position_coordinate_dim=attention_dim)
    model.positions.config = model.config
    optimizer.add_param_group({"params": list(head.parameters())})


def save_trial(model, optimizer, streams, config, plan, root, samples, frames, step, source):
    target = root / f"checkpoint_sample_{samples:06d}.pt"
    if target.exists():
        raise FileExistsError(target)
    temporary = target.with_suffix('.pt.tmp')
    torch.save({"schema": "virtual_readout_comparison_v1", "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "streams": [stream.state_dict() for stream in streams],
                "samples": samples, "input_frames": frames, "step": step, "config": config,
                "architecture": model.configuration(), "plan": plan, "source": source,
                "formal_initialization_allowed": False}, temporary)
    temporary.replace(target)
    write_json(root / 'latest_checkpoint.json', {"path": str(target), "samples": samples,
                                                "step": step, "bytes": target.stat().st_size})


def run(plan, arm_index):
    config = load_config(Path(plan['virtual_config']))
    repo = Path(config['runtime']['repo'])
    active = json.loads((repo / 'model_training/state/current.json').read_text())['active_operation']
    if active['label'] != plan['label'] or active['output'] != plan['output']:
        raise ValueError('training plan differs from the recorded active operation')
    subprocess.run([config['runtime']['python'], '-B', str(repo / 'model_training/tools/agent_work_guard.py'),
                    'check', '--operation', 'train'], check=True, stdout=subprocess.DEVNULL)
    source = source_metadata(config)
    arm = plan['arms'][arm_index]
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.cuda.set_per_process_memory_fraction(config['runtime']['allocator_fraction'], device)
    root = Path(plan['output']) / arm
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=True)
    if file_sha256(plan['checkpoint']) != plan['checkpoint_sha256']:
        raise ValueError('source new-model checkpoint differs from the agreed comparison')
    model, initialization = build_virtual_model(config, 'self_cross_2', device)
    optimizer = optimizer_for(model, config)
    payload = torch.load(plan['checkpoint'], map_location='cpu', mmap=True, weights_only=False)
    if (payload['schema'] != 'virtual_frame_rig_v1' or payload['samples'] != 5040 or payload['step'] != 105
            or VirtualFrameRigConfig(**payload['architecture']) != model.config
            or payload['config']['optimization'] != config['optimization']
            or payload['config']['data'] != config['data'] or len(payload['rank_states']) != 2
            or payload['initialization']['old_method_trained_weights_loaded']):
        raise ValueError('source checkpoint architecture/data/schedule/lineage is not the matched new-model run')
    model.load_state_dict(payload['model'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    entries = load_entries(config)
    dataset = dataset_for(entries['train'], config)
    streams = [FrameBudgetStream(
        dataset, rank=rank, world_size=2, seed=config['data']['seed'],
        frame_budget=plan['frame_budget'], batch_cap=plan['batch_cap'],
        min_frames=config['data']['frames_min'], max_frames=config['data']['frames_max'],
        state=payload['rank_states'][rank]['stream']) for rank in (0, 1)]
    samples, total_frames, step = payload['samples'], payload['input_frames'], payload['step']
    torch.set_rng_state(payload['rank_states'][0]['torch_rng'])
    torch.cuda.set_rng_state(payload['rank_states'][0]['cuda_rng'], device)
    del payload
    if arm == 'coordinate_readout':
        enable_coordinate_readout(model, optimizer, seed=plan['readout_seed'], attention_dim=plan['readout_dim'])
    elif arm != 'control':
        raise ValueError('unknown training arm')
    report = {"complete": False, "arm": arm, "source": source, "plan": plan,
              "initialization": initialization, "architecture": model.configuration(),
              "source_samples": samples, "source_step": step,
              "started_at": datetime.now(timezone.utc).isoformat(),
              "samples": samples, "input_frames": total_frames, "step": step, "evaluations": [],
              "global_asset_batch_replayed_on_one_gpu": True,
              "existing_optimizer_state_preserved": True, "formal_initialization_allowed": False}
    path = root / 'progress.json'
    write_json(path, report)
    model.train()
    milestones = sorted(set(plan['evaluation_samples'] + [plan['target_samples']]))
    if any(value <= samples or value % 2 or value > plan['target_samples'] for value in milestones):
        raise ValueError('invalid continuation milestones')
    if plan['target_samples'] > config['optimization']['schedule_samples']:
        raise ValueError('continuation exceeds the original schedule')
    started = time.monotonic()
    with (root / 'trace.jsonl').open('x', buffering=1) as trace:
        while samples < plan['target_samples']:
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(device)
            lr, beta1 = set_exposure_lr(optimizer, samples, config)
            boundary = min(value for value in milestones if value > samples)
            batches = paired_requests(streams, boundary - samples)
            schedule = [[request._asdict() for request in requests] for requests in batches]
            totals = torch.zeros(5, dtype=torch.float64, device=device)
            for requests in batches:
                surface, targets = prepare(dataset, requests, config, device)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    output = model(surface, targets, xyz_weight=config['optimization']['xyz_weight'])
                if not bool(torch.isfinite(output['loss'])):
                    raise RuntimeError(f'nonfinite training loss in {arm} at step {step + 1}')
                output['loss_per_asset'].sum().backward()
                size, frames = len(requests), requests[0].frames
                totals += torch.stack((output['loss'].detach() * size, output['structure_loss'].detach() * size,
                                       output['xyz_loss'].detach() * size, totals.new_tensor(size),
                                       totals.new_tensor(size * frames)))
                del output, surface, targets
            update_assets, update_frames = int(totals[3]), int(totals[4])
            normalize_asset_gradients(model.parameters(), global_assets=update_assets, world_size=1)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config['optimization']['clip_grad_norm'], error_if_nonfinite=True)
            optimizer.step()
            samples += update_assets
            total_frames += update_frames
            step += 1
            row = {"arm": arm, "step": step, "samples": samples, "input_frames": total_frames,
                   "assets_in_update": update_assets, "frames_in_update": update_frames,
                   "request_sha256": hashlib.sha256(json.dumps(schedule, sort_keys=True).encode()).hexdigest(),
                   "loss": float(totals[0] / update_assets), "structure_ce": float(totals[1] / update_assets),
                   "xyz_l1": float(totals[2] / update_assets), "gradient_norm": float(gradient_norm),
                   "lr": lr, "beta1": beta1, "update_seconds": time.monotonic() - tick,
                   "elapsed_seconds": time.monotonic() - started,
                   "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30}
            if arm == 'coordinate_readout':
                row['coordinate_key_norm'] = float(model.positions.coordinate_readout.key.weight.detach().norm())
            trace.write(json.dumps(row) + '\n')
            report.update(samples=samples, input_frames=total_frames, step=step, latest=row)
            write_json(path, report)
            print('READOUT_TRAIN ' + json.dumps(row), flush=True)
            if samples in milestones:
                save_trial(model, optimizer, streams, config, plan, root, samples, total_frames, step, source)
                evaluate(model, entries, config, root, samples, device, rank=0, world=1)
                report['evaluations'].append(samples)
                summary = json.loads((root / f'evaluation_sample_{samples:06d}' / 'summary.json').read_text())
                print('READOUT_EVAL ' + json.dumps({'arm': arm, 'samples': samples, 'summary': summary['summary']}), flush=True)
                write_json(path, report)
    report.update(complete=True, finished_at=datetime.now(timezone.utc).isoformat())
    write_json(path, report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    plan = load_config(args.config)
    if int(os.environ.get('WORLD_SIZE', 1)) != 2:
        raise ValueError('this experiment needs two independent one-GPU training arms')
    run(plan, int(os.environ['RANK']))


if __name__ == '__main__':
    main()
