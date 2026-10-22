#!/usr/bin/env python3
"""Read-only own80k checkpoint/stream handoff to the cumulative160k schedule."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'model_training/rigweave/src'))
sys.path.insert(0, str(ROOT / 'model_training/rigweave/scripts'))
from structure_xyz_fill_runtime import (
    FAMILY, make_stream, read_entries, restore_sampling_state, sha256,
    validate_config, verify_data_selection, write_json,
)
from train_structure_xyz_fill import training_contract


def plan_handoff(base, future, entries, states):
    reports, plans = [], []
    for rank, local in enumerate(states):
        kwargs = dict(asset_exposures=80000, supervised_frame_exposures=1040352,
                      step=1683, rank=rank, world=2)
        old = restore_sampling_state(base, entries, local['stream'], **kwargs)
        extended = restore_sampling_state(future, entries, local['stream'], **kwargs)
        fresh = make_stream(future, entries, rank=rank, world=2)
        assert old.state_dict() == local['stream'] and extended.cursor == 40000
        for name, count in (('order', 80000), ('frame_counts', 1683), ('update_sizes', 1683)):
            assert np.array_equal(getattr(old, name), getattr(extended, name)[:count])
            assert np.array_equal(getattr(extended, name), getattr(fresh, name))
        assert extended.plan_sha256 == fresh.plan_sha256
        assert np.all(np.bincount(extended.order, minlength=5000) == 32)
        plan = extended.plan_report()
        plans.append(plan)
        frames = np.cumsum(extended.update_sizes.astype(np.int64) * extended.frame_counts.astype(np.int64))
        milestones = []
        for assets in range(90000, 160001, 10000):
            positions = np.flatnonzero(extended.global_ends == assets)
            assert len(positions) == 1
            index = int(positions[0])
            milestones.append({'asset_exposures': assets, 'step': index + 1,
                               'supervised_frame_exposures': int(frames[index])})
        first_updates = []
        while extended.cursor * 2 < 80240:
            update = int(np.searchsorted(extended.local_ends, extended.cursor, side='right'))
            if not first_updates or first_updates[-1]['step'] != update + 1:
                first_updates.append({'step': update + 1, 'asset_exposures': int(extended.global_ends[update]),
                                      'supervised_frame_exposures': int(frames[update]), 'microbatches': []})
            requests = extended.take()
            first_updates[-1]['microbatches'].append({
                'frames': requests[0].frames,
                'requests': [{'item_index': request.item_index, 'asset_id': entries[request.item_index]['asset_id'],
                              'frames': request.frames, 'sampling_seed': str(request.sampling_seed)}
                             for request in requests]})
        assert len(first_updates) == 5 and first_updates[-1]['asset_exposures'] == 80240
        reports.append({'rank': rank, 'saved_cursor': local['stream']['cursor'],
                        'old80000_prefix_exact': True, 'extended_plan_matches_fresh': True,
                        'milestones': milestones, 'first_five_updates': first_updates})
    comparable = deepcopy(plans)
    for rank, plan in enumerate(comparable):
        assert plan['contract'].pop('rank') == rank
    assert comparable[0] == comparable[1]
    assert reports[0]['milestones'] == reports[1]['milestones']
    plan = plans[0]
    return {'plan_sha256': plan['plan_sha256'], 'total_updates': len(plan['update_sizes']),
            'new_updates': len(plan['update_sizes']) - 1683,
            'supervised_frames': plan['supervised_frame_exposures'],
            'new_supervised_frames': plan['supervised_frame_exposures'] - 1040352,
            'all5000_assets_exactly32_visits': True, 'ranks': reports,
            'milestone_counters': reports[0]['milestones']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--terminal-receipt', type=Path, required=True)
    parser.add_argument('--terminal-receipt-sha256', required=True)
    parser.add_argument('--base-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU continuation audit requires explicitly hidden CUDA')
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert sha256(args.terminal_receipt) == args.terminal_receipt_sha256
    terminal = json.loads(args.terminal_receipt.read_text())
    assert terminal['passed'] is True
    stat = args.checkpoint.stat()
    assert stat.st_size == 5480784524 and sha256(args.checkpoint) == args.checkpoint_sha256
    base = json.loads(args.base_config.read_text())
    validate_config(base)
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    assert saved['schema'] == FAMILY and saved['mode'] == 'train' and saved['config'] == base
    assert saved['training_contract'] == training_contract(base)
    assert (saved['asset_exposures'], saved['supervised_frame_exposures'], saved['step']) == (80000, 1040352, 1683)
    assert len(saved['rank_states']) == 2
    future = deepcopy(base)
    boundaries = base['training']['evaluation_assets'] + list(range(90000, 160001, 10000))
    future['training'].update(target_assets=160000, evaluation_assets=boundaries, checkpoint_assets=boundaries)
    validate_config(future)
    entries = read_entries(base)
    verify_data_selection(base, entries)
    report = plan_handoff(base, future, entries['train'], saved['rank_states'])
    named = saved['optimizer_parameter_names']
    groups = saved['optimizer']['param_groups']
    assert len(named) == len(groups) == 4
    count = 0
    for named_group, group in zip(named, groups):
        assert named_group['group'] == group['name']
        names = named_group['parameters']
        assert len(names) == len(group['params'])
        for name, identifier in zip(names, group['params']):
            state = saved['optimizer']['state'][identifier]
            assert float(state['step']) == 1683
            assert state['exp_avg'].shape == saved['model'][name].shape == state['exp_avg_sq'].shape
            count += 1
    assert count == 626 and not torch.cuda.is_initialized()
    assert args.checkpoint.stat().st_mtime_ns == stat.st_mtime_ns and args.checkpoint.stat().st_size == stat.st_size
    report.update(passed=True, complete=True, candidate_plan_only=True,
                  source_checkpoint={'path': str(args.checkpoint), 'sha256': args.checkpoint_sha256,
                                     'bytes': stat.st_size, 'counts': [80000, 1040352, 1683]},
                  terminal_integrity_receipt={'path': str(args.terminal_receipt), 'sha256': args.terminal_receipt_sha256},
                  source_config={'path': str(args.base_config), 'sha256': sha256(args.base_config)},
                  boundaries=boundaries, initialized_adam_states=count,
                  optimizer_groups=[{key: group[key] for key in ('name', 'lr', 'peak_lr')} for group in groups],
                  optimizer_updates_executed=0, CUDA_initialized=False, checkpoint_modified=False,
                  model_constructed=False, finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(args.output, report)
    print(json.dumps({'passed': True, 'path': str(args.output), 'sha256': sha256(args.output),
                      'total_updates': report['total_updates'], 'supervised_frames': report['supervised_frames'],
                      'plan_sha256': report['plan_sha256']}), flush=True)


if __name__ == '__main__':
    main()
