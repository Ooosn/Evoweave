#!/usr/bin/env python3
"""Guarded fixed-5000-asset virtual-on/off training, never a small-fit trial."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import traceback

from virtual_pair_guard import ROOT, verify_launch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('audit', 'capacity', 'train'), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    source = verify_launch(args, config)
    sys.path.insert(0, str(ROOT / 'model_training/rigweave/src'))
    import torch
    import torch.distributed as dist
    from virtual_pair_runtime import audit, capacity, train_pair, read_entries, validate_config, write_json
    torch.set_num_threads(1)
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    claimed = False
    try:
        validate_config(config)
        entries = read_entries(config)
        if args.mode == 'audit':
            if world != 1 or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
                raise RuntimeError('weight audit must be CPU-only')
            args.output.mkdir(parents=True, exist_ok=False)
            claimed = True
            result = audit(config, entries, args.output, source)
        else:
            if world != 2 or torch.cuda.device_count() != 2:
                raise RuntimeError('exactly two approved GPUs required')
            device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
            torch.cuda.set_device(device)
            torch.cuda.set_per_process_memory_fraction(.85, device)
            dist.init_process_group('nccl', timeout=timedelta(minutes=60))
            if rank == 0:
                args.output.mkdir(parents=True, exist_ok=False)
            dist.barrier(device_ids=[device.index])
            claimed = True
            result = (capacity if args.mode == 'capacity' else train_pair)(
                config, entries, args.output, source, device, rank, world)
        result.update(source=source, finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.output / f'result_rank_{rank}.json', result)
        if rank == 0:
            write_json(args.output / 'result.json', result)
        print('RESULT ' + json.dumps({key: value for key, value in result.items()
            if key in ('passed', 'complete', 'mode', 'progress', 'training_assets', 'allocation_handoff_required')}), flush=True)
    except Exception as error:
        if claimed:
            write_json(args.output / f'failure_rank_{rank}.json',
                {'complete': False, 'passed': False, 'error': repr(error), 'traceback': traceback.format_exc()})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
