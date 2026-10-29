#!/usr/bin/env python3
"""Guarded full18,002-asset joint training from official UniRig for300kvisits."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import traceback

from anchor_full_protocol import sha256, validate_config
from virtual_pair_guard import ROOT, verify_launch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mode', choices=('audit', 'capacity', 'train'), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    source = verify_launch(args, config)
    config['_config_sha256'] = sha256(args.config)
    sys.path.insert(0, str(ROOT/'model_training/rigweave/src'))
    import torch
    import torch.distributed as dist
    from anchor_full_runtime import audit, capacity, train, write_json
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.mode == 'audit':
        args.output.mkdir(parents=True, exist_ok=False)
        audit(config, args.output, source)
        return
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if world != 2 or torch.cuda.device_count() != 2:
        raise RuntimeError('exactly two approved GPUs are required')
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(config['runtime']['gpu_memory_fraction_limit'], device)
    claimed = False
    try:
        dist.init_process_group('nccl', timeout=timedelta(minutes=30))
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=False)
        dist.barrier(device_ids=[device.index])
        claimed = True
        result = (capacity if args.mode == 'capacity' else train)(config, args.output, source, device, rank, world)
        result['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(args.output/f'result_rank_{rank}.json', result)
        if rank == 0:
            write_json(args.output/'result.json', result)
            print('RESULT '+json.dumps(result), flush=True)
    except Exception as error:
        if claimed:
            write_json(args.output/f'failure_rank_{rank}.json', {'complete': False, 'passed': False,
                'error': repr(error), 'traceback': traceback.format_exc()})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
