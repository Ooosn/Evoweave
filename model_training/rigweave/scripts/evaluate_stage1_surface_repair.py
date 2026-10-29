#!/usr/bin/env python3
"""Native-guarded zero-update canonical-input repair."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import traceback

from stage1_surface_repair_protocol import validate_config
from virtual_pair_guard import ROOT, verify_launch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('audit','capacity'),required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config,args.mode)
    source = verify_launch(args,config)
    sys.path.insert(0,str(ROOT/'model_training/rigweave/src'))
    import torch
    import torch.distributed as dist
    from stage1_surface_repair_runtime import audit, run, write_json
    torch.set_num_threads(1)
    if args.mode == 'audit':
        args.output.mkdir(parents=True,exist_ok=False)
        audit(config,args.output,source)
        return
    rank,world = int(os.environ['RANK']),int(os.environ['WORLD_SIZE'])
    if world != 2 or torch.cuda.device_count() != 2:
        raise RuntimeError('repair requires exact native two-GPU grant')
    device = torch.device('cuda',int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(config['runtime']['gpu_memory_fraction_limit'],device)
    try:
        dist.init_process_group('nccl',timeout=timedelta(minutes=30))
        if rank == 0:
            args.output.mkdir(parents=True,exist_ok=False)
        dist.barrier(device_ids=[device.index])
        result = run(config,args.output,source,device,rank,world)
        result['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(args.output/f'result_rank_{rank}.json',result)
        if rank == 0:
            write_json(args.output/'result.json',result)
    except Exception as error:
        if args.output.exists():
            write_json(args.output/f'failure_rank_{rank}.json',{'complete':False,'passed':False,
                'error':repr(error),'traceback':traceback.format_exc()})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
