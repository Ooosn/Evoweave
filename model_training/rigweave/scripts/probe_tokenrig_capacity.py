"""Bounded real-data Qwen capacity probe, using the original joint update."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import traceback

import numpy as np
import torch
import torch.distributed as dist

from anchor_full_data import CachedFiles, stress_groups
from anchor_full_runtime import capacity_requests, check_teacher_prefix, initialize
from anchor_joint_runtime import update
from reference_flow_runtime import read_all
from train_virtual_query_xyz import prepare
from virtual_pair_runtime import wrap
from virtual_query_xyz_runtime import fixed_dataset, write_json


ENLARGED_BATCHES = {'frames12': (12, 24), 'frames24': (24, 12)}


def probe(config, output, device, rank, world, profile='default'):
    base, entries = read_all(config)
    print(f'rank={rank} manifest verified', flush=True)
    resolver = CachedFiles(config, base)
    groups, _ = stress_groups(entries['train'])
    dataset = fixed_dataset(entries['train'], base, 2, mode='train', file_resolver=resolver)
    print(f'rank={rank} cache verified; loading released weights', flush=True)
    model, optimizer, tokenizer, codec, base, initial = initialize(config, device)
    wrapped = wrap(model, device, world)
    records = []
    status = {'profile': profile, 'initialization': initial['official']['tokenrig_decoder'],
              'cases': records, 'complete': False}
    write_json(output/f'progress_rank_{rank}.json', status)
    cases = sorted(base['capacity']['cases'], key=lambda row: row['frames'], reverse=True)
    if profile != 'default':
        frames, batch = ENLARGED_BATCHES[profile]
        cases = [{**case, 'batch_per_rank': batch} for case in cases if case['frames'] == frames]
    for case in cases:
        frames, batch = case['frames'], case['batch_per_rank']
        accum, remainder = divmod(config['training']['effective_assets_per_update'], world * batch)
        if not accum or remainder:
            raise ValueError('probe batch does not exactly divide the effective batch')
        indices = groups[case['stress_group']]
        for step in range(config['capacity']['optimizer_steps_per_case']):
            print(f'rank={rank} start T={frames} B={batch} A={accum} {case["stress_group"]} step={step+1}',
                  flush=True)
            batches = [capacity_requests(base, indices, rank, frames, batch,
                200000+frames*1000000+step*100000+k*30000) for k in range(accum)]
            result = update(model, wrapped, optimizer, dataset, batches, tokenizer, codec, base, config,
                            1000, device, world, check_groups=True)
            if result['assets_in_update'] != config['training']['effective_assets_per_update']:
                raise ValueError('probe did not reproduce the effective batch')
            gradient = model.backbone.base.decoder_condition_projection.weight.grad
            if gradient is None or not torch.isfinite(gradient).all() or not gradient.norm() > 0:
                raise ValueError('new condition projection did not receive a finite nonzero gradient')
            result.pop('microbatches')
            records.append({**case, **result, 'step': step+1, 'accumulation': accum})
            write_json(output/f'progress_rank_{rank}.json', status)
            print(json.dumps({'rank': rank, **records[-1]}), flush=True)
    if profile != 'default':
        status['complete'] = True
        write_json(output/f'result_rank_{rank}.json', status)
        return
    optimizer.zero_grad(set_to_none=True)
    optimizer.state.clear()
    del wrapped
    torch.cuda.empty_cache()
    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    req = capacity_requests(base, groups['largest_joints'], rank, 24, 1, 800000)
    surface, targets, _ = prepare(dataset, req, base, tokenizer, codec, device)
    with torch.no_grad():
        features, _ = model.backbone.encode(surface)
        status['teacher_prefix'] = check_teacher_prefix(model.backbone, features, targets)
    ids = targets['main_input_ids'][0, 0][targets['main_attention_mask'][0, 0].bool()].cpu().numpy()
    for i in range(1, len(ids)):
        if ids[i] not in tokenizer.next_posible_token(ids=ids[:i]):
            raise ValueError(f'native grammar rejects teacher token at offset {i}')
    decoded = tokenizer.detokenize(np.asarray(ids))
    if not np.isfinite(decoded.joints).all() or len(decoded.parents) != len(decoded.joints):
        raise ValueError('TokenRig tree cannot feed the existing stage-two interface')
    status.update(complete=True, decoded_joints=len(decoded.joints))
    write_json(output/f'result_rank_{rank}.json', status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--decoder', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--profile', choices=('default', *ENLARGED_BATCHES), default='default')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config['tokenrig_decoder'] = json.loads(args.decoder.read_text())
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if world != 2 or torch.cuda.device_count() != 2:
        raise RuntimeError('this probe requires the existing two-GPU grant')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    total = torch.cuda.get_device_properties(device).total_memory
    # Reserve headroom for NCCL and memory outside PyTorch's allocator.
    fraction = min(76000 * 2**20 / total, 1.)
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    config['runtime']['gpu_memory_fraction_limit'] = fraction
    dist.init_process_group('nccl', timeout=timedelta(minutes=5))
    try:
        probe(config, args.output, device, rank, world, args.profile)
    except Exception:
        (args.output/f'failure_rank_{rank}.txt').write_text(traceback.format_exc())
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
