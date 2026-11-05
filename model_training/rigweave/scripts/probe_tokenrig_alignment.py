"""No-update, matched-input attribution of the pretrained TokenRig interface."""
import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from anchor_full_protocol import scientific_sha, source_hashes, validate_config
from virtual_pair_guard import ROOT, verify_launch


def models(config, device):
    from anchor_full_runtime import initialize
    aligned = initialize(config, device)
    if 'hybrid_dynamic' not in config['alignment_probe']['variants']:
        return aligned, None
    old_config = deepcopy(config)
    old_config['tokenrig_decoder'].pop('conditioning')
    old_config['backbone']['initialization'] = 'official_UniRig_only'
    old = initialize(old_config, device)
    return aligned, old


def audit(config, output, source):
    from anchor_full_runtime import head_digest
    from virtual_query_xyz_runtime import write_json
    aligned, old = models(config, torch.device('cpu'))
    model, optimizer, _, _, _, initial = aligned
    projection = model.backbone.base.decoder_condition_projection
    state = torch.load(config['tokenrig_decoder']['checkpoint'], map_location='cpu', mmap=True, weights_only=False)['state_dict']
    generator = torch.Generator().manual_seed(20261105)
    value = torch.randn((2, 5, 512), generator=generator)
    native = F.rms_norm(F.linear(value, state['output_proj.0.weight'].float(),
                                 state['output_proj.0.bias'].float()),
                        (896,), state['output_proj.1.weight'].float())
    lifted = (torch.cat((value, value), dim=-1) if config['tokenrig_decoder']['conditioning'].get(
        'motion_initialization') == 'identity_residuals_native_norm_v1' else F.pad(value, (0, 512)))
    actual = projection(lifted)
    torch.testing.assert_close(actual, native, atol=1e-5, rtol=1e-5)
    report = {'complete': True, 'passed': True, 'source': source,
        'scientific_sha256': scientific_sha(config), 'source_sha256': source_hashes(),
        'cuda_initialized': torch.cuda.is_initialized(), 'initialization': initial,
        'old_hybrid_initialization': old[-1] if old else None, 'optimizer_entries': len(optimizer.state),
        'projection_max_absolute_error': float((actual-native).abs().max()),
        'projection_tolerances': {'atol': 1e-5, 'rtol': 1e-5},
        'initial_model_sha256': head_digest(model), 'optimizer_updates': 0,
        'cases': config['alignment_probe']}
    if report['cuda_initialized'] or optimizer.state:
        raise RuntimeError('initialization audit must not use CUDA or Adam state')
    write_json(output/'result.json', report)


@torch.inference_mode()
def probe(config, output, source, device, rank, world):
    from anchor_full_data import CachedFiles
    from anchor_full_evaluation import canonical_input
    from anchor_full_runtime import head_digest
    from reference_flow_runtime import read_all
    from stage1_feedback import score_generation, summarize_rows
    from stage1_surface_repair_runtime import canonical_identity
    from virtual_pair_runtime import evaluation_rows, gather, shared_deadline_due
    from virtual_query_xyz_runtime import fixed_dataset, write_json
    from rigweave.dynamic_rig.virtual_query_xyz import original_xyz_vocabulary_processor

    audited = json.loads(Path(config['execution']['CPU_audit']['path']).read_text())
    resources = json.loads(Path(config['execution']['CPU_audit']['resources_path']).read_text())
    if (not audited['passed'] or not resources['passed'] or not resources['payload_cleanup_complete']
            or audited['source'] != source or audited['source_sha256'] != source_hashes()
            or audited['scientific_sha256'] != scientific_sha(config)):
        raise RuntimeError('this exact initialization has no accepted native CPU audit')
    aligned, hybrid = models(config, device)
    model, optimizer, tokenizer, _, base, initial = aligned
    old = hybrid[0] if hybrid else None
    if initial != audited['initialization'] or head_digest(model) != audited['initial_model_sha256']:
        raise RuntimeError('probe initialization differs from the CPU review')
    model.eval()
    if old is not None:
        old.eval()
    _, entries = read_all(config)
    panel = evaluation_rows(base, entries)
    selected = [panel[i-1] for i in config['alignment_probe']['cases']]
    dataset = fixed_dataset(selected, base, 8, mode='seen_eval', file_resolver=CachedFiles(config, base))
    original = json.loads(Path(config['alignment_probe']['input_reference_rows']).read_text())
    original = {(r['asset_id'], r['frame_index']): r for r in original}
    rows = []
    started = time.monotonic()
    for index in range(rank, len(selected), world):
        if shared_deadline_due(config, device, world):
            raise RuntimeError('allocation save-stop deadline reached')
        record, _ = canonical_input(dataset, index, base, config, 'seen_eval', device)
        surface = {k: v.to(device) for k, v in record['surface'].items()}
        features, virtual = model.backbone.encode(surface)
        dynamic = model.backbone.conditions(features, virtual)
        old_features = old_dynamic = None
        if old is not None:
            old_features, old_virtual = old.backbone.encode(surface)
            old_dynamic = old.backbone.conditions(old_features, old_virtual)
        for frame in config['alignment_probe']['frames']:
            identity = canonical_identity(record, frame)
            reference = original[record['asset_id'], frame]
            if any(identity[k] != reference[k] for k in (
                    'raw_frame_index', 'normalization_sha256', 'surface_sha256', 'original_GT_sha256')):
                raise RuntimeError('probe inputs differ from the retained 250k validation')
            args = [surface[k][:, frame] for k in ('dense_points', 'dense_normals', 'query_points', 'query_normals')]
            native_encoder = model.backbone.surface_tokenizer.michelangelo_encoder
            identity_init = config['tokenrig_decoder']['conditioning'].get(
                'motion_initialization') == 'identity_residuals_native_norm_v1'
            if identity_init:
                from rigweave.dynamic_rig.surface_tokenizer import FixedQuerySurfaceTokenizer
                fixed_raw = FixedQuerySurfaceTokenizer(native_encoder)(*args)
                fixed = torch.cat((fixed_raw, fixed_raw), dim=-1)
            else:
                fixed = model.backbone.surface_tokenizer(*args)
            conditions = {'aligned_static': (model.backbone, fixed),
                'aligned_dynamic': (model.backbone, dynamic[:, frame])}
            if identity_init:
                conditions['aligned_dynamic_fullprefix'] = (model.backbone, dynamic[:, frame])
            if old is not None:
                conditions['hybrid_dynamic'] = (old.backbone, old_dynamic[:, frame])
            if 'native_static' in config['alignment_probe']['variants']:
                _, native, _, _ = native_encoder.encode_latents(pc=args[0], feats=args[1])
                native = torch.cat((native, native), dim=-1) if identity_init else F.pad(native, (0, 512))
                conditions['native_static'] = (model.backbone, native)
            start = torch.tensor([tokenizer.bos, tokenizer.cls_name_to_token('articulation')], device=device)
            for variant in config['alignment_probe']['variants']:
                backbone, condition = conditions[variant]
                t0 = time.monotonic()
                prefix_tokens = backbone.base.decoder_condition_tokens
                try:
                    if variant == 'aligned_dynamic_fullprefix':
                        backbone.base.decoder_condition_tokens = None
                    actual_tokens = backbone.base.decoder_condition_length(condition)
                    prediction = backbone.generate_condition(condition, start, eos_token=tokenizer.eos,
                        max_new_tokens=config['evaluation']['max_new_tokens'],
                        logits_processor=original_xyz_vocabulary_processor(tokenizer, start))
                finally:
                    backbone.base.decoder_condition_tokens = prefix_tokens
                row = {**identity, 'variant': variant, 'seconds': time.monotonic()-t0,
                    'condition_tokens': actual_tokens, 'optimizer_updates': 0,
                    **score_generation(prediction, tokenizer, record['truth'][frame].numpy(),
                        record['parents'].tolist(), float(record['diagonals'][frame]))}
                rows.append(row)
                write_json(output/f'rows_rank_{rank}.json', rows)
                print('PROBE '+json.dumps({k:v for k,v in row.items() if k not in (
                    'generated_tokens', 'predicted_joints', 'predicted_parents')}), flush=True)
        del surface, features, dynamic, old_features, old_dynamic, conditions
    combined = sum(gather(rows, world), [])
    expected = len(selected)*len(config['alignment_probe']['frames'])
    summaries = {}
    for variant in config['alignment_probe']['variants']:
        group = [r for r in combined if r['variant'] == variant]
        if len(group) != expected:
            raise RuntimeError('zero-update probe coverage is incomplete')
        summaries[variant] = summarize_rows(group)
    if optimizer.state or (hybrid is not None and hybrid[1].state):
        raise RuntimeError('probe unexpectedly updated an optimizer')
    report = {'complete': True, 'passed': True, 'source': source,
        'scientific_sha256': scientific_sha(config), 'source_sha256': source_hashes(),
        'optimizer_updates': 0, 'initialization': initial, 'summaries': summaries,
        'cases': config['alignment_probe'], 'precision': 'fp32_noTF32', 'decoder_attention': 'eager',
        'stage1_decoding': 'greedy_diagnostic_not_official_15beam', 'stage2_evaluated': False,
        'matched_input_views': expected, 'seconds': time.monotonic()-started}
    if rank == 0:
        write_json(output/'rows.json', combined)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('audit', 'capacity'), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config['execution']['phase'] != 'tokenrig_alignment_probe':
        raise ValueError('this entrypoint cannot train or execute a capacity stress test')
    validate_config(config)
    source = verify_launch(args, config)
    from virtual_query_xyz_runtime import write_json
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.mode == 'audit':
        args.output.mkdir(parents=True, exist_ok=False)
        audit(config, args.output, source)
        return
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if world != 2:
        raise RuntimeError('the original dual-H100 grant is required')
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        dist.init_process_group('nccl', timeout=timedelta(minutes=45))
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=False)
        dist.barrier(device_ids=[device.index])
        result = probe(config, args.output, source, device, rank, world)
        result['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(args.output/f'result_rank_{rank}.json', result)
        if rank == 0:
            write_json(args.output/'result.json', result)
            print('RESULT '+json.dumps(result), flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
