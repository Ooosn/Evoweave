#!/usr/bin/env python3
"""Read-only localization of a fixed two-frame fitting failure."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'model_training/rigweave/src'))
from virtual_query_xyz_runtime import build_model, read_entries, fixed_dataset, prepare_batch, sha256, write_json
from probe_virtual_query_xyz import choose_rows, evaluate


def verify(args, config):
    state = json.loads((ROOT/'model_training/state/current.json').read_text())
    operation = state['active_operation']
    if state['architecture_contract']['id'] != 'virtual_query_xyz_aux_ids_v1':
        raise ValueError('wrong active model route')
    if not operation.get('gpu_work_allowed') or operation.get('formal_training_allowed'):
        raise ValueError('read-only GPU diagnosis is not authorized')
    if operation['mode'] != 'diagnose' or operation['operation'] != 'matched_eval':
        raise ValueError('wrong bounded operation')
    if Path(operation['config']).resolve() != args.config.resolve() or operation['config_sha256'] != sha256(args.config):
        raise ValueError('configuration differs from recorded diagnosis')
    if Path(operation['output']).resolve() != args.output.resolve():
        raise ValueError('wrong diagnostic output')
    subprocess.run([sys.executable, str(ROOT/'model_training/tools/agent_work_guard.py'),
                    'check', '--operation', 'matched_eval', '--entrypoint', str(Path(__file__).resolve())], check=True)
    if subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
        raise ValueError('diagnosis needs a clean published checkout')
    expected = operation['code_commit']
    subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', expected, 'HEAD'], check=True)
    subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet', expected, 'HEAD', '--',
                    'model_training/rigweave', 'model_training/tools'], check=True)
    spec = config['diagnostic']
    for name, key in [('checkpoint', 'checkpoint_sha256'), ('evaluation', 'evaluation_sha256')]:
        if sha256(spec[name]) != spec[key]:
            raise ValueError('diagnostic source hash mismatch: '+name)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(config['runtime']['physical_gpu']):
        raise ValueError('wrong visible GPU')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('diagnosis requires one granted GPU')
    return subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()


def pair_stats(values):
    a, b = values[0, 0].float(), values[0, 1].float()
    delta = a-b
    return {'shape': list(values.shape), 'dtype': str(values.dtype),
            'rms': float(values.float().square().mean().sqrt()),
            'difference_rms': float(delta.square().mean().sqrt()),
            'relative_l2': float(delta.norm()/((a.norm()+b.norm())*.5).clamp_min(1e-12)),
            'mean_absolute_difference': float(delta.abs().mean()),
            'max_absolute_difference': float(delta.abs().max()),
            'equal_fraction': float((a==b).float().mean()),
            'flattened_cosine': float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0)),
            'slot_mean_cosine': float(F.cosine_similarity(a, b, dim=-1).mean())}


def eager_decoder_for_diagnosis(model):
    """Use the installed reference operator with exactly the same learned tensors."""
    from transformers.models.opt.modeling_opt import OPTAttention
    decoder = model.xyz_decoder.model.decoder
    config = deepcopy(model.xyz_decoder.config)
    config._attn_implementation = 'eager'
    for layer in decoder.layers:
        previous = layer.self_attn
        reference = OPTAttention(config=config, is_decoder=True).to(previous.q_proj.weight.device)
        reference.load_state_dict(previous.state_dict(), strict=True)
        if any(not torch.equal(value, reference.state_dict()[key]) for key, value in previous.state_dict().items()):
            raise ValueError('attention backend conversion changed learned tensors')
        layer.self_attn = reference.requires_grad_(False).eval()
    decoder._use_flash_attention_2 = False
    decoder._use_sdpa = False
    decoder.config._attn_implementation = 'eager'


@torch.no_grad()
def run(args, config):
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.95, device)
    model, tokenizer, codec, _ = build_model(config, device)
    payload = torch.load(config['diagnostic']['checkpoint'], map_location='cpu', mmap=True, weights_only=False)
    if payload['schema'] != 'virtual_query_xyz_aux_ids_v1' or payload['step'] != 128:
        raise ValueError('unexpected diagnostic checkpoint')
    model.load_state_dict(payload['model'], strict=True)
    del payload
    model.requires_grad_(False).eval()
    rows = choose_rows(read_entries(config)['train'], config)
    if [r['asset_id'] for r in rows] != config['diagnostic']['asset_ids']:
        raise ValueError('fixed asset replay changed')
    dataset = fixed_dataset(rows, config, config['probe']['frames'])
    items = [prepare_batch(dataset[i], config, tokenizer, codec, device) for i in range(len(dataset))]
    if len(items) != 1 or items[0][2]['selected_frames'].cpu().tolist() != config['diagnostic']['frames']:
        raise ValueError('expected exactly the original two-frame case')
    previous = json.loads(Path(config['diagnostic']['evaluation']).read_text())
    reproduced = evaluate(model, items, tokenizer, config, args.output, 128)
    for old, new in zip(previous['rows'], reproduced['rows'], strict=True):
        for key in ('asset_id', 'raw_frame_index', 'target_tokens', 'generated_tokens', 'status'):
            if old[key] != new[key]:
                raise ValueError('fixed-fit replay failed: '+key)
    model.eval()
    surface, targets, geometry = items[0]
    stages = {'query_geometry': pair_stats(surface['query_points'])}
    hooks = []
    def capture_surface(module, args):
        stages['surface_tokens'] = pair_stats(args[0])
    hooks.append(model.motion_encoder.register_forward_pre_hook(capture_surface))
    # Temporal modules process aligned slots as [Q+registers+role,T+1,D].
    offset = 1+model.config.register_tokens
    def temporal_hook(index):
        def capture(module, args, output):
            values = output[offset:, :2].permute(1, 0, 2)[None]
            stages[f'motion_block_{index:02d}'] = pair_stats(values)
        return capture
    for index, block in enumerate(model.motion_encoder.blocks, 1):
        hooks.append(block.anchor_temporal.register_forward_hook(temporal_hook(index)))
    with torch.autocast('cuda', dtype=torch.bfloat16):
        real, virtual = model.encode(surface)
        conditions = model.conditions(real, virtual)
    for hook in hooks:
        hook.remove()
    stages['final_real'] = pair_stats(real)
    stages['final_condition'] = pair_stats(conditions)
    write_json(args.output/'feature_stages.json', stages)
    ids = targets['main_input_ids'][0]
    mask = targets['main_attention_mask'][0]
    differences = (ids[0] != ids[1]).nonzero().flatten()
    fork = int(differences[0])
    prefix = ids[:1, :fork]
    candidates = ids[:, fork].tolist()
    controls = {'frame_0': conditions[:, 0], 'frame_1': conditions[:, 1],
                'mean': conditions.mean(1), 'zero': torch.zeros_like(conditions[:, 0])}
    fork_rows, teacher_rows = [], []
    for precision in ('bfloat16_flash', 'bfloat16_eager', 'float32_eager'):
        if precision == 'bfloat16_eager':
            eager_decoder_for_diagnosis(model)
        context = nullcontext() if precision == 'float32_eager' else torch.autocast('cuda', dtype=torch.bfloat16)
        with context:
            for name, condition in controls.items():
                logits = model.main_logits(condition.float(), prefix, torch.ones_like(prefix))[:, -1].float()
                probabilities = logits.softmax(-1)[0]
                fork_rows.append({'precision': precision, 'condition': name,
                    'candidate_tokens': candidates, 'probabilities': probabilities[candidates].tolist(),
                    'logits': logits[0, candidates].tolist(), 'argmax': int(logits.argmax(-1)),
                    'logit_margin_token0_minus_token1': float(logits[0, candidates[0]]-logits[0, candidates[1]])})
                if precision == 'bfloat16_flash':
                    for target_frame in range(2):
                        logits = model.main_logits(condition, ids[target_frame:target_frame+1, :-1],
                            mask[target_frame:target_frame+1, :-1]).float()
                        labels = ids[target_frame, 1:]
                        loss = F.cross_entropy(logits[0], labels, reduction='none')
                        valid = mask[target_frame, 1:]
                        teacher_rows.append({'condition': name, 'target_frame': target_frame,
                            'mean_ce': float(loss[valid].mean()), 'first_fork_ce': float(loss[fork-1]),
                            'accuracy': float((logits[0].argmax(-1)[valid]==labels[valid]).float().mean())})
                write_json(args.output/'readout_progress.json', {'first_fork': fork_rows,
                    'teacher_forcing_controls': teacher_rows})
    report = {'accepted': True, 'accepted_meaning': 'read_only_replay_and_measurement_valid_not_model_quality',
        'optimizer_updates': 0, 'replay_exact': True, 'frames': geometry['selected_frames'].cpu().tolist(),
        'first_different_full_token_index': fork, 'different_target_positions': int(len(differences)),
        'target_candidate_tokens': candidates, 'feature_stages': stages, 'first_fork': fork_rows,
        'teacher_forcing_controls': teacher_rows, 'formal_training_allowed': False,
        'precision_control': 'Same BF16-encoded conditions. Eager OPT attention uses exactly equal learned tensors; BF16 eager separates backend from precision effects.',
        'zero_condition_is_out_of_distribution': True}
    write_json(args.output/'diagnosis.json', report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    source = verify(args, config)
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result = run(args, config)
        write_json(args.output/'result.json', {key: result[key] for key in
            ('accepted', 'accepted_meaning', 'optimizer_updates', 'replay_exact', 'formal_training_allowed')})
        print('DIAGNOSIS_COMPLETE '+json.dumps({'code_commit': source, 'optimizer_updates': 0}), flush=True)
    except Exception:
        write_json(args.output/'result.json', {'accepted': False, 'execution_complete': False,
            'formal_training_allowed': False, 'traceback': traceback.format_exc(), 'code_commit': source})
        raise


if __name__ == '__main__':
    main()
