"""Fixed native Qwen candidates on the retained official 12-case inputs."""
import argparse
from datetime import datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import time
import traceback

from tokenrig_comparison_matrix import IDENTITY, digest, matrix, read
from virtual_pair_guard import NATIVE_VISUAL_PHASES, ROOT, verify_launch

CASES = [5, 15, 23, 24, 30, 44, 54, 61, 70, 95, 105, 108]
CONTINUATION_PHASE = 'tokenrig_native_cont400k_visual12_20261107'


def scientific_sha(config):
    value = {k: v for k, v in config.items() if k not in ('audit_result', 'runtime')}
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_config(config):
    from stage1_official_inference import OFFICIAL_OPTIONS, POLICY
    spec = config['evaluation']
    phase = config['execution']['phase']
    if (phase not in NATIVE_VISUAL_PHASES
            or spec['optimizer_updates'] != 0 or spec['cases'] != CASES
            or spec['generation_kwargs'] != OFFICIAL_OPTIONS or spec['policy'] != POLICY
            or spec['base_seed'] != 20261029 or spec['precision'] != 'fp32_noTF32'
            or spec['reference_slot'] != 0 or spec['target_slots'] != list(range(1, 8))
            or spec['capacity_stress'] or spec['automatic_retry'] or spec['static_fallback']):
        raise ValueError('unreviewed native visual comparison')
    if phase == CONTINUATION_PHASE:
        validate_screening(config)
    elif [c['visits'] for c in config['candidates']] != [250000, 300000]:
        raise ValueError('unreviewed native visual candidates')


def validate_screening(config):
    spec = config['screening']
    if digest(spec['path']) != spec['sha256']:
        raise ValueError('frozen development screening changed')
    report = read(spec['path'])
    curve = report['curve']
    if (not report['complete'] or not report['passed'] or report['assets'] != 400000
            or report['optimizer_updates'] != 8360 or not report['same_parent_panel_input_and_GT']
            or not report['all_ten_recoveries_retained']
            or not report['formal_results_not_used_for_selection']
            or report['rule'] != 'descending_greedy_F1_then_lower_GT_MPJPE_then_earlier_visits'
            or [r['visits'] for r in curve] != list(range(310000, 400001, 10000))):
        raise ValueError('incomplete continuation screening')
    ranked = sorted(curve, key=lambda r: (-r['stage1_edge_F1'], r['GT_MPJPE'], r['visits']))
    expected = [{'label': f"native{r['visits']//1000}k", 'visits': r['visits'],
        'path': r['checkpoint']['path'], 'sha256': r['checkpoint']['sha256']} for r in ranked[:2]]
    if config['candidates'] != expected or report['selected_candidates'] != expected:
        raise ValueError('exactly the two preselected development candidates are allowed')
    return report


def baseline(config, label):
    spec = config['baselines'][label]
    for name, expected in spec['sha256'].items():
        if digest(Path(spec['root']) / name) != expected:
            raise ValueError('retained baseline changed: ' + label + '/' + name)
    result = read(Path(spec['root']) / 'result.json')
    if not result['complete'] or not result['passed'] or result['optimizer_updates'] != 0:
        raise ValueError('retained baseline was not accepted')
    refs = read(Path(spec['root']) / 'references_rank_0.json')
    rows = read(Path(spec['root']) / 'rows_rank_0.json')
    if len(refs) != 12 or len(rows) != 336:
        raise ValueError('retained baseline coverage differs')
    return refs, rows


def retained_native_models(config):
    spec = config['retained_native_baselines']
    if digest(spec['path']) != spec['sha256']:
        raise ValueError('retained native comparison changed')
    prior = read(spec['path'])
    if (not prior['complete'] or not prior['passed'] or prior['optimizer_updates'] != 0
            or set(prior['models']) != {'native250k', 'native300k'}):
        raise ValueError('retained native comparison was not completed')
    return prior['models']


def ranked_artifacts(model, *, durable, verify_backup=False):
    refs, rows, examples = {}, [], []
    if sorted(r['rank'] for r in model['artifacts']) != [0, 1]:
        raise ValueError('two native artifact shards are required')
    for rank in model['artifacts']:
        paths = {}
        for kind in ('references', 'rows', 'examples'):
            receipt = rank[kind]
            path = Path(receipt['durable_path' if durable else 'path'])
            if digest(path) != receipt['sha256'] or (verify_backup
                    and digest(receipt['durable_path']) != receipt['sha256']):
                raise ValueError('native inference artifact differs: ' + str(path))
            paths[kind] = path
        shard = read(paths['references'])
        if set(shard) & set(refs):
            raise ValueError('duplicate reference across native artifact shards')
        refs.update(shard)
        rows.extend(read(paths['rows']))
        examples.append((rank['rank'], paths))
    if len(refs) != 12 or len(rows) != 336:
        raise ValueError('native artifact coverage differs')
    return refs, rows, examples


def audit(config, output, source):
    import torch
    from fusion_evaluation import tree_fingerprint
    from stage1_official_inference import reference_seed
    from stage1_surface_repair_runtime import canonical_identity
    from virtual_query_xyz_runtime import write_json
    baselines = {label: baseline(config, label) for label in config['baselines']}
    continuation = config['execution']['phase'] == CONTINUATION_PHASE
    if continuation:
        for label, model in retained_native_models(config).items():
            baselines[label] = ranked_artifacts(model, durable=True)[:2]
    refs, rows = baselines['OPT300k']
    assets = config['evaluation']['asset_ids']
    if set(assets) != set(refs):
        raise ValueError('predeclared hard-case assets differ')
    for label, (other_refs, other_rows) in baselines.items():
        if set(other_refs) != set(assets):
            raise ValueError('baseline cases differ')
        for asset in assets:
            a, b = refs[asset], other_refs[asset]
            if any(a[k] != b[k] for k in (*IDENTITY, 'policy', 'seed', 'generation_kwargs')):
                raise ValueError('baseline input/seed/policy differs')
            if b['seed'] != reference_seed(asset, b['raw_frame_index'], config['evaluation']['base_seed']):
                raise ValueError('fixed per-case seed differs')
    for spec in config['candidates']:
        if digest(spec['path']) != spec['sha256']:
            raise ValueError('native candidate weights changed')
    parent = ROOT / config['training_config']['relative_path']
    if digest(parent) != config['training_config']['sha256']:
        raise ValueError('native training architecture/config changed')
    report_matrix = validate_screening(config) if continuation else matrix(config)
    write_json(output / 'development_matrix.json', report_matrix)
    archive = config['canonical_archive']
    if digest(archive['path']) != archive['sha256']:
        raise ValueError('retained OPT canonical archive changed')
    cache = output / 'canonical_inputs'
    cache.mkdir()
    specs = {}
    with tarfile.open(archive['path'], 'r') as handle:
        for member in handle.getmembers():
            if not member.name.endswith('.json'):
                continue
            sidecar = json.load(handle.extractfile(member))
            asset = sidecar['identity']['asset_id']
            if asset not in assets:
                continue
            if asset in specs:
                raise ValueError('duplicate retained canonical asset')
            tensor_member = member.name.removesuffix('.json') + '.pt'
            content = handle.extractfile(tensor_member).read()
            import hashlib
            if hashlib.sha256(content).hexdigest() != sidecar['sha256']:
                raise ValueError('retained tensor bytes differ')
            record = torch.load(io.BytesIO(content), map_location='cpu', weights_only=True)
            if record['asset_id'] != asset or tree_fingerprint(record['surface']) != record['surface_sha256']:
                raise ValueError('canonical tensor identity differs')
            for frame in range(8):
                identity = canonical_identity(record, frame)
                for label, (old_refs, old_rows) in baselines.items():
                    matches = [old_refs[asset]] if frame == 0 else [r for r in old_rows
                        if r['asset_id'] == asset and r['target_slot'] == frame]
                    if len(matches) != (1 if frame == 0 else 4) or any(
                            identity[k] != old[k] for old in matches for k in IDENTITY):
                        raise ValueError('canonical input/GT differs from baseline ' + label)
            path = cache / Path(tensor_member).name
            with path.open('xb') as target:
                target.write(content)
            specs[asset] = {'path': str(path), 'sha256': sidecar['sha256'],
                'case': CASES[assets.index(asset)], 'asset_id': asset}
    if set(specs) != set(assets) or torch.cuda.is_initialized():
        raise ValueError('CPU audit failed coverage or initialized CUDA')
    write_json(output / 'result.json', {'complete': True, 'passed': True, 'source': source,
        'scientific_sha256': scientific_sha(config), 'records': [specs[a] for a in assets],
        'canonical_archive': archive, 'all_baseline_input_GT_seed_identities_equal': True,
        'baselines': config['baselines'], 'candidate_weights_verified': config['candidates'],
        'matrix_sha256': digest(output / 'development_matrix.json'),
        'optimizer_updates': 0, 'CUDA_initialized': False})


def load_candidate(config, spec, device):
    import torch
    from anchor_full_protocol import validate_config as validate_training
    from rigweave.dynamic_rig.anchor_joint_binding import AnchorJointBinding, AnchorJointConfig
    from rigweave.dynamic_rig.virtual_path_ablation import VirtualPathAblationRig
    from stage1_feedback import backbone_state
    from virtual_query_xyz_runtime import build_model
    training = read(ROOT / config['training_config']['relative_path'])
    base = validate_training(training)
    base['tokenrig_decoder'] = {**training['tokenrig_decoder'], 'eager': True}
    original, tokenizer, _, _ = build_model(base, torch.device('cpu'), load_official_weights=False)
    backbone = VirtualPathAblationRig(original, virtual_enabled=False)
    head = AnchorJointBinding(AnchorJointConfig(**training['architecture']), 'reference_transport')
    if digest(spec['path']) != spec['sha256']:
        raise ValueError('candidate changed after admission')
    saved = torch.load(spec['path'], map_location='cpu', mmap=True, weights_only=False)
    backbone.load_state_dict(backbone_state(saved, backbone.state_dict(), layout='anchor'), strict=True)
    head.load_state_dict({k.removeprefix('head.'): v for k, v in saved['model'].items()
                         if k.startswith('head.')}, strict=True)
    if (backbone.base.decoder_condition_tokens != 512 or tokenizer.skeleton_class != 'articulation'
            or backbone.xyz_decoder.config._attn_implementation != 'eager'):
        raise ValueError('native decoder interface differs')
    return backbone.to(device).eval(), head.to(device).eval(), tokenizer


def run(config, output, source, device, rank, world):
    import gc
    import torch
    from fusion_evaluation import tree_fingerprint
    from predicted_reference_protocol import reference_from_row, summarize
    from predicted_reference_runtime import head_prediction, score_output
    from reference_flow_runtime import mirror
    from stage1_feedback import score_generation
    from stage1_official_inference import generate_condition, reference_seed, OFFICIAL_OPTIONS, POLICY
    from stage1_surface_repair_runtime import canonical_identity
    from virtual_pair_guard import deadline_due
    from virtual_pair_runtime import gather
    from virtual_query_xyz_runtime import write_json
    spec = config['audit_result']
    if not spec or digest(spec['path']) != spec['sha256']:
        raise ValueError('independently reviewed CPU audit is not bound')
    checked = read(spec['path'])
    if (not checked['passed'] or checked['scientific_sha256'] != scientific_sha(config)
            or checked['candidate_weights_verified'] != config['candidates']):
        raise ValueError('CPU scientific admission differs')
    if config['execution']['phase'] == CONTINUATION_PHASE:
        if digest(spec['resources_path']) != spec['resources_sha256']:
            raise ValueError('independent CPU resource receipt changed')
        resources = read(spec['resources_path'])
        if (not resources['complete'] or not resources['passed'] or resources['exit_code'] != 0
                or resources['source'] != checked['source'] or not resources['kernel_scheduler_membership']
                or not resources['payload_cleanup_complete']
                or resources['allocation'] != config['runtime']['allocation']):
            raise ValueError('CPU audit did not pass on the retained native allocation')
    started = time.monotonic()
    def boundary():
        if deadline_due(config) or time.monotonic() - started > config['evaluation']['payload_limit_seconds']:
            raise RuntimeError('bounded native inference deadline reached')
    all_models = {}
    for candidate in config['candidates']:
        boundary()
        label = candidate['label']
        backbone, head, tokenizer = load_candidate(config, candidate, device)
        before = (tree_fingerprint(backbone.state_dict()), tree_fingerprint(head.state_dict()))
        references, rows, examples = {}, [], []
        with torch.inference_mode():
            for index, item in enumerate(checked['records']):
                if index % world != rank:
                    continue
                boundary()
                if digest(item['path']) != item['sha256']:
                    raise ValueError('admitted hard-case input changed')
                record = torch.load(item['path'], map_location='cpu', weights_only=True)
                asset = record['asset_id']
                surface = {k: v.to(device) for k, v in record['surface'].items()}
                if tree_fingerprint(surface) != record['surface_sha256']:
                    raise ValueError('device input differs')
                features, virtual = backbone.encode(surface)
                condition = backbone.conditions(features, virtual)[:, 0]
                seed = reference_seed(asset, record['raw_frames'][0], config['evaluation']['base_seed'])
                tick = time.monotonic()
                generation = generate_condition(backbone.xyz_decoder, condition, tokenizer, seed=seed,
                    prefix_builder=backbone.base.decoder_prefix)
                ref = score_generation(generation, tokenizer, record['truth'][0].numpy(),
                    record['parents'].tolist(), float(record['diagonals'][0]))
                ref.update(canonical_identity(record, 0), policy=POLICY, seed=seed,
                    generation_kwargs=dict(OFFICIAL_OPTIONS), generation_seconds=time.monotonic() - tick,
                    case_index=item['case'], native_class='articulation', native_prefix_tokens=512)
                references[asset] = ref
                target, gt_parents = record['truth'].to(device), record['parents'].tolist()
                asset_rows = []
                for kind in ('GT', 'predicted'):
                    parsed, status = ((record['truth'][0].numpy(), gt_parents), 'valid') if kind == 'GT' else reference_from_row(ref)
                    xyz, parents, prediction = None, None, None
                    if parsed is not None:
                        xyz, parents = torch.as_tensor(parsed[0], device=device), parsed[1]
                        prediction = head_prediction(head, features, surface, xyz, parents, 0)
                    for method in ('head', 'copy'):
                        for frame in range(1, 8):
                            value = (prediction[frame] if method == 'head' else xyz) if parsed is not None else None
                            row = score_output(value, parents, target[frame], gt_parents,
                                float(record['diagonals'][frame]), known_ids=kind == 'GT', status=status)
                            row.update(canonical_identity(record, frame), method=kind + '_' + method,
                                reference_slot=0, target_slot=frame, reference_kind=kind,
                                reference_raw_frame=record['raw_frames'][0], case_index=item['case'])
                            rows.append(row)
                            asset_rows.append(row)
                dense = record['surface']['dense_points'][0]
                step = max(1, (dense.shape[1] + 4095) // 4096)
                examples.append({'asset_id': asset, 'panel_index': item['case'],
                    'raw_frames': record['raw_frames'], 'points': record['surface']['query_points'][0],
                    'visual_points': dense[:, ::step].clone(), 'truth': record['truth'],
                    'parents': record['parents'], 'rows': asset_rows, 'reference': ref})
                write_json(output / f'{label}_references_rank_{rank}.json', references)
                write_json(output / f'progress_rank_{rank}.json', {'candidate': label,
                    'generated_references': len(references), 'total_local_references': 6,
                    'last_case': item['case'], 'seconds': time.monotonic() - started})
                del features, virtual, condition, surface, target, xyz, prediction
        if before != (tree_fingerprint(backbone.state_dict()), tree_fingerprint(head.state_dict())):
            raise RuntimeError('inference changed model weights')
        del backbone, head, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        row_path, example_path = (output / f'{label}_{name}_rank_{rank}.{suffix}'
            for name, suffix in (('rows', 'json'), ('examples', 'pt')))
        write_json(row_path, rows)
        torch.save(examples, example_path)
        receipt = {'rank': rank, 'rows': mirror(row_path, row_path.name, config),
            'references': mirror(output / f'{label}_references_rank_{rank}.json',
                f'{label}_references_rank_{rank}.json', config),
            'examples': mirror(example_path, example_path.name, config)}
        receipts = gather(receipt, world)
        if rank == 0:
            combined = sum([read(r['rows']['path']) for r in receipts], [])
            refs = {a: row for r in receipts for a, row in read(r['references']['path']).items()}
            if len(combined) != 336 or len(refs) != 12:
                raise RuntimeError('complete hard-case coverage differs')
            groups = {m: [r for r in combined if r['method'] == m]
                for m in ('GT_head', 'GT_copy', 'predicted_head', 'predicted_copy')}
            expected = {(a, f) for a in config['evaluation']['asset_ids'] for f in range(1, 8)}
            if any(len(v) != 84 or {(r['asset_id'], r['target_slot']) for r in v} != expected for v in groups.values()):
                raise RuntimeError('hard-case denominators changed')
            all_models[label] = {'checkpoint': candidate, 'summaries': {m: summarize(v) for m, v in groups.items()},
                'artifacts': receipts, 'reference_edge_F1': sum(r['original_edge_f1'] for r in refs.values()) / 12,
                'reference_successes': sum(r['success'] for r in refs.values()), 'model_weights_unchanged': True}
    if rank == 0:
        result = {'complete': True, 'passed': True, 'source': source,
            'scientific_sha256': scientific_sha(config), 'optimizer_updates': 0,
            'policy': config['evaluation'], 'models': all_models,
            'baselines_reused_not_regenerated': config['baselines'],
            'seconds': time.monotonic() - started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
            'quality_accepted': False, 'working_model_changed': False,
            'previous_case105_exact_token_replay_unresolved': True}
        if config['execution']['phase'] == CONTINUATION_PHASE:
            result['retained_native_baselines'] = config['retained_native_baselines']
            result['screening'] = config['screening']
        write_json(output / 'result.json', result)
        mirror(output / 'result.json', 'result.json', config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('audit', 'capacity'), required=True)
    args = parser.parse_args()
    config = read(args.config)
    source = verify_launch(args, config)
    sys.path.insert(0, str(ROOT / 'model_training/rigweave/src'))
    import torch
    import torch.distributed as dist
    from virtual_query_xyz_runtime import write_json
    validate_config(config)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.mode == 'audit':
        args.output.mkdir(parents=True, exist_ok=False)
        audit(config, args.output, source)
        return
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if world != 2 or torch.cuda.device_count() != 2:
        raise ValueError('use only the retained dual-H100 native grant')
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(config['runtime']['gpu_memory_fraction_limit'], device)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        dist.init_process_group('nccl', timeout=timedelta(minutes=45))
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=False)
        dist.barrier(device_ids=[device.index])
        run(config, args.output, source, device, rank, world)
        dist.barrier(device_ids=[device.index])
    except Exception as exc:
        if args.output.exists():
            write_json(args.output / f'failure_rank_{rank}.json', {'complete': False, 'passed': False,
                'error': repr(exc), 'traceback': traceback.format_exc()})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
