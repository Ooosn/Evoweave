"""Fixed paired decoding audit with exact baseline replay before new policies."""
import gc
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from ar_only_runtime import load_evaluation_model
from diagnostic_decoding import POLICIES, generate, options
from fusion_evaluation import tree_fingerprint
from predicted_reference_protocol import ROOT, bound_config, reference_from_row, scientific_sha, sha256, summarize
from predicted_reference_runtime import head_prediction, score_output
from predicted_reference_smoke import load_records
from reference_flow_runtime import mirror
from stage1_feedback import score_generation, summarize_rows
from stage1_official_inference import OFFICIAL_OPTIONS, reference_seed
from stage1_surface_repair_runtime import canonical_identity
from virtual_query_xyz_runtime import write_json
from rigweave.dynamic_rig.anchor_joint_binding import AnchorJointBinding, AnchorJointConfig

VERSION = 'paired_decode_diagnosis_20261031_v1'
OLD_SHA = ('b2a606bd1ed075539c7f7b1630a625454458d0ae092d1fdae759009f71312852',
           '3ea27d7537852a42aab043938fbd7f341aaf3127a2edba907b5648506950e701')
MODEL_NAMES = ('old', 'full')


def validate_diagnosis(config):
    spec = config['inference_diagnosis']
    if (spec['version'] != VERSION or spec['policies'] != list(POLICIES)
            or set(spec['models']) != set(MODEL_NAMES) or not spec['require_exact_baseline_replay']
            or spec['optimizer_updates'] != 0 or spec['select_using_GT'] or spec['retry_failures']
            or config['runtime']['world_size'] != 1 or config['capacity']['full_official_beam_decode_per_rank'] != 0
            or config['official_smoke']['full_beam_capacity_first']):
        raise ValueError('unreviewed paired decoding diagnosis')
    old, full = (spec['models'][name] for name in MODEL_NAMES)
    if (tuple(old['checkpoints'][stage]['sha256'] for stage in ('stage1', 'stage2')) != OLD_SHA
            or full['checkpoints'] != config['checkpoints']
            or any(item['layout'] != 'anchor' for model in (old, full) for item in model['checkpoints'].values())):
        raise ValueError('diagnostic checkpoint selection differs')


def read_bound(spec):
    if sha256(spec['path']) != spec['sha256']:
        raise ValueError('diagnostic primary artifact changed: '+spec['path'])
    return json.loads(Path(spec['path']).read_text())


def saved_references(model_spec):
    result = read_bound(model_spec['recorded_run']['result'])
    resources = read_bound(model_spec['recorded_run']['resources'])
    if (not result['complete'] or not result['passed'] or not resources['passed']
            or resources['exit_code'] != 0 or not resources['payload_cleanup_complete']
            or result['checkpoints'] != model_spec['checkpoints']
            or result['policy']['generation_kwargs'] != OFFICIAL_OPTIONS):
        raise ValueError('prior official run provenance differs')
    profiles = [item['beam_capacity'] for item in result['artifacts']]
    if not profiles or any(not p['passed'] or (p['new_tokens'], p['num_beams'], p['input_frames'])
                           != (2048, 15, 8) for p in profiles):
        raise ValueError('prior full beam capacity is not established')
    references = {}
    for spec in model_spec['recorded_run']['references']:
        rows = read_bound(spec)
        if references.keys() & rows.keys():
            raise ValueError('duplicate saved reference')
        references.update(rows)
    if len(references) != 12:
        raise ValueError('saved official references are incomplete')
    return references


def environment():
    return {'python': sys.version, 'packages': {name: importlib.metadata.version(name)
            for name in ('torch', 'transformers', 'numpy', 'tokenizers')}}


def audit(config, output, source):
    records = load_records(config)
    assets = {r['asset_id'] for r in records}
    for model_spec in config['inference_diagnosis']['models'].values():
        refs = saved_references(model_spec)
        if assets != refs.keys():
            raise ValueError('saved reference assets differ from canonical inputs')
        for checkpoint in model_spec['checkpoints'].values():
            if sha256(checkpoint['path']) != checkpoint['sha256']:
                raise ValueError('frozen diagnostic checkpoint changed')
        for record in records:
            identity = canonical_identity(record, 0)
            if any(refs[record['asset_id']][field] != identity[field] for field in identity):
                raise ValueError('old/new official input identities differ')
    subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet', config['prior_source'], source, '--',
        'model_training/rigweave/src', 'model_training/rigweave/scripts/ar_only_runtime.py',
        'model_training/rigweave/scripts/stage1_feedback.py',
        'model_training/rigweave/scripts/stage1_official_inference.py'], check=True)
    subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s',
        str(ROOT/'model_training/rigweave/tests'), '-p', 'test_diagnostic_decoding.py', '-v'],
        check=True, cwd=ROOT)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU diagnosis audit initialized CUDA')
    result = {'complete': True, 'passed': True, 'source': source, 'scientific_sha256': scientific_sha(config),
        'assets': sorted(assets), 'optimizer_updates': 0, 'cuda_initialized': False,
        'environment': environment(), 'policies': list(POLICIES), 'baseline_reference_count': 24,
        'inherited_capacity': 'Both recorded model pipelines have full15beam/2048token capacity receipts.'}
    write_json(output/'result.json', result)
    return result


@torch.inference_mode()
def run(config, output, source, device, rank, world):
    if (rank, world) != (0, 1):
        raise ValueError('diagnosis uses only the single admitted native GPU subset')
    admitted = read_bound(config['audit_result'])
    if not admitted['passed'] or admitted['scientific_sha256'] != scientific_sha(config):
        raise ValueError('diagnosis CPU admission differs')
    started = time.monotonic()
    def boundary():
        from virtual_pair_guard import deadline_due
        if deadline_due(config) or time.monotonic()-started > config['evaluation']['payload_limit_seconds']:
            raise RuntimeError('bounded diagnosis deadline reached')
    def progress(phase, **details):
        write_json(output/'progress_rank_0.json', dict(phase=phase, seconds=time.monotonic()-started, **details))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    original = bound_config(config['parent_config'])
    records = load_records(config, rank)
    models = config['inference_diagnosis']['models']
    references = {name: {policy: {} for policy in POLICIES} for name in MODEL_NAMES}
    conditions, reproduction = {}, []
    for name in MODEL_NAMES:
        boundary()
        progress('load_baseline', model=name)
        model, tokenizer = load_evaluation_model(original, device, models[name]['checkpoints']['stage1'])
        saved = saved_references(models[name])
        conditions[name] = {}
        for record in records:
            boundary()
            asset = record['asset_id']
            surface = {k: v.to(device) for k, v in record['surface'].items()}
            if tree_fingerprint(surface) != record['surface_sha256']:
                raise ValueError('device canonical surface differs')
            real, virtual = model.encode(surface)
            condition = model.conditions(real, virtual)[:, 0]
            conditions[name][asset] = condition.detach().cpu()
            seed = reference_seed(asset, record['raw_frames'][0], config['official_smoke']['base_seed'])
            tick = time.monotonic()
            generated = generate(model.xyz_decoder, condition, tokenizer, seed=seed, policy='official')
            row = score_generation(generated, tokenizer, record['truth'][0].numpy(),
                record['parents'].tolist(), float(record['diagonals'][0]))
            row.update(canonical_identity(record, 0), policy=generated['policy'], seed=seed,
                generation_kwargs=generated['generation_kwargs'], generation_seconds=time.monotonic()-tick,
                condition_sha256=tree_fingerprint(condition), panel_index=record['panel_index'])
            references[name]['official'][asset] = row
            equal = row['generated_tokens'] == saved[asset]['generated_tokens'] and row['status'] == saved[asset]['status']
            reproduction.append({'model': name, 'case': record['panel_index'], 'exact_tokens_and_status': equal})
            write_json(output/f'references_{name}_official.json', references[name]['official'])
            progress('baseline_reproduction', model=name, completed=len(references[name]['official']),
                case=record['panel_index'], exact_tokens_and_status=equal)
            del real, virtual, condition, surface, generated
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    write_json(output/'baseline_reproduction.json', reproduction)
    if len(reproduction) != 24 or not all(row['exact_tokens_and_status'] for row in reproduction):
        raise RuntimeError('baseline replay differs; stop before any new-policy experiment')
    for name in MODEL_NAMES:
        boundary()
        progress('load_policy_comparison', model=name)
        model, tokenizer = load_evaluation_model(original, device, models[name]['checkpoints']['stage1'])
        for record in records:
            asset = record['asset_id']
            condition = conditions[name][asset].to(device)
            seed = reference_seed(asset, record['raw_frames'][0], config['official_smoke']['base_seed'])
            for policy in POLICIES[1:]:
                boundary()
                tick = time.monotonic()
                generated = generate(model.xyz_decoder, condition, tokenizer, seed=seed, policy=policy)
                row = score_generation(generated, tokenizer, record['truth'][0].numpy(),
                    record['parents'].tolist(), float(record['diagonals'][0]))
                row.update(canonical_identity(record, 0), policy=generated['policy'], seed=seed,
                    generation_kwargs=options(policy), generation_seconds=time.monotonic()-tick,
                    condition_sha256=tree_fingerprint(condition), panel_index=record['panel_index'])
                references[name][policy][asset] = row
                write_json(output/f'references_{name}_{policy}.json', references[name][policy])
                progress('policy_comparison', model=name, policy=policy, case=record['panel_index'],
                    completed=len(references[name][policy]), status=row['status'])
            del condition, generated
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    del conditions
    all_rows, artifacts = [], []
    for name in MODEL_NAMES:
        boundary()
        progress('load_pose_prediction', model=name)
        checkpoint = models[name]['checkpoints']['stage2']
        backbone, _ = load_evaluation_model(original, device, checkpoint)
        head = AnchorJointBinding(AnchorJointConfig(**original['architecture']), 'reference_transport').to(device).eval()
        payload = torch.load(checkpoint['path'], map_location='cpu', mmap=True, weights_only=False)
        head.load_state_dict({k.removeprefix('head.'): v for k, v in payload['model'].items() if k.startswith('head.')}, strict=True)
        del payload
        examples = {policy: [] for policy in POLICIES}
        for record in records:
            boundary()
            asset = record['asset_id']
            surface = {k: v.to(device) for k, v in record['surface'].items()}
            features, _ = backbone.encode(surface)
            truth = record['truth'].to(device)
            gt_parents = record['parents'].tolist()
            for policy in ('GT',)+POLICIES:
                if policy == 'GT':
                    xyz, parents, status = truth[0], gt_parents, 'valid'
                else:
                    parsed, status = reference_from_row(references[name][policy][asset])
                    xyz, parents = (torch.as_tensor(parsed[0], device=device), parsed[1]) if parsed else (None, None)
                prediction = head_prediction(head, features, surface, xyz, parents, 0) if xyz is not None else None
                asset_rows = []
                for method in ('head', 'copy'):
                    for frame in range(1, 8):
                        value = (prediction[frame] if method == 'head' else xyz) if xyz is not None else None
                        row = score_output(value, parents, truth[frame], gt_parents,
                            float(record['diagonals'][frame]), known_ids=policy == 'GT', status=status)
                        row.update(canonical_identity(record, frame), model=name, decoding_policy=policy,
                            method=('GT_' if policy == 'GT' else 'predicted_')+method,
                            reference_slot=0, target_slot=frame, reference_kind='GT' if policy == 'GT' else 'predicted',
                            reference_raw_frame=record['raw_frames'][0], stage1_cache_reused=False)
                        asset_rows.append(row)
                        all_rows.append(row)
                if policy != 'GT':
                    dense = record['surface']['dense_points'][0]
                    step = max(1, (dense.shape[1]+4095)//4096)
                    examples[policy].append({'asset_id': asset, 'raw_frames': record['raw_frames'],
                        'panel_index': record['panel_index'], 'visual_points': dense[:, ::step].clone(),
                        'truth': record['truth'], 'parents': record['parents'], 'rows': asset_rows,
                        'reference': references[name][policy][asset]})
            progress('pose_prediction', model=name, case=record['panel_index'])
            del features, surface, truth, prediction
        for policy, values in examples.items():
            path = output/f'examples_{name}_{policy}.pt'
            torch.save(values, path)
            artifacts.append(mirror(path, path.name, config))
        del backbone, head, examples
        gc.collect()
        torch.cuda.empty_cache()
    path = output/'rows.json'
    write_json(path, all_rows)
    artifacts.append(mirror(path, path.name, config))
    summaries = {}
    for name in MODEL_NAMES:
        summaries[name] = {}
        for policy in ('GT',)+POLICIES:
            rows = [r for r in all_rows if r['model'] == name and r['decoding_policy'] == policy]
            if len(rows) != 168:
                raise ValueError('diagnostic row denominator changed')
            summaries[name][policy] = {method: summarize([r for r in rows if r['method'].endswith('_'+method)])
                                      for method in ('head', 'copy')}
            if policy != 'GT':
                summaries[name][policy]['reference'] = summarize_rows(list(references[name][policy].values()))
                ref_path = output/f'references_{name}_{policy}.json'
                artifacts.append(mirror(ref_path, ref_path.name, config))
    artifacts.append(mirror(output/'baseline_reproduction.json', 'baseline_reproduction.json', config))
    result = {'complete': True, 'passed': True, 'source': source, 'scientific_sha256': scientific_sha(config),
        'optimizer_updates': 0, 'exact_baseline_reproductions': 24, 'environment': environment(),
        'summaries': summaries, 'artifacts': artifacts, 'seconds': time.monotonic()-started,
        'scope': 'Two frozen pipelines x three predeclared policies x12 fixed cases. No retries, fitting, GT selection, or validation-wide acceptance.',
        'quality_accepted': False}
    write_json(output/'result.json', result)
    mirror(output/'result.json', 'result.json', config)
    return result
