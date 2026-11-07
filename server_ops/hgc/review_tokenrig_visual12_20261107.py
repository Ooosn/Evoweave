"""CPU-only rescoring and compact durable evidence for the native visual run."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tarfile

ROOT = Path('/work/wangyy/tokenrig_alignment_20261105')
CODE = ROOT / 'code_native_visual12_20261107'
OUT = ROOT / 'outputs_native_visual12_20261107'
DEST = Path('/home/wangyy/evorig/outputs/tokenrig_alignment_20261105/native_visual12_20261107')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def mean(rows, field):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['asset_id']].append(row[field])
    return sum(sum(v) / len(v) for v in grouped.values()) / len(grouped) if grouped else None


def main():
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0) & {6, 7, 9, 10, 11, 12, 14, 15, 61, 63, 64, 65, 66, 67, 68, 69})})
    os.nice(19)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'RIGWEAVE_DISABLE_OPEN3D', 'RIGWEAVE_DISABLE_LIGHTNING_IMPORT'):
        os.environ[name] = '1'
    sys.path[:0] = [str(CODE / 'model_training/rigweave/src'), str(CODE / 'model_training/rigweave/scripts')]
    import numpy as np
    import torch
    from predicted_reference_protocol import summarize
    from predicted_reference_runtime import score_output
    from stage1_surface_repair_runtime import canonical_identity
    from tokenrig_comparison_matrix import IDENTITY
    from rigweave.dynamic_rig.skeleton_topology_metrics import _topology_metrics
    torch.set_num_threads(1)
    config_path = CODE / 'model_training/experiments/tokenrig_native_visual12_20261107.json'
    config, result, resources = read(config_path), read(OUT / 'capacity/result.json'), read(OUT / 'capacity_resources.json')
    if (not result['complete'] or not result['passed'] or result['optimizer_updates'] != 0
            or not resources['complete'] or not resources['passed'] or resources['exit_code'] != 0
            or not resources['kernel_scheduler_membership'] or not resources['payload_cleanup_complete']):
        raise ValueError('native inference did not complete and clean up')
    if any(v > 78 * 1024 for v in resources['peak_gpu_memory_mib'].values()):
        raise ValueError('whole-card GPU ceiling exceeded')
    audit = read(OUT / 'audit/result.json')
    records = {r['asset_id']: torch.load(r['path'], map_location='cpu', weights_only=True) for r in audit['records']}
    models, files = {}, []
    for label, spec in config['baselines'].items():
        root = Path(spec['root'])
        for name, expected in spec['sha256'].items():
            if sha(root / name) != expected:
                raise ValueError('baseline changed after audit')
        old = read(root / 'result.json')
        example_sha = old['artifacts'][0]['examples']['sha256']
        if sha(root / 'examples_rank_0.pt') != example_sha:
            raise ValueError('baseline visual artifact differs')
        models[label] = {'refs': read(root / 'references_rank_0.json'),
            'rows': read(root / 'rows_rank_0.json'), 'summaries': old['summaries']}
        for name in ('references_rank_0.json', 'rows_rank_0.json', 'examples_rank_0.pt'):
            files.append((root / name, f'models/{label}/{name}'))
    for label, spec in result['models'].items():
        refs, rows = {}, []
        for rank in spec['artifacts']:
            for kind in ('references', 'rows', 'examples'):
                artifact = rank[kind]
                if sha(artifact['path']) != artifact['sha256'] or sha(artifact['durable_path']) != artifact['sha256']:
                    raise ValueError('new inference artifact backup differs')
                files.append((Path(artifact['path']), f'models/{label}/{kind}_rank_{rank["rank"]}.' + ('pt' if kind == 'examples' else 'json')))
            refs.update(read(rank['references']['path']))
            rows.extend(read(rank['rows']['path']))
        models[label] = {'refs': refs, 'rows': rows, 'summaries': spec['summaries']}
    summary, per_case = {}, {}
    expected_keys = {(asset, frame) for asset in records for frame in range(1, 8)}
    checked_reference_rows, checked_target_rows = 0, 0
    for label, data in models.items():
        if set(data['refs']) != set(records) or len(data['rows']) != 336:
            raise ValueError('comparison coverage differs')
        for asset, ref in data['refs'].items():
            record = records[asset]
            identity = canonical_identity(record, 0)
            if any(ref[k] != identity[k] for k in IDENTITY):
                raise ValueError('reference canonical identity differs')
            old = models['OPT330k']['refs'][asset]
            if any(ref[k] != old[k] for k in ('policy', 'seed', 'generation_kwargs')):
                raise ValueError('formal inference seed/policy differs')
            if ref['success']:
                xyz, truth = np.asarray(ref['predicted_joints']), record['truth'][0].numpy()
                f1 = _topology_metrics(xyz, ref['predicted_parents'], truth, record['parents'].tolist())['edge_f1']
                distances = np.linalg.norm(xyz[:, None] - truth[None], axis=-1)
                j2j = .5 * (distances.min(0).mean() + distances.min(1).mean()) / float(record['diagonals'][0])
                if abs(f1 - ref['original_edge_f1']) > 1e-12 or abs(j2j - ref['original_normalized_j2j']) > 1e-7:
                    raise ValueError('independent reference rescoring differs')
            elif ref['original_edge_f1'] != 0 or ref['original_normalized_j2j'] is not None:
                raise ValueError('failed reference received quality')
            checked_reference_rows += 1
        for row in data['rows']:
            record, frame = records[row['asset_id']], row['target_slot']
            if any(row[k] != canonical_identity(record, frame)[k] for k in IDENTITY):
                raise ValueError('target canonical identity differs')
            value = torch.tensor(row['predicted_joints']) if row['success'] else None
            rescored = score_output(value, row.get('predicted_parents'), record['truth'][frame],
                record['parents'].tolist(), float(record['diagonals'][frame]),
                known_ids=row['reference_kind'] == 'GT', status=row['status'])
            for key in ('edge_f1', 'normalized_j2j', 'normalized_mpjpe', 'exact_count', 'success'):
                a, b = row[key], rescored[key]
                if (a is None) != (b is None) or (a is not None and abs(float(a) - float(b)) > 1e-7):
                    raise ValueError('independent target rescoring differs: ' + key)
            checked_target_rows += 1
        groups = {m: [r for r in data['rows'] if r['method'] == m] for m in data['summaries']}
        for method, rows in groups.items():
            if len(rows) != 84 or {(r['asset_id'], r['target_slot']) for r in rows} != expected_keys:
                raise ValueError('paired target coverage differs')
            actual = summarize(rows)
            for metric in ('edge_f1', 'normalized_j2j_successful', 'normalized_mpjpe_known_id'):
                a, b = actual[metric], data['summaries'][method][metric]
                if (a is None) != (b is None) or (a is not None and abs(a - b) > 1e-12):
                    raise ValueError('stored pipeline summary differs')
        summary[label] = {'reference_edge_F1': sum(r['original_edge_f1'] for r in data['refs'].values()) / 12,
            'reference_successes': sum(r['success'] for r in data['refs'].values()),
            'summaries': data['summaries']}
        per_case[label] = {}
        for asset in config['evaluation']['asset_ids']:
            rows = [r for r in groups['predicted_head'] if r['asset_id'] == asset]
            gt = [r for r in groups['GT_head'] if r['asset_id'] == asset]
            ref = data['refs'][asset]
            per_case[label][asset] = {'reference_F1': ref['original_edge_f1'],
                'reference_success': ref['success'], 'reference_joints': ref['predicted_joint_count'],
                'GT_joints': ref['gt_joint_count'], 'predicted_head': summarize(rows),
                'GT_MPJPE': mean(gt, 'normalized_mpjpe')}
        data['head'] = {(r['asset_id'], r['target_slot']): r for r in groups['predicted_head']}
    common = set.intersection(*({k for k, r in v['head'].items() if r['success']} for v in models.values()))
    common_scores = {label: mean([v['head'][k] for k in common], 'normalized_j2j') for label, v in models.items()}
    pairs = {}
    for label in ('native250k', 'native300k'):
        for old in ('OPT300k', 'OPT330k'):
            shared = {k for k in expected_keys if models[label]['head'][k]['success'] and models[old]['head'][k]['success']}
            a, b = ([models[n]['head'][k] for k in shared] for n in (label, old))
            wins = losses = 0
            for asset in records:
                aa, bb = ([r for r in rows if r['asset_id'] == asset] for rows in (a, b))
                if not aa:
                    continue
                delta = mean(aa, 'normalized_j2j') - mean(bb, 'normalized_j2j')
                wins += delta < -1e-10
                losses += delta > 1e-10
            pairs[label + '_vs_' + old] = {'all12_failure_zero_target_F1_delta':
                summary[label]['summaries']['predicted_head']['edge_f1'] - summary[old]['summaries']['predicted_head']['edge_f1'],
                'identical_common_success_views': len(shared), 'assets': len({a for a, _ in shared}),
                'new_J2J': mean(a, 'normalized_j2j'), 'old_J2J': mean(b, 'normalized_j2j'),
                'geometry_case_improvements': wins, 'geometry_case_regressions': losses}
    review = {'complete': True, 'passed': True, 'reviewed_utc': datetime.now(timezone.utc).isoformat(),
        'result_sha256': sha(OUT / 'capacity/result.json'), 'resources_sha256': sha(OUT / 'capacity_resources.json'),
        'models': summary, 'per_case': per_case, 'pairwise': pairs,
        'all4_identical_common_success_geometry': {'views': len(common),
            'asset_ids': sorted({a for a, _ in common}), 'normalized_J2J': common_scores},
        'independently_rescored_reference_rows': checked_reference_rows,
        'independently_rescored_target_rows': checked_target_rows,
        'peak_gpu_memory_mib': resources['peak_gpu_memory_mib'], 'optimizer_updates': 0,
        'owned_payload_cleanup_complete': True, 'baseline_generation_performed': False,
        'quality_accepted': False, 'working330k_changed': False,
        'case105_old_exact_token_replay_unresolved': True, 'CUDA_used_in_review': False}
    if torch.cuda.is_initialized():
        raise ValueError('CPU review initialized CUDA')
    review_path = OUT / 'comparison_review.json'
    encoded = (json.dumps(review, indent=2) + '\n').encode()
    for p in (review_path, DEST / 'comparison_review.json'):
        with p.open('xb') as handle:
            handle.write(encoded)
    files += [(review_path, 'comparison_review.json'), (config_path, 'config.json'),
        (OUT / 'audit/development_matrix.json', 'development_matrix.json'),
        (OUT / 'capacity/result.json', 'result.json'), (OUT / 'capacity_resources.json', 'resources.json'),
        (OUT / 'audit/result.json', 'audit.json'), (OUT / 'audit_resources.json', 'audit_resources.json')]
    archive = DEST / 'comparison_evidence_20261107.tar.gz'
    with tarfile.open(archive, 'x:gz') as handle:
        for path, name in files:
            handle.add(path, arcname=name, recursive=False)
    with tarfile.open(archive, 'r:gz') as handle:
        for path, name in files:
            digest = hashlib.sha256(handle.extractfile(name).read()).hexdigest()
            if digest != sha(path):
                raise ValueError('comparison evidence archive verification failed')
    print(json.dumps({'review': str(review_path), 'review_sha256': sha(review_path),
        'archive': str(archive), 'archive_sha256': sha(archive), 'archive_bytes': archive.stat().st_size,
        'models': summary, 'pairwise': pairs, 'common_geometry': review['all4_identical_common_success_geometry']}), flush=True)


if __name__ == '__main__':
    main()
