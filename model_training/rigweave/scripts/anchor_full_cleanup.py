"""Explicit legacy-checkpoint retirement with verified model-only replacements."""
from collections import Counter
import gc
import json
import os
from pathlib import Path

import torch

from reference_flow_runtime import mirror
from reference_flow_protocol import sha256
from virtual_query_xyz_runtime import write_json

ROOT = Path('/work/wangyy/virtual_query_matched_5k_20261025')
RETAINED = ROOT/'retained_models_20261030'
DURABLE = Path('/home/wangyy/evorig/outputs/retained_models_20261030')
SOURCE_DURABLE = Path('/home/wangyy/evorig/outputs')
STAGES = (1, 2, 3, 4, 7, 8, 9, 10, 16, 17, 18, 19, 21)
KEEP_FULL = {
    'train_v9/fraction20_40_query/checkpoint_020000.pt',
    'train_v17/reference_transport/checkpoint_060000.pt',
    'train_v17/target_content/checkpoint_060000.pt',
    'train_v19/joint_pretrained/checkpoint_080000.pt',
    'train_v19/pretrained_stage2/checkpoint_080000.pt',
    'train_v19/scratch_stage2/checkpoint_080000.pt',
    'train_v21/checkpoint_040000.pt',
}


def validated_paths(spec):
    path, durable = Path(spec['path']), Path(spec['durable_path'])
    allowed_work = (ROOT/'outputs').resolve()
    allowed_durable = SOURCE_DURABLE.resolve()
    if (allowed_work not in path.resolve().parents or allowed_durable not in durable.resolve().parents
            or path.name != durable.name or not path.name.startswith('checkpoint_')
            or path.suffix != '.pt' or path.is_symlink() or durable.is_symlink()):
        raise ValueError('legacy checkpoint escaped the approved model-output roots')
    if not spec.get('complete') or not spec.get('both_copies_verified'):
        raise ValueError('legacy checkpoint has no verified original mirror')
    return path, durable


def best_nodes(directory):
    scores = {}
    for path in sorted(directory.glob('evaluation_*/summary.json')):
        value = json.loads(path.read_text())
        if not value.get('complete'):
            continue
        node = int(path.parent.name.split('_')[-1])
        groups = value.get('groups')
        if isinstance(groups, dict) and 'original_edge_f1' in groups.get('all', {}):
            scores.setdefault('AR_F1', []).append((groups['all']['original_edge_f1'], node))
        elif isinstance(groups, list):
            for group in groups:
                if group.get('split') == 'valid' and group.get('method') == 'head':
                    if 'normalized_mpjpe' in group:
                        scores.setdefault('XYZ_negative_MPJPE', []).append((-group['normalized_mpjpe'], node))
                    if 'edge_f1' in group:
                        scores.setdefault('XYZ_F1', []).append((group['edge_f1'], node))
    return {max(values)[1] for values in scores.values()}, bool(scores)


def plan_legacy():
    rows = []
    for stage in STAGES:
        directory = ROOT/'outputs'/f'train_v{stage}'
        by_parent = {}
        for path in sorted(directory.rglob('checkpoint_*.pt')):
            by_parent.setdefault(path.parent, []).append(path)
        for parent, paths in by_parent.items():
            best, known = best_nodes(parent)
            final = max(int(path.stem.split('_')[-1]) for path in paths)
            for path in paths:
                spec = json.loads(path.with_suffix('.json').read_text())
                local, durable = validated_paths(spec)
                if local != path or any(p.stat().st_size != spec['bytes'] for p in (local, durable)):
                    raise ValueError('legacy file or size differs from its sidecar')
                relative = path.relative_to(ROOT/'outputs').as_posix()
                samples = int(path.stem.split('_')[-1])
                if relative in KEEP_FULL:
                    action, reason = 'keep_full', 'critical donor, final comparator or continuation recovery point'
                elif samples == final or samples % 10000 == 0 or samples in best or not known:
                    action, reason = 'compact_model', 'final/best/10k comparison node; preserve unknown-quality nodes conservatively'
                else:
                    action, reason = 'remove_intermediate', 'redundant 5k intermediate; final, best and10k nodes retained'
                rows.append({'relative': relative, 'action': action, 'reason': reason, 'samples': samples,
                    'checkpoint': spec, 'local_mtime_ns': local.stat().st_mtime_ns,
                    'durable_mtime_ns': durable.stat().st_mtime_ns})
    if not KEEP_FULL <= {r['relative'] for r in rows}:
        raise RuntimeError('a required complete recovery checkpoint is missing')
    return {'complete': True, 'authorized_scope': 'User approved full training and useful-checkpoint cleanup on2026-10-30',
        'rows': rows, 'counts': dict(Counter(r['action'] for r in rows)),
        'full_bytes_before_both_copies': sum(r['checkpoint']['bytes']*2 for r in rows),
        'full_bytes_selected_for_removal_both_copies': sum(r['checkpoint']['bytes']*2 for r in rows if r['action'] != 'keep_full'),
        'never_delete': ['official weights', 'datasets', 'configs', 'logs', 'metrics', 'visuals', 'KEEP_FULL'],
        'new_model_only_roots': [str(RETAINED), str(DURABLE)]}


def compact(row):
    spec = row['checkpoint']
    relative = Path(row['relative']).with_name('weights_'+Path(row['relative']).name.removeprefix('checkpoint_'))
    target = RETAINED/relative
    sidecar = target.with_suffix('.json')
    if sidecar.exists():
        prior = json.loads(sidecar.read_text())
        if (prior.get('source_checkpoint_sha256') != spec['sha256'] or not prior.get('tensor_roundtrip_verified')
                or sha256(prior['path']) != prior['sha256'] or sha256(prior['durable_path']) != prior['sha256']):
            raise RuntimeError('retained model export no longer verifies')
        return prior
    payload = torch.load(spec['path'], map_location='cpu', mmap=True, weights_only=False)
    if not isinstance(payload.get('model'), dict) or not payload['model']:
        raise ValueError('legacy checkpoint does not contain complete model weights')
    metadata = {k: payload[k] for k in ('configuration', 'config', 'initialization', 'variant',
        'asset_exposures', 'additional_asset_exposures', 'global_backbone_asset_exposures',
        'input_frame_exposures', 'step', 'code_commit', 'source_sha256', 'scientific_sha256',
        'contract_sha256') if k in payload}
    export = {**metadata, 'model': payload['model'], 'model_only': True,
        'optimizer_and_rng_removed': True, 'source_checkpoint_sha256': spec['sha256'],
        'original_checkpoint_path': spec['path']}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.pt.tmp')
    if target.exists() or temporary.exists():
        raise RuntimeError('unreviewed partial legacy export exists')
    torch.save(export, temporary)
    checked = torch.load(temporary, map_location='cpu', mmap=True, weights_only=False)
    if (set(checked['model']) != set(payload['model'])
            or any(not torch.equal(value, checked['model'][name]) for name, value in payload['model'].items())):
        raise RuntimeError('legacy model-only tensor verification failed')
    del checked, payload, export
    gc.collect()
    temporary.replace(target)
    record = {**mirror(target, str(relative), {'artifacts': {'durable_root': str(DURABLE)}}),
        'source_checkpoint_sha256': spec['sha256'], 'tensor_roundtrip_verified': True,
        'model_only': True, 'original_checkpoint': spec, 'reason': row['reason']}
    write_json(sidecar, record)
    write_json(Path(record['durable_path']).with_suffix('.json'), record)
    return record


def execute_legacy(spec, output):
    if sha256(spec['path']) != spec['sha256']:
        raise RuntimeError('reviewed legacy retirement plan changed')
    plan = json.loads(Path(spec['path']).read_text())
    if not plan.get('complete'):
        raise ValueError('incomplete retirement plan')
    journal = ROOT/'outputs/legacy_retirement_20261030.jsonl'
    previous = {row['relative']: row for row in
        (json.loads(line) for line in journal.read_text().splitlines())} if journal.exists() else {}
    records, saved_bytes = [], 0
    for row in plan['rows']:
        if row['action'] == 'keep_full':
            continue
        if row['relative'] in previous:
            old = previous[row['relative']]
            if old['source_sha256'] != row['checkpoint']['sha256']:
                raise RuntimeError('legacy retirement journal differs')
            if not old['complete']:
                local, durable = validated_paths(row['checkpoint'])
                replacement = old.get('replacement')
                if row['action'] == 'compact_model' and (not replacement
                        or any(sha256(replacement[k]) != replacement['sha256'] for k in ('path', 'durable_path'))):
                    raise RuntimeError('interrupted legacy retirement has no verified replacement')
                for path in (local, durable):
                    if path.exists():
                        if sha256(path) != old['source_sha256']:
                            raise RuntimeError('interrupted retirement original changed')
                        path.unlink()
                old['complete'] = True
                with journal.open('a') as handle:
                    handle.write(json.dumps(old)+'\n')
            records.append(old)
            saved_bytes += old['net_bytes_freed']
            continue
        local, durable = validated_paths(row['checkpoint'])
        if (local.stat().st_mtime_ns != row['local_mtime_ns']
                or durable.stat().st_mtime_ns != row['durable_mtime_ns']
                or any(sha256(path) != row['checkpoint']['sha256'] for path in (local, durable))):
            raise RuntimeError('legacy checkpoint changed after retirement planning')
        replacement = compact(row) if row['action'] == 'compact_model' else None
        if row['action'] not in ('compact_model', 'remove_intermediate') or row['relative'] in KEEP_FULL:
            raise RuntimeError('unapproved legacy retirement action')
        # No source model is removed until both new copies and tensor equality pass.
        record = {'relative': row['relative'], 'action': row['action'], 'complete': False,
            'source_sha256': row['checkpoint']['sha256'], 'old_checkpoint': row['checkpoint'],
            'replacement': replacement, 'reason': row['reason'],
            'net_bytes_freed': row['checkpoint']['bytes']*2-(replacement['bytes']*2 if replacement else 0)}
        with journal.open('a') as handle:
            handle.write(json.dumps(record)+'\n')
            handle.flush()
            os.fsync(handle.fileno())
        DURABLE.mkdir(parents=True, exist_ok=True)
        (DURABLE/'retirement.jsonl').write_text(journal.read_text())
        local.unlink()
        durable.unlink()
        record['complete'] = True
        with journal.open('a') as handle:
            handle.write(json.dumps(record)+'\n')
        DURABLE.mkdir(parents=True, exist_ok=True)
        (DURABLE/'retirement.jsonl').write_text(journal.read_text())
        write_json(local.with_suffix('.retired.json'), record)
        write_json(durable.with_suffix('.retired.json'), record)
        records.append(record)
        saved_bytes += record['net_bytes_freed']
        write_json(output/'cleanup_progress.json', {'retired': len(records),
            'planned': sum(r['action'] != 'keep_full' for r in plan['rows']), 'net_bytes_freed': saved_bytes})
        print('RETIRE '+json.dumps({'relative': row['relative'], 'action': row['action'],
            'net_GiB_freed_total': saved_bytes/1024**3}), flush=True)
    DURABLE.mkdir(parents=True, exist_ok=True)
    (DURABLE/'retirement.jsonl').write_text(journal.read_text() if journal.exists() else '')
    return {'complete': True, 'plan': spec, 'retired_checkpoints': len(records),
        'net_bytes_freed': saved_bytes, 'kept_full': sorted(KEEP_FULL),
        'journal': str(journal), 'replacement_roots': [str(RETAINED), str(DURABLE)]}
