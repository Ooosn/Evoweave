"""Prune only user-approved historical personal checkpoint copies, never SSD."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess

BASE = Path('/home/wangyy/evorig/outputs')
SCOPES = {
    'OPT300k': BASE/'anchor_full_20261030/v1',
    'OPT330k': BASE/'hardcase3608_20261101/v1',
    'OPT340k400k': BASE/'hardcase3608_20261101/continue400k_v1',
    'Qwen250k': BASE/'tokenrig_full_20261104',
}
KEEPS = {
    ('OPT300k', 'checkpoint_300000.pt'): 'a22824dd9134d1840ea60d75254589f1d95f248d495fe98381cc976dbc0fcaa7',
    ('OPT300k', 'weights_300000.pt'): 'bef030ac403cd26d94f493d868bf426f694f047ed93327e1df51b541fe52ba4d',
    ('OPT330k', 'checkpoint_030000.pt'): 'a62dd25991f948298889a68525ee2b0babb9cceb10cfd435e6cd0668422134f7',
    ('OPT330k', 'weights_030000.pt'): 'abaa87aa7b377f17093850981024a4d63529449f68f082e7c6502ea7a5571018',
    ('Qwen250k', 'checkpoint_250000.pt'): '2d2d84e82fa51b0ce1fb4b194482afdd178652241353bc7e5255c41454d57162',
    ('Qwen250k', 'weights_250000.pt'): 'efc308565d44661b2c9655d9bb0268fa8a606b0e7bcc64a97cbf5df61a3c9b75',
}
LIVE = Path('/work/wangyy/tokenrig_alignment_20261105')
EXCLUDED = BASE/'tokenrig_alignment_20261105'
OUTPUT = BASE/'personal_checkpoint_cleanup_20261106_v1'
PATTERN = re.compile(r'(checkpoint|weights)_[0-9]{6}\.pt')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for data in iter(lambda: handle.read(8 * 1024**2), b''):
            value.update(data)
    return value.hexdigest()


def write_new(path, data):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def decision(scope, relative):
    relative = Path(relative)
    if scope not in SCOPES or not PATTERN.fullmatch(relative.name):
        return 'untouched'
    if relative.parent not in (Path('.'), Path('capacity')):
        return 'untouched'
    if (scope, str(relative)) in KEEPS:
        return 'keep'
    return 'delete'


def checked_path(path, root):
    base = BASE.resolve(strict=True)
    root = root.resolve(strict=True)
    if root == base or base not in root.parents:
        raise RuntimeError('checkpoint root escaped the personal output directory')
    if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
        raise RuntimeError('checkpoint is not a regular, non-symlink file')
    resolved = path.resolve(strict=True)
    if (resolved.parent not in (root, root/'capacity')
            or EXCLUDED.resolve() in resolved.parents or LIVE.resolve() in resolved.parents):
        raise RuntimeError('checkpoint path escaped its reviewed historical scope')
    return resolved


def identity(path):
    s = path.lstat()
    return {key: getattr(s, 'st_'+key) for key in ('dev', 'ino', 'size', 'mtime_ns', 'nlink', 'blocks')}


def live_snapshot():
    output = LIVE/'outputs_native_train'
    resources = json.loads((output/'train_resources.json').read_text())
    progress = [json.loads((output/f'train/progress_rank_{rank}.json').read_text()) for rank in (0, 1)]
    code = LIVE/'code_native_train'
    if subprocess.check_output(['git', '-C', str(code), 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('the running source is unexpectedly dirty; do not modify it')
    return {'observed_utc': now(), 'source': resources['source'],
        'complete': resources['complete'], 'error': resources.get('error'),
        'progress': [{key: v[key] for key in ('additional_asset_exposures', 'step')} for v in progress],
        'current_training_root_excluded': str(EXCLUDED), 'running_source_not_modified': True}


def inventory():
    keep, delete = [], []
    for scope, root in SCOPES.items():
        for parent in (root, root/'capacity'):
            if not parent.exists():
                continue
            for path in sorted(parent.iterdir()):
                kind = decision(scope, path.relative_to(root))
                if kind == 'untouched':
                    continue
                resolved = checked_path(path, root)
                sidecar = path.with_suffix('.json')
                meta = json.loads(sidecar.read_text())
                if not meta.get('complete') or meta.get('bytes') != path.stat().st_size:
                    raise RuntimeError('historical checkpoint lacks its verified save receipt')
                record = {'scope': scope, 'path': str(path), 'resolved': str(resolved),
                    'identity': identity(resolved), 'sidecar': str(sidecar), 'sidecar_sha256': sha(sidecar),
                    'saved_content_sha256': meta['sha256'], 'samples': meta['samples'],
                    'model_only': bool(meta.get('model_only'))}
                if kind == 'keep':
                    if meta['sha256'] != KEEPS[(scope, str(path.relative_to(root)))]:
                        raise RuntimeError('a protected model differs from its pinned identity')
                    keep.append(record)
                else:
                    delete.append(record)
    if len(keep) != len(KEEPS):
        raise RuntimeError('not all six protected full/model-only copies were found')
    return {'schema_version': 1, 'created_utc': now(),
        'authorization': '2026-11-06 user requested deleting older personal checkpoint/Adam copies; retain OPT300k, OPT330k, paused Qwen250k; leave current native run for later discussion.',
        'only_historical_personal_model_files': True, 'SSD_not_touched': True,
        'metadata_metrics_visuals_and_sources_not_deleted': True,
        'scope_roots': {key: str(value) for key, value in SCOPES.items()},
        'keep': keep, 'delete': delete, 'delete_bytes': sum(x['identity']['size'] for x in delete),
        'live_training_before': live_snapshot()}


def open_references(candidates):
    wanted = {record[key] for record in candidates for key in ('resolved', 'path')}
    inodes = {(r['identity']['dev'], r['identity']['ino']): r['resolved'] for r in candidates}
    found = []
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            for fd in (process/'fd').iterdir():
                try:
                    value = os.readlink(fd)
                    s = fd.stat()
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    continue
                key = (s.st_dev, s.st_ino)
                if value in wanted or key in inodes:
                    found.append({'pid': int(process.name), 'file': inodes.get(key, value), 'kind': 'open_fd'})
            maps = (process/'maps').read_text()
            for value in wanted:
                if value in maps:
                    found.append({'pid': int(process.name), 'file': value, 'kind': 'mapped'})
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return found


def execute(expected):
    import resource
    plan_path = OUTPUT/'plan.json'
    if sha(plan_path) != expected:
        raise RuntimeError('cleanup plan digest differs from the reviewed plan')
    plan = json.loads(plan_path.read_text())
    if (OUTPUT/'intent.json').exists():
        raise RuntimeError('an execution intent already exists; inspect it instead of retrying')
    if plan['scope_roots'] != {key: str(value) for key, value in SCOPES.items()}:
        raise RuntimeError('cleanup roots changed')
    for kind in ('keep', 'delete'):
        for row in plan[kind]:
            path = checked_path(Path(row['path']), SCOPES[row['scope']])
            if (str(path) != row['resolved'] or identity(path) != row['identity']
                    or sha(Path(row['sidecar'])) != row['sidecar_sha256']
                    or decision(row['scope'], Path(row['path']).relative_to(SCOPES[row['scope']])) != kind):
                raise RuntimeError('a checkpoint changed after inventory')
    references = open_references(plan['delete'])
    if references:
        raise RuntimeError('a deletion candidate is still in use: '+json.dumps(references))
    verified = []
    for row in plan['keep']:
        path = Path(row['resolved'])
        actual = sha(path)
        if actual != row['saved_content_sha256']:
            raise RuntimeError('a protected full/model-only checkpoint failed its fresh SHA256 check')
        verified.append({'path': row['path'], 'sha256': actual, 'bytes': row['identity']['size']})
        print('KEEP_VERIFIED',row['scope'],path.name,flush=True)
    write_new(OUTPUT/'intent.json', {'started_utc': now(), 'plan_sha256': expected,
        'protected_copies_freshly_verified': verified, 'current_training_not_touched': live_snapshot(),
        'execution': {'CUDA_used': False, 'CPU_budget': 1, 'affinity': sorted(os.sched_getaffinity(0)),
            'kernel_cgroups': Path('/proc/self/cgroup').read_text(), 'native_training_job_not_modified': True}})
    removed = []
    with (OUTPUT/'journal.jsonl').open('x', encoding='utf-8') as journal:
        for row in plan['delete']:
            path = checked_path(Path(row['path']), SCOPES[row['scope']])
            if identity(path) != row['identity']:
                raise RuntimeError('candidate changed immediately before removal')
            journal.write(json.dumps({'event': 'unlink_intent', 'path': str(path), 'utc': now()})+'\n')
            journal.flush()
            os.fsync(journal.fileno())
            # No recursive delete: each reviewed filename is unlinked under its pinned parent.
            parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                current = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
                if (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) != tuple(
                        row['identity'][key] for key in ('dev', 'ino', 'size', 'mtime_ns')):
                    raise RuntimeError('candidate changed under its opened parent')
                os.unlink(path.name, dir_fd=parent)
            finally:
                os.close(parent)
            removed.append(row)
            journal.write(json.dumps({'event': 'unlinked', 'path': str(path), 'utc': now()})+'\n')
            journal.flush()
            os.fsync(journal.fileno())
    if any(Path(row['path']).exists() for row in removed):
        raise RuntimeError('a retired checkpoint still exists')
    for row in plan['keep']:
        if identity(Path(row['resolved'])) != row['identity']:
            raise RuntimeError('a protected checkpoint changed during cleanup')
    result = {'complete': True, 'finished_utc': now(), 'plan_sha256': expected,
        'removed_files': len(removed), 'removed_logical_bytes': sum(x['identity']['size'] for x in removed),
        'removed_single_link_allocated_bytes': sum(x['identity']['blocks'] * 512 for x in removed
            if x['identity']['nlink'] == 1), 'kept_files': verified,
        'all_original_save_receipts_preserved': all(Path(x['sidecar']).exists() for x in removed),
        'live_training_after': live_snapshot(), 'SSD_and_current_training_untouched': True,
        'CPU_peak_RSS_KiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write_new(OUTPUT/'completion.json', result)
    print(json.dumps(result),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'execute'))
    parser.add_argument('--plan-sha256')
    args = parser.parse_args()
    permitted = os.sched_getaffinity(0) & {6, 7, 9, 10, 11, 12, 14, 15, 61, 63, 64, 65, 66, 67, 68, 69}
    if not permitted:
        raise RuntimeError('no approved bounded CPU is available for file maintenance')
    os.sched_setaffinity(0, {min(permitted)})
    os.nice(19)
    if args.mode == 'plan':
        OUTPUT.mkdir(exist_ok=False)
        plan = inventory()
        write_new(OUTPUT/'plan.json', plan)
        print(json.dumps({'plan': str(OUTPUT/'plan.json'), 'sha256': sha(OUTPUT/'plan.json'),
            'keep_files': len(plan['keep']), 'delete_files': len(plan['delete']),
            'delete_bytes': plan['delete_bytes'], 'by_scope': {key: {
                'files': sum(x['scope'] == key for x in plan['delete']),
                'bytes': sum(x['identity']['size'] for x in plan['delete'] if x['scope'] == key)} for key in SCOPES}}))
    else:
        if not args.plan_sha256:
            raise ValueError('execution needs the reviewed immutable plan SHA256')
        execute(args.plan_sha256)


if __name__ == '__main__':
    main()
