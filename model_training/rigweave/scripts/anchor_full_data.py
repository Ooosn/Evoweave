"""Manifest-bound SSD staging and complete full-dataset exposure plans."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from anchor_full_protocol import TRAIN_ROWS, cache_identity, sha256
from fit_cohort_selection import _joint_count
from virtual_fusion_normal_runtime import MatchedFrameStream
from virtual_query_xyz_runtime import write_json


def make_stream(base, rank, world, state=None):
    spec = base['data']['manifests']['train']
    return MatchedFrameStream(range(spec['rows']), rank=rank, world_size=world,
        seed=base['data']['stream_seed'], target_assets=base['training']['target_assets'],
        effective_assets=base['training']['effective_assets_per_update'],
        frame_policy=base['training']['frame_policy'], manifest_sha256=spec['sha256'],
        state=state, update_boundaries=base['training']['checkpoint_assets'])


def planned_microbatches(base, stream, limit):
    from virtual_pair_runtime import next_update_batches
    planner = make_stream(base, stream.rank, stream.world_size, stream.state_dict())
    samples = planner.cursor * planner.world_size
    while samples < limit:
        goal, batches = next_update_batches(planner, samples, planner.world_size)
        yield from batches
        samples += goal


def stress_groups(rows):
    counts = [_joint_count(row)[0] for row in rows]
    if len(rows) != TRAIN_ROWS or min(counts) < 1 or max(counts) > 256:
        raise ValueError('full training joint-count metadata differs')
    def geometry(index):
        item = rows[index]['canonical_metrics']
        vertices, faces = item['filtered_vertex_count'], item['filtered_face_count']
        if min(vertices, faces) < 1:
            raise ValueError('invalid full training geometry metadata')
        return vertices * 24 + faces * 3
    return {'largest_joints': sorted(range(len(rows)), key=lambda i: (-counts[i], i))[:3],
        'largest_geometry': sorted(range(len(rows)), key=lambda i: (-geometry(i), i))[:3]}, counts


def verify_ssd(root):
    device = subprocess.check_output(['findmnt', '-n', '-o', 'SOURCE', '-T', str(root)], text=True).strip()
    rows = subprocess.check_output(['lsblk', '-s', '-n', '-o', 'ROTA', device], text=True).split()
    if not rows or any(value != '0' for value in rows):
        raise RuntimeError('training cache destination is not verified nonrotational storage')
    return {'device': device, 'rotational_flags': rows}


def stage_cache(config, base, entries, output):
    spec = config['data_cache']
    root = Path(spec['root'])
    root.mkdir(parents=True, exist_ok=True)
    storage = verify_ssd(root)
    sources, aliases = {}, {}
    for split in ('train', 'valid'):
        for row in entries[split]:
            for original, expected in ((row['path'], None),
                    (row['root_rt']['cache_path'], row['root_rt']['cache_sha256'])):
                source = Path(original).resolve(strict=True)
                key = str(source)
                if key not in sources:
                    stat = source.stat()
                    name = hashlib.sha256(key.encode()).hexdigest()
                    sources[key] = {'source': key, 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                        'expected_sha256': expected, 'cached_path': str(root/name[:2]/(name+'.npz'))}
                elif sources[key]['expected_sha256'] != expected:
                    raise ValueError('conflicting cache source identity')
                aliases[original] = key
    required = sum(v['bytes'] for v in sources.values() if not Path(v['cached_path']).exists())
    if shutil.disk_usage(root).free < required + spec['reserve_free_bytes']:
        raise RuntimeError('insufficient SSD space for the full dataset plus safety reserve')
    started, finished = time.monotonic(), 0

    def copy_one(item):
        source, target = Path(item['source']), Path(item['cached_path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        sidecar = target.with_suffix('.json')
        if target.exists() and sidecar.exists():
            prior = json.loads(sidecar.read_text())
            if (all(prior.get(k) == item[k] for k in ('source', 'bytes', 'mtime_ns'))
                    and target.stat().st_size == item['bytes'] and sha256(target) == prior['sha256']):
                if item['expected_sha256'] and prior['sha256'] != item['expected_sha256']:
                    raise ValueError('retained RT cache digest differs')
                return prior
            raise RuntimeError('existing SSD cache failed verification; do not overwrite it')
        if target.exists() or sidecar.exists():
            raise RuntimeError('incomplete existing SSD cache needs explicit repair')
        temporary = target.with_suffix('.npz.tmp')
        digest = hashlib.sha256()
        with source.open('rb') as src, temporary.open('xb') as dst:
            for block in iter(lambda: src.read(8 * 1024**2), b''):
                dst.write(block)
                digest.update(block)
            dst.flush()
            os.fsync(dst.fileno())
        current = source.stat()
        value = digest.hexdigest()
        if (current.st_size != item['bytes'] or current.st_mtime_ns != item['mtime_ns']
                or temporary.stat().st_size != item['bytes'] or sha256(temporary) != value
                or item['expected_sha256'] not in (None, value)):
            raise RuntimeError('source changed or SSD copy verification failed')
        temporary.replace(target)
        target.chmod(0o444)
        result = {**item, 'sha256': value}
        write_json(sidecar, result)
        return result

    verified = {}
    with ThreadPoolExecutor(max_workers=spec['workers']) as pool:
        for value in pool.map(copy_one, sources.values()):
            verified[value['source']] = value
            finished += 1
            if finished % 250 == 0 or finished == len(sources):
                progress = {'complete': False, 'verified_files': finished, 'files': len(sources),
                    'seconds': time.monotonic()-started}
                write_json(output/'cache_progress.json', progress)
                print('CACHE ' + json.dumps(progress), flush=True)
    index = {'complete': True, 'originals_unchanged': True, 'manifests': cache_identity(base),
        'storage': storage, 'files': verified, 'aliases': aliases,
        'total_bytes': sum(v['bytes'] for v in verified.values())}
    write_json(Path(spec['index']), index)
    return {'path': spec['index'], 'sha256': sha256(spec['index']), 'files': len(verified),
        'total_bytes': index['total_bytes'], 'seconds': time.monotonic()-started, 'storage': storage}


class CachedFiles:
    def __init__(self, config, base):
        self.root = Path(config['data_cache']['root']).resolve()
        path = Path(config['data_cache']['index'])
        index = json.loads(path.read_text())
        if not index.get('complete') or index['manifests'] != cache_identity(base):
            raise ValueError('SSD cache is not bound to the full immutable dataset')
        self.files, self.aliases = index['files'], index['aliases']
        for value in self.files.values():
            target = Path(value['cached_path'])
            if self.root not in target.resolve().parents or target.stat().st_size != value['bytes']:
                raise ValueError('cached file escaped the SSD root or changed size')
        self.index_sha256 = sha256(path)

    def __call__(self, original):
        return self.files[self.aliases[str(original)]]['cached_path']


def audit_plan(base):
    from virtual_pair_runtime import next_update_batches, target_selection
    streams = [make_stream(base, rank, 2) for rank in range(2)]
    first, samples, frame_visits, updates = streams[0], 0, 0, 0
    rows = base['data']['manifests']['train']['rows']
    for start in range(0, len(first.order), rows):
        part = first.order[start:start+rows]
        if len(np.unique(part)) != len(part) or (len(part) == rows and not np.array_equal(np.sort(part), np.arange(rows))):
            raise ValueError('full-data exposure plan skipped or duplicated an asset within a pass')
    while samples < base['training']['target_assets']:
        consumed = []
        for stream in streams:
            goal, batches = next_update_batches(stream, samples, 2)
            for batch in batches:
                consumed.extend(r.item_index for r in batch)
                frame_visits += sum(r.frames for r in batch)
                if any(len(slots) < 2 or len(set(slots)) != len(slots)
                        for slots in target_selection(base, batch)):
                    raise ValueError('AR supervision selection differs')
            if updates < 2 or samples + goal in base['training']['checkpoint_assets']:
                restored = make_stream(base, stream.rank, 2, stream.state_dict())
                if restored.cursor != stream.cursor or restored.plan_sha256 != stream.plan_sha256:
                    raise ValueError('full sampler restoration differs')
        if consumed != first.order[samples:samples+goal].tolist():
            raise ValueError('two-rank stream does not implement the global exposure plan')
        samples += goal
        updates += 1
    return {'assets': samples, 'updates': updates, 'input_frame_visits': frame_visits,
        'complete_passes': samples // rows, 'partial_pass_assets': samples % rows,
        'unique_assets': int(np.unique(first.order).size), 'plan_sha256': first.plan_sha256,
        'all_full_passes_cover_every_asset_once': True, 'rank_resume_roundtrip': True}
