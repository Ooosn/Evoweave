"""Materialize validation geometry once; evaluators replay the saved tensors."""
import hashlib
import json
from pathlib import Path
import tarfile

import torch

from fusion_evaluation import tree_fingerprint
from reference_xyz_protocol import sha256
from virtual_query_xyz_runtime import write_json
from rigweave.dynamic_rig.virtual_frame_data import prepare_virtual_surface


def input_identity(row, base, mode, frames):
    return {**{key: row[key] for key in ('asset_id', 'path', 'root_rt')},
            'data': base['data'], 'surface_sampling': base['surface_sampling'],
            'mode': mode, 'frames': frames}


def _geometry_identity(identity):
    # These fields select the training stream or skeleton tokens, not evaluation geometry.
    training_only = {'manifests', 'stream_seed', 'frames_min', 'frames_max', 'max_main_tokens'}
    return {**identity, 'data': {k: v for k, v in identity['data'].items() if k not in training_only}}


def _check_partition(record, mode, frames):
    raw, heldout = set(record['raw_frames']), set(record['heldout_raw_ids'])
    if (len(raw) != frames or record['frame_partition'] != mode
            or (mode == 'heldout_eval' and not raw <= heldout)
            or (mode != 'heldout_eval' and raw & heldout)):
        raise RuntimeError('canonical evaluation frame partition differs')


def read_record(spec, *, verify=False):
    path = Path(spec['path'])
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f'missing frozen evaluation input: {path}')
    if verify and sha256(path) != spec['sha256']:
        raise RuntimeError(f'frozen evaluation file digest differs: {path}')
    record = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    identity = spec['identity']
    if record['asset_id'] != identity['asset_id']:
        raise RuntimeError(f'frozen evaluation asset differs: {path}')
    _check_partition(record, identity['mode'], identity['frames'])
    if verify and tree_fingerprint(record['surface']) != record['surface_sha256']:
        raise RuntimeError(f'frozen evaluation surface tensor digest differs: {path}')
    return record


def _diagonals(batch):
    vertices = batch['frame_vertices'][0]
    return torch.linalg.vector_norm(vertices.amax(1) - vertices.amin(1), dim=-1).cpu()


def verify_source(record, batch):
    """Audit the source frames, normalization and labels without resampling surfaces."""
    count = int(batch['joint_mask'][0].sum())
    if (record['asset_id'] != batch['asset_id']
            or record['raw_frames'] != batch['selected_frames'][0].tolist()
            or record['heldout_raw_ids'] != batch['heldout_raw_ids']
            or record['normalization_sha256'] != tree_fingerprint(
                {k: batch[k] for k in ('query_center', 'query_scale')})
            or not torch.equal(record['truth'], batch['frame_target_joints'][0, :, :count])
            or not torch.equal(record['parents'], batch['target_parents'][0, :count])
            or not torch.equal(record['diagonals'], _diagonals(batch))):
        raise RuntimeError(f"frozen evaluation source differs for {batch['asset_id']}")


def panel_digest(specs):
    entries = sorted((s['identity']['mode'], s['identity']['asset_id'], s['sha256']) for s in specs)
    return hashlib.sha256(json.dumps(entries, separators=(',', ':')).encode()).hexdigest()


def archived_panel(archive, selected, base, directory, *, mode='seen_eval', frames=8,
                   verify_archive=False):
    """Bind a retained packed panel to its SSD files; never rebuild a missing entry."""
    if verify_archive and sha256(archive['path']) != archive['sha256']:
        raise RuntimeError('retained evaluation archive digest differs')
    specs = {}
    with tarfile.open(archive['path'], 'r:') as tar:
        for member in tar:
            if not member.name.endswith('.json'):
                continue
            if not member.isfile():
                raise RuntimeError('evaluation archive metadata must be regular files')
            spec = json.load(tar.extractfile(member))
            asset_id = spec['identity']['asset_id']
            if asset_id in specs:
                raise RuntimeError(f'duplicate evaluation asset: {asset_id}')
            specs[asset_id] = spec
    if len(specs) != len(selected) or set(specs) != {row['asset_id'] for row in selected}:
        raise RuntimeError('retained archive membership differs from the requested evaluation panel')
    panel = []
    for row in selected:
        spec = specs[row['asset_id']]
        identity = input_identity(row, base, mode, frames)
        if _geometry_identity(spec['identity']) != _geometry_identity(identity):
            raise RuntimeError(f"retained evaluation geometry differs for {row['asset_id']}")
        path = Path(directory) / Path(spec['path']).name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f'missing retained evaluation input: {path}')
        panel.append({**spec, 'path': str(path)})
    return panel


def canonical_input(dataset, index, base, config, mode, device, *, frames=8,
                    directory=None, require_existing=False, verify=False):
    if 'native_frame_control' in config:
        from tokenrig_native_frame_control import cache_base
        base = cache_base(base)
    identity = input_identity(dataset.entries[index], base, mode, frames)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = Path(config['data_cache']['root'])/'evaluation_inputs' if directory is None else Path(directory)
    path = directory/(key+'.pt')
    sidecar = path.with_suffix('.json')
    if path.exists() or sidecar.exists():
        if path.is_symlink() or sidecar.is_symlink() or not (path.is_file() and sidecar.is_file()):
            raise RuntimeError(f'incomplete frozen evaluation input: {path}')
        spec = json.loads(sidecar.read_text())
        if spec['identity'] != identity or spec['path'] != str(path):
            raise RuntimeError('frozen evaluation input metadata changed')
        return read_record(spec, verify=verify), spec
    if require_existing or 'native_continuation' in config or 'native_frame_control' in config:
        raise RuntimeError(f'missing frozen evaluation input; do not resample: {path}')

    batch = dataset[index]
    surface, geometry = prepare_virtual_surface([batch], device, **base['surface_sampling'])
    count = int(geometry['joint_mask'][0].sum())
    surface = {k: v.detach().cpu() for k, v in surface.items()}
    record = {'asset_id': batch['asset_id'], 'surface': surface,
        'truth': geometry['frame_target_joints'][0, :, :count].cpu(),
        'parents': geometry['target_parents'][0, :count].cpu(),
        'raw_frames': geometry['selected_frames'][0].tolist(), 'diagonals': _diagonals(batch),
        'normalization_sha256': tree_fingerprint({k: geometry[k] for k in ('query_center', 'query_scale')}),
        'surface_sha256': tree_fingerprint(surface), 'heldout_raw_ids': batch['heldout_raw_ids'],
        'frame_partition': mode}
    _check_partition(record, mode, frames)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pt.tmp')
    with temporary.open('xb') as handle:
        torch.save(record, handle)
    temporary.replace(path)
    path.chmod(0o444)
    spec = {'path': str(path), 'sha256': sha256(path), 'identity': identity}
    write_json(sidecar, spec)
    return record, spec
