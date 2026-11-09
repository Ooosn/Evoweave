"""Prepare and score a paired sample from the published Articulation-XL2.0 test split."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import re
import runpy
import struct
import sys
from urllib.parse import unquote, urlsplit
from zipfile import ZipFile

from review_skintokens_stage1_20261109 import digest


SCOPE = 'skintokens_original_test_20261109'
ROOT = Path('/work/wangyy') / SCOPE
UPSTREAM = ROOT / 'upstream'
DATA = ROOT / 'data'
PANEL = ROOT / 'panel'
RELEASE = UPSTREAM / 'experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt'
VAE = UPSTREAM / 'experiments/skin_vae_2_10_32768/last.ckpt'
FINE = Path('/work/wangyy/tokenrig_alignment_20261105/outputs_skintokens_static_5k_20261109/train/checkpoint_030000.pt')
OVERLAY = ROOT / 'skintokens_static_030000_full_vocabulary.ckpt'
ARTICULATION_SHA = '62d16254b1ed33e0f5e39d474d9036d7045dceeb78f998c619651c205ede8406'
METADATA_SHA = 'f5174ce23e319c9663e0d0ecf8ec8854d37497020ade7ac344d9b4ed13323a75'
RELEASE_SHA = 'f4e4706a11cfb520cdde65156a0358545e4fbf8f36237aca01ea5e79d5cb5692'
VAE_SHA = '4843f49e58afff88345806b94ca82e6cc9d8def6e7432e2853c677b154de0ed4'
FINE_SHA = '58a999bde6669cb43ced784dc679f9a99675230ace7a27bd5ceeb33f3a378b59'
TRAIN_SHA = 'a17b6be96fb33a32a1bee7117b07789da6261c6b5693104f3d2cce07c09bab88'
UPSTREAM_COMMIT = '273b691d35989d71cd17ff2895fdc735097b92d1'
DATA_REV = '11f777d83a1e7ee7c5a22201da4e140cafac1837'
QWEN_CONFIG_SHA = '660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd'
SEED = 20261109
CASES = 12
SKELETON_ROWS = 267
ORIGINS = [Path('/home/wangyy/evorig') / dataset / 'raw/download_manifest.jsonl'
           for dataset in ('evoweave_objaverse_xl', 'evoweave_objaverse_xl_more_anim')]


def write_new(path, data):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write('\n')


def require_digest(path, sha):
    if digest(path) != sha:
        raise ValueError('digest mismatch: ' + str(path))


def source_key(value):
    url = urlsplit(value.strip())
    path = unquote(url.path).strip('/')
    fields = path.split('/')
    host = (url.hostname or '').lower()
    if host in ('www.sketchfab.com', 'sketchfab.com'):
        token = re.search(r'[a-fA-F0-9]{32}(?=$|/)', path)
        if token:
            return ('sketchfab', token[0].lower())
    if host == 'github.com' and len(fields) >= 5 and fields[2] in ('blob', 'raw'):
        return ('github', fields[0].lower(), fields[1].lower(), '/'.join(fields[4:]))
    if host == 'raw.githubusercontent.com' and len(fields) >= 4:
        return ('github', fields[0].lower(), fields[1].lower(), '/'.join(fields[3:]))
    return (host, path) if host and path else None


def training_sources(manifest):
    require_digest(manifest, TRAIN_SHA)
    ids = {json.loads(line)['asset_id'] for line in manifest.read_text().splitlines()}
    if len(ids) != 5000:
        raise ValueError('expected the fixed 5000 training asset identities')
    objaverse = {uid for uid in ids if uid.startswith('oxl_')}
    origins = {}
    for path in ORIGINS:
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                uid = row['asset_id']
                key = source_key(row.get('fileIdentifier') or row.get('download_url') or '')
                if uid in objaverse and key:
                    origins.setdefault(uid, set()).add(key)
    if set(origins) != objaverse:
        raise ValueError(f'{len(objaverse-set(origins))} training assets lack original source identity')
    return ids, set().union(*origins.values()) | {('sketchfab', uid) for uid in ids - objaverse}


class DatasetUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        import numpy as np
        if module == 'numpy' and name in ('ndarray', 'dtype'):
            return getattr(np, name)
        if module in ('numpy.core.multiarray', 'numpy._core.multiarray') and name in ('_reconstruct', 'scalar'):
            return getattr(np.core.multiarray, name)
        raise pickle.UnpicklingError(f'unexpected dataset object {module}.{name}')


def read_test(path):
    import numpy as np
    with ZipFile(path) as archive, archive.open('arr_0.npy') as handle:
        version = np.lib.format.read_magic(handle)
        readers = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}
        shape, _, dtype = readers[version](handle)
        if len(shape) != 1 or dtype.kind != 'O':
            raise ValueError('unexpected published test archive format')
        rows = DatasetUnpickler(handle).load()
        if not isinstance(rows, np.ndarray) or rows.shape != shape or rows.dtype != dtype or handle.read(1):
            raise ValueError('test archive did not decode to the declared rows')
        return rows


def validate_mesh_rig(row):
    import numpy as np
    v, f, j, b = (np.asarray(row[k]) for k in ('vertices', 'faces', 'joints', 'bones'))
    if (v.ndim != 2 or v.shape[1] != 3 or len(v) < 3 or f.ndim != 2 or f.shape[1] != 3
            or not len(f) or f.dtype.kind not in 'iu' or f.min() < 0 or f.max() >= len(v)
            or j.ndim != 2 or j.shape[1] != 3 or len(j) < 2 or b.ndim != 2 or b.shape[1] != 2
            or not len(b) or b.dtype.kind not in 'iu' or b.min() < 0 or b.max() >= len(j)
            or not np.isfinite(v).all() or not np.isfinite(j).all() or not np.ptp(v, axis=0).max() > 0):
        raise ValueError('malformed mesh or GT skeleton')
    return v, f, j, b


def prepare_panel(manifest):
    import numpy as np
    import trimesh
    source, meta = DATA/'articulation_xlv2_test.npz', DATA/'meta_Articulation_XL_2.0.csv'
    require_digest(source, ARTICULATION_SHA)
    require_digest(meta, METADATA_SHA)
    train_ids, train_sources = training_sources(manifest)
    with meta.open(encoding='utf-8-sig', newline='') as handle:
        sources = {r['uuid']: r['fileIdentifier'] for r in csv.DictReader(handle)}
    rows = read_test(source)
    ordered, excluded, invalid = [], 0, 0
    seen = set()
    for row in rows:
        uid = str(row['uuid'])
        if not re.fullmatch(r'[A-Za-z0-9_-]{5,128}', uid) or uid in seen or uid not in sources:
            raise ValueError('test UUID is unsafe, duplicated, or not in published metadata')
        seen.add(uid)
        key = source_key(sources[uid])
        if key is None:
            raise ValueError('cannot trace test UUID to a published source')
        if uid.replace('-', '').lower() in train_ids or key in train_sources:
            excluded += 1
            continue
        try:
            validate_mesh_rig(row)
        except ValueError:
            invalid += 1
            continue
        ordered.append((hashlib.sha256(f'{SEED}:{uid}'.encode()).hexdigest(), uid, row))
    if len(ordered) < CASES:
        raise ValueError('not enough traced eligible test assets')
    PANEL.mkdir()
    cases = []
    for i, (_, uid, row) in enumerate(sorted(ordered, key=lambda x:x[:2])[:CASES], 1):
        v, f, j, b = validate_mesh_rig(row)
        target = PANEL/f'{i:02d}_{uid}'
        mesh = target.with_suffix('.glb')
        truth = target.with_suffix('.npz')
        mesh.write_bytes(trimesh.Trimesh(vertices=v, faces=f, process=False).export(file_type='glb'))
        if glb_json(mesh).get('skins'):
            raise ValueError('test inputs must not contain any rig or skin')
        np.savez(truth, joints=j, bones=b, mesh_min=v.min(axis=0), mesh_max=v.max(axis=0))
        cases.append({'id': uid, 'ordinal': i, 'seed': SEED+i, 'mesh': str(mesh),
                      'mesh_sha256': digest(mesh), 'truth': str(truth), 'truth_sha256': digest(truth),
                      'source_identifier': sources[uid]})
    report = {'dataset': 'Seed3D/Articulation-XL2.0', 'revision': DATA_REV, 'split': 'main test',
              'archive_sha256': ARTICULATION_SHA, 'metadata_sha256': METADATA_SHA, 'split_rows': len(rows),
              'fine_tuning_manifest_sha256': TRAIN_SHA, 'traced_train_assets': len(train_ids),
              'training_source_overlap_excluded': excluded, 'invalid_geometry_excluded': invalid,
              'selection': 'ascending SHA256 of fixed seed and UUID before inference', 'cases': cases}
    write_new(PANEL/'manifest.json', report)
    return report


def overlay_state(base, fine):
    import torch
    projection = {'decoder_condition_projection.weight': 'output_proj.0.weight',
                  'decoder_condition_projection.bias': 'output_proj.0.bias',
                  'decoder_condition_projection.normalization.weight': 'output_proj.1.weight'}
    vocab = {'transformer.model.embed_tokens.weight', 'transformer.lm_head.weight'}
    expected = {key for key in base if key.startswith(('mesh_encoder.', 'output_proj.', 'transformer.'))}
    updates = {}
    for key, value in fine.items():
        name = projection.get(key, key.replace('xyz_decoder.', 'transformer.', 1))
        if name not in expected or name in updates or not torch.isfinite(value).all():
            raise ValueError('unexpected fine-tuned tensor ' + key)
        original = base[name]
        if name in vocab:
            if original.shape != (33036, 896) or value.shape != (SKELETON_ROWS, 896):
                raise ValueError('full and skeleton-only vocabularies do not align')
            table = original.to(value.dtype).clone()
            table[:SKELETON_ROWS].copy_(value)
            if not torch.equal(table[SKELETON_ROWS:].to(original.dtype), original[SKELETON_ROWS:]):
                raise ValueError('unchanged skin vocabulary rows changed')
            updates[name] = table
        else:
            if original.shape != value.shape:
                raise ValueError('shape changed for ' + name)
            updates[name] = value
    if set(updates) != expected or not vocab <= expected:
        raise ValueError('fine-tuned state did not cover the full skeleton branch')
    names = sorted(vocab)
    if not torch.equal(base[names[0]], base[names[1]]) or not torch.equal(updates[names[0]], updates[names[1]]):
        raise ValueError('released and fine-tuned tied tables must remain identical')
    return {**base, **updates}, sorted(updates)


def prepare_weights():
    import torch
    for path, sha in ((RELEASE, RELEASE_SHA), (FINE, FINE_SHA), (VAE, VAE_SHA),
                      (UPSTREAM/'models/Qwen3-0.6B/config.json', QWEN_CONFIG_SHA)):
        require_digest(path, sha)
    source = torch.load(RELEASE, map_location='cpu', mmap=True, weights_only=False)
    tuned = torch.load(FINE, map_location='cpu', mmap=True, weights_only=False)
    if ((tuned['samples'], tuned['step'], tuned['stage2_enabled']) != (30000, 627, False)
            or tuned['configuration']['tokenrig_decoder']['checkpoint_sha256'] != RELEASE_SHA
            or tuned['configuration']['train_manifest']['sha256'] != TRAIN_SHA):
        raise ValueError('not the accepted single-pose 30k checkpoint')
    state, names = overlay_state(source['state_dict'], tuned['model'])
    untouched = set(source['state_dict']) - set(names)
    if not untouched or any(not k.startswith('vae.') or state[k] is not source['state_dict'][k] for k in untouched):
        raise ValueError('the unchanged full rig tensors must be exactly the released VAE')
    if OVERLAY.exists():
        raise FileExistsError(OVERLAY)
    with OVERLAY.open('xb') as handle:
        torch.save({**source, 'state_dict': state}, handle)
    restored = torch.load(OVERLAY, map_location='cpu', mmap=True, weights_only=False)
    if (restored['hyper_parameters'] != source['hyper_parameters']
            or restored['state_dict'].keys() != state.keys()
            or any(not torch.equal(value, restored['state_dict'][name]) for name,value in state.items())):
        raise ValueError('full checkpoint overlay did not round trip')
    report = {'release_sha256': RELEASE_SHA, 'fine_tune_sha256': FINE_SHA,
              'overlay_path': str(OVERLAY), 'overlay_sha256': digest(OVERLAY), 'vae_sha256': VAE_SHA,
              'replaced_skeleton_tensors': len(names), 'unchanged_VAE_tensors': len(untouched),
              'skeleton_vocab_rows_replaced': SKELETON_ROWS, 'full_vocab_rows': 33036}
    write_new(ROOT/'weights_review.json', report)
    return report


def glb_json(path):
    with path.open('rb') as f:
        magic, version, total = struct.unpack('<4sII', f.read(12))
        size, kind = struct.unpack('<I4s', f.read(8))
        if magic != b'glTF' or version != 2 or total != path.stat().st_size or kind != b'JSON':
            raise ValueError('not a valid GLB v2: '+str(path))
        return json.loads(f.read(size))


def glb_skeleton(path):
    import numpy as np
    from scipy.spatial.transform import Rotation
    gltf = glb_json(path)
    nodes = gltf['nodes']
    joints = sorted({int(j) for skin in gltf.get('skins', []) for j in skin['joints']})
    if len(joints) < 2:
        raise ValueError('export did not contain a usable joint hierarchy')
    parents = {}
    for i, node in enumerate(nodes):
        for child in node.get('children', []):
            if child in parents:
                raise ValueError('GLB node has multiple parents')
            parents[child] = i
    matrices = {}

    def matrix(i):
        if i not in matrices:
            node = nodes[i]
            if 'matrix' in node:
                transform = np.array(node['matrix'], np.float64).reshape(4, 4).T
            else:
                transform = np.eye(4)
                transform[:3, :3] = Rotation.from_quat(node.get('rotation', [0,0,0,1])).as_matrix() * node.get('scale', [1,1,1])
                transform[:3, 3] = node.get('translation', [0,0,0])
            matrices[i] = matrix(parents[i]) @ transform if i in parents else transform
        return matrices[i]

    ids = {j:i for i,j in enumerate(joints)}
    edges = []
    for j in joints:
        p = parents.get(j)
        while p is not None and p not in ids:
            p = parents.get(p)
        if p is not None:
            edges.append((ids[p], ids[j]))
    result = np.stack([matrix(j)[:3,3] for j in joints])
    if not edges or not np.isfinite(result).all():
        raise ValueError('exported skeleton is empty or nonfinite')
    return result, np.array(edges)


def distances(a, ea, b, eb, span):
    import numpy as np
    from scipy.spatial import cKDTree
    # RigNet-style symmetric Chamfer; uniformly sample each bone at 100 positions.
    def bones(joints, edges):
        start, end = joints[edges[:,0]], joints[edges[:,1]]
        mix = np.arange(100, dtype=np.float64)[:, None, None] / 100.
        return (mix*start + (1.-mix)*end).reshape(-1,3)
    def near(x, y):
        return float(cKDTree(y).query(x)[0].mean())
    x, y = bones(a, ea), bones(b, eb)
    factor = 50. / span
    return {'J2J_percent': factor*(near(a,b)+near(b,a)), 'J2B_percent': factor*(near(a,y)+near(b,x)),
            'B2B_percent': factor*(near(x,y)+near(y,x))}


def score_case(case, path):
    import numpy as np
    import trimesh
    require_digest(Path(case['truth']), case['truth_sha256'])
    with np.load(case['truth'], allow_pickle=False) as gt:
        j, b = glb_skeleton(path)
        span = float((gt['mesh_max'] - gt['mesh_min']).max())
        expected = np.stack((gt['mesh_min'], gt['mesh_max']))
        actual = trimesh.load_scene(path, process=False).bounds
        if actual is None or not np.allclose(actual, expected, rtol=0, atol=2e-5*span):
            raise ValueError('official export did not preserve the test mesh coordinate frame')
        return {'joints': len(j), 'bones': len(b), 'gt_joints': len(gt['joints']),
                **distances(j, b, gt['joints'], gt['bones'], span)}


def seeded_demo(seed, argv):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    sys.argv = ['demo.py', *argv]
    sys.path.insert(0, str(UPSTREAM))
    os.chdir(UPSTREAM)
    runpy.run_path('demo.py', run_name='__main__')


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == 'demo':
        seeded_demo(int(sys.argv[2]), sys.argv[3:])
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('mode', choices=('prepare',))
        parser.add_argument('--train-manifest', type=Path, required=True)
        args = parser.parse_args()
        if PANEL.exists() or OVERLAY.exists():
            raise RuntimeError('refuse to replace the fixed panel or derived checkpoint')
        report = prepare_panel(args.train_manifest)
        print(json.dumps({k:v for k,v in report.items() if k != 'cases'}), flush=True)
        print(json.dumps(prepare_weights()), flush=True)
