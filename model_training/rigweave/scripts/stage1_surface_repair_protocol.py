"""Bounded inference-only repair of inconsistent materialized surface inputs."""
import copy
import hashlib
import json
from pathlib import Path

from reference_flow_protocol import sha256

SCOPE = 'stage1_surface_repair_20261029_v1'
ENTRYPOINT = 'model_training/rigweave/scripts/evaluate_stage1_surface_repair.py'
ROOT = Path(__file__).resolve().parents[3]
LABELS = ('donor','stage2_080000','joint_080000',
    *('joint_'+str(n).zfill(6) for n in (10000,20000,30000,40000)),
    *('ar_'+str(n).zfill(6) for n in (10000,20000,30000,40000)))
PAIR_FIELDS = ('raw_frame_index','normalization_sha256','original_GT_sha256',
               'target_mesh_diagonal','gt_joint_count')


def scientific_sha(config):
    fields = {k:config[k] for k in ('family','parent_config','prior_source','prior_output','checkpoints',
        'prior_evaluations','affected_asset_ids','evaluation','capacity','repair','parent_failure')}
    return hashlib.sha256(json.dumps(fields,sort_keys=True).encode()).hexdigest()


def parent_config(config):
    spec = config['parent_config']
    path = ROOT/spec['relative_path']
    if sha256(path) != spec['sha256']:
        raise ValueError('parent inference configuration changed')
    return json.loads(path.read_text())


def validate_config(config, mode):
    if mode not in ('audit','capacity') or config.get('family') != SCOPE or config.get('schema_version') != 1:
        raise ValueError('surface repair permits audit or zero-update evaluation only')
    original = parent_config(config)
    if (config['evaluation'] != original['evaluation']
            or config['repair'] != {'optimizer_updates':0,'recompute_assets_per_checkpoint':18,
                'reuse_assets_per_checkpoint':110,'checkpoint_count':11,'recomputed_views':1584,
                'canonical_surface':'materialize_once_serialize_then_reuse_exact_tensors'}
            or len(config['affected_asset_ids']) != 18
            or len(set(config['affected_asset_ids'])) != 18
            or set(config['checkpoints']) != set(LABELS)
            or set(config['prior_evaluations']) != set(LABELS)
            or config['capacity']['input_frames'] != 24 or config['capacity']['max_new_tokens'] != 2048
            or config['prior_source'] != 'd2dfff503923ff6a4038b7aa50eddc7aba8be4a7'):
        raise ValueError('unreviewed surface repair recipe')
    return original


def plan_inputs(summaries, rows):
    if set(summaries) != set(rows) or set(rows) != set(LABELS):
        raise ValueError('eleven checkpoint identities required')
    keyed = {}
    for label,values in rows.items():
        keyed[label] = {(r['asset_id'],r['frame_index']):r for r in values}
        if len(keyed[label]) != len(values):
            raise ValueError('duplicate asset/frame rows')
    reference = keyed['donor']
    assets = {key[0] for key in reference}
    if len(reference) != 1024 or len(assets) != 128 or set(reference) != {(a,f) for a in assets for f in range(8)}:
        raise ValueError('fixed128x8 coverage differs')
    identity, affected = None,set()
    for label,values in keyed.items():
        if values.keys() != reference.keys():
            raise ValueError('prior coverage differs')
        current = copy.deepcopy(summaries[label]['identity'])
        current.pop('checkpoint_sha256')
        current.pop('checkpoint_layout')
        if identity is None:
            identity = current
        elif identity != current:
            raise ValueError('prior decoding/data/source contract differs')
        for key,row in values.items():
            if any(row[field] != reference[key][field] for field in PAIR_FIELDS):
                raise ValueError('repair is limited to materialized surfaces, not changed frames/GT/normalization')
            if row['surface_sha256'] != reference[key]['surface_sha256']:
                affected.add(key[0])
        for asset in assets:
            if len({values[asset,f]['surface_sha256'] for f in range(8)}) != 1:
                raise ValueError('one asset must bind one complete multi-frame surface')
    return {'affected_asset_ids':sorted(affected),'retained_asset_ids':sorted(assets-affected),
            'input_contract':identity,'all_non_surface_fields_equal':True}


def merge_rows(original, replacements, affected):
    affected = set(affected)
    new = {(r['asset_id'],r['frame_index']):r for r in replacements}
    required = {(r['asset_id'],r['frame_index']) for r in original if r['asset_id'] in affected}
    if len(new) != len(replacements) or set(new) != required:
        raise ValueError('repair must replace all and only affected views')
    result = []
    for row in original:
        key = row['asset_id'],row['frame_index']
        if key in new:
            if any(row[field] != new[key][field] for field in PAIR_FIELDS):
                raise ValueError('canonical repair changed rawframe/GT/normalization')
            result.append(new[key])
        else:
            result.append(row)
    return result
