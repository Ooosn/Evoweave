"""Reuse the final full-model evaluation's exact inputs for fixed visual cases."""
import hashlib
import json
from pathlib import Path

import torch

from anchor_full_evaluation import canonical_input
from predicted_reference_protocol import bound_config, sha256
from reference_flow_runtime import read_all
from stage1_surface_repair_runtime import validate_canonical
from virtual_pair_runtime import evaluation_rows
from virtual_query_xyz_runtime import fixed_dataset


def load_records(config, rank=None):
    specs = config['full_model_inputs']
    for spec in specs.values():
        if sha256(spec['path']) != spec['sha256']:
            raise ValueError('completed full-model input provenance changed')
    result = json.loads(Path(specs['result']['path']).read_text())
    resource = json.loads(Path(specs['resources']['path']).read_text())
    summary = json.loads(Path(specs['summary']['path']).read_text())
    rows = json.loads(Path(specs['rows']['path']).read_text())
    if (not result['passed'] or not result['complete'] or result['samples'] != 300000
            or not resource['passed'] or resource['exit_code'] != 0
            or not resource['payload_cleanup_complete'] or not summary['complete']
            or summary['samples'] != 300000 or len(rows) != 1024):
        raise ValueError('full300k producer did not complete as recorded')
    original = bound_config(config['parent_config'])
    if original['family'] != 'anchor_full_20261030_v1':
        raise ValueError('visual inputs require the full300k configuration')
    base, entries = read_all(original)
    panel = evaluation_rows(base, entries)
    dataset = fixed_dataset(panel, base, 8, mode='seen_eval')
    records = []
    for index in config['official_smoke']['case_indices']:
        if rank is not None and index % 2 != rank:
            continue
        row = panel[index]
        identity = {'asset_id': row['asset_id'], 'path': row['path'], 'root_rt': row['root_rt'],
            'data': base['data'], 'surface_sampling': base['surface_sampling'], 'mode': 'seen_eval', 'frames': 8}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = Path(original['data_cache']['root'])/'evaluation_inputs'/(key+'.pt')
        if not path.is_file() or not path.with_suffix('.json').is_file():
            raise ValueError('retained canonical input is missing; do not regenerate it')
        record, _ = canonical_input(dataset, index, base, original, 'seen_eval', torch.device('cpu'))
        validate_canonical(record, rows)
        records.append(dict(record, panel_index=index))
    expected = [i for i in config['official_smoke']['case_indices'] if rank is None or i % 2 == rank]
    if [r['panel_index'] for r in records] != expected:
        raise ValueError('fixed full-model visual case coverage differs')
    return records
