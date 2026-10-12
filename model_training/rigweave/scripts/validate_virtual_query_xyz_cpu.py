#!/usr/bin/env python3
"""Real-manifest auxiliary codec validation; no CUDA or learned model execution."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--config', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[3]
for name, suffix in [('rigweave', 'rigweave'), ('rigweave.dynamic_rig', 'rigweave/dynamic_rig')]:
    module = types.ModuleType(name)
    module.__path__ = [str(root/'model_training/rigweave/src'/suffix)]
    sys.modules[name] = module
from rigweave.dynamic_rig.aux_skeleton_tokens import AuxSkeletonTokens

config = json.loads(a.config.read_text())
codec = AuxSkeletonTokens(**config['auxiliary_codec'])
report = {'status': 'passed', 'assets': 0, 'max_joints': 0, 'max_tokens': 0, 'splits': {}, 'errors': []}
for split, spec in config['data']['manifests'].items():
    blob = Path(spec['path']).read_bytes()
    assert hashlib.sha256(blob).hexdigest() == spec['sha256']
    rows = [json.loads(line) for line in blob.splitlines()]
    assert len(rows) == spec['rows']
    for row in rows:
        try:
            assert row['root_rt']['version'] == 'retained_root_rigid_v1'
            with np.load(row['path'], allow_pickle=False) as data:
                parents = data['target_parents']
            encoded = codec.encode(parents)
            decoded = codec.decode(encoded.tokens)
            expected = np.asarray([-1 if parents[i] < 0 else encoded.source_to_dfs[parents[i]]
                                   for i in encoded.dfs_order], dtype=np.int64)
            assert np.array_equal(decoded, expected)
            report['max_joints'] = max(report['max_joints'], len(parents))
            report['max_tokens'] = max(report['max_tokens'], len(encoded.tokens))
            report['assets'] += 1
        except Exception as exc:
            report['errors'].append({'path': row['path'], 'error': repr(exc)})
        if (report['assets']+len(report['errors'])) % 2000 == 0:
            print('CODEC', report['assets'], 'errors', len(report['errors']), flush=True)
    report['splits'][split] = len(rows)
report['status'] = 'passed' if not report['errors'] else 'failed'
report['manifest_sha256'] = {s: v['sha256'] for s,v in config['data']['manifests'].items()}
a.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report), flush=True)
raise SystemExit(bool(report['errors']))
