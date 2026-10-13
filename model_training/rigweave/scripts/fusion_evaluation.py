"""Free auxiliary generation and matched-input checks for fusion experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch


def tree_fingerprint(value):
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            visit(['tensor', str(item.dtype), list(item.shape)])
            digest.update(item.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
        elif isinstance(item, np.ndarray):
            visit(['array', str(item.dtype), list(item.shape)])
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            digest.update(b'{')
            for key in sorted(item):
                visit(key)
                visit(item[key])
            digest.update(b'}')
        elif isinstance(item, (list, tuple)):
            digest.update(b'[')
            for child in item:
                visit(child)
            digest.update(b']')
        else:
            digest.update(json.dumps(item, sort_keys=True, allow_nan=False).encode() + b'\0')

    visit(value)
    return digest.hexdigest()


def auxiliary_record(prediction, codec, source_parents, *, asset_id):
    target_tokens = codec.encode(source_parents).tokens.tolist()
    target_parents = codec.decode(target_tokens).tolist()
    record = {'asset_id': asset_id, 'status': prediction['status'],
              'natural_eos': bool(prediction['success']), 'success': False,
              'generated_tokens': prediction['tokens'], 'target_tokens': target_tokens,
              'target_parents': target_parents, 'exact_count': False,
              'exact_parents': False, 'exact_tokens': prediction['tokens'] == target_tokens}
    if prediction['success']:
        try:
            parents = codec.decode(prediction['tokens']).tolist()
            record.update(success=True, predicted_parents=parents,
                          exact_count=len(parents) == len(target_parents),
                          exact_parents=parents == target_parents)
        except ValueError as exc:
            record.update(status='invalid_eos_sequence', error=str(exc))
    return record


def auxiliary_summary(rows):
    if not rows:
        raise ValueError('auxiliary evaluation needs at least one asset')
    return {key: sum(bool(row[key]) for row in rows) / len(rows)
            for key in ('natural_eos', 'success', 'exact_count', 'exact_parents', 'exact_tokens')}


def comparison_contract(config):
    contract = json.loads(json.dumps(config))
    contract['architecture'].pop('fusion_mode', None)
    contract['runtime'].pop('physical_gpu', None)
    contract['probe'].pop('paired_output', None)
    return tree_fingerprint(contract)


def pair_ready(output, peer_output, initial, items, config, *, timeout=600):
    from virtual_query_xyz_runtime import write_json

    output, peer_output = Path(output), Path(peer_output)
    if output.resolve() == peer_output.resolve():
        raise ValueError('a paired trial cannot compare itself')
    own = {'full_initial_state_sha256': initial['full_initial_state_sha256'],
           'inputs_sha256': tree_fingerprint(items),
           'comparison_contract_sha256': comparison_contract(config),
           'code_commit': initial['code_commit'],
           'assets': [item[2]['asset_id'][0] for item in items]}
    write_json(output / 'pair_ready.json', own)
    deadline = time.monotonic() + timeout
    while not (peer_output / 'pair_ready.json').is_file():
        if (peer_output / 'result.json').is_file():
            raise RuntimeError('paired trial finished before declaring matched inputs')
        if time.monotonic() >= deadline:
            raise TimeoutError('paired initialization/input check timed out')
        time.sleep(2)
    peer = json.loads((peer_output / 'pair_ready.json').read_text())
    if own != peer:
        raise RuntimeError('paired trials have different initial weights or actual inputs')
    write_json(output / 'pair_verified.json', {'passed': True, 'peer': str(peer_output), **own})
    return own
