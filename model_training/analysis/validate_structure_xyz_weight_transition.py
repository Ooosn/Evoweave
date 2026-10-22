"""Validate an explicitly approved loss-weight-only checkpoint continuation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re


_WEIGHT_PATH = ('optimization', 'structure_weight')
_TRANSITION_FIELDS = {'kind', 'checkpoint_sha256', 'from_weight', 'to_weight',
                      'diagnostic', 'primary_accepted'}


def _sha256(value, name):
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError(f'{name} must be a lowercase SHA-256 digest')
    return value


def _weight(value, name):
    if type(value) not in (int, float):
        raise ValueError(f'{name} must be a positive finite number, not a boolean')
    try:
        valid = math.isfinite(value) and value > 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f'{name} must be a positive finite number')
    return value


def _validate_tree(value):
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError('training contract keys must be strings')
        for item in value.values():
            _validate_tree(item)
    elif type(value) in (list, tuple):
        for item in value:
            _validate_tree(item)
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError('training contract contains a nonfinite value')
    elif value is not None and type(value) not in (bool, int, str):
        raise ValueError('training contract contains an unsupported value type')


def _equal(left, right, *, ignore_weight=False, path=()):
    if ignore_weight and path == _WEIGHT_PATH:
        return True
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return (left.keys() == right.keys()
                and all(_equal(left[key], right[key], ignore_weight=ignore_weight, path=path + (key,))
                        for key in left))
    if isinstance(left, (list, tuple)):
        return (len(left) == len(right)
                and all(_equal(a, b, ignore_weight=ignore_weight, path=path + (index,))
                        for index, (a, b) in enumerate(zip(left, right))))
    return left == right


def _contract_weight(contract, name):
    if type(contract) is not dict or type(contract.get('optimization')) is not dict:
        raise ValueError(f'{name} must contain an optimization mapping')
    _validate_tree(contract)
    return _weight(contract['optimization'].get('structure_weight'), f'{name}.optimization.structure_weight')


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'diagnostic JSON contains duplicate key {key!r}')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f'diagnostic JSON contains nonfinite value {value}')


def _read_diagnostic(spec):
    if type(spec) is not dict or set(spec) != {'path', 'sha256'}:
        raise ValueError('diagnostic must contain exactly path and sha256')
    if not isinstance(spec['path'], str) or not spec['path'].strip():
        raise ValueError('diagnostic path must be a nonempty string')
    expected = _sha256(spec['sha256'], 'diagnostic.sha256')
    try:
        payload = Path(spec['path']).read_bytes()
    except OSError as exc:
        raise ValueError('cannot read the pinned diagnostic JSON') from exc
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError('diagnostic SHA-256 mismatch')
    try:
        result = json.loads(payload, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('diagnostic is not valid JSON') from exc
    if type(result) is not dict:
        raise ValueError('diagnostic JSON must be an object')
    _validate_tree(result)
    return result


def validate_resume_contract(saved_training_contract, target_training_contract, transition=None,
                             *, checkpoint_sha256):
    """Return None when valid; raise ValueError without modifying either contract.

    The caller verifies checkpoint bytes. A transition authorizes only the weight
    values named here; the diagnostic proves a completed, unchanged-model probe.
    """
    checkpoint_sha256 = _sha256(checkpoint_sha256, 'checkpoint_sha256')
    old = _contract_weight(saved_training_contract, 'saved_training_contract')
    new = _contract_weight(target_training_contract, 'target_training_contract')
    if transition is None:
        if not _equal(saved_training_contract, target_training_contract):
            raise ValueError('changed training contract requires an approved weight-only transition')
        return None
    if type(transition) is not dict or set(transition) != _TRANSITION_FIELDS:
        raise ValueError('transition has missing or extra fields')
    if transition['kind'] != 'structure_weight_only_v1' or transition['primary_accepted'] is not True:
        raise ValueError('transition kind or explicit primary acceptance differs')
    if _sha256(transition['checkpoint_sha256'], 'transition.checkpoint_sha256') != checkpoint_sha256:
        raise ValueError('transition checkpoint SHA-256 mismatch')
    before = _weight(transition['from_weight'], 'transition.from_weight')
    after = _weight(transition['to_weight'], 'transition.to_weight')
    if old == new:
        raise ValueError('unchanged loss weight must not declare a transition')
    if not _equal(saved_training_contract, target_training_contract, ignore_weight=True):
        raise ValueError('only optimization.structure_weight may change')
    diagnostic = _read_diagnostic(transition['diagnostic'])
    if diagnostic.get('complete') is not True or diagnostic.get('passed') is not True:
        raise ValueError('diagnostic must be complete and passed')
    if (diagnostic.get('optimizer_created') is not False
            or type(diagnostic.get('optimizer_updates')) is not int
            or diagnostic['optimizer_updates'] != 0):
        raise ValueError('diagnostic must not create or update an optimizer')
    checkpoint = diagnostic.get('checkpoint')
    if type(checkpoint) is not dict or checkpoint.get('sha256') != checkpoint_sha256:
        raise ValueError('diagnostic checkpoint SHA-256 mismatch')
    model_hash = _sha256(checkpoint.get('model_sha256'), 'diagnostic.checkpoint.model_sha256')
    if diagnostic.get('model_sha256_unchanged') != model_hash:
        raise ValueError('diagnostic model fingerprint changed')
    if before != old or after != new:
        raise ValueError('transition from_weight/to_weight do not match the saved/target contracts')
    return None
