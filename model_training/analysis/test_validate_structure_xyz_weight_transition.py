from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_structure_xyz_weight_transition import validate_resume_contract


CHECKPOINT_SHA = 'a' * 64
MODEL_SHA = 'b' * 64


@pytest.fixture
def example(tmp_path):
    saved = {'purpose': 'structure_xyz_token_fill_v1',
             'optimization': {'structure_weight': 1., 'surface_lr': 3e-5, 'betas': [.9, .999]},
             'data': {'normalization': 'input_sequence_mesh_bbox', 'root_rt': True},
             'evaluation': {'frames': [0, 2, 4, 6]},
             'architecture': {'width': 1024},
             'execution': {'world_size': 2, 'precision': 'bfloat16_autocast'}}
    target = deepcopy(saved)
    target['optimization']['structure_weight'] = .25
    diagnostic = {'complete': True, 'passed': True, 'optimizer_created': False, 'optimizer_updates': 0,
                  'checkpoint': {'sha256': CHECKPOINT_SHA, 'model_sha256': MODEL_SHA},
                  'model_sha256_unchanged': MODEL_SHA, 'groups': {'shared': {'ratio': .7}}}
    path = tmp_path / 'diagnostic.json'
    path.write_text(json.dumps(diagnostic), encoding='utf-8')
    transition = {'kind': 'structure_weight_only_v1', 'checkpoint_sha256': CHECKPOINT_SHA,
                  'from_weight': 1., 'to_weight': .25, 'primary_accepted': True,
                  'diagnostic': {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}
    return saved, target, transition, diagnostic


def check(example, transition=True, checkpoint=CHECKPOINT_SHA):
    saved, target, declared, _ = example
    return validate_resume_contract(saved, target, declared if transition else None,
                                    checkpoint_sha256=checkpoint)


def repin(example, text=None):
    _, _, transition, diagnostic = example
    path = Path(transition['diagnostic']['path'])
    path.write_text(json.dumps(diagnostic) if text is None else text, encoding='utf-8')
    transition['diagnostic']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_unchanged_contract_needs_no_receipt(example):
    saved, _, _, _ = example
    assert validate_resume_contract(saved, deepcopy(saved), checkpoint_sha256=CHECKPOINT_SHA) is None
    reordered = dict(reversed(list(saved.items())))
    assert validate_resume_contract(saved, reordered, checkpoint_sha256=CHECKPOINT_SHA) is None


def test_legitimate_weight_only_transition_is_read_only(example):
    before = deepcopy(example)
    path = Path(example[2]['diagnostic']['path'])
    contents = path.read_bytes()
    assert check(example) is None
    assert example == before and path.read_bytes() == contents


def test_integer_weights_are_valid_but_other_types_remain_exact(example):
    example[0]['optimization']['structure_weight'] = 1
    assert check(example) is None
    example[1]['optimization']['surface_lr'] = 0
    with pytest.raises(ValueError, match='only optimization.structure_weight'):
        check(example)


def test_changed_weight_without_transition_is_rejected(example):
    with pytest.raises(ValueError, match='requires an approved'):
        check(example, transition=False)


def test_unchanged_weight_with_transition_is_rejected(example):
    example[1]['optimization']['structure_weight'] = 1.
    with pytest.raises(ValueError, match='unchanged loss weight'):
        check(example)


@pytest.mark.parametrize('edit', [
    lambda c: c['optimization'].update(surface_lr=1e-4),
    lambda c: c['optimization']['betas'].__setitem__(0, .95),
    lambda c: c['data'].update(root_rt=False),
    lambda c: c['architecture'].update(width=2048),
    lambda c: c['evaluation']['frames'].append(7),
    lambda c: c['execution'].update(world_size=1),
    lambda c: c.update(extra='new field'),
    lambda c: c.pop('purpose'),
    lambda c: c['optimization'].update(betas=(.9, .999)),
    lambda c: c['architecture'].update(width=1024.),
    lambda c: c['data'].update(root_rt=1),
])
def test_any_other_recursive_contract_change_is_rejected(example, edit):
    edit(example[1])
    with pytest.raises(ValueError, match='only optimization.structure_weight'):
        check(example)


@pytest.mark.parametrize('value', [True, False, 0, -1, float('nan'), float('inf'), -float('inf'), '1', None])
@pytest.mark.parametrize('where', ['saved', 'target', 'from', 'to'])
def test_weights_must_be_positive_finite_numbers_not_booleans(example, value, where):
    if where in ('saved', 'target'):
        example[0 if where == 'saved' else 1]['optimization']['structure_weight'] = value
    else:
        example[2][where + '_weight'] = value
    with pytest.raises(ValueError):
        check(example)


@pytest.mark.parametrize('edit', [
    lambda t: t.update(extra=True), lambda t: t.pop('kind'),
    lambda t: t.update(kind='anything_else'), lambda t: t.update(primary_accepted=1),
    lambda t: t.update(primary_accepted=False), lambda t: t.update(checkpoint_sha256='c' * 64),
    lambda t: t.update(from_weight=2.), lambda t: t.update(to_weight=.5),
    lambda t: t['diagnostic'].update(extra=True), lambda t: t['diagnostic'].pop('sha256'),
    lambda t: t['diagnostic'].update(path='  '), lambda t: t['diagnostic'].update(path=False),
    lambda t: t['diagnostic'].update(sha256='F' * 64),
])
def test_transition_schema_acceptance_and_values_are_strict(example, edit):
    edit(example[2])
    with pytest.raises(ValueError):
        check(example)


@pytest.mark.parametrize('checkpoint', [None, True, '', 'a' * 63, 'A' * 64, 'g' * 64, 'c' * 64])
def test_input_checkpoint_digest_must_be_valid_and_match(example, checkpoint):
    with pytest.raises(ValueError):
        check(example, checkpoint=checkpoint)


@pytest.mark.parametrize('edit', [
    lambda d: d.update(complete=False), lambda d: d.update(complete=1),
    lambda d: d.update(passed=False), lambda d: d.pop('passed'),
    lambda d: d.update(optimizer_created=True), lambda d: d.update(optimizer_created=0),
    lambda d: d.update(optimizer_updates=1), lambda d: d.update(optimizer_updates=False),
    lambda d: d.update(optimizer_updates=0.),
    lambda d: d['checkpoint'].update(sha256='c' * 64),
    lambda d: d['checkpoint'].update(model_sha256='c' * 64),
    lambda d: d['checkpoint'].pop('model_sha256'),
    lambda d: d.update(model_sha256_unchanged='c' * 64),
    lambda d: d.pop('model_sha256_unchanged'), lambda d: d.pop('checkpoint'),
    lambda d: d['groups']['shared'].update(ratio=float('nan')),
])
def test_diagnostic_receipt_must_prove_completed_unchanged_checkpoint(example, edit):
    edit(example[3])
    repin(example)
    with pytest.raises(ValueError):
        check(example)


def test_diagnostic_bytes_must_match_pinned_hash(example):
    Path(example[2]['diagnostic']['path']).write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='diagnostic SHA-256 mismatch'):
        check(example)


def test_missing_diagnostic_is_rejected(example):
    example[2]['diagnostic']['path'] += '.missing'
    with pytest.raises(ValueError, match='cannot read'):
        check(example)


@pytest.mark.parametrize('text', ['not JSON', '[]', '{"passed":false,"passed":true}',
                                 '{"metric":1e9999}'])
def test_invalid_ambiguous_or_overflowed_json_is_rejected(example, text):
    repin(example, text)
    with pytest.raises(ValueError):
        check(example)


def test_nonfinite_unchanged_contract_and_boolean_integer_alias_are_rejected(example):
    saved = example[0]
    target = deepcopy(saved)
    target['data']['root_rt'] = 1
    with pytest.raises(ValueError, match='changed training contract'):
        validate_resume_contract(saved, target, checkpoint_sha256=CHECKPOINT_SHA)
    saved['optimization']['surface_lr'] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        validate_resume_contract(saved, deepcopy(saved), checkpoint_sha256=CHECKPOINT_SHA)
