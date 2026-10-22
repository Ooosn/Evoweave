from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'model_training/rigweave/src'))
sys.path.insert(0, str(ROOT / 'model_training/rigweave/scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import structure_xyz_fill_evaluation as evaluation
import structure_xyz_fill_runtime as runtime
import structure_xyz_large_evaluation as large


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def reference_row(asset, group, frame):
    index = int(asset[1:])
    return {'group': group, 'asset_id': asset, 'frame_index': frame,
            'raw_frame_index': frame + (8 if group == 'train_heldout' else 0),
            'target_tokens': [100 + index, 10 + frame], 'original_edge_f1': .4,
            'original_target_joints': [[float(index), float(frame), 0.]],
            'original_target_parents': [-1]}


@pytest.fixture
def case(tmp_path):
    entries = {split: [{'asset_id': f'{prefix}{index:03d}', 'path': f'/fixture/{prefix}{index:03d}.npz',
                       'root_rt': {'version': 'retained_root_rigid_v1', 'offset': index}}
                      for index in range(count)]
               for split, prefix, count in (('train', 't', 32), ('valid', 'v', 128))}
    selection = {'train_seen': entries['train'], 'train_heldout': entries['train'], 'valid': entries['valid']}
    files = {}
    for name, payload in (('selection', selection), ('summary', {'complete': True, 'samples': 65000})):
        path = tmp_path / f'{name}.json'
        save(path, payload)
        files[name] = {'path': str(path), 'sha256': digest(path)}
    for rank in (0, 1):
        payload = {'complete': True, 'samples': 65000, 'rank': rank, 'groups': {}}
        for group, rows in selection.items():
            payload['groups'][group] = {'rows': [reference_row(row['asset_id'], group, frame)
                                                for row in rows[rank::2] for frame in range(8)]}
        path = tmp_path / f'rank_{rank}.json'
        save(path, payload)
        files[f'rank_{rank}'] = {'path': str(path), 'sha256': digest(path)}
    config = {'evaluation': deepcopy(large.SMALL_EVALUATION),
              'large_evaluation': {'enabled': True, **deepcopy(large.LARGE_EVALUATION),
                                   'implementation_sha256': digest(large.__file__)},
              'reference_evaluation': {'files': files},
              'data': {'unchanged_marker': {'normalization': 'input_sequence_mesh_bbox'}},
              'optimization': {'structure_weight': .1},
              'quantization': {'bins': 256, 'continuous_range': [-1., 1.]}}
    panel, reference, manifest = runtime.load_small_reference_panel(config, entries)
    return SimpleNamespace(config=config, entries=entries, panel=deepcopy(panel), reference=reference,
                           panel_hash=manifest['panel_sha256'], output=tmp_path / 'output',
                           tokenizer=object(), codec=object())


class StubModel:
    def __init__(self, training=True):
        self.training = training
        self.weights = [1., 2., 3.]
        self.encodings = []

    def train(self, mode=True):
        self.training = mode
        return self

    def eval(self):
        return self.train(False)

    def encode(self, surface):
        assert self.training is False
        self.encodings.append((surface['asset_id'], tuple(surface['raw_frames'])))


class StubDataset(list):
    def __init__(self, rows, mode):
        super().__init__(rows)
        self.mode = mode


@pytest.fixture
def harness(case, monkeypatch):
    original_config = deepcopy(case.config)

    def fixed_dataset(rows, config, frames, mode):
        assert config == original_config and frames == 8
        return StubDataset(rows, mode)

    def prepare(dataset, indices, config, codec, device):
        assert config == original_config and codec is case.codec and device == 'cpu'
        assert len(indices) == 1
        asset = dataset[indices[0]]['asset_id']
        raw_frames = list(range(8, 16)) if dataset.mode == 'heldout_eval' else list(range(8))
        geometry = {'selected_frames': np.asarray([raw_frames]), 'heldout_raw_ids': [list(range(8, 16))]}
        return {'asset_id': asset, 'raw_frames': raw_frames}, {}, geometry, {}

    def evaluate(model, items, codec, *, frame_indices, group, original_tokenizer,
                 reference_lookup, structure_weight, **quantization):
        assert codec is case.codec and original_tokenizer is case.tokenizer
        assert frame_indices == list(range(8)) and structure_weight == .1
        assert quantization == original_config['quantization']
        rows, structures = [], []
        for surface, _, _ in items:
            model.encode(surface)
            asset = surface['asset_id']
            index = int(asset[1:])
            success = index % 9 != 0
            structures.append({'group': group, 'asset_id': asset, 'dfs_edge_f1': float(success),
                               'target_tokens': [501, 2], 'generated_tokens': [501, 2] if success else [501],
                               'success': success, 'status': 'eos' if success else 'hit_max'})
            for frame in frame_indices:
                original = reference_row(asset, group, frame)
                key = (group, asset, frame)
                assert reference_lookup[key]['target_tokens'] == original['target_tokens']
                rows.append({**original, 'target_tokens': [501, 2],
                             'success': success, 'exact_count': success,
                             'original_edge_f1': index % 17 / 20 + frame / 100 if success else 0.,
                             'j2j': .1 if success else None, 'gt_structure_xyz_ce': 2. + frame / 10,
                             'gt_structure_xyz_accuracy': .5, 'gt_structure_xyz_l1': .2,
                             'gt_structure_edge_f1': .8, 'input_anchor_points': [[index, frame, 1]],
                             'reference_original_edge_f1': reference_lookup[key]['original_edge_f1']})
        return {'rows': rows, 'structure_rows': structures,
                'summary': evaluation.summarize_fill_rows(rows, structures)}

    monkeypatch.setattr(runtime, 'read_entries', lambda config: deepcopy(case.entries))
    monkeypatch.setattr(runtime, 'fixed_dataset', fixed_dataset)
    monkeypatch.setattr(runtime, 'prepare_fill_batch', prepare)
    monkeypatch.setattr(evaluation, 'evaluate_fill_items', evaluate)
    import torch.distributed as dist
    monkeypatch.setattr(dist, 'barrier', lambda: None)
    return SimpleNamespace(case=case, evaluate=evaluate, prepare=prepare)


def run(case, model, *, rank=0, world=1):
    return large.run_large_evaluation(model, case.tokenizer, case.codec, case.config,
        SimpleNamespace(output=case.output), 'cpu', rank, world, 80000,
        case.panel, case.reference, case.panel_hash)


def test_large_panel_is_deterministic_deep_copy_and_exact_small_superset(case):
    original, original_entries = deepcopy(case.config), deepcopy(case.entries)
    first = large.load_large_reference_panel(case.config, case.entries)
    assert first == large.load_large_reference_panel(case.config, case.entries)
    panel, reference, manifest = first
    assert {key: len(rows) for key, rows in panel.items()} == large.GROUP_SIZES
    assert len(reference) == 1536 and len(case.reference) == 128
    assert all(reference[key] == row for key, row in case.reference.items())
    assert manifest['scored_frame_indices'] == list(range(8))
    assert manifest['panel_sha256'] != case.panel_hash
    assert case.config == original and case.entries == original_entries


@pytest.mark.parametrize('key,value', [
    ('enabled', False), ('enabled', 1), ('train_assets', 31), ('valid_assets', 127),
    ('input_frames', 4), ('seed', 20261023), ('frame_indices', [0, 2, 4, 6]),
    ('frame_indices', [False, 1, 2, 3, 4, 5, 6, 7]), ('implementation_sha256', '0' * 64),
])
def test_large_configuration_and_implementation_hash_fail_closed(case, key, value):
    case.config['large_evaluation'][key] = value
    with pytest.raises(ValueError, match='large_evaluation'):
        large.load_large_reference_panel(case.config, case.entries)


def test_small_configuration_cannot_be_expanded_in_place(case):
    case.config['evaluation']['valid_assets'] = 128
    with pytest.raises(ValueError, match='small evaluation contract'):
        large.load_large_reference_panel(case.config, case.entries)


def rewrite_reference(case, edit, *, repin=True):
    spec = case.config['reference_evaluation']['files']['rank_0']
    path = Path(spec['path'])
    value = json.loads(path.read_text())
    edit(value['groups']['valid']['rows'])
    save(path, value)
    if repin:
        spec['sha256'] = digest(path)


def test_pinned_reference_checksum_is_required(case):
    rewrite_reference(case, lambda rows: rows.pop(), repin=False)
    with pytest.raises(ValueError, match='reference changed'):
        large.load_large_reference_panel(case.config, case.entries)


@pytest.mark.parametrize('edit', [
    lambda rows: rows.pop(),
    lambda rows: rows.append(deepcopy(rows[0])),
    lambda rows: rows[0].update(frame_index=8),
    lambda rows: rows[0].update(group='train_seen'),
    lambda rows: rows[0].update(raw_frame_index=rows[1]['raw_frame_index']),
])
def test_missing_duplicate_wrong_group_or_frame_reference_is_rejected(case, edit):
    rewrite_reference(case, edit)
    with pytest.raises(ValueError):
        large.load_large_reference_panel(case.config, case.entries)


def test_heldout_selection_must_retain_original_manifest_identity(case):
    spec = case.config['reference_evaluation']['files']['selection']
    path = Path(spec['path'])
    value = json.loads(path.read_text())
    value['train_heldout'][0]['root_rt']['offset'] = -1
    save(path, value)
    spec['sha256'] = digest(path)
    with pytest.raises(ValueError, match='manifest/raw-GT descriptor'):
        large.load_large_reference_panel(case.config, case.entries)


@pytest.mark.parametrize('training', [True, False])
def test_both_summaries_project_identical_values_without_duplicate_encoding(harness, training):
    case = harness.case
    original_config = deepcopy(case.config)
    model = StubModel(training)
    summary = run(case, model)
    assert model.training is training and model.weights == [1., 2., 3.]
    assert case.config == original_config
    assert len(model.encodings) == len(set(model.encodings)) == 192
    large_root = case.output / 'evaluation_large_080000'
    small_root = case.output / 'evaluation_080000'
    large_report = json.loads((large_root / 'rank_0.json').read_text())
    small_report = json.loads((small_root / 'rank_0.json').read_text())
    large_summary = json.loads((large_root / 'summary.json').read_text())
    assert summary == json.loads((small_root / 'summary.json').read_text())
    assert summary['panel_sha256'] == case.panel_hash
    assert summary['evaluation_assets'] == large.SMALL_GROUP_SIZES
    assert large_summary['evaluation_assets'] == large.GROUP_SIZES
    assert sum(group['score_frame_count'] for group in large_summary['groups'].values()) == 1536
    assert sum(group['score_frame_count'] for group in summary['groups'].values()) == 128
    assert summary['full_validation'] is False and summary['failed_generations_scored_zero'] is True
    assert large_summary['quality_acceptance'] is False
    for group, result in small_report['groups'].items():
        large_rows = {(row['asset_id'], row['frame_index']): row for row in large_report['groups'][group]['rows']}
        for row in result['rows']:
            assert row == large_rows[row['asset_id'], row['frame_index']]
        assert all(row['original_edge_f1'] == 0 for row in result['rows'] if not row['success'])


def test_uneven_distributed_projection_has_exact_global_counts(harness):
    models = [StubModel(False) for _ in range(3)]
    for rank in (1, 2):
        assert run(harness.case, models[rank], rank=rank, world=3) is None
    summary = run(harness.case, models[0], rank=0, world=3)
    assert all(model.training is False for model in models)
    assert sum(len(model.encodings) for model in models) == 192
    assert sum(group['score_frame_count'] for group in summary['groups'].values()) == 128
    for group, count in large.SMALL_GROUP_SIZES.items():
        keys = []
        for rank in range(3):
            payload = json.loads((harness.case.output / 'evaluation_080000' / f'rank_{rank}.json').read_text())
            keys += [(row['asset_id'], row['frame_index']) for row in payload['groups'][group]['rows']]
        assert len(keys) == len(set(keys)) == count * 4


def test_empty_local_groups_are_explicit_and_do_not_call_empty_evaluator(harness):
    model = StubModel(False)
    assert run(harness.case, model, rank=35, world=40) is None
    assert model.training is False
    for label in ('evaluation_large_080000', 'evaluation_080000'):
        payload = json.loads((harness.case.output / label / 'rank_35.json').read_text())
        for group in ('train_seen', 'train_heldout'):
            result = payload['groups'][group]
            assert result['rows'] == result['structure_rows'] == []
            assert result['summary']['score_frame_count'] == 0
            assert result['summary']['original_edge_f1'] is None


def test_nonempty_large_shard_can_have_an_empty_small_projection(harness):
    case = harness.case
    panel, _, _ = large.load_large_reference_panel(case.config, case.entries)
    small_ids = {row['asset_id'] for row in case.panel['train_seen']}
    rank = next(number for number in range(1, 16)
                if not small_ids.intersection(row['asset_id'] for row in panel['train_seen'][number::16]))
    run(case, StubModel(False), rank=rank, world=16)
    large_report = json.loads((case.output / 'evaluation_large_080000' / f'rank_{rank}.json').read_text())
    small_report = json.loads((case.output / 'evaluation_080000' / f'rank_{rank}.json').read_text())
    assert len(large_report['groups']['train_seen']['rows']) == 16
    assert small_report['groups']['train_seen']['rows'] == []
    assert small_report['groups']['train_seen']['summary']['asset_count'] == 0


@pytest.mark.parametrize('which', ['hash', 'asset', 'raw_frame', 'GT', 'AR_tokens'])
def test_supplied_small_panel_must_be_the_exact_frozen_subset(harness, which):
    case = harness.case
    key = next(iter(case.reference))
    if which == 'hash':
        case.panel_hash = '0' * 64
    elif which == 'asset':
        case.panel['valid'][0]['asset_id'] = 'wrong'
    elif which == 'raw_frame':
        case.reference[key]['raw_frame_index'] = 999
    elif which == 'GT':
        case.reference[key]['original_target_joints'][0][0] = -999
    else:
        case.reference[key]['target_tokens'] = [999]
    model = StubModel(False)
    with pytest.raises(ValueError, match='supplied small panel'):
        run(case, model)
    assert not model.encodings and model.training is False


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'frame', 'raw_frame', 'GT',
                                 'group', 'structure', 'failure_F1', 'nonfinite'])
def test_bad_evaluator_output_fails_and_restores_mode(harness, monkeypatch, kind):
    def broken(*args, **kwargs):
        result = harness.evaluate(*args, **kwargs)
        if kind == 'missing':
            result['rows'].pop()
        elif kind == 'duplicate':
            result['rows'].append(deepcopy(result['rows'][0]))
        elif kind == 'frame':
            result['rows'][0]['frame_index'] = 99
        elif kind == 'raw_frame':
            result['rows'][0]['raw_frame_index'] = 999
        elif kind == 'GT':
            result['rows'][0]['original_target_joints'] = [[999., 0., 0.]]
        elif kind == 'group':
            result['rows'][0]['group'] = 'wrong'
        elif kind == 'structure':
            result['structure_rows'].append(deepcopy(result['structure_rows'][0]))
        elif kind == 'failure_F1':
            result['rows'][0].update(success=False, original_edge_f1=.5)
        else:
            result['rows'][0]['original_edge_f1'] = float('nan')
        return result

    monkeypatch.setattr(evaluation, 'evaluate_fill_items', broken)
    model = StubModel(True)
    with pytest.raises(RuntimeError):
        run(harness.case, model)
    assert model.training is True
    assert not (harness.case.output / 'evaluation_080000/summary.json').exists()


@pytest.mark.parametrize('training', [True, False])
def test_frame_partition_errors_restore_incoming_mode(harness, monkeypatch, training):
    def crossed(*args, **kwargs):
        surface, targets, geometry, decision = harness.prepare(*args, **kwargs)
        geometry['heldout_raw_ids'] = [list(range(8))]
        return surface, targets, geometry, decision

    monkeypatch.setattr(runtime, 'prepare_fill_batch', crossed)
    model = StubModel(training)
    with pytest.raises(ValueError, match='heldout partition'):
        run(harness.case, model)
    assert model.training is training and not model.encodings


@pytest.mark.parametrize('kind', ['incomplete', 'wrong_rank', 'wrong_budget', 'projection_value'])
def test_cross_rank_aggregation_rejects_stale_or_changed_reports(harness, kind):
    case = harness.case
    run(case, StubModel(), rank=1, world=2)
    path = case.output / 'evaluation_080000/rank_1.json'
    payload = json.loads(path.read_text())
    if kind == 'incomplete':
        payload['complete'] = False
    elif kind == 'wrong_rank':
        payload['rank'] = 0
    elif kind == 'wrong_budget':
        payload['asset_exposures'] = 79000
    else:
        payload['groups']['valid']['rows'][0]['gt_structure_xyz_l1'] += .01
    save(path, payload)
    model = StubModel(False)
    with pytest.raises(RuntimeError):
        run(case, model, rank=0, world=2)
    assert model.training is False
    assert not (case.output / 'evaluation_large_080000/summary.json').exists()
