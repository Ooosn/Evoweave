import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from plot_structure_xyz_fill import load_rows


def test_normal_evaluation_requires_complete_ranks(tmp_path):
    path = tmp_path / 'rank_0.json'
    path.write_text(json.dumps({'complete': False, 'groups': {'valid': {'rows': [1]}}}))
    with pytest.raises(ValueError, match='complete'):
        load_rows(tmp_path, 'valid')
    path.write_text(json.dumps({'complete': True, 'groups': {'valid': {'rows': [1]}}}))
    assert load_rows(tmp_path, 'valid') == ([1], 'valid')


def test_fit_evidence_is_an_explicit_format_and_step(tmp_path):
    for rank in range(2):
        (tmp_path / f'fit_eval_0032_rank_{rank}.json').write_text(json.dumps({'rows': [rank]}))
    (tmp_path / 'fit_eval_0000_rank_0.json').write_text(json.dumps({'rows': [99]}))
    assert load_rows(tmp_path, 'valid', 32) == ([0, 1], 'fixed_fit')
    with pytest.raises(ValueError, match='no saved'):
        load_rows(tmp_path, 'valid', 16)
    with pytest.raises(ValueError, match='nonnegative'):
        load_rows(tmp_path, 'valid', -1)
