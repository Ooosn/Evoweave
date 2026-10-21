from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'analysis'))
from compare_structure_xyz_trials import localization_diagnostics


def row(prediction, indices=(0, 1)):
    original = np.asarray([[-.5, 0, 0], [.5, 0, 0]], dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    return {'original_target_joints': original.tolist(), 'fill_source_indices': list(indices),
            'gt_structure_predicted_joints': prediction.tolist(),
            'gt_structure_xyz_l1': float(np.abs(prediction - original[list(indices)]).mean())}


def test_center_bin_collapse_has_zero_improvement_and_span():
    result = localization_diagnostics([row(np.full((2, 3), 1 / 256))])
    assert result['relative_l1_reduction_vs_center_bin'] == 0
    assert result['origin_constant_l1'] == pytest.approx(1 / 6)
    assert result['single_point_prediction_views'] == 1
    assert result['predicted_bbox_diagonal'] == 0
    assert result['target_bbox_diagonal'] == 1


def test_perfect_prediction_has_full_improvement_and_target_span():
    result = localization_diagnostics([row([[-.5, 0, 0], [.5, 0, 0]])])
    assert result['relative_l1_reduction_vs_center_bin'] == 1
    assert result['predicted_bbox_diagonal'] == result['target_bbox_diagonal'] == 1
    assert result['mean_distinct_predicted_xyz'] == 2


def test_reordered_and_pruned_mapping_is_respected():
    result = localization_diagnostics([row([[.5, 0, 0], [-.5, 0, 0]], (1, 0)), row([[.5, 0, 0]], (1,))])
    assert result['relative_l1_reduction_vs_center_bin'] == 1
    assert result['views'] == 2


@pytest.mark.parametrize('change', ['metric', 'indices', 'nonfinite'])
def test_inconsistent_saved_evidence_is_rejected(change):
    sample = deepcopy(row([[-.5, 0, 0], [.5, 0, 0]]))
    if change == 'metric':
        sample['gt_structure_xyz_l1'] = .1
    elif change == 'indices':
        sample['fill_source_indices'] = [0, 0]
    else:
        sample['gt_structure_predicted_joints'][0][0] = float('nan')
    with pytest.raises(ValueError):
        localization_diagnostics([sample])


def test_empty_diagnostics():
    assert localization_diagnostics([]) == {'views': 0}
