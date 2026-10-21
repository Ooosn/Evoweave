from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'analysis'))
from compare_structure_xyz_trials import localization_diagnostics, pose_response_diagnostics


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


def test_pose_response_matches_joint_identities_and_distinguishes_static_prediction():
    first = dict(row([[-.5, 0, 0], [.5, 0, 0]]), asset_id='a', frame_index=0)
    second = dict(row([[.5, 0, 0], [-.5, 0, 0]], (1, 0)), asset_id='a', frame_index=2)
    second['original_target_joints'] = (np.asarray(second['original_target_joints']) + .1).tolist()
    static = pose_response_diagnostics([first, second])
    assert static['predicted_to_GT_change_ratio'] == 0
    assert static['mean_GT_coordinate_change'] == pytest.approx(.1)
    second['gt_structure_predicted_joints'] = (np.asarray(second['gt_structure_predicted_joints']) + .1).tolist()
    moving = pose_response_diagnostics([first, second])
    assert moving['predicted_to_GT_change_ratio'] == pytest.approx(1)
    assert moving['pose_pairs'] == 1
    second['frame_index'] = 0
    with pytest.raises(ValueError):
        pose_response_diagnostics([first, second])


def test_no_pose_pairs_or_motion_does_not_invent_a_ratio():
    first = dict(row([[-.5, 0, 0], [.5, 0, 0]]), asset_id='a', frame_index=0)
    assert pose_response_diagnostics([first])['pose_pairs'] == 0
    second = dict(first, frame_index=2)
    assert pose_response_diagnostics([first, second])['predicted_to_GT_change_ratio'] is None
