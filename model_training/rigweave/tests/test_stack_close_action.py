from __future__ import annotations

import torch

from rigweave.stack_close.model import _stack_action_targets


def test_stack_action_targets_align_with_next_token_positions() -> None:
    labels = torch.tensor(
        [[266, 1, 2, 3, 4, 5, 6, 256, 256, 258]],
        dtype=torch.long,
    )
    coordinate_positions = torch.tensor(
        [[[2, 3, 4], [5, 6, 7]]],
        dtype=torch.long,
    )
    joint_count = torch.tensor([2], dtype=torch.long)

    result = _stack_action_targets(
        labels,
        coordinate_positions,
        joint_count,
        close_token=256,
    )

    assert result.tolist() == [
        [-100, -100, -100, -100, 0, -100, -100, 1, 1, -100]
    ]
