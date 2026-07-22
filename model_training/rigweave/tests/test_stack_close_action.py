from __future__ import annotations

import torch

from rigweave.stack_close.model import StackActionHead, _stack_action_targets


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


def test_condition_aware_stack_action_head_shape_and_gradients() -> None:
    torch.manual_seed(7)
    head = StackActionHead(8, condition_dim=4, heads=2)
    query = torch.randn(2, 3, 8)
    condition = torch.randn(2, 5, 8)

    logits = head(query, condition)
    assert logits.shape == (2, 3, 2)
    logits.sum().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in head.parameters()
    )


def test_conditionless_stack_action_head_ignores_condition_values() -> None:
    torch.manual_seed(11)
    head = StackActionHead(8, condition_dim=0, heads=2)
    query = torch.randn(1, 4, 8)
    condition_a = torch.randn(1, 5, 8)
    condition_b = torch.randn(1, 5, 8)

    torch.testing.assert_close(
        head(query, condition_a),
        head(query, condition_b),
    )
