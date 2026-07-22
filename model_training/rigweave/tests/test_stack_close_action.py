from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from rigweave.stack_close.model import StackActionHead, _stack_action_targets
from rigweave.stack_close.tokenizer import StackCloseTokenizer


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


def test_stack_action_targets_match_serialized_tree_decisions() -> None:
    legacy = SimpleNamespace(
        num_discrete=256,
        continuous_range=(-1.0, 1.0),
        vocab_size=267,
        token_id_branch=256,
        bos=257,
        eos=258,
        pad=259,
        token_id_cls_none=263,
        cls_token_id={"articulationxl": 266},
    )
    tokenizer = StackCloseTokenizer(legacy)
    serialization = tokenizer.serialize_tree(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.25, 0.0, 0.0],
                [0.0, 0.25, 0.0],
                [0.0, 0.5, 0.0],
            ],
            dtype=np.float32,
        ),
        np.asarray([-1, 0, 0, 2], dtype=np.int64),
        cls="articulationxl",
        sibling_rng=None,
    )
    labels = torch.from_numpy(serialization.tokens[1:]).unsqueeze(0)
    positions = torch.from_numpy(serialization.coordinate_token_positions).unsqueeze(0)
    actions = _stack_action_targets(
        labels,
        positions,
        torch.tensor([serialization.joints.shape[0]]),
        close_token=tokenizer.token_id_close,
    )[0]

    expected = torch.full_like(actions, -100)
    for coordinate_start in serialization.coordinate_token_positions[1:, 0]:
        expected[int(coordinate_start) - 1] = 0
    close_positions = np.flatnonzero(serialization.tokens == tokenizer.token_id_close)
    for close_position in close_positions:
        expected[int(close_position) - 1] = 1

    torch.testing.assert_close(actions, expected)


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
