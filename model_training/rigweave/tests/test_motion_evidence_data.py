from __future__ import annotations

import torch

from rigweave.motion_evidence.data import _pad_skin_weights, _token_joint_indices


def test_pad_skin_weights_preserves_each_vertex_joint_block() -> None:
    first = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    second = torch.arange(10, dtype=torch.float32).reshape(2, 5)
    padded = _pad_skin_weights([first, second])
    assert padded.shape == (2, 4, 5)
    torch.testing.assert_close(padded[0, :4, :3], first)
    torch.testing.assert_close(padded[1, :2, :5], second)
    assert torch.count_nonzero(padded[0, :, 3:]) == 0
    assert torch.count_nonzero(padded[1, 2:]) == 0


def test_token_joint_indices_cover_chain_and_branch_coordinates() -> None:
    branch = 256
    eos = 258
    tokens = torch.tensor(
        [257, 266, 10, 11, 12, 20, 21, 22, branch, 10, 11, 12, 30, 31, 32, eos]
    )
    mapping = _token_joint_indices(
        tokens,
        torch.tensor([-1, 0, 0]),
        branch_token=branch,
        eos_token=eos,
    )
    assert mapping.tolist() == [
        -1,
        0,
        0,
        0,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        2,
        2,
        2,
        -1,
        -1,
    ]
