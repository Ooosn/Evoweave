from __future__ import annotations

import torch

from rigweave.motion_evidence.data import _pad_skin_weights


def test_pad_skin_weights_preserves_each_vertex_joint_block() -> None:
    first = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    second = torch.arange(10, dtype=torch.float32).reshape(2, 5)
    padded = _pad_skin_weights([first, second])
    assert padded.shape == (2, 4, 5)
    torch.testing.assert_close(padded[0, :4, :3], first)
    torch.testing.assert_close(padded[1, :2, :5], second)
    assert torch.count_nonzero(padded[0, :, 3:]) == 0
    assert torch.count_nonzero(padded[1, 2:]) == 0
