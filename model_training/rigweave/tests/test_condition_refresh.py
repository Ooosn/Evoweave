from __future__ import annotations

import torch

from rigweave.stack_close_refresh import BottleneckConditionRefresh


def test_zero_gate_is_exact_identity_and_opens_gradient_path() -> None:
    torch.manual_seed(7)
    module = BottleneckConditionRefresh(
        32,
        refresh_dim=16,
        heads=4,
    )
    query = torch.randn(2, 5, 32, requires_grad=True)
    condition = torch.randn(2, 11, 32, requires_grad=True)

    initial = module(query, condition)
    assert torch.equal(initial, query)

    initial.square().mean().backward()
    assert module.gate.grad is not None
    assert torch.isfinite(module.gate.grad).all()
    assert module.gate.grad.abs().sum() > 0

    with torch.no_grad():
        module.gate.add_(0.01)
    module.zero_grad(set_to_none=True)
    query.grad = None
    condition.grad = None
    module(query, condition).square().mean().backward()
    assert module.query_proj.weight.grad is not None
    assert module.query_proj.weight.grad.abs().sum() > 0
    assert module.key_proj.weight.grad is not None
    assert module.key_proj.weight.grad.abs().sum() > 0
