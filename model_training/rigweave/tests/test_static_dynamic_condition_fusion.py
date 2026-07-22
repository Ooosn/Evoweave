from __future__ import annotations

import torch

from rigweave.dynamic_rig.unirig_wrapper import StaticDynamicConditionFusion


def test_zero_update_is_exact_static_identity_and_trainable() -> None:
    torch.manual_seed(17)
    module = StaticDynamicConditionFusion(
        dim=32,
        heads=4,
        gate_init=0.25,
        zero_init_update=True,
        depth=1,
    )
    static = torch.randn(2, 7, 32, requires_grad=True)
    dynamic = torch.randn(2, 7, 32, requires_grad=True)

    initial = module(static, dynamic)
    assert torch.equal(initial, static)

    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    initial.square().mean().backward()
    final_update = module.blocks[0].update[-1]
    assert final_update.weight.grad is not None
    assert torch.isfinite(final_update.weight.grad).all()
    assert final_update.weight.grad.abs().sum() > 0
    assert static.grad is not None
    assert static.grad.abs().sum() > 0

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    changed_a = module(static.detach(), dynamic.detach())
    changed_b = module(static.detach(), dynamic.detach() + 0.5)

    assert not torch.equal(changed_a, static.detach())
    assert not torch.equal(changed_a, changed_b)


def test_fusion_rejects_misaligned_condition_shapes() -> None:
    module = StaticDynamicConditionFusion(dim=16, heads=4, zero_init_update=True)

    try:
        module(torch.zeros(1, 8, 16), torch.zeros(1, 7, 16))
    except ValueError as exc:
        assert "same shape" in str(exc)
    else:
        raise AssertionError("misaligned condition tokens must be rejected")
