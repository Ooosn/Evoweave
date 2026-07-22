from __future__ import annotations

import torch

from rigweave.dynamic_rig.unirig_wrapper import (
    AnchorWiseMotionResidualFusion,
    StaticDynamicConditionFusion,
)


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


def test_anchor_residual_is_exact_identity_and_does_not_mix_anchor_indices() -> None:
    torch.manual_seed(23)
    module = AnchorWiseMotionResidualFusion(dim=16, gate_init=0.25, zero_init_update=True)
    static = torch.randn(1, 5, 16)
    dynamic = torch.randn(1, 5, 16)

    assert torch.equal(module(static, dynamic), static)

    with torch.no_grad():
        module.update[-1].weight.normal_(std=0.02)
        module.update[-1].bias.zero_()
    baseline = module(static, dynamic)
    changed_dynamic = dynamic.clone()
    changed_dynamic[:, 2] += torch.linspace(-1.0, 1.0, 16)
    changed = module(static, changed_dynamic)

    difference = (changed - baseline).abs().sum(dim=-1)
    assert difference[0, 2] > 0
    assert torch.equal(difference[0, :2], torch.zeros_like(difference[0, :2]))
    assert torch.equal(difference[0, 3:], torch.zeros_like(difference[0, 3:]))
