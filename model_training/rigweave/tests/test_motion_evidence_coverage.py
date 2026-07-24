from __future__ import annotations

import torch

from rigweave.motion_evidence import (
    CoverageAwareMotionEvidenceCrossAttention,
    MotionEvidenceCrossAttention,
    prefix_support_distribution_loss,
    prefix_support_targets,
)
from rigweave.motion_evidence.data import (
    _token_coverage_state,
    _token_joint_indices,
)


def _token_state() -> tuple[torch.Tensor, ...]:
    branch = 256
    eos = 258
    tokens = torch.tensor(
        [257, 266, 10, 11, 12, 20, 21, 22, branch, 10, 11, 12, 30, 31, 32, eos]
    )
    parents = torch.tensor([-1, 0, 0])
    joint_indices = _token_joint_indices(
        tokens,
        parents,
        branch_token=branch,
        eos_token=eos,
    )
    completed, branch_decision = _token_coverage_state(
        tokens,
        parents,
        branch_token=branch,
        eos_token=eos,
    )
    return tokens, joint_indices, completed, branch_decision


def test_prefix_support_targets_preserve_branch_parent_child_and_null_roles() -> None:
    _, joint_indices, completed, branch_decision = _token_state()
    query_skin = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        ]
    )
    targets = prefix_support_targets(
        query_skin,
        joint_indices[None],
        completed[None],
        branch_decision[None],
        torch.tensor([3]),
    )

    # A sequential child coordinate reads that child's surface support.
    torch.testing.assert_close(
        targets.anchor_distribution[0, 4],
        torch.tensor([0.0, 1.0, 0.0, 0.0]),
    )
    # The branch decision keeps both the referenced parent and remaining child.
    torch.testing.assert_close(
        targets.anchor_distribution[0, 7],
        torch.tensor([0.5, 0.0, 0.5, 0.0]),
    )
    # Parent replay and child coordinates retain their distinct support targets.
    torch.testing.assert_close(
        targets.anchor_distribution[0, 8],
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        targets.anchor_distribution[0, 11],
        torch.tensor([0.0, 0.0, 1.0, 0.0]),
    )
    assert float(targets.null_probability[0, 14]) == 1.0
    assert torch.count_nonzero(targets.anchor_distribution[0, 14]) == 0


def test_prefix_support_distribution_loss_is_finite_and_trains_logits() -> None:
    tokens, joint_indices, completed, branch_decision = _token_state()
    query_skin = torch.eye(3).new_zeros((1, 4, 3))
    query_skin[0, :3] = torch.eye(3)
    targets = prefix_support_targets(
        query_skin,
        joint_indices[None],
        completed[None],
        branch_decision[None],
        torch.tensor([3]),
    )
    logits = torch.randn(1, tokens.numel(), 5, requires_grad=True)
    output = prefix_support_distribution_loss(
        logits,
        targets,
        torch.ones((1, tokens.numel())),
        static_prefix_steps=4,
    )
    output["prefix_support_loss"].backward()
    assert torch.isfinite(output["prefix_support_loss"])
    assert float(output["prefix_support_branch_valid_fraction"]) > 0.0
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0


def test_coverage_attention_matches_ungated_attention_with_uniform_prior() -> None:
    torch.manual_seed(91)
    baseline = MotionEvidenceCrossAttention(16, 4, residual_scale=0.1)
    coverage = CoverageAwareMotionEvidenceCrossAttention(
        16,
        4,
        residual_scale=0.1,
        coverage_bias_strength=0.0,
        support_projection_size=8,
    )
    coverage.load_attention_state(baseline)
    with torch.no_grad():
        coverage.support_head.query_projection.weight.zero_()
        coverage.support_head.key_projection.weight.zero_()
        coverage.support_head.null_projection.weight.zero_()
        coverage.support_head.null_projection.bias.fill_(-50.0)
    prefix = torch.randn(2, 7, 16)
    static = torch.randn(2, 5, 16)
    motion = torch.randn(2, 5, 16)
    torch.testing.assert_close(
        coverage(prefix, static, motion),
        baseline(prefix, static, motion),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_coverage_attention_is_exact_noop_for_zero_motion_values() -> None:
    torch.manual_seed(92)
    attention = CoverageAwareMotionEvidenceCrossAttention(
        16,
        4,
        residual_scale=0.1,
        support_projection_size=8,
    )
    prefix = torch.randn(2, 7, 16)
    static = torch.randn(2, 5, 16)
    refined = attention(prefix, static, torch.zeros_like(static))
    assert torch.equal(refined, prefix)
    weights = attention.attention_weights(prefix, static, torch.zeros_like(static))
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones((2, 4, 7)))


def test_coverage_attention_null_state_suppresses_motion_update() -> None:
    torch.manual_seed(93)
    attention = CoverageAwareMotionEvidenceCrossAttention(
        16,
        4,
        residual_scale=0.1,
        support_projection_size=8,
    )
    prefix = torch.randn(2, 7, 16)
    static = torch.randn(2, 5, 16)
    motion = torch.randn(2, 5, 16)
    with torch.no_grad():
        attention.support_head.query_projection.weight.zero_()
        attention.support_head.key_projection.weight.zero_()
        attention.support_head.null_projection.weight.zero_()
        attention.support_head.null_projection.bias.fill_(-50.0)
    active = attention(prefix, static, motion)
    assert float((active - prefix).detach().abs().sum()) > 0.0

    with torch.no_grad():
        attention.support_head.null_projection.bias.fill_(50.0)
    suppressed = attention(prefix, static, motion)
    torch.testing.assert_close(suppressed, prefix, atol=1.0e-7, rtol=0.0)
