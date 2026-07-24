"""Prefix-conditioned surface support and explicit null supervision."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PrefixSupportTargets:
    """Training-only distribution over query anchors plus a null state."""

    anchor_distribution: torch.Tensor
    null_probability: torch.Tensor
    valid_mask: torch.BoolTensor
    raw_support: torch.Tensor

    @property
    def distribution(self) -> torch.Tensor:
        return torch.cat(
            (self.anchor_distribution, self.null_probability[..., None]),
            dim=-1,
        )


class PrefixSurfaceSupportHead(nn.Module):
    """Predict a query-anchor distribution and an explicit abstention logit."""

    def __init__(
        self,
        hidden_size: int,
        *,
        projection_size: int = 128,
        detach_static_keys: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or projection_size <= 0:
            raise ValueError("hidden and projection sizes must be positive")
        self.hidden_size = int(hidden_size)
        self.projection_size = int(projection_size)
        self.detach_static_keys = bool(detach_static_keys)
        self.query_norm = nn.LayerNorm(hidden_size)
        self.key_norm = nn.LayerNorm(hidden_size)
        self.query_projection = nn.Linear(hidden_size, projection_size, bias=False)
        self.key_projection = nn.Linear(hidden_size, projection_size, bias=False)
        self.null_projection = nn.Linear(hidden_size, 1)
        self.scale = float(projection_size) ** -0.5

    def forward(
        self,
        prefix_states: torch.Tensor,
        static_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if prefix_states.ndim != 3 or static_tokens.ndim != 3:
            raise ValueError("prefix_states and static_tokens must be rank-3")
        if prefix_states.shape[0] != static_tokens.shape[0]:
            raise ValueError("prefix and static-token batch sizes differ")
        if prefix_states.shape[-1] != self.hidden_size:
            raise ValueError("prefix hidden width does not match support head")
        if static_tokens.shape[-1] != self.hidden_size:
            raise ValueError("static-token width does not match support head")

        keys = static_tokens.detach() if self.detach_static_keys else static_tokens
        compute_dtype = self.query_norm.weight.dtype
        with torch.autocast(device_type=prefix_states.device.type, enabled=False):
            normalized_query = self.query_norm(prefix_states.to(compute_dtype))
            normalized_keys = self.key_norm(keys.to(compute_dtype))
            query = self.query_projection(normalized_query)
            key = self.key_projection(normalized_keys)
            anchor_logits = torch.matmul(query, key.transpose(1, 2)) * self.scale
            null_logits = self.null_projection(normalized_query)
        return torch.cat((anchor_logits, null_logits), dim=-1)


def prefix_support_targets(
    query_skin: torch.Tensor,
    token_joint_indices: torch.LongTensor,
    completed_joint_counts: torch.LongTensor,
    branch_decision_mask: torch.BoolTensor,
    joint_counts: torch.LongTensor,
    *,
    eps: float = 1.0e-8,
) -> PrefixSupportTargets:
    """Build next-token support from skin weights without changing inference input."""

    if query_skin.ndim != 3:
        raise ValueError("query_skin must have shape (B,Q,J)")
    state_tensors = (
        token_joint_indices,
        completed_joint_counts,
        branch_decision_mask,
    )
    if any(value.ndim != 2 for value in state_tensors):
        raise ValueError("token support-state tensors must be rank-2")
    if any(value.shape != token_joint_indices.shape for value in state_tensors):
        raise ValueError("token support-state tensors must have identical shapes")
    if query_skin.shape[0] != token_joint_indices.shape[0]:
        raise ValueError("skin and token-state batch sizes differ")
    if joint_counts.shape != (query_skin.shape[0],):
        raise ValueError("joint_counts must identify every batch item")

    batch_size, query_count, max_joints = query_skin.shape
    if max_joints <= 0:
        raise ValueError("query_skin must contain at least one joint column")
    device = query_skin.device
    joint_indices = token_joint_indices.to(device=device, dtype=torch.long)
    completed = completed_joint_counts.to(device=device, dtype=torch.long)
    branch_mask = branch_decision_mask.to(device=device, dtype=torch.bool)
    counts = joint_counts.to(device=device, dtype=torch.long)
    if bool((counts <= 0).any()) or bool((counts > max_joints).any()):
        raise ValueError("joint_counts fall outside the padded skin width")

    skin_by_joint = query_skin.transpose(1, 2)
    valid_joint = (joint_indices >= 0) & (joint_indices < counts[:, None])
    safe_joint = joint_indices.clamp(min=0, max=max_joints - 1)
    joint_support = torch.gather(
        skin_by_joint,
        1,
        safe_joint[..., None].expand(-1, -1, query_count),
    )
    joint_support = joint_support * valid_joint[..., None].to(joint_support.dtype)

    suffix_support = query_skin.flip(dims=(-1,)).cumsum(dim=-1).flip(dims=(-1,))
    suffix_support = nn.functional.pad(suffix_support, (0, 1), value=0.0)
    safe_completed = completed.clamp(min=0, max=max_joints)
    remaining_support = torch.gather(
        suffix_support.transpose(1, 2),
        1,
        safe_completed[..., None].expand(-1, -1, query_count),
    )

    support = joint_support
    branch_support = (joint_support + remaining_support).clamp(0.0, 1.0)
    support = torch.where(branch_mask[..., None], branch_support, support)
    valid_state = (completed >= 0) & (completed <= counts[:, None])
    support = support * valid_state[..., None].to(support.dtype)

    mass = support.sum(dim=-1)
    has_support = mass > eps
    anchor_distribution = support / mass[..., None].clamp_min(eps)
    anchor_distribution = torch.where(
        has_support[..., None],
        anchor_distribution,
        torch.zeros_like(anchor_distribution),
    )
    null_probability = (valid_state & ~has_support).to(query_skin.dtype)
    return PrefixSupportTargets(
        anchor_distribution=anchor_distribution,
        null_probability=null_probability,
        valid_mask=valid_state,
        raw_support=support,
    )


def prefix_support_distribution_loss(
    logits: torch.Tensor,
    targets: PrefixSupportTargets,
    attention_mask: torch.Tensor,
    *,
    static_prefix_steps: int,
) -> dict[str, torch.Tensor]:
    """Balance supported next-token states against terminal/null states."""

    if static_prefix_steps < 0:
        raise ValueError("static_prefix_steps must be non-negative")
    if logits.shape[:-1] != targets.valid_mask.shape:
        raise ValueError("support logits and target positions differ")
    if logits.shape[-1] != targets.anchor_distribution.shape[-1] + 1:
        raise ValueError("support logits must contain every anchor plus null")
    if attention_mask.shape != targets.valid_mask.shape:
        raise ValueError("attention_mask and support target positions differ")

    log_probabilities = nn.functional.log_softmax(logits.float(), dim=-1)
    per_position = -(targets.distribution.float() * log_probabilities).sum(dim=-1)
    predicts_token = torch.zeros_like(targets.valid_mask)
    predicts_token[:, :-1] = attention_mask[:, 1:] != 0
    positions = torch.arange(logits.shape[1], device=logits.device)[None]
    valid = (
        targets.valid_mask
        & predicts_token
        & (positions >= int(static_prefix_steps))
    )
    terminal = valid & (targets.null_probability > 0.5)
    supported = valid & ~terminal
    group_losses = []
    if bool(supported.any()):
        group_losses.append(per_position[supported].mean())
    if bool(terminal.any()):
        group_losses.append(per_position[terminal].mean())
    if not group_losses:
        raise ValueError("no valid prefix support targets")
    loss = torch.stack(group_losses).mean()
    probabilities = log_probabilities.exp()
    return {
        "prefix_support_loss": loss,
        "prefix_support_valid_fraction": valid.float().mean(),
        "prefix_support_null_probability_supported": (
            probabilities[..., -1][supported].mean()
            if bool(supported.any())
            else loss.new_zeros(())
        ),
        "prefix_support_null_probability_terminal": (
            probabilities[..., -1][terminal].mean()
            if bool(terminal.any())
            else loss.new_zeros(())
        ),
    }
