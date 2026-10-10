"""Motion observations from fixed query-pose groups, without skeleton inputs."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class MotionEvidence:
    states: torch.Tensor
    anchor_features: torch.Tensor
    valid_groups: torch.Tensor


@torch.no_grad()
def grouped_motion_evidence(
    points: torch.Tensor,
    query_indices: torch.Tensor,
    *,
    translation_scale: float = 0.1,
    relative_scale: float = 0.1,
    assignment_chunk: int = 4096,
) -> MotionEvidence:
    """Return [unknown, co-moving/undecided, relative-change] evidence.

    ``points`` is (B,T,N,3), in the existing query-mesh coordinate system.
    Membership is fixed using nearest query anchors, not recomputed per frame.
    Rank-deficient local rotations are unobserved, not evidence of rigidity.
    """
    if points.ndim != 4 or points.shape[-1] != 3 or points.shape[1] < 2:
        raise ValueError("motion evidence requires (B,T>=2,N,3) points")
    if query_indices.ndim != 2 or query_indices.shape[0] != points.shape[0]:
        raise ValueError("query_indices must have shape (B,Q)")
    if query_indices.dtype != torch.long or query_indices.device != points.device:
        raise ValueError("query_indices must be int64 on the points device")
    if translation_scale <= 0 or relative_scale <= 0 or assignment_chunk < 1:
        raise ValueError("evidence scales and assignment_chunk must be positive")
    frames, count, groups = points.shape[1], points.shape[2], query_indices.shape[1]
    if groups < 1 or groups > count:
        raise ValueError("query anchor count must be between 1 and N")
    states_batch, features_batch, valid_batch = [], [], []
    # Geometry must stay outside ambient BF16 training autocast. Only the
    # cancellation-sensitive pair-distance accumulation uses FP64.
    with torch.autocast(device_type=points.device.type, enabled=False):
        for cloud, indices in zip(points.detach().float(), query_indices):
            query = cloud[0]
            anchors = query.index_select(0, indices)
            anchor_norm = anchors.square().sum(-1).unsqueeze(0)
            labels = torch.empty(count, dtype=torch.long, device=cloud.device)
            for start in range(0, count, assignment_chunk):
                block = query[start:start + assignment_chunk]
                distance = block.square().sum(-1, keepdim=True) + anchor_norm
                distance.addmm_(block, anchors.T, beta=1.0, alpha=-2.0)
                labels[start:start + assignment_chunk] = distance.argmin(-1)
            counts = torch.bincount(labels, minlength=groups)
            denominator = counts.clamp_min(1).to(cloud.dtype)
            centers = cloud.new_zeros(frames, groups, 3)
            centers.index_add_(1, labels, cloud)
            centers /= denominator[None, :, None]
            centered = cloud - centers.index_select(1, labels)
            outer = centered[0][None, :, :, None] * centered[:, :, None, :]
            covariance = cloud.new_zeros(frames, groups, 9)
            covariance.index_add_(1, labels, outer.reshape(frames, count, 9))
            covariance = covariance.reshape(frames, groups, 3, 3)
            covariance /= denominator[None, :, None, None]
            left, singular, right_t = torch.linalg.svd(covariance, full_matrices=False)
            right, left_t = right_t.transpose(-1, -2), left.transpose(-1, -2)
            sign = torch.where(torch.linalg.det(right @ left_t) < 0, -1.0, 1.0)
            correction = torch.ones_like(singular)
            correction[..., 2] = sign
            rotation = (right * correction.unsqueeze(-2)) @ left_t
            rotation[0] = torch.eye(3, device=cloud.device, dtype=cloud.dtype)
            translation = centers - (rotation @ centers[0][None, :, :, None]).squeeze(-1)
            translation[0] = 0
            rank_threshold = torch.maximum(singular[..., 0] * 1e-5, singular.new_tensor(1e-10))
            valid = ((counts[None, :] >= 3) & (singular[..., 1] > rank_threshold)).all(0)
            displacement = torch.linalg.vector_norm(centers - centers[:1], dim=-1)
            cosine = ((rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5).clamp(-1, 1)
            skew = torch.stack((rotation[..., 2, 1] - rotation[..., 1, 2],
                                rotation[..., 0, 2] - rotation[..., 2, 0],
                                rotation[..., 1, 0] - rotation[..., 0, 1]), dim=-1)
            angle = torch.atan2(torch.linalg.vector_norm(skew, dim=-1) * 0.5, cosine)
            o = torch.stack((displacement, angle), dim=-1)[1:].square().mean(0).sqrt()
            # Identical observations carry no motion evidence. Scatter/SVD
            # roundoff must not turn a static clip into a weak positive signal.
            if torch.equal(cloud, cloud[:1].expand_as(cloud)):
                o.zero_()
            descriptor = torch.cat((translation, rotation.reshape(frames, groups, 9) / math.sqrt(3)), -1).double()
            flat = descriptor[1:].permute(1, 0, 2).reshape(groups, -1) / math.sqrt(frames - 1)
            norm = flat.square().sum(-1)
            m_squared = norm[:, None] + norm[None, :]
            m_squared.addmm_(flat, flat.T, beta=1.0, alpha=-2.0)
            m_squared.clamp_min_(0)
            m_squared.diagonal().zero_()
            relative = -torch.expm1(-m_squared.sqrt().float() / relative_scale)
            activity = -torch.expm1(-o[:, 0] / translation_scale - o[:, 1])
            pair_activity = torch.maximum(activity[:, None], activity[None, :])
            states = torch.stack((1 - pair_activity, pair_activity * (1 - relative),
                                  pair_activity * relative), dim=-1)
            pair_valid = valid[:, None] & valid[None, :]
            unknown = states.new_tensor((1.0, 0.0, 0.0))
            states = torch.where(pair_valid[..., None], states, unknown)
            local_activity = -torch.expm1(-o / o.new_tensor((translation_scale, 1.0)))
            local_activity = torch.where(valid[:, None], local_activity, 0.0)
            # Node fusion is a deliberately pooled alternative to pair bias:
            # it retains observation types, but not each neighbor's identity.
            features = torch.cat((local_activity, states.mean(dim=1)), dim=-1)
            states_batch.append(states)
            features_batch.append(features)
            valid_batch.append(valid)
    return MotionEvidence(torch.stack(states_batch), torch.stack(features_batch), torch.stack(valid_batch))


class MotionEvidenceInjection(nn.Module):
    """Zero-initialized pair-score and/or node-feature fusion ablations."""

    def __init__(self, dim: int, heads: int, *, fusion: str = "off", biased_heads: int = 0) -> None:
        super().__init__()
        if fusion not in {"off", "bias", "token", "hybrid"}:
            raise ValueError(f"unsupported motion evidence fusion: {fusion}")
        if fusion in {"bias", "hybrid"}:
            if not 1 <= biased_heads <= heads:
                raise ValueError("bias/hybrid fusion requires 1..heads biased heads")
        elif biased_heads != 0:
            raise ValueError("off/token fusion requires biased_heads=0")
        self.fusion = fusion
        self.biased_heads = int(biased_heads)
        # Explicit zeros avoid changing the baseline initialization RNG stream.
        self.bias_weights = nn.Parameter(torch.zeros(biased_heads, 3)) if biased_heads else None
        self.token_weight = nn.Parameter(torch.zeros(dim, 5)) if fusion in {"token", "hybrid"} else None

    def forward(self, layer: nn.TransformerEncoderLayer, tokens: torch.Tensor, evidence: MotionEvidence) -> torch.Tensor:
        batch, frames, slots, dim = tokens.shape
        anchors = evidence.states.shape[1]
        prefix = slots - anchors
        if prefix < 0 or evidence.states.shape != (batch, anchors, anchors, 3):
            raise ValueError("motion evidence must match the anchor tokens")
        if self.token_weight is not None:
            delta = F.linear(evidence.anchor_features.to(self.token_weight.dtype), self.token_weight)
            tokens = torch.cat((tokens[:, :, :prefix], tokens[:, :, prefix:] + delta[:, None]), dim=2)
        if self.bias_weights is None:
            return layer(tokens.reshape(batch * frames, slots, dim)).reshape(batch, frames, slots, dim)
        if not layer.norm_first or not layer.self_attn.batch_first:
            raise ValueError("motion bias requires the retained batch-first pre-LN layer")
        heads = layer.self_attn.num_heads
        head_dim = dim // heads
        changed_heads = self.biased_heads
        x = tokens.reshape(batch * frames, slots, dim)
        normalized = layer.norm1(x)
        qkv = F.linear(normalized, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias)
        qkv = qkv.reshape(batch, frames, slots, 3, heads, head_dim).permute(3, 0, 1, 4, 2, 5)
        query, key, value = qkv.unbind(0)
        bias = F.linear(evidence.states.to(self.bias_weights.dtype), self.bias_weights).permute(0, 3, 1, 2)
        padded_width = (slots + 7) // 8 * 8
        bias = F.pad(bias, (prefix, padded_width - slots, prefix, 0)).to(query.dtype)[..., :slots]
        dropout = layer.self_attn.dropout if layer.training else 0.0
        changed = torch.stack([
            F.scaled_dot_product_attention(
                query[index, :, :changed_heads], key[index, :, :changed_heads],
                value[index, :, :changed_heads], attn_mask=bias[index:index + 1], dropout_p=dropout)
            for index in range(batch)
        ], dim=0)
        if changed_heads < heads:
            plain = F.scaled_dot_product_attention(
                query[:, :, changed_heads:].reshape(batch * frames, heads - changed_heads, slots, head_dim),
                key[:, :, changed_heads:].reshape(batch * frames, heads - changed_heads, slots, head_dim),
                value[:, :, changed_heads:].reshape(batch * frames, heads - changed_heads, slots, head_dim),
                dropout_p=dropout).reshape(batch, frames, heads - changed_heads, slots, head_dim)
            attended = torch.cat((changed, plain), dim=2)
        else:
            attended = changed
        attended = attended.permute(0, 1, 3, 2, 4).reshape(batch * frames, slots, dim)
        attended = F.linear(attended, layer.self_attn.out_proj.weight, layer.self_attn.out_proj.bias)
        x = x + layer.dropout1(attended)
        feedforward = layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(x)))))
        return (x + layer.dropout2(feedforward)).reshape(batch, frames, slots, dim)
