from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.functional import pad

from .model import DynamicRigConditioner
from .sampling import sample_trackable_surface


GRAMMAR_STATE_FEATURE_DIM = 16


class StaticDynamicConditionFusionBlock(nn.Module):
    """One static-prior-preserving dynamic update block."""

    def __init__(self, dim: int, heads: int = 8, zero_init_update: bool = False) -> None:
        super().__init__()
        self.dim = int(dim)
        self.heads = int(heads)
        self.zero_init_update = bool(zero_init_update)
        self.static_norm = nn.LayerNorm(dim)
        self.dynamic_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.update = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        if zero_init_update:
            nn.init.zeros_(self.update[-1].weight)
            nn.init.zeros_(self.update[-1].bias)

    def reset_parameters(self, *, zero_init_update: bool | None = None) -> None:
        zero = self.zero_init_update if zero_init_update is None else bool(zero_init_update)
        self.static_norm.reset_parameters()
        self.dynamic_norm.reset_parameters()
        self.cross_attn._reset_parameters()
        for module in self.update:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        if zero:
            nn.init.zeros_(self.update[-1].weight)
            nn.init.zeros_(self.update[-1].bias)

    def forward(self, static_cond: torch.Tensor, dynamic_cond: torch.Tensor) -> torch.Tensor:
        dtype = static_cond.dtype
        q = self.static_norm(static_cond.float())
        kv = self.dynamic_norm(dynamic_cond.float())
        attended, _ = self.cross_attn(q, kv, kv, need_weights=False)
        return self.update(attended).to(dtype=dtype)


class StaticDynamicConditionFusion(nn.Module):
    """Fuse dynamic evidence into static-condition token slots.

    This is an optional ablation path.  The current clean route uses dynamic
    condition tokens directly, then appends coarse branch-prior proposal tokens.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        gate_init: float = 0.25,
        zero_init_update: bool = False,
        depth: int = 1,
    ) -> None:
        super().__init__()
        gate_init = min(max(float(gate_init), 1.0e-4), 1.0 - 1.0e-4)
        self.blocks = nn.ModuleList(
            StaticDynamicConditionFusionBlock(dim, heads=heads, zero_init_update=zero_init_update)
            for _ in range(max(int(depth), 1))
        )
        self.gate_logit = nn.Parameter(torch.tensor(np.log(gate_init / (1.0 - gate_init)), dtype=torch.float32))

    def reset_parameters(self, *, gate_init: float = 0.25, zero_init_update: bool = True) -> None:
        gate_init = min(max(float(gate_init), 1.0e-4), 1.0 - 1.0e-4)
        with torch.no_grad():
            self.gate_logit.fill_(float(np.log(gate_init / (1.0 - gate_init))))
        for block in self.blocks:
            block.reset_parameters(zero_init_update=zero_init_update)

    def forward(self, static_cond: torch.Tensor, dynamic_cond: torch.Tensor) -> torch.Tensor:
        if static_cond.shape != dynamic_cond.shape:
            raise ValueError(
                "static and dynamic condition tokens must have the same shape, "
                f"got {tuple(static_cond.shape)} vs {tuple(dynamic_cond.shape)}"
            )
        out = static_cond
        gate = torch.sigmoid(self.gate_logit).to(dtype=static_cond.dtype)
        for block in self.blocks:
            out = out + gate * block(out, dynamic_cond)
        return out


class CoarseBranchPrior(nn.Module):
    """Predict coarse branch proposals from dynamic condition tokens.

    The proposals are prompt tokens and a
    supervised auxiliary structure map: each proposal predicts whether a branch
    action exists, the branch parent/root coordinate, and the first child
    coordinate of that branch.
    """

    def __init__(self, dim: int, proposals: int = 32, heads: int = 8) -> None:
        super().__init__()
        self.dim = int(dim)
        self.proposals = int(proposals)
        self.query = nn.Parameter(torch.randn(self.proposals, dim) * 0.02)
        self.cond_norm = nn.LayerNorm(dim)
        self.query_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, int(heads), batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.token_norm = nn.LayerNorm(dim)
        self.exist_head = nn.Linear(dim, 1)
        self.root_head = nn.Linear(dim, 3)
        self.child_head = nn.Linear(dim, 3)
        self.coord_token_proj = nn.Sequential(
            nn.LayerNorm(7),
            nn.Linear(7, dim),
        )

    def forward(self, cond: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.proposals <= 0:
            raise RuntimeError("CoarseBranchPrior was called with proposals <= 0")
        batch_size = int(cond.shape[0])
        q = self.query.view(1, self.proposals, self.dim).expand(batch_size, -1, -1)
        attended, _ = self.cross_attn(
            self.query_norm(q.float()),
            self.cond_norm(cond.float()),
            self.cond_norm(cond.float()),
            need_weights=False,
        )
        h = q.float() + attended
        h = h + self.ff(h)
        exist_logits = self.exist_head(h).squeeze(-1)
        root_xyz = self.root_head(h)
        child_xyz = self.child_head(h)
        coord_features = torch.cat([exist_logits.sigmoid().unsqueeze(-1), root_xyz, child_xyz], dim=-1)
        tokens = self.token_norm(h + self.coord_token_proj(coord_features))
        return {
            "tokens": tokens.to(dtype=cond.dtype),
            "exist_logits": exist_logits,
            "root_xyz": root_xyz,
            "child_xyz": child_xyz,
        }


class ExplicitTreeDecoder(nn.Module):
    """Tree-state decoder with explicit action, parent pointer, and child xyz.

    This is the model-side replacement path for UniRig's implicit
    `BRANCH + parent_xyz + child_xyz` pointer.  During teacher forcing it sees
    only the already generated tree prefix; during generation it feeds back its
    own predicted joints and parent indices.
    """

    ACTION_EOS = 0
    ACTION_CHILD = 1
    ACTION_BRANCH = 2
    FEATURE_DIM = 16

    def __init__(
        self,
        dim: int,
        depth: int = 4,
        heads: int = 8,
        ff_mult: int = 4,
        topology_mode: str = "geometry",
        coordinate_mode: str = "absolute",
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.topology_mode = str(topology_mode)
        if self.topology_mode not in {"geometry", "topology", "hybrid", "split", "planner", "topomlp"}:
            raise ValueError(f"unknown explicit-tree topology_mode={self.topology_mode!r}")
        self.coordinate_mode = str(coordinate_mode)
        if self.coordinate_mode not in {"absolute", "parent_delta"}:
            raise ValueError(f"unknown explicit-tree coordinate_mode={self.coordinate_mode!r}")
        self.state_mlp = nn.Sequential(
            nn.LayerNorm(self.FEATURE_DIM),
            nn.Linear(self.FEATURE_DIM, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.step_mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=int(heads),
            dim_feedforward=int(dim) * int(ff_mult),
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=int(depth))
        self.out_norm = nn.LayerNorm(dim)
        self.action_head = nn.Linear(dim, 3)
        self.xyz_head = nn.Linear(dim, 3)
        self.parent_query = nn.Linear(dim, dim)
        self.parent_key = nn.Sequential(
            nn.LayerNorm(10),
            nn.Linear(10, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.topology_state_mlp = nn.Sequential(
            nn.LayerNorm(self.FEATURE_DIM),
            nn.Linear(self.FEATURE_DIM, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.topology_step_mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        topology_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=int(heads),
            dim_feedforward=int(dim) * int(ff_mult),
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.topology_decoder = nn.TransformerDecoder(topology_layer, num_layers=int(depth))
        self.topology_out_norm = nn.LayerNorm(dim)
        self.topology_action_head = nn.Linear(dim, 3)
        self.topology_parent_query = nn.Linear(dim, dim)
        self.topology_parent_key = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.topomlp_cond_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.planner_query_norm = nn.LayerNorm(dim)
        self.planner_memory_norm = nn.LayerNorm(dim)
        self.planner_cross_attn = nn.MultiheadAttention(dim, int(heads), batch_first=True)
        self.planner_ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    @staticmethod
    def _depths(parent_values: list[int]) -> list[int]:
        depths: list[int] = []
        for idx, parent in enumerate(parent_values):
            if parent < 0 or parent >= idx:
                depths.append(0)
            else:
                depths.append(depths[parent] + 1)
        return depths

    @staticmethod
    def _norm_index(value: int, denom: float = 256.0) -> float:
        if value < 0:
            return 0.0
        return min(float(value) / float(denom), 1.0)

    @staticmethod
    def _norm_count(value: int, denom: float = 32.0) -> float:
        return min(max(float(value), 0.0) / float(denom), 1.0)

    @staticmethod
    def _previous_child_count(parent_values: list[int], parent: int, end: int) -> int:
        if parent < 0:
            return 0
        return sum(1 for idx in range(max(0, min(end, len(parent_values)))) if parent_values[idx] == parent)

    def _prefix_features(
        self,
        joints: torch.Tensor,
        parents: torch.Tensor,
        joint_count: torch.Tensor,
        steps: int,
        mode: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature_mode = self.topology_mode if mode is None else str(mode)
        if feature_mode in {"hybrid", "split", "planner", "topomlp"}:
            feature_mode = "topology"
        device = joints.device
        batch_size = int(joints.shape[0])
        features = joints.new_zeros((batch_size, steps, self.FEATURE_DIM), dtype=torch.float32)
        valid_steps = torch.zeros((batch_size, steps), device=device, dtype=torch.bool)
        max_joints = int(joints.shape[1])
        for b in range(batch_size):
            count = int(joint_count[b].detach().cpu())
            count = max(0, min(count, max_joints))
            valid_steps[b, : min(steps, count + 1)] = True
            parent_values = [int(x) for x in parents[b, :count].detach().cpu().tolist()]
            depths = self._depths(parent_values)
            for t in range(1, min(steps, count + 1)):
                prev = t - 1
                parent = parent_values[prev] if prev < len(parent_values) else -1
                prev_joint = joints[b, prev].to(dtype=torch.float32)
                if 0 <= parent < count:
                    parent_joint = joints[b, parent].to(dtype=torch.float32)
                else:
                    parent_joint = prev_joint
                delta = prev_joint - parent_joint
                if prev == 0 or parent == prev - 1:
                    prev_action = self.ACTION_CHILD
                else:
                    prev_action = self.ACTION_BRANCH
                action_onehot = torch.zeros((3,), device=device, dtype=torch.float32)
                action_onehot[int(prev_action)] = 1.0
                depth_norm = min(float(depths[prev]) / 32.0, 1.0) if prev < len(depths) else 0.0
                step_norm = min(float(t) / 256.0, 1.0)
                if feature_mode == "topology":
                    parent_depth = depths[parent] if 0 <= parent < len(depths) else 0
                    previous_siblings = self._previous_child_count(parent_values, parent, prev)
                    parent_children = self._previous_child_count(parent_values, parent, t)
                    topological = torch.cat(
                        [
                            torch.tensor(
                                [
                                    self._norm_index(prev),
                                    self._norm_index(parent),
                                    self._norm_count(previous_siblings),
                                    depth_norm,
                                    self._norm_count(parent_depth),
                                    self._norm_count(parent_children),
                                    1.0 if prev == 0 else 0.0,
                                    1.0 if parent == prev - 1 else 0.0,
                                    1.0 if parent == 0 else 0.0,
                                    step_norm,
                                    depth_norm,
                                    1.0,
                                ],
                                device=device,
                                dtype=torch.float32,
                            ),
                            action_onehot,
                            torch.tensor(
                                [1.0 if prev_action == self.ACTION_BRANCH else 0.0],
                                device=device,
                                dtype=torch.float32,
                            ),
                        ],
                        dim=0,
                    )
                    features[b, t] = topological
                else:
                    features[b, t] = torch.cat(
                        [
                            prev_joint,
                            parent_joint,
                            delta,
                            torch.tensor([step_norm, depth_norm, 1.0], device=device, dtype=torch.float32),
                            action_onehot,
                            torch.tensor(
                                [1.0 if prev_action == self.ACTION_BRANCH else 0.0],
                                device=device,
                                dtype=torch.float32,
                            ),
                        ],
                        dim=0,
                    )
        return features, valid_steps

    def _parent_key_features(
        self,
        joints: torch.Tensor,
        parents: torch.Tensor,
        joint_count: torch.Tensor,
        mode: str | None = None,
    ) -> torch.Tensor:
        feature_mode = self.topology_mode if mode is None else str(mode)
        if feature_mode in {"hybrid", "split", "planner", "topomlp"}:
            feature_mode = "topology"
        device = joints.device
        batch_size, max_joints, _ = joints.shape
        features = joints.new_zeros((batch_size, max_joints, 10), dtype=torch.float32)
        for b in range(batch_size):
            count = int(joint_count[b].detach().cpu())
            count = max(0, min(count, max_joints))
            parent_values = [int(x) for x in parents[b, :count].detach().cpu().tolist()]
            depths = self._depths(parent_values)
            for j in range(count):
                parent = parent_values[j]
                joint = joints[b, j].to(dtype=torch.float32)
                if 0 <= parent < count:
                    parent_joint = joints[b, parent].to(dtype=torch.float32)
                else:
                    parent_joint = joint
                depth_norm = min(float(depths[j]) / 32.0, 1.0) if j < len(depths) else 0.0
                if feature_mode == "topology":
                    parent_depth = depths[parent] if 0 <= parent < len(depths) else 0
                    sibling_ordinal = self._previous_child_count(parent_values, parent, j)
                    is_chain = j == 0 or parent == j - 1
                    features[b, j] = torch.tensor(
                        [
                            self._norm_index(j),
                            self._norm_index(parent),
                            depth_norm,
                            1.0 if j == 0 else 0.0,
                            1.0 if parent == 0 else 0.0,
                            1.0 if is_chain else 0.0,
                            0.0 if is_chain else 1.0,
                            self._norm_count(sibling_ordinal),
                            self._norm_count(parent_depth),
                            1.0,
                        ],
                        device=device,
                        dtype=torch.float32,
                    )
                else:
                    features[b, j] = torch.cat(
                        [
                            joint,
                            parent_joint,
                            joint - parent_joint,
                            torch.tensor([depth_norm], device=device, dtype=torch.float32),
                        ],
                        dim=0,
                    )
        return features

    def _parent_base_for_targets(
        self,
        joints: torch.Tensor,
        parents: torch.Tensor,
        steps: int,
        max_count: int,
        like: torch.Tensor,
    ) -> torch.Tensor:
        base = like.new_zeros(like.shape)
        if max_count <= 0:
            return base
        parent_idx = parents[:, :max_count].clamp_min(0)
        gather_idx = parent_idx.unsqueeze(-1).expand(-1, -1, 3)
        parent_xyz = torch.gather(joints[:, :max_count].to(device=like.device, dtype=torch.float32), 1, gather_idx)
        root_mask = parents[:, :max_count].to(device=like.device) < 0
        parent_xyz = parent_xyz.masked_fill(root_mask.unsqueeze(-1), 0.0)
        used = min(int(steps), int(max_count))
        base[:, :used] = parent_xyz[:, :used]
        return base

    def child_xyz_from_step(
        self,
        step_out: dict[str, torch.Tensor | None],
        prefix_joints: torch.Tensor,
        parent: int,
    ) -> torch.Tensor:
        xyz = step_out["xyz"]
        assert isinstance(xyz, torch.Tensor)
        if self.coordinate_mode != "parent_delta":
            return xyz[0].to(dtype=torch.float32)
        delta = step_out.get("xyz_delta", xyz)
        assert isinstance(delta, torch.Tensor)
        base = torch.zeros_like(delta[0], dtype=torch.float32)
        parent_i = int(parent)
        if parent_i >= 0 and prefix_joints.numel() and parent_i < int(prefix_joints.shape[1]):
            base = prefix_joints[0, parent_i].to(device=delta.device, dtype=torch.float32)
        return (base + delta[0].to(dtype=torch.float32)).to(dtype=torch.float32)

    def _planner_hidden(
        self,
        cond: torch.Tensor,
        topo_features: torch.Tensor,
        step_ids: torch.Tensor,
    ) -> torch.Tensor:
        tgt = self.topology_state_mlp(topo_features.to(device=cond.device)) + self.topology_step_mlp(step_ids)
        memory = cond.to(dtype=torch.float32)
        attended, _ = self.planner_cross_attn(
            self.planner_query_norm(tgt),
            self.planner_memory_norm(memory),
            self.planner_memory_norm(memory),
            need_weights=False,
        )
        hidden = tgt + attended
        hidden = hidden + self.planner_ff(hidden)
        return self.topology_out_norm(hidden)

    def forward(
        self,
        cond: torch.Tensor,
        joints: torch.Tensor,
        parents: torch.Tensor,
        joint_count: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        max_count = int(joint_count.detach().cpu().max().item()) if joint_count.numel() else 0
        max_count = max(1, min(max_count, int(joints.shape[1])))
        joints = joints[:, :max_count].to(dtype=torch.float32)
        parents = parents[:, :max_count]
        steps = max_count + 1
        step_ids = torch.arange(steps, device=cond.device, dtype=torch.float32).view(1, steps, 1) / 256.0
        causal = torch.triu(torch.ones((steps, steps), device=cond.device, dtype=torch.bool), diagonal=1)
        if self.topology_mode == "split":
            geo_features, valid_steps = self._prefix_features(joints, parents, joint_count, steps, mode="geometry")
            topo_features, _ = self._prefix_features(joints, parents, joint_count, steps, mode="topology")
            geo_tgt = self.state_mlp(geo_features.to(device=cond.device)) + self.step_mlp(step_ids)
            topo_tgt = self.topology_state_mlp(topo_features.to(device=cond.device)) + self.topology_step_mlp(step_ids)
            geo_hidden = self.decoder(
                tgt=geo_tgt,
                memory=cond.to(dtype=torch.float32),
                tgt_mask=causal,
                tgt_key_padding_mask=~valid_steps.to(device=cond.device),
            )
            topo_hidden = self.topology_decoder(
                tgt=topo_tgt,
                memory=cond.to(dtype=torch.float32),
                tgt_mask=causal,
                tgt_key_padding_mask=~valid_steps.to(device=cond.device),
            )
            hidden = self.topology_out_norm(topo_hidden)
            xyz_hidden = self.out_norm(geo_hidden)
            parent_keys = self.topology_parent_key(hidden[:, 1 : max_count + 1])
            action_logits = self.topology_action_head(hidden)
            xyz = self.xyz_head(xyz_hidden)
            parent_logits = (
                torch.einsum("bsd,bjd->bsj", self.topology_parent_query(hidden), parent_keys)
                / (self.dim ** 0.5)
            )
        elif self.topology_mode == "planner":
            geo_features, valid_steps = self._prefix_features(joints, parents, joint_count, steps, mode="geometry")
            topo_features, _ = self._prefix_features(joints, parents, joint_count, steps, mode="topology")
            geo_tgt = self.state_mlp(geo_features.to(device=cond.device)) + self.step_mlp(step_ids)
            geo_hidden = self.decoder(
                tgt=geo_tgt,
                memory=cond.to(dtype=torch.float32),
                tgt_mask=causal,
                tgt_key_padding_mask=~valid_steps.to(device=cond.device),
            )
            hidden = self._planner_hidden(cond, topo_features, step_ids)
            xyz_hidden = self.out_norm(geo_hidden)
            parent_keys = self.topology_parent_key(hidden[:, 1 : max_count + 1])
            action_logits = self.topology_action_head(hidden)
            xyz = self.xyz_head(xyz_hidden)
            parent_logits = (
                torch.einsum("bsd,bjd->bsj", self.topology_parent_query(hidden), parent_keys)
                / (self.dim ** 0.5)
            )
        elif self.topology_mode == "topomlp":
            geo_features, valid_steps = self._prefix_features(joints, parents, joint_count, steps, mode="geometry")
            topo_features, _ = self._prefix_features(joints, parents, joint_count, steps, mode="topology")
            geo_tgt = self.state_mlp(geo_features.to(device=cond.device)) + self.step_mlp(step_ids)
            geo_hidden = self.decoder(
                tgt=geo_tgt,
                memory=cond.to(dtype=torch.float32),
                tgt_mask=causal,
                tgt_key_padding_mask=~valid_steps.to(device=cond.device),
            )
            cond_token = self.topomlp_cond_mlp(cond.to(dtype=torch.float32).mean(dim=1)).unsqueeze(1)
            hidden = self.topology_state_mlp(topo_features.to(device=cond.device)) + self.topology_step_mlp(step_ids)
            hidden = self.topology_out_norm(hidden + cond_token)
            xyz_hidden = self.out_norm(geo_hidden)
            parent_features = self._parent_key_features(
                joints.to(device=cond.device), parents.to(device=cond.device), joint_count, mode="topology"
            )
            action_logits = self.topology_action_head(hidden)
            xyz = self.xyz_head(xyz_hidden)
            parent_keys = self.parent_key(parent_features)
            parent_logits = torch.einsum("bsd,bjd->bsj", self.parent_query(hidden), parent_keys) / (self.dim ** 0.5)
        elif self.topology_mode == "hybrid":
            geo_features, valid_steps = self._prefix_features(joints, parents, joint_count, steps, mode="geometry")
            topo_features, _ = self._prefix_features(joints, parents, joint_count, steps, mode="topology")
            geo_tgt = self.state_mlp(geo_features.to(device=cond.device)) + self.step_mlp(step_ids)
            topo_tgt = self.state_mlp(topo_features.to(device=cond.device)) + self.step_mlp(step_ids)
            geo_hidden = self.decoder(
                tgt=geo_tgt,
                memory=cond.to(dtype=torch.float32),
                tgt_mask=causal,
                tgt_key_padding_mask=~valid_steps.to(device=cond.device),
            )
            topo_hidden = self.decoder(
                tgt=topo_tgt,
                memory=cond.to(dtype=torch.float32),
                tgt_mask=causal,
                tgt_key_padding_mask=~valid_steps.to(device=cond.device),
            )
            hidden = self.out_norm(topo_hidden)
            xyz_hidden = self.out_norm(geo_hidden)
            parent_features = self._parent_key_features(
                joints.to(device=cond.device), parents.to(device=cond.device), joint_count, mode="topology"
            )
            action_logits = self.action_head(hidden)
            xyz = self.xyz_head(xyz_hidden)
            parent_keys = self.parent_key(parent_features)
            parent_logits = torch.einsum("bsd,bjd->bsj", self.parent_query(hidden), parent_keys) / (self.dim ** 0.5)
        else:
            features, valid_steps = self._prefix_features(joints, parents, joint_count, steps)
            tgt = self.state_mlp(features.to(device=cond.device)) + self.step_mlp(step_ids)
            hidden_raw = self.decoder(
                tgt=tgt,
                memory=cond.to(dtype=torch.float32),
                tgt_mask=causal,
                tgt_key_padding_mask=~valid_steps.to(device=cond.device),
            )
            hidden = self.out_norm(hidden_raw)
            xyz_hidden = hidden
            parent_features = self._parent_key_features(
                joints.to(device=cond.device), parents.to(device=cond.device), joint_count
            )
            action_logits = self.action_head(hidden)
            xyz = self.xyz_head(xyz_hidden)
            parent_keys = self.parent_key(parent_features)
            parent_logits = torch.einsum("bsd,bjd->bsj", self.parent_query(hidden), parent_keys) / (self.dim ** 0.5)
        xyz_delta = xyz
        if self.coordinate_mode == "parent_delta":
            parent_base = self._parent_base_for_targets(
                joints.to(device=cond.device),
                parents.to(device=cond.device),
                steps,
                max_count,
                xyz_delta,
            )
            xyz = parent_base + xyz_delta
        step_index = torch.arange(steps, device=cond.device).view(1, steps, 1)
        joint_index = torch.arange(max_count, device=cond.device).view(1, 1, max_count)
        count_mask = joint_index < joint_count.to(device=cond.device).view(-1, 1, 1)
        parent_allowed = (joint_index < step_index) & count_mask
        parent_logits = parent_logits.masked_fill(~parent_allowed, -torch.finfo(parent_logits.dtype).max)
        return {
            "hidden": hidden,
            "action_logits": action_logits,
            "parent_logits": parent_logits,
            "xyz": xyz,
            "xyz_delta": xyz_delta,
            "valid_steps": valid_steps.to(device=cond.device),
            "max_count": torch.tensor(max_count, device=cond.device, dtype=torch.long),
        }

    def step_from_prefix(
        self,
        cond: torch.Tensor,
        joints: torch.Tensor,
        parents: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        """Predict one explicit-tree step from an already generated prefix."""

        if int(cond.shape[0]) != 1:
            raise ValueError("step_from_prefix expects batch size 1")
        device = cond.device
        count = int(joints.shape[1]) if joints.numel() else 0
        if count > 0:
            joint_tensor = joints.to(device=device, dtype=torch.float32)
            parent_tensor = parents.to(device=device, dtype=torch.long)
            count_tensor = torch.tensor([count], device=device, dtype=torch.long)
        else:
            joint_tensor = torch.zeros((1, 1, 3), device=device, dtype=torch.float32)
            parent_tensor = torch.full((1, 1), -1, device=device, dtype=torch.long)
            count_tensor = torch.zeros((1,), device=device, dtype=torch.long)
        step_ids = torch.arange(count + 1, device=device, dtype=torch.float32).view(1, count + 1, 1) / 256.0
        causal = torch.triu(torch.ones((count + 1, count + 1), device=device, dtype=torch.bool), diagonal=1)
        split_parent_keys = None
        if self.topology_mode == "split":
            geo_features, valid = self._prefix_features(joint_tensor, parent_tensor, count_tensor, count + 1, mode="geometry")
            topo_features, _ = self._prefix_features(
                joint_tensor, parent_tensor, count_tensor, count + 1, mode="topology"
            )
            geo_tgt = self.state_mlp(geo_features.to(device=device)) + self.step_mlp(step_ids)
            topo_tgt = self.topology_state_mlp(topo_features.to(device=device)) + self.topology_step_mlp(step_ids)
            geo_hidden = self.decoder(
                tgt=geo_tgt,
                memory=cond.to(dtype=torch.float32),
                tgt_mask=causal,
                tgt_key_padding_mask=~valid.to(device=device),
            )
            topo_hidden = self.topology_decoder(
                tgt=topo_tgt,
                memory=cond.to(dtype=torch.float32),
                tgt_mask=causal,
                tgt_key_padding_mask=~valid.to(device=device),
            )
            topo_hidden = self.topology_out_norm(topo_hidden)
            row = topo_hidden[:, -1]
            xyz_row = self.out_norm(geo_hidden)[:, -1]
            split_parent_keys = self.topology_parent_key(topo_hidden[:, 1:])
            action_logits = self.topology_action_head(row)
        elif self.topology_mode == "planner":
            geo_features, valid = self._prefix_features(joint_tensor, parent_tensor, count_tensor, count + 1, mode="geometry")
            topo_features, _ = self._prefix_features(
                joint_tensor, parent_tensor, count_tensor, count + 1, mode="topology"
            )
            geo_tgt = self.state_mlp(geo_features.to(device=device)) + self.step_mlp(step_ids)
            geo_hidden = self.decoder(
                tgt=geo_tgt,
                memory=cond.to(dtype=torch.float32),
                tgt_mask=causal,
                tgt_key_padding_mask=~valid.to(device=device),
            )
            topo_hidden = self._planner_hidden(cond, topo_features, step_ids)
            row = topo_hidden[:, -1]
            xyz_row = self.out_norm(geo_hidden)[:, -1]
            split_parent_keys = self.topology_parent_key(topo_hidden[:, 1:])
            action_logits = self.topology_action_head(row)
        elif self.topology_mode == "topomlp":
            geo_features, valid = self._prefix_features(joint_tensor, parent_tensor, count_tensor, count + 1, mode="geometry")
            topo_features, _ = self._prefix_features(
                joint_tensor, parent_tensor, count_tensor, count + 1, mode="topology"
            )
            geo_tgt = self.state_mlp(geo_features.to(device=device)) + self.step_mlp(step_ids)
            geo_hidden = self.decoder(
                tgt=geo_tgt,
                memory=cond.to(dtype=torch.float32),
                tgt_mask=causal,
                tgt_key_padding_mask=~valid.to(device=device),
            )
            cond_token = self.topomlp_cond_mlp(cond.to(dtype=torch.float32).mean(dim=1)).unsqueeze(1)
            topo_hidden = self.topology_state_mlp(topo_features.to(device=device)) + self.topology_step_mlp(step_ids)
            topo_hidden = self.topology_out_norm(topo_hidden + cond_token)
            row = topo_hidden[:, -1]
            xyz_row = self.out_norm(geo_hidden)[:, -1]
            action_logits = self.topology_action_head(row)
        elif self.topology_mode == "hybrid":
            geo_features, valid = self._prefix_features(joint_tensor, parent_tensor, count_tensor, count + 1, mode="geometry")
            topo_features, _ = self._prefix_features(
                joint_tensor, parent_tensor, count_tensor, count + 1, mode="topology"
            )
            geo_tgt = self.state_mlp(geo_features.to(device=device)) + self.step_mlp(step_ids)
            topo_tgt = self.state_mlp(topo_features.to(device=device)) + self.step_mlp(step_ids)
            geo_hidden = self.decoder(
                tgt=geo_tgt,
                memory=cond.to(dtype=torch.float32),
                tgt_mask=causal,
                tgt_key_padding_mask=~valid.to(device=device),
            )
            topo_hidden = self.decoder(
                tgt=topo_tgt,
                memory=cond.to(dtype=torch.float32),
                tgt_mask=causal,
                tgt_key_padding_mask=~valid.to(device=device),
            )
            row = self.out_norm(topo_hidden)[:, -1]
            xyz_row = self.out_norm(geo_hidden)[:, -1]
            action_logits = self.action_head(row)
        else:
            features, valid = self._prefix_features(joint_tensor, parent_tensor, count_tensor, count + 1)
            tgt = self.state_mlp(features.to(device=device)) + self.step_mlp(step_ids)
            hidden = self.decoder(
                tgt=tgt,
                memory=cond.to(dtype=torch.float32),
                tgt_mask=causal,
                tgt_key_padding_mask=~valid.to(device=device),
            )
            row = self.out_norm(hidden)[:, -1]
            xyz_row = row
            action_logits = self.action_head(row)
        xyz = self.xyz_head(xyz_row)
        xyz_delta = xyz
        parent_logits = None
        if count > 0:
            if self.topology_mode in {"split", "planner"}:
                assert isinstance(split_parent_keys, torch.Tensor)
                parent_logits = (
                    torch.einsum("bd,bjd->bj", self.topology_parent_query(row), split_parent_keys)
                    / (self.dim ** 0.5)
                )
            elif self.topology_mode == "topomlp":
                parent_features = self._parent_key_features(joint_tensor, parent_tensor, count_tensor, mode="topology")
                parent_keys = self.parent_key(parent_features)
                parent_logits = torch.einsum("bd,bjd->bj", self.parent_query(row), parent_keys) / (self.dim ** 0.5)
            else:
                parent_mode = "topology" if self.topology_mode == "hybrid" else None
                parent_features = self._parent_key_features(joint_tensor, parent_tensor, count_tensor, mode=parent_mode)
                parent_keys = self.parent_key(parent_features)
                parent_logits = torch.einsum("bd,bjd->bj", self.parent_query(row), parent_keys) / (self.dim ** 0.5)
            valid_parent = torch.arange(count, device=device).view(1, -1) < count
            parent_logits = parent_logits[:, :count].masked_fill(~valid_parent, -torch.finfo(parent_logits.dtype).max)
        return {"action_logits": action_logits, "parent_logits": parent_logits, "xyz": xyz, "xyz_delta": xyz_delta}

    @torch.no_grad()
    def generate(
        self,
        cond: torch.Tensor,
        *,
        max_joints: int = 128,
        min_joints: int = 1,
        joint_count_hint: float | None = None,
        count_guidance_joint_margin: float = 1.0,
        count_guidance_early_eos_penalty: float = 6.0,
        count_guidance_eos_bias: float = 6.0,
    ) -> dict[str, Any]:
        if int(cond.shape[0]) != 1:
            raise ValueError("ExplicitTreeDecoder.generate currently expects batch size 1")
        device = cond.device
        joints: list[torch.Tensor] = []
        parents: list[int] = []
        has_eos = False
        steps = 0
        for step in range(int(max_joints)):
            if joints:
                joint_tensor = torch.stack(joints, dim=0).view(1, len(joints), 3)
                parent_tensor = torch.tensor([parents], device=device, dtype=torch.long)
            else:
                joint_tensor = torch.zeros((1, 0, 3), device=device, dtype=torch.float32)
                parent_tensor = torch.zeros((1, 0), device=device, dtype=torch.long)
            step_out = self.step_from_prefix(cond, joint_tensor, parent_tensor)
            action_logits = step_out["action_logits"]
            assert isinstance(action_logits, torch.Tensor)
            if len(joints) < int(min_joints):
                action_logits[:, self.ACTION_EOS] = -torch.finfo(action_logits.dtype).max
            if joint_count_hint is not None and np.isfinite(float(joint_count_hint)):
                hint = max(float(min_joints), float(joint_count_hint))
                margin = max(0.0, float(count_guidance_joint_margin))
                current = float(len(joints))
                if current + margin < hint:
                    action_logits[:, self.ACTION_EOS] -= float(count_guidance_early_eos_penalty)
                if current >= hint - margin:
                    over = max(0.0, current - (hint - margin))
                    action_logits[:, self.ACTION_EOS] += float(count_guidance_eos_bias) * (1.0 + over)
            if not joints:
                action = self.ACTION_CHILD
            else:
                action = int(action_logits.argmax(dim=-1).item())
            if action == self.ACTION_EOS:
                has_eos = True
                steps = step + 1
                break
            if not joints:
                parent = -1
            elif action == self.ACTION_CHILD:
                parent = len(joints) - 1
            else:
                parent_logits = step_out["parent_logits"]
                assert isinstance(parent_logits, torch.Tensor)
                parent = int(parent_logits.argmax(dim=-1).item())
            child = self.child_xyz_from_step(step_out, joint_tensor, parent)
            joints.append(child)
            parents.append(parent)
            steps = step + 1
        if joints:
            joint_arr = torch.stack(joints, dim=0).detach().cpu().numpy().astype(np.float32)
        else:
            joint_arr = np.zeros((0, 3), dtype=np.float32)
        return {
            "joints": joint_arr,
            "parents": parents,
            "has_eos": bool(has_eos),
            "steps": int(steps),
        }


class DynamicRigUniRigAR(nn.Module):
    """Replace UniRig static condition with motion-aware query-frame tokens."""

    def __init__(
        self,
        unirig_ar: nn.Module,
        conditioner: DynamicRigConditioner,
        tokenizer: Any,
        *,
        num_surface_samples: int = 65536,
        vertex_samples: int = 8192,
        query_tokens: int = 1024,
        surface_no_grad: bool = True,
        latent_align_weight: float = 0.0,
        motion_contrast_weight: float = 0.0,
        motion_contrast_margin: float = 0.05,
        contrast_controls: tuple[str, ...] = ("zero", "reverse"),
        condition_control_ce_weight: float = 0.0,
        condition_control_ce_controls: tuple[str, ...] = ("zero", "shuffle"),
        eos_loss_weight: float = 1.0,
        decision_loss_weight: float = 0.0,
        loop_recovery_loss_weight: float = 0.0,
        loop_recovery_repeats: int = 4,
        prefix_decision_recovery_weight: float = 0.0,
        prefix_decision_recovery_states: int = 4,
        prefix_decision_recovery_variants: int = 1,
        prefix_decision_recovery_jitter: int = 4,
        prefix_token_recovery_weight: float = 0.0,
        prefix_token_recovery_states: int = 4,
        prefix_token_recovery_variants: int = 1,
        prefix_token_recovery_jitter: int = 4,
        prefix_token_recovery_max_rows: int = 4,
        prefix_action_recovery_weight: float = 0.0,
        prefix_action_recovery_states: int = 4,
        prefix_action_recovery_variants: int = 1,
        prefix_action_recovery_jitter: int = 4,
        prefix_action_recovery_max_rows: int = 4,
        generated_prefix_recovery_weight: float = 0.0,
        generated_prefix_recovery_states: int = 4,
        generated_prefix_recovery_max_new_tokens: int = 128,
        generated_prefix_recovery_max_rows: int = 4,
        structure_count_loss_weight: float = 0.0,
        structure_action_loss_weight: float = 0.0,
        condition_fusion: str = "dynamic",
        condition_fusion_heads: int = 8,
        condition_fusion_gate_init: float = 0.25,
        condition_fusion_depth: int = 1,
        condition_static_blend_weight: float = 0.0,
        use_grammar_state_embedding: bool = False,
        use_action_group_bias: bool = False,
        use_condition_action_group_bias: bool = False,
        branch_prior_proposals: int = 0,
        branch_prior_heads: int = 8,
        branch_prior_loss_weight: float = 0.0,
        branch_prior_coord_loss_weight: float = 0.0,
        explicit_tree_loss_weight: float = 0.0,
        explicit_tree_generated_prefix_weight: float = 0.0,
        explicit_tree_generated_prefix_states: int = 4,
        explicit_tree_generated_prefix_max_steps: int = 64,
        explicit_tree_generated_prefix_max_rows: int = 4,
        explicit_tree_oracle_prefix_weight: float = 0.0,
        explicit_tree_oracle_prefix_states: int = 4,
        explicit_tree_oracle_prefix_max_steps: int = 64,
        explicit_tree_oracle_prefix_max_rows: int = 4,
        explicit_tree_prefix_jitter_weight: float = 0.0,
        explicit_tree_prefix_jitter_std: float = 0.0,
        explicit_tree_depth: int = 4,
        explicit_tree_heads: int = 8,
        explicit_tree_topology_mode: str = "geometry",
        explicit_tree_coordinate_mode: str = "absolute",
        explicit_tree_action_eos_loss_weight: float = 1.0,
        explicit_tree_action_child_loss_weight: float = 1.0,
        explicit_tree_action_branch_loss_weight: float = 1.0,
        explicit_tree_xyz_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.unirig_ar = unirig_ar
        self.conditioner = conditioner
        self.tokenizer = tokenizer
        self.num_surface_samples = int(num_surface_samples)
        self.vertex_samples = int(vertex_samples)
        self.query_tokens = int(query_tokens)
        self.surface_no_grad = bool(surface_no_grad)
        self.latent_align_weight = float(latent_align_weight)
        self.motion_contrast_weight = float(motion_contrast_weight)
        self.motion_contrast_margin = float(motion_contrast_margin)
        self.contrast_controls = tuple(contrast_controls)
        self.condition_control_ce_weight = float(condition_control_ce_weight)
        self.condition_control_ce_controls = tuple(condition_control_ce_controls)
        self.eos_loss_weight = float(eos_loss_weight)
        self.decision_loss_weight = float(decision_loss_weight)
        self.loop_recovery_loss_weight = float(loop_recovery_loss_weight)
        self.loop_recovery_repeats = int(loop_recovery_repeats)
        self.prefix_decision_recovery_weight = float(prefix_decision_recovery_weight)
        self.prefix_decision_recovery_states = int(prefix_decision_recovery_states)
        self.prefix_decision_recovery_variants = int(prefix_decision_recovery_variants)
        self.prefix_decision_recovery_jitter = int(prefix_decision_recovery_jitter)
        self.prefix_token_recovery_weight = float(prefix_token_recovery_weight)
        self.prefix_token_recovery_states = int(prefix_token_recovery_states)
        self.prefix_token_recovery_variants = int(prefix_token_recovery_variants)
        self.prefix_token_recovery_jitter = int(prefix_token_recovery_jitter)
        self.prefix_token_recovery_max_rows = int(prefix_token_recovery_max_rows)
        self.prefix_action_recovery_weight = float(prefix_action_recovery_weight)
        self.prefix_action_recovery_states = int(prefix_action_recovery_states)
        self.prefix_action_recovery_variants = int(prefix_action_recovery_variants)
        self.prefix_action_recovery_jitter = int(prefix_action_recovery_jitter)
        self.prefix_action_recovery_max_rows = int(prefix_action_recovery_max_rows)
        self.generated_prefix_recovery_weight = float(generated_prefix_recovery_weight)
        self.generated_prefix_recovery_states = int(generated_prefix_recovery_states)
        self.generated_prefix_recovery_max_new_tokens = int(generated_prefix_recovery_max_new_tokens)
        self.generated_prefix_recovery_max_rows = int(generated_prefix_recovery_max_rows)
        self.structure_count_loss_weight = float(structure_count_loss_weight)
        self.structure_action_loss_weight = float(structure_action_loss_weight)
        self.condition_fusion = str(condition_fusion)
        self.condition_static_blend_weight = float(condition_static_blend_weight)
        self.use_grammar_state_embedding = bool(use_grammar_state_embedding)
        self.use_action_group_bias = bool(use_action_group_bias)
        self.use_condition_action_group_bias = bool(use_condition_action_group_bias)
        self.branch_prior_loss_weight = float(branch_prior_loss_weight)
        self.branch_prior_coord_loss_weight = float(branch_prior_coord_loss_weight)
        self.explicit_tree_loss_weight = float(explicit_tree_loss_weight)
        self.explicit_tree_generated_prefix_weight = float(explicit_tree_generated_prefix_weight)
        self.explicit_tree_generated_prefix_states = int(explicit_tree_generated_prefix_states)
        self.explicit_tree_generated_prefix_max_steps = int(explicit_tree_generated_prefix_max_steps)
        self.explicit_tree_generated_prefix_max_rows = int(explicit_tree_generated_prefix_max_rows)
        self.explicit_tree_oracle_prefix_weight = float(explicit_tree_oracle_prefix_weight)
        self.explicit_tree_oracle_prefix_states = int(explicit_tree_oracle_prefix_states)
        self.explicit_tree_oracle_prefix_max_steps = int(explicit_tree_oracle_prefix_max_steps)
        self.explicit_tree_oracle_prefix_max_rows = int(explicit_tree_oracle_prefix_max_rows)
        self.explicit_tree_prefix_jitter_weight = float(explicit_tree_prefix_jitter_weight)
        self.explicit_tree_prefix_jitter_std = float(explicit_tree_prefix_jitter_std)
        self.explicit_tree_topology_mode = str(explicit_tree_topology_mode)
        self.explicit_tree_coordinate_mode = str(explicit_tree_coordinate_mode)
        self.explicit_tree_action_eos_loss_weight = float(explicit_tree_action_eos_loss_weight)
        self.explicit_tree_action_child_loss_weight = float(explicit_tree_action_child_loss_weight)
        self.explicit_tree_action_branch_loss_weight = float(explicit_tree_action_branch_loss_weight)
        self.explicit_tree_xyz_loss_weight = float(explicit_tree_xyz_loss_weight)
        self.branch_prior = (
            CoarseBranchPrior(unirig_ar.hidden_size, proposals=int(branch_prior_proposals), heads=int(branch_prior_heads))
            if int(branch_prior_proposals) > 0
            else None
        )
        self.explicit_tree_decoder = (
            ExplicitTreeDecoder(
                unirig_ar.hidden_size,
                depth=int(explicit_tree_depth),
                heads=int(explicit_tree_heads),
                topology_mode=self.explicit_tree_topology_mode,
                coordinate_mode=self.explicit_tree_coordinate_mode,
            )
            if (
                self.explicit_tree_loss_weight > 0.0
                or self.explicit_tree_generated_prefix_weight > 0.0
                or self.explicit_tree_oracle_prefix_weight > 0.0
                or self.explicit_tree_prefix_jitter_weight > 0.0
            )
            else None
        )
        if self.condition_fusion not in {"dynamic", "static_blend", "static_cross_attn", "static_cross_attn_zero"}:
            raise ValueError(f"unknown condition_fusion={self.condition_fusion!r}")
        self.condition_fuser = StaticDynamicConditionFusion(
            unirig_ar.hidden_size,
            heads=int(condition_fusion_heads),
            gate_init=float(condition_fusion_gate_init),
            zero_init_update=self.condition_fusion == "static_cross_attn_zero",
            depth=int(condition_fusion_depth),
        )
        if self.condition_fusion in {"dynamic", "static_blend"}:
            for param in self.condition_fuser.parameters():
                param.requires_grad_(False)
        self.structure_count_head = nn.Sequential(
            nn.LayerNorm(unirig_ar.hidden_size),
            nn.Linear(unirig_ar.hidden_size, 2),
        )
        self.structure_action_head = nn.Sequential(
            nn.LayerNorm(unirig_ar.hidden_size),
            nn.Linear(unirig_ar.hidden_size, 4),
        )
        self.grammar_state_proj = nn.Sequential(
            nn.LayerNorm(GRAMMAR_STATE_FEATURE_DIM),
            nn.Linear(GRAMMAR_STATE_FEATURE_DIM, unirig_ar.hidden_size),
        )
        nn.init.zeros_(self.grammar_state_proj[-1].weight)
        nn.init.zeros_(self.grammar_state_proj[-1].bias)
        if not self.use_grammar_state_embedding:
            for param in self.grammar_state_proj.parameters():
                param.requires_grad_(False)
        self.action_group_bias_head = nn.Sequential(
            nn.LayerNorm(unirig_ar.hidden_size),
            nn.Linear(unirig_ar.hidden_size, 4),
        )
        nn.init.zeros_(self.action_group_bias_head[-1].weight)
        nn.init.zeros_(self.action_group_bias_head[-1].bias)
        self.condition_action_group_bias_head = nn.Sequential(
            nn.LayerNorm(unirig_ar.hidden_size * 2),
            nn.Linear(unirig_ar.hidden_size * 2, unirig_ar.hidden_size),
            nn.GELU(),
            nn.Linear(unirig_ar.hidden_size, 4),
        )
        nn.init.zeros_(self.condition_action_group_bias_head[-1].weight)
        nn.init.zeros_(self.condition_action_group_bias_head[-1].bias)
        self.register_buffer("action_group_for_token", self._build_action_group_for_vocab(), persistent=False)
        if not self.use_action_group_bias:
            for param in self.action_group_bias_head.parameters():
                param.requires_grad_(False)
        if not self.use_condition_action_group_bias:
            for param in self.condition_action_group_bias_head.parameters():
                param.requires_grad_(False)
        if self.structure_count_loss_weight <= 0.0:
            for param in self.structure_count_head.parameters():
                param.requires_grad_(False)
        if self.structure_action_loss_weight <= 0.0 and self.prefix_action_recovery_weight <= 0.0:
            for param in self.structure_action_head.parameters():
                param.requires_grad_(False)
        if self.branch_prior is not None and self.branch_prior_loss_weight <= 0.0:
            # Proposal tokens can still condition the AR decoder; the coordinate
            # heads only need gradients when the branch prior loss is enabled.
            for module in (self.branch_prior.exist_head, self.branch_prior.root_head, self.branch_prior.child_head):
                for param in module.parameters():
                    param.requires_grad_(False)

    def _explicit_tree_action_weights(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(
            [
                self.explicit_tree_action_eos_loss_weight,
                self.explicit_tree_action_child_loss_weight,
                self.explicit_tree_action_branch_loss_weight,
            ],
            device=device,
            dtype=torch.float32,
        )

    @property
    def transformer(self) -> nn.Module:
        return self.unirig_ar.transformer

    def _control_sequence(self, sequence: torch.Tensor | None, control: str) -> torch.Tensor | None:
        if sequence is None:
            return None
        if control == "normal":
            return sequence
        frames = sequence.clone()
        if control == "zero":
            frames[:, 1:] = frames[:, :1]
        elif control == "reverse":
            if frames.shape[1] > 2:
                frames[:, 1:] = torch.flip(frames[:, 1:], dims=[1])
        else:
            raise ValueError(f"unknown motion control {control!r}")
        return frames

    def sample_references(self, batch: dict[str, Any]) -> Any:
        frame_vertices = batch["frame_vertices"]
        return sample_trackable_surface(
            frame_vertices[:, 0],
            batch["faces"],
            num_samples=self.num_surface_samples,
            vertex_samples=self.vertex_samples,
            query_tokens=self.query_tokens,
            vertex_counts=batch.get("vertex_count"),
            face_counts=batch.get("face_count"),
        )

    def build_condition(
        self,
        batch: dict[str, Any],
        control: str = "normal",
        refs: Any | None = None,
        return_branch_prior: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        frame_vertices = self._control_sequence(batch["frame_vertices"], control)
        vertex_normals = self._control_sequence(batch.get("vertex_normals"), control)
        face_normals = self._control_sequence(batch.get("face_normals"), control)
        faces = batch["faces"]
        if refs is None:
            refs = self.sample_references(batch)
        dynamic_cond = self.conditioner(
            frame_vertices,
            faces,
            refs,
            vertex_normals=vertex_normals,
            face_normals=face_normals,
        )
        if self.condition_fusion == "dynamic":
            cond = dynamic_cond
        else:
            static_cond = self.build_static_condition(batch).to(device=dynamic_cond.device, dtype=dynamic_cond.dtype)
            if self.condition_fusion == "static_blend":
                weight = min(max(self.condition_static_blend_weight, 0.0), 1.0)
                cond = (1.0 - weight) * dynamic_cond + weight * static_cond
            else:
                cond = self.condition_fuser(static_cond, dynamic_cond)
        branch_prior: dict[str, torch.Tensor] | None = None
        if self.branch_prior is not None:
            branch_prior = self.branch_prior(cond)
            cond = torch.cat([cond, branch_prior["tokens"].to(dtype=cond.dtype)], dim=1)
        if return_branch_prior:
            return cond, branch_prior
        return cond

    @torch.no_grad()
    def build_static_condition(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.unirig_ar.encode_mesh_cond(
            vertices=batch["frame_vertices"][:, 0],
            normals=batch["vertex_normals"][:, 0],
        )

    def _ar_losses(
        self,
        cond: torch.Tensor,
        batch: dict[str, Any],
        *,
        include_loop_recovery: bool = True,
        include_generated_prefix_recovery: bool = True,
    ) -> dict[str, torch.Tensor]:
        cond = cond.to(dtype=self.transformer.dtype)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        batch_size = input_ids.shape[0]

        inputs_embeds = self.token_inputs_embeds(input_ids, attention_mask)
        inputs_embeds = torch.cat([cond, inputs_embeds], dim=1)
        full_attention = pad(attention_mask, (cond.shape[1], 0, 0, 0), value=1.0)

        need_hidden = self.structure_action_loss_weight > 0.0 or self.uses_action_group_bias
        output = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention,
            use_cache=False,
            output_hidden_states=need_hidden,
        )
        logits = output.logits[:, cond.shape[1] :].reshape(batch_size, -1, self.tokenizer.vocab_size)
        token_hidden_full = output.hidden_states[-1][:, cond.shape[1] :] if need_hidden else None
        logits = self.apply_action_group_bias(logits, token_hidden_full, cond)
        logits = logits[:, :-1]

        labels = input_ids[:, 1:].clone()
        labels[attention_mask[:, 1:] == 0] = -100
        eos_mask = labels == self.tokenizer.eos

        if self.eos_loss_weight == 1.0:
            ce_loss = nn.functional.cross_entropy(logits.permute(0, 2, 1), labels)
        else:
            flat_logits = logits.reshape(-1, logits.shape[-1])
            flat_labels = labels.reshape(-1)
            token_loss = nn.functional.cross_entropy(flat_logits, flat_labels, ignore_index=-100, reduction="none")
            valid = flat_labels != -100
            weights = torch.ones_like(token_loss)
            weights[flat_labels == self.tokenizer.eos] = self.eos_loss_weight
            ce_loss = (token_loss[valid] * weights[valid]).sum() / weights[valid].sum().clamp_min(1.0)

        num_discrete = self.tokenizer.num_discrete
        mask = labels < num_discrete
        valid_labels = labels.clamp_min(0)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        dis = torch.arange(num_discrete, device=logits.device).view(1, 1, -1)
        dis = (dis - valid_labels.unsqueeze(2).repeat(1, 1, num_discrete)).to(torch.float32) / num_discrete
        if mask.any():
            dis_loss = (probs[:, :, :num_discrete] * torch.abs(dis))[mask].sum() / 50.0
        else:
            dis_loss = torch.zeros((), device=logits.device, dtype=ce_loss.dtype)

        if eos_mask.any():
            eos_logits = logits[eos_mask]
            eos_targets = labels[eos_mask]
            eos_loss = nn.functional.cross_entropy(eos_logits, eos_targets)
            eos_acc = (eos_logits.argmax(dim=-1) == self.tokenizer.eos).to(torch.float32).mean()
        else:
            eos_loss = torch.zeros((), device=logits.device, dtype=ce_loss.dtype)
            eos_acc = torch.zeros((), device=logits.device, dtype=torch.float32)

        decision_loss, decision_acc, decision_count = self._decision_losses(logits, input_ids, labels)
        prefix_decision_loss, prefix_decision_acc, prefix_decision_count = self._prefix_decision_recovery_loss(
            cond,
            input_ids,
            attention_mask,
        )
        prefix_token_loss, prefix_token_acc, prefix_token_count = self._prefix_token_recovery_loss(
            cond,
            input_ids,
            attention_mask,
        )
        prefix_action_loss, prefix_action_acc, prefix_action_count = self._prefix_action_recovery_loss(
            cond,
            input_ids,
            attention_mask,
        )
        if include_generated_prefix_recovery:
            (
                generated_prefix_loss,
                generated_prefix_acc,
                generated_prefix_count,
                generated_prefix_rollout_count,
                generated_prefix_diverged_count,
                generated_prefix_invalid_target_count,
                generated_prefix_candidate_count,
            ) = self._generated_prefix_recovery_loss(
                cond,
                input_ids,
                attention_mask,
            )
        else:
            generated_prefix_loss = torch.zeros((), device=input_ids.device, dtype=ce_loss.dtype)
            generated_prefix_acc = torch.zeros((), device=input_ids.device, dtype=torch.float32)
            generated_prefix_count = torch.zeros((), device=input_ids.device, dtype=torch.float32)
            generated_prefix_rollout_count = torch.zeros((), device=input_ids.device, dtype=torch.float32)
            generated_prefix_diverged_count = torch.zeros((), device=input_ids.device, dtype=torch.float32)
            generated_prefix_invalid_target_count = torch.zeros((), device=input_ids.device, dtype=torch.float32)
            generated_prefix_candidate_count = torch.zeros((), device=input_ids.device, dtype=torch.float32)
        structure_count_loss, joint_count_mae, branch_count_mae = self._structure_count_losses(
            cond,
            input_ids,
            attention_mask,
        )
        token_hidden = None if token_hidden_full is None else token_hidden_full[:, :-1]
        structure_action_loss, structure_action_acc, structure_action_count = self._structure_action_losses(
            token_hidden,
            input_ids,
            labels,
        )
        loop_loss, loop_acc, loop_count = self._loop_recovery_loss(
            cond,
            input_ids,
            attention_mask,
            include_loop_recovery=include_loop_recovery,
        )
        return {
            "ce_loss": ce_loss,
            "dis_loss": dis_loss,
            "eos_loss": eos_loss,
            "eos_acc": eos_acc,
            "decision_loss": decision_loss,
            "decision_acc": decision_acc,
            "decision_count": decision_count,
            "prefix_decision_recovery_loss": prefix_decision_loss,
            "prefix_decision_recovery_acc": prefix_decision_acc,
            "prefix_decision_recovery_count": prefix_decision_count,
            "prefix_token_recovery_loss": prefix_token_loss,
            "prefix_token_recovery_acc": prefix_token_acc,
            "prefix_token_recovery_count": prefix_token_count,
            "prefix_action_recovery_loss": prefix_action_loss,
            "prefix_action_recovery_acc": prefix_action_acc,
            "prefix_action_recovery_count": prefix_action_count,
            "generated_prefix_recovery_loss": generated_prefix_loss,
            "generated_prefix_recovery_acc": generated_prefix_acc,
            "generated_prefix_recovery_count": generated_prefix_count,
            "generated_prefix_recovery_rollout_count": generated_prefix_rollout_count,
            "generated_prefix_recovery_diverged_count": generated_prefix_diverged_count,
            "generated_prefix_recovery_invalid_target_count": generated_prefix_invalid_target_count,
            "generated_prefix_recovery_candidate_count": generated_prefix_candidate_count,
            "structure_count_loss": structure_count_loss,
            "joint_count_mae": joint_count_mae,
            "branch_count_mae": branch_count_mae,
            "structure_action_loss": structure_action_loss,
            "structure_action_acc": structure_action_acc,
            "structure_action_count": structure_action_count,
            "loop_recovery_loss": loop_loss,
            "loop_recovery_acc": loop_acc,
            "loop_recovery_count": loop_count,
            "loss": ce_loss,
        }

    def _branch_prior_loss(
        self,
        branch_prior: dict[str, torch.Tensor] | None,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.branch_prior_loss_weight <= 0.0 or branch_prior is None:
            device = next(self.parameters()).device
            zero = torch.zeros((), device=device)
            return zero, zero.float(), zero.float()
        target_roots = batch["branch_prior_roots"].to(device=branch_prior["root_xyz"].device, dtype=torch.float32)
        target_children = batch["branch_prior_children"].to(device=branch_prior["child_xyz"].device, dtype=torch.float32)
        target_mask = batch["branch_prior_mask"].to(device=branch_prior["exist_logits"].device, dtype=torch.bool)
        proposals = int(branch_prior["exist_logits"].shape[1])
        target_count = min(int(target_mask.shape[1]), proposals)
        exist_target = branch_prior["exist_logits"].new_zeros(branch_prior["exist_logits"].shape)
        root_target = branch_prior["root_xyz"].new_zeros(branch_prior["root_xyz"].shape)
        child_target = branch_prior["child_xyz"].new_zeros(branch_prior["child_xyz"].shape)
        positive = torch.zeros_like(exist_target, dtype=torch.bool)
        if target_count > 0:
            mask = target_mask[:, :target_count]
            positive[:, :target_count] = mask
            exist_target[:, :target_count] = mask.to(dtype=exist_target.dtype)
            root_target[:, :target_count] = target_roots[:, :target_count]
            child_target[:, :target_count] = target_children[:, :target_count]
        exist_loss = nn.functional.binary_cross_entropy_with_logits(branch_prior["exist_logits"], exist_target)
        if positive.any():
            coord_loss = 0.5 * (
                nn.functional.smooth_l1_loss(branch_prior["root_xyz"][positive], root_target[positive])
                + nn.functional.smooth_l1_loss(branch_prior["child_xyz"][positive], child_target[positive])
            )
            coord_mae = 0.5 * (
                (branch_prior["root_xyz"][positive] - root_target[positive]).abs().mean()
                + (branch_prior["child_xyz"][positive] - child_target[positive]).abs().mean()
            )
        else:
            coord_loss = (
                branch_prior["root_xyz"].sum() * 0.0
                + branch_prior["child_xyz"].sum() * 0.0
            ).to(dtype=exist_loss.dtype)
            coord_mae = torch.zeros((), device=exist_loss.device, dtype=torch.float32)
        pred = branch_prior["exist_logits"].sigmoid() >= 0.5
        exist_acc = (pred == exist_target.to(dtype=torch.bool)).to(dtype=torch.float32).mean()
        loss = exist_loss + self.branch_prior_coord_loss_weight * coord_loss
        return loss.to(dtype=branch_prior["tokens"].dtype), exist_acc, coord_mae.float()

    def _explicit_tree_loss(
        self,
        cond: torch.Tensor,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.explicit_tree_loss_weight <= 0.0 or self.explicit_tree_decoder is None:
            zero = torch.zeros((), device=cond.device, dtype=cond.dtype)
            return zero, {
                "explicit_tree_action_loss": zero,
                "explicit_tree_parent_loss": zero,
                "explicit_tree_xyz_loss": zero,
                "explicit_tree_prefix_jitter_loss": zero,
                "explicit_tree_action_acc": zero.float(),
                "explicit_tree_parent_acc": zero.float(),
                "explicit_tree_xyz_mae": zero.float(),
            }

        target_joints = batch["target_joints"].to(device=cond.device, dtype=torch.float32)
        target_parents = batch["target_parents"].to(device=cond.device, dtype=torch.long)
        joint_count = batch["joint_count"].to(device=cond.device, dtype=torch.long)
        pred = self.explicit_tree_decoder(cond, target_joints, target_parents, joint_count)
        action_logits = pred["action_logits"]
        parent_logits = pred["parent_logits"]
        xyz_pred = pred["xyz"]
        batch_size, steps, _ = action_logits.shape
        max_count = min(steps - 1, int(target_joints.shape[1]))

        action_targets = torch.full((batch_size, steps), -100, device=cond.device, dtype=torch.long)
        parent_targets = torch.full((batch_size, steps), -100, device=cond.device, dtype=torch.long)
        xyz_targets = torch.zeros((batch_size, steps, 3), device=cond.device, dtype=torch.float32)
        xyz_mask = torch.zeros((batch_size, steps), device=cond.device, dtype=torch.bool)

        for b in range(batch_size):
            count = int(joint_count[b].detach().cpu())
            count = max(0, min(count, max_count))
            for t in range(count):
                parent = int(target_parents[b, t].detach().cpu())
                if t == 0 or parent == t - 1:
                    action = ExplicitTreeDecoder.ACTION_CHILD
                else:
                    action = ExplicitTreeDecoder.ACTION_BRANCH
                action_targets[b, t] = int(action)
                xyz_targets[b, t] = target_joints[b, t]
                xyz_mask[b, t] = True
                if t > 0 and parent >= 0:
                    parent_targets[b, t] = int(parent)
            if count < steps:
                action_targets[b, count] = ExplicitTreeDecoder.ACTION_EOS

        action_loss = nn.functional.cross_entropy(
            action_logits.reshape(-1, action_logits.shape[-1]),
            action_targets.reshape(-1),
            ignore_index=-100,
            weight=self._explicit_tree_action_weights(cond.device),
        )
        action_valid = action_targets != -100
        if action_valid.any():
            action_acc = (
                action_logits.argmax(dim=-1)[action_valid] == action_targets[action_valid]
            ).to(dtype=torch.float32).mean()
        else:
            action_acc = torch.zeros((), device=cond.device, dtype=torch.float32)

        parent_mask = parent_targets >= 0
        if parent_mask.any():
            parent_loss = nn.functional.cross_entropy(parent_logits[parent_mask], parent_targets[parent_mask])
            parent_acc = (
                parent_logits[parent_mask].argmax(dim=-1) == parent_targets[parent_mask]
            ).to(dtype=torch.float32).mean()
        else:
            parent_loss = (parent_logits.sum() * 0.0).to(dtype=action_loss.dtype)
            parent_acc = torch.zeros((), device=cond.device, dtype=torch.float32)

        if xyz_mask.any():
            xyz_loss = nn.functional.smooth_l1_loss(xyz_pred[xyz_mask], xyz_targets[xyz_mask])
            xyz_mae = (xyz_pred[xyz_mask] - xyz_targets[xyz_mask]).abs().mean()
        else:
            xyz_loss = (xyz_pred.sum() * 0.0).to(dtype=action_loss.dtype)
            xyz_mae = torch.zeros((), device=cond.device, dtype=torch.float32)
        total = action_loss + parent_loss + self.explicit_tree_xyz_loss_weight * xyz_loss
        jitter_loss = torch.zeros((), device=cond.device, dtype=action_loss.dtype)
        if self.explicit_tree_prefix_jitter_weight > 0.0 and self.explicit_tree_prefix_jitter_std > 0.0:
            count_mask = (
                torch.arange(target_joints.shape[1], device=cond.device).view(1, -1)
                < joint_count.to(device=cond.device).view(-1, 1)
            )
            noise = torch.randn_like(target_joints) * float(self.explicit_tree_prefix_jitter_std)
            noisy_joints = target_joints + noise * count_mask.unsqueeze(-1).to(dtype=target_joints.dtype)
            jitter_pred = self.explicit_tree_decoder(cond, noisy_joints, target_parents, joint_count)
            jitter_action_loss = nn.functional.cross_entropy(
                jitter_pred["action_logits"].reshape(-1, jitter_pred["action_logits"].shape[-1]),
                action_targets.reshape(-1),
                ignore_index=-100,
                weight=self._explicit_tree_action_weights(cond.device),
            )
            if parent_mask.any():
                jitter_parent_loss = nn.functional.cross_entropy(
                    jitter_pred["parent_logits"][parent_mask],
                    parent_targets[parent_mask],
                )
            else:
                jitter_parent_loss = (jitter_pred["parent_logits"].sum() * 0.0).to(dtype=jitter_action_loss.dtype)
            if xyz_mask.any():
                jitter_xyz_loss = nn.functional.smooth_l1_loss(jitter_pred["xyz"][xyz_mask], xyz_targets[xyz_mask])
            else:
                jitter_xyz_loss = (jitter_pred["xyz"].sum() * 0.0).to(dtype=jitter_action_loss.dtype)
            jitter_loss = jitter_action_loss + jitter_parent_loss + self.explicit_tree_xyz_loss_weight * jitter_xyz_loss
            total = total + self.explicit_tree_prefix_jitter_weight * jitter_loss
        return total.to(dtype=cond.dtype), {
            "explicit_tree_action_loss": action_loss.to(dtype=cond.dtype),
            "explicit_tree_parent_loss": parent_loss.to(dtype=cond.dtype),
            "explicit_tree_xyz_loss": xyz_loss.to(dtype=cond.dtype),
            "explicit_tree_prefix_jitter_loss": jitter_loss.to(dtype=cond.dtype),
            "explicit_tree_action_acc": action_acc,
            "explicit_tree_parent_acc": parent_acc,
            "explicit_tree_xyz_mae": xyz_mae.float(),
        }

    def _explicit_tree_generated_prefix_loss(
        self,
        cond: torch.Tensor,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zero = torch.zeros((), device=cond.device, dtype=cond.dtype)
        if (
            self.explicit_tree_generated_prefix_weight <= 0.0
            or self.explicit_tree_decoder is None
            or self.explicit_tree_generated_prefix_states <= 0
            or self.explicit_tree_generated_prefix_max_steps <= 0
        ):
            return zero, {
                "explicit_tree_generated_prefix_loss": zero,
                "explicit_tree_generated_prefix_action_acc": zero.float(),
                "explicit_tree_generated_prefix_parent_acc": zero.float(),
                "explicit_tree_generated_prefix_xyz_mae": zero.float(),
                "explicit_tree_generated_prefix_count": zero.float(),
            }

        target_joints = batch["target_joints"].to(device=cond.device, dtype=torch.float32)
        target_parents = batch["target_parents"].to(device=cond.device, dtype=torch.long)
        joint_count = batch["joint_count"].to(device=cond.device, dtype=torch.long)
        max_rows = int(self.explicit_tree_generated_prefix_max_rows)
        variants: list[tuple[int, torch.Tensor, torch.Tensor, int, int, torch.Tensor | None]] = []

        with torch.no_grad():
            for batch_idx in range(int(cond.shape[0])):
                count = int(joint_count[batch_idx].detach().cpu())
                count = max(0, min(count, int(target_joints.shape[1])))
                cond_row = cond[batch_idx : batch_idx + 1]
                gen_joints: list[torch.Tensor] = []
                gen_parents: list[int] = []
                collected: list[tuple[int, torch.Tensor, torch.Tensor, int, int, torch.Tensor | None]] = []
                horizon = min(int(self.explicit_tree_generated_prefix_max_steps), max(count + 1, 1))
                for _ in range(horizon):
                    prefix_count = len(gen_joints)
                    if prefix_count > count:
                        break
                    if gen_joints:
                        prefix_joints = torch.stack(gen_joints, dim=0).view(1, prefix_count, 3).detach()
                        prefix_parents = torch.tensor([gen_parents], device=cond.device, dtype=torch.long)
                    else:
                        prefix_joints = torch.zeros((1, 0, 3), device=cond.device, dtype=torch.float32)
                        prefix_parents = torch.zeros((1, 0), device=cond.device, dtype=torch.long)

                    if prefix_count == count:
                        target_action = ExplicitTreeDecoder.ACTION_EOS
                        target_parent = -100
                        target_xyz = None
                    else:
                        target_parent = int(target_parents[batch_idx, prefix_count].detach().cpu())
                        if prefix_count == 0 or target_parent == prefix_count - 1:
                            target_action = ExplicitTreeDecoder.ACTION_CHILD
                        else:
                            target_action = ExplicitTreeDecoder.ACTION_BRANCH
                        target_xyz = target_joints[batch_idx, prefix_count].detach()
                    collected.append(
                        (
                            batch_idx,
                            prefix_joints.detach(),
                            prefix_parents.detach(),
                            int(target_action),
                            int(target_parent),
                            None if target_xyz is None else target_xyz.detach(),
                        )
                    )

                    step_out = self.explicit_tree_decoder.step_from_prefix(cond_row, prefix_joints, prefix_parents)
                    action_logits = step_out["action_logits"]
                    assert isinstance(action_logits, torch.Tensor)
                    if prefix_count == 0:
                        pred_action = ExplicitTreeDecoder.ACTION_CHILD
                    else:
                        pred_action = int(action_logits.argmax(dim=-1).item())
                    if pred_action == ExplicitTreeDecoder.ACTION_EOS:
                        break
                    if prefix_count == 0:
                        pred_parent = -1
                    elif pred_action == ExplicitTreeDecoder.ACTION_CHILD:
                        pred_parent = prefix_count - 1
                    else:
                        parent_logits = step_out["parent_logits"]
                        if not isinstance(parent_logits, torch.Tensor):
                            break
                        pred_parent = int(parent_logits.argmax(dim=-1).item())
                    child = self.explicit_tree_decoder.child_xyz_from_step(
                        step_out,
                        prefix_joints.to(device=cond.device, dtype=torch.float32),
                        int(pred_parent),
                    ).detach()
                    gen_joints.append(child)
                    gen_parents.append(int(pred_parent))

                if len(collected) > self.explicit_tree_generated_prefix_states:
                    idxs = np.linspace(0, len(collected) - 1, self.explicit_tree_generated_prefix_states)
                    collected = [collected[int(round(x))] for x in idxs]
                variants.extend(collected)

        if max_rows > 0 and len(variants) > max_rows:
            idxs = np.linspace(0, len(variants) - 1, max_rows)
            variants = [variants[int(round(x))] for x in idxs]
        if not variants:
            return zero, {
                "explicit_tree_generated_prefix_loss": zero,
                "explicit_tree_generated_prefix_action_acc": zero.float(),
                "explicit_tree_generated_prefix_parent_acc": zero.float(),
                "explicit_tree_generated_prefix_xyz_mae": zero.float(),
                "explicit_tree_generated_prefix_count": zero.float(),
            }

        action_logits_list: list[torch.Tensor] = []
        action_targets: list[int] = []
        parent_losses: list[torch.Tensor] = []
        parent_correct: list[torch.Tensor] = []
        xyz_preds: list[torch.Tensor] = []
        xyz_targets: list[torch.Tensor] = []
        for batch_idx, prefix_joints, prefix_parents, target_action, target_parent, target_xyz in variants:
            cond_row = cond[batch_idx : batch_idx + 1]
            step_out = self.explicit_tree_decoder.step_from_prefix(cond_row, prefix_joints, prefix_parents)
            action_logits = step_out["action_logits"]
            assert isinstance(action_logits, torch.Tensor)
            action_logits_list.append(action_logits.squeeze(0))
            action_targets.append(int(target_action))
            if target_action != ExplicitTreeDecoder.ACTION_EOS and target_xyz is not None:
                xyz_pred = self.explicit_tree_decoder.child_xyz_from_step(
                    step_out,
                    prefix_joints.to(device=cond.device, dtype=torch.float32),
                    int(target_parent),
                )
                xyz_preds.append(xyz_pred)
                xyz_targets.append(target_xyz.to(device=cond.device, dtype=torch.float32))
                parent_logits = step_out["parent_logits"]
                if target_parent >= 0 and isinstance(parent_logits, torch.Tensor):
                    parent_logits_row = parent_logits.squeeze(0)
                    if int(target_parent) < int(parent_logits_row.shape[-1]):
                        parent_target_tensor = torch.tensor(
                            [int(target_parent)], device=cond.device, dtype=torch.long
                        )
                        parent_losses.append(
                            nn.functional.cross_entropy(
                                parent_logits_row.view(1, -1).float(), parent_target_tensor
                            )
                        )
                        parent_correct.append(
                            (parent_logits_row.argmax(dim=-1, keepdim=True) == parent_target_tensor)
                            .to(dtype=torch.float32)
                            .mean()
                        )

        action_logits_stacked = torch.stack(action_logits_list, dim=0)
        action_target_tensor = torch.tensor(action_targets, device=cond.device, dtype=torch.long)
        action_loss = nn.functional.cross_entropy(
            action_logits_stacked.float(),
            action_target_tensor,
            weight=self._explicit_tree_action_weights(cond.device),
        )
        action_acc = (action_logits_stacked.argmax(dim=-1) == action_target_tensor).to(dtype=torch.float32).mean()

        if parent_losses:
            parent_loss = torch.stack(parent_losses, dim=0).mean()
            parent_acc = torch.stack(parent_correct, dim=0).mean()
        else:
            parent_loss = (action_logits_stacked.sum() * 0.0).to(dtype=action_loss.dtype)
            parent_acc = torch.zeros((), device=cond.device, dtype=torch.float32)

        if xyz_preds:
            xyz_pred_tensor = torch.stack(xyz_preds, dim=0)
            xyz_target_tensor = torch.stack(xyz_targets, dim=0)
            xyz_loss = nn.functional.smooth_l1_loss(xyz_pred_tensor, xyz_target_tensor)
            xyz_mae = (xyz_pred_tensor - xyz_target_tensor).abs().mean()
        else:
            xyz_loss = (action_logits_stacked.sum() * 0.0).to(dtype=action_loss.dtype)
            xyz_mae = torch.zeros((), device=cond.device, dtype=torch.float32)

        total = action_loss + parent_loss + self.explicit_tree_xyz_loss_weight * xyz_loss
        return total.to(dtype=cond.dtype), {
            "explicit_tree_generated_prefix_loss": total.to(dtype=cond.dtype),
            "explicit_tree_generated_prefix_action_acc": action_acc,
            "explicit_tree_generated_prefix_parent_acc": parent_acc,
            "explicit_tree_generated_prefix_xyz_mae": xyz_mae.float(),
            "explicit_tree_generated_prefix_count": torch.tensor(float(len(variants)), device=cond.device),
        }

    def _explicit_tree_oracle_prefix_loss(
        self,
        cond: torch.Tensor,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zero = torch.zeros((), device=cond.device, dtype=cond.dtype)
        if (
            self.explicit_tree_oracle_prefix_weight <= 0.0
            or self.explicit_tree_decoder is None
            or self.explicit_tree_oracle_prefix_states <= 0
            or self.explicit_tree_oracle_prefix_max_steps <= 0
        ):
            return zero, {
                "explicit_tree_oracle_prefix_loss": zero,
                "explicit_tree_oracle_prefix_action_acc": zero.float(),
                "explicit_tree_oracle_prefix_parent_acc": zero.float(),
                "explicit_tree_oracle_prefix_xyz_mae": zero.float(),
                "explicit_tree_oracle_prefix_count": zero.float(),
            }

        target_joints = batch["target_joints"].to(device=cond.device, dtype=torch.float32)
        target_parents = batch["target_parents"].to(device=cond.device, dtype=torch.long)
        joint_count = batch["joint_count"].to(device=cond.device, dtype=torch.long)
        variants: list[tuple[int, torch.Tensor, torch.Tensor, int, int, torch.Tensor | None]] = []

        with torch.no_grad():
            for batch_idx in range(int(cond.shape[0])):
                count = int(joint_count[batch_idx].detach().cpu())
                count = max(0, min(count, int(target_joints.shape[1])))
                cond_row = cond[batch_idx : batch_idx + 1]
                gen_joints: list[torch.Tensor] = []
                gen_parents: list[int] = []
                collected: list[tuple[int, torch.Tensor, torch.Tensor, int, int, torch.Tensor | None]] = []
                horizon = min(int(self.explicit_tree_oracle_prefix_max_steps), max(count + 1, 1))
                for _ in range(horizon):
                    prefix_count = len(gen_joints)
                    if gen_joints:
                        prefix_joints = torch.stack(gen_joints, dim=0).view(1, prefix_count, 3).detach()
                        prefix_parents = torch.tensor([gen_parents], device=cond.device, dtype=torch.long)
                    else:
                        prefix_joints = torch.zeros((1, 0, 3), device=cond.device, dtype=torch.float32)
                        prefix_parents = torch.zeros((1, 0), device=cond.device, dtype=torch.long)

                    if prefix_count == count:
                        collected.append(
                            (
                                batch_idx,
                                prefix_joints.detach(),
                                prefix_parents.detach(),
                                ExplicitTreeDecoder.ACTION_EOS,
                                -100,
                                None,
                            )
                        )
                        break
                    target_parent = int(target_parents[batch_idx, prefix_count].detach().cpu())
                    if prefix_count == 0 or target_parent == prefix_count - 1:
                        target_action = ExplicitTreeDecoder.ACTION_CHILD
                    else:
                        target_action = ExplicitTreeDecoder.ACTION_BRANCH
                    target_xyz = target_joints[batch_idx, prefix_count].detach()
                    collected.append(
                        (
                            batch_idx,
                            prefix_joints.detach(),
                            prefix_parents.detach(),
                            int(target_action),
                            int(target_parent),
                            target_xyz.detach(),
                        )
                    )

                    step_out = self.explicit_tree_decoder.step_from_prefix(cond_row, prefix_joints, prefix_parents)
                    child = self.explicit_tree_decoder.child_xyz_from_step(
                        step_out,
                        prefix_joints.to(device=cond.device, dtype=torch.float32),
                        int(target_parent),
                    ).detach()
                    gen_joints.append(child)
                    gen_parents.append(int(target_parent))

                if len(collected) > self.explicit_tree_oracle_prefix_states:
                    idxs = np.linspace(0, len(collected) - 1, self.explicit_tree_oracle_prefix_states)
                    collected = [collected[int(round(x))] for x in idxs]
                variants.extend(collected)

        max_rows = int(self.explicit_tree_oracle_prefix_max_rows)
        if max_rows > 0 and len(variants) > max_rows:
            idxs = np.linspace(0, len(variants) - 1, max_rows)
            variants = [variants[int(round(x))] for x in idxs]
        if not variants:
            return zero, {
                "explicit_tree_oracle_prefix_loss": zero,
                "explicit_tree_oracle_prefix_action_acc": zero.float(),
                "explicit_tree_oracle_prefix_parent_acc": zero.float(),
                "explicit_tree_oracle_prefix_xyz_mae": zero.float(),
                "explicit_tree_oracle_prefix_count": zero.float(),
            }

        action_logits_list: list[torch.Tensor] = []
        action_targets: list[int] = []
        parent_losses: list[torch.Tensor] = []
        parent_correct: list[torch.Tensor] = []
        xyz_preds: list[torch.Tensor] = []
        xyz_targets: list[torch.Tensor] = []
        for batch_idx, prefix_joints, prefix_parents, target_action, target_parent, target_xyz in variants:
            cond_row = cond[batch_idx : batch_idx + 1]
            step_out = self.explicit_tree_decoder.step_from_prefix(cond_row, prefix_joints, prefix_parents)
            action_logits = step_out["action_logits"]
            assert isinstance(action_logits, torch.Tensor)
            action_logits_list.append(action_logits.squeeze(0))
            action_targets.append(int(target_action))
            if target_action != ExplicitTreeDecoder.ACTION_EOS and target_xyz is not None:
                xyz_pred = self.explicit_tree_decoder.child_xyz_from_step(
                    step_out,
                    prefix_joints.to(device=cond.device, dtype=torch.float32),
                    int(target_parent),
                )
                xyz_preds.append(xyz_pred)
                xyz_targets.append(target_xyz.to(device=cond.device, dtype=torch.float32))
                parent_logits = step_out["parent_logits"]
                if target_parent >= 0 and isinstance(parent_logits, torch.Tensor):
                    parent_logits_row = parent_logits.squeeze(0)
                    if int(target_parent) < int(parent_logits_row.shape[-1]):
                        parent_target_tensor = torch.tensor(
                            [int(target_parent)], device=cond.device, dtype=torch.long
                        )
                        parent_losses.append(
                            nn.functional.cross_entropy(
                                parent_logits_row.view(1, -1).float(), parent_target_tensor
                            )
                        )
                        parent_correct.append(
                            (parent_logits_row.argmax(dim=-1, keepdim=True) == parent_target_tensor)
                            .to(dtype=torch.float32)
                            .mean()
                        )

        action_logits_stacked = torch.stack(action_logits_list, dim=0)
        action_target_tensor = torch.tensor(action_targets, device=cond.device, dtype=torch.long)
        action_loss = nn.functional.cross_entropy(
            action_logits_stacked.float(),
            action_target_tensor,
            weight=self._explicit_tree_action_weights(cond.device),
        )
        action_acc = (action_logits_stacked.argmax(dim=-1) == action_target_tensor).to(dtype=torch.float32).mean()

        if parent_losses:
            parent_loss = torch.stack(parent_losses, dim=0).mean()
            parent_acc = torch.stack(parent_correct, dim=0).mean()
        else:
            parent_loss = (action_logits_stacked.sum() * 0.0).to(dtype=action_loss.dtype)
            parent_acc = torch.zeros((), device=cond.device, dtype=torch.float32)

        if xyz_preds:
            xyz_pred_tensor = torch.stack(xyz_preds, dim=0)
            xyz_target_tensor = torch.stack(xyz_targets, dim=0)
            xyz_loss = nn.functional.smooth_l1_loss(xyz_pred_tensor, xyz_target_tensor)
            xyz_mae = (xyz_pred_tensor - xyz_target_tensor).abs().mean()
        else:
            xyz_loss = (action_logits_stacked.sum() * 0.0).to(dtype=action_loss.dtype)
            xyz_mae = torch.zeros((), device=cond.device, dtype=torch.float32)

        total = action_loss + parent_loss + self.explicit_tree_xyz_loss_weight * xyz_loss
        return total.to(dtype=cond.dtype), {
            "explicit_tree_oracle_prefix_loss": total.to(dtype=cond.dtype),
            "explicit_tree_oracle_prefix_action_acc": action_acc,
            "explicit_tree_oracle_prefix_parent_acc": parent_acc,
            "explicit_tree_oracle_prefix_xyz_mae": xyz_mae.float(),
            "explicit_tree_oracle_prefix_count": torch.tensor(float(len(variants)), device=cond.device),
        }

    @torch.no_grad()
    def generate_explicit_tree(
        self,
        batch: dict[str, Any],
        *,
        max_joints: int = 128,
        min_joints: int = 1,
        count_guidance: str = "none",
        count_guidance_kwargs: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        if self.explicit_tree_decoder is None:
            raise RuntimeError("explicit tree decoder is not enabled for this checkpoint")
        refs = self.sample_references(batch)
        cond = self.build_condition(batch, refs=refs)
        joint_count_hint: float | None = None
        if count_guidance != "none":
            if count_guidance == "predicted":
                joint_count_hint = float(self.predict_structure_counts(cond).detach().float().cpu()[0, 0])
            elif count_guidance == "oracle":
                joint_count_hint = float(batch["joint_count"][0].detach().cpu())
            else:
                raise ValueError(f"unknown count guidance mode {count_guidance!r}")
        kwargs = count_guidance_kwargs or {}
        return self.explicit_tree_decoder.generate(
            cond,
            max_joints=max_joints,
            min_joints=min_joints,
            joint_count_hint=joint_count_hint,
            count_guidance_joint_margin=float(kwargs.get("joint_margin", 1.0)),
            count_guidance_early_eos_penalty=float(kwargs.get("early_eos_penalty", 6.0)),
            count_guidance_eos_bias=float(kwargs.get("eos_bias", 6.0)),
        )

    def predict_structure_counts(self, cond: torch.Tensor) -> torch.Tensor:
        """Predict global joint/branch counts from dynamic condition tokens."""

        pooled = cond.to(dtype=torch.float32).mean(dim=1)
        log_counts = self.structure_count_head(pooled)
        return torch.expm1(log_counts).clamp_min(0.0)

    def _structure_count_losses(
        self,
        cond: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.structure_count_loss_weight <= 0.0:
            zero = torch.zeros((), device=cond.device, dtype=cond.dtype)
            return zero, zero.float(), zero.float()
        targets = self._target_structure_counts(input_ids, attention_mask).to(device=cond.device, dtype=torch.float32)
        pred_log = self.structure_count_head(cond.to(dtype=torch.float32).mean(dim=1))
        target_log = torch.log1p(targets)
        loss = nn.functional.smooth_l1_loss(pred_log, target_log)
        pred_counts = torch.expm1(pred_log).clamp_min(0.0)
        mae = (pred_counts - targets).abs().mean(dim=0)
        return loss.to(dtype=cond.dtype), mae[0], mae[1]

    def _target_structure_counts(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        counts: list[tuple[float, float]] = []
        branch = int(getattr(self.tokenizer, "token_id_branch"))
        eos = int(self.tokenizer.eos)
        for batch_idx in range(input_ids.shape[0]):
            valid_ids = input_ids[batch_idx][attention_mask[batch_idx].to(torch.bool)]
            ids = [int(x) for x in valid_ids.detach().cpu().tolist()]
            if eos in ids:
                ids = ids[: ids.index(eos) + 1]
            branch_count = float(sum(1 for x in ids if x == branch))
            try:
                out = self.tokenizer.detokenize(np.asarray(ids, dtype=np.int64))
                joint_count = float(len(out.joints))
            except Exception:
                joint_count = float(self._count_completed_joints(ids))
            counts.append((joint_count, branch_count))
        return torch.tensor(counts, device=input_ids.device, dtype=torch.float32)

    def _build_action_group_for_vocab(self) -> torch.Tensor:
        num_discrete = int(self.tokenizer.num_discrete)
        eos = int(self.tokenizer.eos)
        branch = int(getattr(self.tokenizer, "token_id_branch"))
        groups = torch.full((int(self.tokenizer.vocab_size),), 3, dtype=torch.long)
        groups[:num_discrete] = 2
        if 0 <= branch < groups.numel():
            groups[branch] = 1
        if 0 <= eos < groups.numel():
            groups[eos] = 0
        return groups

    @property
    def uses_action_group_bias(self) -> bool:
        return self.use_action_group_bias or self.use_condition_action_group_bias

    def apply_action_group_bias(
        self,
        logits: torch.Tensor,
        hidden: torch.Tensor | None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.uses_action_group_bias or hidden is None:
            return logits
        hidden_f = hidden.to(dtype=torch.float32)
        group_logits = logits.new_zeros((*hidden.shape[:-1], 4), dtype=torch.float32)
        if self.use_action_group_bias:
            group_logits = group_logits + self.action_group_bias_head(hidden_f)
        if self.use_condition_action_group_bias:
            if cond is None:
                raise ValueError("cond is required when use_condition_action_group_bias=True")
            pooled = cond.to(dtype=torch.float32).mean(dim=1)
            while pooled.ndim < hidden_f.ndim:
                pooled = pooled.unsqueeze(1)
            pooled = pooled.expand(*hidden_f.shape[:-1], pooled.shape[-1])
            group_logits = group_logits + self.condition_action_group_bias_head(torch.cat([hidden_f, pooled], dim=-1))
        token_groups = self.action_group_for_token.to(device=logits.device)
        token_bias = group_logits[..., token_groups].to(dtype=logits.dtype)
        return logits + token_bias

    def apply_action_group_bias_row(
        self,
        logits: torch.Tensor,
        hidden: torch.Tensor | None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden is None:
            return logits
        return self.apply_action_group_bias(logits.unsqueeze(1), hidden.unsqueeze(1), cond).squeeze(1)

    def token_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeds = self.transformer.get_input_embeddings()(input_ids).to(dtype=self.transformer.dtype)
        if not self.use_grammar_state_embedding:
            return embeds
        state = self._grammar_state_features(input_ids, attention_mask)
        state_embed = self.grammar_state_proj(state.to(dtype=torch.float32)).to(dtype=embeds.dtype)
        return embeds + state_embed

    def next_token_embed_with_state(self, full_prefix_ids: list[int], device: torch.device) -> torch.Tensor:
        ids = torch.tensor([full_prefix_ids], device=device, dtype=torch.long)
        mask = torch.ones_like(ids, dtype=torch.long)
        return self.token_inputs_embeds(ids, mask)[:, -1:, :]

    def _grammar_state_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Parser-state features available under teacher forcing and inference.

        These features depend only on the already consumed prefix and UniRig's
        grammar.  They are not target ids, corruption labels, GT joint slots, or
        any future information.
        """

        ids_cpu = input_ids.detach().cpu()
        if attention_mask is None:
            valid_cpu = torch.ones_like(ids_cpu, dtype=torch.bool)
        else:
            valid_cpu = attention_mask.detach().cpu().to(torch.bool)
        features_cpu = torch.zeros(
            (*input_ids.shape, GRAMMAR_STATE_FEATURE_DIM),
            dtype=torch.float32,
        )

        num_discrete = int(self.tokenizer.num_discrete)
        eos = int(self.tokenizer.eos)
        branch = int(getattr(self.tokenizer, "token_id_branch"))

        for batch_idx in range(input_ids.shape[0]):
            valid_len = int(valid_cpu[batch_idx].sum().item())
            if valid_len <= 0:
                continue
            ids = [int(x) for x in ids_cpu[batch_idx, :valid_len].tolist()]
            completed_joints = 0
            branch_count = 0
            coord_count = 0
            coord_run = 0
            coords_remaining = 0
            last_branch_offset = 1_000_000
            for token_idx in range(valid_len):
                token = int(ids[token_idx])
                if token == branch:
                    branch_count += 1
                    coords_remaining = 6
                    coord_run = 0
                    last_branch_offset = 0
                elif 0 <= token < num_discrete:
                    coord_count += 1
                    coord_run += 1
                    if coords_remaining <= 0:
                        coords_remaining = 2
                    else:
                        coords_remaining -= 1
                    if coords_remaining == 0:
                        completed_joints += 1
                    if last_branch_offset < 1_000_000:
                        last_branch_offset += 1
                else:
                    coord_run = 0
                    if last_branch_offset < 1_000_000:
                        last_branch_offset += 1

                in_coord_block = coords_remaining > 0
                can_decide = completed_joints > 0 and not in_coord_block
                allowed_coord = 1.0 if (in_coord_block or not can_decide) else 1.0
                allowed_eos = float(can_decide)
                allowed_branch = float(can_decide)
                allowed_special = float(not in_coord_block)
                possible_count_norm = 1.0 / 256.0 if in_coord_block else 4.0 / 256.0
                feat = [
                    allowed_eos,
                    allowed_branch,
                    allowed_coord,
                    allowed_special,
                    min(completed_joints / 128.0, 1.0),
                    min(branch_count / 128.0, 1.0),
                    min(coord_count / 384.0, 1.0),
                    min(coord_run / 6.0, 1.0),
                    float(token == eos),
                    float(token == branch),
                    float(0 <= token < num_discrete),
                    float(token >= num_discrete and token not in {eos, branch}),
                    float(token_idx / max(valid_len - 1, 1)),
                    possible_count_norm,
                    float(0 <= last_branch_offset < 3),
                    float(3 <= last_branch_offset < 6),
                ]
                features_cpu[batch_idx, token_idx] = torch.tensor(feat, dtype=torch.float32)
        return features_cpu.to(device=input_ids.device, non_blocking=True)

    def _grammar_state_feature_for_prefix(
        self,
        prefix: list[int],
        *,
        valid_len: int,
        token_idx: int,
        num_discrete: int,
        eos: int,
        branch: int,
        vocab_size: float,
    ) -> list[float]:
        try:
            possible = set(int(x) for x in self.tokenizer.next_posible_token(ids=np.asarray(prefix, dtype=np.int64)))
        except Exception:
            possible = set()

        last = int(prefix[-1]) if prefix else -1
        coord_count = sum(1 for x in prefix if 0 <= int(x) < num_discrete)
        branch_count = sum(1 for x in prefix if int(x) == branch)
        completed_joints = self._count_completed_joints(prefix)
        coord_run = 0
        for token in reversed(prefix):
            if 0 <= int(token) < num_discrete:
                coord_run += 1
            else:
                break
        last_branch_pos = max((i for i, token in enumerate(prefix) if int(token) == branch), default=-1000000)
        branch_offset = len(prefix) - last_branch_pos - 1

        return [
            float(eos in possible),
            float(branch in possible),
            float(any(0 <= x < num_discrete for x in possible)),
            float(any(x >= num_discrete and x not in {eos, branch} for x in possible)),
            min(completed_joints / 128.0, 1.0),
            min(branch_count / 128.0, 1.0),
            min(coord_count / 384.0, 1.0),
            min(coord_run / 6.0, 1.0),
            float(last == eos),
            float(last == branch),
            float(0 <= last < num_discrete),
            float(last >= num_discrete and last not in {eos, branch}),
            float(token_idx / max(valid_len - 1, 1)),
            min(len(possible) / vocab_size, 1.0),
            float(0 <= branch_offset < 3),
            float(3 <= branch_offset < 6),
        ]

    def _count_completed_joints(self, ids: list[int]) -> int:
        num_discrete = int(self.tokenizer.num_discrete)
        branch = int(getattr(self.tokenizer, "token_id_branch"))
        state = "expect_bos"
        coords_needed = 0
        joints = 0
        for token in ids:
            if state == "expect_bos":
                state = "expect_cls_or_part_or_joint"
            elif state in {"expect_cls_or_part_or_joint", "expect_part_or_joint", "expect_branch_or_part_or_joint"}:
                if token == branch:
                    coords_needed = 6
                    state = "expect_coords"
                elif 0 <= token < num_discrete:
                    coords_needed = 2
                    state = "expect_coords"
                else:
                    state = "expect_part_or_joint"
            elif state == "expect_joint":
                if 0 <= token < num_discrete:
                    coords_needed = 2
                    state = "expect_coords"
            elif state == "expect_coords":
                if 0 <= token < num_discrete:
                    coords_needed -= 1
                    if coords_needed <= 0:
                        joints += 1
                        state = "expect_branch_or_part_or_joint"
        return joints

    def _possible_action_groups(self, possible: set[int]) -> set[int]:
        num_discrete = int(self.tokenizer.num_discrete)
        eos = int(self.tokenizer.eos)
        branch = int(getattr(self.tokenizer, "token_id_branch"))
        groups: set[int] = set()
        if eos in possible:
            groups.add(0)
        if branch in possible:
            groups.add(1)
        if any(0 <= x < num_discrete for x in possible):
            groups.add(2)
        if any(x >= num_discrete and x not in {eos, branch} for x in possible):
            groups.add(3)
        return groups

    def _token_action_group(self, token: int) -> int:
        num_discrete = int(self.tokenizer.num_discrete)
        eos = int(self.tokenizer.eos)
        branch = int(getattr(self.tokenizer, "token_id_branch"))
        if token == eos:
            return 0
        if token == branch:
            return 1
        if 0 <= token < num_discrete:
            return 2
        return 3

    def _structure_action_losses(
        self,
        hidden: torch.Tensor | None,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.structure_action_loss_weight <= 0.0 or hidden is None:
            zero = torch.zeros((), device=labels.device, dtype=self.transformer.dtype)
            return zero, zero.float(), zero.float()

        action_logits = []
        action_targets = []
        for batch_idx in range(input_ids.shape[0]):
            full_ids = input_ids[batch_idx].detach().cpu().numpy().astype(np.int64)
            for t in range(labels.shape[1]):
                label = int(labels[batch_idx, t].detach().cpu())
                if label < 0:
                    continue
                prefix = full_ids[: t + 1]
                try:
                    possible = set(int(x) for x in self.tokenizer.next_posible_token(ids=prefix))
                except Exception:
                    continue
                groups = self._possible_action_groups(possible)
                if len(groups) < 2:
                    continue
                target = self._token_action_group(label)
                if target not in groups:
                    continue
                logits = self.structure_action_head(hidden[batch_idx, t].to(dtype=torch.float32))
                mask = torch.full_like(logits, -1.0e9)
                mask[list(groups)] = 0.0
                action_logits.append(logits + mask)
                action_targets.append(target)

        if not action_logits:
            zero = torch.zeros((), device=labels.device, dtype=self.transformer.dtype)
            return zero, zero.float(), zero.float()
        stacked_logits = torch.stack(action_logits, dim=0)
        targets = torch.tensor(action_targets, device=labels.device, dtype=torch.long)
        loss = nn.functional.cross_entropy(stacked_logits, targets)
        acc = (stacked_logits.argmax(dim=-1) == targets).to(torch.float32).mean()
        count = torch.tensor(float(targets.numel()), device=labels.device)
        return loss.to(dtype=self.transformer.dtype), acc, count

    def _decision_group_logits(
        self,
        token_logits: torch.Tensor,
        possible: set[int],
    ) -> torch.Tensor | None:
        eos = int(self.tokenizer.eos)
        branch = int(getattr(self.tokenizer, "token_id_branch"))
        if eos not in possible:
            return None
        continue_ids = sorted(x for x in possible if x not in {eos, branch})
        if not continue_ids:
            return None
        eos_logit = token_logits[eos]
        if branch in possible:
            branch_logit = token_logits[branch]
        else:
            branch_logit = token_logits.new_full((), -1.0e9)
        continue_logit = torch.logsumexp(token_logits[continue_ids], dim=0)
        return torch.stack([eos_logit, branch_logit, continue_logit], dim=0)

    def _decision_target(self, label: int) -> int:
        eos = int(self.tokenizer.eos)
        branch = int(getattr(self.tokenizer, "token_id_branch"))
        if label == eos:
            return 0
        if label == branch:
            return 1
        return 2

    def _decision_losses(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Auxiliary EOS/BRANCH/CONTINUE loss at UniRig decision states.

        UniRig grammar allows EOS and BRANCH immediately after a completed
        joint.  CE supervises the exact next token, but it does not explicitly
        separate the control decision "stop / branch / continue".  The free
        generation failures we observed are exactly failures of this control
        decision, so this optional loss gives a direct training signal there.
        """

        if self.decision_loss_weight <= 0.0:
            zero = torch.zeros((), device=logits.device, dtype=logits.dtype)
            return zero, zero.float(), zero.float()

        decision_logits = []
        decision_targets = []
        num_discrete = int(self.tokenizer.num_discrete)
        eos = int(self.tokenizer.eos)
        branch = int(getattr(self.tokenizer, "token_id_branch"))

        for batch_idx in range(input_ids.shape[0]):
            full_ids = input_ids[batch_idx].detach().cpu().numpy().astype(np.int64)
            for t in range(labels.shape[1]):
                label = int(labels[batch_idx, t].detach().cpu())
                if label < 0:
                    continue
                prefix = full_ids[: t + 1]
                # Exclude the empty pre-root state.  It is legal in the
                # tokenizer to emit EOS very early, but such examples are not
                # useful for the skeleton control decision we care about.
                if int(np.sum((prefix >= 0) & (prefix < num_discrete))) < 3:
                    continue
                try:
                    possible = set(int(x) for x in self.tokenizer.next_posible_token(ids=prefix))
                except Exception:
                    continue
                if eos not in possible:
                    continue
                continue_ids = sorted(x for x in possible if x not in {eos, branch})
                if not continue_ids:
                    continue

                grouped = self._decision_group_logits(logits[batch_idx, t], possible)
                if grouped is None:
                    continue
                decision_logits.append(grouped)
                decision_targets.append(self._decision_target(label))

        if not decision_logits:
            zero = torch.zeros((), device=logits.device, dtype=logits.dtype)
            return zero, zero.float(), zero.float()
        stacked_logits = torch.stack(decision_logits, dim=0)
        targets = torch.tensor(decision_targets, device=logits.device, dtype=torch.long)
        loss = nn.functional.cross_entropy(stacked_logits.float(), targets)
        acc = (stacked_logits.argmax(dim=-1) == targets).to(torch.float32).mean()
        count = torch.tensor(float(targets.numel()), device=logits.device)
        return loss.to(dtype=logits.dtype), acc, count

    def _prefix_decision_recovery_loss(
        self,
        cond: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decision supervision on slightly drifted intermediate prefixes.

        Free generation failures on the fixed failure panel drift early from
        BRANCH to coordinate tokens, or the reverse.  Teacher-forced CE only
        trains exact GT prefixes, so this loss asks the model to make the same
        EOS/BRANCH/CONTINUE decision after small coordinate drift in valid
        intermediate prefixes.
        """

        if (
            self.prefix_decision_recovery_weight <= 0.0
            or self.prefix_decision_recovery_states <= 0
            or self.prefix_decision_recovery_variants <= 0
        ):
            zero = torch.zeros((), device=input_ids.device, dtype=cond.dtype)
            return zero, zero.float(), zero.float()

        num_discrete = int(self.tokenizer.num_discrete)
        eos = int(self.tokenizer.eos)
        branch = int(getattr(self.tokenizer, "token_id_branch"))
        max_positions = getattr(getattr(self.transformer, "config", None), "max_position_embeddings", None)
        max_total_len = int(max_positions) if max_positions is not None else None

        variants: list[list[int]] = []
        possible_sets: list[set[int]] = []
        targets: list[int] = []
        cond_indices: list[int] = []

        for batch_idx in range(input_ids.shape[0]):
            valid_ids = input_ids[batch_idx][attention_mask[batch_idx].to(torch.bool)]
            ids = [int(x) for x in valid_ids.detach().cpu().tolist()]
            if eos in ids:
                ids = ids[: ids.index(eos) + 1]
            candidates: list[tuple[list[int], int]] = []
            for t in range(len(ids) - 1):
                label = ids[t + 1]
                if label < 0:
                    continue
                prefix = ids[: t + 1]
                if sum(0 <= x < num_discrete for x in prefix) < 3:
                    continue
                try:
                    possible = set(int(x) for x in self.tokenizer.next_posible_token(ids=np.asarray(prefix)))
                except Exception:
                    continue
                if eos not in possible:
                    continue
                continue_ids = [x for x in possible if x not in {eos, branch}]
                if not continue_ids:
                    continue
                candidates.append((prefix, self._decision_target(label)))

            if not candidates:
                continue
            if len(candidates) <= self.prefix_decision_recovery_states:
                selected = candidates
            else:
                idxs = np.linspace(0, len(candidates) - 1, self.prefix_decision_recovery_states)
                selected = [candidates[int(round(x))] for x in idxs]

            for prefix, target in selected:
                for variant_idx in range(self.prefix_decision_recovery_variants):
                    corrupted = self._corrupt_decision_prefix(prefix, variant_idx)
                    if not self._loop_variant_fits(corrupted, cond.shape[1], max_total_len):
                        continue
                    try:
                        possible = set(int(x) for x in self.tokenizer.next_posible_token(ids=np.asarray(corrupted)))
                    except Exception:
                        continue
                    if self._decision_group_logits(cond.new_zeros(self.tokenizer.vocab_size), possible) is None:
                        continue
                    variants.append(corrupted)
                    possible_sets.append(possible)
                    targets.append(target)
                    cond_indices.append(batch_idx)

        if not variants:
            zero = torch.zeros((), device=input_ids.device, dtype=cond.dtype)
            return zero, zero.float(), zero.float()

        pad_id = int(self.tokenizer.pad)
        max_len = max(len(x) for x in variants)
        prefix_ids = torch.full((len(variants), max_len), pad_id, device=input_ids.device, dtype=torch.long)
        prefix_mask = torch.zeros((len(variants), max_len), device=input_ids.device, dtype=attention_mask.dtype)
        for row_idx, ids in enumerate(variants):
            prefix_ids[row_idx, : len(ids)] = torch.tensor(ids, device=input_ids.device, dtype=torch.long)
            prefix_mask[row_idx, : len(ids)] = 1

        cond_index_tensor = torch.tensor(cond_indices, device=cond.device, dtype=torch.long)
        selected_cond = cond.index_select(0, cond_index_tensor).to(dtype=self.transformer.dtype)
        prefix_embeds = self.token_inputs_embeds(prefix_ids, prefix_mask)
        inputs_embeds = torch.cat([selected_cond, prefix_embeds], dim=1)
        full_attention = pad(prefix_mask, (selected_cond.shape[1], 0, 0, 0), value=1.0)
        output = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention,
            use_cache=False,
        )
        last_index = prefix_mask.to(torch.long).sum(dim=1) - 1 + selected_cond.shape[1]
        row_index = torch.arange(prefix_ids.shape[0], device=input_ids.device)
        row_logits = output.logits[row_index, last_index]

        grouped_logits = []
        kept_targets = []
        for row_idx, possible in enumerate(possible_sets):
            grouped = self._decision_group_logits(row_logits[row_idx], possible)
            if grouped is None:
                continue
            grouped_logits.append(grouped)
            kept_targets.append(targets[row_idx])

        if not grouped_logits:
            zero = torch.zeros((), device=input_ids.device, dtype=cond.dtype)
            return zero, zero.float(), zero.float()
        stacked_logits = torch.stack(grouped_logits, dim=0)
        target_tensor = torch.tensor(kept_targets, device=input_ids.device, dtype=torch.long)
        loss = nn.functional.cross_entropy(stacked_logits.float(), target_tensor)
        acc = (stacked_logits.argmax(dim=-1) == target_tensor).to(torch.float32).mean()
        count = torch.tensor(float(target_tensor.numel()), device=input_ids.device)
        return loss.to(dtype=cond.dtype), acc, count

    def _prefix_token_recovery_loss(
        self,
        cond: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exact next-token supervision under small generated-prefix drift.

        Free generation does not see perfect GT prefixes. Once a coordinate is
        slightly off, teacher-forced CE no longer describes the state the model
        must recover from. This loss keeps the grammar-valid prefix shape but
        perturbs recent coordinate values, then asks the decoder to predict the
        original next token under the same UniRig vocab mask.
        """

        if (
            self.prefix_token_recovery_weight <= 0.0
            or self.prefix_token_recovery_states <= 0
            or self.prefix_token_recovery_variants <= 0
        ):
            zero = torch.zeros((), device=input_ids.device, dtype=cond.dtype)
            return zero, zero.float(), zero.float()

        variants = self._collect_corrupted_next_token_prefixes(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cond_len=cond.shape[1],
            state_count=self.prefix_token_recovery_states,
            variant_count=self.prefix_token_recovery_variants,
            jitter=self.prefix_token_recovery_jitter,
            max_rows=self.prefix_token_recovery_max_rows,
            require_ambiguous_action=False,
        )
        if not variants:
            zero = torch.zeros((), device=input_ids.device, dtype=cond.dtype)
            return zero, zero.float(), zero.float()

        row_logits, possible_sets, targets = self._run_recovery_prefixes(cond, input_ids, variants, output_hidden=False)
        if row_logits is None:
            zero = torch.zeros((), device=input_ids.device, dtype=cond.dtype)
            return zero, zero.float(), zero.float()

        masked_logits = []
        kept_targets = []
        for row_idx, possible in enumerate(possible_sets):
            target = int(targets[row_idx])
            if target not in possible:
                continue
            mask = torch.full_like(row_logits[row_idx], -1.0e9)
            mask[list(possible)] = 0.0
            masked_logits.append(row_logits[row_idx] + mask)
            kept_targets.append(target)

        if not masked_logits:
            zero = torch.zeros((), device=input_ids.device, dtype=cond.dtype)
            return zero, zero.float(), zero.float()
        stacked_logits = torch.stack(masked_logits, dim=0)
        target_tensor = torch.tensor(kept_targets, device=input_ids.device, dtype=torch.long)
        loss = nn.functional.cross_entropy(stacked_logits.float(), target_tensor)
        acc = (stacked_logits.argmax(dim=-1) == target_tensor).to(torch.float32).mean()
        count = torch.tensor(float(target_tensor.numel()), device=input_ids.device)
        return loss.to(dtype=cond.dtype), acc, count

    def _prefix_action_recovery_loss(
        self,
        cond: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Action-head supervision under corrupted prefixes.

        The plain structure action loss is easy under teacher forcing and did
        not transfer to free generation. This variant trains the same action
        head on prefixes whose recent coordinates have already drifted, which
        is the state observed in the failure panel before branch/EOS collapse.
        """

        if (
            self.prefix_action_recovery_weight <= 0.0
            or self.prefix_action_recovery_states <= 0
            or self.prefix_action_recovery_variants <= 0
        ):
            zero = torch.zeros((), device=input_ids.device, dtype=cond.dtype)
            return zero, zero.float(), zero.float()

        variants = self._collect_corrupted_next_token_prefixes(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cond_len=cond.shape[1],
            state_count=self.prefix_action_recovery_states,
            variant_count=self.prefix_action_recovery_variants,
            jitter=self.prefix_action_recovery_jitter,
            max_rows=self.prefix_action_recovery_max_rows,
            require_ambiguous_action=True,
        )
        if not variants:
            zero = torch.zeros((), device=input_ids.device, dtype=cond.dtype)
            return zero, zero.float(), zero.float()

        row_logits, possible_sets, targets, row_hidden = self._run_recovery_prefixes(
            cond,
            input_ids,
            variants,
            output_hidden=True,
        )
        if row_hidden is None:
            zero = torch.zeros((), device=input_ids.device, dtype=cond.dtype)
            return zero, zero.float(), zero.float()

        action_logits = []
        action_targets = []
        for row_idx, possible in enumerate(possible_sets):
            groups = self._possible_action_groups(possible)
            target = self._token_action_group(int(targets[row_idx]))
            if len(groups) < 2 or target not in groups:
                continue
            logits = self.structure_action_head(row_hidden[row_idx].to(dtype=torch.float32))
            mask = torch.full_like(logits, -1.0e9)
            mask[list(groups)] = 0.0
            action_logits.append(logits + mask)
            action_targets.append(target)

        if not action_logits:
            zero = torch.zeros((), device=input_ids.device, dtype=cond.dtype)
            return zero, zero.float(), zero.float()
        stacked_logits = torch.stack(action_logits, dim=0)
        target_tensor = torch.tensor(action_targets, device=input_ids.device, dtype=torch.long)
        loss = nn.functional.cross_entropy(stacked_logits, target_tensor)
        acc = (stacked_logits.argmax(dim=-1) == target_tensor).to(torch.float32).mean()
        count = torch.tensor(float(target_tensor.numel()), device=input_ids.device)
        return loss.to(dtype=cond.dtype), acc, count

    def _generated_prefix_recovery_loss(
        self,
        cond: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Next-token recovery on prefixes produced by the model itself.

        The hard failures are early free-generation trajectory errors, not just
        missing EOS at the end.  The older recovery losses perturb GT prefixes;
        this one first greedily rolls the current model forward for a short
        horizon, collects grammar-valid states after it diverges from the GT
        sequence, then supervises the GT next token at the same absolute token
        position.  It is an optional training diagnostic, default-off.
        """

        if (
            self.generated_prefix_recovery_weight <= 0.0
            or self.generated_prefix_recovery_states <= 0
            or self.generated_prefix_recovery_max_new_tokens <= 0
        ):
            zero = torch.zeros((), device=input_ids.device, dtype=cond.dtype)
            return zero, zero.float(), zero.float(), zero.float(), zero.float(), zero.float(), zero.float()

        variants, stats = self._collect_generated_next_token_prefixes(
            cond=cond,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        rollout_steps = torch.tensor(float(stats["rollout_steps"]), device=input_ids.device, dtype=torch.float32)
        diverged_steps = torch.tensor(float(stats["diverged_steps"]), device=input_ids.device, dtype=torch.float32)
        invalid_target_count = torch.tensor(
            float(stats["invalid_target_count"]),
            device=input_ids.device,
            dtype=torch.float32,
        )
        candidate_count = torch.tensor(float(stats["candidate_count"]), device=input_ids.device, dtype=torch.float32)
        if not variants:
            zero = torch.zeros((), device=input_ids.device, dtype=cond.dtype)
            return zero, zero.float(), zero.float(), rollout_steps, diverged_steps, invalid_target_count, candidate_count

        row_logits, possible_sets, targets = self._run_recovery_prefixes(cond, input_ids, variants, output_hidden=False)
        if row_logits is None:
            zero = torch.zeros((), device=input_ids.device, dtype=cond.dtype)
            return zero, zero.float(), zero.float(), rollout_steps, diverged_steps, invalid_target_count, candidate_count

        masked_logits = []
        kept_targets = []
        for row_idx, possible in enumerate(possible_sets):
            target = int(targets[row_idx])
            if target not in possible:
                continue
            mask = torch.full_like(row_logits[row_idx], -1.0e9)
            mask[list(possible)] = 0.0
            masked_logits.append(row_logits[row_idx] + mask)
            kept_targets.append(target)

        if not masked_logits:
            zero = torch.zeros((), device=input_ids.device, dtype=cond.dtype)
            return zero, zero.float(), zero.float(), rollout_steps, diverged_steps, invalid_target_count, candidate_count
        stacked_logits = torch.stack(masked_logits, dim=0)
        target_tensor = torch.tensor(kept_targets, device=input_ids.device, dtype=torch.long)
        loss = nn.functional.cross_entropy(stacked_logits.float(), target_tensor)
        acc = (stacked_logits.argmax(dim=-1) == target_tensor).to(torch.float32).mean()
        count = torch.tensor(float(target_tensor.numel()), device=input_ids.device)
        return (
            loss.to(dtype=cond.dtype),
            acc,
            count,
            rollout_steps,
            diverged_steps,
            invalid_target_count,
            candidate_count,
        )

    @torch.no_grad()
    def _collect_generated_next_token_prefixes(
        self,
        *,
        cond: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[list[tuple[int, list[int], set[int], int]], dict[str, int]]:
        num_discrete = int(self.tokenizer.num_discrete)
        eos = int(self.tokenizer.eos)
        max_positions = getattr(getattr(self.transformer, "config", None), "max_position_embeddings", None)
        max_total_len = int(max_positions) if max_positions is not None else None

        rows: list[tuple[int, list[int], set[int], int]] = []
        stats = {
            "rollout_steps": 0,
            "diverged_steps": 0,
            "invalid_target_count": 0,
            "candidate_count": 0,
        }
        for batch_idx in range(input_ids.shape[0]):
            valid_ids = input_ids[batch_idx][attention_mask[batch_idx].to(torch.bool)]
            target_ids = [int(x) for x in valid_ids.detach().cpu().tolist()]
            if eos in target_ids:
                target_ids = target_ids[: target_ids.index(eos) + 1]
            if len(target_ids) < 4:
                continue

            prefix = target_ids[:2]
            collected: list[tuple[int, list[int], set[int], int]] = []
            diverged = False
            horizon = min(self.generated_prefix_recovery_max_new_tokens, max(len(target_ids) - 2, 1))
            for _ in range(horizon):
                target_pos = len(prefix)
                if target_pos >= len(target_ids):
                    break
                target = int(target_ids[target_pos])
                try:
                    possible = set(int(x) for x in self.tokenizer.next_posible_token(ids=np.asarray(prefix)))
                except Exception:
                    break
                if not possible:
                    break
                stats["rollout_steps"] += 1

                logits = self._next_logits_for_prefix_no_grad(cond, batch_idx, prefix)
                if logits is None:
                    break
                mask = torch.full_like(logits, -float("inf"))
                mask[list(possible)] = 0.0
                pred = int(torch.argmax(logits + mask).item())

                if pred != target:
                    diverged = True
                if diverged:
                    stats["diverged_steps"] += 1
                    if target not in possible:
                        stats["invalid_target_count"] += 1
                    elif (
                        self._prefix_matches_gt_structure(prefix, target_ids[:target_pos], num_discrete)
                        and self._generated_prefix_state_is_useful(prefix, pred, target, num_discrete)
                    ):
                        stats["candidate_count"] += 1
                        collected.append((batch_idx, list(prefix), possible, target))

                prefix.append(pred)
                if pred == eos:
                    break
                if not self._loop_variant_fits(prefix, cond.shape[1], max_total_len):
                    break

            if len(collected) > self.generated_prefix_recovery_states:
                idxs = np.linspace(0, len(collected) - 1, self.generated_prefix_recovery_states)
                collected = [collected[int(round(x))] for x in idxs]
            rows.extend(collected)

        max_rows = int(self.generated_prefix_recovery_max_rows)
        if max_rows > 0 and len(rows) > max_rows:
            idxs = np.linspace(0, len(rows) - 1, max_rows)
            rows = [rows[int(round(x))] for x in idxs]
        return rows, stats

    @torch.no_grad()
    def _next_logits_for_prefix_no_grad(
        self,
        cond: torch.Tensor,
        batch_idx: int,
        prefix: list[int],
    ) -> torch.Tensor | None:
        if not prefix:
            return None
        device = cond.device
        prefix_ids = torch.tensor([prefix], device=device, dtype=torch.long)
        prefix_mask = torch.ones_like(prefix_ids, dtype=torch.long)
        selected_cond = cond[batch_idx : batch_idx + 1].to(dtype=self.transformer.dtype)
        prefix_embeds = self.token_inputs_embeds(prefix_ids, prefix_mask)
        inputs_embeds = torch.cat([selected_cond, prefix_embeds], dim=1)
        full_attention = pad(prefix_mask, (selected_cond.shape[1], 0, 0, 0), value=1.0)
        output = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention,
            use_cache=False,
            output_hidden_states=self.uses_action_group_bias,
        )
        logits = output.logits[0, selected_cond.shape[1] + prefix_ids.shape[1] - 1]
        if self.uses_action_group_bias:
            assert output.hidden_states is not None
            hidden = output.hidden_states[-1][0, selected_cond.shape[1] + prefix_ids.shape[1] - 1]
            logits = self.apply_action_group_bias_row(logits.unsqueeze(0), hidden.unsqueeze(0), selected_cond).squeeze(0)
        return logits

    @staticmethod
    def _generated_prefix_state_is_useful(prefix: list[int], pred: int, target: int, num_discrete: int) -> bool:
        if pred != target:
            return True
        if len(prefix) < 6:
            return False
        recent = [int(x) for x in prefix[-6:] if 0 <= int(x) < num_discrete]
        if len(recent) < 6:
            return False
        return recent[-3:] == recent[-6:-3]

    @staticmethod
    def _prefix_matches_gt_structure(prefix: list[int], gt_prefix: list[int], num_discrete: int) -> bool:
        if len(prefix) != len(gt_prefix):
            return False
        for pred_token, gt_token in zip(prefix, gt_prefix, strict=False):
            pred_is_coord = 0 <= int(pred_token) < num_discrete
            gt_is_coord = 0 <= int(gt_token) < num_discrete
            if pred_is_coord or gt_is_coord:
                if pred_is_coord != gt_is_coord:
                    return False
                continue
            if int(pred_token) != int(gt_token):
                return False
        return True

    def _collect_corrupted_next_token_prefixes(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cond_len: int,
        state_count: int,
        variant_count: int,
        jitter: int,
        max_rows: int,
        require_ambiguous_action: bool,
    ) -> list[tuple[int, list[int], set[int], int]]:
        num_discrete = int(self.tokenizer.num_discrete)
        eos = int(self.tokenizer.eos)
        branch = int(getattr(self.tokenizer, "token_id_branch"))
        max_positions = getattr(getattr(self.transformer, "config", None), "max_position_embeddings", None)
        max_total_len = int(max_positions) if max_positions is not None else None

        rows: list[tuple[int, list[int], set[int], int]] = []
        for batch_idx in range(input_ids.shape[0]):
            valid_ids = input_ids[batch_idx][attention_mask[batch_idx].to(torch.bool)]
            ids = [int(x) for x in valid_ids.detach().cpu().tolist()]
            if eos in ids:
                ids = ids[: ids.index(eos) + 1]
            candidates: list[tuple[list[int], int]] = []
            for t in range(len(ids) - 1):
                label = int(ids[t + 1])
                if label < 0:
                    continue
                prefix = ids[: t + 1]
                has_coord_prefix = any(0 <= x < num_discrete for x in prefix)
                label_is_coord = 0 <= label < num_discrete
                # Keep the recovery task off the class-token prologue while still
                # covering the first root/child coordinate states where hitmax
                # failures start.
                if not has_coord_prefix and not label_is_coord:
                    continue
                try:
                    possible = set(int(x) for x in self.tokenizer.next_posible_token(ids=np.asarray(prefix)))
                except Exception:
                    continue
                if label not in possible:
                    continue
                if require_ambiguous_action:
                    groups = self._possible_action_groups(possible)
                    if len(groups) < 2:
                        continue
                    if branch not in possible and eos not in possible:
                        continue
                candidates.append((prefix, label))

            if not candidates:
                continue
            if len(candidates) <= state_count:
                selected = candidates
            else:
                idxs = np.linspace(0, len(candidates) - 1, state_count)
                selected = [candidates[int(round(x))] for x in idxs]

            for prefix, label in selected:
                for variant_idx in range(variant_count):
                    corrupted = self._corrupt_coordinate_prefix(prefix, variant_idx, jitter=jitter)
                    if not self._loop_variant_fits(corrupted, cond_len, max_total_len):
                        continue
                    try:
                        possible = set(int(x) for x in self.tokenizer.next_posible_token(ids=np.asarray(corrupted)))
                    except Exception:
                        continue
                    if label not in possible:
                        continue
                    rows.append((batch_idx, corrupted, possible, label))
        if max_rows > 0 and len(rows) > max_rows:
            idxs = np.linspace(0, len(rows) - 1, max_rows)
            rows = [rows[int(round(x))] for x in idxs]
        return rows

    def _run_recovery_prefixes(
        self,
        cond: torch.Tensor,
        input_ids: torch.Tensor,
        variants: list[tuple[int, list[int], set[int], int]],
        *,
        output_hidden: bool,
    ) -> tuple[torch.Tensor | None, list[set[int]], list[int]] | tuple[torch.Tensor | None, list[set[int]], list[int], torch.Tensor | None]:
        if not variants:
            if output_hidden:
                return None, [], [], None
            return None, [], []

        pad_id = int(self.tokenizer.pad)
        max_len = max(len(ids) for _, ids, _, _ in variants)
        prefix_ids = torch.full((len(variants), max_len), pad_id, device=input_ids.device, dtype=torch.long)
        prefix_mask = torch.zeros((len(variants), max_len), device=input_ids.device, dtype=torch.long)
        cond_indices = []
        possible_sets = []
        targets = []
        for row_idx, (batch_idx, ids, possible, label) in enumerate(variants):
            prefix_ids[row_idx, : len(ids)] = torch.tensor(ids, device=input_ids.device, dtype=torch.long)
            prefix_mask[row_idx, : len(ids)] = 1
            cond_indices.append(batch_idx)
            possible_sets.append(possible)
            targets.append(label)

        cond_index_tensor = torch.tensor(cond_indices, device=cond.device, dtype=torch.long)
        selected_cond = cond.index_select(0, cond_index_tensor).to(dtype=self.transformer.dtype)
        prefix_embeds = self.token_inputs_embeds(prefix_ids, prefix_mask)
        inputs_embeds = torch.cat([selected_cond, prefix_embeds], dim=1)
        full_attention = pad(prefix_mask, (selected_cond.shape[1], 0, 0, 0), value=1.0)
        need_hidden = output_hidden or self.uses_action_group_bias
        output = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention,
            use_cache=False,
            output_hidden_states=need_hidden,
        )
        last_index = prefix_mask.sum(dim=1) - 1 + selected_cond.shape[1]
        row_index = torch.arange(prefix_ids.shape[0], device=input_ids.device)
        row_logits = output.logits[row_index, last_index]
        row_hidden = None
        if need_hidden:
            assert output.hidden_states is not None
            row_hidden = output.hidden_states[-1][row_index, last_index]
            row_logits = self.apply_action_group_bias_row(row_logits, row_hidden, selected_cond)
        if output_hidden:
            return row_logits, possible_sets, targets, row_hidden
        return row_logits, possible_sets, targets

    def _corrupt_decision_prefix(self, prefix: list[int], variant_idx: int) -> list[int]:
        return self._corrupt_coordinate_prefix(prefix, variant_idx, jitter=self.prefix_decision_recovery_jitter)

    def _corrupt_coordinate_prefix(self, prefix: list[int], variant_idx: int, *, jitter: int) -> list[int]:
        num_discrete = int(self.tokenizer.num_discrete)
        jitter = max(1, int(jitter))
        out = list(prefix)
        coord_positions = [i for i, token in enumerate(out) if 0 <= token < num_discrete]
        if not coord_positions:
            return out
        last_triplet = coord_positions[-min(3, len(coord_positions)) :]
        mode = variant_idx % 3
        if mode == 0:
            offsets = (-jitter, 0, jitter)[-len(last_triplet) :]
            for pos, offset in zip(last_triplet, offsets, strict=False):
                out[pos] = int(np.clip(out[pos] + offset, 0, num_discrete - 1))
        elif mode == 1:
            repeated = out[last_triplet[0]]
            for pos in last_triplet:
                out[pos] = repeated
        else:
            sign = -1 if (variant_idx // 3) % 2 == 0 else 1
            for pos in last_triplet:
                out[pos] = int(np.clip(out[pos] + sign * jitter, 0, num_discrete - 1))
        return out

    def _loop_recovery_loss(
        self,
        cond: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        include_loop_recovery: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Train the AR decoder to terminate after synthetic looped prefixes.

        The no-EOS failures we inspected are not illegal-token errors.  They are
        free-running prefixes that drift into repeated coordinate or
        BRANCH/coordinate motifs.  Teacher-forced CE never shows the model such
        bad prefixes, so this optional diagnostic loss appends short legal-ish
        loop suffixes after the GT terminal decision state and asks for EOS.

        The hook is deliberately default-off.  It is an experiment for the
        observed prefix-drift failure mode, not an alternate decoder.
        """

        if (
            not include_loop_recovery
            or self.loop_recovery_loss_weight <= 0.0
            or self.loop_recovery_repeats <= 0
        ):
            zero = torch.zeros((), device=input_ids.device, dtype=cond.dtype)
            return zero, zero.float(), zero.float()

        num_discrete = int(self.tokenizer.num_discrete)
        eos = int(self.tokenizer.eos)
        branch = int(getattr(self.tokenizer, "token_id_branch"))
        max_positions = getattr(getattr(self.transformer, "config", None), "max_position_embeddings", None)
        max_total_len = int(max_positions) if max_positions is not None else None

        variants: list[list[int]] = []
        cond_indices: list[int] = []
        for batch_idx in range(input_ids.shape[0]):
            valid_ids = input_ids[batch_idx][attention_mask[batch_idx].to(torch.bool)]
            ids = [int(x) for x in valid_ids.detach().cpu().tolist()]
            if eos not in ids:
                continue
            eos_pos = ids.index(eos)
            terminal_prefix = ids[:eos_pos]
            if len(terminal_prefix) < 4:
                continue
            coord_tokens = [x for x in terminal_prefix if 0 <= x < num_discrete]
            if len(coord_tokens) < 3:
                continue
            try:
                possible = set(int(x) for x in self.tokenizer.next_posible_token(ids=np.asarray(terminal_prefix)))
            except Exception:
                continue
            if eos not in possible:
                continue

            last_coord = coord_tokens[-1]
            coord_loop = terminal_prefix + [last_coord] * (3 * self.loop_recovery_repeats)
            if self._loop_variant_fits(coord_loop, cond.shape[1], max_total_len):
                variants.append(coord_loop)
                cond_indices.append(batch_idx)

            branch_loop = self._make_branch_loop_prefix(terminal_prefix, coord_tokens[-3:], self.loop_recovery_repeats)
            if branch_loop is not None and self._loop_variant_fits(branch_loop, cond.shape[1], max_total_len):
                variants.append(branch_loop)
                cond_indices.append(batch_idx)

        if not variants:
            zero = torch.zeros((), device=input_ids.device, dtype=cond.dtype)
            return zero, zero.float(), zero.float()

        pad_id = int(self.tokenizer.pad)
        max_len = max(len(x) for x in variants)
        prefix_ids = torch.full((len(variants), max_len), pad_id, device=input_ids.device, dtype=torch.long)
        prefix_mask = torch.zeros((len(variants), max_len), device=input_ids.device, dtype=attention_mask.dtype)
        for row_idx, ids in enumerate(variants):
            prefix_ids[row_idx, : len(ids)] = torch.tensor(ids, device=input_ids.device, dtype=torch.long)
            prefix_mask[row_idx, : len(ids)] = 1

        cond_index_tensor = torch.tensor(cond_indices, device=cond.device, dtype=torch.long)
        selected_cond = cond.index_select(0, cond_index_tensor).to(dtype=self.transformer.dtype)
        prefix_embeds = self.token_inputs_embeds(prefix_ids, prefix_mask)
        inputs_embeds = torch.cat([selected_cond, prefix_embeds], dim=1)
        full_attention = pad(prefix_mask, (selected_cond.shape[1], 0, 0, 0), value=1.0)
        output = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention,
            use_cache=False,
        )
        last_index = prefix_mask.to(torch.long).sum(dim=1) - 1 + selected_cond.shape[1]
        row_index = torch.arange(prefix_ids.shape[0], device=input_ids.device)
        logits = output.logits[row_index, last_index]
        targets = torch.full((prefix_ids.shape[0],), eos, device=input_ids.device, dtype=torch.long)
        loss = nn.functional.cross_entropy(logits.float(), targets)
        acc = (logits.argmax(dim=-1) == targets).to(torch.float32).mean()
        count = torch.tensor(float(targets.numel()), device=input_ids.device)
        return loss.to(dtype=cond.dtype), acc, count

    @staticmethod
    def _loop_variant_fits(ids: list[int], cond_len: int, max_total_len: int | None) -> bool:
        if max_total_len is None:
            return True
        return cond_len + len(ids) <= max_total_len

    def _make_branch_loop_prefix(
        self,
        prefix: list[int],
        seed_coords: list[int],
        repeats: int,
    ) -> list[int] | None:
        branch = int(getattr(self.tokenizer, "token_id_branch"))
        num_discrete = int(self.tokenizer.num_discrete)
        out = list(prefix)
        for _ in range(repeats):
            try:
                possible = set(int(x) for x in self.tokenizer.next_posible_token(ids=np.asarray(out)))
            except Exception:
                return None
            if branch in possible:
                out.append(branch)
            for coord in seed_coords:
                try:
                    possible = set(int(x) for x in self.tokenizer.next_posible_token(ids=np.asarray(out)))
                except Exception:
                    return None
                if coord in possible:
                    out.append(int(coord))
                    continue
                coordinate_options = sorted(x for x in possible if 0 <= x < num_discrete)
                if not coordinate_options:
                    return None
                out.append(coordinate_options[len(coordinate_options) // 2])
        return out

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        refs = self.sample_references(batch)
        cond, branch_prior = self.build_condition(batch, refs=refs, return_branch_prior=True)
        out = self._ar_losses(cond, batch)
        branch_prior_loss, branch_prior_exist_acc, branch_prior_coord_mae = self._branch_prior_loss(branch_prior, batch)
        explicit_tree_loss, explicit_tree_metrics = self._explicit_tree_loss(cond, batch)
        explicit_tree_generated_prefix_loss, explicit_tree_generated_prefix_metrics = self._explicit_tree_generated_prefix_loss(
            cond,
            batch,
        )
        explicit_tree_oracle_prefix_loss, explicit_tree_oracle_prefix_metrics = self._explicit_tree_oracle_prefix_loss(
            cond,
            batch,
        )
        out["branch_prior_loss"] = branch_prior_loss
        out["branch_prior_exist_acc"] = branch_prior_exist_acc
        out["branch_prior_coord_mae"] = branch_prior_coord_mae
        out["explicit_tree_loss"] = explicit_tree_loss
        out["explicit_tree_oracle_prefix_loss"] = explicit_tree_oracle_prefix_loss
        out.update(explicit_tree_metrics)
        out.update(explicit_tree_generated_prefix_metrics)
        out.update(explicit_tree_oracle_prefix_metrics)
        total = out["ce_loss"]
        if self.decision_loss_weight > 0.0:
            total = total + self.decision_loss_weight * out["decision_loss"]
        if self.prefix_decision_recovery_weight > 0.0:
            total = total + self.prefix_decision_recovery_weight * out["prefix_decision_recovery_loss"]
        if self.prefix_token_recovery_weight > 0.0:
            total = total + self.prefix_token_recovery_weight * out["prefix_token_recovery_loss"]
        if self.prefix_action_recovery_weight > 0.0:
            total = total + self.prefix_action_recovery_weight * out["prefix_action_recovery_loss"]
        if self.generated_prefix_recovery_weight > 0.0:
            total = total + self.generated_prefix_recovery_weight * out["generated_prefix_recovery_loss"]
        if self.structure_count_loss_weight > 0.0:
            total = total + self.structure_count_loss_weight * out["structure_count_loss"]
        if self.structure_action_loss_weight > 0.0:
            total = total + self.structure_action_loss_weight * out["structure_action_loss"]
        if self.loop_recovery_loss_weight > 0.0:
            total = total + self.loop_recovery_loss_weight * out["loop_recovery_loss"]
        if self.branch_prior_loss_weight > 0.0:
            total = total + self.branch_prior_loss_weight * branch_prior_loss
        if self.explicit_tree_loss_weight > 0.0:
            total = total + self.explicit_tree_loss_weight * explicit_tree_loss
        if self.explicit_tree_generated_prefix_weight > 0.0:
            total = total + self.explicit_tree_generated_prefix_weight * explicit_tree_generated_prefix_loss
        if self.explicit_tree_oracle_prefix_weight > 0.0:
            total = total + self.explicit_tree_oracle_prefix_weight * explicit_tree_oracle_prefix_loss

        if self.latent_align_weight > 0.0:
            static_cond = self.build_static_condition(batch).to(device=cond.device, dtype=cond.dtype)
            align_cond = cond[:, : static_cond.shape[1]]
            align_loss = nn.functional.mse_loss(
                nn.functional.layer_norm(align_cond.float(), align_cond.shape[-1:]),
                nn.functional.layer_norm(static_cond.float(), static_cond.shape[-1:]),
            )
            total = total + self.latent_align_weight * align_loss
            out["latent_align_loss"] = align_loss

        if self.motion_contrast_weight > 0.0:
            contrast_terms = []
            for control in self.contrast_controls:
                control_cond = self.build_condition(batch, control=control, refs=refs)
                control_out = self._ar_losses(
                    control_cond,
                    batch,
                    include_loop_recovery=False,
                    include_generated_prefix_recovery=False,
                )
                control_ce = control_out["ce_loss"]
                contrast = nn.functional.relu(self.motion_contrast_margin + out["ce_loss"] - control_ce)
                contrast_terms.append(contrast)
                out[f"{control}_ce_loss"] = control_ce.detach()
            if contrast_terms:
                contrast_loss = torch.stack(contrast_terms).mean()
                total = total + self.motion_contrast_weight * contrast_loss
                out["motion_contrast_loss"] = contrast_loss

        if self.condition_control_ce_weight > 0.0:
            control_terms = []
            for control in self.condition_control_ce_controls:
                # This term is a decoder robustness probe: when motion evidence is
                # weak or corrupted, the AR decoder should still fall back to a
                # legal skeleton prior.  The corrupted condition itself is not a
                # target for the motion encoder, so detach it to avoid retaining a
                # second full motion-encoder graph on H100 qlogin runs.
                with torch.no_grad():
                    control_cond = self.build_condition(batch, control=control, refs=refs)
                control_cond = control_cond.detach()
                control_out = self._ar_losses(
                    control_cond,
                    batch,
                    include_loop_recovery=False,
                    include_generated_prefix_recovery=False,
                )
                control_ce = control_out["ce_loss"]
                control_terms.append(control_ce)
                out[f"{control}_control_ce_loss"] = control_ce.detach()
            if control_terms:
                control_ce_loss = torch.stack(control_terms).mean()
                total = total + self.condition_control_ce_weight * control_ce_loss
                out["condition_control_ce_loss"] = control_ce_loss

        out["loss"] = total
        return out
