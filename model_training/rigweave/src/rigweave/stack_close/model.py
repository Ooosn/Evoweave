from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import pad

from rigweave.dynamic_rig.sampling import TrackableSurfaceReferences
from rigweave.dynamic_rig.unirig_wrapper import DynamicRigUniRigAR

from .tokenizer import StackCloseTokenizer


class StackActionHead(nn.Module):
    """Predict child versus CLOSE, optionally reading condition tokens directly."""

    def __init__(
        self,
        hidden_size: int,
        *,
        condition_dim: int = 0,
        heads: int = 8,
    ) -> None:
        super().__init__()
        if condition_dim < 0:
            raise ValueError("condition_dim must be non-negative")
        if heads <= 0:
            raise ValueError("heads must be positive")
        if condition_dim > 0 and condition_dim % heads != 0:
            raise ValueError("condition_dim must be divisible by heads")
        self.hidden_size = int(hidden_size)
        self.condition_dim = int(condition_dim)
        self.heads = int(heads)
        self.query_norm = nn.LayerNorm(self.hidden_size)
        if self.condition_dim > 0:
            self.condition_norm = nn.LayerNorm(self.hidden_size)
            self.query_proj = nn.Linear(self.hidden_size, self.condition_dim)
            self.key_proj = nn.Linear(self.hidden_size, self.condition_dim)
            self.value_proj = nn.Linear(self.hidden_size, self.condition_dim)
            classifier_size = self.hidden_size + self.condition_dim
        else:
            self.condition_norm = None
            self.query_proj = None
            self.key_proj = None
            self.value_proj = None
            classifier_size = self.hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_size),
            nn.Linear(classifier_size, 2),
        )

    def _heads(self, values: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = values.shape
        head_dim = self.condition_dim // self.heads
        return (
            values.reshape(batch, tokens, self.heads, head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        query: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if query.ndim != 3 or condition.ndim != 3:
            raise ValueError("query and condition must have shape (B,T,D)")
        if query.shape[0] != condition.shape[0]:
            raise ValueError("query and condition batch sizes differ")
        if query.shape[-1] != self.hidden_size or condition.shape[-1] != self.hidden_size:
            raise ValueError(
                f"expected hidden size {self.hidden_size}, "
                f"got query={query.shape[-1]} condition={condition.shape[-1]}"
            )
        query_f = self.query_norm(query.float()).to(dtype=query.dtype)
        if self.condition_dim <= 0:
            return self.classifier(query_f)

        assert self.condition_norm is not None
        assert self.query_proj is not None
        assert self.key_proj is not None
        assert self.value_proj is not None
        condition_f = self.condition_norm(condition.float()).to(dtype=condition.dtype)
        q = self._heads(self.query_proj(query_f))
        k = self._heads(self.key_proj(condition_f))
        v = self._heads(self.value_proj(condition_f))
        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
        )
        context = (
            context.transpose(1, 2)
            .contiguous()
            .reshape(query.shape[0], query.shape[1], self.condition_dim)
        )
        return self.classifier(torch.cat([query_f, context], dim=-1))


def _stack_action_targets(
    labels: torch.LongTensor,
    coordinate_positions: torch.LongTensor,
    joint_count: torch.LongTensor,
    *,
    close_token: int,
) -> torch.LongTensor:
    """Build child-versus-close labels aligned with next-token logits."""

    targets = torch.full_like(labels, -100)
    targets[labels == int(close_token)] = 1
    for row, count_tensor in enumerate(joint_count):
        count = int(count_tensor.item())
        if count <= 1:
            continue
        predictor_positions = coordinate_positions[row, 1:count, 0] - 1
        predictor_positions = predictor_positions[
            (predictor_positions >= 0)
            & (predictor_positions < targets.shape[1])
        ]
        targets[row, predictor_positions] = 0
    return targets


@dataclass(frozen=True)
class PrefixPerturbationConfig:
    row_probability: float = 0.5
    axial_fraction_max: float = 0.05
    radial_fraction_max: float = 0.05
    max_perturbed_joints: int = 4
    max_joint_fraction: float = 0.08
    warmup_samples: int = 5_000
    ramp_samples: int = 15_000
    radial_clearance_cap: float = 0.5
    nearest_surface_multiplier: float = 4.0

    def validate(self) -> None:
        if not 0.0 <= self.row_probability <= 1.0:
            raise ValueError("row_probability must be in [0,1]")
        if not 0.0 <= self.axial_fraction_max <= 0.25:
            raise ValueError("axial_fraction_max must be in [0,0.25]")
        if not 0.0 <= self.radial_fraction_max <= 0.25:
            raise ValueError("radial_fraction_max must be in [0,0.25]")
        if self.max_perturbed_joints <= 0:
            raise ValueError("max_perturbed_joints must be positive")
        if not 0.0 < self.max_joint_fraction <= 1.0:
            raise ValueError("max_joint_fraction must be in (0,1]")
        if self.warmup_samples < 0 or self.ramp_samples < 0:
            raise ValueError("warmup_samples and ramp_samples must be non-negative")


class StackCloseDynamicRigAR(DynamicRigUniRigAR):
    """UniRig dynamic decoder with an isolated stack-close target contract."""

    def __init__(
        self,
        unirig_ar: nn.Module,
        conditioner: nn.Module,
        tokenizer: StackCloseTokenizer,
        *,
        perturbation: PrefixPerturbationConfig,
        stack_action_loss_weight: float = 0.0,
        stack_action_condition_dim: int = 0,
        stack_action_condition_heads: int = 8,
        num_surface_samples: int = 65_536,
        vertex_samples: int = 8_192,
        query_tokens: int = 1_024,
    ) -> None:
        perturbation.validate()
        super().__init__(
            unirig_ar,
            conditioner,
            tokenizer,
            num_surface_samples=num_surface_samples,
            vertex_samples=vertex_samples,
            query_tokens=query_tokens,
            latent_align_weight=0.0,
            motion_contrast_weight=0.0,
            condition_control_ce_weight=0.0,
            eos_loss_weight=1.0,
            decision_loss_weight=0.0,
            loop_recovery_loss_weight=0.0,
            prefix_decision_recovery_weight=0.0,
            prefix_token_recovery_weight=0.0,
            prefix_action_recovery_weight=0.0,
            generated_prefix_recovery_weight=0.0,
            structure_count_loss_weight=0.0,
            structure_action_loss_weight=0.0,
            condition_fusion="dynamic",
            use_grammar_state_embedding=False,
            use_action_group_bias=False,
            use_condition_action_group_bias=False,
            branch_prior_proposals=0,
            branch_prior_loss_weight=0.0,
            explicit_tree_loss_weight=0.0,
            explicit_tree_generated_prefix_weight=0.0,
            explicit_tree_oracle_prefix_weight=0.0,
        )
        self.perturbation = perturbation
        self.stack_action_loss_weight = float(stack_action_loss_weight)
        if self.stack_action_loss_weight < 0.0:
            raise ValueError("stack_action_loss_weight must be non-negative")
        self.stack_action_head = (
            StackActionHead(
                unirig_ar.hidden_size,
                condition_dim=stack_action_condition_dim,
                heads=stack_action_condition_heads,
            )
            if self.stack_action_loss_weight > 0.0
            else None
        )
        for module in (
            self.condition_fuser,
            self.grammar_state_proj,
            self.action_group_bias_head,
            self.condition_action_group_bias_head,
            self.structure_count_head,
            self.structure_action_head,
        ):
            module.requires_grad_(False)
        self.register_buffer(
            "_sample_seen",
            torch.zeros((), dtype=torch.long),
            persistent=False,
        )

    def set_training_sample_seen(self, sample_seen: int) -> None:
        self._sample_seen.fill_(int(sample_seen))

    def perturbation_scale(self) -> float:
        seen = int(self._sample_seen.item())
        if seen < self.perturbation.warmup_samples:
            return 0.0
        if self.perturbation.ramp_samples <= 0:
            return 1.0
        return min(
            1.0,
            (seen - self.perturbation.warmup_samples)
            / self.perturbation.ramp_samples,
        )

    @staticmethod
    def _batched_faces(
        faces: torch.LongTensor,
        batch_size: int,
    ) -> torch.LongTensor:
        if faces.dim() == 2:
            return faces.unsqueeze(0).expand(batch_size, -1, -1)
        if faces.dim() == 3 and faces.shape[0] == batch_size:
            return faces
        raise ValueError(f"faces must be (F,3) or (B,F,3), got {tuple(faces.shape)}")

    @classmethod
    def _query_surface_points(
        cls,
        vertices: torch.Tensor,
        faces: torch.LongTensor,
        refs: TrackableSurfaceReferences,
    ) -> torch.Tensor:
        """Materialize only the already-selected FPS points, not all 65k points."""

        batch_size = int(vertices.shape[0])
        refs = refs.to(vertices.device)
        faces_b = cls._batched_faces(faces.to(vertices.device), batch_size)
        vertex_sample_count = int(refs.vertex_indices.shape[1])
        outputs: list[torch.Tensor] = []
        for row in range(batch_size):
            query = refs.query_indices[row]
            from_vertex = query < vertex_sample_count
            points = vertices.new_empty((query.shape[0], 3))
            if from_vertex.any():
                vertex_slots = query[from_vertex]
                vertex_ids = refs.vertex_indices[row, vertex_slots]
                points[from_vertex] = vertices[row, vertex_ids]
            if (~from_vertex).any():
                surface_slots = query[~from_vertex] - vertex_sample_count
                face_ids = refs.face_indices[row, surface_slots]
                triangles = vertices[row, faces_b[row, face_ids]]
                barycentric = refs.barycentric[row, surface_slots]
                points[~from_vertex] = (
                    triangles * barycentric.unsqueeze(-1)
                ).sum(dim=1)
            outputs.append(points)
        return torch.stack(outputs, dim=0)

    def _quantize(self, values: torch.Tensor) -> torch.LongTensor:
        lo, hi = self.tokenizer.continuous_range
        scaled = (values.float() - lo) / (hi - lo)
        return (
            (scaled * self.tokenizer.num_discrete)
            .round()
            .clamp(0, self.tokenizer.num_discrete - 1)
            .to(dtype=torch.long)
        )

    def _selected_perturbation_joints(
        self,
        batch: dict[str, Any],
        scale: float,
    ) -> list[torch.LongTensor]:
        selected: list[torch.LongTensor] = []
        probability = self.perturbation.row_probability * scale
        for row, count_tensor in enumerate(batch["joint_count"]):
            count = int(count_tensor.item())
            valid = batch["perturb_valid_mask"][row, :count]
            candidates = torch.nonzero(valid, as_tuple=False).flatten()
            if (
                candidates.numel() == 0
                or torch.rand((), device=candidates.device) >= probability
            ):
                selected.append(candidates[:0])
                continue
            fraction_cap = max(
                1,
                int(np.ceil(count * self.perturbation.max_joint_fraction)),
            )
            count_cap = min(
                int(candidates.numel()),
                self.perturbation.max_perturbed_joints,
                fraction_cap,
            )
            perturb_count = int(
                torch.randint(
                    1,
                    count_cap + 1,
                    (),
                    device=candidates.device,
                ).item()
            )
            order = torch.randperm(candidates.numel(), device=candidates.device)
            selected.append(candidates[order[:perturb_count]])
        return selected

    def build_decoder_input_ids(
        self,
        batch: dict[str, Any],
        refs: TrackableSurfaceReferences,
        *,
        force_scale: float | None = None,
    ) -> tuple[torch.LongTensor, dict[str, torch.Tensor]]:
        clean_ids = batch["target_ids"]
        scale = self.perturbation_scale() if force_scale is None else float(force_scale)
        if scale <= 0.0:
            zero = clean_ids.new_zeros((), dtype=torch.float32)
            return clean_ids, {
                "perturb_selected_joints": zero,
                "perturb_changed_joints": zero,
                "perturb_changed_fraction": zero,
                "perturb_scale": zero,
            }

        selected_rows = self._selected_perturbation_joints(batch, scale)
        selected_total = sum(int(values.numel()) for values in selected_rows)
        if selected_total == 0:
            zero = clean_ids.new_zeros((), dtype=torch.float32)
            return clean_ids, {
                "perturb_selected_joints": zero,
                "perturb_changed_joints": zero,
                "perturb_changed_fraction": zero,
                "perturb_scale": zero + scale,
            }

        query_points = self._query_surface_points(
            batch["frame_vertices"][:, 0],
            batch["faces"],
            refs,
        ).float()
        decoder_ids = clean_ids.clone()
        changed_total = 0
        axial_displacements: list[torch.Tensor] = []
        radial_displacements: list[torch.Tensor] = []

        for row, selected in enumerate(selected_rows):
            if selected.numel() == 0:
                continue
            joints = batch["target_joints"][row, selected].float()
            axes = batch["perturb_axes"][row, selected].float()
            lengths = batch["perturb_lengths"][row, selected].float()

            random_direction = torch.randn_like(axes)
            radial = random_direction - (
                random_direction * axes
            ).sum(dim=-1, keepdim=True) * axes
            radial_norm = radial.norm(dim=-1, keepdim=True)
            degenerate = radial_norm.squeeze(-1) < 1.0e-6
            if degenerate.any():
                basis = torch.zeros_like(axes[degenerate])
                basis[:, 0] = 1.0
                parallel = axes[degenerate, 0].abs() > 0.8
                basis[parallel] = axes.new_tensor([0.0, 1.0, 0.0])
                radial[degenerate] = torch.cross(
                    axes[degenerate],
                    basis,
                    dim=-1,
                )
            radial = radial / radial.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
            side = torch.cross(axes, radial, dim=-1)

            offsets = query_points[row].unsqueeze(0) - joints.unsqueeze(1)
            ray_distance = (offsets * radial.unsqueeze(1)).sum(dim=-1)
            axial_offset = (offsets * axes.unsqueeze(1)).sum(dim=-1)
            side_offset = (offsets * side.unsqueeze(1)).sum(dim=-1)
            ray_score = axial_offset.square() + side_offset.square()
            ray_score = ray_score.masked_fill(ray_distance <= 1.0e-5, float("inf"))
            nearest_ray_index = ray_score.argmin(dim=1)
            clearance = ray_distance.gather(
                1,
                nearest_ray_index.unsqueeze(1),
            ).squeeze(1)
            no_positive_point = ~torch.isfinite(
                ray_score.gather(1, nearest_ray_index.unsqueeze(1)).squeeze(1)
            )
            nearest_surface = offsets.norm(dim=-1).amin(dim=1)
            clearance = torch.minimum(
                clearance,
                nearest_surface * self.perturbation.nearest_surface_multiplier,
            )
            clearance = clearance.clamp(
                min=0.0,
                max=self.perturbation.radial_clearance_cap,
            )
            clearance[no_positive_point] = 0.0

            axial_fraction = (
                torch.rand_like(lengths) * 2.0 - 1.0
            ) * self.perturbation.axial_fraction_max * scale
            radial_fraction = (
                torch.rand_like(clearance)
                * self.perturbation.radial_fraction_max
                * scale
            )
            axial_delta = axial_fraction * lengths
            radial_delta = radial_fraction * clearance
            perturbed = (
                joints
                + axial_delta.unsqueeze(1) * axes
                + radial_delta.unsqueeze(1) * radial
            )
            quantized = self._quantize(perturbed)

            positions = batch["coordinate_token_positions"][row, selected]
            previous = decoder_ids[row, positions]
            decoder_ids[row, positions] = quantized
            changed_total += int((quantized != previous).any(dim=1).sum().item())
            axial_displacements.append(axial_delta.abs())
            radial_displacements.append(radial_delta)

        selected_tensor = clean_ids.new_tensor(float(selected_total), dtype=torch.float32)
        changed_tensor = clean_ids.new_tensor(float(changed_total), dtype=torch.float32)
        axial_mean = (
            torch.cat(axial_displacements).mean()
            if axial_displacements
            else selected_tensor * 0.0
        )
        radial_mean = (
            torch.cat(radial_displacements).mean()
            if radial_displacements
            else selected_tensor * 0.0
        )
        return decoder_ids, {
            "perturb_selected_joints": selected_tensor,
            "perturb_changed_joints": changed_tensor,
            "perturb_changed_fraction": changed_tensor / selected_tensor.clamp_min(1.0),
            "perturb_axial_displacement_mean": axial_mean,
            "perturb_radial_displacement_mean": radial_mean,
            "perturb_scale": selected_tensor.new_tensor(scale),
        }

    def _stack_close_losses(
        self,
        cond: torch.Tensor,
        batch: dict[str, Any],
        decoder_input_ids: torch.LongTensor,
    ) -> dict[str, torch.Tensor]:
        cond = cond.to(dtype=self.transformer.dtype)
        attention_mask = batch["attention_mask"]
        targets = batch["target_ids"]
        batch_size = int(targets.shape[0])

        token_embeds = self.transformer.get_input_embeddings()(
            decoder_input_ids
        ).to(dtype=self.transformer.dtype)
        inputs_embeds = torch.cat([cond, token_embeds], dim=1)
        full_attention = pad(
            attention_mask,
            (cond.shape[1], 0, 0, 0),
            value=1.0,
        )
        need_action_hidden = self.stack_action_head is not None
        output = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention,
            use_cache=False,
            output_hidden_states=need_action_hidden,
        )
        logits = output.logits[:, cond.shape[1] :].reshape(
            batch_size,
            -1,
            self.tokenizer.vocab_size,
        )
        logits = logits[:, :-1]
        labels = targets[:, 1:].clone()
        labels[attention_mask[:, 1:] == 0] = -100
        ce_loss = nn.functional.cross_entropy(
            logits.permute(0, 2, 1),
            labels,
            ignore_index=-100,
        )

        predictions = logits.argmax(dim=-1)
        valid = labels != -100
        coordinate_mask = valid & (labels < self.tokenizer.num_discrete)
        close_mask = valid & (labels == self.tokenizer.token_id_close)
        eos_mask = valid & (labels == self.tokenizer.eos)

        def accuracy(mask: torch.Tensor) -> torch.Tensor:
            if not mask.any():
                return ce_loss.detach() * 0.0
            return (predictions[mask] == labels[mask]).float().mean()

        if coordinate_mask.any():
            probabilities = torch.softmax(logits[coordinate_mask].float(), dim=-1)
            coordinate_labels = labels[coordinate_mask]
            bins = torch.arange(
                self.tokenizer.num_discrete,
                device=logits.device,
                dtype=torch.float32,
            )
            distance = (
                bins.unsqueeze(0) - coordinate_labels.float().unsqueeze(1)
            ).abs() / self.tokenizer.num_discrete
            dis_loss = (
                probabilities[:, : self.tokenizer.num_discrete] * distance
            ).sum() / 50.0
        else:
            dis_loss = ce_loss.detach() * 0.0

        if self.stack_action_head is None:
            stack_action_loss = ce_loss.detach() * 0.0
            stack_action_acc = ce_loss.detach() * 0.0
            stack_action_count = ce_loss.detach() * 0.0
        else:
            assert output.hidden_states is not None
            action_targets = _stack_action_targets(
                labels,
                batch["coordinate_token_positions"],
                batch["joint_count"],
                close_token=self.tokenizer.token_id_close,
            )
            action_mask = action_targets != -100
            action_hidden = output.hidden_states[-1][
                :, cond.shape[1] : -1
            ]
            action_logits = self.stack_action_head(action_hidden, cond)[action_mask]
            stack_action_loss = nn.functional.cross_entropy(
                action_logits.float(),
                action_targets[action_mask],
            )
            stack_action_acc = (
                action_logits.argmax(dim=-1) == action_targets[action_mask]
            ).float().mean()
            stack_action_count = action_mask.sum().float()

        total_loss = ce_loss + self.stack_action_loss_weight * stack_action_loss

        return {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "dis_loss": dis_loss,
            "token_acc": accuracy(valid),
            "coordinate_acc": accuracy(coordinate_mask),
            "close_acc": accuracy(close_mask),
            "eos_acc": accuracy(eos_mask),
            "close_count": close_mask.sum().float(),
            "eos_count": eos_mask.sum().float(),
            "stack_action_loss": stack_action_loss,
            "stack_action_acc": stack_action_acc,
            "stack_action_count": stack_action_count,
        }

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        refs = self.sample_references(batch)
        decoder_input_ids, perturb_metrics = self.build_decoder_input_ids(
            batch,
            refs,
        )
        cond = self.build_condition(batch, refs=refs)
        out = self._stack_close_losses(
            cond,
            batch,
            decoder_input_ids,
        )
        out.update(perturb_metrics)
        return out

    @torch.no_grad()
    def generate_from_condition(
        self,
        cond: torch.Tensor,
        *,
        cls: str | None,
        max_new_tokens: int = 1_100,
    ) -> np.ndarray:
        if cond.shape[0] != 1:
            raise ValueError("stack-close generation currently requires batch size 1")
        ids = [self.tokenizer.bos, self.tokenizer.cls_name_to_token(cls)]
        prefix = torch.tensor([ids], device=cond.device, dtype=torch.long)
        token_embeds = self.transformer.get_input_embeddings()(prefix).to(
            dtype=self.transformer.dtype
        )
        initial_embeds = torch.cat(
            [cond.to(dtype=self.transformer.dtype), token_embeds],
            dim=1,
        )
        attention_mask = torch.ones(
            initial_embeds.shape[:2],
            device=cond.device,
            dtype=torch.long,
        )
        need_action_hidden = self.stack_action_head is not None
        output = self.transformer(
            inputs_embeds=initial_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=need_action_hidden,
        )
        for _ in range(max_new_tokens):
            logits = output.logits[0, -1].float()
            possible = self.tokenizer.next_posible_token(
                np.asarray(ids, dtype=np.int64)
            )
            if not possible:
                break
            allowed = torch.tensor(possible, device=logits.device, dtype=torch.long)
            if (
                self.stack_action_head is not None
                and self.tokenizer.token_id_close in possible
            ):
                assert output.hidden_states is not None
                action_logits = self.stack_action_head(
                    output.hidden_states[-1][:, -1:],
                    cond,
                )[0, 0]
                if int(action_logits.argmax(dim=-1).item()) == 1:
                    next_token = int(self.tokenizer.token_id_close)
                else:
                    coordinate_ids = allowed[
                        allowed < self.tokenizer.num_discrete
                    ]
                    next_token = int(
                        coordinate_ids[logits[coordinate_ids].argmax()].item()
                    )
            else:
                next_token = int(allowed[logits[allowed].argmax()].item())
            ids.append(next_token)
            if next_token == self.tokenizer.eos:
                break
            next_ids = torch.tensor(
                [[next_token]],
                device=cond.device,
                dtype=torch.long,
            )
            next_embed = self.transformer.get_input_embeddings()(next_ids).to(
                dtype=self.transformer.dtype
            )
            attention_mask = torch.ones(
                (1, cond.shape[1] + len(ids)),
                device=cond.device,
                dtype=torch.long,
            )
            output = self.transformer(
                inputs_embeds=next_embed,
                attention_mask=attention_mask,
                past_key_values=output.past_key_values,
                use_cache=True,
                output_hidden_states=need_action_hidden,
            )
        return np.asarray(ids, dtype=np.int64)

    @torch.no_grad()
    def generate_batch_item(
        self,
        batch: dict[str, Any],
        *,
        row: int = 0,
        max_new_tokens: int = 1_100,
    ) -> np.ndarray:
        refs = self.sample_references(batch)
        cond = self.build_condition(batch, refs=refs)
        return self.generate_from_condition(
            cond[row : row + 1],
            cls=batch["cls"][row],
            max_new_tokens=max_new_tokens,
        )
