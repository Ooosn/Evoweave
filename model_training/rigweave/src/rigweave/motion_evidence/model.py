"""Isolated flat-UniRig route with separate static and motion memories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn
from torch.nn.functional import pad

from rigweave.dynamic_rig.sampling import (
    TrackableSurfaceReferences,
    materialize_trackable_surface,
    sample_trackable_surface,
)

from .attention import MotionEvidenceCrossAttention
from .encoder import (
    MotionEvidenceValues,
    TopologyLocalMotionEvidence,
    TopologyMotionValueEncoder,
)
from .supervision import (
    query_aligned_skin_boundary_targets,
    query_aligned_skin_weights,
)


@dataclass(frozen=True)
class MotionEvidenceMemory:
    """Aligned static addresses and motion-only values for one batch."""

    static_tokens: torch.Tensor
    motion_values: torch.Tensor
    boundary_logits: torch.Tensor
    anchor_confidence: torch.Tensor
    confidence: torch.Tensor
    raw_evidence: MotionEvidenceValues

    def controlled(
        self,
        control: str,
        *,
        generator: torch.Generator | None = None,
    ) -> "MotionEvidenceMemory":
        if control == "normal":
            return self
        if control == "zero":
            return MotionEvidenceMemory(
                static_tokens=self.static_tokens,
                motion_values=torch.zeros_like(self.motion_values),
                boundary_logits=torch.zeros_like(self.boundary_logits),
                anchor_confidence=torch.zeros_like(self.anchor_confidence),
                confidence=torch.zeros_like(self.confidence),
                raw_evidence=self.raw_evidence,
            )
        if control != "corrupt_correspondence":
            raise ValueError(f"unknown motion evidence control {control!r}")

        rows = []
        boundary_rows = []
        confidence_rows = []
        for batch_index in range(self.motion_values.shape[0]):
            permutation_device = (
                self.motion_values.device if generator is None else generator.device
            )
            permutation = torch.randperm(
                self.motion_values.shape[1],
                device=permutation_device,
                generator=generator,
            ).to(self.motion_values.device)
            rows.append(self.motion_values[batch_index, permutation])
            boundary_rows.append(self.boundary_logits[batch_index, permutation])
            confidence_rows.append(self.anchor_confidence[batch_index, permutation])
        return MotionEvidenceMemory(
            static_tokens=self.static_tokens,
            motion_values=torch.stack(rows, dim=0),
            boundary_logits=torch.stack(boundary_rows, dim=0),
            anchor_confidence=torch.stack(confidence_rows, dim=0),
            confidence=self.confidence,
            raw_evidence=self.raw_evidence,
        )


@dataclass(frozen=True)
class MotionEvidenceTeacherForcingOutput:
    logits: torch.Tensor
    baseline_logits: torch.Tensor
    token_hidden: torch.Tensor
    refined_hidden: torch.Tensor


class StaticQueryMotionEvidenceConditioner(nn.Module):
    """Build Q from the query pose and E from topology-local deformation."""

    def __init__(
        self,
        surface_tokenizer: nn.Module,
        hidden_size: int,
        *,
        evidence_intermediate_size: int | None = None,
        active_threshold: float = 1.0e-3,
        confidence_scale: float = 1.0e-2,
    ) -> None:
        super().__init__()
        self.surface_tokenizer = surface_tokenizer
        self.extractor = TopologyLocalMotionEvidence(
            active_threshold=active_threshold,
            confidence_scale=confidence_scale,
        )
        self.value_encoder = TopologyMotionValueEncoder(
            hidden_size,
            intermediate_size=evidence_intermediate_size,
        )

    def forward(
        self,
        frame_vertices: torch.Tensor,
        faces: torch.LongTensor,
        refs: TrackableSurfaceReferences,
        *,
        vertex_normals: torch.Tensor | None = None,
        face_normals: torch.Tensor | None = None,
        vertex_counts: torch.LongTensor | None = None,
        face_counts: torch.LongTensor | None = None,
    ) -> MotionEvidenceMemory:
        query_vertex_normals = None if vertex_normals is None else vertex_normals[:, 0]
        query_face_normals = None if face_normals is None else face_normals[:, 0]
        query_samples = materialize_trackable_surface(
            frame_vertices[:, 0],
            faces,
            refs,
            vertex_normals=query_vertex_normals,
            face_normals=query_face_normals,
        )
        static_tokens = self.surface_tokenizer(
            query_samples.dense_points,
            query_samples.dense_normals,
            query_samples.query_points,
            query_samples.query_normals,
        )
        raw = self.extractor(
            frame_vertices,
            faces,
            refs,
            vertex_counts=vertex_counts,
            face_counts=face_counts,
        )
        encoded = self.value_encoder.forward_with_boundary(
            raw.query_features,
            raw.anchor_confidence,
        )
        motion_values = encoded.values
        if static_tokens.shape != motion_values.shape:
            raise RuntimeError(
                "static query tokens and motion values are not aligned: "
                f"{tuple(static_tokens.shape)} != {tuple(motion_values.shape)}"
            )
        return MotionEvidenceMemory(
            static_tokens=static_tokens,
            motion_values=motion_values,
            boundary_logits=encoded.boundary_logits,
            anchor_confidence=raw.anchor_confidence,
            confidence=raw.confidence,
            raw_evidence=raw,
        )


class MotionEvidenceDecoderAdapter(nn.Module):
    """Read E after the causal decoder has formed a prefix-dependent state."""

    def __init__(
        self,
        hidden_size: int,
        heads: int,
        *,
        residual_scale: float = 0.1,
        static_prefix_steps: int = 4,
    ) -> None:
        super().__init__()
        if static_prefix_steps < 0:
            raise ValueError("static_prefix_steps must be non-negative")
        self.static_prefix_steps = int(static_prefix_steps)
        self.attention = MotionEvidenceCrossAttention(
            hidden_size,
            heads,
            residual_scale=residual_scale,
            detach_static_keys=True,
        )

    def refine_hidden(
        self,
        prefix_hidden: torch.Tensor,
        memory: MotionEvidenceMemory,
        prefix_positions: torch.LongTensor,
    ) -> torch.Tensor:
        if prefix_positions.shape != (prefix_hidden.shape[1],):
            raise ValueError(
                "prefix_positions must identify every decoder position, "
                f"got {tuple(prefix_positions.shape)} for length {prefix_hidden.shape[1]}"
            )
        refined = self.attention(
            prefix_hidden,
            memory.static_tokens,
            memory.motion_values,
        )
        use_evidence = prefix_positions >= self.static_prefix_steps
        return torch.where(use_evidence[None, :, None], refined, prefix_hidden)

    def logits_from_hidden(
        self,
        transformer: nn.Module,
        prefix_hidden: torch.Tensor,
        memory: MotionEvidenceMemory,
        prefix_positions: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        refined = self.refine_hidden(prefix_hidden, memory, prefix_positions)
        return self.project_hidden(transformer, refined), refined

    @staticmethod
    def project_hidden(transformer: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
        """Project hidden states through the precision-stable vocabulary head."""

        output_embedding = transformer.get_output_embeddings()
        if output_embedding is None:
            raise TypeError("causal transformer does not expose output embeddings")
        weight = getattr(output_embedding, "weight", None)
        projection_dtype = weight.dtype if isinstance(weight, torch.Tensor) else hidden.dtype
        # Teacher forcing projects many prefix positions while generation projects
        # one. Keep this projection out of autocast so both shapes use the same
        # parameter precision instead of shape-dependent bf16 GEMM rounding.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return output_embedding(hidden.to(dtype=projection_dtype))

    def teacher_forcing(
        self,
        transformer: nn.Module,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        memory: MotionEvidenceMemory,
        *,
        token_embedder: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> MotionEvidenceTeacherForcingOutput:
        if token_embedder is None:
            input_embedding = transformer.get_input_embeddings()
            token_embeds = input_embedding(input_ids)
        else:
            token_embeds = token_embedder(input_ids, attention_mask)
        transformer_dtype = getattr(transformer, "dtype", memory.static_tokens.dtype)
        static_tokens = memory.static_tokens.to(dtype=transformer_dtype)
        token_embeds = token_embeds.to(dtype=transformer_dtype)
        inputs_embeds = torch.cat((static_tokens, token_embeds), dim=1)
        full_attention = pad(attention_mask, (static_tokens.shape[1], 0), value=1.0)
        output = transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention,
            use_cache=False,
            output_hidden_states=True,
        )
        if output.hidden_states is None:
            raise RuntimeError("causal transformer did not return hidden states")
        token_hidden = output.hidden_states[-1][:, static_tokens.shape[1] :]
        prefix_positions = torch.arange(token_hidden.shape[1], device=token_hidden.device)
        logits, refined = self.logits_from_hidden(
            transformer,
            token_hidden,
            memory,
            prefix_positions,
        )
        baseline_logits = output.logits[:, static_tokens.shape[1] :]
        static_steps = min(self.static_prefix_steps, logits.shape[1])
        if static_steps:
            logits = torch.cat(
                (baseline_logits[:, :static_steps], logits[:, static_steps:]),
                dim=1,
            )
        zero_rows = memory.anchor_confidence.amax(dim=1) == 0
        if zero_rows.any():
            logits = torch.where(zero_rows[:, None, None], baseline_logits, logits)
        return MotionEvidenceTeacherForcingOutput(
            logits=logits,
            baseline_logits=baseline_logits,
            token_hidden=token_hidden,
            refined_hidden=refined,
        )

    def generation_step(
        self,
        transformer: nn.Module,
        transformer_output: Any,
        memory: MotionEvidenceMemory,
        *,
        prefix_position: int,
    ) -> torch.Tensor:
        """Project the newest causal hidden state through the same E path."""

        if prefix_position < 0:
            raise ValueError("prefix_position must be non-negative")
        if transformer_output.hidden_states is None:
            raise RuntimeError("generation requires output_hidden_states=True")
        hidden = transformer_output.hidden_states[-1][:, -1:]
        positions = torch.tensor([prefix_position], device=hidden.device)
        logits, _ = self.logits_from_hidden(transformer, hidden, memory, positions)
        baseline_logits = transformer_output.logits[:, -1:]
        if prefix_position < self.static_prefix_steps:
            return baseline_logits[:, -1]
        zero_rows = memory.anchor_confidence.amax(dim=1) == 0
        if zero_rows.any():
            logits = torch.where(zero_rows[:, None, None], baseline_logits, logits)
        return logits[:, -1]


class TopologyMotionEvidenceUniRigAR(nn.Module):
    """Training-capable route that leaves DynamicRigUniRigAR unchanged."""

    def __init__(
        self,
        unirig_ar: nn.Module,
        surface_tokenizer: nn.Module,
        tokenizer: Any,
        *,
        num_surface_samples: int = 65536,
        vertex_samples: int = 8192,
        query_tokens: int = 1024,
        evidence_heads: int = 8,
        evidence_residual_scale: float = 0.1,
        evidence_static_prefix_steps: int = 4,
        evidence_intermediate_size: int | None = None,
        active_threshold: float = 1.0e-3,
        confidence_scale: float = 1.0e-2,
        eos_loss_weight: float = 1.0,
        boundary_loss_weight: float = 1.0,
        attention_alignment_loss_weight: float = 1.0,
        counterfactual_correspondence_loss_weight: float = 0.0,
        counterfactual_gain_loss_weight: float = 0.0,
        terminal_noharm_loss_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.unirig_ar = unirig_ar
        self.tokenizer = tokenizer
        self.num_surface_samples = int(num_surface_samples)
        self.vertex_samples = int(vertex_samples)
        self.query_tokens = int(query_tokens)
        self.eos_loss_weight = float(eos_loss_weight)
        if boundary_loss_weight < 0.0:
            raise ValueError("boundary_loss_weight must be non-negative")
        self.boundary_loss_weight = float(boundary_loss_weight)
        if attention_alignment_loss_weight < 0.0:
            raise ValueError("attention_alignment_loss_weight must be non-negative")
        self.attention_alignment_loss_weight = float(attention_alignment_loss_weight)
        counterfactual_weights = (
            counterfactual_correspondence_loss_weight,
            counterfactual_gain_loss_weight,
            terminal_noharm_loss_weight,
        )
        if any(weight < 0.0 for weight in counterfactual_weights):
            raise ValueError("counterfactual loss weights must be non-negative")
        self.counterfactual_correspondence_loss_weight = float(
            counterfactual_correspondence_loss_weight
        )
        self.counterfactual_gain_loss_weight = float(counterfactual_gain_loss_weight)
        self.terminal_noharm_loss_weight = float(terminal_noharm_loss_weight)
        hidden_size = int(unirig_ar.hidden_size)
        self.conditioner = StaticQueryMotionEvidenceConditioner(
            surface_tokenizer,
            hidden_size,
            evidence_intermediate_size=evidence_intermediate_size,
            active_threshold=active_threshold,
            confidence_scale=confidence_scale,
        )
        self.evidence_adapter = MotionEvidenceDecoderAdapter(
            hidden_size,
            evidence_heads,
            residual_scale=evidence_residual_scale,
            static_prefix_steps=evidence_static_prefix_steps,
        )

    @property
    def transformer(self) -> nn.Module:
        return self.unirig_ar.transformer

    def sample_references(self, batch: dict[str, Any]) -> TrackableSurfaceReferences:
        return sample_trackable_surface(
            batch["frame_vertices"][:, 0],
            batch["faces"],
            num_samples=self.num_surface_samples,
            vertex_samples=self.vertex_samples,
            query_tokens=self.query_tokens,
            vertex_counts=batch.get("vertex_count"),
            face_counts=batch.get("face_count"),
        )

    def build_memory(
        self,
        batch: dict[str, Any],
        *,
        refs: TrackableSurfaceReferences | None = None,
        control: str = "normal",
        generator: torch.Generator | None = None,
    ) -> MotionEvidenceMemory:
        if refs is None:
            refs = self.sample_references(batch)
        memory = self.conditioner(
            batch["frame_vertices"],
            batch["faces"],
            refs,
            vertex_normals=batch.get("vertex_normals"),
            face_normals=batch.get("face_normals"),
            vertex_counts=batch.get("vertex_count"),
            face_counts=batch.get("face_count"),
        )
        return memory.controlled(control, generator=generator)

    def teacher_forcing(
        self,
        batch: dict[str, Any],
        *,
        memory: MotionEvidenceMemory | None = None,
        refs: TrackableSurfaceReferences | None = None,
        control: str = "normal",
        generator: torch.Generator | None = None,
    ) -> MotionEvidenceTeacherForcingOutput:
        if memory is None:
            memory = self.build_memory(
                batch,
                refs=refs,
                control=control,
                generator=generator,
            )
        return self.evidence_adapter.teacher_forcing(
            self.transformer,
            batch["input_ids"],
            batch["attention_mask"],
            memory,
        )

    def _token_losses(
        self,
        logits: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        logits = logits[:, :-1]
        labels = input_ids[:, 1:].clone()
        labels[attention_mask[:, 1:] == 0] = -100
        if self.eos_loss_weight == 1.0:
            ce_loss = nn.functional.cross_entropy(logits.transpose(1, 2), labels)
        else:
            flat_logits = logits.reshape(-1, logits.shape[-1])
            flat_labels = labels.reshape(-1)
            losses = nn.functional.cross_entropy(
                flat_logits,
                flat_labels,
                ignore_index=-100,
                reduction="none",
            )
            valid = flat_labels != -100
            weights = torch.ones_like(losses)
            weights[flat_labels == int(self.tokenizer.eos)] = self.eos_loss_weight
            ce_loss = (losses[valid] * weights[valid]).sum() / weights[valid].sum().clamp_min(1.0)

        valid = labels != -100
        token_acc = (logits.argmax(dim=-1)[valid] == labels[valid]).float().mean()
        num_discrete = int(self.tokenizer.num_discrete)
        coordinate_mask = valid & (labels < num_discrete)
        if coordinate_mask.any():
            probabilities = nn.functional.softmax(logits[..., :num_discrete], dim=-1)
            bins = torch.arange(num_discrete, device=logits.device).view(1, 1, -1)
            distances = (bins - labels.clamp_min(0).unsqueeze(-1)).abs().float() / num_discrete
            dis_loss = (probabilities * distances)[coordinate_mask].sum() / 50.0
        else:
            dis_loss = torch.zeros((), device=logits.device, dtype=ce_loss.dtype)
        return {
            "loss": ce_loss,
            "ce_loss": ce_loss,
            "dis_loss": dis_loss,
            "token_acc": token_acc,
        }

    def boundary_auxiliary_loss(
        self,
        batch: dict[str, Any],
        refs: TrackableSurfaceReferences,
        memory: MotionEvidenceMemory,
    ) -> dict[str, torch.Tensor]:
        skin_weights = batch.get("target_skin_weights")
        if skin_weights is None:
            raise KeyError("motion-evidence training requires target_skin_weights")
        targets = query_aligned_skin_boundary_targets(
            skin_weights,
            batch["faces"],
            refs,
            vertex_counts=batch.get("vertex_count"),
            face_counts=batch.get("face_count"),
        )
        if targets.values.shape != memory.boundary_logits.shape:
            raise RuntimeError(
                "boundary logits and targets are not query aligned: "
                f"{tuple(memory.boundary_logits.shape)} != {tuple(targets.values.shape)}"
            )
        per_value = nn.functional.binary_cross_entropy_with_logits(
            memory.boundary_logits.float(),
            targets.values.float(),
            reduction="none",
        ).mean(dim=-1)
        weights = (
            targets.valid_mask.float()
            * memory.anchor_confidence.float()
        )
        loss = (per_value * weights).sum() / weights.sum().clamp_min(1.0)
        predictions = torch.sigmoid(memory.boundary_logits.float())
        mae = (
            (predictions - targets.values.float()).abs().mean(dim=-1) * weights
        ).sum() / weights.sum().clamp_min(1.0)
        return {
            "boundary_loss": loss,
            "boundary_mae": mae,
            "boundary_valid_fraction": targets.valid_mask.float().mean(),
        }

    def attention_alignment_loss(
        self,
        batch: dict[str, Any],
        refs: TrackableSurfaceReferences,
        memory: MotionEvidenceMemory,
        token_hidden: torch.Tensor,
        *,
        eps: float = 1.0e-8,
    ) -> dict[str, torch.Tensor]:
        """Supervise each joint-producing prefix step to read its skinned region."""

        skin_weights = batch.get("target_skin_weights")
        if skin_weights is None:
            raise KeyError("motion-evidence training requires target_skin_weights")
        token_joint_indices = batch.get("token_joint_indices")
        if token_joint_indices is None:
            raise KeyError("motion-evidence training requires token_joint_indices")
        if token_joint_indices.shape != token_hidden.shape[:2]:
            raise ValueError(
                "token_joint_indices must identify every causal hidden position, "
                f"got {tuple(token_joint_indices.shape)} for {tuple(token_hidden.shape[:2])}"
            )
        attention_mask = batch["attention_mask"]
        if attention_mask.shape != token_joint_indices.shape:
            raise ValueError(
                "attention_mask and token_joint_indices must have identical shape, "
                f"got {tuple(attention_mask.shape)} and {tuple(token_joint_indices.shape)}"
            )

        query_skin = query_aligned_skin_weights(
            skin_weights,
            batch["faces"],
            refs,
        )
        if query_skin.shape[:2] != memory.static_tokens.shape[:2]:
            raise RuntimeError(
                "query skin targets and static keys are not anchor aligned: "
                f"{tuple(query_skin.shape[:2])} != {tuple(memory.static_tokens.shape[:2])}"
            )
        joint_count = query_skin.shape[-1]
        if joint_count <= 0:
            raise ValueError("target_skin_weights must contain at least one joint column")

        joint_indices = token_joint_indices.to(device=query_skin.device, dtype=torch.long)
        valid_joint = (joint_indices >= 0) & (joint_indices < joint_count)
        safe_joint_indices = joint_indices.clamp(min=0, max=joint_count - 1)
        skin_by_joint = query_skin.transpose(1, 2)
        targets = torch.gather(
            skin_by_joint,
            1,
            safe_joint_indices[..., None].expand(-1, -1, query_skin.shape[1]),
        )
        target_mass = targets.sum(dim=-1)
        targets = targets / target_mass[..., None].clamp_min(eps)

        attention = self.evidence_adapter.attention.attention_weights(
            token_hidden,
            memory.static_tokens,
            memory.motion_values,
        )
        expected_shape = (
            token_hidden.shape[0],
            token_hidden.shape[1],
            query_skin.shape[1],
        )
        if attention.shape[0] != expected_shape[0] or attention.shape[2:] != expected_shape[1:]:
            raise RuntimeError(
                "attention weights are not prefix-to-query aligned: "
                f"got {tuple(attention.shape)}, expected (*,{expected_shape[1]},{expected_shape[2]})"
            )
        mean_attention = attention.float().mean(dim=1).clamp_min(eps)
        per_step_kl = (
            targets
            * (targets.clamp_min(eps).log() - mean_attention.log())
        ).sum(dim=-1)

        prediction_valid = torch.zeros_like(valid_joint)
        prediction_valid[:, :-1] = attention_mask[:, 1:] != 0
        positions = torch.arange(token_hidden.shape[1], device=token_hidden.device)[None]
        later_position = positions >= self.evidence_adapter.static_prefix_steps
        supervised = (
            valid_joint
            & prediction_valid
            & later_position
            & (target_mass > eps)
        )
        target_observability = (
            targets * memory.anchor_confidence.float()[:, None, :]
        ).sum(dim=-1)
        loss_weights = supervised.float() * target_observability
        weight_sum = loss_weights.sum()
        loss = (per_step_kl * loss_weights).sum() / weight_sum.clamp_min(1.0)

        attention_entropy = -(mean_attention * mean_attention.log()).sum(dim=-1)
        if mean_attention.shape[-1] > 1:
            attention_entropy = attention_entropy / torch.log(
                mean_attention.new_tensor(float(mean_attention.shape[-1]))
            )
        target_peak = targets.max(dim=-1).values
        later_valid_count = (prediction_valid & later_position).sum().clamp_min(1)
        return {
            "attention_alignment_loss": loss,
            "attention_alignment_valid_fraction": supervised.float().sum()
            / later_valid_count,
            "attention_alignment_entropy": (
                attention_entropy * loss_weights
            ).sum()
            / weight_sum.clamp_min(1.0),
            "attention_alignment_target_peak": (target_peak * loss_weights).sum()
            / weight_sum.clamp_min(1.0),
        }

    def counterfactual_condition_losses(
        self,
        batch: dict[str, Any],
        refs: TrackableSurfaceReferences,
        memory: MotionEvidenceMemory,
        teacher: MotionEvidenceTeacherForcingOutput,
        *,
        generator: torch.Generator | None = None,
        margin: float = 1.0e-3,
        temperature: float = 1.0e-2,
        eps: float = 1.0e-8,
    ) -> dict[str, torch.Tensor]:
        """Make correct local correspondence useful and terminal states abstain."""

        if margin < 0.0 or temperature <= 0.0:
            raise ValueError(
                "counterfactual margin must be non-negative and temperature positive"
            )
        skin_weights = batch.get("target_skin_weights")
        token_joint_indices = batch.get("token_joint_indices")
        if skin_weights is None or token_joint_indices is None:
            raise KeyError(
                "counterfactual evidence training requires skin weights and token-joint indices"
            )

        query_skin = query_aligned_skin_weights(
            skin_weights,
            batch["faces"],
            refs,
        )
        joint_count = query_skin.shape[-1]
        joint_indices = token_joint_indices.to(device=query_skin.device, dtype=torch.long)
        valid_joint = (joint_indices >= 0) & (joint_indices < joint_count)
        safe_joint_indices = joint_indices.clamp(min=0, max=joint_count - 1)
        targets = torch.gather(
            query_skin.transpose(1, 2),
            1,
            safe_joint_indices[..., None].expand(-1, -1, query_skin.shape[1]),
        )
        target_mass = targets.sum(dim=-1)
        targets = targets / target_mass[..., None].clamp_min(eps)
        step_observability = (
            targets * memory.anchor_confidence.float()[:, None, :]
        ).sum(dim=-1)

        positions = torch.arange(
            teacher.token_hidden.shape[1],
            device=teacher.token_hidden.device,
        )
        corrupted_memory = memory.controlled(
            "corrupt_correspondence",
            generator=generator,
        )
        corrupt_logits, _ = self.evidence_adapter.logits_from_hidden(
            self.transformer,
            teacher.token_hidden,
            corrupted_memory,
            positions,
        )
        zero_logits = self.evidence_adapter.project_hidden(
            self.transformer,
            teacher.token_hidden,
        ).detach()

        labels = batch["input_ids"][:, 1:]
        prediction_valid = batch["attention_mask"][:, 1:] != 0
        prediction_positions = positions[:-1][None]
        later = prediction_positions >= self.evidence_adapter.static_prefix_steps

        def token_nll(logits: torch.Tensor) -> torch.Tensor:
            return nn.functional.cross_entropy(
                logits[:, :-1].float().transpose(1, 2),
                labels,
                reduction="none",
            )

        normal_nll = token_nll(teacher.logits)
        corrupt_nll = token_nll(corrupt_logits)
        zero_nll = token_nll(zero_logits)
        supervised = (
            prediction_valid
            & later
            & valid_joint[:, :-1]
            & (target_mass[:, :-1] > eps)
        )
        weights = supervised.float() * step_observability[:, :-1]
        weight_sum = weights.sum().clamp_min(1.0)
        correspondence = nn.functional.softplus(
            (normal_nll - corrupt_nll + margin) / temperature
        ) * temperature
        gain = nn.functional.softplus(
            (normal_nll - zero_nll + margin) / temperature
        ) * temperature
        correspondence_loss = (correspondence * weights).sum() / weight_sum
        gain_loss = (gain * weights).sum() / weight_sum

        terminal = prediction_valid & later & (labels == int(self.tokenizer.eos))
        if terminal.any():
            terminal_noharm_loss = nn.functional.relu(
                normal_nll[terminal] - zero_nll[terminal]
            ).mean()
        else:
            terminal_noharm_loss = normal_nll.new_zeros(())
        return {
            "counterfactual_correspondence_loss": correspondence_loss,
            "counterfactual_gain_loss": gain_loss,
            "terminal_noharm_loss": terminal_noharm_loss,
            "counterfactual_observable_step_fraction": supervised.float().mean(),
            "counterfactual_corrupt_minus_normal_nll": (
                (corrupt_nll - normal_nll) * weights
            ).sum()
            / weight_sum,
            "counterfactual_zero_minus_normal_nll": (
                (zero_nll - normal_nll) * weights
            ).sum()
            / weight_sum,
        }

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        refs = self.sample_references(batch)
        memory = self.build_memory(batch, refs=refs)
        teacher = self.teacher_forcing(batch, memory=memory)
        out = self._token_losses(
            teacher.logits,
            batch["input_ids"],
            batch["attention_mask"],
        )
        hidden_delta = teacher.refined_hidden.float() - teacher.token_hidden.float()
        out["evidence_hidden_delta_rms"] = torch.sqrt(hidden_delta.square().mean())
        boundary = self.boundary_auxiliary_loss(batch, refs, memory)
        out.update(boundary)
        alignment = self.attention_alignment_loss(
            batch,
            refs,
            memory,
            teacher.token_hidden,
        )
        out.update(alignment)
        counterfactual_weight = (
            self.counterfactual_correspondence_loss_weight
            + self.counterfactual_gain_loss_weight
            + self.terminal_noharm_loss_weight
        )
        if counterfactual_weight > 0.0:
            counterfactual = self.counterfactual_condition_losses(
                batch,
                refs,
                memory,
                teacher,
            )
            out.update(counterfactual)
        out["loss"] = (
            out["ce_loss"]
            + self.boundary_loss_weight * boundary["boundary_loss"]
            + self.attention_alignment_loss_weight
            * alignment["attention_alignment_loss"]
        )
        if counterfactual_weight > 0.0:
            out["loss"] = (
                out["loss"]
                + self.counterfactual_correspondence_loss_weight
                * counterfactual["counterfactual_correspondence_loss"]
                + self.counterfactual_gain_loss_weight
                * counterfactual["counterfactual_gain_loss"]
                + self.terminal_noharm_loss_weight
                * counterfactual["terminal_noharm_loss"]
            )
        return out
