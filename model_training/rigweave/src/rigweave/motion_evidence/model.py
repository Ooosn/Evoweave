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


@dataclass(frozen=True)
class MotionEvidenceMemory:
    """Aligned static addresses and motion-only values for one batch."""

    static_tokens: torch.Tensor
    motion_values: torch.Tensor
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
                confidence=torch.zeros_like(self.confidence),
                raw_evidence=self.raw_evidence,
            )
        if control != "corrupt_correspondence":
            raise ValueError(f"unknown motion evidence control {control!r}")

        rows = []
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
        return MotionEvidenceMemory(
            static_tokens=self.static_tokens,
            motion_values=torch.stack(rows, dim=0),
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
        motion_values = self.value_encoder(raw.query_features, raw.confidence)
        if static_tokens.shape != motion_values.shape:
            raise RuntimeError(
                "static query tokens and motion values are not aligned: "
                f"{tuple(static_tokens.shape)} != {tuple(motion_values.shape)}"
            )
        return MotionEvidenceMemory(
            static_tokens=static_tokens,
            motion_values=motion_values,
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
        gate_init: float = 1.0e-2,
    ) -> None:
        super().__init__()
        self.attention = MotionEvidenceCrossAttention(
            hidden_size,
            heads,
            gate_init=gate_init,
            detach_static_keys=True,
        )

    def refine_hidden(
        self,
        prefix_hidden: torch.Tensor,
        memory: MotionEvidenceMemory,
    ) -> torch.Tensor:
        return self.attention(
            prefix_hidden,
            memory.static_tokens,
            memory.motion_values,
            memory.confidence,
        )

    def logits_from_hidden(
        self,
        transformer: nn.Module,
        prefix_hidden: torch.Tensor,
        memory: MotionEvidenceMemory,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output_embedding = transformer.get_output_embeddings()
        if output_embedding is None:
            raise TypeError("causal transformer does not expose output embeddings")
        refined = self.refine_hidden(prefix_hidden, memory)
        weight = getattr(output_embedding, "weight", None)
        projection_dtype = weight.dtype if isinstance(weight, torch.Tensor) else refined.dtype
        # Teacher forcing projects many prefix positions while generation projects
        # one. Keep this projection out of autocast so both shapes use the same
        # parameter precision instead of shape-dependent bf16 GEMM rounding.
        with torch.autocast(device_type=refined.device.type, enabled=False):
            logits = output_embedding(refined.to(dtype=projection_dtype))
        return logits, refined

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
        logits, refined = self.logits_from_hidden(transformer, token_hidden, memory)
        baseline_logits = output.logits[:, static_tokens.shape[1] :]
        zero_rows = memory.confidence == 0
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
    ) -> torch.Tensor:
        """Project the newest causal hidden state through the same E path."""

        if transformer_output.hidden_states is None:
            raise RuntimeError("generation requires output_hidden_states=True")
        hidden = transformer_output.hidden_states[-1][:, -1:]
        logits, _ = self.logits_from_hidden(transformer, hidden, memory)
        baseline_logits = transformer_output.logits[:, -1:]
        zero_rows = memory.confidence == 0
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
        evidence_gate_init: float = 1.0e-2,
        evidence_intermediate_size: int | None = None,
        active_threshold: float = 1.0e-3,
        confidence_scale: float = 1.0e-2,
        eos_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.unirig_ar = unirig_ar
        self.tokenizer = tokenizer
        self.num_surface_samples = int(num_surface_samples)
        self.vertex_samples = int(vertex_samples)
        self.query_tokens = int(query_tokens)
        self.eos_loss_weight = float(eos_loss_weight)
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
            gate_init=evidence_gate_init,
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

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        teacher = self.teacher_forcing(batch)
        out = self._token_losses(
            teacher.logits,
            batch["input_ids"],
            batch["attention_mask"],
        )
        hidden_delta = teacher.refined_hidden.float() - teacher.token_hidden.float()
        out["evidence_hidden_delta_rms"] = torch.sqrt(hidden_delta.square().mean())
        return out
