from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rigweave.dynamic_rig.sampling import TrackableSurfaceReferences
from rigweave.motion_evidence import (
    CoverageAwareTopologyMotionEvidenceUniRigAR,
    MotionEvidenceDecoderAdapter,
    MotionEvidenceMemory,
    StaticQueryMotionEvidenceConditioner,
    TopologyMotionEvidenceUniRigAR,
)


class _SurfaceTokenizer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(6, hidden_size)

    def forward(
        self,
        dense_points: torch.Tensor,
        dense_normals: torch.Tensor,
        query_points: torch.Tensor,
        query_normals: torch.Tensor,
    ) -> torch.Tensor:
        del dense_points, dense_normals
        return self.projection(torch.cat((query_points, query_normals), dim=-1))


class _CausalTransformer(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.core = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.output = nn.Linear(hidden_size, vocab_size, bias=False)

    @property
    def dtype(self) -> torch.dtype:
        return self.embedding.weight.dtype

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def get_output_embeddings(self) -> nn.Module:
        return self.output

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        output_hidden_states: bool,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        hidden = self.core(inputs_embeds)
        return SimpleNamespace(
            logits=self.output(hidden),
            hidden_states=(hidden,) if output_hidden_states else None,
        )


class _UniRig(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.transformer = _CausalTransformer(hidden_size, vocab_size)


class _Tokenizer:
    num_discrete = 16
    eos = 19
    vocab_size = 24


def _references(device: torch.device) -> TrackableSurfaceReferences:
    return TrackableSurfaceReferences(
        vertex_indices=torch.tensor([[0, 1, 2, 3]], device=device),
        face_indices=torch.tensor([[0]], device=device),
        barycentric=torch.tensor([[[0.25, 0.25, 0.50]]], device=device),
        query_indices=torch.tensor([[0, 1, 2, 3, 4]], device=device),
    )


def _batch(device: torch.device) -> dict[str, torch.Tensor]:
    query = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        device=device,
    )
    moved = query.clone()
    moved[2:] += torch.tensor([0.3, 0.6, 0.0], device=device)
    frames = torch.stack((query, moved), dim=0)[None]
    faces = torch.tensor([[[0, 1, 2], [0, 2, 3]]], device=device, dtype=torch.long)
    return {
        "frame_vertices": frames,
        "faces": faces,
        "vertex_normals": torch.zeros_like(frames),
        "face_normals": torch.zeros((1, 2, 2, 3), device=device),
        "vertex_count": torch.tensor([4], device=device),
        "face_count": torch.tensor([2], device=device),
        "target_skin_weights": torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                ]
            ],
            device=device,
        ),
        "input_ids": torch.tensor([[18, 17, 2, 3, 4, 5, 6, 7, 19]], device=device),
        "attention_mask": torch.ones((1, 9), device=device),
        "token_joint_indices": torch.tensor(
            [[-1, 0, 0, 0, 1, 1, 1, -1, -1]],
            device=device,
        ),
        "token_completed_joint_counts": torch.tensor(
            [[-1, 0, 0, 0, 1, 1, 1, 2, -1]],
            device=device,
        ),
        "token_branch_decision_mask": torch.zeros(
            (1, 9), device=device, dtype=torch.bool
        ),
        "joint_count": torch.tensor([2], device=device),
    }


def test_static_query_memory_does_not_change_with_evidence_frames() -> None:
    torch.manual_seed(21)
    device = torch.device("cpu")
    batch = _batch(device)
    conditioner = StaticQueryMotionEvidenceConditioner(_SurfaceTokenizer(16), 16)
    normal = conditioner(
        batch["frame_vertices"],
        batch["faces"],
        _references(device),
        vertex_normals=batch["vertex_normals"],
        face_normals=batch["face_normals"],
        vertex_counts=batch["vertex_count"],
        face_counts=batch["face_count"],
    )
    altered_frames = batch["frame_vertices"].clone()
    altered_frames[:, 1, 1:] *= 1.8
    altered = conditioner(
        altered_frames,
        batch["faces"],
        _references(device),
        vertex_normals=batch["vertex_normals"],
        face_normals=batch["face_normals"],
        vertex_counts=batch["vertex_count"],
        face_counts=batch["face_count"],
    )
    torch.testing.assert_close(normal.static_tokens, altered.static_tokens)
    assert not torch.allclose(normal.motion_values, altered.motion_values)


def test_zero_evidence_teacher_forcing_is_exact_decoder_noop() -> None:
    torch.manual_seed(23)
    device = torch.device("cpu")
    batch = _batch(device)
    model = TopologyMotionEvidenceUniRigAR(
        _UniRig(16, _Tokenizer.vocab_size),
        _SurfaceTokenizer(16),
        _Tokenizer(),
        num_surface_samples=5,
        vertex_samples=4,
        query_tokens=5,
        evidence_heads=4,
    )
    memory = model.build_memory(batch, refs=_references(device)).controlled("zero")
    teacher = model.teacher_forcing(batch, memory=memory)
    assert torch.equal(teacher.refined_hidden, teacher.token_hidden)
    assert torch.equal(teacher.logits, teacher.baseline_logits)


def test_locally_unobservable_evidence_is_noop_despite_global_motion() -> None:
    torch.manual_seed(24)
    device = torch.device("cpu")
    batch = _batch(device)
    model = TopologyMotionEvidenceUniRigAR(
        _UniRig(16, _Tokenizer.vocab_size),
        _SurfaceTokenizer(16),
        _Tokenizer(),
        num_surface_samples=5,
        vertex_samples=4,
        query_tokens=5,
        evidence_heads=4,
    )
    memory = model.build_memory(batch, refs=_references(device))
    no_local_evidence = replace(
        memory,
        motion_values=torch.zeros_like(memory.motion_values),
        anchor_confidence=torch.zeros_like(memory.anchor_confidence),
        confidence=torch.ones_like(memory.confidence),
    )
    teacher = model.teacher_forcing(batch, memory=no_local_evidence)
    boundary = model.boundary_auxiliary_loss(batch, _references(device), no_local_evidence)

    assert torch.equal(teacher.logits, teacher.baseline_logits)
    assert float(boundary["boundary_loss"].detach()) == 0.0


def test_zero_residual_scale_matches_precision_stable_projection() -> None:
    torch.manual_seed(43)
    device = torch.device("cpu")
    batch = _batch(device)
    model = TopologyMotionEvidenceUniRigAR(
        _UniRig(16, _Tokenizer.vocab_size),
        _SurfaceTokenizer(16),
        _Tokenizer(),
        num_surface_samples=5,
        vertex_samples=4,
        query_tokens=5,
        evidence_heads=4,
        evidence_residual_scale=0.0,
    )
    memory = model.build_memory(batch, refs=_references(device))
    teacher = model.teacher_forcing(batch, memory=memory)
    projected = model.evidence_adapter.project_hidden(model.transformer, teacher.token_hidden)

    torch.testing.assert_close(
        teacher.refined_hidden,
        teacher.token_hidden,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        teacher.logits[:, model.evidence_adapter.static_prefix_steps :],
        projected[:, model.evidence_adapter.static_prefix_steps :],
        atol=0.0,
        rtol=0.0,
    )


def test_corrupt_control_preserves_values_but_breaks_alignment() -> None:
    torch.manual_seed(29)
    device = torch.device("cpu")
    batch = _batch(device)
    model = TopologyMotionEvidenceUniRigAR(
        _UniRig(16, _Tokenizer.vocab_size),
        _SurfaceTokenizer(16),
        _Tokenizer(),
        num_surface_samples=5,
        vertex_samples=4,
        query_tokens=5,
        evidence_heads=4,
    )
    memory = model.build_memory(batch, refs=_references(device))
    generator = torch.Generator(device=device).manual_seed(31)
    corrupted = memory.controlled("corrupt_correspondence", generator=generator)
    torch.testing.assert_close(
        memory.motion_values.sort(dim=1).values,
        corrupted.motion_values.sort(dim=1).values,
    )
    assert not torch.equal(memory.motion_values, corrupted.motion_values)
    assert torch.equal(memory.static_tokens, corrupted.static_tokens)
    assert torch.equal(memory.confidence, corrupted.confidence)
    torch.testing.assert_close(
        memory.anchor_confidence.sort(dim=1).values,
        corrupted.anchor_confidence.sort(dim=1).values,
    )


def test_teacher_forcing_and_generation_step_use_the_same_prefix_position() -> None:
    torch.manual_seed(33)
    device = torch.device("cpu")
    batch = _batch(device)
    model = TopologyMotionEvidenceUniRigAR(
        _UniRig(16, _Tokenizer.vocab_size),
        _SurfaceTokenizer(16),
        _Tokenizer(),
        num_surface_samples=5,
        vertex_samples=4,
        query_tokens=5,
        evidence_heads=4,
    )
    memory = model.build_memory(batch, refs=_references(device))
    teacher = model.teacher_forcing(batch, memory=memory)
    prefix_length = 5
    prefix_ids = batch["input_ids"][:, :prefix_length]
    token_embeds = model.transformer.get_input_embeddings()(prefix_ids)
    prompt = torch.cat((memory.static_tokens, token_embeds), dim=1)
    prompt_mask = torch.ones(prompt.shape[:2])
    transformer_output = model.transformer(
        inputs_embeds=prompt,
        attention_mask=prompt_mask,
        use_cache=False,
        output_hidden_states=True,
    )
    generation_logits = model.evidence_adapter.generation_step(
        model.transformer,
        transformer_output,
        memory,
        prefix_position=prefix_length - 1,
    )
    torch.testing.assert_close(generation_logits, teacher.logits[:, prefix_length - 1])


def test_motion_evidence_leaves_class_and_root_predictions_unchanged() -> None:
    torch.manual_seed(35)
    device = torch.device("cpu")
    batch = _batch(device)
    model = TopologyMotionEvidenceUniRigAR(
        _UniRig(16, _Tokenizer.vocab_size),
        _SurfaceTokenizer(16),
        _Tokenizer(),
        num_surface_samples=5,
        vertex_samples=4,
        query_tokens=5,
        evidence_heads=4,
    )
    teacher = model.teacher_forcing(batch, refs=_references(device))
    torch.testing.assert_close(
        teacher.refined_hidden[:, :4],
        teacher.token_hidden[:, :4],
        atol=0.0,
        rtol=0.0,
    )
    assert torch.equal(teacher.logits[:, :4], teacher.baseline_logits[:, :4])
    assert not torch.equal(teacher.refined_hidden[:, 4:], teacher.token_hidden[:, 4:])


def test_attention_weights_are_normalized_per_step_and_head() -> None:
    torch.manual_seed(36)
    device = torch.device("cpu")
    batch = _batch(device)
    model = TopologyMotionEvidenceUniRigAR(
        _UniRig(16, _Tokenizer.vocab_size),
        _SurfaceTokenizer(16),
        _Tokenizer(),
        num_surface_samples=5,
        vertex_samples=4,
        query_tokens=5,
        evidence_heads=4,
    )
    memory = model.build_memory(batch, refs=_references(device))
    teacher = model.teacher_forcing(batch, memory=memory)
    weights = model.evidence_adapter.attention.attention_weights(
        teacher.token_hidden,
        memory.static_tokens,
        memory.motion_values,
    )
    assert weights.shape == (1, 4, 9, 5)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones((1, 4, 9)))


def test_spatially_constant_motion_cannot_create_a_global_decoder_bias() -> None:
    torch.manual_seed(37)
    attention = MotionEvidenceDecoderAdapter(16, heads=4).attention
    prefix = torch.randn(2, 7, 16)
    static_keys = torch.randn(2, 5, 16)
    constant_motion = torch.ones(2, 5, 16)
    refined = attention(
        prefix,
        static_keys,
        constant_motion,
    )
    assert torch.equal(refined, prefix)


def test_attention_alignment_loss_trains_prefix_to_region_addressing() -> None:
    torch.manual_seed(38)
    device = torch.device("cpu")
    batch = _batch(device)
    model = TopologyMotionEvidenceUniRigAR(
        _UniRig(16, _Tokenizer.vocab_size),
        _SurfaceTokenizer(16),
        _Tokenizer(),
        num_surface_samples=5,
        vertex_samples=4,
        query_tokens=5,
        evidence_heads=4,
    )
    refs = _references(device)
    memory = model.build_memory(batch, refs=refs)
    teacher = model.teacher_forcing(batch, memory=memory)
    alignment = model.attention_alignment_loss(
        batch,
        refs,
        memory,
        teacher.token_hidden,
    )
    assert torch.isfinite(alignment["attention_alignment_loss"])
    assert float(alignment["attention_alignment_loss"].detach()) > 0.0
    alignment["attention_alignment_loss"].backward()
    addressing_grad = model.evidence_adapter.attention.cross_attention.in_proj_weight.grad
    assert addressing_grad is not None
    assert float(addressing_grad.abs().sum()) > 0.0


def test_training_route_has_nonzero_evidence_and_backbone_gradients() -> None:
    torch.manual_seed(39)
    device = torch.device("cpu")
    batch = _batch(device)
    model = TopologyMotionEvidenceUniRigAR(
        _UniRig(16, _Tokenizer.vocab_size),
        _SurfaceTokenizer(16),
        _Tokenizer(),
        num_surface_samples=8,
        vertex_samples=4,
        query_tokens=5,
        evidence_heads=4,
    )
    output = model(batch)
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    evidence_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.evidence_adapter.parameters()
        if parameter.grad is not None
    )
    value_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.conditioner.value_encoder.parameters()
        if parameter.grad is not None
    )
    boundary_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.conditioner.value_encoder.boundary_head.parameters()
        if parameter.grad is not None
    )
    backbone_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.transformer.parameters()
        if parameter.grad is not None
    )
    assert evidence_grad > 0.0
    assert value_grad > 0.0
    assert boundary_grad > 0.0
    assert backbone_grad > 0.0
    assert torch.isfinite(output["boundary_loss"])
    assert torch.isfinite(output["attention_alignment_loss"])
    assert float(output["attention_alignment_valid_fraction"]) > 0.0


def test_coverage_aware_route_trains_support_null_and_evidence_paths() -> None:
    torch.manual_seed(40)
    device = torch.device("cpu")
    batch = _batch(device)
    model = CoverageAwareTopologyMotionEvidenceUniRigAR(
        _UniRig(16, _Tokenizer.vocab_size),
        _SurfaceTokenizer(16),
        _Tokenizer(),
        num_surface_samples=8,
        vertex_samples=4,
        query_tokens=5,
        evidence_heads=4,
        support_projection_size=8,
    )
    output = model(batch)
    output["loss"].backward()
    support_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.evidence_adapter.attention.support_head.parameters()
        if parameter.grad is not None
    )
    evidence_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.evidence_adapter.attention.value_projection.parameters()
        if parameter.grad is not None
    )
    assert torch.isfinite(output["loss"])
    assert torch.isfinite(output["prefix_support_loss"])
    assert float(output["prefix_support_valid_fraction"]) > 0.0
    assert support_grad > 0.0
    assert evidence_grad > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_adapter_logits_are_prefix_length_stable_under_bfloat16_autocast() -> None:
    torch.manual_seed(41)
    device = torch.device("cuda")
    hidden_size = 256
    transformer = _CausalTransformer(hidden_size, 8192).to(device)
    adapter = MotionEvidenceDecoderAdapter(hidden_size, heads=8).to(device)
    memory = MotionEvidenceMemory(
        static_tokens=torch.randn(1, 64, hidden_size, device=device, dtype=torch.bfloat16),
        motion_values=torch.randn(1, 64, hidden_size, device=device, dtype=torch.bfloat16),
        boundary_logits=torch.randn(1, 64, 2, device=device),
        anchor_confidence=torch.ones(1, 64, device=device),
        confidence=torch.tensor([0.8], device=device),
        raw_evidence=None,  # type: ignore[arg-type]
    )
    prefix = torch.randn(1, 33, hidden_size, device=device, dtype=torch.bfloat16)
    full_positions = torch.arange(prefix.shape[1], device=device)
    step_positions = full_positions[-1:]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        full_logits, full_hidden = adapter.logits_from_hidden(
            transformer,
            prefix,
            memory,
            full_positions,
        )
        step_logits, step_hidden = adapter.logits_from_hidden(
            transformer,
            prefix[:, -1:],
            memory,
            step_positions,
        )
    torch.testing.assert_close(full_hidden[:, -1], step_hidden[:, 0], atol=1.0e-5, rtol=1.0e-5)
    torch.testing.assert_close(full_logits[:, -1], step_logits[:, 0], atol=1.0e-4, rtol=1.0e-5)
