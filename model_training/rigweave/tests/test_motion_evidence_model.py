from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rigweave.dynamic_rig.sampling import TrackableSurfaceReferences
from rigweave.motion_evidence import (
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
        "input_ids": torch.tensor([[18, 17, 2, 3, 4, 19]], device=device),
        "attention_mask": torch.ones((1, 6), device=device),
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
    assert not torch.equal(teacher.refined_hidden[:, 4:], teacher.token_hidden[:, 4:])


def test_training_route_has_nonzero_evidence_and_backbone_gradients() -> None:
    torch.manual_seed(37)
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
    backbone_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.transformer.parameters()
        if parameter.grad is not None
    )
    assert evidence_grad > 0.0
    assert value_grad > 0.0
    assert backbone_grad > 0.0


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
