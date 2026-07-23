from __future__ import annotations

import pytest
import torch

from rigweave.dynamic_rig.sampling import TrackableSurfaceReferences
from rigweave.motion_evidence import (
    MotionEvidenceCrossAttention,
    TopologyLocalMotionEvidence,
    TopologyMotionValueEncoder,
)


def _references(device: torch.device) -> TrackableSurfaceReferences:
    return TrackableSurfaceReferences(
        vertex_indices=torch.tensor([[0, 1, 2, 3]], device=device),
        face_indices=torch.tensor([[0]], device=device),
        barycentric=torch.tensor([[[0.25, 0.25, 0.50]]], device=device),
        query_indices=torch.tensor([[0, 1, 2, 3, 4]], device=device),
    )


def _square(device: torch.device) -> tuple[torch.Tensor, torch.LongTensor]:
    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        device=device,
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], device=device, dtype=torch.long)
    return vertices, faces


def test_zero_motion_is_exactly_zero_after_value_encoding() -> None:
    device = torch.device("cpu")
    query, faces = _square(device)
    frames = query[None, None].repeat(1, 5, 1, 1)
    extractor = TopologyLocalMotionEvidence()
    evidence = extractor(frames, faces, _references(device))
    assert torch.count_nonzero(evidence.query_features) == 0
    assert torch.equal(evidence.confidence, torch.zeros_like(evidence.confidence))

    encoder = TopologyMotionValueEncoder(hidden_size=16)
    values = encoder(evidence.query_features, evidence.confidence)
    assert torch.equal(values, torch.zeros_like(values))


def test_global_rigid_transform_does_not_create_motion_evidence() -> None:
    device = torch.device("cpu")
    query, faces = _square(device)
    angle = torch.tensor(0.7)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    moved = query @ rotation.T + torch.tensor([3.0, -2.0, 0.5])
    frames = torch.stack((query, moved), dim=0)[None]
    evidence = TopologyLocalMotionEvidence()(frames, faces, _references(device))
    torch.testing.assert_close(evidence.query_features, torch.zeros_like(evidence.query_features), atol=2e-6, rtol=0)


def test_vertex_and_face_references_preserve_evidence_alignment() -> None:
    device = torch.device("cpu")
    query, faces = _square(device)
    moved = query.clone()
    moved[2] = torch.tensor([1.6, 1.4, 0.0])
    frames = torch.stack((query, moved), dim=0)[None]
    evidence = TopologyLocalMotionEvidence()(frames, faces, _references(device))
    assert evidence.query_features.shape == (1, 5, 8)
    expected_face = (
        0.25 * evidence.query_features[:, 0]
        + 0.25 * evidence.query_features[:, 1]
        + 0.50 * evidence.query_features[:, 2]
    )
    torch.testing.assert_close(evidence.query_features[:, 4], expected_face)
    assert float(evidence.confidence[0]) > 0.0


def test_attention_uses_motion_values_and_preserves_zero_confidence() -> None:
    torch.manual_seed(7)
    attention = MotionEvidenceCrossAttention(hidden_size=16, heads=4)
    prefix = torch.randn(2, 3, 16, requires_grad=True)
    static = torch.randn(2, 5, 16, requires_grad=True)
    motion = torch.randn(2, 5, 16, requires_grad=True)
    confidence = torch.tensor([0.0, 0.8])
    output = attention(prefix, static, motion, confidence)
    torch.testing.assert_close(output[0], prefix[0])
    assert not torch.allclose(output[1], prefix[1])

    output[1].square().mean().backward()
    attention_grad = attention.cross_attention.in_proj_weight.grad
    assert attention_grad is not None and torch.isfinite(attention_grad).all()
    assert motion.grad is not None and float(motion.grad.abs().sum()) > 0.0
    assert static.grad is None


def test_extractor_encoder_attention_path_has_finite_nonzero_gradients() -> None:
    torch.manual_seed(11)
    device = torch.device("cpu")
    query, faces = _square(device)
    moved = query.clone()
    moved[2:] += torch.tensor([0.4, 0.7, 0.0])
    frames = torch.stack((query, moved), dim=0)[None]
    evidence = TopologyLocalMotionEvidence()(frames, faces, _references(device))
    encoder = TopologyMotionValueEncoder(hidden_size=16)
    attention = MotionEvidenceCrossAttention(hidden_size=16, heads=4)
    values = encoder(evidence.query_features, evidence.confidence)
    prefix = torch.randn(1, 4, 16)
    static = torch.randn(1, 5, 16)
    output = attention(prefix, static, values, evidence.confidence)
    output.square().mean().backward()

    encoder_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in encoder.parameters()
        if parameter.grad is not None
    )
    attention_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in attention.parameters()
        if parameter.grad is not None
    )
    assert encoder_grad > 0.0
    assert attention_grad > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_attention_supports_bfloat16_module_on_cuda() -> None:
    torch.manual_seed(13)
    device = torch.device("cuda")
    attention = MotionEvidenceCrossAttention(hidden_size=32, heads=4).to(
        device=device,
        dtype=torch.bfloat16,
    )
    prefix = torch.randn(2, 3, 32, device=device, dtype=torch.bfloat16)
    static = torch.randn(2, 5, 32, device=device, dtype=torch.bfloat16)
    motion = torch.randn(2, 5, 32, device=device, dtype=torch.bfloat16)
    confidence = torch.tensor([0.0, 0.8], device=device)
    output = attention(prefix, static, motion, confidence)
    assert output.dtype == torch.bfloat16
    assert torch.equal(output[0], prefix[0])
    assert torch.isfinite(output).all()

    encoder = TopologyMotionValueEncoder(hidden_size=32).to(
        device=device,
        dtype=torch.bfloat16,
    )
    features = torch.rand(2, 5, 8, device=device, dtype=torch.float32)
    values = encoder(features, confidence)
    assert values.dtype == torch.bfloat16
    assert torch.equal(values[0], torch.zeros_like(values[0]))
    assert torch.isfinite(values).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_float32_attention_is_prefix_length_stable_under_bfloat16_autocast() -> None:
    torch.manual_seed(17)
    device = torch.device("cuda")
    attention = MotionEvidenceCrossAttention(hidden_size=64, heads=4).to(device)
    prefix = torch.randn(1, 17, 64, device=device, dtype=torch.bfloat16)
    static = torch.randn(1, 32, 64, device=device, dtype=torch.bfloat16)
    motion = torch.randn(1, 32, 64, device=device, dtype=torch.bfloat16)
    confidence = torch.tensor([0.8], device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        full = attention(prefix, static, motion, confidence)
        step = attention(prefix[:, -1:], static, motion, confidence)
    torch.testing.assert_close(full[:, -1], step[:, 0], atol=1.0e-5, rtol=1.0e-5)
