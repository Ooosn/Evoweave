"""Capture the actual motion-encoder boundary for fixed-input interventions."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch


def rng_snapshot():
    return {"cpu": torch.get_rng_state().clone(),
            "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


def restore_rng(state):
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def rng_digest(state):
    return {"cpu": hashlib.sha256(state["cpu"].numpy().tobytes()).hexdigest(),
            "cuda": [hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest() for value in state["cuda"]]}


def tensor_delta(reference, other):
    if reference.shape != other.shape:
        raise ValueError("replayed tensor shape changed")
    a, b = reference.detach().float(), other.detach().float()
    delta = b - a
    return {"equal": bool(torch.equal(reference, other)), "max_abs": float(delta.abs().max()),
            "relative_l2": float(delta.norm() / a.norm().clamp_min(1e-20))}


@dataclass
class MotionInput:
    tokens: torch.Tensor
    query_points: torch.Tensor
    evidence: object
    rng: dict

    @classmethod
    def capture(cls, args, kwargs):
        if len(args) != 1 or set(kwargs) != {"query_points", "motion_evidence"}:
            raise ValueError("unexpected motion-encoder invocation; no diagnostic fallback")
        evidence = kwargs["motion_evidence"]
        frozen_evidence = type(evidence)(*(getattr(evidence, key).detach().clone()
            for key in ("states", "anchor_features", "valid_groups")))
        return cls(args[0].detach().clone(), kwargs["query_points"].detach().clone(), frozen_evidence, rng_snapshot())

    def forward(self, encoder):
        restore_rng(self.rng)
        return encoder(self.tokens, query_points=self.query_points, motion_evidence=self.evidence)

    def delta(self, other):
        return {"frame_tokens": tensor_delta(self.tokens, other.tokens),
                "query_points": tensor_delta(self.query_points, other.query_points),
                **{key: tensor_delta(getattr(self.evidence, key), getattr(other.evidence, key))
                   for key in ("states", "anchor_features", "valid_groups")}}


class MotionCapture:
    def __init__(self, model):
        if model.condition_fusion != "dynamic" or model.branch_prior is not None:
            raise ValueError("cached diagnosis supports only the exact dynamic/no-prior contract")
        self.model = model
        self.motion_input = None
        self.rng_after_motion = None
        self.calls = 0

    def __enter__(self):
        encoder = self.model.conditioner.motion_encoder
        self.pre = encoder.register_forward_pre_hook(self._before, with_kwargs=True)
        self.post = encoder.register_forward_hook(self._after)
        return self

    def _before(self, module, args, kwargs):
        self.calls += 1
        if self.calls != 1:
            raise RuntimeError("one full condition must invoke motion encoder exactly once")
        self.motion_input = MotionInput.capture(args, kwargs)

    def _after(self, module, args, output):
        self.rng_after_motion = rng_snapshot()

    def __exit__(self, exc_type, exc, traceback):
        self.pre.remove()
        self.post.remove()


def captured_forward(model, batch, refs):
    before = rng_snapshot()
    with MotionCapture(model) as captured:
        condition = model.build_condition(batch, refs=refs)
        loss = model._ar_losses(condition, batch)
    if captured.motion_input is None:
        raise RuntimeError("motion boundary was not captured")
    trace = {"start": rng_digest(before), "motion_entry": rng_digest(captured.motion_input.rng),
             "motion_exit": rng_digest(captured.rng_after_motion), "after_ce": rng_digest(rng_snapshot())}
    return condition, loss, captured.motion_input, before, trace
