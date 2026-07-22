#!/usr/bin/env python3
"""Compare official UniRig conditioning with trackable frame-0 tokens.

This is a pre-training compatibility test. It keeps the official UniRig
tokenizer and autoregressive decoder fixed, changes only the mesh condition
construction, and evaluates both teacher forcing and free generation on the
same deterministic query poses.
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader


MODEL_TRAINING_ROOT = Path(__file__).resolve().parents[1]
RIGWEAVE_SCRIPTS = MODEL_TRAINING_ROOT / "rigweave" / "scripts"
sys.path.insert(0, str(RIGWEAVE_SCRIPTS))

from eval_dynamic_rig_generation import (
    _dynamic_generate,
    _run_model,
    _static_generate,
    _summarize,
)
from train_dynamic_rig import (
    build_tokenizer,
    load_unirig,
    move_batch,
    move_dynamic_model_to_device,
)


class _TrackableFrame0ConditionOverride:
    """Delegate the decoder API while replacing only condition construction."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)

    def build_condition(
        self,
        batch: dict[str, Any],
        *,
        return_branch_prior: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, None]:
        refs = self.model.sample_references(batch)
        frame_tokens, _query_points = self.model.conditioner.tokenize_frames(
            batch["frame_vertices"],
            batch["faces"],
            refs,
            vertex_normals=batch["vertex_normals"],
            face_normals=batch["face_normals"],
        )
        condition = frame_tokens[:, 0]
        if return_branch_prior:
            return condition, None
        return condition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--unirig-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=18)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--surface-samples", type=int, default=65536)
    parser.add_argument("--vertex-samples", type=int, default=8192)
    parser.add_argument("--query-tokens", type=int, default=1024)
    parser.add_argument("--register-tokens", type=int, default=96)
    parser.add_argument("--motion-depth", type=int, default=12)
    parser.add_argument("--motion-heads", type=int, default=8)
    parser.add_argument("--motion-fps-ratio", type=float, default=0.7)
    parser.add_argument("--motion-vertex-samples", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=1400)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    return parser.parse_args()


def _teacher_logits(model: Any, condition: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
    condition = condition.to(dtype=model.transformer.dtype)
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    token_embeddings = model.token_inputs_embeds(input_ids, attention_mask)
    inputs_embeds = torch.cat([condition, token_embeddings], dim=1)
    full_attention = pad(attention_mask, (condition.shape[1], 0, 0, 0), value=1.0)
    output = model.transformer(
        inputs_embeds=inputs_embeds,
        attention_mask=full_attention,
        use_cache=False,
        output_hidden_states=False,
    )
    logits = output.logits[:, condition.shape[1] :].reshape(
        input_ids.shape[0], -1, model.tokenizer.vocab_size
    )
    return logits[:, :-1]


def _teacher_metrics(logits: torch.Tensor, batch: dict[str, Any]) -> dict[str, float]:
    labels = batch["input_ids"][:, 1:].clone()
    labels[batch["attention_mask"][:, 1:] == 0] = -100
    loss = torch.nn.functional.cross_entropy(logits.permute(0, 2, 1), labels)
    valid = labels != -100
    accuracy = (logits.argmax(dim=-1)[valid] == labels[valid]).float().mean()
    prefix_count = min(8, int(valid[0].sum().item()))
    prefix_labels = labels[:, :prefix_count]
    prefix_valid = prefix_labels != -100
    prefix_accuracy = (
        (logits[:, :prefix_count].argmax(dim=-1)[prefix_valid] == prefix_labels[prefix_valid])
        .float()
        .mean()
    )
    return {
        "ce": float(loss.detach().float().cpu()),
        "token_accuracy": float(accuracy.detach().float().cpu()),
        "first_8_token_accuracy": float(prefix_accuracy.detach().float().cpu()),
    }


@torch.no_grad()
def _condition_teacher_audit(
    model: Any,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, batch in enumerate(loader):
        batch = move_batch(batch, device)
        torch.manual_seed(seed + index)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + index)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            refs = model.sample_references(batch)
            frame_tokens, _query_points = model.conditioner.tokenize_frames(
                batch["frame_vertices"],
                batch["faces"],
                refs,
                vertex_normals=batch["vertex_normals"],
                face_normals=batch["face_normals"],
            )
            trackable = frame_tokens[:, 0]
            official = model.build_static_condition(batch).to(
                device=trackable.device,
                dtype=trackable.dtype,
            )
            trackable_logits = _teacher_logits(model, trackable, batch)
            official_logits = _teacher_logits(model, official, batch)

        rows.append(
            {
                "index": index,
                "path": batch["path"][0],
                "query_frame": int(batch["selected_frames"][0, 0].detach().cpu()),
                "vertex_count": int(batch["vertex_count"][0].detach().cpu()),
                "trackable_frame0": _teacher_metrics(trackable_logits, batch),
                "official_static": _teacher_metrics(official_logits, batch),
                "condition_stats": {
                    "trackable_rms": float(trackable.float().square().mean().sqrt().cpu()),
                    "official_rms": float(official.float().square().mean().sqrt().cpu()),
                    "trackable_token_std_mean": float(trackable.float().std(dim=1).mean().cpu()),
                    "official_token_std_mean": float(official.float().std(dim=1).mean().cpu()),
                    "mean_token_l2": float(
                        (trackable.float().mean(dim=1) - official.float().mean(dim=1)).norm(dim=-1).mean().cpu()
                    ),
                },
            }
        )
    return rows


def _mean(rows: list[dict[str, Any]], route: str, metric: str) -> float:
    values = [float(row[route][metric]) for row in rows]
    return float(sum(values) / max(len(values), 1))


def main() -> None:
    args = parse_args()
    for path in (args.manifest, args.tokenizer_config, args.model_config, args.unirig_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    from rigweave.dynamic_rig import AnchorWiseAlternatingMotionEncoder, FixedQuerySurfaceTokenizer
    from rigweave.dynamic_rig.data import DynamicRigManifestDataset, dynamic_rig_collate
    from rigweave.dynamic_rig.model import DynamicRigConditioner
    from rigweave.dynamic_rig.unirig_wrapper import DynamicRigUniRigAR

    tokenizer = build_tokenizer(args.tokenizer_config)
    unirig = load_unirig(tokenizer, args.model_config, args.unirig_checkpoint)
    surface_tokenizer = FixedQuerySurfaceTokenizer(unirig.mesh_encoder, unirig.output_proj)
    motion_encoder = AnchorWiseAlternatingMotionEncoder(
        dim=unirig.hidden_size,
        depth=args.motion_depth,
        heads=args.motion_heads,
        register_tokens=args.register_tokens,
        max_frames=max(args.frames, 48),
        use_motion_features=False,
        use_time_embedding=False,
        gradient_checkpointing=False,
    )
    conditioner = DynamicRigConditioner(surface_tokenizer, motion_encoder)
    model = DynamicRigUniRigAR(
        unirig,
        conditioner,
        tokenizer,
        num_surface_samples=args.surface_samples,
        vertex_samples=args.vertex_samples,
        query_tokens=args.query_tokens,
        condition_fusion="anchor_motion_residual_zero",
        condition_fusion_gate_init=0.25,
        condition_fusion_depth=1,
        branch_prior_proposals=0,
    )
    model.condition_fuser.reset_parameters(gate_init=0.25, zero_init_update=True)
    move_dynamic_model_to_device(model, device)
    model.eval()

    dataset = DynamicRigManifestDataset(
        args.manifest,
        tokenizer,
        frame_count=args.frames,
        limit=args.limit,
        random_query=False,
        seed=args.seed,
        motion_fps_ratio=args.motion_fps_ratio,
        motion_vertex_samples=args.motion_vertex_samples,
        target_start_policy="joint0",
        target_root_policy="legacy",
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(dynamic_rig_collate, pad_token=tokenizer.pad),
    )

    teacher_rows = _condition_teacher_audit(model, loader, device, amp_dtype, args.seed)
    generation_rows = [{"index": index, "path": str(path)} for index, path in enumerate(dataset.paths)]
    generation_kwargs = {"do_sample": False, "num_beams": 1, "num_return_sequences": 1}
    trackable_model = _TrackableFrame0ConditionOverride(model)
    _run_model(
        "trackable_frame0",
        lambda batch, target_ids: _dynamic_generate(
            trackable_model,
            tokenizer,
            batch,
            args.max_new_tokens,
            target_ids,
            generation_kwargs,
        ),
        generation_rows,
        loader,
        tokenizer,
        device,
        amp_dtype,
        args.max_new_tokens,
        args.seed,
    )
    _run_model(
        "official_static",
        lambda batch, _target_ids: _static_generate(
            unirig,
            tokenizer,
            batch,
            args.max_new_tokens,
            generation_kwargs,
        ),
        generation_rows,
        loader,
        tokenizer,
        device,
        amp_dtype,
        args.max_new_tokens,
        args.seed,
    )

    summary = {
        "manifest": str(args.manifest),
        "rows": len(dataset),
        "seed": args.seed,
        "weights": "official_unirig_only_no_dynamic_training",
        "teacher": {
            "trackable_frame0_ce": _mean(teacher_rows, "trackable_frame0", "ce"),
            "official_static_ce": _mean(teacher_rows, "official_static", "ce"),
            "trackable_frame0_first_8_accuracy": _mean(
                teacher_rows, "trackable_frame0", "first_8_token_accuracy"
            ),
            "official_static_first_8_accuracy": _mean(
                teacher_rows, "official_static", "first_8_token_accuracy"
            ),
        },
        "trackable_frame0": _summarize(generation_rows, "trackable_frame0"),
        "official_static": _summarize(generation_rows, "official_static"),
    }
    payload = {
        "summary": summary,
        "teacher_rows": teacher_rows,
        "generation_rows": generation_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
