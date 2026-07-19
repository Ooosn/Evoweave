#!/usr/bin/env python3
from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_stack_close_generation import _build_model  # noqa: E402
from train_dynamic_rig import move_batch, move_dynamic_model_to_device  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit zero-residual condition refresh before training.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--unirig-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--refresh-layers", default="7,15,23")
    parser.add_argument("--refresh-dim", type=int, default=256)
    parser.add_argument("--refresh-heads", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _decoder_inputs(
    model: torch.nn.Module,
    cond: torch.Tensor,
    batch: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    token_embeds = model.transformer.get_input_embeddings()(
        batch["target_ids"]
    ).to(dtype=model.transformer.dtype)
    inputs_embeds = torch.cat(
        [cond.to(dtype=model.transformer.dtype), token_embeds],
        dim=1,
    )
    attention_mask = pad(
        batch["attention_mask"],
        (cond.shape[1], 0, 0, 0),
        value=1.0,
    )
    return inputs_embeds, attention_mask


def _finite_nonzero(parameters: list[torch.nn.Parameter]) -> dict[str, Any]:
    gradients = [
        parameter.grad.detach().float()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        return {
            "parameter_tensors": len(parameters),
            "gradient_tensors": 0,
            "finite": False,
            "abs_max": 0.0,
            "abs_sum": 0.0,
        }
    return {
        "parameter_tensors": len(parameters),
        "gradient_tensors": len(gradients),
        "finite": bool(
            all(torch.isfinite(gradient).all().item() for gradient in gradients)
        ),
        "abs_max": float(
            max(gradient.abs().max().item() for gradient in gradients)
        ),
        "abs_sum": float(
            sum(gradient.abs().sum().item() for gradient in gradients)
        ),
    }


def main() -> None:
    args = _parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2**32))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    amp_dtype = torch.bfloat16

    train_args = {
        "tokenizer_config": str(args.tokenizer_config),
        "model_config": str(args.model_config),
        "unirig_checkpoint": str(args.unirig_checkpoint),
        "frames": 24,
        "surface_samples": 65_536,
        "vertex_samples": 8_192,
        "query_tokens": 1_024,
        "register_tokens": 96,
        "motion_depth": 12,
        "motion_heads": 8,
        "motion_fps_ratio": 0.7,
        "motion_vertex_samples": 512,
        "perturb_row_probability": 0.0,
        "condition_refresh_layers": args.refresh_layers,
        "condition_refresh_dim": args.refresh_dim,
        "condition_refresh_heads": args.refresh_heads,
    }
    legacy_tokenizer, tokenizer, model = _build_model(
        train_args,
        device=device,
    )
    if not hasattr(model, "condition_refresh_adapters"):
        raise RuntimeError("refresh model was not constructed")
    move_dynamic_model_to_device(model, device)
    model.condition_refresh_adapters.to(device)

    from rigweave.stack_close import (
        StackCloseManifestDataset,
        stack_close_collate,
    )

    dataset = StackCloseManifestDataset(
        args.manifest,
        legacy_tokenizer=legacy_tokenizer,
        stack_tokenizer=tokenizer,
        frame_count=24,
        limit=1,
        random_query=False,
        random_sibling_order=False,
        seed=args.seed,
        motion_fps_ratio=0.7,
        motion_vertex_samples=512,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=partial(stack_close_collate, pad_token=tokenizer.pad),
    )
    batch = move_batch(next(iter(loader)), device)

    model.eval()
    initial_gate_max_abs = float(
        max(
            adapter.gate.detach().abs().max().cpu().item()
            for adapter in model.condition_refresh_adapters
        )
    )
    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=amp_dtype,
        enabled=device.type == "cuda",
    ):
        refs = model.sample_references(batch)
        cond = model.build_condition(batch, refs=refs)
        inputs_embeds, attention_mask = _decoder_inputs(
            model,
            cond,
            batch,
        )
        direct = model.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
        with model.condition_refresh(cond):
            refreshed = model.transformer(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
        max_logit_diff = float((direct - refreshed).abs().max().cpu())

        benchmark_repeats = 3
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(benchmark_repeats):
            model.transformer(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        direct_seconds = (
            time.perf_counter() - start
        ) / benchmark_repeats
        start = time.perf_counter()
        for _ in range(benchmark_repeats):
            with model.condition_refresh(cond):
                model.transformer(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        refresh_seconds = (
            time.perf_counter() - start
        ) / benchmark_repeats

    gate_parameters = [
        adapter.gate
        for adapter in model.condition_refresh_adapters
    ]
    projection_parameters = [
        parameter
        for adapter in model.condition_refresh_adapters
        for name, parameter in adapter.named_parameters()
        if name != "gate"
    ]
    refresh_optimizer = torch.optim.AdamW(
        model.condition_refresh_adapters.parameters(),
        lr=1.0e-3,
        weight_decay=0.0,
    )

    model.train()
    refresh_optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type="cuda",
        dtype=amp_dtype,
        enabled=device.type == "cuda",
    ):
        first_loss = model(batch)["loss"]
    first_loss.backward()
    first_gate_grad = _finite_nonzero(gate_parameters)
    first_projection_grad = _finite_nonzero(projection_parameters)
    refresh_optimizer.step()
    gate_after_step = float(model.refresh_gate_max_abs().detach().cpu())

    model.zero_grad(set_to_none=True)
    refresh_optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type="cuda",
        dtype=amp_dtype,
        enabled=device.type == "cuda",
    ):
        second_loss = model(batch)["loss"]
    second_loss.backward()
    second_gate_grad = _finite_nonzero(gate_parameters)
    second_projection_grad = _finite_nonzero(projection_parameters)

    report = {
        "contract": "stack_close_condition_refresh_v1",
        "path": batch["path"][0],
        "query_frame": int(batch["selected_frames"][0, 0].item()),
        "condition_shape": list(cond.shape),
        "target_tokens": int(batch["attention_mask"][0].sum().item()),
        "refresh_layers": list(model.refresh_layer_indices),
        "refresh_dim": int(model.refresh_dim),
        "refresh_heads": int(model.refresh_heads),
        "refresh_parameters": int(
            sum(
                parameter.numel()
                for parameter in model.condition_refresh_adapters.parameters()
            )
        ),
        "initial_gate_max_abs": initial_gate_max_abs,
        "initial_max_abs_logit_diff": max_logit_diff,
        "direct_forward_seconds": direct_seconds,
        "refresh_forward_seconds": refresh_seconds,
        "forward_time_ratio": refresh_seconds / max(direct_seconds, 1.0e-9),
        "first_loss": float(first_loss.detach().cpu()),
        "first_gate_grad": first_gate_grad,
        "first_projection_grad": first_projection_grad,
        "gate_max_abs_after_one_step": gate_after_step,
        "second_loss": float(second_loss.detach().cpu()),
        "second_gate_grad": second_gate_grad,
        "second_projection_grad": second_projection_grad,
    }
    if max_logit_diff != 0.0:
        raise RuntimeError(
            f"zero-gated refresh changed initial logits by {max_logit_diff}"
        )
    if initial_gate_max_abs != 0.0:
        raise RuntimeError(
            f"refresh gate was not zero at initialization: {initial_gate_max_abs}"
        )
    if not first_gate_grad["finite"] or first_gate_grad["abs_sum"] <= 0.0:
        raise RuntimeError("refresh gate did not receive a finite nonzero gradient")
    if gate_after_step <= 0.0:
        raise RuntimeError("refresh gate did not change after one optimizer step")
    if (
        not second_projection_grad["finite"]
        or second_projection_grad["abs_sum"] <= 0.0
    ):
        raise RuntimeError(
            "refresh projections did not receive gradients after the gate opened"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
