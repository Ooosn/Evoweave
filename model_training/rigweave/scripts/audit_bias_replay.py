"""Bounded localization of final-checkpoint forward replay noise; no updates."""
from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch

from bias_replay import captured_forward, restore_rng, rng_digest, rng_snapshot, tensor_delta
from motion_experiment_runtime import guard, load_config, source_metadata, verify_manifests, write_json


def main():
    from eval_dynamic_rig_ce import CHECKPOINT_DEFAULTS, _build_dynamic_model
    from train_dynamic_rig import build_tokenizer, move_batch
    from rigweave.dynamic_rig.data import DynamicRigManifestDataset, dynamic_rig_collate

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    plan = config["bias_replay_audit"]
    guard(config, "matched_eval")
    source, manifests = source_metadata(config), verify_manifests(config)
    torch.set_num_threads(1)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("replay audit requires exactly one recorded GPU")
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.93)
    torch.backends.mha.set_fastpath_enabled(False)
    payload = torch.load(plan["checkpoint"], map_location="cpu", mmap=True, weights_only=False)
    model_args = SimpleNamespace(**{**CHECKPOINT_DEFAULTS, **payload["args"], "checkpoint": Path(plan["checkpoint"])})
    step = int(payload["step"])
    del payload
    tokenizer = build_tokenizer(model_args.tokenizer_config)
    model = _build_dynamic_model(model_args, tokenizer, torch.device("cuda:0"))
    model.requires_grad_(False)
    named = [(name, value) for name, value in model.named_parameters() if name.endswith(".motion_injection.bias_weights")]
    if len(named) != 12 or any(value.shape != (8, 3) for _, value in named):
        raise ValueError("expected the recorded twelve-layer eight-head model")
    parameters = [value.requires_grad_(True) for _, value in named]
    versions = {name: value._version for name, value in model.named_parameters()}
    dataset = DynamicRigManifestDataset(model_args.val_manifest, tokenizer, frame_count=plan["frames"],
        limit=plan["validation_rows"], random_query=False, seed=model_args.seed + 17, motion_fps_ratio=0.75,
        minimum_random_frames=1, motion_vertex_samples=model_args.motion_vertex_samples)
    collate = partial(dynamic_rig_collate, pad_token=tokenizer.pad)
    report = {"source": source, "manifests": manifests, "plan": plan, "checkpoint_step": step,
              "training_modules": [name for name, module in model.named_modules() if module.training],
              "rows": [], "complete": False, "scope": "Read-only parameter audit. Same actual batch and all surface references; full recomputation versus detached, unmodified motion-boundary tensors with restored Torch CPU/CUDA RNG. No model/data/target changes, optimizer, or checkpoint writes."}
    write_json(args.output, report)
    for index in range(len(dataset)):
        batch = move_batch(collate([dataset[index]]), torch.device("cuda:0"))
        torch.manual_seed(model_args.seed + 17 + index)
        refs = model.sample_references(batch)
        row = {"index": index, "path": batch["path"][0], "frames": plan["frames"], "runs": {}}
        with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            condition, losses, frozen, start_rng, trace = captured_forward(model, batch, refs)
        first_ce = float(losses["ce_loss"].detach())
        first_condition = condition.detach().clone()
        gradient = torch.cat([value.detach().float().reshape(-1) for value in torch.autograd.grad(losses["ce_loss"], parameters)])
        row["runs"]["full_first"] = {"ce": first_ce, "rng": trace}
        del condition, losses
        for name, reset in (("full_repeat", False), ("full_rng_restored", True)):
            if reset:
                restore_rng(start_rng)
            with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                condition, losses, other, _, trace = captured_forward(model, batch, refs)
            row["runs"][name] = {"ce": float(losses["ce_loss"].detach()), "rng": trace,
                "inputs": frozen.delta(other), "condition": tensor_delta(first_condition, condition)}
            del condition, losses, other
        cached_ce, cached_condition = None, None
        for name in ("cached_first", "cached_repeat"):
            with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                condition = frozen.forward(model.conditioner.motion_encoder)
                after_motion = rng_digest(rng_snapshot())
                losses = model._ar_losses(condition, batch)
            ce = float(losses["ce_loss"].detach())
            current_gradient = torch.cat([value.detach().float().reshape(-1) for value in torch.autograd.grad(losses["ce_loss"], parameters)])
            row["runs"][name] = {"ce": ce, "ce_abs_vs_full_first": abs(ce - first_ce),
                "condition_vs_full_first": tensor_delta(first_condition, condition),
                "gradient_vs_full_first": tensor_delta(gradient, current_gradient),
                "rng_motion_exit": after_motion, "rng_after_ce": rng_digest(rng_snapshot())}
            if cached_ce is not None:
                row["cached_replay_ce_abs_difference"] = abs(ce - cached_ce)
                row["cached_replay_condition"] = tensor_delta(cached_condition, condition)
            cached_ce, cached_condition = ce, condition.detach().clone()
            del condition, losses, current_gradient
        row["cached_contract_passed"] = (row["cached_replay_ce_abs_difference"] <= 2e-5
            and all(row["runs"][name]["ce_abs_vs_full_first"] <= 2e-5 for name in ("cached_first", "cached_repeat")))
        report["rows"].append(row)
        write_json(args.output, report)
        if not row["cached_contract_passed"]:
            raise RuntimeError("cached boundary does not meet the unchanged replay resolution")
        del batch, refs, frozen, first_condition, gradient, cached_condition
    changed = [name for name, value in model.named_parameters() if value._version != versions[name]]
    report["parameters_changed"] = changed
    if changed:
        write_json(args.output, report)
        raise RuntimeError("read-only audit changed parameter versions")
    report["complete"] = True
    write_json(args.output, report)
    print("COMPLETE replay audit", flush=True)


if __name__ == "__main__":
    main()
