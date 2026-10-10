"""Paired bias-scale and sample-gradient diagnostics without optimizer updates."""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from eval_dynamic_rig_ce import CHECKPOINT_DEFAULTS, _build_dynamic_model, _control_batch
from eval_dynamic_rig_generation import _continuous_range, _dynamic_generate, _output_metrics
from motion_experiment_runtime import guard, load_config, source_metadata, verify_manifests, write_json
from train_dynamic_rig import build_tokenizer, move_batch
from rigweave.dynamic_rig.data import DynamicRigManifestDataset, dynamic_rig_collate
from rigweave.dynamic_rig.motion_evidence import MotionEvidence


def gradient_summary(vectors, coefficients):
    gradients = np.asarray(vectors, dtype=np.float64)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if gradients.ndim != 2 or gradients.shape[1:] != coefficients.shape:
        raise ValueError("gradient vectors must align with the recorded coefficients")
    if not np.isfinite(gradients).all() or not np.isfinite(coefficients).all():
        raise ValueError("non-finite gradient or coefficient")
    norms = np.linalg.norm(gradients, axis=1)
    active = norms > 0
    if not active.any():
        raise ValueError("all sample gradients are zero")
    directions = gradients[active] / norms[active, None]
    cosines = (directions @ directions.T)[np.triu_indices(len(directions), 1)]
    return {
        "samples": len(gradients), "zero_gradient_samples": int((~active).sum()),
        "norm_sum_over_sum_norms": float(np.linalg.norm(gradients.sum(0)) / norms.sum()),
        "mean_gradient_norm": float(np.linalg.norm(gradients.mean(0))),
        "median_sample_gradient_norm": float(np.median(norms)),
        "mean_pairwise_cosine": float(cosines.mean()) if len(cosines) else None,
        "negative_pairwise_cosine_fraction": float((cosines < 0).mean()) if len(cosines) else None,
        "mean_scale_derivative_at_one": float((gradients @ coefficients).mean()),
        "negative_scale_derivative_fraction": float(((gradients @ coefficients) < 0).mean()),
        "interpretation": "Gradients of equal-weight per-asset CE at this fixed checkpoint. R=1 means aligned, R=0 cancellation. This is not a reconstruction of historical training gradients. A negative scale derivative locally favors increasing the common bias multiplier.",
    }


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row["control"], row["scale"], row["frames"])
        groups.setdefault(key, []).append(row)
    baseline = {(row["index"], row["frames"]): row for row in rows
                if row["control"] == "normal" and row["scale"] == 1}
    output = {}
    for (control, scale, frames), group in groups.items():
        deltas = [row["ce"] - baseline[(row["index"], frames)]["ce"] for row in group
                  if (row["index"], frames) in baseline]
        generated = [row["generation"] for row in group if "generation" in row]
        entry = {"count": len(group), "mean_asset_ce": float(np.mean([row["ce"] for row in group])),
                 "paired_ce_delta_vs_normal_scale1": float(np.mean(deltas)) if deltas else None,
                 "generation_count": len(generated), "generation_success": sum(row["success"] for row in generated),
                 "topology_f1_all_rows": float(np.mean([row["topology_f1_or_zero"] for row in generated])) if generated else None}
        if deltas:
            rng = np.random.default_rng(20261010)
            data = np.asarray(deltas)
            estimates = data[rng.integers(0, len(data), (4000, len(data)))].mean(1)
            entry["paired_asset_bootstrap_95pct_ce_delta"] = np.quantile(estimates, [0.025, 0.975]).tolist()
        output[f"{control}_scale{scale:g}_t{frames}"] = entry
    return output


@torch.no_grad()
def evidence_summary(evidence, coefficients):
    states = evidence.states[0].float()
    result = {"valid_anchor_fraction": float(evidence.valid_groups.float().mean()), "channels": {}, "layers": []}
    for channel, name in enumerate(("u", "c", "d")):
        value = states[..., channel]
        result["channels"][name] = {"mean": float(value.mean()), "std": float(value.std(unbiased=False)),
                                    "mean_row_std": float(value.std(-1, unbiased=False).mean())}
    for layer, weights in enumerate(coefficients):
        bias = torch.einsum("ijr,hr->hij", states, weights.float())
        span = bias.max(-1).values - bias.min(-1).values
        result["layers"].append({"layer": layer, "max_row_span": float(span.max()),
                                  "median_row_span": float(span.median()),
                                  "mean_row_std": float(bias.std(-1, unbiased=False).mean())})
    result["precision_note"] = "FP32 reconstruction of anchor-only pre-softmax bias; actual attention casts to its query dtype."
    return result


@torch.no_grad()
def generation_result(model, tokenizer, batch, condition, limit):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        tokens = _dynamic_generate(model, tokenizer, batch, limit,
                                   generation_kwargs={"do_sample": False, "num_beams": 1}, precomputed_cond=condition)
    eos = bool(np.any(tokens == int(tokenizer.eos)))
    hit_max = len(tokens) - 2 >= limit and not eos
    prediction, error = None, None
    try:
        prediction = tokenizer.detokenize(tokens)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    success = eos and prediction is not None and not hit_max
    row = {"generated_ids": tokens.tolist(), "has_eos": eos, "hit_max_without_eos": hit_max,
           "success": success, "error": error, "topology_f1_or_zero": 0.0}
    if success:
        count = int(batch["joint_count"][0])
        parents = batch["target_parents"][0, :count].cpu().tolist()
        target = SimpleNamespace(joints=batch["target_joints"][0, :count].cpu().numpy(),
                                 parents=[None if value < 0 else int(value) for value in parents])
        metrics = _output_metrics(prediction, target, _continuous_range(tokenizer))
        row.update(metrics=metrics, topology_f1_or_zero=float(metrics["topology"]["edge_f1"]))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    plan = config["bias_diagnostic"]
    guard(config, "matched_eval")
    source, manifests = source_metadata(config), verify_manifests(config)
    torch.set_num_threads(1)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("bias diagnosis requires exactly one recorded GPU")
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.93)
    device = torch.device("cuda:0")
    checkpoint = Path(plan["checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    settings = {**CHECKPOINT_DEFAULTS, **payload["args"], "checkpoint": checkpoint}
    checkpoint_step = payload["step"]
    del payload
    model_args = SimpleNamespace(**settings)
    tokenizer = build_tokenizer(model_args.tokenizer_config)
    model = _build_dynamic_model(model_args, tokenizer, device)
    model.requires_grad_(False)
    named = [(name, parameter) for name, parameter in model.named_parameters()
             if name.endswith(".motion_injection.bias_weights")]
    if len(named) != 12 or any(tuple(parameter.shape) != (8, 3) for _, parameter in named):
        raise RuntimeError("expected the unchanged twelve-layer, eight-head bias model")
    parameters = [parameter.requires_grad_(True) for _, parameter in named]
    coefficients = [parameter.detach().clone() for parameter in parameters]
    flat_coefficients = torch.cat([value.reshape(-1) for value in coefficients]).cpu().tolist()
    capture = {"control": "normal", "permutation": None, "evidence": None}

    def inspect_or_permute(module, inputs):
        layer, tokens, evidence = inputs
        capture["evidence"] = evidence
        if capture["control"] != "misaligned":
            return None
        order = capture["permutation"]
        altered = MotionEvidence(evidence.states[:, order][:, :, order],
                                 evidence.anchor_features[:, order], evidence.valid_groups[:, order])
        return layer, tokens, altered

    hooks = [block.motion_injection.register_forward_pre_hook(inspect_or_permute)
             for block in model.conditioner.motion_encoder.blocks]
    report = {"source": source, "manifests": manifests, "checkpoint": str(checkpoint),
              "checkpoint_step": checkpoint_step, "plan": plan, "rows": [], "gradients": [],
              "parameter_names": [name for name, _ in named], "coefficients": flat_coefficients,
              "scope": "No optimizer or checkpoint writes. Identical query, selected frames, surface references and complete GT within each paired asset/frame comparison. Nonquery permutation is not used as a negative control.",
              "limits": "Small fixed validation subset, one training seed. Scaling a learned checkpoint is not equivalent to retraining at a higher LR; gradients describe the final checkpoint, not historical minibatches."}
    collate = partial(dynamic_rig_collate, pad_token=tokenizer.pad)
    seed = model_args.seed + 17
    try:
        for frames in plan["frames"]:
            dataset = DynamicRigManifestDataset(model_args.val_manifest, tokenizer, frame_count=frames,
                limit=plan["validation_rows"], random_query=False, seed=seed, motion_fps_ratio=0.75,
                minimum_random_frames=1, motion_vertex_samples=model_args.motion_vertex_samples)
            for index in range(len(dataset)):
                original = move_batch(collate([dataset[index]]), device)
                torch.manual_seed(seed + index)
                refs = model.sample_references(original)
                ref_hash = hashlib.sha256(refs.query_indices.detach().cpu().numpy().tobytes()).hexdigest()
                generator = torch.Generator(device=device).manual_seed(seed + index + 991)
                capture["permutation"] = torch.randperm(model_args.query_tokens, generator=generator, device=device)
                variants = [("normal", scale) for scale in plan["scales"]]
                if frames == plan["gradient_frames"] and index < plan["control_rows"]:
                    variants += [("zero", 0.0), ("zero", 1.0), ("misaligned", 1.0)]
                for control, scale in variants:
                    capture["control"] = control
                    with torch.no_grad():
                        for parameter, value in zip(parameters, coefficients):
                            parameter.copy_(value * scale)
                    batch = _control_batch(original, "zero" if control == "zero" else "normal", seed + index)
                    needs_gradient = control == "normal" and scale == 1 and frames == plan["gradient_frames"]
                    with torch.set_grad_enabled(needs_gradient), torch.autocast("cuda", dtype=torch.bfloat16):
                        condition = model.build_condition(batch, refs=refs)
                        losses = model._ar_losses(condition, batch)
                    ce = float(losses["ce_loss"].detach())
                    if not np.isfinite(ce):
                        raise RuntimeError("non-finite diagnostic CE")
                    row = {"index": index, "path": batch["path"][0], "frames": frames,
                           "control": control, "scale": scale, "ce": ce, "eos_accuracy": float(losses["eos_acc"]),
                           "selected_frames": original["selected_frames"][0].tolist(),
                           "query_center": original["query_center"][0].tolist(), "query_scale": float(original["query_scale"][0]),
                           "surface_reference_sha256": ref_hash, "joint_count": int(original["joint_count"][0])}
                    if needs_gradient:
                        gradients = torch.autograd.grad(losses["ce_loss"], parameters)
                        vector = torch.cat([value.detach().float().reshape(-1) for value in gradients]).cpu().tolist()
                        if not np.isfinite(vector).all():
                            raise RuntimeError("non-finite sample bias gradient")
                        report["gradients"].append({"index": index, "path": row["path"], "frames": frames, "vector": vector})
                        row["evidence"] = evidence_summary(capture["evidence"], coefficients)
                        if index < 2:
                            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                                repeated = model.build_condition(batch, refs=refs)
                                repeated_ce = float(model._ar_losses(repeated, batch)["ce_loss"])
                            row["replay_ce_abs_difference"] = abs(repeated_ce - ce)
                            if row["replay_ce_abs_difference"] > 2e-5:
                                raise RuntimeError("fixed-input replay noise exceeds diagnostic resolution")
                            del repeated
                        del gradients
                    if control == "normal" and frames in plan["generation_frames"] and index < plan["generation_rows"]:
                        row["generation"] = generation_result(model, tokenizer, batch, condition.detach(), plan["max_new_tokens"])
                    report["rows"].append(row)
                    del losses, condition
                capture["evidence"] = None
                del batch, original, refs
                if (index + 1) % 4 == 0:
                    report["summary"] = summarize(report["rows"])
                    write_json(args.output, report)
                    print(json.dumps({"frames": frames, "assets_complete": index + 1, "rows": len(report["rows"])}), flush=True)
    finally:
        for hook in hooks:
            hook.remove()
        with torch.no_grad():
            for parameter, value in zip(parameters, coefficients):
                parameter.copy_(value)
        assert all(torch.equal(parameter, value) for parameter, value in zip(parameters, coefficients))
    report["summary"] = summarize(report["rows"])
    report["gradient_summary"] = gradient_summary([row["vector"] for row in report["gradients"]], flat_coefficients)
    report["complete"] = True
    report["parameters_restored"] = True
    write_json(args.output, report)
    print("COMPLETE", json.dumps({"gradient_summary": report["gradient_summary"], "summary": report["summary"]}), flush=True)


if __name__ == "__main__":
    main()
