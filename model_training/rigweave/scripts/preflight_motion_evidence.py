from __future__ import annotations

import argparse
from functools import partial
import gc
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
from transformers import LogitsProcessor, LogitsProcessorList

from eval_dynamic_rig_ce import CHECKPOINT_DEFAULTS, _build_dynamic_model
from motion_experiment_runtime import load_config, source_metadata, verify_manifests, write_json
from train_dynamic_rig import build_tokenizer, move_batch
from rigweave.dynamic_rig.data import DynamicRigManifestDataset, dynamic_rig_collate
from rigweave.dynamic_rig.motion_evidence import MotionEvidenceInjection


def set_fusion(model, fusion, heads):
    encoder = model.conditioner.motion_encoder
    encoder.motion_evidence_fusion = fusion
    encoder.motion_evidence_heads = heads
    for block in encoder.blocks:
        block.motion_injection = MotionEvidenceInjection(
            encoder.dim, block.pose_inner.self_attn.num_heads, fusion=fusion, biased_heads=heads,
        ).to(next(encoder.parameters()).device)


def difference(reference, value):
    reference, value = reference.detach().float(), value.detach().float()
    error = reference - value
    return {"max_abs": float(error.abs().max()), "rms": float(error.square().mean().sqrt()),
            "relative_l2": float(error.norm() / reference.norm().clamp_min(1e-20))}


class ForceGroundTruthPrefix(LogitsProcessor):
    """Test-only forcing to compare actual generation-cache logits on GT prefixes."""
    def __init__(self, tokens):
        self.tokens = tokens
        self.offset = 0

    def __call__(self, input_ids, scores):
        token = int(self.tokens[self.offset])
        self.offset += 1
        forced = torch.full_like(scores, -float("inf"))
        forced[:, token] = 0.0
        return forced


@torch.no_grad()
def prefix_contract(model, cond, batch):
    ids = batch["input_ids"][:, :12]
    mask = batch["attention_mask"][:, :ids.shape[1]]
    valid_count = int(mask[0].sum())
    count = min(8, valid_count - 2)
    if count < 1:
        raise RuntimeError("target is too short for a same-prefix contract check")
    cond = cond.to(model.transformer.dtype)
    embedded = model.token_inputs_embeds(ids, mask)
    inputs = torch.cat((cond, embedded), dim=1)
    attention = torch.nn.functional.pad(mask, (cond.shape[1], 0), value=1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        teacher = model.transformer(inputs_embeds=inputs, attention_mask=attention, use_cache=False).logits
        expected = teacher[:, cond.shape[1] + 1:cond.shape[1] + 1 + count].clone()
        prompt = torch.cat((cond, embedded[:, :2]), dim=1)
        captured = []
        hook = model.transformer.register_forward_hook(
            lambda module, arguments, output: captured.append(output.logits[:, -1].detach().clone()))
        try:
            model.transformer.generate(
                inputs_embeds=prompt,
                attention_mask=torch.ones(prompt.shape[:2], dtype=torch.long, device=prompt.device),
                max_new_tokens=count, do_sample=False, num_beams=1, use_cache=True,
                eos_token_id=None, pad_token_id=model.tokenizer.pad,
                logits_processor=LogitsProcessorList([ForceGroundTruthPrefix(ids[0, 2:2 + count])]),
            )
        finally:
            hook.remove()
    if len(captured) != count:
        raise RuntimeError(f"generation contract captured {len(captured)} of {count} steps")
    actual = torch.stack(captured, dim=1)
    report = difference(expected, actual)
    report["steps"] = count
    report["top1_agreement"] = float((expected.argmax(-1) == actual.argmax(-1)).float().mean())
    if report["max_abs"] > 0.25 or report["relative_l2"] > 0.005:
        raise RuntimeError(f"teacher-forcing / generation-cache mismatch: {report}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    torch.set_num_threads(1)
    torch.manual_seed(config["changes"]["initialization_seed"])
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(0.93)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("preflight must use one allocated physical H100")
    report = {"scope": "Recorded preflight only; mutation of the in-memory reference model is not a training checkpoint.",
              "source": source_metadata(config), "manifests": verify_manifests(config),
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "cases": [], "optimizer_checks": []}
    model_args = SimpleNamespace(**{**CHECKPOINT_DEFAULTS, **config["baseline"]["args"],
        "checkpoint": Path(config["baseline"]["checkpoint"]), "motion_evidence_fusion": "off",
        "motion_evidence_heads": 0, "minimum_random_frames": 1})
    tokenizer = build_tokenizer(model_args.tokenizer_config)
    model = _build_dynamic_model(model_args, tokenizer, device)
    dataset = DynamicRigManifestDataset(
        model_args.val_manifest, tokenizer, frame_count=24, limit=4, random_query=False,
        seed=model_args.seed + 17, motion_fps_ratio=0.75, minimum_random_frames=1,
        motion_vertex_samples=model_args.motion_vertex_samples,
    )
    samples = [dataset[index] for index in range(3)]
    collate = partial(dynamic_rig_collate, pad_token=tokenizer.pad)
    original = move_batch(collate([samples[0]]), device)
    variants = [("off", 0), ("bias", 2), ("bias", 4), ("bias", 8), ("token", 0),
                ("hybrid", 2), ("hybrid", 4), ("hybrid", 8)]
    for frames in [2, 8, 24]:
        batch = dict(original)
        for key in ("frame_vertices", "vertex_normals", "face_normals"):
            batch[key] = original[key][:, :frames]
        torch.manual_seed(923)
        refs = model.sample_references(batch)
        set_fusion(model, "off", 0)
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            baseline_cond = model.build_condition(batch, refs=refs)
            baseline_ce = float(model._ar_losses(baseline_cond, batch)["ce_loss"])
        for fusion, heads in variants:
            set_fusion(model, fusion, heads)
            model.eval()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                cond = model.build_condition(batch, refs=refs)
                ce = float(model._ar_losses(cond, batch)["ce_loss"])
            torch.cuda.synchronize()
            row = {"frames": frames, "fusion": fusion, "heads": heads,
                   "zero_initial_condition": difference(baseline_cond, cond),
                   "baseline_ce": baseline_ce, "ce": ce, "ce_delta": ce - baseline_ce,
                   "forward_ms": 1000 * (time.perf_counter() - start),
                   "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
            if not np.isfinite(ce) or row["zero_initial_condition"]["relative_l2"] > 0.02 or abs(row["ce_delta"]) > 0.03:
                raise RuntimeError(f"unexpected zero-initialization drift: {row}")
            if frames in (2, 24) and (fusion, heads) in (("off", 0), ("bias", 2), ("bias", 8), ("hybrid", 2)):
                row["prefix_contract"] = prefix_contract(model, cond, batch)
            report["cases"].append(row)
            write_json(args.output, report)
            print(json.dumps(row), flush=True)
            del cond
        del baseline_cond, refs
    # All original trainable routes remain active. Exercise the real maximum-T
    # microbatch including Adam states, then the minimum-T batch with the same optimizer.
    set_fusion(model, "hybrid", 8)
    model.conditioner.motion_encoder.gradient_checkpointing = True
    model.train()
    groups = [{"params": [p for p in module.parameters() if p.requires_grad], "lr": 2e-5}
              for module in (model.conditioner.motion_encoder, model.transformer, model.conditioner.surface_tokenizer)]
    optimizer = torch.optim.AdamW(groups, weight_decay=0.04)
    large = move_batch(collate(samples), device)
    for frames in [24, 2]:
        batch = dict(large)
        for key in ("frame_vertices", "vertex_normals", "face_normals"):
            batch[key] = large[key][:, :frames]
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        before = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch)
            loss = out["loss"]
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        gradients = {name: float(parameter.grad.norm()) for name, parameter in model.named_parameters()
                     if ".motion_injection." in name and parameter.grad is not None}
        if len(gradients) != 24 or not all(np.isfinite(value) and value > 0 for value in gradients.values()):
            raise RuntimeError(f"missing/non-finite evidence gradients: {gradients}")
        optimizer.step()
        torch.cuda.synchronize()
        row = {"frames": frames, "batch": 3, "loss": float(loss), "gradient_norm": float(norm),
               "adapter_gradients": gradients, "seconds": time.perf_counter() - before,
               "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
               "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
        report["optimizer_checks"].append(row)
        write_json(args.output, report)
        print("OPTIMIZER", json.dumps(row), flush=True)
        del out, loss
    del optimizer, large, original, model
    gc.collect()
    torch.cuda.empty_cache()
    report["passed"] = True
    write_json(args.output, report)
    print("PASS", args.output, flush=True)


if __name__ == "__main__":
    main()
