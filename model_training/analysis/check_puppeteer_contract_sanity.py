#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


class DummyConditioner(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.motion_encoder = SimpleNamespace(dim=int(dim))

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise RuntimeError("decoder-only sanity should not call conditioner")


def _insert_paths(model_root: Path) -> None:
    for path in (
        model_root / "rigweave" / "src",
        model_root / "rigweave" / "scripts",
        model_root / "third_party_references" / "Puppeteer" / "skeleton",
    ):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _infer_decoder(model_root: Path, state: dict[str, torch.Tensor], train_args: dict[str, Any]) -> tuple[nn.Module, int, dict[str, int]]:
    from rigweave.dynamic_rig import import_puppeteer_decoder

    puppeteer_root = Path(train_args.get("puppeteer_root") or model_root / "third_party_references" / "Puppeteer")
    if not puppeteer_root.exists():
        puppeteer_root = model_root / "third_party_references" / "Puppeteer"
    SkeletonOPTConfig, SkeletonOPT = import_puppeteer_decoder(puppeteer_root)

    embed = state["decoder.model.decoder.embed_tokens.weight"]
    vocab_size = int(embed.shape[0])
    hidden = int(embed.shape[1])
    layer_ids = [
        int(match.group(1))
        for key in state
        for match in [re.match(r"decoder\.model\.decoder\.layers\.(\d+)\.", key)]
        if match is not None
    ]
    layers = max(layer_ids) + 1
    ffn_dim = int(state["decoder.model.decoder.layers.0.fc1.weight"].shape[0])
    position_rows = int(state["decoder.model.decoder.embed_positions.weight"].shape[0])
    max_position_embeddings = max(2, position_rows - 2)
    heads = int(train_args.get("decoder_heads") or 16)
    if hidden == 1024 and layers == 24:
        heads = 16
    has_final_layer_norm = "decoder.model.decoder.final_layer_norm.weight" in state

    config = SkeletonOPTConfig(
        vocab_size=vocab_size,
        hidden_size=hidden,
        word_embed_proj_dim=hidden,
        ffn_dim=ffn_dim,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        max_position_embeddings=max_position_embeddings,
        n_positions=max_position_embeddings,
        dropout=float(train_args.get("decoder_dropout", 0.0) or 0.0),
        attention_dropout=float(train_args.get("decoder_attention_dropout", 0.0) or 0.0),
        activation_dropout=float(train_args.get("decoder_activation_dropout", 0.0) or 0.0),
        layerdrop=float(train_args.get("decoder_layerdrop", 0.0) or 0.0),
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        do_layer_norm_before=bool(has_final_layer_norm),
        _attn_implementation=str(train_args.get("attn_implementation") or "flash_attention_2"),
    )
    config.joint_token = True
    config.n_discrete_size = int(train_args.get("n_discrete_size", vocab_size - 3))
    config.bone_per_token = 4
    config.cond_length = int(train_args.get("cond_length", 257))
    config.word_embed_proj_dim = hidden

    decoder = SkeletonOPT(config)
    decoder_state = {key[len("decoder.") :]: value for key, value in state.items() if key.startswith("decoder.")}
    missing, unexpected = decoder.load_state_dict(decoder_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"decoder state mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    return decoder, hidden, {
        "decoder_position_rows": position_rows,
        "decoder_layers": layers,
        "decoder_heads": heads,
        "decoder_ffn_dim": ffn_dim,
    }


def _run_checkpoint(
    *,
    model_root: Path,
    manifest: Path,
    checkpoint: Path,
    output: Path,
    limit: int,
    max_positions: int,
    max_diff: float,
) -> dict[str, Any]:
    from rigweave.dynamic_rig import (
        PuppeteerDynamicRigDataset,
        PuppeteerDynamicRigModel,
        PuppeteerJointTokenizer,
        puppeteer_dynamic_collate,
    )
    from train_dynamic_rig import move_batch

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    train_args = dict(ckpt.get("args", {}) or {})
    state = ckpt["model"]
    decoder, hidden, decoder_info = _infer_decoder(model_root, state, train_args)

    n_max_joints = int(train_args.get("n_max_joints", 101))
    n_discrete_size = int(train_args.get("n_discrete_size", 128))
    cond_length = int(train_args.get("cond_length", 257))
    slot_tensor = state.get("target_aware_pos_embed")
    original_slots = None if slot_tensor is None else int(slot_tensor.shape[1])

    tokenizer = PuppeteerJointTokenizer(
        n_discrete_size=n_discrete_size,
        target_coord_scale=float(train_args.get("target_coord_scale", 0.25)),
        strict_range=not bool(train_args.get("no_strict_target_range", False)),
    )
    model = PuppeteerDynamicRigModel(
        conditioner=DummyConditioner(hidden),
        decoder=decoder,
        tokenizer=tokenizer,
        num_surface_samples=int(train_args.get("surface_samples", 65536)),
        vertex_samples=int(train_args.get("vertex_samples", 8192)),
        query_tokens=int(train_args.get("query_tokens", 1024)),
        cond_length=cond_length,
        projector_heads=int(train_args.get("projector_heads", 8)),
        max_joints=n_max_joints,
        use_joint_slot_embedding=not bool(train_args.get("no_joint_slot_embedding", False)),
        target_aware_pos_embed=slot_tensor,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    dataset = PuppeteerDynamicRigDataset(
        manifest,
        frame_count=int(train_args.get("frames", 24)),
        limit=int(limit),
        random_query=False,
        seed=int(train_args.get("seed", 20260707)) + 17,
        motion_fps_ratio=float(train_args.get("motion_fps_ratio", 0.7)),
        motion_vertex_samples=int(train_args.get("motion_vertex_samples", 512)),
        max_joints=n_max_joints,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=puppeteer_dynamic_collate)

    rows: list[dict[str, Any]] = []
    max_abs = 0.0
    failed = 0
    autocast_dtype = torch.bfloat16
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device.type == "cuda"):
        for index, batch in enumerate(loader):
            batch = move_batch(batch, device)
            raw = torch.zeros(
                (1, cond_length, hidden),
                device=device,
                dtype=autocast_dtype if device.type == "cuda" else torch.float32,
            )
            cond = model._condition_embeds_from_raw(raw)
            forced_logits, token_batch, _loss = model.teacher_forced_logits(batch, cond=cond)
            labels = token_batch.labels.to(device)
            valid_positions = (labels[0] != -100).nonzero(as_tuple=False).flatten().tolist()
            checked = 0
            row_max = 0.0
            row_mean_total = 0.0
            first_pos = -1
            first_max = 0.0
            for pos in valid_positions[: max(0, int(max_positions))]:
                if pos <= 0:
                    continue
                prefix = token_batch.input_ids[:, :pos].to(device)
                step_logits = model._next_token_logits(cond, prefix)
                diff = (forced_logits[:, pos] - step_logits).float().abs()
                item_max = float(diff.max().detach().cpu())
                item_mean = float(diff.mean().detach().cpu())
                if checked == 0:
                    first_pos = int(pos)
                    first_max = item_max
                row_max = max(row_max, item_max)
                row_mean_total += item_mean
                checked += 1
            row = {
                "index": index,
                "path": batch["path"][0],
                "joint_count": int(batch["joint_count"][0].detach().cpu()),
                "checked_positions": int(checked),
                "max_abs_logit_diff": float(row_max),
                "mean_abs_logit_diff": float(row_mean_total / max(checked, 1)),
                "first_checked_position": int(first_pos),
                "first_position_max_abs_logit_diff": float(first_max),
            }
            rows.append(row)
            max_abs = max(max_abs, row_max)
            if checked <= 0 or row_max > max_diff:
                failed += 1

    summary = {
        "checkpoint": str(checkpoint),
        "manifest": str(manifest),
        "rows": len(rows),
        "failed_rows": int(failed),
        "max_abs_logit_diff_over_rows": float(max_abs),
        "mean_abs_logit_diff_over_rows": float(sum(row["mean_abs_logit_diff"] for row in rows) / max(len(rows), 1)),
        "threshold": float(max_diff),
        "n_max_joints": int(n_max_joints),
        "original_target_aware_slots": original_slots,
        "expected_current_slots": int(n_max_joints + 1),
        "slot_contract_ok": bool(original_slots is None or original_slots == n_max_joints + 1),
        "random_init": bool(train_args.get("random_init", False)),
        **decoder_info,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Puppeteer/joint-token teacher-forcing vs generation contract.")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--name", action="append", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--max-positions", type=int, default=64)
    parser.add_argument("--max-diff", type=float, default=3.0e-2)
    args = parser.parse_args()

    _insert_paths(args.model_root)
    names = args.name or [Path(path).parent.name for path in args.checkpoint]
    if len(names) != len(args.checkpoint):
        raise ValueError("--name count must match --checkpoint count")
    summaries: list[dict[str, Any]] = []
    for name, checkpoint in zip(names, args.checkpoint):
        checkpoint_path = Path(checkpoint)
        print(f"==== {name} {checkpoint_path}", flush=True)
        summary = _run_checkpoint(
            model_root=args.model_root,
            manifest=args.manifest,
            checkpoint=checkpoint_path,
            output=args.output_dir / f"{name}.json",
            limit=args.limit,
            max_positions=args.max_positions,
            max_diff=args.max_diff,
        )
        summary = {"name": name, **summary}
        summaries.append(summary)
        print("SUMMARY " + json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"summaries": summaries}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
