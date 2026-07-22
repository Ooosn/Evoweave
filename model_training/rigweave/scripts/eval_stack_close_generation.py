#!/usr/bin/env python3
from __future__ import annotations

import argparse
from functools import partial
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_dynamic_rig_generation import (  # noqa: E402
    _continuous_range,
    _output_metrics,
    _summarize,
)
from train_dynamic_rig import (  # noqa: E402
    build_tokenizer,
    load_unirig,
    move_batch,
    move_dynamic_model_to_device,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Free-generation evaluation for the isolated stack-close route.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=1_100)
    parser.add_argument("--visual-limit", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument(
        "--surface-seed",
        type=int,
        help=(
            "Base seed for order-invariant per-row surface/FPS sampling. "
            "Defaults to --seed."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--indices-file",
        type=Path,
        help="Optional newline-delimited manifest indices. Preserves original dataset indices and poses.",
    )
    parser.add_argument("--ablate-motion-features", action="store_true")
    parser.add_argument("--ablate-time-embedding", action="store_true")
    return parser


def _required_path(
    train_args: dict[str, Any],
    key: str,
    override: Path | None = None,
) -> Path:
    value = override if override is not None else train_args.get(key)
    if value is None:
        raise ValueError(f"checkpoint is missing required training argument {key!r}")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _build_model(
    train_args: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[Any, Any, torch.nn.Module]:
    from rigweave.dynamic_rig import (
        AnchorWiseAlternatingMotionEncoder,
        FixedQuerySurfaceTokenizer,
    )
    from rigweave.dynamic_rig.model import DynamicRigConditioner
    from rigweave.stack_close import (
        PrefixPerturbationConfig,
        StackCloseDynamicRigAR,
        StackCloseTokenizer,
    )

    legacy_tokenizer = build_tokenizer(
        _required_path(train_args, "tokenizer_config"),
    )
    stack_tokenizer = StackCloseTokenizer(legacy_tokenizer)
    unirig = load_unirig(
        stack_tokenizer,
        _required_path(train_args, "model_config"),
        _required_path(train_args, "unirig_checkpoint"),
    )
    surface_tokenizer = FixedQuerySurfaceTokenizer(
        unirig.mesh_encoder,
        unirig.output_proj,
    )
    motion_encoder = AnchorWiseAlternatingMotionEncoder(
        dim=unirig.hidden_size,
        depth=int(train_args.get("motion_depth", 12)),
        heads=int(train_args.get("motion_heads", 8)),
        register_tokens=int(train_args.get("register_tokens", 96)),
        max_frames=max(int(train_args.get("frames", 24)), 48),
        use_motion_features=bool(train_args.get("use_motion_features", True)),
        use_time_embedding=bool(train_args.get("use_time_embedding", True)),
        gradient_checkpointing=True,
    )
    conditioner = DynamicRigConditioner(surface_tokenizer, motion_encoder)
    perturbation = PrefixPerturbationConfig(
        row_probability=float(train_args.get("perturb_row_probability", 0.5)),
        axial_fraction_max=float(
            train_args.get("perturb_axial_fraction_max", 0.05)
        ),
        radial_fraction_max=float(
            train_args.get("perturb_radial_fraction_max", 0.05)
        ),
        max_perturbed_joints=int(train_args.get("perturb_max_joints", 4)),
        max_joint_fraction=float(
            train_args.get("perturb_max_joint_fraction", 0.08)
        ),
        warmup_samples=int(train_args.get("perturb_warmup_samples", 5_000)),
        ramp_samples=int(train_args.get("perturb_ramp_samples", 15_000)),
    )
    refresh_layers = tuple(
        int(part.strip())
        for part in str(
            train_args.get("condition_refresh_layers", "")
        ).split(",")
        if part.strip()
    )
    model_kwargs = {
        "perturbation": perturbation,
        "stack_action_loss_weight": float(
            train_args.get("stack_action_loss_weight", 0.0)
        ),
        "stack_action_condition_dim": int(
            train_args.get("stack_action_condition_dim", 0)
        ),
        "stack_action_condition_heads": int(
            train_args.get("stack_action_condition_heads", 8)
        ),
        "num_surface_samples": int(
            train_args.get("surface_samples", 65_536)
        ),
        "vertex_samples": int(train_args.get("vertex_samples", 8_192)),
        "query_tokens": int(train_args.get("query_tokens", 1_024)),
    }
    if refresh_layers:
        from rigweave.stack_close_refresh import (
            ConditionRefreshStackCloseDynamicRigAR,
        )

        model = ConditionRefreshStackCloseDynamicRigAR(
            unirig,
            conditioner,
            stack_tokenizer,
            refresh_layer_indices=refresh_layers,
            refresh_dim=int(
                train_args.get("condition_refresh_dim", 256)
            ),
            refresh_heads=int(
                train_args.get("condition_refresh_heads", 8)
            ),
            **model_kwargs,
        )
    else:
        model = StackCloseDynamicRigAR(
            unirig,
            conditioner,
            stack_tokenizer,
            **model_kwargs,
        )
    return legacy_tokenizer, stack_tokenizer, model


def _parents_array(output: Any) -> np.ndarray:
    return np.asarray(
        [-1 if value is None else int(value) for value in output.parents],
        dtype=np.int64,
    )


def _plot_skeleton(
    axis: Any,
    joints: np.ndarray,
    parents: np.ndarray,
    *,
    color: str,
    linewidth: float,
    alpha: float,
) -> None:
    for child, parent in enumerate(parents.tolist()):
        if parent < 0 or parent >= joints.shape[0]:
            continue
        axis.plot(
            [joints[parent, 0], joints[child, 0]],
            [joints[parent, 1], joints[child, 1]],
            [joints[parent, 2], joints[child, 2]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )
    if joints.shape[0] > 0:
        axis.scatter(
            joints[:, 0],
            joints[:, 1],
            joints[:, 2],
            s=13,
            color=color,
            alpha=alpha,
            depthshade=False,
        )
        axis.scatter(
            [joints[0, 0]],
            [joints[0, 1]],
            [joints[0, 2]],
            s=58,
            color=color,
            edgecolors="black",
            depthshade=False,
        )


def _set_equal_axes(axis: Any, points: np.ndarray) -> None:
    if points.size == 0:
        center = np.zeros((3,), dtype=np.float32)
        radius = 1.0
    else:
        lo = points.min(axis=0)
        hi = points.max(axis=0)
        center = (lo + hi) * 0.5
        radius = max(float((hi - lo).max()) * 0.55, 0.1)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.view_init(elev=18, azim=-65)
    axis.set_xlabel("x", fontsize=6)
    axis.set_ylabel("y", fontsize=6)
    axis.set_zlabel("z", fontsize=6)
    axis.tick_params(labelsize=5)


def _mesh_for_row(row: dict[str, Any]) -> np.ndarray:
    with np.load(row["path"], allow_pickle=True) as raw:
        frame = np.asarray(
            raw["frame_vertices_rootspace"],
            dtype=np.float32,
        )[int(row["query_frame"])]
    center = np.asarray(row["query_center"], dtype=np.float32)
    scale = max(float(row["query_scale"]), 1.0e-8)
    mesh = (frame - center) / scale
    if mesh.shape[0] > 2_500:
        rng = np.random.default_rng(7_000 + int(row["index"]))
        mesh = mesh[rng.choice(mesh.shape[0], size=2_500, replace=False)]
    return mesh


def _make_visual(
    row: dict[str, Any],
    tokenizer: Any,
    output: Path,
) -> None:
    target = tokenizer.detokenize(np.asarray(row["target_ids"], dtype=np.int64))
    generated_ids = np.asarray(
        row["stack_close"]["generated_ids"],
        dtype=np.int64,
    )
    pred = tokenizer.detokenize(generated_ids)
    target_joints = np.asarray(target.joints, dtype=np.float32)
    pred_joints = np.asarray(pred.joints, dtype=np.float32)
    target_parents = _parents_array(target)
    pred_parents = _parents_array(pred)
    mesh = _mesh_for_row(row)
    all_points = np.concatenate([mesh, target_joints, pred_joints], axis=0)
    metrics = row["stack_close"]["metrics"]
    f1 = float(metrics["topology"]["edge_f1"])
    j2j = float(metrics["official"]["j2j"])
    target_count = int(metrics["target_joint_count"])
    pred_count = int(metrics["pred_joint_count"])

    figure = plt.figure(figsize=(15, 5), dpi=160)
    figure.suptitle(
        f"idx={row['index']} target={target_count} pred={pred_count} "
        f"F1={f1:.3f} J2J={j2j:.4f} {Path(row['path']).name}",
        fontsize=9,
    )
    for column, (title, show_target, show_pred) in enumerate(
        (
            ("GT", True, False),
            ("Prediction", False, True),
            ("Overlay", True, True),
        ),
        start=1,
    ):
        axis = figure.add_subplot(1, 3, column, projection="3d")
        axis.scatter(
            mesh[:, 0],
            mesh[:, 1],
            mesh[:, 2],
            s=0.6,
            color="#aeb7c2",
            alpha=0.16,
            depthshade=False,
        )
        if show_target:
            _plot_skeleton(
                axis,
                target_joints,
                target_parents,
                color="#1b9e77",
                linewidth=1.8,
                alpha=0.92,
            )
        if show_pred:
            _plot_skeleton(
                axis,
                pred_joints,
                pred_parents,
                color="#d95f02",
                linewidth=1.25,
                alpha=0.82,
            )
        axis.set_title(title, fontsize=9)
        _set_equal_axes(axis, all_points)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)


def _visual_indices(rows: list[dict[str, Any]], limit: int) -> list[int]:
    candidates = [
        row
        for row in rows
        if row["stack_close"].get("detokenize_ok", False)
    ]
    candidates.sort(
        key=lambda row: float(
            row["stack_close"]["metrics"]["topology"]["edge_f1"]
        )
    )
    if len(candidates) <= limit:
        return [int(row["index"]) for row in candidates]
    positions = np.linspace(0, len(candidates) - 1, num=limit)
    return [
        int(candidates[int(round(position))]["index"])
        for position in positions.tolist()
    ]


def _finite_mean(rows: list[dict[str, Any]], path: tuple[str, ...]) -> float | None:
    values: list[float] = []
    for row in rows:
        value: Any = row
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return None if not values else float(np.mean(values))


def _load_manifest_indices(path: Path, dataset_size: int) -> list[int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    indices = [
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not indices:
        raise ValueError("--indices-file is empty")
    if len(indices) != len(set(indices)):
        raise ValueError("--indices-file contains duplicate indices")
    invalid = [index for index in indices if index < 0 or index >= dataset_size]
    if invalid:
        raise ValueError(
            f"manifest indices outside [0,{dataset_size}): {invalid[:20]}"
        )
    return indices


def _row_surface_seed(base_seed: int, manifest_index: int) -> int:
    return (
        int(base_seed) * 1_000_003 + int(manifest_index) * 97_409
    ) % (2**63 - 1)


def main() -> None:
    args = _parser().parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA was requested but is not available")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = dict(payload.get("args", {}) or {})
    manifest = _required_path(train_args, "train_manifest", args.manifest)
    device = torch.device(args.device)
    amp_name = str(train_args.get("amp_dtype", "bf16"))
    amp_dtype = torch.bfloat16 if amp_name == "bf16" else torch.float16

    legacy_tokenizer, tokenizer, model = _build_model(
        train_args,
        device=device,
    )
    model.load_state_dict(payload["model"], strict=True)
    move_dynamic_model_to_device(model, device)
    if model.stack_action_head is not None:
        model.stack_action_head.to(device)
    if hasattr(model, "condition_refresh_adapters"):
        model.condition_refresh_adapters.to(device)
    motion_encoder = model.conditioner.motion_encoder
    trained_condition_flags = {
        "use_motion_features": bool(motion_encoder.use_motion_features),
        "use_time_embedding": bool(motion_encoder.use_time_embedding),
    }
    if args.ablate_motion_features:
        motion_encoder.use_motion_features = False
    if args.ablate_time_embedding:
        motion_encoder.use_time_embedding = False
    model.eval()

    from rigweave.stack_close import (
        StackCloseManifestDataset,
        stack_close_collate,
    )

    dataset = StackCloseManifestDataset(
        manifest,
        legacy_tokenizer=legacy_tokenizer,
        stack_tokenizer=tokenizer,
        frame_count=int(train_args.get("frames", 24)),
        limit=0 if args.indices_file is not None else args.limit,
        random_query=False,
        random_sibling_order=False,
        seed=args.seed,
        motion_fps_ratio=float(train_args.get("motion_fps_ratio", 0.7)),
        motion_vertex_samples=int(
            train_args.get("motion_vertex_samples", 512)
        ),
    )
    selected_indices = None
    evaluation_dataset = dataset
    if args.indices_file is not None:
        selected_indices = _load_manifest_indices(args.indices_file, len(dataset))
        evaluation_dataset = Subset(dataset, selected_indices)
    loader = DataLoader(
        evaluation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=partial(stack_close_collate, pad_token=tokenizer.pad),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "rows.jsonl"
    rows_path.write_text("", encoding="utf-8")
    rows: list[dict[str, Any]] = []
    metric_range = _continuous_range(tokenizer)
    surface_seed_base = int(
        args.seed if args.surface_seed is None else args.surface_seed
    )
    fork_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    with torch.inference_mode():
        for evaluation_position, batch in enumerate(loader):
            index = (
                selected_indices[evaluation_position]
                if selected_indices is not None
                else evaluation_position
            )
            batch = move_batch(batch, device)
            valid = batch["attention_mask"][0].bool()
            target_ids = (
                batch["target_ids"][0][valid]
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64)
            )
            target = tokenizer.detokenize(target_ids)
            block: dict[str, Any] = {"detokenize_ok": False}
            try:
                row_surface_seed = _row_surface_seed(surface_seed_base, index)
                with torch.random.fork_rng(devices=fork_devices):
                    torch.manual_seed(row_surface_seed)
                    if device.type == "cuda":
                        torch.cuda.manual_seed_all(row_surface_seed)
                    with torch.autocast(
                        device_type="cuda",
                        dtype=amp_dtype,
                        enabled=device.type == "cuda",
                    ):
                        generated_ids = model.generate_batch_item(
                            batch,
                            row=0,
                            max_new_tokens=args.max_new_tokens,
                        )
                eos_hits = np.flatnonzero(generated_ids == int(tokenizer.eos))
                pred = tokenizer.detokenize(generated_ids)
                block = {
                    "detokenize_ok": True,
                    "generated_ids": generated_ids.astype(int).tolist(),
                    "generated_new_tokens": int(generated_ids.shape[0] - 2),
                    "has_eos": bool(eos_hits.size),
                    "eos_index": (
                        int(eos_hits[0]) if eos_hits.size else None
                    ),
                    "eos_new_token_index": (
                        int(eos_hits[0] - 2)
                        if eos_hits.size and int(eos_hits[0]) >= 2
                        else None
                    ),
                    "hit_max_without_eos": bool(
                        generated_ids.shape[0] - 2 >= args.max_new_tokens
                        and not eos_hits.size
                    ),
                    "metrics": _output_metrics(pred, target, metric_range),
                }
            except Exception as exc:
                block = {
                    "detokenize_ok": False,
                    "generated_ids": [],
                    "generated_new_tokens": 0,
                    "has_eos": False,
                    "hit_max_without_eos": False,
                    "error": repr(exc),
                    "metrics": _output_metrics(None, target, metric_range),
                }

            selected_frames = (
                batch["selected_frames"][0]
                .detach()
                .cpu()
                .numpy()
                .astype(int)
                .tolist()
            )
            row = {
                "index": index,
                "evaluation_position": evaluation_position,
                "path": batch["path"][0],
                "selected_frames": selected_frames,
                "query_frame": int(selected_frames[0]),
                "surface_seed": _row_surface_seed(surface_seed_base, index),
                "query_center": (
                    batch["query_center"][0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(float)
                    .tolist()
                ),
                "query_scale": float(
                    batch["query_scale"][0].detach().cpu()
                ),
                "target_ids": target_ids.astype(int).tolist(),
                "target_token_count": int(target_ids.shape[0]),
                "target_joint_count": int(target.joints.shape[0]),
                "max_new_tokens": args.max_new_tokens,
                "eos": int(tokenizer.eos),
                "stack_close": block,
            }
            rows.append(row)
            with rows_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            metrics = block.get("metrics", {})
            print(
                json.dumps(
                    {
                        "index": index,
                        "evaluation_position": evaluation_position,
                        "target_joints": int(target.joints.shape[0]),
                        "pred_joints": metrics.get("pred_joint_count"),
                        "has_eos": block.get("has_eos"),
                        "hit_max": block.get("hit_max_without_eos"),
                        "j2j": metrics.get("official", {}).get("j2j"),
                        "topology_f1": metrics.get("topology", {}).get(
                            "edge_f1"
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": int(payload.get("step", -1)),
        "checkpoint_sample_seen": payload.get("sample_seen"),
        "manifest": str(manifest),
        "manifest_indices": selected_indices,
        "condition_flags": {
            "trained": trained_condition_flags,
            "evaluated": {
                "use_motion_features": bool(motion_encoder.use_motion_features),
                "use_time_embedding": bool(motion_encoder.use_time_embedding),
            },
        },
        "stack_action": {
            "enabled": model.stack_action_head is not None,
            "loss_weight": float(model.stack_action_loss_weight),
            "condition_dim": int(
                getattr(model.stack_action_head, "condition_dim", 0)
                if model.stack_action_head is not None
                else 0
            ),
        },
        "metric_contract": "shared_eval_dynamic_rig_generation",
        "surface_sampling": {
            "base_seed": surface_seed_base,
            "per_manifest_index": True,
        },
        "generation": _summarize(rows, "stack_close"),
        "mean_topology_f1": _finite_mean(
            rows,
            ("stack_close", "metrics", "topology", "edge_f1"),
        ),
        "mean_j2j": _finite_mean(
            rows,
            ("stack_close", "metrics", "official", "j2j"),
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    visual_dir = args.output_dir / "visuals"
    row_by_index = {int(row["index"]): row for row in rows}
    for index in _visual_indices(rows, max(args.visual_limit, 0)):
        row = row_by_index[index]
        metrics = row["stack_close"]["metrics"]
        f1 = float(metrics["topology"]["edge_f1"])
        _make_visual(
            row,
            tokenizer,
            visual_dir / f"idx{index:03d}_f1{f1:.3f}.png",
        )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
