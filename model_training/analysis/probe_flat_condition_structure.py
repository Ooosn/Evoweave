#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


COUNT_TARGETS = (
    "joint_count",
    "leaf_count",
    "branch_node_count",
    "max_depth",
    "root_child_count",
)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _seed_all(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("path"):
                raise ValueError(f"{path}:{line_number} has no path")
            rows.append(row)
    return rows


def _row_joint_count(row: dict[str, Any]) -> int:
    candidates = (
        row.get("canonical_metrics", {}).get("target_joint_count"),
        row.get("rootless_target_strict_metrics", {}).get("target_joint_count"),
        row.get("final_bbox_consistency_screening", {})
        .get("metrics", {})
        .get("rootless_joint_count"),
        row.get("pass1_precheck_metrics", {}).get("joint_count"),
    )
    for value in candidates:
        if value is not None:
            return int(value)
    raise ValueError(f"manifest row has no joint count: {row.get('path')}")


def _parse_bin_uppers(text: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if not values or tuple(sorted(set(values))) != values:
        raise ValueError(f"bin uppers must be strictly increasing, got {values}")
    return values


def _count_bin(value: int, uppers: tuple[int, ...]) -> int:
    for index, upper in enumerate(uppers):
        if value <= upper:
            return index
    return len(uppers)


def _stratified_rows(
    rows: list[dict[str, Any]],
    *,
    per_bin: int,
    bin_uppers: tuple[int, ...],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_count_bin(_row_joint_count(row), bin_uppers)].append(row)
    rng = np.random.default_rng(seed)
    selected: list[dict[str, Any]] = []
    report: dict[str, int] = {}
    for bin_index in range(len(bin_uppers) + 1):
        group = groups.get(bin_index, [])
        take = min(int(per_bin), len(group))
        if take:
            indices = rng.choice(len(group), size=take, replace=False)
            selected.extend(group[int(index)] for index in indices.tolist())
        low = 1 if bin_index == 0 else bin_uppers[bin_index - 1] + 1
        high = bin_uppers[bin_index] if bin_index < len(bin_uppers) else math.inf
        label = f"{low}-{high if math.isfinite(high) else 'inf'}"
        report[label] = take
    rng.shuffle(selected)
    return selected, report


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _tree_targets(parents: np.ndarray) -> dict[str, Any]:
    parents = np.asarray(parents, dtype=np.int64).reshape(-1)
    if parents.size <= 0:
        raise ValueError("empty parent tree")
    roots = np.flatnonzero(parents < 0)
    if roots.tolist() != [0]:
        raise ValueError(f"roots={roots.tolist()}, expected [0]")

    children: list[list[int]] = [[] for _ in range(parents.size)]
    depths = np.zeros(parents.size, dtype=np.int64)
    for child_index, parent_index in enumerate(parents.tolist()):
        if parent_index < 0:
            continue
        if not 0 <= parent_index < child_index:
            raise ValueError(
                f"tree is not in topological order: child={child_index} parent={parent_index}"
            )
        children[parent_index].append(child_index)
        depths[child_index] = depths[parent_index] + 1
    child_counts = np.asarray([len(value) for value in children], dtype=np.int64)
    depth_hist = np.bincount(depths, minlength=16)[:16]
    degree_hist = np.bincount(np.minimum(child_counts, 5), minlength=6)
    return {
        "joint_count": int(parents.size),
        "leaf_count": int((child_counts == 0).sum()),
        "branch_node_count": int((child_counts > 1).sum()),
        "max_depth": int(depths.max()),
        "root_child_count": int(child_counts[0]),
        "max_children": int(child_counts.max()),
        "depth_hist": depth_hist.astype(np.float32),
        "degree_hist": degree_hist.astype(np.float32),
        "topology_signature": ",".join(str(int(value)) for value in parents.tolist()),
    }


def _metadata_features(batch: dict[str, Any]) -> np.ndarray:
    vertex_count = int(batch["vertex_count"][0].item())
    face_count = int(batch["face_count"][0].item())
    vertices = batch["frame_vertices"][0, :, :vertex_count].float()
    query = vertices[0]
    displacement = torch.linalg.vector_norm(vertices - query.unsqueeze(0), dim=-1)
    max_vertex_motion = displacement.max(dim=0).values
    covariance = torch.cov(query.T)
    eigvals = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    extents = query.amax(dim=0) - query.amin(dim=0)
    values = [
        math.log1p(vertex_count),
        math.log1p(face_count),
        float(displacement.mean().item()),
        float(torch.quantile(max_vertex_motion, 0.50).item()),
        float(torch.quantile(max_vertex_motion, 0.90).item()),
        float(torch.quantile(max_vertex_motion, 0.99).item()),
        float(max_vertex_motion.max().item()),
        float((max_vertex_motion > 1.0e-3).float().mean().item()),
        float((max_vertex_motion > 1.0e-2).float().mean().item()),
        *[float(value) for value in extents.tolist()],
        *[float(value) for value in eigvals.tolist()],
    ]
    return np.asarray(values, dtype=np.float32)


@torch.no_grad()
def _extract_features(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    loader: DataLoader,
    device: torch.device,
    seed: int,
    label: str,
) -> dict[str, Any]:
    features: dict[str, list[np.ndarray]] = {
        "metadata": [],
        "condition_mean": [],
        "condition_mean_std": [],
        "decoder_class": [],
    }
    targets: dict[str, list[Any]] = defaultdict(list)
    paths: list[str] = []
    query_frames: list[int] = []

    for index, batch in enumerate(loader):
        batch = _move_batch(batch, device)
        _seed_all(seed + index, device)
        refs = model.sample_references(batch)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            condition = model.build_condition(batch, refs=refs).to(dtype=model.transformer.dtype)
            start_tokens = torch.tensor(
                [tokenizer.bos, tokenizer.cls_name_to_token(batch["cls"][0])],
                device=device,
                dtype=torch.long,
            )
            start_mask = torch.ones((1, 2), device=device, dtype=torch.long)
            start_embed = model.token_inputs_embeds(start_tokens.unsqueeze(0), start_mask)
            prompt = torch.cat([condition, start_embed], dim=1)
            output = model.transformer(
                inputs_embeds=prompt,
                attention_mask=torch.ones(prompt.shape[:2], device=device, dtype=torch.long),
                use_cache=False,
                output_hidden_states=True,
            )

        condition_f = condition[0].float()
        condition_mean = condition_f.mean(dim=0)
        condition_std = condition_f.std(dim=0, unbiased=False)
        decoder_class = output.hidden_states[-1][0, -1].float()
        joint_count = int(batch["joint_count"][0].item())
        parents = (
            batch["target_parents"][0, :joint_count].detach().cpu().numpy().astype(np.int64)
        )
        tree = _tree_targets(parents)

        features["metadata"].append(_metadata_features(batch))
        features["condition_mean"].append(condition_mean.cpu().numpy())
        features["condition_mean_std"].append(
            torch.cat([condition_mean, condition_std], dim=0).cpu().numpy()
        )
        features["decoder_class"].append(decoder_class.cpu().numpy())
        for target_name in COUNT_TARGETS:
            targets[target_name].append(float(tree[target_name]))
        targets["depth_hist"].append(tree["depth_hist"])
        targets["degree_hist"].append(tree["degree_hist"])
        targets["topology_signature"].append(tree["topology_signature"])
        targets["max_children"].append(float(tree["max_children"]))
        paths.append(str(batch["path"][0]))
        query_frames.append(int(batch["selected_frames"][0, 0].item()))

        if (index + 1) % 10 == 0 or index + 1 == len(loader):
            print(f"[{label}] extracted {index + 1}/{len(loader)}", flush=True)

        del output, prompt, start_embed, condition

    return {
        "features": {
            name: np.stack(values, axis=0).astype(np.float32)
            for name, values in features.items()
        },
        "targets": {
            name: (
                np.stack(values, axis=0).astype(np.float32)
                if name in {"depth_hist", "degree_hist"}
                else np.asarray(values, dtype=object if name == "topology_signature" else np.float32)
            )
            for name, values in targets.items()
        },
        "paths": np.asarray(paths, dtype=object),
        "query_frames": np.asarray(query_frames, dtype=np.int64),
    }


def _safe_spearman(target: np.ndarray, prediction: np.ndarray) -> float | None:
    from scipy.stats import spearmanr

    value = spearmanr(target, prediction).statistic
    return float(value) if np.isfinite(value) else None


def _fit_count_probes(
    train: dict[str, Any],
    valid: dict[str, Any],
    *,
    alphas: np.ndarray,
) -> dict[str, Any]:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    report: dict[str, Any] = {}
    for feature_name, train_values in train["features"].items():
        valid_values = valid["features"][feature_name]
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_values)
        valid_scaled = scaler.transform(valid_values)
        feature_report: dict[str, Any] = {}
        for target_name in COUNT_TARGETS:
            train_target = train["targets"][target_name].astype(np.float64)
            valid_target = valid["targets"][target_name].astype(np.float64)
            model = RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_absolute_error")
            model.fit(train_scaled, train_target)
            prediction = model.predict(valid_scaled)
            baseline = np.full_like(valid_target, np.median(train_target))
            residual = valid_target - prediction
            total = valid_target - valid_target.mean()
            r2 = 1.0 - float(np.square(residual).sum() / max(np.square(total).sum(), 1.0e-12))
            feature_report[target_name] = {
                "alpha": float(model.alpha_),
                "mae": float(np.abs(residual).mean()),
                "rmse": float(np.sqrt(np.square(residual).mean())),
                "r2": r2,
                "spearman": _safe_spearman(valid_target, prediction),
                "rounded_exact_rate": float((np.rint(prediction) == valid_target).mean()),
                "within_1_rate": float((np.abs(prediction - valid_target) <= 1.0).mean()),
                "median_baseline_mae": float(np.abs(valid_target - baseline).mean()),
                "mae_fraction_of_baseline": float(
                    np.abs(residual).mean() / max(np.abs(valid_target - baseline).mean(), 1.0e-12)
                ),
            }
        report[feature_name] = feature_report
    return report


def _fit_histogram_probes(
    train: dict[str, Any],
    valid: dict[str, Any],
    *,
    alphas: np.ndarray,
) -> dict[str, Any]:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    report: dict[str, Any] = {}
    for feature_name, train_values in train["features"].items():
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_values)
        valid_scaled = scaler.transform(valid["features"][feature_name])
        feature_report: dict[str, Any] = {}
        for target_name in ("depth_hist", "degree_hist"):
            train_target = train["targets"][target_name].astype(np.float64)
            valid_target = valid["targets"][target_name].astype(np.float64)
            model = RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_absolute_error")
            model.fit(train_scaled, train_target)
            prediction = np.clip(model.predict(valid_scaled), 0.0, None)
            baseline_row = np.median(train_target, axis=0, keepdims=True)
            baseline = np.repeat(baseline_row, valid_target.shape[0], axis=0)
            mae = float(np.abs(prediction - valid_target).mean())
            baseline_mae = float(np.abs(baseline - valid_target).mean())
            per_row_l1 = np.abs(prediction - valid_target).sum(axis=1)
            feature_report[target_name] = {
                "alpha": float(model.alpha_),
                "cell_mae": mae,
                "row_l1_mean": float(per_row_l1.mean()),
                "median_baseline_cell_mae": baseline_mae,
                "mae_fraction_of_baseline": float(mae / max(baseline_mae, 1.0e-12)),
            }
        report[feature_name] = feature_report
    return report


def _cosine_nearest_indices(train: np.ndarray, valid: np.ndarray) -> np.ndarray:
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train).astype(np.float64)
    valid_scaled = scaler.transform(valid).astype(np.float64)
    train_norm = np.linalg.norm(train_scaled, axis=1, keepdims=True)
    valid_norm = np.linalg.norm(valid_scaled, axis=1, keepdims=True)
    train_scaled /= np.maximum(train_norm, 1.0e-12)
    valid_scaled /= np.maximum(valid_norm, 1.0e-12)
    return np.argmax(valid_scaled @ train_scaled.T, axis=1)


def _topology_retrieval(train: dict[str, Any], valid: dict[str, Any]) -> dict[str, Any]:
    train_signature = train["targets"]["topology_signature"].astype(str)
    valid_signature = valid["targets"]["topology_signature"].astype(str)
    train_count = train["targets"]["joint_count"].astype(int)
    valid_count = valid["targets"]["joint_count"].astype(int)
    global_mode = Counter(train_signature.tolist()).most_common(1)[0][0]
    count_modes: dict[int, str] = {}
    for count in np.unique(train_count):
        values = train_signature[train_count == count].tolist()
        count_modes[int(count)] = Counter(values).most_common(1)[0][0]
    count_oracle_prediction = np.asarray(
        [count_modes.get(int(count), global_mode) for count in valid_count],
        dtype=str,
    )

    report: dict[str, Any] = {
        "global_mode_exact_rate": float((valid_signature == global_mode).mean()),
        "joint_count_oracle_mode_exact_rate": float(
            (valid_signature == count_oracle_prediction).mean()
        ),
    }
    for feature_name, train_values in train["features"].items():
        nearest = _cosine_nearest_indices(train_values, valid["features"][feature_name])
        predicted_signature = train_signature[nearest]
        predicted_count = train_count[nearest]
        report[feature_name] = {
            "nearest_exact_topology_rate": float(
                (predicted_signature == valid_signature).mean()
            ),
            "nearest_joint_count_exact_rate": float(
                (predicted_count == valid_count).mean()
            ),
            "nearest_joint_count_mae": float(np.abs(predicted_count - valid_count).mean()),
        }
    return report


def _save_cache(path: Path, payload: dict[str, Any]) -> None:
    arrays: dict[str, np.ndarray] = {
        "paths": payload["paths"],
        "query_frames": payload["query_frames"],
    }
    arrays.update(
        {f"feature__{name}": value for name, value in payload["features"].items()}
    )
    arrays.update(
        {f"target__{name}": value for name, value in payload["targets"].items()}
    )
    np.savez_compressed(path, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a trained flat UniRig model and test whether its 1024 condition "
            "tokens linearly expose target tree structure."
        )
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--unirig-checkpoint", type=Path, required=True)
    parser.add_argument("--train-per-bin", type=int, default=32)
    parser.add_argument("--valid-per-bin", type=int, default=16)
    parser.add_argument("--joint-count-bin-uppers", type=str, default="10,20,40,60,80,101")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    scripts_dir = args.model_root / "rigweave" / "scripts"
    source_dir = args.model_root / "rigweave" / "src"
    sys.path.insert(0, str(source_dir))
    sys.path.insert(0, str(scripts_dir))

    from eval_dynamic_rig_ce import (
        CHECKPOINT_DEFAULTS,
        _build_dynamic_model,
        apply_checkpoint_eval_defaults,
    )
    from rigweave.dynamic_rig.data import (
        DynamicRigManifestDataset,
        dynamic_rig_collate,
    )
    from train_dynamic_rig import build_tokenizer

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bin_uppers = _parse_bin_uppers(args.joint_count_bin_uppers)
    train_rows, train_selection = _stratified_rows(
        _read_manifest(args.train_manifest),
        per_bin=args.train_per_bin,
        bin_uppers=bin_uppers,
        seed=args.seed,
    )
    valid_rows, valid_selection = _stratified_rows(
        _read_manifest(args.valid_manifest),
        per_bin=args.valid_per_bin,
        bin_uppers=bin_uppers,
        seed=args.seed + 1,
    )
    train_selected_manifest = args.output_dir / "train_selected.jsonl"
    valid_selected_manifest = args.output_dir / "valid_selected.jsonl"
    _write_manifest(train_selected_manifest, train_rows)
    _write_manifest(valid_selected_manifest, valid_rows)
    print(
        json.dumps(
            {
                "train_selection": train_selection,
                "valid_selection": valid_selection,
                "train_rows": len(train_rows),
                "valid_rows": len(valid_rows),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    namespace = argparse.Namespace(
        checkpoint=args.checkpoint,
        tokenizer_config=args.tokenizer_config,
        model_config=args.model_config,
        unirig_checkpoint=args.unirig_checkpoint,
    )
    for name in CHECKPOINT_DEFAULTS:
        if not hasattr(namespace, name):
            setattr(namespace, name, None)
    apply_checkpoint_eval_defaults(namespace)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = build_tokenizer(args.tokenizer_config)

    def make_loader(manifest: Path, seed: int) -> DataLoader:
        dataset = DynamicRigManifestDataset(
            manifest,
            tokenizer,
            frame_count=namespace.frames,
            random_query=True,
            seed=seed,
            motion_fps_ratio=namespace.motion_fps_ratio,
            motion_vertex_samples=namespace.motion_vertex_samples,
            target_active_skin_only=namespace.target_active_skin_only,
            active_skin_threshold=namespace.active_skin_threshold,
            target_start_policy=namespace.target_start_policy,
            target_root_policy=namespace.target_root_policy,
            input_space_policy=namespace.input_space_policy,
        )
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=partial(dynamic_rig_collate, pad_token=tokenizer.pad),
        )

    model = _build_dynamic_model(namespace, tokenizer, device)
    model.eval()
    train_payload = _extract_features(
        model=model,
        tokenizer=tokenizer,
        loader=make_loader(train_selected_manifest, args.seed),
        device=device,
        seed=args.seed,
        label="train",
    )
    _save_cache(args.output_dir / "train_features.npz", train_payload)
    valid_payload = _extract_features(
        model=model,
        tokenizer=tokenizer,
        loader=make_loader(valid_selected_manifest, args.seed + 1),
        device=device,
        seed=args.seed + 100_000,
        label="valid",
    )
    _save_cache(args.output_dir / "valid_features.npz", valid_payload)

    alphas = np.logspace(-4, 4, num=17)
    report = {
        "checkpoint": str(args.checkpoint),
        "train_manifest": str(args.train_manifest),
        "valid_manifest": str(args.valid_manifest),
        "seed": args.seed,
        "selection": {
            "bin_uppers": list(bin_uppers),
            "train": train_selection,
            "valid": valid_selection,
            "train_rows": len(train_rows),
            "valid_rows": len(valid_rows),
        },
        "count_probes": _fit_count_probes(
            train_payload,
            valid_payload,
            alphas=alphas,
        ),
        "histogram_probes": _fit_histogram_probes(
            train_payload,
            valid_payload,
            alphas=alphas,
        ),
        "topology_retrieval": _topology_retrieval(train_payload, valid_payload),
    }
    output = args.output_dir / "condition_structure_probe.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
