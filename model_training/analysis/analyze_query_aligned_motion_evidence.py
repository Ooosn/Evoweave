#!/usr/bin/env python3
"""Test whether local motion survives aggregation to the 1024 decoder anchors."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_TRAINING_ROOT = SCRIPT_DIR.parent
RIGWEAVE_SCRIPTS = MODEL_TRAINING_ROOT / "rigweave" / "scripts"
if str(RIGWEAVE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(RIGWEAVE_SCRIPTS))

from analyze_local_motion_evidence import (  # noqa: E402
    _corrupt_within_groups,
    _per_asset_metrics,
    _score_metrics,
)
from train_dynamic_rig import build_tokenizer, move_batch  # noqa: E402

from rigweave.dynamic_rig.data import (  # noqa: E402
    DynamicRigManifestDataset,
    dynamic_rig_collate,
)
from rigweave.dynamic_rig.sampling import sample_trackable_surface  # noqa: E402
from rigweave.motion_evidence import (  # noqa: E402
    TopologyLocalMotionEvidence,
    query_aligned_skin_boundary_targets,
)


@dataclass(frozen=True)
class QueryArrays:
    features: np.ndarray
    labels: np.ndarray
    weights: np.ndarray
    groups: np.ndarray
    paths: np.ndarray
    confidence: np.ndarray
    motion_q90_rms: np.ndarray
    valid_fraction: np.ndarray
    spatial_std: np.ndarray
    elapsed_seconds: float


class QueryEvidenceMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=1024)
    parser.add_argument("--valid-limit", type=int, default=0)
    parser.add_argument("--frame-count", type=int, default=24)
    parser.add_argument("--surface-samples", type=int, default=65536)
    parser.add_argument("--vertex-samples", type=int, default=8192)
    parser.add_argument("--query-tokens", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def make_dataset(
    manifest: Path,
    tokenizer: Any,
    *,
    frame_count: int,
    seed: int,
) -> DynamicRigManifestDataset:
    return DynamicRigManifestDataset(
        manifest,
        tokenizer,
        frame_count=frame_count,
        random_query=False,
        seed=seed,
        motion_fps_ratio=0.7,
        motion_vertex_samples=512,
        motion_alignment_policy="none",
        target_active_skin_only=False,
        active_skin_threshold=1.0e-4,
        target_start_policy="joint0",
        target_root_policy="legacy",
        input_space_policy="mesh_query_bbox",
    )


def select_dataset(dataset: DynamicRigManifestDataset, limit: int, seed: int) -> Subset:
    count = len(dataset)
    if limit <= 0 or limit >= count:
        indices = np.arange(count, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(count, size=limit, replace=False)).astype(np.int64)
    return Subset(dataset, indices.tolist())


def make_loader(dataset: Subset, tokenizer: Any) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: dynamic_rig_collate(batch, pad_token=tokenizer.pad),
    )


@torch.no_grad()
def extract_split(
    loader: DataLoader,
    *,
    split: str,
    device: torch.device,
    extractor: TopologyLocalMotionEvidence,
    surface_samples: int,
    vertex_samples: int,
    query_tokens: int,
    seed: int,
) -> QueryArrays:
    started = time.perf_counter()
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    groups: list[tuple[int, int]] = []
    paths: list[str] = []
    confidences: list[float] = []
    motion_amounts: list[float] = []
    valid_fractions: list[float] = []
    spatial_stds: list[float] = []
    offset = 0

    for row_index, raw_batch in enumerate(loader):
        path = str(raw_batch["path"][0])
        batch = move_batch(raw_batch, device)
        generator = torch.Generator(device=device).manual_seed(seed + row_index)
        refs = sample_trackable_surface(
            batch["frame_vertices"][:, 0],
            batch["faces"],
            num_samples=surface_samples,
            vertex_samples=vertex_samples,
            query_tokens=query_tokens,
            vertex_counts=batch["vertex_count"],
            face_counts=batch["face_count"],
            generator=generator,
        )
        evidence = extractor(
            batch["frame_vertices"],
            batch["faces"],
            refs,
            vertex_counts=batch["vertex_count"],
            face_counts=batch["face_count"],
        )
        with np.load(path, allow_pickle=True) as raw:
            skin_np = np.asarray(raw["target_skin_weights"], dtype=np.float32)
        vertex_count = int(batch["vertex_count"][0])
        if skin_np.shape[0] != vertex_count:
            raise ValueError(
                f"{path} skin vertices {skin_np.shape[0]} != loader vertices {vertex_count}"
            )
        skin = torch.from_numpy(skin_np).to(device=device)[None]
        targets = query_aligned_skin_boundary_targets(
            skin,
            batch["faces"],
            refs,
            vertex_counts=batch["vertex_count"],
            face_counts=batch["face_count"],
        )
        valid = targets.valid_mask[0]
        count = int(valid.sum())
        if count == 0:
            raise ValueError(f"{path} has no query-aligned skin-boundary targets")
        confidence = float(evidence.confidence[0])
        row_features = torch.cat(
            (
                evidence.query_features[0],
                evidence.confidence[0].expand(query_tokens, 1),
            ),
            dim=-1,
        )
        features.append(row_features[valid].cpu().numpy().astype(np.float32, copy=False))
        labels.append(targets.values[0, valid].cpu().numpy().astype(np.float32, copy=False))
        weights.append(np.full((count,), confidence, dtype=np.float32))
        groups.append((offset, offset + count))
        offset += count
        paths.append(path)
        confidences.append(confidence)
        motion_amounts.append(float(evidence.example_motion_q90_rms[0]))
        valid_fractions.append(count / query_tokens)
        spatial_stds.append(float(evidence.query_features[0, valid].std(dim=0).mean()))

        if (row_index + 1) % 100 == 0 or row_index + 1 == len(loader):
            print(
                f"[{split}] rows={row_index + 1}/{len(loader)} queries={offset} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    return QueryArrays(
        features=np.concatenate(features, axis=0),
        labels=np.concatenate(labels, axis=0),
        weights=np.concatenate(weights, axis=0),
        groups=np.asarray(groups, dtype=np.int64),
        paths=np.asarray(paths, dtype=str),
        confidence=np.asarray(confidences, dtype=np.float32),
        motion_q90_rms=np.asarray(motion_amounts, dtype=np.float32),
        valid_fraction=np.asarray(valid_fractions, dtype=np.float32),
        spatial_std=np.asarray(spatial_stds, dtype=np.float32),
        elapsed_seconds=time.perf_counter() - started,
    )


def fit_model(
    arrays: QueryArrays,
    *,
    device: torch.device,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[QueryEvidenceMLP, np.ndarray, np.ndarray, list[dict[str, float]]]:
    torch.manual_seed(seed)
    mean = arrays.features.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(
        arrays.features.std(axis=0, dtype=np.float64).astype(np.float32),
        1.0e-6,
    )
    x = torch.from_numpy((arrays.features - mean) / std).to(device)
    y = torch.from_numpy(arrays.labels).to(device)
    weights = torch.from_numpy(arrays.weights).to(device)
    model = QueryEvidenceMLP(x.shape[1], hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        permutation = torch.randperm(x.shape[0], device=device)
        total = 0.0
        total_weight = 0.0
        for start in range(0, x.shape[0], batch_size):
            indices = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[indices])
            per_value = nn.functional.binary_cross_entropy_with_logits(
                logits,
                y[indices],
                reduction="none",
            ).mean(dim=-1)
            item_weights = weights[indices]
            loss = (per_value * item_weights).sum() / item_weights.sum().clamp_min(1.0)
            loss.backward()
            optimizer.step()
            total += float((per_value.detach() * item_weights).sum())
            total_weight += float(item_weights.sum())
        record = {"epoch": epoch + 1, "loss": total / max(total_weight, 1.0)}
        history.append(record)
        print(json.dumps(record), flush=True)
    return model, mean, std, history


@torch.no_grad()
def predict(
    model: QueryEvidenceMLP,
    features: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    rows: list[np.ndarray] = []
    for start in range(0, features.shape[0], batch_size):
        normalized = (features[start : start + batch_size] - mean) / std
        values = torch.from_numpy(normalized.astype(np.float32, copy=False)).to(device)
        rows.append(torch.sigmoid(model(values)).cpu().numpy())
    return np.concatenate(rows, axis=0).astype(np.float32, copy=False)


def metric_bundle(
    arrays: QueryArrays,
    correct: np.ndarray,
    corrupted: np.ndarray,
) -> dict[str, Any]:
    names = ("incident_boundary_mean", "incident_boundary_max")
    result: dict[str, Any] = {}
    for index, name in enumerate(names):
        labels = arrays.labels[:, index]
        result[name] = {
            "mlp_correct": _score_metrics(labels, correct[:, index]),
            "mlp_corrupted": _score_metrics(labels, corrupted[:, index]),
            "raw_rms_mean": _score_metrics(labels, arrays.features[:, 0]),
            "raw_rms_max": _score_metrics(labels, arrays.features[:, 4]),
            "mlp_correct_per_asset": _per_asset_metrics(labels, correct[:, index], arrays.groups),
            "mlp_corrupted_per_asset": _per_asset_metrics(
                labels,
                corrupted[:, index],
                arrays.groups,
            ),
        }
    return result


def split_summary(arrays: QueryArrays) -> dict[str, Any]:
    return {
        "assets": int(arrays.groups.shape[0]),
        "queries": int(arrays.features.shape[0]),
        "feature_dim": int(arrays.features.shape[1]),
        "valid_fraction_mean": float(arrays.valid_fraction.mean()),
        "valid_fraction_min": float(arrays.valid_fraction.min()),
        "confidence_mean": float(arrays.confidence.mean()),
        "confidence_lt_0.1": int((arrays.confidence < 0.1).sum()),
        "spatial_feature_std_mean": float(arrays.spatial_std.mean()),
        "spatial_feature_std_median": float(np.median(arrays.spatial_std)),
        "boundary_mean": arrays.labels.mean(axis=0).tolist(),
        "elapsed_seconds": arrays.elapsed_seconds,
    }


def main() -> None:
    args = parse_args()
    if args.query_tokens != 1024:
        raise ValueError("current decoder contract requires exactly 1024 query tokens")
    device = torch.device("cuda:0")
    tokenizer = build_tokenizer(args.tokenizer_config)
    train_base = make_dataset(
        args.train_manifest,
        tokenizer,
        frame_count=args.frame_count,
        seed=args.seed,
    )
    valid_base = make_dataset(
        args.valid_manifest,
        tokenizer,
        frame_count=args.frame_count,
        seed=args.seed + 1,
    )
    train_loader = make_loader(select_dataset(train_base, args.train_limit, args.seed), tokenizer)
    valid_loader = make_loader(
        select_dataset(valid_base, args.valid_limit, args.seed + 1),
        tokenizer,
    )
    extractor = TopologyLocalMotionEvidence().to(device)
    torch.cuda.reset_peak_memory_stats()
    train = extract_split(
        train_loader,
        split="train",
        device=device,
        extractor=extractor,
        surface_samples=args.surface_samples,
        vertex_samples=args.vertex_samples,
        query_tokens=args.query_tokens,
        seed=args.seed,
    )
    valid = extract_split(
        valid_loader,
        split="valid",
        device=device,
        extractor=extractor,
        surface_samples=args.surface_samples,
        vertex_samples=args.vertex_samples,
        query_tokens=args.query_tokens,
        seed=args.seed + 1,
    )
    model, mean, std, history = fit_model(
        train,
        device=device,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    correct = predict(
        model,
        valid.features,
        mean=mean,
        std=std,
        device=device,
        batch_size=args.batch_size,
    )
    corrupted_features = _corrupt_within_groups(
        valid.features,
        valid.groups,
        args.seed + 17,
    )
    corrupted = predict(
        model,
        corrupted_features,
        mean=mean,
        std=std,
        device=device,
        batch_size=args.batch_size,
    )
    summary = {
        "status": "query_aligned_evidence_analysis_complete",
        "skeleton_checkpoint_loaded": False,
        "weights_changed": False,
        "query_contract": {
            "surface_samples": args.surface_samples,
            "vertex_samples": args.vertex_samples,
            "query_tokens": args.query_tokens,
            "feature_order": [
                "incident_mean_rms",
                "incident_mean_std",
                "incident_mean_max_abs",
                "incident_mean_active_ratio",
                "incident_max_rms",
                "incident_max_std",
                "incident_max_max_abs",
                "incident_max_active_ratio",
                "example_confidence",
            ],
        },
        "train": split_summary(train),
        "valid": split_summary(valid),
        "training_history": history,
        "metrics": metric_bundle(valid, correct, corrupted),
        "runtime": {
            "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
            "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        args.output_dir / "valid_query_features.npz",
        features=valid.features,
        labels=valid.labels,
        groups=valid.groups,
        paths=valid.paths,
        confidence=valid.confidence,
        motion_q90_rms=valid.motion_q90_rms,
    )
    torch.save(
        {
            "model": model.state_dict(),
            "feature_mean": mean,
            "feature_std": std,
            "args": vars(args),
        },
        args.output_dir / "diagnostic_query_classifier.pt",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    del model, extractor
    gc.collect()


if __name__ == "__main__":
    main()
