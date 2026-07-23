#!/usr/bin/env python3
"""Validate local relative-motion evidence against held-out skin relations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
RIGWEAVE_SRC = REPO_ROOT / "model_training" / "rigweave" / "src"
if str(RIGWEAVE_SRC) not in sys.path:
    sys.path.insert(0, str(RIGWEAVE_SRC))

from rigweave.dynamic_rig.data import _select_query_sequence  # noqa: E402
from rigweave.dynamic_rig.motion_evidence import (  # noqa: E402
    FEATURE_NAMES,
    LocalMotionEvidence,
    extract_local_motion_evidence,
    unique_mesh_edges,
)


DEFAULT_ROOT = Path(
    "/ssdwork/liuhaohan/evorig/evoweave_rebuild_rootless_v3_20260706/"
    "quality_distributions/rootless_bbox_consistency/final_manifests"
)


@dataclass(frozen=True)
class SplitArrays:
    features: np.ndarray
    labels: np.ndarray
    observability: np.ndarray
    groups: np.ndarray
    paths: np.ndarray
    query_frames: np.ndarray
    selected_frame_rows: np.ndarray
    source_edge_counts: np.ndarray
    sampled_edge_counts: np.ndarray
    dropped_degenerate_edges: np.ndarray
    example_motion_amounts: np.ndarray
    elapsed_seconds: float


class EvidenceMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_ROOT / "train_manifest.jsonl")
    parser.add_argument("--valid-manifest", type=Path, default=DEFAULT_ROOT / "valid_manifest.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--valid-limit", type=int, default=0)
    parser.add_argument("--max-edges-per-asset", type=int, default=1024)
    parser.add_argument("--frame-count", type=int, default=24)
    parser.add_argument("--query-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-features", action="store_true")
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--visual-count", type=int, default=6)
    parser.add_argument("--visual-max-edges", type=int, default=20000)
    return parser.parse_args()


def _read_manifest(path: Path, limit: int, seed: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        value = row.get("path")
        if not isinstance(value, str) or not value:
            raise ValueError(f"manifest row lacks path: {row}")
        if not Path(value).is_file():
            raise FileNotFoundError(value)
    if limit > 0 and limit < len(rows):
        rng = np.random.default_rng(int(seed))
        indices = np.sort(rng.choice(len(rows), size=int(limit), replace=False))
        rows = [rows[int(index)] for index in indices]
    return rows


def _path_seed(path: Path, seed: int, repeat: int) -> int:
    digest = hashlib.blake2b(digest_size=8)
    digest.update(str(path).encode("utf-8"))
    digest.update(int(seed).to_bytes(8, "little", signed=False))
    digest.update(int(repeat).to_bytes(4, "little", signed=False))
    return int.from_bytes(digest.digest(), "little") % (2**32)


def _select_edge_indices(
    edge_count: int,
    *,
    max_edges: int,
    seed: int,
) -> np.ndarray:
    if max_edges <= 0 or edge_count <= max_edges:
        return np.arange(edge_count, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(edge_count, size=int(max_edges), replace=False)).astype(np.int64)


def _extract_one(
    path: Path,
    *,
    manifest_index: int,
    repeat: int,
    frame_count: int,
    seed: int,
    max_edges: int,
) -> tuple[LocalMotionEvidence, np.ndarray, int]:
    with np.load(path, allow_pickle=True) as raw:
        required = (
            "frame_vertices_rootspace",
            "faces",
            "target_joints_rootspace",
            "target_skin_weights",
        )
        missing = [name for name in required if name not in raw.files]
        if missing:
            raise KeyError(f"{path} is missing evidence fields {missing}")
        frames = np.asarray(raw["frame_vertices_rootspace"], dtype=np.float32)
        faces = np.asarray(raw["faces"], dtype=np.int64)
        joints = np.asarray(raw["target_joints_rootspace"], dtype=np.float32)
        skin = np.asarray(raw["target_skin_weights"], dtype=np.float32)

    selected_frames, _, _, selected = _select_query_sequence(
        frames,
        joints,
        None,
        frame_count=int(frame_count),
        path=path,
        index=int(manifest_index),
        random_query=False,
        seed=int(seed + repeat * 1_000_003),
        motion_fps_ratio=0.7,
        motion_vertex_samples=512,
        motion_alignment_policy="none",
        input_space_policy="mesh_query_bbox",
    )
    source_edges = unique_mesh_edges(faces, vertex_count=int(selected_frames.shape[1]))
    selected_edge_indices = _select_edge_indices(
        int(source_edges.shape[0]),
        max_edges=max_edges,
        seed=_path_seed(path, seed, repeat),
    )
    evidence = extract_local_motion_evidence(
        selected_frames,
        faces,
        skin,
        edges=source_edges[selected_edge_indices],
        query_index=0,
    )
    return evidence, selected, int(source_edges.shape[0])


def _save_split_cache(path: Path, arrays: SplitArrays) -> None:
    np.savez(
        path,
        features=arrays.features,
        labels=arrays.labels,
        observability=arrays.observability,
        groups=arrays.groups,
        paths=arrays.paths,
        query_frames=arrays.query_frames,
        selected_frame_rows=arrays.selected_frame_rows,
        source_edge_counts=arrays.source_edge_counts,
        sampled_edge_counts=arrays.sampled_edge_counts,
        dropped_degenerate_edges=arrays.dropped_degenerate_edges,
        example_motion_amounts=arrays.example_motion_amounts,
        elapsed_seconds=np.asarray(arrays.elapsed_seconds, dtype=np.float64),
    )


def _load_split_cache(path: Path) -> SplitArrays:
    with np.load(path, allow_pickle=False) as raw:
        return SplitArrays(
            features=np.asarray(raw["features"], dtype=np.float32),
            labels=np.asarray(raw["labels"], dtype=np.float32),
            observability=np.asarray(raw["observability"], dtype=np.float32),
            groups=np.asarray(raw["groups"], dtype=np.int64),
            paths=np.asarray(raw["paths"]),
            query_frames=np.asarray(raw["query_frames"], dtype=np.int64),
            selected_frame_rows=np.asarray(raw["selected_frame_rows"], dtype=np.int64),
            source_edge_counts=np.asarray(raw["source_edge_counts"], dtype=np.int64),
            sampled_edge_counts=np.asarray(raw["sampled_edge_counts"], dtype=np.int64),
            dropped_degenerate_edges=np.asarray(raw["dropped_degenerate_edges"], dtype=np.int64),
            example_motion_amounts=np.asarray(raw["example_motion_amounts"], dtype=np.float32),
            elapsed_seconds=float(np.asarray(raw["elapsed_seconds"]).item()),
        )


def extract_split(
    rows: list[dict[str, Any]],
    *,
    split: str,
    frame_count: int,
    query_repeats: int,
    max_edges: int,
    seed: int,
) -> SplitArrays:
    started = time.perf_counter()
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    observability: list[np.ndarray] = []
    groups: list[tuple[int, int]] = []
    paths: list[str] = []
    query_frames: list[int] = []
    selected_frame_rows: list[np.ndarray] = []
    source_edge_counts: list[int] = []
    sampled_edge_counts: list[int] = []
    dropped_edges: list[int] = []
    example_motion_amounts: list[float] = []
    offset = 0

    for manifest_index, row in enumerate(rows):
        path = Path(row["path"])
        for repeat in range(int(query_repeats)):
            evidence, selected_frames, source_edge_count = _extract_one(
                path,
                manifest_index=manifest_index,
                repeat=repeat,
                frame_count=frame_count,
                seed=seed,
                max_edges=max_edges,
            )
            feature_values = evidence.features
            label_values = evidence.boundary
            observation_values = evidence.observability
            count = int(evidence.edges.shape[0])
            features.append(feature_values)
            labels.append(label_values)
            observability.append(observation_values)
            groups.append((offset, offset + count))
            offset += count
            paths.append(str(path))
            query_frames.append(int(selected_frames[0]))
            selected_frame_rows.append(np.asarray(selected_frames, dtype=np.int64))
            source_edge_counts.append(source_edge_count)
            sampled_edge_counts.append(count)
            dropped_edges.append(int(evidence.dropped_degenerate_edges))
            example_motion_amounts.append(float(np.quantile(evidence.observability, 0.90)))

        if (manifest_index + 1) % 100 == 0 or manifest_index + 1 == len(rows):
            elapsed = time.perf_counter() - started
            print(
                f"[{split}] rows={manifest_index + 1}/{len(rows)} "
                f"examples={len(groups)} edges={offset} elapsed={elapsed:.1f}s",
                flush=True,
            )

    return SplitArrays(
        features=np.concatenate(features, axis=0).astype(np.float32, copy=False),
        labels=np.concatenate(labels, axis=0).astype(np.float32, copy=False),
        observability=np.concatenate(observability, axis=0).astype(np.float32, copy=False),
        groups=np.asarray(groups, dtype=np.int64),
        paths=np.asarray(paths, dtype=str),
        query_frames=np.asarray(query_frames, dtype=np.int64),
        selected_frame_rows=np.stack(selected_frame_rows, axis=0),
        source_edge_counts=np.asarray(source_edge_counts, dtype=np.int64),
        sampled_edge_counts=np.asarray(sampled_edge_counts, dtype=np.int64),
        dropped_degenerate_edges=np.asarray(dropped_edges, dtype=np.int64),
        example_motion_amounts=np.asarray(example_motion_amounts, dtype=np.float32),
        elapsed_seconds=float(time.perf_counter() - started),
    )


def _fit_model(
    train: SplitArrays,
    *,
    device: torch.device,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[EvidenceMLP, np.ndarray, np.ndarray, list[dict[str, float]]]:
    torch.manual_seed(int(seed))
    mean = train.features.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train.features.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1.0e-6)
    x = torch.from_numpy((train.features - mean) / std).to(device)
    y = torch.from_numpy(train.labels).to(device)
    model = EvidenceMLP(x.shape[1], int(hidden_dim)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    # Unweighted BCE is a proper scoring rule for the soft histogram-intersection
    # targets. A positive-class weight would shift the optimum away from the
    # soft label and make the reported calibration errors uninterpretable.
    loss_fn = nn.BCEWithLogitsLoss()
    history: list[dict[str, float]] = []

    for epoch in range(int(epochs)):
        model.train()
        permutation = torch.randperm(x.shape[0], device=device)
        total_loss = 0.0
        total_count = 0
        for start in range(0, x.shape[0], int(batch_size)):
            indices = permutation[start : start + int(batch_size)]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[indices])
            loss = loss_fn(logits, y[indices])
            loss.backward()
            optimizer.step()
            count = int(indices.shape[0])
            total_loss += float(loss.detach()) * count
            total_count += count
        item = {
            "epoch": float(epoch + 1),
            "loss": float(total_loss / max(1, total_count)),
        }
        history.append(item)
        print(f"[fit] epoch={epoch + 1}/{epochs} loss={item['loss']:.6f}", flush=True)
    return model, mean, std, history


@torch.no_grad()
def _predict(
    model: EvidenceMLP,
    features: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    for start in range(0, features.shape[0], int(batch_size)):
        values = (features[start : start + int(batch_size)] - mean) / std
        tensor = torch.from_numpy(values.astype(np.float32, copy=False)).to(device)
        out.append(torch.sigmoid(model(tensor)).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32, copy=False)


def _corrupt_within_groups(values: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    corrupted = np.empty_like(values)
    for start, end in groups.tolist():
        count = int(end - start)
        permutation = rng.permutation(count)
        corrupted[start:end] = values[start:end][permutation]
    return corrupted


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _safe_binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float | int | None]:
    hard = labels >= float(threshold)
    positives = int(hard.sum())
    negatives = int((~hard).sum())
    result: dict[str, float | int] = {
        "threshold": float(threshold),
        "rows": int(labels.shape[0]),
        "positives": positives,
        "negatives": negatives,
    }
    if positives == 0 or negatives == 0:
        result["auroc"] = None
        result["auprc"] = None
    else:
        result["auroc"] = float(roc_auc_score(hard, scores))
        result["auprc"] = float(average_precision_score(hard, scores))
    return result


def _score_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    sample_count = min(int(labels.shape[0]), 500_000)
    if labels.shape[0] > sample_count:
        indices = np.linspace(0, labels.shape[0] - 1, sample_count, dtype=np.int64)
    else:
        indices = np.arange(labels.shape[0], dtype=np.int64)
    correlation = spearmanr(labels[indices], scores[indices]).statistic
    top_count = max(1, int(round(0.10 * scores.shape[0])))
    top_indices = np.argpartition(scores, -top_count)[-top_count:]
    return {
        "soft_mae": float(np.mean(np.abs(scores - labels))),
        "soft_mse": float(np.mean(np.square(scores - labels))),
        "spearman": _finite_or_none(float(correlation)),
        "label_mean": float(labels.mean()),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "top10_label_mean": float(labels[top_indices].mean()),
        "top10_lift": float(labels[top_indices].mean() / max(float(labels.mean()), 1.0e-12)),
        "binary": {
            str(threshold): _safe_binary_metrics(labels, scores, threshold)
            for threshold in (0.10, 0.25, 0.50)
        },
    }


def _metrics_by_example_motion(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    example_motion_amounts: np.ndarray,
) -> dict[str, Any]:
    if groups.shape[0] != example_motion_amounts.shape[0]:
        raise ValueError("one example motion amount is required for every edge group")
    ranked_examples = np.argsort(example_motion_amounts, kind="stable")
    strata = np.array_split(ranked_examples, 4)
    out: dict[str, Any] = {
        "definition": "per-example q90 edge-motion RMS, split into equal-count rank quartiles",
        "strata": {},
    }
    for index, example_ids in enumerate(strata):
        mask = np.zeros(labels.shape[0], dtype=bool)
        for example_id in example_ids.tolist():
            start, end = groups[int(example_id)]
            mask[int(start) : int(end)] = True
        amounts = example_motion_amounts[example_ids]
        out["strata"][str(index)] = {
            "examples": int(example_ids.shape[0]),
            "edges": int(mask.sum()),
            "example_motion_min": float(amounts.min()),
            "example_motion_mean": float(amounts.mean()),
            "example_motion_max": float(amounts.max()),
            "metrics": _score_metrics(labels[mask], scores[mask]),
        }
    return out


def _per_asset_metrics(labels: np.ndarray, scores: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    aucs: list[float] = []
    aps: list[float] = []
    correlations: list[float] = []
    for start, end in groups.tolist():
        local_labels = labels[start:end]
        local_scores = scores[start:end]
        hard = local_labels >= 0.25
        if bool(hard.any()) and bool((~hard).any()):
            aucs.append(float(roc_auc_score(hard, local_scores)))
            aps.append(float(average_precision_score(hard, local_scores)))
        if local_labels.shape[0] >= 3 and float(local_labels.std()) > 1.0e-8:
            correlation = float(spearmanr(local_labels, local_scores).statistic)
            if np.isfinite(correlation):
                correlations.append(correlation)

    def summary(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"count": 0}
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": int(array.shape[0]),
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "q25": float(np.quantile(array, 0.25)),
            "q75": float(np.quantile(array, 0.75)),
        }

    return {
        "auroc_at_boundary_0.25": summary(aucs),
        "auprc_at_boundary_0.25": summary(aps),
        "spearman": summary(correlations),
    }


def _render_distribution_plot(
    labels: np.ndarray,
    observability: np.ndarray,
    correct_scores: np.ndarray,
    corrupted_scores: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    low = labels <= 0.05
    high = labels >= 0.25
    bins = np.linspace(-8.0, 1.0, 80)
    axes[0].hist(np.log10(observability[low] + 1.0e-8), bins=bins, density=True, alpha=0.6, label="same segment")
    axes[0].hist(np.log10(observability[high] + 1.0e-8), bins=bins, density=True, alpha=0.6, label="boundary")
    axes[0].set_xlabel("log10 edge-motion RMS")
    axes[0].set_ylabel("density")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].scatter(labels[:: max(1, labels.shape[0] // 50_000)], correct_scores[:: max(1, labels.shape[0] // 50_000)], s=2, alpha=0.15, label="correct")
    axes[1].scatter(labels[:: max(1, labels.shape[0] // 50_000)], corrupted_scores[:: max(1, labels.shape[0] // 50_000)], s=2, alpha=0.15, label="corrupted")
    axes[1].set_xlabel("soft skin boundary label")
    axes[1].set_ylabel("predicted boundary evidence")
    axes[1].legend(markerscale=4)
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _render_examples(
    rows: list[dict[str, Any]],
    *,
    model: EvidenceMLP,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    output_dir: Path,
    frame_count: int,
    seed: int,
    count: int,
    max_edges: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if count <= 0:
        return
    indices = np.linspace(0, len(rows) - 1, min(count, len(rows)), dtype=np.int64)
    for visual_index, row_index in enumerate(indices.tolist()):
        path = Path(rows[int(row_index)]["path"])
        evidence, selected, _ = _extract_one(
            path,
            manifest_index=int(row_index),
            repeat=0,
            frame_count=frame_count,
            seed=seed,
            max_edges=max_edges,
        )
        with np.load(path, allow_pickle=True) as raw:
            query = np.asarray(raw["frame_vertices_rootspace"], dtype=np.float32)[int(selected[0])]
        lo = query.min(axis=0)
        hi = query.max(axis=0)
        scale = max(float(np.max(hi - lo)), 1.0e-8)
        query = (query - (lo + hi) * 0.5) / scale
        scores = _predict(
            model,
            evidence.features,
            mean=mean,
            std=std,
            device=device,
            batch_size=131072,
        )
        values = (evidence.boundary, evidence.observability, scores)
        titles = ("skin boundary", "motion RMS", "MLP evidence")
        segments = query[evidence.edges]
        fig = plt.figure(figsize=(15, 4.8))
        for panel, (title, edge_values) in enumerate(zip(titles, values), start=1):
            axis = fig.add_subplot(1, 3, panel, projection="3d")
            vmax = max(float(np.quantile(edge_values, 0.99)), 1.0e-8)
            colors = plt.get_cmap("viridis")(np.clip(edge_values / vmax, 0.0, 1.0))
            collection = Line3DCollection(segments, colors=colors, linewidths=0.6, alpha=0.85)
            axis.add_collection3d(collection)
            axis.scatter(query[:, 0], query[:, 1], query[:, 2], s=0.3, c="0.75", alpha=0.25)
            axis.set_xlim(float(query[:, 0].min()), float(query[:, 0].max()))
            axis.set_ylim(float(query[:, 1].min()), float(query[:, 1].max()))
            axis.set_zlim(float(query[:, 2].min()), float(query[:, 2].max()))
            axis.set_title(title)
            axis.set_axis_off()
        fig.suptitle(f"{path.name} query_frame={int(selected[0])}")
        fig.tight_layout()
        fig.savefig(output_dir / f"example_{visual_index:02d}_{path.stem}.png", dpi=180)
        plt.close(fig)


def _split_summary(arrays: SplitArrays) -> dict[str, Any]:
    return {
        "examples": int(arrays.groups.shape[0]),
        "edges": int(arrays.features.shape[0]),
        "feature_dim": int(arrays.features.shape[1]),
        "label_mean": float(arrays.labels.mean()),
        "label_q50": float(np.quantile(arrays.labels, 0.50)),
        "label_q90": float(np.quantile(arrays.labels, 0.90)),
        "label_q99": float(np.quantile(arrays.labels, 0.99)),
        "observability_mean": float(arrays.observability.mean()),
        "observability_q50": float(np.quantile(arrays.observability, 0.50)),
        "observability_q90": float(np.quantile(arrays.observability, 0.90)),
        "observability_q99": float(np.quantile(arrays.observability, 0.99)),
        "example_motion_q10": float(np.quantile(arrays.example_motion_amounts, 0.10)),
        "example_motion_q50": float(np.quantile(arrays.example_motion_amounts, 0.50)),
        "example_motion_q90": float(np.quantile(arrays.example_motion_amounts, 0.90)),
        "source_edges_total": int(arrays.source_edge_counts.sum()),
        "sampled_edges_total": int(arrays.sampled_edge_counts.sum()),
        "dropped_degenerate_edges": int(arrays.dropped_degenerate_edges.sum()),
        "elapsed_seconds": float(arrays.elapsed_seconds),
        "examples_per_second": float(arrays.groups.shape[0] / max(arrays.elapsed_seconds, 1.0e-12)),
    }


def _peak_rss_bytes() -> int | None:
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    except (ImportError, OSError, ValueError):
        return None


def main() -> int:
    total_started = time.perf_counter()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "visuals").mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    train_rows = _read_manifest(args.train_manifest, args.train_limit, args.seed)
    valid_rows = _read_manifest(args.valid_manifest, args.valid_limit, args.seed + 1)

    train_cache = args.output_dir / "train_features.npz"
    valid_cache = args.output_dir / "valid_features.npz"
    if args.reuse_cache:
        train = _load_split_cache(train_cache)
        valid = _load_split_cache(valid_cache)
    else:
        train = extract_split(
            train_rows,
            split="train",
            frame_count=args.frame_count,
            query_repeats=args.query_repeats,
            max_edges=args.max_edges_per_asset,
            seed=args.seed,
        )
        valid = extract_split(
            valid_rows,
            split="valid",
            frame_count=args.frame_count,
            query_repeats=args.query_repeats,
            max_edges=args.max_edges_per_asset,
            seed=args.seed + 1,
        )
        if args.cache_features:
            _save_split_cache(train_cache, train)
            _save_split_cache(valid_cache, valid)

    model, mean, std, history = _fit_model(
        train,
        device=device,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    correct_scores = _predict(
        model,
        valid.features,
        mean=mean,
        std=std,
        device=device,
        batch_size=args.batch_size,
    )
    zero_scores = _predict(
        model,
        np.zeros_like(valid.features),
        mean=mean,
        std=std,
        device=device,
        batch_size=args.batch_size,
    )
    corrupted_features = _corrupt_within_groups(valid.features, valid.groups, args.seed + 17)
    corrupted_scores = _predict(
        model,
        corrupted_features,
        mean=mean,
        std=std,
        device=device,
        batch_size=args.batch_size,
    )
    rms_scores = valid.observability

    summary = {
        "status": "evidence_analysis_complete",
        "skeleton_model": {
            "checkpoint_loaded": False,
            "weights_changed": False,
        },
        "diagnostic_edge_classifier": {
            "trained": True,
            "architecture": f"MLP({len(FEATURE_NAMES)}->{args.hidden_dim}->{args.hidden_dim}->1)",
        },
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "feature_names": list(FEATURE_NAMES),
        "train": _split_summary(train),
        "valid": _split_summary(valid),
        "training_history": history,
        "metrics": {
            "mlp_correct": _score_metrics(valid.labels, correct_scores),
            "mlp_zero_motion": _score_metrics(valid.labels, zero_scores),
            "mlp_corrupted_correspondence": _score_metrics(valid.labels, corrupted_scores),
            "rms_untrained": _score_metrics(valid.labels, rms_scores),
            "mlp_correct_by_example_motion": _metrics_by_example_motion(
                valid.labels,
                correct_scores,
                valid.groups,
                valid.example_motion_amounts,
            ),
            "mlp_correct_per_asset": _per_asset_metrics(valid.labels, correct_scores, valid.groups),
            "mlp_corrupted_per_asset": _per_asset_metrics(valid.labels, corrupted_scores, valid.groups),
        },
        "runtime_and_memory": {
            "total_elapsed_seconds_before_render": float(time.perf_counter() - total_started),
            "peak_process_rss_bytes": _peak_rss_bytes(),
            "peak_gpu_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {
            "model": model.state_dict(),
            "feature_mean": mean,
            "feature_std": std,
            "feature_names": FEATURE_NAMES,
            "args": summary["args"],
        },
        args.output_dir / "diagnostic_edge_classifier.pt",
    )
    _render_distribution_plot(
        valid.labels,
        valid.observability,
        correct_scores,
        corrupted_scores,
        args.output_dir / "evidence_distributions.png",
    )
    _render_examples(
        valid_rows,
        model=model,
        mean=mean,
        std=std,
        device=device,
        output_dir=args.output_dir / "visuals",
        frame_count=args.frame_count,
        seed=args.seed + 1,
        count=args.visual_count,
        max_edges=args.visual_max_edges,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
