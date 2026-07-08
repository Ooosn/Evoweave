#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def decode_puppeteer_ids(
    ids: list[int],
    *,
    n_discrete_size: int = 128,
    target_coord_scale: float = 0.25,
    low: float = -0.5,
    high: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    raw_ids = [int(x) for x in ids]
    if raw_ids and raw_ids[0] == 0:
        raw_ids = raw_ids[1:]
    if 1 in raw_ids:
        raw_ids = raw_ids[: raw_ids.index(1)]
    raw_ids = [x for x in raw_ids if x not in {0, 1, 2}]
    usable = (len(raw_ids) // 4) * 4
    raw = np.asarray(raw_ids[:usable], dtype=np.int64).reshape(-1, 4) - 3
    coords = raw[:, :3].astype(np.float32) / float(n_discrete_size)
    coords = coords * float(high - low) + float(low)
    coords = coords / max(float(target_coord_scale), 1.0e-12)

    parents: list[int] = []
    for child, value in enumerate(raw[:, 3].tolist()):
        parent_bin = int(value)
        if child == 0:
            parents.append(-1)
        elif 0 < parent_bin <= child:
            parents.append(parent_bin - 1)
        else:
            parents.append(-2)
    return coords.astype(np.float32), np.asarray(parents, dtype=np.int64)


def edges(parents: np.ndarray) -> list[tuple[int, int]]:
    return [(int(parent), child) for child, parent in enumerate(parents.tolist()) if 0 <= int(parent) < child]


def subsample(points: np.ndarray, count: int, *, seed: int = 7) -> np.ndarray:
    if points.shape[0] <= count:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(points.shape[0], size=count, replace=False)]


def set_equal_axes(ax: plt.Axes, points: np.ndarray) -> None:
    finite = points[np.isfinite(points).all(axis=1)]
    if finite.size == 0:
        return
    lo = finite.min(axis=0)
    hi = finite.max(axis=0)
    center = (lo + hi) * 0.5
    radius = max(float((hi - lo).max() * 0.5), 1.0e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_skeleton(ax: plt.Axes, joints: np.ndarray, parents: np.ndarray, *, color: str, label: str, lw: float, alpha: float) -> None:
    for parent, child in edges(parents):
        ax.plot(
            [joints[parent, 0], joints[child, 0]],
            [joints[parent, 1], joints[child, 1]],
            [joints[parent, 2], joints[child, 2]],
            color=color,
            lw=lw,
            alpha=alpha,
        )
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], s=16, color=color, alpha=alpha, label=label, depthshade=False)
    if joints.shape[0] > 0:
        ax.scatter(
            [joints[0, 0]],
            [joints[0, 1]],
            [joints[0, 2]],
            s=64,
            color=color,
            edgecolors="black",
            depthshade=False,
        )


def row_metric(row: dict, key: str) -> float | int:
    metrics = row["dynamic_puppeteer"]["metrics"]
    if key == "f1":
        return float(metrics["topology"]["edge_f1"])
    return metrics[key]


def make_plot(row: dict, output_dir: Path, rank: int) -> Path:
    npz_path = Path(row["path"])
    raw = np.load(npz_path, allow_pickle=True)
    query_frame = int(row.get("query_frame", row.get("selected_frames", [0])[0]))
    center = np.asarray(row["query_center"], dtype=np.float32)
    scale = float(row["query_scale"])

    mesh = (np.asarray(raw["frame_vertices_rootspace"], dtype=np.float32)[query_frame] - center) / scale
    gt_joints = (np.asarray(raw["target_joints_rootspace"], dtype=np.float32)[query_frame] - center) / scale
    gt_parents = np.asarray(raw["target_parents"], dtype=np.int64).reshape(-1)
    pred_joints, pred_parents = decode_puppeteer_ids(row["dynamic_puppeteer"]["generated_ids"])

    mesh_view = subsample(mesh, 2500)
    all_points = np.concatenate([mesh_view, gt_joints, pred_joints], axis=0)

    target_count = int(row_metric(row, "target_joint_count"))
    pred_count = int(row_metric(row, "pred_joint_count"))
    f1 = float(row_metric(row, "f1"))
    fig = plt.figure(figsize=(15, 5), dpi=170)
    fig.suptitle(
        f"idx={row['index']} target={target_count} pred={pred_count} F1={f1:.3f} {npz_path.name}",
        fontsize=10,
    )

    panels = [
        ("GT", "gt"),
        ("Prediction", "pred"),
        ("Overlay", "overlay"),
    ]
    for col, (title, mode) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, col, projection="3d")
        ax.scatter(mesh_view[:, 0], mesh_view[:, 1], mesh_view[:, 2], s=0.6, c="#b9c0c8", alpha=0.16, depthshade=False)
        if mode in {"gt", "overlay"}:
            plot_skeleton(ax, gt_joints, gt_parents, color="#1b9e77", label="gt", lw=2.0, alpha=0.95)
        if mode in {"pred", "overlay"}:
            plot_skeleton(ax, pred_joints, pred_parents, color="#d95f02", label="pred", lw=1.2, alpha=0.72)
        ax.set_title(title, fontsize=9)
        ax.view_init(elev=18, azim=-65)
        set_equal_axes(ax, all_points)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.tick_params(labelsize=6)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = output_dir / f"rank{rank:02d}_idx{row['index']:02d}_t{target_count}_p{pred_count}_f1{f1:.3f}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def choose_rows(rows: list[dict], limit: int) -> list[dict]:
    ordered = sorted(rows, key=lambda row: (row_metric(row, "f1"), -abs(float(row_metric(row, "joint_count_error")))))
    chosen: list[dict] = []
    seen: set[int] = set()
    for row in ordered:
        if int(row["index"]) in seen:
            continue
        chosen.append(row)
        seen.add(int(row["index"]))
        if len(chosen) >= limit:
            break
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    payload = json.loads(args.eval_json.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = choose_rows(payload["rows"], args.limit)
    outputs = [make_plot(row, args.output_dir, rank) for rank, row in enumerate(rows)]
    print(json.dumps({"count": len(outputs), "outputs": [str(path) for path in outputs]}, indent=2))


if __name__ == "__main__":
    main()
