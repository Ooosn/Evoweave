#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MAIN_METRIC = "target_joint_mesh_bbox_diag_consistency"
ACTIVE_METRIC = "active_joint_mesh_bbox_diag_consistency"


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("status") == "ok"]
    for row in rows:
        for key in (MAIN_METRIC, ACTIVE_METRIC):
            row[key] = float(row[key])
    return rows


def evenly_sample(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return rows
    ids = np.linspace(0, len(rows) - 1, limit).round().astype(np.int64)
    return [rows[int(i)] for i in ids]


def frame0(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    if out.ndim == 3:
        return out[0]
    return out


def sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    ids = np.linspace(0, points.shape[0] - 1, max_points).round().astype(np.int64)
    return points[ids]


def finite_bbox(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[-1] != 3 or pts.shape[0] == 0:
        return None
    finite = pts[np.isfinite(pts).all(axis=1)]
    if finite.shape[0] == 0:
        return None
    return finite.min(axis=0), finite.max(axis=0)


def best_dims(vertices: np.ndarray) -> tuple[int, int, str]:
    bbox = finite_bbox(vertices)
    if bbox is None:
        return 0, 1, "XY"
    lo, hi = bbox
    span = hi - lo
    candidates = [((0, 1), "XY"), ((0, 2), "XZ"), ((1, 2), "YZ")]
    dims, name = max(candidates, key=lambda item: float(span[item[0][0]] * span[item[0][1]]))
    return int(dims[0]), int(dims[1]), name


def draw_bbox(ax, points: np.ndarray, dims: tuple[int, int], color: str, lw: float, alpha: float) -> None:
    bbox = finite_bbox(points)
    if bbox is None:
        return
    lo, hi = bbox
    x, y = dims
    xs = [lo[x], hi[x], hi[x], lo[x], lo[x]]
    ys = [lo[y], lo[y], hi[y], hi[y], lo[y]]
    ax.plot(xs, ys, color=color, lw=lw, alpha=alpha)


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        vertices = frame0(np.asarray(data["frame_vertices_rootspace"], dtype=np.float32))
        joints = frame0(np.asarray(data["target_joints_rootspace"], dtype=np.float32))
        parents = np.asarray(data["target_parents"], dtype=np.int64).reshape(-1)
    return vertices, joints, parents


def plot_overlay(
    ax,
    vertices: np.ndarray,
    joints: np.ndarray,
    parents: np.ndarray,
    dims: tuple[int, int],
    *,
    max_points: int,
) -> None:
    x, y = dims
    pts = sample_points(vertices, max_points)
    ax.scatter(pts[:, x], pts[:, y], s=1, c="#9ca3af", alpha=0.16, linewidths=0)
    draw_bbox(ax, vertices, dims, "#111827", 0.75, 0.55)
    draw_bbox(ax, joints, dims, "#dc2626", 1.0, 0.85)
    for j, parent in enumerate(parents.tolist()):
        if int(parent) >= 0:
            ax.plot(
                [joints[int(parent), x], joints[j, x]],
                [joints[int(parent), y], joints[j, y]],
                c="#2563eb",
                lw=0.8,
                alpha=0.9,
            )
    ax.scatter(joints[:, x], joints[:, y], s=9, c="#f97316", edgecolors="#111827", linewidths=0.25, zorder=3)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.14)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_contact_sheet(rows: list[dict[str, Any]], out_path: Path, title: str, *, max_points: int) -> None:
    cols = 4
    rows_n = int(math.ceil(len(rows) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.0 * cols, 3.6 * rows_n), dpi=160)
    axes = np.asarray(axes).reshape(-1)
    for ax, row in zip(axes, rows):
        try:
            vertices, joints, parents = load_npz(Path(str(row["path"])))
            x, y, plane = best_dims(vertices)
            plot_overlay(ax, vertices, joints, parents, (x, y), max_points=max_points)
            ax.set_title(
                f"{row['asset_id'][:12]} {plane}\nC={row[MAIN_METRIC]:.3f}",
                fontsize=7,
            )
        except Exception as exc:  # noqa: BLE001
            ax.set_title(f"{row.get('asset_id','?')}\n{exc!r}", fontsize=6)
            ax.axis("off")
    for ax in axes[len(rows) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_triview(row: dict[str, Any], out_path: Path, *, max_points: int) -> None:
    vertices, joints, parents = load_npz(Path(str(row["path"])))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=160)
    for ax, dims, plane in zip(axes, [(0, 1), (0, 2), (1, 2)], ["XY", "XZ", "YZ"]):
        plot_overlay(ax, vertices, joints, parents, dims, max_points=max_points)
        ax.set_title(plane, fontsize=9)
    fig.suptitle(
        f"{row['asset_id']} split={row.get('split','')} consistency={row[MAIN_METRIC]:.6g}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_bucket_csv(buckets: dict[str, list[dict[str, Any]]], out_path: Path) -> None:
    fields = ["bucket", "asset_id", "split", "path", MAIN_METRIC, ACTIVE_METRIC]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for bucket, rows in buckets.items():
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-per-bucket", type=int, default=24)
    parser.add_argument("--detail-count", type=int, default=12)
    parser.add_argument("--max-points", type=int, default=7000)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(load_rows(args.metrics_csv), key=lambda row: row[MAIN_METRIC])

    bucket_defs = [
        ("lt_0p1", lambda r: r[MAIN_METRIC] < 0.1),
        ("0p1_0p2", lambda r: 0.1 <= r[MAIN_METRIC] < 0.2),
        ("0p2_0p3", lambda r: 0.2 <= r[MAIN_METRIC] < 0.3),
        ("0p3_0p4", lambda r: 0.3 <= r[MAIN_METRIC] < 0.4),
        ("0p4_0p5", lambda r: 0.4 <= r[MAIN_METRIC] < 0.5),
    ]
    buckets: dict[str, list[dict[str, Any]]] = {}
    for name, predicate in bucket_defs:
        selected = [row for row in rows if predicate(row)]
        selected = sorted(selected, key=lambda row: row[MAIN_METRIC])
        if name.startswith("gt_"):
            selected = list(reversed(selected))
        buckets[name] = selected

    write_bucket_csv(buckets, args.out_dir / "bbox_consistency_review_buckets.csv")

    for name, selected in buckets.items():
        sample = evenly_sample(selected, int(args.sample_per_bucket))
        if not sample:
            continue
        title = f"{name}: showing {len(sample)} / {len(selected)}"
        plot_contact_sheet(
            sample,
            args.out_dir / f"{name}_contact_sheet.png",
            title,
            max_points=int(args.max_points),
        )

    detail_dir = args.out_dir / "details"
    detail_dir.mkdir(exist_ok=True)
    detail_rows = rows[: int(args.detail_count)] + list(reversed(rows[-int(args.detail_count) :]))
    seen: set[str] = set()
    for row in detail_rows:
        asset_id = str(row["asset_id"])
        if asset_id in seen:
            continue
        seen.add(asset_id)
        plot_triview(
            row,
            detail_dir / f"{asset_id}_triview.png",
            max_points=int(args.max_points),
        )

    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
