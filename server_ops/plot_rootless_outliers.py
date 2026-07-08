#!/usr/bin/env python3
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_rows(metrics_csv: Path):
    with metrics_csv.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sample_points(points, max_points=6000):
    if len(points) <= max_points:
        return points
    idx = np.linspace(0, len(points) - 1, max_points).round().astype(np.int64)
    return points[idx]


def frame0(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3:
        return arr[0]
    return arr


def plot_one(ax, verts, joints, parents, dims, title):
    x, y = dims
    verts_s = sample_points(verts)
    ax.scatter(verts_s[:, x], verts_s[:, y], s=1, c="#9ca3af", alpha=0.18, linewidths=0)
    for j, p in enumerate(parents):
        if p >= 0:
            ax.plot([joints[p, x], joints[j, x]], [joints[p, y], joints[j, y]], c="#2563eb", lw=1.0, alpha=0.9)
    ax.scatter(joints[:, x], joints[:, y], s=14, c="#f97316", edgecolors="#111827", linewidths=0.35, zorder=3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.2)


def main() -> int:
    metrics_csv = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(read_rows(metrics_csv), key=lambda r: float(r["active_joint_mesh_bbox_diag_consistency"]))[:n]

    for row in rows:
        path = Path(row["path"])
        with np.load(path, allow_pickle=True) as z:
            verts = frame0(z["frame_vertices_rootspace"]).astype(np.float32)
            joints = frame0(z["target_joints_rootspace"]).astype(np.float32)
            parents = z["target_parents"].astype(int)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=160)
        title = f"{row['asset_id']} consistency={float(row['active_joint_mesh_bbox_diag_consistency']):.3f}"
        plot_one(axes[0], verts, joints, parents, (0, 1), "XY")
        plot_one(axes[1], verts, joints, parents, (0, 2), "XZ")
        plot_one(axes[2], verts, joints, parents, (1, 2), "YZ")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out_dir / f"{row['asset_id']}_rootless_overlay.png")
        plt.close(fig)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
