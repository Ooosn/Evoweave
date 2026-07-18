#!/usr/bin/env python3
"""Measure query-normalized root, prefix geometry, and topology by skeleton length."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _target_joint_count(row: dict[str, Any]) -> int:
    candidates = [
        row.get("canonical_metrics", {}).get("target_joint_count"),
        row.get("final_bbox_consistency_screening", {})
        .get("metrics", {})
        .get("rootless_joint_count"),
        row.get("rootless_target_strict_metrics", {}).get("target_joint_count"),
    ]
    for value in candidates:
        if value is not None:
            return int(value)
    raise ValueError(f"manifest row has no target joint count: {row.get('path')}")


def _resolve_path(manifest: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest.parent / path
    return path.resolve()


def _summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot summarize an empty array")
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def _entropy(counter: collections.Counter[Any]) -> float:
    total = sum(counter.values())
    if total <= 0:
        raise ValueError("cannot compute entropy of an empty counter")
    probabilities = np.asarray(list(counter.values()), dtype=np.float64) / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def _counter_report(counter: collections.Counter[Any]) -> dict[str, Any]:
    total = sum(counter.values())
    top = counter.most_common(10)
    return {
        "entropy_bits": _entropy(counter),
        "occupied_bins": len(counter),
        "mode_share": float(top[0][1] / total),
        "top": [
            {
                "value": list(value) if isinstance(value, tuple) else int(value),
                "count": int(count),
                "share": float(count / total),
            }
            for value, count in top
        ],
    }


def _quantize(roots: np.ndarray) -> np.ndarray:
    scaled = np.asarray(roots, dtype=np.float64) * 0.25
    if bool(((scaled < -0.500001) | (scaled > 0.500001)).any()):
        raise ValueError(
            f"joint-0 target exceeds Puppeteer token range: "
            f"min={scaled.min():.6g} max={scaled.max():.6g}"
        )
    scaled = np.clip(scaled, -0.5, np.nextafter(0.5, -0.5))
    return np.floor((scaled + 0.5) * 128.0).astype(np.int64)


def _load_rows(manifest: Path, max_joints: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        count = _target_joint_count(row)
        if count > max_joints:
            continue
        value = row.get("path") or row.get("npz_path") or row.get("file")
        if value is None:
            raise ValueError("manifest JSON row has no path-like field")
        rows.append(
            {
                "path": _resolve_path(manifest, str(value)),
                "joint_count": count,
                "dataset_source": row.get("dataset_source"),
            }
        )
    return rows


def _sample_rows(
    rows: list[dict[str, Any]],
    predicate: Callable[[int], bool],
    *,
    limit: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    selected = [row for row in rows if predicate(int(row["joint_count"]))]
    if not selected:
        raise ValueError("joint-count group is empty")
    if limit > 0 and len(selected) > limit:
        indices = np.sort(rng.choice(len(selected), size=limit, replace=False))
        selected = [selected[int(index)] for index in indices]
    return selected


def _analyze_group(
    name: str,
    rows: list[dict[str, Any]],
    *,
    prefix_joint_count: int,
    shared_prefix_joint_count: int,
) -> dict[str, Any]:
    frame_roots: list[np.ndarray] = []
    asset_means: list[np.ndarray] = []
    within_asset_rms: list[float] = []
    shared_prefix_asset_means: list[np.ndarray] = []
    shared_prefix_within_asset_rms: list[float] = []
    frame_counts: list[int] = []
    source_counts: collections.Counter[str] = collections.Counter()
    topology_counter: collections.Counter[tuple[int, ...]] = collections.Counter()
    shared_prefix_topology_counter: collections.Counter[tuple[int, ...]] = (
        collections.Counter()
    )
    prefix_xyz_counters = [
        collections.Counter() for _ in range(prefix_joint_count)
    ]
    prefix_axis_counters = [
        [collections.Counter() for _axis in range(3)]
        for _joint in range(prefix_joint_count)
    ]

    for index, row in enumerate(rows):
        path = Path(row["path"])
        with np.load(path, allow_pickle=False) as raw:
            required = {
                "frame_vertices_rootspace",
                "target_joints_rootspace",
                "target_parents",
            }
            missing = required.difference(raw.files)
            if missing:
                raise KeyError(f"{path} is missing {sorted(missing)}")
            vertices = np.asarray(raw["frame_vertices_rootspace"], dtype=np.float32)
            joints = np.asarray(raw["target_joints_rootspace"], dtype=np.float32)
            parents = np.asarray(raw["target_parents"], dtype=np.int64).reshape(-1)

        if vertices.ndim != 3 or vertices.shape[-1] != 3:
            raise ValueError(f"{path} invalid frame vertices shape {vertices.shape}")
        if joints.ndim != 3 or joints.shape[0] != vertices.shape[0] or joints.shape[-1] != 3:
            raise ValueError(f"{path} invalid target joints shape {joints.shape}")
        if int(joints.shape[1]) != int(row["joint_count"]):
            raise ValueError(
                f"{path} manifest joints={row['joint_count']} NPZ joints={joints.shape[1]}"
            )
        if parents.shape != (joints.shape[1],):
            raise ValueError(
                f"{path} target parents shape {parents.shape} does not match "
                f"{joints.shape[1]} joints"
            )
        if int(joints.shape[1]) < shared_prefix_joint_count:
            raise ValueError(
                f"{path} has {joints.shape[1]} joints, fewer than the required "
                f"shared prefix {shared_prefix_joint_count}"
            )

        lo = vertices.min(axis=1)
        hi = vertices.max(axis=1)
        centers = (lo + hi) * 0.5
        scales = ((hi - lo) * 0.5).max(axis=1)
        if bool((~np.isfinite(scales) | (scales < 1.0e-8)).any()):
            raise ValueError(f"{path} has an invalid query-mesh bbox scale")
        normalized_joints = (joints - centers[:, None]) / scales[:, None, None]
        roots = normalized_joints[:, 0]
        if not bool(np.isfinite(roots).all()):
            raise ValueError(f"{path} produced non-finite normalized joint-0 coordinates")

        asset_mean = roots.mean(axis=0)
        shared_prefix = normalized_joints[:, :shared_prefix_joint_count]
        shared_prefix_mean = shared_prefix.mean(axis=0)
        frame_roots.append(roots.astype(np.float64))
        asset_means.append(asset_mean.astype(np.float64))
        within_asset_rms.append(
            float(np.sqrt(np.mean(np.square(roots - asset_mean[None]))))
        )
        shared_prefix_asset_means.append(shared_prefix_mean.astype(np.float64))
        shared_prefix_within_asset_rms.append(
            float(
                np.sqrt(
                    np.mean(
                        np.square(
                            shared_prefix - shared_prefix_mean[None]
                        )
                    )
                )
            )
        )
        topology_counter[tuple(int(value) for value in parents.tolist())] += 1
        shared_prefix_topology_counter[
            tuple(int(value) for value in parents[:shared_prefix_joint_count].tolist())
        ] += 1
        analyzed_prefix_count = min(prefix_joint_count, int(joints.shape[1]))
        quantized_prefix = _quantize(
            normalized_joints[:, :analyzed_prefix_count].reshape(-1, 3)
        ).reshape(normalized_joints.shape[0], analyzed_prefix_count, 3)
        for joint_index in range(analyzed_prefix_count):
            values = quantized_prefix[:, joint_index]
            prefix_xyz_counters[joint_index].update(map(tuple, values.tolist()))
            for axis in range(3):
                prefix_axis_counters[joint_index][axis].update(
                    values[:, axis].tolist()
                )
        frame_counts.append(int(roots.shape[0]))
        source_counts[str(row["dataset_source"])] += 1
        if (index + 1) % 50 == 0 or index + 1 == len(rows):
            print(f"{name}: {index + 1}/{len(rows)}", flush=True)

    roots = np.concatenate(frame_roots, axis=0)
    means = np.stack(asset_means, axis=0)
    shared_prefix_means = np.stack(shared_prefix_asset_means, axis=0)
    quantized = _quantize(roots)
    tuple_counter = collections.Counter(map(tuple, quantized.tolist()))
    axis_counters = [
        collections.Counter(quantized[:, axis].tolist()) for axis in range(3)
    ]
    global_asset_mean = means.mean(axis=0)
    between_asset_rms = np.sqrt(
        np.mean(np.square(means - global_asset_mean[None]), axis=1)
    )
    global_shared_prefix_mean = shared_prefix_means.mean(axis=0)
    shared_prefix_between_asset_rms = np.sqrt(
        np.mean(
            np.square(
                shared_prefix_means - global_shared_prefix_mean[None]
            ),
            axis=(1, 2),
        )
    )

    return {
        "asset_count": len(rows),
        "frame_count": int(roots.shape[0]),
        "joint_count": _summary(
            np.asarray([row["joint_count"] for row in rows], dtype=np.float64)
        ),
        "dataset_sources": dict(sorted(source_counts.items())),
        "frames_per_asset": _summary(np.asarray(frame_counts, dtype=np.float64)),
        "root_coordinate": {
            axis: _summary(roots[:, index])
            for index, axis in enumerate(("x", "y", "z"))
        },
        "root_radius": _summary(np.linalg.norm(roots, axis=1)),
        "outside_query_bbox_rate": float((np.abs(roots) > 1.0).any(axis=1).mean()),
        "within_asset_pose_rms": _summary(np.asarray(within_asset_rms)),
        "between_asset_mean_rms": _summary(between_asset_rms),
        "topology": {
            "full_tree": _counter_report(topology_counter),
            "shared_prefix": {
                "joint_count": int(shared_prefix_joint_count),
                **_counter_report(shared_prefix_topology_counter),
            },
        },
        "shared_prefix_geometry": {
            "joint_count": int(shared_prefix_joint_count),
            "within_asset_pose_rms": _summary(
                np.asarray(shared_prefix_within_asset_rms)
            ),
            "between_asset_mean_rms": _summary(
                shared_prefix_between_asset_rms
            ),
        },
        "quantized_axis": {
            axis: _counter_report(axis_counters[index])
            for index, axis in enumerate(("x", "y", "z"))
        },
        "quantized_xyz": _counter_report(tuple_counter),
        "prefix_joint_quantization": [
            {
                "joint_index": joint_index,
                "frame_occurrences": int(
                    sum(prefix_xyz_counters[joint_index].values())
                ),
                "xyz": _counter_report(prefix_xyz_counters[joint_index]),
                "axis": {
                    axis: _counter_report(
                        prefix_axis_counters[joint_index][axis_index]
                    )
                    for axis_index, axis in enumerate(("x", "y", "z"))
                },
            }
            for joint_index in range(prefix_joint_count)
            if prefix_xyz_counters[joint_index]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-joints", type=int, default=101)
    parser.add_argument("--max-assets-per-group", type=int, default=400)
    parser.add_argument("--prefix-joints", type=int, default=10)
    parser.add_argument("--shared-prefix-joints", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = _load_rows(args.manifest.resolve(), args.max_joints)
    rng = np.random.default_rng(args.seed)
    groups = {
        "low_le10": lambda count: count <= 10,
        "dominant_eq52": lambda count: count == 52,
        "common_20_to75_not52": lambda count: 20 <= count <= 75 and count != 52,
        "high_76_to101": lambda count: 76 <= count <= 101,
    }
    report: dict[str, Any] = {
        "manifest": str(args.manifest.resolve()),
        "max_joints": int(args.max_joints),
        "max_assets_per_group": int(args.max_assets_per_group),
        "prefix_joints": int(args.prefix_joints),
        "shared_prefix_joints": int(args.shared_prefix_joints),
        "seed": int(args.seed),
        "eligible_rows": len(rows),
        "groups": {},
    }
    for name, predicate in groups.items():
        selected = _sample_rows(
            rows,
            predicate,
            limit=args.max_assets_per_group,
            rng=rng,
        )
        report["groups"][name] = _analyze_group(
            name,
            selected,
            prefix_joint_count=args.prefix_joints,
            shared_prefix_joint_count=args.shared_prefix_joints,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
