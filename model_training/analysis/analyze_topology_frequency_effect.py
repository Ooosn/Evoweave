#!/usr/bin/env python3
"""Relate Puppeteer evaluation quality to target topology frequency in training."""

from __future__ import annotations

import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any

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


def _load_training_paths(manifest: Path, max_joints: int) -> list[Path]:
    paths: list[Path] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if _target_joint_count(row) > max_joints:
            continue
        value = row.get("path") or row.get("npz_path") or row.get("file")
        if value is None:
            raise ValueError("manifest row has no path-like field")
        paths.append(_resolve_path(manifest, str(value)))
    return paths


def _load_parent_signature(path: Path) -> tuple[int, ...]:
    with np.load(path, allow_pickle=False) as raw:
        parents = np.asarray(raw["target_parents"], dtype=np.int64).reshape(-1)
    return tuple(int(value) for value in parents.tolist())


def _parents_from_token_ids(ids: list[int]) -> tuple[int, ...]:
    payload = list(int(value) for value in ids)
    if payload and payload[0] == 0:
        payload = payload[1:]
    if 1 in payload:
        payload = payload[: payload.index(1)]
    if len(payload) % 4 != 0:
        raise ValueError(
            f"Puppeteer token payload length {len(payload)} is not divisible by four"
        )
    parents: list[int] = []
    for position in range(3, len(payload), 4):
        raw_parent = payload[position] - 3
        parents.append(-1 if raw_parent == 0 else raw_parent - 1)
    return tuple(parents)


def _nearest_match(
    signature: tuple[int, ...],
    counter: collections.Counter[tuple[int, ...]],
) -> tuple[float, int]:
    if not counter:
        return float("nan"), 0
    best_match = -1.0
    best_frequency = 0
    for candidate, frequency in counter.items():
        match = sum(
            left == right for left, right in zip(signature, candidate)
        ) / len(signature)
        if match > best_match or (
            match == best_match and int(frequency) > best_frequency
        ):
            best_match = float(match)
            best_frequency = int(frequency)
    return best_match, best_frequency


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "target_topology_frequency": _summary(
            [float(row["target_topology_frequency"]) for row in rows]
        ),
        "coordinate_nll": _summary(
            [float(row["coordinate_nll"]) for row in rows]
        ),
        "coordinate_accuracy": _summary(
            [float(row["coordinate_accuracy"]) for row in rows]
        ),
        "joint_count_abs_error": _summary(
            [float(row["joint_count_abs_error"]) for row in rows]
        ),
        "j2j": _summary([float(row["j2j"]) for row in rows]),
        "topology_f1": _summary(
            [float(row["topology_f1"]) for row in rows]
        ),
        "prediction_exact_training_topology_rate": float(
            np.mean(
                [
                    int(row["predicted_topology_frequency"]) > 0
                    for row in rows
                ]
            )
        )
        if rows
        else float("nan"),
    }


def _frequency_bin(frequency: int) -> str:
    if frequency <= 0:
        return "unseen"
    if frequency == 1:
        return "singleton"
    if frequency <= 9:
        return "2_to_9"
    if frequency <= 99:
        return "10_to_99"
    return "100_plus"


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _correlation(left: list[float], right: list[float]) -> dict[str, float]:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size < 2 or x.std() == 0.0 or y.std() == 0.0:
        return {"pearson": float("nan"), "spearman": float("nan")}
    return {
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "spearman": float(np.corrcoef(_rank(x), _rank(y))[0, 1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--teacher-report", type=Path, required=True)
    parser.add_argument("--generation-report", type=Path, required=True)
    parser.add_argument("--max-joints", type=int, default=101)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train_manifest = args.train_manifest.resolve()
    train_paths = _load_training_paths(train_manifest, args.max_joints)
    topology_by_count: dict[
        int, collections.Counter[tuple[int, ...]]
    ] = collections.defaultdict(collections.Counter)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, signature in enumerate(
            executor.map(_load_parent_signature, train_paths),
            start=1,
        ):
            topology_by_count[len(signature)][signature] += 1
            if index % 3000 == 0 or index == len(train_paths):
                print(f"loaded training topology {index}/{len(train_paths)}", flush=True)

    teacher_rows = json.loads(
        args.teacher_report.read_text(encoding="utf-8")
    )["rows"]
    generation_rows = json.loads(
        args.generation_report.read_text(encoding="utf-8")
    )["rows"]
    if len(teacher_rows) != len(generation_rows):
        raise ValueError(
            f"teacher/generation row counts differ: "
            f"{len(teacher_rows)} vs {len(generation_rows)}"
        )

    rows: list[dict[str, Any]] = []
    for teacher, generation in zip(teacher_rows, generation_rows):
        if teacher["path"] != generation["path"]:
            raise ValueError(
                f"teacher/generation paths differ: "
                f"{teacher['path']} vs {generation['path']}"
            )
        dynamic = generation["dynamic_puppeteer"]
        metrics = dynamic["metrics"]
        target_signature = _parents_from_token_ids(generation["target_ids"])
        predicted_signature = _parents_from_token_ids(dynamic["generated_ids"])
        target_counter = topology_by_count[len(target_signature)]
        predicted_counter = topology_by_count[len(predicted_signature)]
        predicted_nearest_match, predicted_nearest_frequency = _nearest_match(
            predicted_signature,
            predicted_counter,
        )
        coordinate_positions = [
            position
            for position in teacher["forced_positions"]
            if position["role"] in {"x", "y", "z"}
        ]
        rows.append(
            {
                "path": teacher["path"],
                "eval_stratum": teacher.get("eval_stratum"),
                "target_joint_count": len(target_signature),
                "predicted_joint_count": len(predicted_signature),
                "target_topology_frequency": int(
                    target_counter[target_signature]
                ),
                "target_count_rows": int(sum(target_counter.values())),
                "target_count_unique_topologies": int(len(target_counter)),
                "predicted_topology_frequency": int(
                    predicted_counter[predicted_signature]
                ),
                "predicted_nearest_training_match": float(
                    predicted_nearest_match
                ),
                "predicted_nearest_training_frequency": int(
                    predicted_nearest_frequency
                ),
                "coordinate_nll": float(
                    np.mean(
                        [
                            position["target_nll"]
                            for position in coordinate_positions
                        ]
                    )
                ),
                "coordinate_accuracy": float(
                    np.mean(
                        [
                            position["correct"]
                            for position in coordinate_positions
                        ]
                    )
                ),
                "joint_count_abs_error": int(
                    metrics["joint_count_abs_error"]
                ),
                "j2j": float(metrics["official"]["j2j"]),
                "topology_f1": float(metrics["topology"]["edge_f1"]),
            }
        )

    by_frequency: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    by_stratum: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_frequency[_frequency_bin(row["target_topology_frequency"])].append(
            row
        )
        by_stratum[str(row.get("eval_stratum"))].append(row)

    log_frequency = [
        float(np.log1p(row["target_topology_frequency"])) for row in rows
    ]
    report = {
        "train_manifest": str(train_manifest),
        "teacher_report": str(args.teacher_report.resolve()),
        "generation_report": str(args.generation_report.resolve()),
        "train_rows": len(train_paths),
        "by_frequency": {
            name: _aggregate(group) for name, group in sorted(by_frequency.items())
        },
        "by_stratum": {
            name: _aggregate(group) for name, group in sorted(by_stratum.items())
        },
        "log_frequency_correlations": {
            metric: _correlation(
                log_frequency,
                [float(row[metric]) for row in rows],
            )
            for metric in (
                "coordinate_nll",
                "coordinate_accuracy",
                "joint_count_abs_error",
                "j2j",
                "topology_f1",
            )
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
