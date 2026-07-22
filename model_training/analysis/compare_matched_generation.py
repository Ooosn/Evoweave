#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Result:
    label: str
    path: Path
    payload: dict[str, Any]
    block: str

    @property
    def rows(self) -> list[dict[str, Any]]:
        return self.payload["rows"]

    @property
    def summary(self) -> dict[str, Any]:
        return self.payload["summary"]


@dataclass(frozen=True)
class Stratum:
    name: str
    start: int
    end: int


def _parse_result(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("result must use LABEL=PATH")
    return label, Path(raw_path)


def _parse_stratum(value: str) -> Stratum:
    name, separator, raw_range = value.partition("=")
    start_text, colon, end_text = raw_range.partition(":")
    if not separator or not colon or not name:
        raise argparse.ArgumentTypeError("stratum must use NAME=START:END")
    start = int(start_text)
    end = int(end_text)
    if start < 0 or end <= start:
        raise argparse.ArgumentTypeError("stratum range must satisfy 0 <= START < END")
    return Stratum(name=name, start=start, end=end)


def _infer_block(payload: dict[str, Any]) -> str:
    rows = payload.get("rows", [])
    if not rows:
        raise ValueError("evaluation result has no rows")
    candidates = [
        name
        for name in ("dynamic", "dynamic_puppeteer", "stack_close")
        if name in rows[0]
    ]
    if len(candidates) != 1:
        raise ValueError(f"expected one generated-model block, found {candidates}")
    return candidates[0]


def _load_payload(path: Path) -> dict[str, Any]:
    if path.is_dir():
        summary_path = path / "summary.json"
        rows_path = path / "rows.jsonl"
        if not summary_path.is_file() or not rows_path.is_file():
            raise ValueError(
                f"evaluation directory must contain summary.json and rows.jsonl: {path}"
            )
        return {
            "summary": json.loads(summary_path.read_text(encoding="utf-8")),
            "rows": [
                json.loads(line)
                for line in rows_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ],
        }
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"evaluation result must be a JSON object: {path}")
    return payload


def _load_results(specs: list[tuple[str, Path]]) -> list[Result]:
    labels: set[str] = set()
    results: list[Result] = []
    for label, path in specs:
        if label in labels:
            raise ValueError(f"duplicate result label {label!r}")
        labels.add(label)
        payload = _load_payload(path)
        results.append(Result(label=label, path=path, payload=payload, block=_infer_block(payload)))
    if len(results) < 2:
        raise ValueError("matched comparison requires at least two results")
    return results


def _contract_value(row: dict[str, Any], key: str) -> Any:
    if key not in row:
        raise ValueError(f"row {row.get('index')} is missing matched-contract field {key!r}")
    return row[key]


def _validate_matched_contract(results: list[Result]) -> dict[str, Any]:
    reference = results[0]
    row_count = len(reference.rows)
    if row_count == 0:
        raise ValueError("matched comparison has no rows")

    fields = ("index", "path", "query_frame", "selected_frames", "target_joint_count")
    float_fields = ("query_center", "query_scale")
    for result in results[1:]:
        if len(result.rows) != row_count:
            raise ValueError(
                f"row-count mismatch: {reference.label}={row_count}, {result.label}={len(result.rows)}"
            )
        for row_index, (left, right) in enumerate(zip(reference.rows, result.rows, strict=True)):
            for field in fields:
                left_value = _contract_value(left, field)
                right_value = _contract_value(right, field)
                if left_value != right_value:
                    raise ValueError(
                        f"matched-contract mismatch at row {row_index}, field {field}: "
                        f"{reference.label}={left_value!r}, {result.label}={right_value!r}"
                    )
            for field in float_fields:
                left_value = np.asarray(_contract_value(left, field), dtype=np.float64)
                right_value = np.asarray(_contract_value(right, field), dtype=np.float64)
                if left_value.shape != right_value.shape or not np.allclose(
                    left_value,
                    right_value,
                    rtol=0.0,
                    atol=1.0e-7,
                ):
                    raise ValueError(
                        f"matched-contract mismatch at row {row_index}, field {field}: "
                        f"{reference.label}={left_value.tolist()}, {result.label}={right_value.tolist()}"
                    )

    eval_contracts = {
        result.label: {
            "manifest": result.summary.get("manifest"),
            "checkpoint": result.summary.get("checkpoint"),
            "eval_contract": result.summary.get("eval_contract"),
            "model_block": result.block,
        }
        for result in results
    }
    return {
        "matched": True,
        "row_count": row_count,
        "reference": reference.label,
        "checked_row_fields": [*fields, *float_fields],
        "results": eval_contracts,
    }


def _finite_stats(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"count": 0, "mean": None, "median": None, "p90": None, "min": None, "max": None}
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def _metric(block: dict[str, Any], *path: str) -> Any:
    value: Any = block.get("metrics", {})
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _aggregate_rows(result: Result, rows: list[dict[str, Any]]) -> dict[str, Any]:
    blocks = [row[result.block] for row in rows]
    valid_blocks = [
        block
        for block in blocks
        if bool(block.get("detokenize_ok")) and block.get("error") is None
    ]

    def values(*path: str) -> list[float]:
        output: list[float] = []
        for block in valid_blocks:
            value = _metric(block, *path)
            if value is not None:
                output.append(float(value))
        return output

    count_errors = values("joint_count_error")
    abs_count_errors = [abs(value) for value in count_errors]
    exact_count = sum(abs(value) < 0.5 for value in count_errors)
    topology_f1_all_rows = []
    extreme_overgeneration = 0
    for row, block in zip(rows, blocks, strict=True):
        topology_f1 = _metric(block, "topology", "edge_f1")
        topology_f1_all_rows.append(
            float(topology_f1)
            if topology_f1 is not None and np.isfinite(float(topology_f1))
            else 0.0
        )
        predicted_count = _metric(block, "pred_joint_count")
        if (
            predicted_count is not None
            and float(predicted_count) - float(row["target_joint_count"]) > 50.0
        ):
            extreme_overgeneration += 1
    return {
        "count": len(rows),
        "detokenize_ok": sum(bool(block.get("detokenize_ok")) for block in blocks),
        "errors": sum(block.get("error") is not None for block in blocks),
        "has_eos": sum(bool(block.get("has_eos")) for block in blocks),
        "hit_max_without_eos": sum(bool(block.get("hit_max_without_eos")) for block in blocks),
        "extreme_overgeneration_gt50": int(extreme_overgeneration),
        "joint_count_exact": int(exact_count),
        "joint_count_exact_rate": float(exact_count / max(len(valid_blocks), 1)),
        "joint_count_error": _finite_stats(count_errors),
        "joint_count_abs_error": _finite_stats(abs_count_errors),
        "j2j": _finite_stats(values("official", "j2j")),
        "j2b": _finite_stats(values("official", "j2b")),
        "b2b": _finite_stats(values("official", "b2b")),
        "topology_f1": _finite_stats(values("topology", "edge_f1")),
        "topology_f1_all_rows_zero_for_failure": _finite_stats(
            topology_f1_all_rows
        ),
        "topology_precision": _finite_stats(values("topology", "edge_precision")),
        "topology_recall": _finite_stats(values("topology", "edge_recall")),
    }


def _build_summary(
    results: list[Result],
    strata: list[Stratum],
    contract: dict[str, Any],
) -> dict[str, Any]:
    row_count = contract["row_count"]
    for stratum in strata:
        if stratum.end > row_count:
            raise ValueError(f"stratum {stratum.name!r} ends at {stratum.end}, but row_count={row_count}")

    output: dict[str, Any] = {"contract": contract, "strata": {}}
    all_strata = [Stratum(name="all", start=0, end=row_count), *strata]
    for stratum in all_strata:
        output["strata"][stratum.name] = {
            "range": [stratum.start, stratum.end],
            "results": {
                result.label: _aggregate_rows(result, result.rows[stratum.start : stratum.end])
                for result in results
            },
        }
    return output


def _write_csv(summary: dict[str, Any], output: Path) -> None:
    columns = [
        "stratum",
        "label",
        "count",
        "detokenize_ok",
        "errors",
        "has_eos",
        "hit_max_without_eos",
        "joint_count_exact_rate",
        "joint_count_mae",
        "j2j_mean",
        "j2b_mean",
        "b2b_mean",
        "topology_f1_mean",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for stratum, stratum_payload in summary["strata"].items():
            for label, metrics in stratum_payload["results"].items():
                writer.writerow(
                    {
                        "stratum": stratum,
                        "label": label,
                        "count": metrics["count"],
                        "detokenize_ok": metrics["detokenize_ok"],
                        "errors": metrics["errors"],
                        "has_eos": metrics["has_eos"],
                        "hit_max_without_eos": metrics["hit_max_without_eos"],
                        "joint_count_exact_rate": metrics["joint_count_exact_rate"],
                        "joint_count_mae": metrics["joint_count_abs_error"]["mean"],
                        "j2j_mean": metrics["j2j"]["mean"],
                        "j2b_mean": metrics["j2b"]["mean"],
                        "b2b_mean": metrics["b2b"]["mean"],
                        "topology_f1_mean": metrics["topology_f1"]["mean"],
                    }
                )


def _load_flat_tokenizer(config: Path) -> Any:
    repository_root = Path(__file__).resolve().parents[2]
    scripts_dir = repository_root / "model_training" / "rigweave" / "scripts"
    sys.path.insert(0, str(scripts_dir))
    from train_dynamic_rig import build_tokenizer

    return build_tokenizer(config)


def _decode_puppeteer(
    ids: list[int],
    *,
    n_discrete_size: int,
    target_coord_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    raw_ids = [int(value) for value in ids]
    if raw_ids and raw_ids[0] == 0:
        raw_ids = raw_ids[1:]
    if 1 in raw_ids:
        raw_ids = raw_ids[: raw_ids.index(1)]
    raw_ids = [value for value in raw_ids if value not in {0, 1, 2}]
    usable = (len(raw_ids) // 4) * 4
    if usable == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    raw = np.asarray(raw_ids[:usable], dtype=np.int64).reshape(-1, 4) - 3
    coordinates = raw[:, :3].astype(np.float32) / float(n_discrete_size)
    coordinates = coordinates - 0.5
    coordinates = coordinates / max(float(target_coord_scale), 1.0e-12)
    parents: list[int] = []
    for child, value in enumerate(raw[:, 3].tolist()):
        if child == 0:
            parents.append(-1)
        elif 0 < int(value) <= child:
            parents.append(int(value) - 1)
        else:
            parents.append(-2)
    return coordinates.astype(np.float32), np.asarray(parents, dtype=np.int64)


def _decode_prediction(result: Result, row: dict[str, Any], flat_tokenizer: Any | None) -> tuple[np.ndarray, np.ndarray]:
    block = row[result.block]
    if not bool(block.get("detokenize_ok")) or block.get("error") is not None:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    ids = block.get("generated_ids", [])
    if result.block == "dynamic_puppeteer":
        contract = result.summary["eval_contract"]
        return _decode_puppeteer(
            ids,
            n_discrete_size=int(contract["n_discrete_size"]),
            target_coord_scale=float(contract["target_coord_scale"]),
        )
    if flat_tokenizer is None:
        raise ValueError("flat/stack visualization requires --tokenizer-config")
    tokenizer = flat_tokenizer
    if result.block == "stack_close":
        rigweave_src = Path(__file__).resolve().parents[1] / "rigweave" / "src"
        if str(rigweave_src) not in sys.path:
            sys.path.insert(0, str(rigweave_src))
        from rigweave.stack_close import StackCloseTokenizer

        tokenizer = StackCloseTokenizer(flat_tokenizer)
    decoded = tokenizer.detokenize(np.asarray(ids, dtype=np.int64))
    joints = np.asarray(decoded.joints, dtype=np.float32)
    parents = np.asarray([-1 if value is None else int(value) for value in decoded.parents], dtype=np.int64)
    return joints, parents


def _edges(parents: np.ndarray) -> list[tuple[int, int]]:
    return [
        (int(parent), child)
        for child, parent in enumerate(parents.tolist())
        if 0 <= int(parent) < child
    ]


def _plot_skeleton(
    axis: plt.Axes,
    joints: np.ndarray,
    parents: np.ndarray,
    *,
    color: str,
    linewidth: float,
    alpha: float,
) -> None:
    for parent, child in _edges(parents):
        axis.plot(
            [joints[parent, 0], joints[child, 0]],
            [joints[parent, 1], joints[child, 1]],
            [joints[parent, 2], joints[child, 2]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )
    if joints.shape[0]:
        axis.scatter(
            joints[:, 0],
            joints[:, 1],
            joints[:, 2],
            s=12,
            color=color,
            alpha=alpha,
            depthshade=False,
        )
        axis.scatter(
            [joints[0, 0]],
            [joints[0, 1]],
            [joints[0, 2]],
            s=52,
            color=color,
            edgecolors="black",
            linewidths=0.6,
            depthshade=False,
        )


def _axis_bounds(mesh: np.ndarray, gt_joints: np.ndarray) -> tuple[np.ndarray, float]:
    points = np.concatenate([mesh, gt_joints], axis=0)
    finite = points[np.isfinite(points).all(axis=1)]
    low = finite.min(axis=0)
    high = finite.max(axis=0)
    center = (low + high) * 0.5
    radius = max(float((high - low).max()) * 0.58, 1.0e-3)
    return center, radius


def _set_axes(axis: plt.Axes, center: np.ndarray, radius: float) -> None:
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_xlabel("x", fontsize=6)
    axis.set_ylabel("y", fontsize=6)
    axis.set_zlabel("z", fontsize=6)
    axis.tick_params(labelsize=5)
    axis.view_init(elev=18, azim=-65)


def _metric_text(result: Result, row: dict[str, Any]) -> str:
    block = row[result.block]
    metrics = block.get("metrics", {})
    official = metrics.get("official", {})
    topology = metrics.get("topology", {})
    pred_count = metrics.get("pred_joint_count")
    j2j = official.get("j2j")
    f1 = topology.get("edge_f1")
    if pred_count is None:
        return "generation error"
    return (
        f"pred={int(pred_count)} "
        f"J2J={float(j2j):.4f} F1={float(f1):.3f} "
        f"EOS={int(bool(block.get('has_eos')))} max={int(bool(block.get('hit_max_without_eos')))}"
    )


def _make_comparison_plot(
    results: list[Result],
    row_index: int,
    output: Path,
    flat_tokenizer: Any | None,
) -> None:
    reference_row = results[0].rows[row_index]
    raw = np.load(reference_row["path"], allow_pickle=True)
    query_frame = int(reference_row["query_frame"])
    center = np.asarray(reference_row["query_center"], dtype=np.float32)
    scale = float(reference_row["query_scale"])
    mesh = (
        np.asarray(raw["frame_vertices_rootspace"], dtype=np.float32)[query_frame] - center
    ) / scale
    gt_joints = (
        np.asarray(raw["target_joints_rootspace"], dtype=np.float32)[query_frame] - center
    ) / scale
    gt_parents = np.asarray(raw["target_parents"], dtype=np.int64).reshape(-1)
    rng = np.random.default_rng(7000 + row_index)
    if mesh.shape[0] > 2500:
        mesh_view = mesh[rng.choice(mesh.shape[0], size=2500, replace=False)]
    else:
        mesh_view = mesh
    axis_center, axis_radius = _axis_bounds(mesh_view, gt_joints)

    figure = plt.figure(figsize=(4.2 * (len(results) + 1), 4.4), dpi=150)
    figure.suptitle(
        f"idx={row_index} target={gt_joints.shape[0]} frame={query_frame} {Path(reference_row['path']).name}",
        fontsize=9,
    )
    panels: list[tuple[str, np.ndarray, np.ndarray]] = [("GT", gt_joints, gt_parents)]
    for result in results:
        pred_joints, pred_parents = _decode_prediction(result, result.rows[row_index], flat_tokenizer)
        panels.append((f"{result.label}\n{_metric_text(result, result.rows[row_index])}", pred_joints, pred_parents))

    for column, (title, joints, parents) in enumerate(panels, start=1):
        axis = figure.add_subplot(1, len(panels), column, projection="3d")
        axis.scatter(
            mesh_view[:, 0],
            mesh_view[:, 1],
            mesh_view[:, 2],
            s=0.55,
            color="#aeb7c2",
            alpha=0.16,
            depthshade=False,
        )
        _plot_skeleton(axis, gt_joints, gt_parents, color="#1b9e77", linewidth=1.8, alpha=0.75)
        if column > 1:
            _plot_skeleton(axis, joints, parents, color="#d95f02", linewidth=1.25, alpha=0.82)
        axis.set_title(title, fontsize=7)
        _set_axes(axis, axis_center, axis_radius)
    figure.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)


def _representative_indices(strata: list[Stratum], row_count: int) -> list[int]:
    source = strata or [Stratum(name="all", start=0, end=row_count)]
    output: list[int] = []
    for stratum in source:
        positions = np.linspace(stratum.start, stratum.end - 1, num=min(4, stratum.end - stratum.start))
        output.extend(int(round(value)) for value in positions.tolist())
    return list(dict.fromkeys(output))


def _make_montage(images: list[Path], output: Path) -> None:
    columns = 2
    rows = int(np.ceil(len(images) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(20, 3.4 * rows), dpi=130)
    axes_array = np.asarray(axes, dtype=object).reshape(-1)
    for axis, image_path in zip(axes_array, images, strict=False):
        axis.imshow(plt.imread(image_path))
        axis.set_title(image_path.stem, fontsize=7)
        axis.axis("off")
    for axis in axes_array[len(images) :]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def _print_table(summary: dict[str, Any]) -> None:
    print(
        "stratum\tlabel\tEOS/max/over50\tcount_MAE\tJ2J\tJ2B\tB2B\t"
        "topology_F1_all"
    )
    for stratum, stratum_payload in summary["strata"].items():
        for label, metrics in stratum_payload["results"].items():
            print(
                "\t".join(
                    [
                        stratum,
                        label,
                        (
                            f"{metrics['has_eos']}/"
                            f"{metrics['hit_max_without_eos']}/"
                            f"{metrics['extreme_overgeneration_gt50']}"
                        ),
                        f"{metrics['joint_count_abs_error']['mean']:.4f}",
                        f"{metrics['j2j']['mean']:.6f}",
                        f"{metrics['j2b']['mean']:.6f}",
                        f"{metrics['b2b']['mean']:.6f}",
                        f"{metrics['topology_f1_all_rows_zero_for_failure']['mean']:.6f}",
                    ]
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and compare matched skeleton-generation results.")
    parser.add_argument("--result", action="append", type=_parse_result, required=True, metavar="LABEL=PATH")
    parser.add_argument("--stratum", action="append", type=_parse_stratum, default=[], metavar="NAME=START:END")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--visual-dir", type=Path)
    parser.add_argument("--visual-indices", type=str)
    parser.add_argument("--tokenizer-config", type=Path)
    args = parser.parse_args()

    results = _load_results(args.result)
    contract = _validate_matched_contract(results)
    summary = _build_summary(results, args.stratum, contract)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.output_csv is not None:
        _write_csv(summary, args.output_csv)

    if args.visual_dir is not None:
        needs_legacy_tokenizer = any(
            result.block in {"dynamic", "stack_close"} for result in results
        )
        if needs_legacy_tokenizer and args.tokenizer_config is None:
            raise ValueError(
                "--tokenizer-config is required to visualize a flat/stack result"
            )
        flat_tokenizer = (
            _load_flat_tokenizer(args.tokenizer_config)
            if needs_legacy_tokenizer
            else None
        )
        if args.visual_indices:
            indices = [int(value) for value in args.visual_indices.split(",") if value.strip()]
        else:
            indices = _representative_indices(args.stratum, contract["row_count"])
        for index in indices:
            if index < 0 or index >= contract["row_count"]:
                raise ValueError(f"visual index {index} is outside [0, {contract['row_count']})")
        images: list[Path] = []
        for index in indices:
            output = args.visual_dir / f"matched_idx{index:02d}.png"
            _make_comparison_plot(results, index, output, flat_tokenizer)
            images.append(output)
        _make_montage(images, args.visual_dir / "matched_montage.png")
        summary["visuals"] = {
            "indices": indices,
            "images": [str(path) for path in images],
            "montage": str(args.visual_dir / "matched_montage.png"),
        }
        args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    _print_table(summary)


if __name__ == "__main__":
    main()
