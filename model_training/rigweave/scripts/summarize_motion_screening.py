"""Compare complete screening reports after checking their paired inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from motion_experiment_runtime import guard, load_config, write_json


def row_key(row):
    return row["path"], row["frames"], row["control"]


def training_summary(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip().startswith("{")]
    steps = [row for row in rows if "gradient_norm_before_clip" in row]
    if not steps:
        raise ValueError(f"no measured optimizer steps: {path}")
    scalar_keys = ("loss", "ce", "gradient_norm_before_clip", "gpu_peak_gb", "seconds")
    if not all(np.isfinite(row[key]) for row in steps for key in scalar_keys):
        raise ValueError(f"non-finite training measurement: {path}")
    return {"logged_steps": len(steps), "last_step": steps[-1]["step"],
            "last_loss": steps[-1]["loss"], "median_step_seconds": float(np.median([row["seconds"] for row in steps[1:]])),
            "peak_gpu_gib": max(row["gpu_peak_gb"] for row in steps),
            "frame_counts_seen_in_logged_steps": sorted({count for row in steps for count in row["frames_in_step"]}),
            "adapter_parameter_norm": steps[-1]["motion_adapter_parameter_norm"],
            "adapter_gradient_norm_after_clip": steps[-1]["motion_adapter_gradient_norm_after_clip"],
            "paired_frame_schedule": {str(row["step"]): row["frames_in_step"] for row in steps}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    guard(config, "report")
    job = Path(config["runtime"]["job_root"])
    reports = {}
    for name in args.names:
        path = job / f"results/screen_{name}_evaluation.json"
        report = json.loads(path.read_text())
        if not report.get("complete") or report["checkpoint_step"] != config["screening"]["optimizer_steps"]:
            raise ValueError(f"incomplete or mismatched screening report: {path}")
        reports[name] = report
    if "control" not in reports:
        raise ValueError("the paired fresh control must be included")
    baseline = {row_key(row): row for row in reports["control"]["rows"]}
    by_name = {}
    for name in ["control", *[value for value in reports if value != "control"]]:
        report = reports[name]
        rows = {row_key(row): row for row in report["rows"]}
        if rows.keys() != baseline.keys():
            raise ValueError(f"evaluation row set differs: {name}")
        for key, row in rows.items():
            for field in ("index", "selected_frames", "query_center", "query_scale"):
                if row[field] != baseline[key][field]:
                    raise ValueError(f"unpaired {field} for {name} at {key}")
        normal = [row for row in rows.values() if row["control"] == "normal"]
        generation = [row["generation"] for row in normal if "generation" in row]
        differences = {}
        for row in normal:
            differences.setdefault(row["path"], []).append(row["ce"] - baseline[row_key(row)]["ce"])
        paired = np.asarray([np.mean(values) for values in differences.values()])
        rng = np.random.default_rng(config["changes"]["initialization_seed"])
        interval = np.quantile(rng.choice(paired, (2000, len(paired))).mean(axis=1), [0.025, 0.975]).tolist()
        train = training_summary(Path(config["runtime"]["output_root"]) / "screen" / name / "logs/train.log")
        if by_name and train["paired_frame_schedule"] != by_name["control"]["training"]["paired_frame_schedule"]:
            raise ValueError(f"training frame schedule differs: {name}")
        by_name[name] = {"ce": report["mean_multiframe_ce"], "summary": report["summary"],
                         "paired_mean_ce_delta_vs_control": float(paired.mean()),
                         "asset_bootstrap_95pct_ce_delta_interval": interval,
                         "generation_success": sum(value["success"] for value in generation),
                         "generation_count": len(generation),
                         "topology_f1_all_rows": float(np.mean([value["topology_f1_or_zero"] for value in generation])),
                         "training": train}
    ranking = sorted(by_name, key=lambda name: by_name[name]["ce"])
    best_ce = by_name[ranking[0]]["ce"]
    result = {"paired_inputs_verified": True, "names": args.names, "candidates": by_name,
              "ce_ranking": ranking, "within_one_percent_of_best_ce": [name for name in ranking if by_name[name]["ce"] <= best_ce * 1.01],
              "interpretation": "Short paired runs screen suitability, not full-budget quality. Bootstrap resamples whole assets across all T; it is descriptive and does not account for initialization or dataset-selection uncertainty. Selection remains a primary-agent decision including generation and cost."}
    write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "paired": True, "candidates": {
        name: {key: value for key, value in values.items() if key not in {"summary", "training"}}
        for name, values in by_name.items()}}), flush=True)


if __name__ == "__main__":
    main()
