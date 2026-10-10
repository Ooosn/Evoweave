"""Paired asset-cluster uncertainty for the predeclared relation probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from motion_experiment_runtime import write_json


def metric_value(row, name):
    if name == "ce":
        return float(row["ce"]), 1.0
    generation = row["generation"]
    if name == "f1":
        return float(generation["topology_f1_or_zero"]), 1.0
    stratum, metric = name.split(":")
    values = generation["strata"][stratum]
    weight = float(values["edges" if metric == "edge_recall" else "joints"])
    return float(values[metric]) if weight else 0.0, weight


def paired_measurement(reference, candidate, metric, *, seed=20261011, replicates=4000):
    a = {(row["asset_id"], row["view"]): row for row in reference}
    b = {(row["asset_id"], row["view"]): row for row in candidate}
    if len(a) != len(reference) or len(b) != len(candidate) or a.keys() != b.keys():
        raise ValueError("paired rows must have identical, unique asset/view keys")
    assets = sorted({key[0] for key in a})
    counts = np.zeros(len(assets))
    sums_a, sums_b = np.zeros(len(assets)), np.zeros(len(assets))
    for i, asset in enumerate(assets):
        for key in a:
            if key[0] != asset:
                continue
            av, aw = metric_value(a[key], metric)
            bv, bw = metric_value(b[key], metric)
            if aw != bw or not np.isfinite([av, bv, aw]).all():
                raise ValueError("paired metric weight changed or nonfinite measurement")
            counts[i] += aw
            sums_a[i] += av * aw
            sums_b[i] += bv * bw
    active = counts > 0
    total = counts.sum()
    result = {"assets": len(assets), "contributing_assets": int(active.sum()), "weight": float(total)}
    if total == 0:
        return {**result, "reference": None, "candidate": None, "delta": None, "bootstrap95": None}
    first, second = float(sums_a.sum() / total), float(sums_b.sum() / total)
    difference = sums_b - sums_a
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(assets), (replicates, len(assets)))
    denominators = counts[selected].sum(1)
    eligible = denominators > 0
    estimates = difference[selected[eligible]].sum(1) / denominators[eligible]
    interval = np.quantile(estimates, [0.025, 0.975]).tolist() if active.sum() >= 3 else None
    return {**result, "reference": first, "candidate": second, "delta": second - first,
            "relative_delta_pct": 100 * (second - first) / first if first else None,
            "bootstrap95": interval,
            "assets_candidate_lower": int((difference[active] < -1e-8).sum()),
            "assets_candidate_higher": int((difference[active] > 1e-8).sum())}


def analysis(report):
    if not report.get("complete") or report["stage"] not in {"relation_screen", "relation_anchor_screen"} or report["changed_base_parameters"]:
        raise ValueError("require a completed, unchanged-base relation screen")
    groups = {"base": report["baseline"]["valid"], **report["evaluation"]}
    anchored = report["stage"] == "relation_anchor_screen"
    pairs = (("base", "actual"),) if anchored else (("base", "actual"), ("unknown", "actual"),
             ("base", "unknown"), ("actual", "actual_evidence_ablated"))
    metrics = ["ce", "f1"] + [f"{stratum}:{metric}" for stratum in
        ("quiet_strict", "quiet_loose", "hidden_demonstrated_strict", "hidden_demonstrated_loose", "observed_active")
        for metric in ("joint_coverage_005", "edge_recall")]
    results = {}
    for reference, candidate in pairs:
        comparisons = {}
        for view in ("all", *report["plan"]["views"]):
            aa = [row for row in groups[reference] if view == "all" or row["view"] == view]
            bb = [row for row in groups[candidate] if view == "all" or row["view"] == view]
            comparisons[view] = {metric: paired_measurement(aa, bb, metric) for metric in metrics}
        results[candidate + "_minus_" + reference] = comparisons
    if anchored:
        for control, values in report["ce_controls"].items():
            results[control + "_minus_actual"] = {}
            for view in ("all", *report["plan"]["views"]):
                aa = [row for row in groups["actual"] if view == "all" or row["view"] == view]
                bb = [row for row in values if view == "all" or row["view"] == view]
                results[control + "_minus_actual"][view] = {"ce": paired_measurement(aa, bb, "ce")}
    training = {}
    baseline = report["baseline"]["train"]
    plan = report["plan"]
    rng = np.random.default_rng(plan["seed"] + 1)
    order = []
    while len(order) < plan["steps"] * plan["accumulation"]:
        order.extend(rng.permutation(len(baseline)).tolist())
    for arm, rows in report["training"].items():
        relative = []
        for row in rows:
            step = row["step"]
            entries = [baseline[i] for i in order[(step - 1) * plan["accumulation"]:step * plan["accumulation"]]]
            base = sum(e["ce"] * e["token_weight"] for e in entries) / sum(e["token_weight"] for e in entries)
            relative.append(row["ce"] / base - 1)
        training[arm] = {"steps": len(rows), "samples": rows[-1]["samples"],
                         "last20_mean_relative_ce_change_pct": float(np.mean(relative[-20:]) * 100)}
    return {"stage": report["stage"], "summary": report["summary"], "paired": results, "training": training,
            "source": report["source"], "source_checkpoint": report["source_checkpoint"],
            "notes": ["All-view confidence intervals resample whole assets, retaining paired frame views.",
                      "Selected stratified16 assets and one training seed; exploratory intervals are not multiplicity-corrected.",
                      "Lower CE is better; higher F1, coverage and edge recall are better.",
                      "Local-transform activity is only a proxy for what mesh motion reveals, not guaranteed surface observability.",
                      "Ablation of an already trained arm is an intervention, not an independently trained geometry baseline.",
                      "Anchored follow-up reuses the first screen's validation assets and verified frozen-base generations; it is not independent confirmation." if anchored else "Plain residual is an unconstrained feature update even for allunknown evidence."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, analysis(json.loads(args.input.read_text())))


if __name__ == "__main__":
    main()
