"""Same-joint baseline comparisons with one-to-one location checks and figures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


VIEWS = ("normal8", "weak2", "static8")


def unique_matches(target, target_parents, prediction, pred_parents, tolerance=0.05):
    target = np.asarray(target, dtype=np.float64)
    count = len(target)
    assignment = np.full(count, -1, dtype=int)
    if prediction is not None and len(prediction):
        prediction = np.asarray(prediction, dtype=np.float64)
        distance = np.linalg.norm(target[:, None] - prediction[None], axis=-1)
        # One extra match outweighs every possible within-tolerance distance.
        real_cost = np.where(distance <= tolerance, distance / tolerance / (count + 1), 2.0)
        cost = np.concatenate((real_cost, np.ones((count, count))), axis=1)
        source, destination = linear_sum_assignment(cost)
        for i, j in zip(source, destination):
            if j < len(prediction) and distance[i, j] <= tolerance:
                assignment[i] = j
    covered = assignment >= 0
    pred_edges = {(int(parent), child) for child, parent in enumerate(pred_parents)
                  if parent is not None and parent >= 0 and parent != child}
    edges = np.asarray([parent >= 0 and covered[i] and covered[parent]
                        and (int(assignment[parent]), int(assignment[i])) in pred_edges
                        for i, parent in enumerate(target_parents)], dtype=bool)
    return {"assignment": assignment.tolist(), "covered": covered.tolist(), "located_edge_found": edges.tolist()}


def count_mask(row, mask):
    mask = np.asarray(mask, dtype=bool)
    valid_edges = mask & (np.asarray(row["target_parents"]) >= 0)
    unique = row["unique_matches"]
    return {"joints": int(mask.sum()), "edges": int(valid_edges.sum()),
            "nearest_covered": int(np.asarray(row["joint_matches"]["covered"])[mask].sum()),
            "unique_covered": int(np.asarray(unique["covered"])[mask].sum()),
            "unique_located_edges": int(np.asarray(unique["located_edge_found"])[valid_edges].sum()),
            "success": bool(row["success"])}


def total_counts(values):
    fields = ("joints", "edges", "nearest_covered", "unique_covered", "unique_located_edges")
    result = {field: sum(value[field] for value in values) for field in fields}
    result["cases"] = len(values)
    result["successes"] = sum(value["success"] for value in values)
    return result


def compare_saved(report):
    rows = report["rows"]
    for row in rows:
        row["unique_matches"] = unique_matches(row["target_joints"], row["target_parents"],
                                                row["pred_joints"], row["pred_parents"])
    by_key = {(row["index"], row["view"]): row for row in rows}
    if len(by_key) != len(rows):
        raise ValueError("duplicate asset/view rows")
    paired = []
    for item in report["same_joint_comparisons"]:
        index = item["index"]
        for name, group in item["matched_masks"].items():
            mask = np.zeros(len(by_key[(index, "normal8")]["target_joints"]), dtype=bool)
            mask[group["joint_indices"]] = True
            paired.append({"index": index, "asset_id": item["asset_id"], "mask": name,
                           "joint_indices": group["joint_indices"],
                           "all_views_success": all(by_key[(index, view)]["success"] for view in VIEWS),
                           "views": {view: count_mask(by_key[(index, view)], mask) for view in VIEWS}})
    summary = {}
    for name in sorted({row["mask"] for row in paired}):
        selected = [row for row in paired if row["mask"] == name and row["joint_indices"]]
        summary[name] = {}
        for cohort in ("all_cases", "all_three_views_success"):
            cases = selected if cohort == "all_cases" else [row for row in selected if row["all_views_success"]]
            summary[name][cohort] = {view: total_counts([row["views"][view] for row in cases]) for view in VIEWS}
    return rows, paired, summary


def render(rows, comparisons, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    selected = (385, 202, 128)
    by_key = {(row["index"], row["view"]): row for row in rows}
    pairs = {(row["index"], row["mask"]): row for row in comparisons}
    figure, axes = plt.subplots(3, 4, figsize=(15, 10.5), constrained_layout=True)
    for row_number, index in enumerate(selected):
        reference = by_key[(index, "weak2")]
        points = np.asarray(reference["query_points"])
        center = points.mean(0)
        _, _, basis = np.linalg.svd(points - center, full_matrices=False)
        projection = basis[:2].T
        projected = (points - center) @ projection
        target = np.asarray(reference["target_joints"])
        target2 = (target - center) @ projection
        if index == 385:
            quiet = np.asarray(reference["observation"]["strata"]["quiet_strict"], dtype=bool)
            mask_label = "quiet in the two-pose input"
        else:
            quiet = np.zeros(len(target), dtype=bool)
            quiet[pairs[(index, "quiet_in_both_with_activity_elsewhere")]["joint_indices"]] = True
            mask_label = "quiet in both moving inputs"
        extent = np.ptp(projected, axis=0)
        pad = max(float(extent.max()) * 0.08, 0.02)
        bounds = projected.min(0) - pad, projected.max(0) + pad
        for column, view in enumerate(("truth", *VIEWS)):
            axis = axes[row_number, column]
            axis.scatter(projected[:, 0], projected[:, 1], c="#B8BFC6", s=2, alpha=0.32, linewidths=0)
            axis.set_aspect("equal")
            axis.set_xlim(bounds[0][0], bounds[1][0])
            axis.set_ylim(bounds[0][1], bounds[1][1])
            axis.axis("off")
            if view == "truth":
                for j, parent in enumerate(reference["target_parents"]):
                    if parent >= 0:
                        axis.plot(target2[[parent, j], 0], target2[[parent, j], 1], color="#505B66", lw=1.1)
                axis.scatter(target2[quiet, 0], target2[quiet, 1], c="#16836E", s=20, zorder=4)
                axis.set_title(f"Asset {index}: ground truth ({len(target)} joints)\n{int(quiet.sum())} {mask_label}", fontsize=10)
                continue
            value = by_key[(index, view)]
            if value["success"]:
                prediction = (np.asarray(value["pred_joints"]) - center) @ projection
                for j, parent in enumerate(value["pred_parents"]):
                    if parent is not None and 0 <= parent < len(prediction):
                        axis.plot(prediction[[parent, j], 0], prediction[[parent, j], 1], color="#2867A0", lw=1.3)
                axis.scatter(prediction[:, 0], prediction[:, 1], c="#2867A0", s=7, zorder=3)
                uncovered = quiet & ~np.asarray(value["unique_matches"]["covered"], dtype=bool)
                axis.scatter(target2[uncovered, 0], target2[uncovered, 1], facecolors="none", edgecolors="#CC4046",
                             s=40, linewidths=1.1, zorder=5)
                score = count_mask(value, quiet)
                axis.set_title(f"{view}: {len(prediction)} predicted joints\nQuiet matched {score['unique_covered']}/{score['joints']}; edges {score['unique_located_edges']}/{score['edges']}", fontsize=10)
            else:
                axis.text(0.5, 0.5, "Generation did not finish", ha="center", va="center", transform=axis.transAxes,
                          fontsize=11, color="#A83238")
                axis.set_title(view, fontsize=10)
    figure.suptitle("Already-trained model: inactive joints, matched query geometry", fontsize=16)
    figure.legend(handles=[Line2D([0], [0], color="#16836E", marker="o", lw=0, label="Quiet target joint"),
                           Line2D([0], [0], color="#2867A0", lw=2, label="Predicted skeleton"),
                           Line2D([0], [0], color="#CC4046", marker="o", markerfacecolor="none", lw=0, label="Unmatched quiet target")],
                  loc="outside lower center", ncol=3, frameon=False, fontsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    data = args.input.read_bytes()
    original = json.loads(data)
    rows, comparisons, summary = compare_saved(original)
    output = {"complete": True, "source_sha256": hashlib.sha256(data).hexdigest(),
              "checkpoint": original["checkpoint"], "summary": summary, "same_joint_comparisons": comparisons,
              "rows": rows, "matching": "Maximum-cardinality one-to-one matches within0.05 query units, then minimum distance. An edge counts only if both endpoints are matched and the directed predicted edge exists. This is a geometric proxy, not an anatomical identity oracle.",
              "limits": original["limits"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    if args.figure:
        render(rows, comparisons, args.figure)
    print(json.dumps({"complete": True, "output": str(args.output), "figure": str(args.figure), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
