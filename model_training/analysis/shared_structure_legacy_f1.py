"""Recompute the existing topology P/R/F1 on saved shared-skeleton predictions."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


TRAINING = Path(__file__).resolve().parents[1]
LEGACY = TRAINING / "rigweave/src/rigweave/dynamic_rig/skeleton_topology_metrics.py"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_legacy_metric(path=LEGACY):
    # Load the unchanged NumPy metric without importing GPU/model dependencies.
    names = {"_edge_set", "_edge_overlap", "_topology_metrics"}
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != names or len(functions) != len(names):
        raise ValueError("legacy topology metric functions are missing or duplicated")
    namespace = {"np": np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_topology_metrics"]


def score_view(saved, metric):
    selection = saved["selection"]
    frame_ids = selection["target_raw_frames"]
    heldout = selection["heldout_target_positions"]
    if len(heldout) != 8 or len(set(heldout)) != 8 or any(i < 0 or i >= len(frame_ids) for i in heldout):
        raise ValueError("expected eight distinct saved heldout target frames")
    query = frame_ids.index(selection["query_raw_frame"])
    parents = np.asarray(saved["gt_parents"]).reshape(-1).tolist()
    gt = np.asarray(saved["gt_joints_mesh_local"], dtype=np.float32)[0]
    if gt.shape != (len(frame_ids), len(parents), 3) or any(p < -1 or p >= len(parents) for p in parents):
        raise ValueError("invalid unpadded saved target skeleton")
    success = bool(saved["generation"]["success"])
    prediction = np.asarray(saved["pred_joints_mesh_local"], dtype=np.float32)[0] if success else None
    predicted_parents = saved["generation"].get("parents", [])
    if success and (prediction.shape != (len(frame_ids), len(predicted_parents), 3) or not np.isfinite(prediction).all()):
        raise ValueError("invalid successful saved prediction")
    frames = {}
    for i in sorted(set([query, *heldout])):
        values = metric(prediction[i], predicted_parents, gt[i], parents) if success else {
            "edge_precision": 0.0, "edge_recall": 0.0, "edge_f1": 0.0}
        frames[i] = {"target_position": i, "raw_frame": frame_ids[i], **values}
    return {"success": success, "query": frames[query], "heldout": [frames[i] for i in heldout],
            "heldout_mean": {name: float(np.mean([frames[i][name] for i in heldout]))
                             for name in ("edge_precision", "edge_recall", "edge_f1")}}


def teacher_control(saved, metric):
    parents = np.asarray(saved["gt_parents"]).reshape(-1).tolist()
    teacher = np.asarray(saved["teacher_joints_mesh_local"], dtype=np.float32)
    target = np.asarray(saved["gt_joints_mesh_local"], dtype=np.float32)
    if teacher.shape != target.shape or not np.isfinite(teacher).all():
        raise ValueError("invalid saved GT-prefix prediction")
    controlled = {**saved, "pred_joints_mesh_local": teacher,
                  "generation": {"success": True, "parents": parents}}
    result = score_view(controlled, metric)
    result["ground_truth_topology_supplied"] = True
    heldout = saved["selection"]["heldout_target_positions"]
    errors = np.linalg.norm(teacher[0] - target[0], axis=-1)
    result["heldout_identity_mpjpe"] = float(errors[heldout].mean())
    result["heldout_stable_identity_coverage"] = float((errors[heldout] <= .05).all(axis=0).mean())
    if saved["generation"].get("exact_tree", False):
        free = np.asarray(saved["pred_joints_mesh_local"], dtype=np.float32)
        if free.shape != teacher.shape:
            raise ValueError("exact-tree prediction has the wrong coordinate shape")
        difference = np.linalg.norm(free[0] - teacher[0], axis=-1)
        result["exact_tree_free_teacher_mpjpe"] = float(difference.mean())
        result["exact_tree_free_teacher_max_joint_distance"] = float(difference.max())
    return result


def summarize(rows):
    result = {}
    for key in sorted({(row["label"], row["split"], row["view"]) for row in rows}):
        selected = [row for row in rows if (row["label"], row["split"], row["view"]) == key]
        block = {"assets": len(selected), "successful_assets": sum(row["success"] for row in selected)}
        for pose in ("query", "heldout_mean"):
            block[pose] = {name: float(np.mean([row[pose][name] for row in selected]))
                           for name in ("edge_precision", "edge_recall", "edge_f1")}
            scores = [row[pose]["edge_f1"] for row in selected]
            block[pose].update(f1_median=float(np.median(scores)), f1_min=float(min(scores)),
                               f1_max=float(max(scores)), zero_f1_assets=sum(value == 0 for value in scores))
        for name in ("heldout_identity_mpjpe", "heldout_stable_identity_coverage", "exact_tree_free_teacher_mpjpe",
                     "exact_tree_free_teacher_max_joint_distance"):
            values = [row[name] for row in selected if name in row]
            if values:
                block[name] = {"assets": len(values), "mean": float(np.mean(values)), "max": float(max(values))}
        result["/".join(key)] = block
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assessment", type=Path, required=True)
    parser.add_argument("--assessment-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-controls", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if file_hash(args.assessment) != args.assessment_sha256:
        raise ValueError("assessment report checksum mismatch")
    report = json.loads(args.assessment.read_text(encoding="utf-8"))
    if not report.get("complete") or not report.get("accepted"):
        raise ValueError("assessment has not completed successfully")
    import torch
    torch.set_num_threads(1)
    metric = load_legacy_metric()
    rows, artifacts = [], []
    for label in ("initialized", "trained"):
        reference = {(row["asset_id"], row["view"]): row for row in report[label]["rows"] if row["view"] == "T8"}
        if len(reference) != sum(row["view"] == "T8" for row in report[label]["rows"]):
            raise ValueError("duplicate T8 reference asset")
        seen = set()
        for artifact in report[label]["artifacts"]:
            path = Path(artifact["path"])
            if file_hash(path) != artifact["sha256"]:
                raise ValueError(f"saved prediction checksum mismatch: {path}")
            payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
            key = (payload["asset_id"], "T8")
            if key not in reference or key in seen:
                raise ValueError("unexpected or duplicated saved prediction")
            source = reference[key]
            saved = payload["views"]["T8"]
            if saved["selection"] != source["selection"] or saved["generation"] != source["generation"]:
                raise ValueError("saved prediction and assessment metadata disagree")
            rows.append({"label": label, **{name: source[name] for name in ("split", "index", "asset_id", "view")},
                         "exact_tree": source["generation"]["exact_tree"], **score_view(saved, metric)})
            if args.teacher_controls:
                rows.append({"label": label + "_gt_prefix",
                             **{name: source[name] for name in ("split", "index", "asset_id", "view")},
                             **teacher_control(saved, metric)})
            artifacts.append({"label": label, "path": str(path), "sha256": artifact["sha256"]})
            seen.add(key)
            del payload, saved
        if seen != set(reference):
            raise ValueError("incomplete saved prediction coverage")
        print(f"SCORED {label} assets={len(seen)}", flush=True)
    result = {"complete": True, "source_assessment": {"path": str(args.assessment), "sha256": args.assessment_sha256},
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=TRAINING, text=True).strip(),
              "legacy_metric": {"path": str(LEGACY), "sha256": file_hash(LEGACY), "function": "_topology_metrics"},
              "protocol": "Unchanged legacy directed-edge P/R/F1 after independent nearest-position mappings in each direction; no distance cutoff or one-to-one assignment. Failed/invalid generation scores zero. Compute per-frame F1, then mean across the eight heldout frames per asset, then macro-average assets. Query uses the saved source-query raw frame. Shared translation/isotropic scale does not change this metric. Same formula as prior training, not the same asset/frame protocol, and not an identity-reuse or geometry-distance metric.",
              "teacher_controls": {"enabled": args.teacher_controls,
                  "meaning": "GT parent prefix and GT topology with the saved teacher-forced coordinate predictions. A diagnostic control, not free generation. Exact-tree free/teacher coordinate differences also expose potential prefix-reuse mismatch; frozen encoder outputs were recomputed separately in the original evaluation."},
              "no_model_inference": True, "optimizer_steps": 0, "artifacts": artifacts,
              "summary": summarize(rows), "rows": rows}
    if file_hash(args.assessment) != args.assessment_sha256:
        raise ValueError("source assessment changed during scoring")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
