"""Read saved predictions on identical inactive joints; never run a model."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "rigweave/scripts"
sys.path.insert(0, str(SCRIPTS))


def joint_matches(joints, parents, target, target_parents):
    count = len(target)
    if joints is None or not len(joints):
        return {"distance": [None] * count, "nearest_prediction": [-1] * count,
                "covered": [False] * count, "edge_found": [False] * count}
    distances = np.linalg.norm(np.asarray(target)[:, None] - np.asarray(joints)[None], axis=-1)
    nearest = distances.argmin(1)
    minimum = distances.min(1)
    edges = {(int(parent), child) for child, parent in enumerate(parents)
             if parent is not None and parent >= 0 and parent != child}
    found = [parent >= 0 and (int(nearest[parent]), int(nearest[child])) in edges
             for child, parent in enumerate(target_parents)]
    return {"distance": minimum.tolist(), "nearest_prediction": nearest.tolist(),
            "covered": (minimum <= 0.05).tolist(), "edge_found": found}


def score_mask(row, mask):
    ids = np.flatnonzero(mask)
    matches = row["joint_matches"]
    distances = [matches["distance"][index] for index in ids]
    return {"joints": len(ids),
            "covered": sum(bool(matches["covered"][index]) for index in ids),
            "edges_found": sum(bool(matches["edge_found"][index]) for index in ids),
            "mean_distance": float(np.mean(distances)) if distances and None not in distances else None,
            "success": row["success"]}


def summarize_same_joints(rows):
    output = []
    for index in sorted({row["index"] for row in rows}):
        views = {row["view"]: row for row in rows if row["index"] == index}
        if set(views) != {"normal8", "weak2", "static8"}:
            raise ValueError("every asset needs exactly the three original views")
        quiet8 = np.asarray(views["normal8"]["observation"]["strata"]["quiet_strict"], dtype=bool)
        quiet2 = np.asarray(views["weak2"]["observation"]["strata"]["quiet_strict"], dtype=bool)
        hidden2 = np.asarray(views["weak2"]["observation"]["strata"]["hidden_demonstrated_strict"], dtype=bool)
        any_active8 = any(views["normal8"]["observation"]["strata"]["observed_active"])
        any_active2 = any(views["weak2"]["observation"]["strata"]["observed_active"])
        masks = {"quiet_in_both_inputs": quiet8 & quiet2,
                 "quiet_in_both_with_activity_elsewhere": quiet8 & quiet2 & any_active8 & any_active2,
                 "hidden_in_weak2": hidden2}
        output.append({"index": index, "asset_id": views["normal8"]["asset_id"],
                       "matched_masks": {name: {"joint_indices": np.flatnonzero(mask).tolist(),
                           "views": {view: score_mask(row, mask) for view, row in views.items()}}
                           for name, mask in masks.items()}})
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("this saved-output audit must have CUDA_VISIBLE_DEVICES empty")
    import torch
    from eval_dynamic_rig_generation import _parents_from_output
    from motion_experiment_runtime import guard, load_config, source_metadata, write_json
    from relation_probe_metrics import file_hash, structure_by_stratum
    from train_dynamic_rig import build_tokenizer

    config = load_config(args.config)
    guard(config, "inspect")
    source = source_metadata(config)
    plan = config["relation_anchor_probe"]
    hashes = {}
    for name in ("source_cache", "source_evaluation"):
        hashes[name] = file_hash(Path(plan[name]))
        if hashes[name] != plan[name + "_sha256"]:
            raise ValueError(f"source hash changed: {name}")
    report = json.loads(Path(plan["source_evaluation"]).read_text())
    if report.get("complete") is not True or report["plan"]["checkpoint"] != plan["checkpoint"]:
        raise ValueError("source evaluation is not the completed base-matched report")
    torch.set_num_threads(1)
    cache = torch.load(plan["source_cache"], map_location="cpu", mmap=True, weights_only=False)
    tokenizer = build_tokenizer(Path(config["baseline"]["args"]["tokenizer_config"]))
    baseline = {(row["index"], row["view"]): row for row in report["baseline"]["valid"]}
    rows = []
    for entry in cache["valid"]:
        key = entry["index"], entry["view"]
        recorded = baseline[key]["generation"]
        count = int(entry["batch"]["joint_count"][0])
        target = entry["batch"]["target_joints"][0, :count].numpy()
        target_parents = entry["batch"]["target_parents"][0, :count].numpy()
        prediction = tokenizer.detokenize(np.asarray(recorded["generated_ids"])) if recorded["success"] else None
        joints = None if prediction is None else np.asarray(prediction.joints)
        parents = [] if prediction is None else _parents_from_output(prediction)
        recomputed = structure_by_stratum(joints, parents, target, target_parents, entry["observation"])
        if recomputed != recorded["strata"]:
            raise ValueError(f"saved stratum metrics do not replay exactly: {key}")
        query_points = entry["input"]["query_points"][0].float().numpy()
        rows.append({"index": entry["index"], "asset_id": entry["asset_id"], "view": entry["view"],
                     "success": recorded["success"], "hitmax": recorded["hit_max_without_eos"],
                     "target_joints": target.tolist(), "target_parents": target_parents.tolist(),
                     "pred_joints": None if joints is None else joints.tolist(), "pred_parents": parents,
                     "query_points": query_points[0].tolist(),
                     "anchor_displacement_max": np.linalg.norm(query_points - query_points[:1], axis=-1).max(0).tolist(),
                     "observation": entry["observation"],
                     "joint_matches": joint_matches(joints, parents, target, target_parents)})
    for index in {row["index"] for row in rows}:
        views = [row for row in rows if row["index"] == index]
        for row in views[1:]:
            for name in ("target_joints", "target_parents", "query_points"):
                if row[name] != views[0][name]:
                    raise ValueError(f"paired geometry or target changed: {index}, {name}")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU audit unexpectedly initialized CUDA")
    output = {"complete": True, "source": source, "input_hashes": hashes,
              "checkpoint": plan["checkpoint"], "rows": rows, "same_joint_comparisons": summarize_same_joints(rows),
              "limits": ["Saved original-model predictions only; no forward, training, or parameter change.",
                         "Quiet is relative-to-parent joint movement, not proof the nearby surface is fixed.",
                         "Joint coverage uses distance <=0.05 query units; nearest-neighbor matching is not one-to-one.",
                         "Matched-view changes measure supplied pose information; they do not isolate the learned explicit bias from the original multiframe path."]}
    write_json(args.output, output)
    print(json.dumps({"complete": True, "rows": len(rows), "assets": len(output["same_joint_comparisons"]),
                      "output": str(args.output), "cuda_initialized": torch.cuda.is_initialized()}))


if __name__ == "__main__":
    main()
