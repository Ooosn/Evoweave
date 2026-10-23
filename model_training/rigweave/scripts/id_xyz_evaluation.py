"""Original-GT scoring and explicit cross-pose structure agreement."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from rigweave.dynamic_rig.aux_skeleton_tokens import AuxSkeletonTokens
from rigweave.dynamic_rig.skeleton_topology_metrics import _topology_metrics
from structure_xyz_fill_evaluation import _j2j, _reference_rows
from structure_xyz_fill_runtime import load_small_reference_panel
from id_xyz_runtime import fixed_dataset, prepare_batch, write_json


def summarize(rows):
    if not rows:
        return {"assets": 0, "views": 0}
    assets = defaultdict(list)
    for row in rows:
        assets[row["asset_id"]].append(row)
    values = []
    for identity, frames in assets.items():
        signatures = [tuple(row["predicted_structure_tokens"]) if row["success"] else None for row in frames]
        pairs = [(a, b) for a in range(len(frames)) for b in range(a + 1, len(frames))]
        values.append({
            "asset_id": identity, "gt_joints": frames[0]["gt_joint_count"],
            "original_edge_f1": float(np.mean([row["original_edge_f1"] for row in frames])),
            "success": float(np.mean([row["success"] for row in frames])),
            "exact_count": float(np.mean([row["exact_count"] for row in frames])),
            "exact_structure": float(np.mean([row["exact_structure"] for row in frames])),
            "all_frames_structure_equal": all(s is not None for s in signatures) and len(set(signatures)) == 1,
            "pairwise_structure_agreement": (sum(signatures[a] is not None and signatures[a] == signatures[b]
                                                for a, b in pairs) / len(pairs)) if pairs else None,
        })
    def aggregate(items):
        if not items:
            return {"assets": 0}
        return {"assets": len(items), **{
            key: float(np.mean([item[key] for item in items]))
            for key in ("original_edge_f1", "success", "exact_count", "exact_structure",
                        "all_frames_structure_equal", "pairwise_structure_agreement")
            if all(item[key] is not None for item in items)}}
    success = [row for row in rows if row["success"]]
    return {**aggregate(values), "views": len(rows), "successful_views": len(success),
            "j2j_successful_only": float(np.mean([row["j2j"] for row in success])) if success else None,
            "reference_B65_f1": float(np.mean([row["reference_B65_f1"] for row in rows]))
                if all("reference_B65_f1" in row for row in rows) else None,
            "gt52": aggregate([item for item in values if item["gt_joints"] == 52]),
            "non52": aggregate([item for item in values if item["gt_joints"] != 52]),
            "per_asset": values}


@torch.no_grad()
def evaluate_item(model, tokenizer, codec, surface, target, geometry, *,
                  frame_indices, group, reference=None, max_new_tokens=2048):
    identity = geometry["asset_id"][0]
    count = int(geometry["joint_count"][0])
    raw_frames = geometry["selected_frames"][0].cpu().numpy()
    gt_joints = geometry["frame_target_joints"][0, :, :count].cpu().numpy()
    gt_parents = geometry["target_parents"][0, :count].cpu().numpy()
    scores = (_reference_rows(geometry, tokenizer, AuxSkeletonTokens(),
        group=group, asset_id=identity, frame_indices=frame_indices, raw_frame_indices=raw_frames,
        reference_lookup=reference) if reference is not None else {})
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=surface["query_points"].is_cuda):
        condition = model.encode(surface)
        output = model.forward_encoded(condition, target, consistency_weight=1.)
        rows = []
        for frame in frame_indices:
            generated = model.generate(condition[:, frame], codec, max_new_tokens=max_new_tokens)
            labels = target["input_ids"][0, frame][target["attention_mask"][0, frame]].cpu().numpy()
            truth = codec.decode(labels)
            status, parsed = generated["status"], None
            if generated["success"]:
                try:
                    parsed = codec.decode(generated["tokens"])
                except ValueError as error:
                    status = "invalid_structure:" + str(error)
            success = parsed is not None
            row = {
                "asset_id": identity, "group": group, "frame_index": frame,
                "raw_frame_index": int(raw_frames[frame]), "selected_raw_frames": raw_frames.tolist(),
                "success": success, "status": status, "gt_joint_count": count,
                "predicted_tokens": generated["tokens"], "target_tokens": labels.tolist(),
                "predicted_structure_tokens": parsed["structure_tokens"].tolist() if success else [],
                "target_structure_tokens": truth["structure_tokens"].tolist(),
                "original_target_joints": gt_joints[frame].tolist(),
                "original_target_parents": gt_parents.tolist(),
                "predicted_joints": parsed["joints"].tolist() if success else None,
                "predicted_parents": parsed["parents"].tolist() if success else None,
                "exact_count": success and len(parsed["parents"]) == count,
                "exact_structure": success and np.array_equal(parsed["parents"], truth["parents"]),
                "original_edge_f1": float(_topology_metrics(parsed["joints"], parsed["parents"].tolist(),
                    gt_joints[frame], gt_parents.tolist())["edge_f1"]) if success else 0.,
                "j2j": _j2j(parsed["joints"], gt_joints[frame]) if success else None,
                "teacher_main_ce": float(output["main_loss"]),
                "teacher_structure_ce": float(output["structure_ce"]),
                "teacher_consistency_kl": float(output["consistency_kl"]),
                "eos_only_legal_choice": generated.get("eos_only_legal_choice", False),
            }
            key = (group, identity, frame)
            if reference is not None:
                row["reference_B65_f1"] = scores[key]
            rows.append(row)
    return rows


def evaluate_panel(model, tokenizer, codec, entries, config, output, *, samples, rank, world, final=False):
    selected_config = deepcopy(config)
    if final:
        selected_config["evaluation"] = config["final_evaluation"]
    panel, reference, manifest = load_small_reference_panel(selected_config, entries)
    root = Path(output) / (f"evaluation_full_{samples:06d}" if final else f"evaluation_{samples:06d}")
    root.mkdir(parents=True, exist_ok=True)
    local = {"complete": False, "rank": rank, "asset_exposures": samples, "groups": {}}
    was_training = model.training
    model.eval()
    try:
        for group, all_rows in panel.items():
            rows = all_rows[rank::world]
            dataset = fixed_dataset(rows, config, selected_config["evaluation"]["input_frames"],
                mode="heldout_eval" if group == "train_heldout" else "seen_eval")
            result = []
            for index in range(len(dataset)):
                surface, target, geometry = prepare_batch(dataset, [index], config, codec,
                                                          next(model.parameters()).device)
                chosen = set(geometry["selected_frames"][0].tolist())
                heldout = set(geometry["heldout_raw_ids"][0])
                if (group == "train_heldout" and chosen != heldout) or (group != "train_heldout" and chosen & heldout):
                    raise RuntimeError("evaluation frame partition changed")
                result.extend(evaluate_item(model, tokenizer, codec, surface, target, geometry,
                    frame_indices=selected_config["evaluation"]["frame_indices"], group=group,
                    reference=reference, max_new_tokens=selected_config["evaluation"]["max_new_tokens"]))
                del surface, target, geometry
            expected = {(row["asset_id"], frame) for row in rows
                        for frame in selected_config["evaluation"]["frame_indices"]}
            actual = [(row["asset_id"], row["frame_index"]) for row in result]
            if len(actual) != len(expected) or set(actual) != expected:
                raise RuntimeError("evaluation missed or duplicated an asset/frame")
            local["groups"][group] = {"rows": result, "summary": summarize(result)}
            write_json(root / f"rank_{rank}.json", local)
        local["complete"] = True
        write_json(root / f"rank_{rank}.json", local)
        if world > 1:
            dist.barrier()
        summary = None
        if rank == 0:
            reports = [json.loads((root / f"rank_{r}.json").read_text()) for r in range(world)]
            if not all(report["complete"] for report in reports):
                raise RuntimeError("incomplete distributed evaluation")
            summary = {"complete": True, "asset_exposures": samples, "panel": manifest,
                       "failed_generations_scored_zero": True, "full_validation": False,
                       "groups": {group: summarize([row for report in reports
                           for row in report["groups"][group]["rows"]]) for group in panel}}
            write_json(root / "summary.json", summary)
            compact = {group: {key: value for key, value in record.items() if key != "per_asset"}
                       for group, record in summary["groups"].items()}
            print("EVALUATION " + json.dumps({"samples": samples, "final_panel": final, "groups": compact}), flush=True)
        if world > 1:
            dist.barrier()
        return summary
    finally:
        model.train(was_training)
