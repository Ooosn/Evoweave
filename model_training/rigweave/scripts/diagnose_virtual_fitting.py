"""Bounded fitting diagnosis using the actual trial model, loss and optimizer."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from motion_experiment_runtime import guard, load_config, source_metadata, write_json
from train_virtual_frame import dataset_for, evaluation_rows, load_entries, optimizer_for, prepare
from virtual_frame_runtime import build_virtual_model, file_sha256, normalize_asset_gradients, set_exposure_lr
from rigweave.dynamic_rig.virtual_frame_rig import coordinate_regression_loss


@torch.no_grad()
def measure(model, surface, targets, xyz_weight, cached_reference=None):
    from eval_dynamic_rig_generation import _topology_metrics
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        anchors, memory = model.encode(surface)
        teacher = model.forward_encoded(anchors, memory, surface["query_points"], targets, xyz_weight=xyz_weight)
        shuffled = model.forward_encoded(anchors.flip(0), memory, surface["query_points"].flip(0), targets, xyz_weight=xyz_weight)
        wrong_memory = model.forward_encoded(anchors, memory.flip(0), surface["query_points"], targets, xyz_weight=xyz_weight)
        replay = {}
        if cached_reference is not None:
            fixed_xyz = model.positions(cached_reference[1], targets["joint_mask"], cached_reference[0], surface["query_points"])
            replay = {"cached_input_postupdate_l1": float(coordinate_regression_loss(fixed_xyz, targets["frame_target_joints"], targets["joint_mask"], targets["frame_mask"]).mean()),
                      "refreshed_anchor_meanabs_change": float((anchors-cached_reference[0]).abs().float().mean()),
                      "refreshed_H_meanabs_change": float((teacher["joint_features"]-cached_reference[1]).abs().float().mean())}
        model.positions.train()
        with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            training_xyz = model.positions(teacher["joint_features"].detach(), targets["joint_mask"], anchors.detach(), surface["query_points"])
            training_loss = coordinate_regression_loss(training_xyz, targets["frame_target_joints"], targets["joint_mask"], targets["frame_mask"]).mean()
            bias_gradient, = torch.autograd.grad(training_loss, model.positions.xyz_head.bias)
            valid = (targets["joint_mask"][:, None, :, None] & targets["frame_mask"][:, :, None, None]).expand_as(training_xyz)
            counts = valid.sum((1, 2, 3))
            expected = (torch.sign(training_xyz.detach().float() - targets["frame_target_joints"]) * valid
                        / counts[:, None, None, None] / training_xyz.shape[0]).sum((0, 1, 2))
            delta = (training_xyz.detach().float() - teacher["joints"].float()).abs()[valid]
            consistency = {"same_weight_train_eval_xyz_meanabs": float(delta.mean()),
                           "same_weight_train_eval_xyz_maxabs": float(delta.max()),
                           "last_bias_gradient_actual": bias_gradient.detach().float().cpu().tolist(),
                           "last_bias_gradient_expected": expected.cpu().tolist(),
                           "last_bias_gradient_maxabs_error": float((bias_gradient.float() - expected).abs().max())}
            del training_xyz, training_loss, bias_gradient
        model.positions.eval()
        rows = []
        for i in range(len(targets["asset_id"])):
            count = int(targets["joint_count"][i])
            gt = targets["frame_target_joints"][i, :, :count].cpu().numpy()
            parents = targets["target_parents"][i, :count].tolist()
            xyz = teacher["joints"][i, :, :count].float().cpu().numpy()
            generated = model.generate_encoded(anchors[i:i+1], memory[i:i+1], surface["query_points"][i:i+1])
            pred = generated["joints"][0].float().cpu().numpy() if generated["success"] else None
            rows.append({"asset_id": targets["asset_id"][i], "parents": parents, "target": gt,
                         "teacher_xyz": xyz, "generated_xyz": pred, "generated_parents": generated.get("parents"),
                         "success": generated["success"], "teacher_F1": _topology_metrics(xyz[0], parents, gt[0], parents)["edge_f1"],
                         "generation_F1": _topology_metrics(pred[0], generated["parents"], gt[0], parents)["edge_f1"] if pred is not None else 0.})
    summary = {"structure_ce": float(teacher["structure_loss"]), "xyz_l1": float(teacher["xyz_loss"]),
               "query_xyz_l1": float(np.mean([np.abs(row["teacher_xyz"][0] - row["target"][0]).mean() for row in rows])),
               "teacher_F1": float(np.mean([row["teacher_F1"] for row in rows])),
               "generation_F1": float(np.mean([row["generation_F1"] for row in rows])),
               "wrong_asset_anchors_xyz_l1": float(shuffled["xyz_loss"]),
               "wrong_asset_structure_memory_xyz_l1": float(wrong_memory["xyz_loss"]), **consistency, **replay}
    return summary, rows, (anchors.detach(), teacher["joint_features"].detach())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    plan = load_config(args.config)
    config = load_config(plan["virtual_config"])
    guard(config, "preflight")
    state = json.loads((Path(config["runtime"]["repo"]) / "model_training/state/current.json").read_text())
    if state["active_operation"]["label"] != plan["label"] or state["active_operation"]["config_sha256"] != file_sha256(args.config):
        raise ValueError("fitting diagnostic differs from recorded operation")
    source = source_metadata(config)
    rank = int(os.environ.get("LOCAL_RANK", 0))
    mode = plan["rank_modes"][rank]
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.9, device)
    root = Path(plan["output"])
    root.mkdir(exist_ok=True, parents=True)
    report_path = root / f"{mode}.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    if file_sha256(plan["checkpoint"]) != plan["checkpoint_sha256"]:
        raise ValueError("new trial checkpoint changed")
    model, initialization = build_virtual_model(config, "self_cross_2", device)
    optimizer = optimizer_for(model, config)
    payload = torch.load(plan["checkpoint"], map_location="cpu", mmap=True, weights_only=False)
    assert payload["samples"] == 5040 and payload["step"] == 105
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    del payload
    selected = evaluation_rows(load_entries(config), config)["train"]
    positions = [next(i for i, row in enumerate(selected) if row["manifest_index"] == index) for index in plan["manifest_indices"]]
    seen = set(torch.randperm(config["data"]["manifests"]["train"]["rows"], generator=torch.Generator().manual_seed(0))[:5040].tolist())
    if not set(plan["manifest_indices"]).issubset(seen):
        raise ValueError("diagnostic assets must actually have been exposed during the trial")
    dataset = dataset_for(selected, config, mode="seen_eval")
    surface, targets = prepare(dataset, positions, config, device)
    xyz_weight = config["optimization"]["xyz_weight"]
    initial, initial_rows, cached = measure(model, surface, targets, xyz_weight)
    report = {"complete": False, "mode": mode, "source": source, "checkpoint": plan["checkpoint"],
              "initialization": initialization, "initial": initial, "updates": [], "evaluations": {"0": initial},
              "assets": targets["asset_id"], "selected_frames": targets["selected_frames"].cpu().tolist(),
              "started_at": datetime.now(timezone.utc).isoformat(), "diagnostic_only_no_formal_weight_reuse": True}
    write_json(report_path, report)
    if mode != "end_to_end":
        model.requires_grad_(False)
        model.positions.requires_grad_(True)
    model.train()
    size = len(positions)
    print("FIT_INITIAL " + json.dumps({"mode": mode, **initial}), flush=True)
    for step in range(1, plan["updates"] + 1):
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        lr, _ = set_exposure_lr(optimizer, 5040 + (step - 1) * size, config)
        multiplier = plan.get("diagnostic_lr_multipliers", {}).get(mode, 1.)
        lr *= multiplier
        for group in optimizer.param_groups:
            group["lr"] = lr
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if mode == "end_to_end":
                output = model(surface, targets, xyz_weight=xyz_weight)
                per_asset = output["loss_per_asset"]
                xyz_loss = output["xyz_loss"]
                structure_loss = output["structure_loss"]
            else:
                xyz = model.positions(cached[1], targets["joint_mask"], cached[0], surface["query_points"])
                per_xyz = coordinate_regression_loss(xyz, targets["frame_target_joints"], targets["joint_mask"], targets["frame_mask"])
                xyz_loss = per_xyz.mean()
                structure_loss = xyz_loss.new_tensor(initial["structure_ce"])
                if plan.get("diagnostic_loss", {}).get(mode) == "smooth_l1":
                    valid = (targets["joint_mask"][:, None, :, None] & targets["frame_mask"][:, :, None, None]).expand_as(xyz)
                    smooth = torch.nn.functional.smooth_l1_loss(xyz.float(), targets["frame_target_joints"], reduction="none", beta=plan["smooth_l1_beta"])
                    per_asset = xyz_weight * (smooth * valid).sum((1, 2, 3)) / valid.sum((1, 2, 3))
                else:
                    per_asset = xyz_weight * per_xyz
            if not bool(torch.isfinite(per_asset).all()):
                raise ValueError("nonfinite diagnostic loss")
            per_asset.sum().backward()
        normalize_asset_gradients(model.parameters(), global_assets=size, world_size=1)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        row = {"step": step, "xyz_l1": float(xyz_loss.detach()), "structure_ce": float(structure_loss.detach()),
               "gradient_norm": float(gradient_norm), "lr": lr, "seconds": time.monotonic()-started}
        report["updates"].append(row)
        if step in plan["evaluation_updates"]:
            measured, records, _ = measure(model, surface, targets, xyz_weight, cached if mode != "end_to_end" else None)
            report["evaluations"][str(step)] = measured
            torch.save({"initial": initial_rows, "current": records}, root / f"{mode}_predictions_step{step}.pt")
            print("FIT_EVAL " + json.dumps({"mode": mode, "step": step, **measured}), flush=True)
            model.train()
        write_json(report_path, report)
        print("FIT " + json.dumps({"mode": mode, **row}), flush=True)
        del per_asset, xyz_loss, structure_loss
        if mode == "end_to_end":
            del output
        else:
            del xyz, per_xyz
    report.update(complete=True, finished_at=datetime.now(timezone.utc).isoformat())
    write_json(report_path, report)


if __name__ == "__main__":
    main()
