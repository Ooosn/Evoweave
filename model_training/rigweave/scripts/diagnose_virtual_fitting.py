"""Bounded fitting diagnosis using the actual trial model, loss and optimizer."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
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
from virtual_frame_runtime import build_virtual_model, file_sha256, finite_gradient_groups, normalize_asset_gradients, set_exposure_lr
from rigweave.dynamic_rig.virtual_frame_rig import coordinate_regression_loss


def single_asset_inputs(surface, targets, mode):
    if mode not in {"single_static", "single_dynamic"} or len(targets["asset_id"]) != 1:
        raise ValueError("single-asset diagnosis requires exactly one asset and a known mode")
    surface, targets = dict(surface), dict(targets)
    if mode == "single_static":
        for key in ("dense_points", "dense_normals", "query_points", "query_normals"):
            surface[key] = surface[key][:, :1].expand_as(surface[key]).clone()
        for key in ("frame_target_joints", "selected_frames", "frame_mesh_bbox_side"):
            targets[key] = targets[key][:, :1].expand_as(targets[key]).clone()
        targets["target_joints"] = targets["frame_target_joints"][:, 0]
    return surface, targets


def input_digest(surface, targets):
    digest = hashlib.sha256()
    for prefix, mapping in (("surface", surface), ("targets", targets)):
        for key, value in sorted(mapping.items()):
            if isinstance(value, torch.Tensor):
                digest.update(f"{prefix}.{key}:{tuple(value.shape)}:{value.dtype}".encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def fitting_passed(measured, threshold):
    return (measured["xyz_l1"] <= threshold and measured["query_xyz_l1"] <= threshold
            and measured["structure_ce"] <= .01 and measured["all_generated_parents_exact"]
            and measured["generation_F1"] >= measured["oracle_query_F1"] - .05)


def check_backward_paths(model, surface, targets, xyz_weight):
    """Compare every trainable gradient at identical weights without Adam steps."""
    original_config = model.config
    original_positions = model.positions.config
    original_motion = model.motion_encoder.gradient_checkpointing
    original_backbone = model.structure.backbone.is_gradient_checkpointing
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()

    def backward(inside_autocast):
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        model.zero_grad(set_to_none=True)
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(surface, targets, xyz_weight=xyz_weight)
            loss = output["loss_per_asset"].sum()
            if inside_autocast:
                loss.backward()
        if not inside_autocast:
            loss.backward()
        return float(loss.detach()), output["joints"].detach().float().cpu()

    reference_loss, reference_xyz = backward(False)
    reference = {name: parameter.grad.detach().cpu() for name, parameter in model.named_parameters()
                 if parameter.requires_grad and parameter.grad is not None}
    results = {}
    try:
        for label, inside in (("backward_inside_autocast", True), ("without_checkpointing", False)):
            if label == "without_checkpointing":
                model.config = replace(original_config, surface_checkpointing=False)
                model.positions.config = replace(original_positions, position_checkpointing=False)
                model.motion_encoder.gradient_checkpointing = False
                model.structure.backbone.gradient_checkpointing_disable()
            loss, xyz = backward(inside)
            groups = {}
            actual_names = set()
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad or parameter.grad is None:
                    continue
                actual_names.add(name)
                if name not in reference:
                    raise RuntimeError(f"gradient presence changed for {name}")
                left, right = reference[name].double(), parameter.grad.detach().cpu().double()
                if not bool(torch.isfinite(right).all()):
                    raise RuntimeError(f"nonfinite comparison gradient for {name}")
                delta = right-left
                group = groups.setdefault(name.split('.')[0], dict(tensors=0, reference_squared=0., actual_squared=0.,
                                                                    difference_squared=0., dot=0., maxabs=0.))
                group['tensors'] += 1
                group['reference_squared'] += float(left.square().sum())
                group['actual_squared'] += float(right.square().sum())
                group['difference_squared'] += float(delta.square().sum())
                group['dot'] += float((left*right).sum())
                group['maxabs'] = max(group['maxabs'], float(delta.abs().max()))
            if actual_names != set(reference):
                raise RuntimeError("comparison lost trainable gradients")
            for group in groups.values():
                group['relative_l2'] = (group['difference_squared']/max(group['reference_squared'], 1e-30))**.5
                group['cosine'] = group['dot']/max((group['reference_squared']*group['actual_squared'])**.5, 1e-30)
            results[label] = dict(reference_loss=reference_loss, actual_loss=loss, groups=groups,
                                  forward_xyz_maxabs=float((xyz-reference_xyz).abs().max()),
                                  forward_xyz_meanabs=float((xyz-reference_xyz).abs().mean()))
            print('BACKWARD_CHECK '+json.dumps({"mode": label, **results[label]}), flush=True)
            if any(x['relative_l2'] > .1 for x in groups.values()):
                raise RuntimeError("backward paths differ materially; stop fitting and inspect evidence")
    finally:
        model.config = original_config
        model.positions.config = original_positions
        model.motion_encoder.gradient_checkpointing = original_motion
        if original_backbone:
            model.structure.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
    return results


@torch.no_grad()
def measure(model, surface, targets, xyz_weight, cached_reference=None, full_path_check=False):
    from eval_dynamic_rig_generation import _topology_metrics
    full_train = None
    if full_path_check:
        model.train()
        with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            output = model(surface, targets, xyz_weight=xyz_weight)
            if not output["loss"].requires_grad:
                raise RuntimeError("training loss has no gradient graph")
            full_train = output["joints"].detach()
            del output
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
                         "oracle_query_F1": _topology_metrics(gt[0], parents, gt[0], parents)["edge_f1"],
                         "generated_parents_exact": generated.get("parents") == parents,
                         "frame_xyz_l1": np.abs(xyz-gt).mean((1, 2)).tolist(),
                         "generation_F1": _topology_metrics(pred[0], generated["parents"], gt[0], parents)["edge_f1"] if pred is not None else 0.})
    summary = {"structure_ce": float(teacher["structure_loss"]), "xyz_l1": float(teacher["xyz_loss"]),
               "query_xyz_l1": float(np.mean([np.abs(row["teacher_xyz"][0] - row["target"][0]).mean() for row in rows])),
               "teacher_F1": float(np.mean([row["teacher_F1"] for row in rows])),
               "generation_F1": float(np.mean([row["generation_F1"] for row in rows])),
               "oracle_query_F1": float(np.mean([row["oracle_query_F1"] for row in rows])),
               "all_generated_parents_exact": all(row["generated_parents_exact"] for row in rows),
               "per_asset_frame_xyz_l1": [row["frame_xyz_l1"] for row in rows],
               "wrong_asset_anchors_xyz_l1": float(shuffled["xyz_loss"]),
               "wrong_asset_structure_memory_xyz_l1": float(wrong_memory["xyz_loss"]), **consistency, **replay}
    if full_train is not None:
        difference = (full_train.float() - teacher["joints"].float()).abs()
        summary.update(full_train_eval_xyz_meanabs=float(difference.mean()),
                       full_train_eval_xyz_maxabs=float(difference.max()))
    return summary, rows, (anchors.detach(), teacher["joint_features"].detach())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    plan = load_config(args.config)
    config = load_config(plan["virtual_config"])
    guard(config, plan.get("operation", "preflight"))
    state = json.loads((Path(config["runtime"]["repo"]) / "model_training/state/current.json").read_text())
    if state["active_operation"]["label"] != plan["label"] or state["active_operation"]["config_sha256"] != file_sha256(args.config):
        raise ValueError("fitting diagnostic differs from recorded operation")
    source = source_metadata(config)
    rank = int(os.environ.get("LOCAL_RANK", 0))
    mode = plan["rank_modes"][rank]
    single = mode in {"single_static", "single_dynamic"}
    end_to_end = single or mode == "end_to_end"
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
    if not plan.get("reset_optimizer", False):
        optimizer.load_state_dict(payload["optimizer"])
    del payload
    selected = evaluation_rows(load_entries(config), config)["train"]
    positions = [next(i for i, row in enumerate(selected) if row["manifest_index"] == index) for index in plan["manifest_indices"]]
    seen = set(torch.randperm(config["data"]["manifests"]["train"]["rows"], generator=torch.Generator().manual_seed(0))[:5040].tolist())
    if not set(plan["manifest_indices"]).issubset(seen):
        raise ValueError("diagnostic assets must actually have been exposed during the trial")
    dataset = dataset_for(selected, config, mode="seen_eval")
    surface, targets = prepare(dataset, positions, config, device)
    original_frames = targets["selected_frames"].cpu().tolist()
    if single:
        surface, targets = single_asset_inputs(surface, targets, mode)
    frozen_digest = input_digest(surface, targets)
    xyz_weight = config["optimization"]["xyz_weight"]
    backward_checks = check_backward_paths(model, surface, targets, xyz_weight) if plan.get("check_backward_paths") else None
    initial, initial_rows, cached = measure(model, surface, targets, xyz_weight, full_path_check=single)
    report = {"complete": False, "mode": mode, "source": source, "checkpoint": plan["checkpoint"],
              "initialization": initialization, "initial": initial, "updates": [], "evaluations": {"0": initial},
              "assets": targets["asset_id"], "selected_frames": targets["selected_frames"].cpu().tolist(),
              "original_selected_frames": original_frames, "input_sha256": frozen_digest,
              "optimizer_reset": plan.get("reset_optimizer", False),
              "backward_checks": backward_checks, "diagnostic_schedule": plan.get("diagnostic_schedule", "unchanged"),
              "started_at": datetime.now(timezone.utc).isoformat(), "diagnostic_only_no_formal_weight_reuse": True}
    write_json(report_path, report)
    if not end_to_end:
        model.requires_grad_(False)
        model.positions.requires_grad_(True)
    model.train()
    size = len(positions)
    stable_passes = 0
    print("FIT_INITIAL " + json.dumps({"mode": mode, **initial}), flush=True)
    for step in range(1, plan["updates"] + 1):
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        if plan.get("diagnostic_schedule") == "original_onecycle_by_update":
            schedule = {"optimization": {**config["optimization"], "schedule_samples": plan["updates"]}}
            lr, _ = set_exposure_lr(optimizer, step-1, schedule)
        elif "constant_lr" in plan:
            lr = plan["constant_lr"]
        else:
            lr, _ = set_exposure_lr(optimizer, 5040 + (step - 1) * size, config)
        multiplier = plan.get("diagnostic_lr_multipliers", {}).get(mode, 1.)
        lr *= multiplier
        for group in optimizer.param_groups:
            group["lr"] = lr
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if end_to_end:
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
        if single and step == 1:
            report["first_gradient_groups"] = finite_gradient_groups(model)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        row = {"step": step, "xyz_l1": float(xyz_loss.detach()), "structure_ce": float(structure_loss.detach()),
               "gradient_norm": float(gradient_norm), "lr": lr, "beta1": optimizer.param_groups[0]['betas'][0],
               "seconds": time.monotonic()-started}
        report["updates"].append(row)
        if step in plan["evaluation_updates"]:
            measured, records, _ = measure(model, surface, targets, xyz_weight, cached if not end_to_end else None,
                                          full_path_check=single)
            if input_digest(surface, targets) != frozen_digest:
                raise RuntimeError("fixed training inputs or labels mutated")
            if single:
                passed = fitting_passed(measured, plan["acceptance_xyz_l1"])
                stable_passes = stable_passes + 1 if passed else 0
                measured.update(fitting_threshold_passed=passed, consecutive_passes=stable_passes)
            report["evaluations"][str(step)] = measured
            torch.save({"initial": initial_rows, "current": records}, root / f"{mode}_predictions_step{step}.pt")
            print("FIT_EVAL " + json.dumps({"mode": mode, "step": step, **measured}), flush=True)
            model.train()
        write_json(report_path, report)
        print("FIT " + json.dumps({"mode": mode, **row}), flush=True)
        del per_asset, xyz_loss, structure_loss
        if end_to_end:
            del output
        else:
            del xyz, per_xyz
        if single and stable_passes >= 2:
            break
    if plan.get("save_final_checkpoint"):
        path = root / f"{mode}_diagnostic_final.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "updates": step,
                    "plan": plan, "source": source, "diagnostic_only_no_formal_weight_reuse": True}, path)
        report['diagnostic_checkpoint'] = {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
    report.update(complete=True, finished_at=datetime.now(timezone.utc).isoformat(),
                  final_input_sha256=input_digest(surface, targets))
    if single:
        report.update(fitting_accepted=stable_passes >= 2,
                      stop_reason="two_consecutive_fit_checks" if stable_passes >= 2 else "update_limit")
    write_json(report_path, report)


if __name__ == "__main__":
    main()
