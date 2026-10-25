#!/usr/bin/env python3
"""Frozen-checkpoint role diagnostics; model/data imports use the trained checkout."""
from collections import Counter
from datetime import timedelta
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import traceback

from query_diagnostics_guard import ROOT, check_deadline, file_sha256, verify_launch


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def checked_json(spec):
    if file_sha256(spec["path"]) != spec["sha256"]:
        raise ValueError(f"pinned input changed: {spec['path']}")
    return json.loads(Path(spec["path"]).read_text())


def trained_imports(config):
    frozen = config["trained_source"]
    root = Path(frozen["repo"])
    if (subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() != frozen["commit"]
            or subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip()):
        raise RuntimeError("the trained source is no longer the frozen clean checkout")
    if file_sha256(frozen["config"]) != frozen["config_sha256"]:
        raise RuntimeError("the immutable training configuration changed")
    for path in (root / "model_training/rigweave/scripts", root / "model_training/rigweave/src"):
        sys.path.insert(0, str(path))
    import id_xyz_runtime as runtime
    import rigweave.dynamic_rig.id_xyz_consistency as model_module
    if not Path(runtime.__file__).resolve().is_relative_to(root) or not Path(model_module.__file__).resolve().is_relative_to(root):
        raise RuntimeError("model/data runtime resolved outside the trained checkout")
    name = "rigweave.dynamic_rig.id_xyz_diagnostics"
    path = ROOT / "model_training/rigweave/src/rigweave/dynamic_rig/id_xyz_diagnostics.py"
    spec = importlib.util.spec_from_file_location(name, path)
    diagnostics = importlib.util.module_from_spec(spec)
    sys.modules[name] = diagnostics
    spec.loader.exec_module(diagnostics)
    return runtime, diagnostics, json.loads(Path(frozen["config"]).read_text())


def references(config):
    result = {}
    for spec in config["reference"].values():
        report = checked_json(spec)
        if report.get("complete") is not True or report["asset_exposures"] != 100000:
            raise ValueError("reference is not the completed ID100k small panel")
        for group, block in report["groups"].items():
            for row in block["rows"]:
                key = (group, row["asset_id"], row["frame_index"])
                if key in result:
                    raise ValueError("duplicated reference view")
                result[key] = row
    if len(result) != 128:
        raise ValueError("the ID100k reference must contain128 views")
    return result


def restore(config, trained, runtime, device):
    import torch
    checkpoint = config["checkpoint"]
    path = Path(checkpoint["path"])
    stat = path.stat()
    if stat.st_size != checkpoint["bytes"] or file_sha256(path) != checkpoint["sha256"]:
        raise ValueError("the completed checkpoint changed")
    saved = torch.load(path, mmap=True, map_location="cpu", weights_only=False)
    if (saved["asset_exposures"] != checkpoint["asset_exposures"] or saved["step"] != checkpoint["step"]
            or saved["configuration"] != trained or saved["source_sha256"] != runtime.source_hashes()
            or saved["code_commit"] != config["trained_source"]["commit"]):
        raise ValueError("checkpoint/configuration/model source identity differs")
    model, tokenizer, codec, _ = runtime.build_model(trained, device)
    report = model.load_state_dict(saved["model"], strict=True)
    actual = model.state_dict()
    if report.missing_keys or report.unexpected_keys or any(
            actual[key].dtype != value.dtype or actual[key].shape != value.shape
            or not torch.equal(actual[key].detach().cpu(), value) for key, value in saved["model"].items()):
        raise RuntimeError("strict restoration did not preserve every saved tensor")
    from probe_virtual_query_lambda import parameter_fingerprint
    fingerprint = parameter_fingerprint(model)
    if fingerprint != checkpoint["model_sha256"]:
        raise RuntimeError("restored model fingerprint differs from the completed training receipt")
    result = {"all_saved_tensors_equal": True, "tensors": len(actual), "model_sha256": fingerprint,
        "checkpoint_sha256": checkpoint["sha256"], "checkpoint_stat": [stat.st_size, stat.st_mtime_ns],
        "trained_source_hashes_equal": True, "optimizer_created": False, "backward_performed": False}
    del actual, saved
    model.eval().requires_grad_(False)
    return model, tokenizer, codec, result


def verify_geometry(geometry, targets, group, reference):
    import numpy as np
    identity = geometry["asset_id"][0]
    raw = geometry["selected_frames"][0].cpu().tolist()
    count = int(geometry["joint_count"][0])
    frames = {}
    for slot in (0, 2, 4, 6):
        row = reference[group, identity, slot]
        ids = targets["input_ids"][0, slot][targets["attention_mask"][0, slot]].cpu().tolist()
        if (row["selected_raw_frames"] != raw or row["target_tokens"] != ids
                or row["gt_joint_count"] != count
                or row["original_target_parents"] != geometry["target_parents"][0, :count].cpu().tolist()
                or not np.array_equal(np.asarray(row["original_target_joints"], dtype=np.float32),
                    geometry["frame_target_joints"][0, slot, :count].cpu().numpy())):
            raise ValueError(f"original frame/GT identity changed: {group}/{identity}/{slot}")
        frames[slot] = ids
    return frames


def structural_divergence(codec, predicted, target):
    left = codec.decode(predicted)["structure_tokens"].tolist()
    right = codec.decode(target)["structure_tokens"].tolist()
    index = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))
    if index == len(left) == len(right):
        return {"equal": True, "index": None, "role": None}
    token = right[index] if index < len(right) else None
    role = ("extra_structure" if token is None else "eos" if token == codec.eos else
            "branch" if token == codec.branch else "parent_id" if index and right[index-1] == codec.branch
            else "new_joint_id")
    return {"equal": False, "index": index, "role": role,
        "generated_token": left[index] if index < len(left) else None, "GT_token": token,
        "positional_only": True}


def gt_structure_free_xyz(model, codec, condition, target, roles):
    """Oracle intervention: force every GT structural decision, predict only XYZ."""
    import torch
    ids = torch.tensor([[codec.bos, codec.class_token]], device=condition.device)
    embedding = model.xyz_decoder.get_input_embeddings()(ids)
    mask = torch.ones((1, condition.shape[1] + 2), dtype=torch.long, device=condition.device)
    output = model.xyz_decoder(inputs_embeds=torch.cat((condition.to(embedding.dtype), embedding), dim=1),
        attention_mask=mask, use_cache=True, return_dict=True)
    for index in range(2, len(target)):
        scores = output.logits[0, -1].float()
        if not bool(torch.isfinite(scores).all()):
            raise RuntimeError("nonfinite oracle logits")
        allowed = codec.allowed_tokens(ids[0].tolist())
        if roles[index] in ("x", "y", "z"):
            if set(allowed) != set(codec.coordinate_ids):
                raise RuntimeError("oracle coordinate position is not a native XYZ decision")
            candidates = torch.as_tensor(allowed, device=scores.device)
            token = int(candidates[scores[candidates].argmax()])
        else:
            token = target[index]
            if token not in allowed:
                raise RuntimeError("forced GT structure is not legal under the oracle prefix")
        ids = torch.cat((ids, ids.new_tensor([[token]])), dim=1)
        if token == codec.eos:
            if index != len(target) - 1:
                raise RuntimeError("oracle stopped before the complete GT structure")
            return ids[0].tolist()
        if output.past_key_values is None:
            raise RuntimeError("oracle decoder lost its KV cache")
        mask = torch.cat((mask, mask.new_ones((1, 1))), dim=1)
        output = model.xyz_decoder(input_ids=ids[:, -1:], attention_mask=mask,
            past_key_values=output.past_key_values, use_cache=True, return_dict=True)
    raise RuntimeError("GT oracle stream did not terminate")


def summarize(rows):
    def full_mismatch_label(value):
        if value.get("status") == "match":
            return "equal"
        return value.get("role") or value.get("status", "unknown")

    def one(selected):
        if not selected:
            return {"views": 0}
        result = {"views": len(selected), "assets": len({r["asset_id"] for r in selected}),
            "free_original_F1": statistics.mean(r["original_edge_f1"] for r in selected),
            "GT_structure_oracle_original_F1": statistics.mean(r["oracle_original_edge_f1"] for r in selected),
            "GT_roundtrip_original_F1": statistics.mean(r["GT_roundtrip_original_edge_f1"] for r in selected),
            "first_full_mismatch_roles": dict(Counter(full_mismatch_label(r["free_first_divergence"]) for r in selected)),
            "first_structure_mismatch_roles": dict(Counter(r["free_structure_divergence"].get("role") or "equal" for r in selected)),
            "all_saved_generation_tokens_equal": all(r["saved_generation_tokens_equal"] for r in selected),
            "oracle_not_model_quality": True, "roles": {}}
        for role in selected[0]["teacher"]["per_role"]:
            values = [r["teacher"]["per_role"][role] for r in selected]
            count = sum(v["full_vocab_count"] for v in values)
            legal = [v["legal_choice"] for v in values]
            nontrivial = sum(v["nontrivial_choice_count"] for v in legal)
            legal_correct = sum(v["top1_correct"] for v in legal)
            wrong = [r["other_pose_teacher"]["per_role"][role] for r in selected]
            result["roles"][role] = {"tokens": count,
                "teacher_NLL": sum(v["full_vocab_nll_sum"] for v in values) / count if count else None,
                "teacher_accuracy": sum(v["full_vocab_top1_correct"] for v in values) / count if count else None,
                "legal_accuracy": legal_correct / count if count else None,
                "nontrivial_choices": nontrivial,
                "nontrivial_legal_accuracy": (legal_correct - count + nontrivial) / nontrivial if nontrivial else None,
                "other_pose_teacher_NLL": sum(v["full_vocab_nll_sum"] for v in wrong) / count if count else None,
                "token_weighted": True}
        return result
    return {group: {"all": one([r for r in rows if r["group"] == group]),
        "non52": one([r for r in rows if r["group"] == group and r["gt_joint_count"] != 52]),
        "gt52": one([r for r in rows if r["group"] == group and r["gt_joint_count"] == 52])}
        for group in sorted({r["group"] for r in rows})}


def run(args, config, source):
    import numpy as np
    import torch
    import torch.distributed as dist
    runtime, diagnostics, trained = trained_imports(config)
    from rigweave.dynamic_rig.virtual_frame_data import collate_virtual_targets, prepare_virtual_surface
    from rigweave.dynamic_rig.skeleton_topology_metrics import _topology_metrics
    from structure_xyz_fill_runtime import load_small_reference_panel
    from probe_virtual_query_lambda import parameter_fingerprint
    torch.set_num_threads(1)
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    cpu = args.mode == "audit"
    if (cpu and (rank, world) != (0, 1)) or (not cpu and (world != 2 or rank not in (0, 1))):
        raise RuntimeError("diagnostic rank topology differs")
    if not cpu:
        torch.cuda.set_device(rank)
        torch.cuda.set_per_process_memory_fraction(.85, rank)
        dist.init_process_group("nccl", timeout=timedelta(minutes=45))
    device = torch.device("cpu" if cpu else f"cuda:{rank}")
    model, tokenizer, codec, restored = restore(config, trained, runtime, device)
    reference = references(config)
    entries = runtime.read_entries(trained)
    panel, _, manifest = load_small_reference_panel(trained, entries)
    if manifest["panel_sha256"] != "da1b4eca4df249fa14f03af2b78d0b075b8fac23eec2b60141593be3f98fbe6f":
        raise RuntimeError("the matched small panel changed")
    work = []
    for group in config["diagnostics"]["groups"]:
        for shard in (range(2) if cpu else (rank,)):
            subset = panel[group][shard::2]
            dataset = runtime.fixed_dataset(subset, trained, 8,
                mode="heldout_eval" if group == "train_heldout" else "seen_eval")
            for index in range(len(subset)):
                if args.mode == "smoke" and (group != "valid" or index != 0):
                    continue
                work.append((group, dataset, index))
    args.output.mkdir(parents=True, exist_ok=True)
    rows, audited = [], []
    with torch.inference_mode():
        for group, dataset, index in work:
            check_deadline(config)
            batch = dataset[index]
            geometry = collate_virtual_targets([batch], "cpu")
            targets = runtime.build_targets(geometry, codec, "cpu")
            frames = verify_geometry(geometry, targets, group, reference)
            audited.append({"group": group, "asset_id": batch["asset_id"],
                "selected_raw_frames": geometry["selected_frames"][0].tolist(), "GT_matches": True})
            if cpu:
                for ids in frames.values():
                    diagnostics.classify_gt_roles(codec, ids)
                continue
            surface, gpu_geometry = prepare_virtual_surface([batch], device, **trained["surface_sampling"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                condition = model.encode(surface)
            del surface, gpu_geometry
            for slot, ids in frames.items():
                ref = reference[group, batch["asset_id"], slot]
                gt = torch.tensor([ids], device=device)
                attention = torch.ones_like(gt, dtype=torch.bool)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model.logits(condition[:, slot], gt[:, :-1], attention[:, :-1])[0].float().cpu()
                    other = model.logits(condition[:, (slot + 2) % 8], gt[:, :-1], attention[:, :-1])[0].float().cpu()
                    generated = model.generate(condition[:, slot], codec, max_new_tokens=2048)
                    oracle_ids = gt_structure_free_xyz(model, codec, condition[:, slot], ids,
                        diagnostics.classify_gt_roles(codec, ids))
                matches = generated["tokens"] == ref["predicted_tokens"]
                if not matches:
                    write(args.output / f"replay_mismatch_rank_{rank}.json", {"group": group,
                        "asset_id": batch["asset_id"], "slot": slot,
                        "actual": generated, "reference": ref})
                    raise RuntimeError("free generation did not reproduce the frozen reference tokens")
                parsed, truth = codec.decode(oracle_ids), codec.decode(ids)
                if not np.array_equal(parsed["parents"], truth["parents"]):
                    raise RuntimeError("GT-structure oracle changed the GT parent identities")
                target_xyz = np.asarray(ref["original_target_joints"], dtype=np.float32)
                parents = ref["original_target_parents"]
                metric = lambda output: float(_topology_metrics(output["joints"], output["parents"].tolist(),
                    target_xyz, parents)["edge_f1"])
                row = {"group": group, "asset_id": batch["asset_id"], "frame_index": slot,
                    "gt_joint_count": ref["gt_joint_count"], "raw_frame_index": ref["raw_frame_index"],
                    "saved_generation_tokens_equal": matches, "original_edge_f1": ref["original_edge_f1"],
                    "teacher": diagnostics.teacher_forced_diagnostics(codec, ids, logits, legal_choice=True),
                    "other_pose_teacher": diagnostics.teacher_forced_diagnostics(codec, ids, other),
                    "other_pose_slot": (slot + 2) % 8,
                    "free_first_divergence": diagnostics.first_divergence_summary(codec, generated["tokens"], ids),
                    "free_structure_divergence": structural_divergence(codec, generated["tokens"], ids),
                    "oracle_original_edge_f1": metric(parsed), "GT_roundtrip_original_edge_f1": metric(truth),
                    "oracle_tokens": oracle_ids, "oracle_not_model_quality": True}
                rows.append(row)
                del logits, other, gt, attention
            write(args.output / f"progress_rank_{rank}.json", {"assets_complete": len(audited), "views": len(rows)})
            print(json.dumps({"rank": rank, "group": group, "asset_complete": batch["asset_id"]}), flush=True)
            del condition, batch, targets, geometry
    if parameter_fingerprint(model) != restored["model_sha256"]:
        raise RuntimeError("inference changed model weights")
    stat = Path(config["checkpoint"]["path"]).stat()
    if [stat.st_size, stat.st_mtime_ns] != restored["checkpoint_stat"]:
        raise RuntimeError("source checkpoint file changed during diagnostics")
    references(config)
    if runtime.source_hashes() != torch.load(config["checkpoint"]["path"], mmap=True,
            map_location="cpu", weights_only=False)["source_sha256"]:
        raise RuntimeError("trained source changed during diagnostics")
    local = {"complete": True, "mode": args.mode, "source": source, "rank": rank,
        "restoration": restored, "audited_assets": audited, "rows": rows,
        "model_unchanged": True, "checkpoint_stat_unchanged": True,
        "cuda_initialized": torch.cuda.is_initialized(), "optimizer_created": False, "backward_performed": False}
    if cpu and local["cuda_initialized"]:
        raise RuntimeError("CPU audit unexpectedly initialized CUDA")
    write(args.output / f"rank_{rank}.json", local)
    if not cpu:
        dist.barrier()
    if rank == 0:
        reports = [local] if cpu else [json.loads((args.output / f"rank_{r}.json").read_text()) for r in range(2)]
        combined = [row for report in reports for row in report["rows"]]
        expected = 0 if cpu else 8 if args.mode == "smoke" else 128
        keys = {(row["group"], row["asset_id"], row["frame_index"]) for row in combined}
        if len(combined) != expected or len(keys) != expected:
            raise RuntimeError("missing or duplicated diagnostic views")
        summary = {"complete": True, "mode": args.mode, "views": len(combined),
            "audited_group_assets": sum(len(report["audited_assets"]) for report in reports),
            "trained_source": config["trained_source"], "checkpoint": config["checkpoint"],
            "restoration": restored, "groups": summarize(combined), "optimizer_created": False,
            "backward_performed": False, "model_unchanged": True, "quality_approval": False,
            "oracle_is_diagnostic_not_model_quality": True,
            "B65_ablation_status": "blocked_missing_deleted_checkpoint_not_substituted"}
        write(args.output / "summary.json", summary)
    if not cpu:
        dist.barrier()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("audit", "smoke", "diagnose"), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    source = verify_launch(args, config)
    try:
        run(args, config, source)
    except BaseException as error:
        write(args.output / f"failure_rank_{os.environ.get('RANK', '0')}.json",
            {"complete": True, "passed": False, "error": repr(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
