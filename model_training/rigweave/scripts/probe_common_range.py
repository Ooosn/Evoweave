"""Read-only paired old-model inference on fixed common-coordinate inputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from prepare_common_range_probe import sha256


def fixed_sample(dataset, index, case, gauge):
    from rigweave.dynamic_rig import data as data_module
    selected = np.asarray(case["selected_frames"], dtype=np.int64)

    def select(vertices, joints, tails, **kwargs):
        if kwargs["frame_count"] != len(selected) or len(set(selected.tolist())) != len(selected):
            raise ValueError("fixed input frame contract differs")
        query = int(selected[0])
        return vertices[selected], joints[query], None if tails is None else tails[query], selected

    with patch.object(data_module, "_select_query_sequence", select):
        sample = dataset[index]
    if not np.array_equal(sample.query_center.numpy(), np.asarray(gauge["query_center"], dtype=np.float32)):
        raise ValueError("query normalization center differs from CPU audit")
    if float(sample.query_scale) != gauge["query_scale"] or sample.joint_count != case["target_joint_count"]:
        raise ValueError("query scale or full GT joint count differs")
    return sample


def transform_batch(batch, gauge, mode):
    import torch
    if mode == "original":
        return batch
    result = dict(batch)
    if mode == "static":
        for key in ("frame_vertices", "vertex_normals", "face_normals"):
            result[key] = batch[key][:, :1].expand_as(batch[key]).clone()
    else:
        value = gauge["gauges"][mode]
        if not value["available"] or value.get("requires_gt_skeleton"):
            raise ValueError("only available mesh-only gauges may enter this model diagnostic")
        positions = gauge["selected_positions"]
        points = batch["frame_vertices"]
        rotations = torch.as_tensor(np.asarray(value["rotations"])[positions], dtype=points.dtype, device=points.device)
        translations = torch.as_tensor(np.asarray(value["translations_common"])[positions], dtype=points.dtype, device=points.device)
        for key in ("frame_vertices", "vertex_normals", "face_normals"):
            result[key] = torch.einsum("tij,btnj->btni", rotations, batch[key])
            if key == "frame_vertices":
                result[key] += translations[None, :, None]
            result[key][:, 0] = batch[key][:, 0]
    if not torch.equal(result["frame_vertices"][:, 0], batch["frame_vertices"][:, 0]):
        raise ValueError("diagnostic changed the query mesh")
    for key in ("input_ids", "target_joints", "target_parents", "query_center", "query_scale"):
        if result[key] is not batch[key]:
            raise ValueError("diagnostic changed query GT or normalization")
    return result


def point_stats(points):
    maxima = points.detach().float().abs().amax(-1)
    return {"maxabs": float(maxima.max()), "point_fraction_outside_1": float((maxima > 1).float().mean()),
            "point_fraction_outside_2": float((maxima > 2).float().mean()),
            "point_fraction_outside_4": float((maxima > 4).float().mean())}


def teacher_metrics(model, condition, batch):
    import torch
    ids, mask = batch["input_ids"], batch["attention_mask"]
    embeddings = model.token_inputs_embeds(ids, mask)
    prompt = torch.cat((condition.to(model.transformer.dtype), embeddings), dim=1)
    attention = torch.nn.functional.pad(mask, (condition.shape[1], 0), value=1.)
    result = model.transformer(inputs_embeds=prompt, attention_mask=attention, use_cache=False)
    logits = result.logits[:, condition.shape[1]:-1].float()
    labels = ids[:, 1:]
    losses = torch.nn.functional.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
    predictions = logits.argmax(-1)
    groups = {"all": mask[:, 1:] > 0, "coordinate": (labels < model.tokenizer.num_discrete) & (labels >= 0),
              "eos": labels == model.tokenizer.eos, "branch": labels == model.tokenizer.token_id_branch}
    metrics = {}
    for name, selected in groups.items():
        metrics[name] = {"tokens": int(selected.sum()), "ce": float(losses[selected].mean()) if selected.any() else None,
                         "accuracy": float((predictions[selected] == labels[selected]).float().mean()) if selected.any() else None}
    return metrics


def generation_metrics(model, tokenizer, batch, condition, max_tokens):
    from eval_dynamic_rig_generation import _continuous_range, _dynamic_generate, _output_metrics
    ids = _dynamic_generate(model, tokenizer, batch, max_tokens,
                            generation_kwargs={"do_sample": False, "num_beams": 1}, precomputed_cond=condition)
    has_eos = bool(np.any(ids == int(tokenizer.eos)))
    hit_max = len(ids) - 2 >= max_tokens and not has_eos
    prediction, error = None, None
    try:
        prediction = tokenizer.detokenize(ids)
        if not np.isfinite(prediction.joints).all():
            raise ValueError("nonfinite generated joints")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        prediction = None
    success = has_eos and prediction is not None and not hit_max
    count = int(batch["joint_count"][0])
    parents = batch["target_parents"][0, :count].tolist()
    target = SimpleNamespace(joints=batch["target_joints"][0, :count].cpu().numpy(),
                             parents=[None if value < 0 else int(value) for value in parents])
    result = {"success": success, "has_eos": has_eos, "hit_max_without_eos": hit_max, "error": error,
              "generated_ids": ids.tolist(), "predicted_joints": None, "predicted_parents": None,
              "edge_f1_or_zero": 0., "metrics": None}
    if success:
        result["metrics"] = _output_metrics(prediction, target, _continuous_range(tokenizer))
        result["edge_f1_or_zero"] = result["metrics"]["topology"]["edge_f1"]
        result["predicted_joints"] = np.asarray(prediction.joints).tolist()
        result["predicted_parents"] = [None if p is None else int(p) for p in prediction.parents]
    return result


def summarize(rows):
    summary = {}
    for stratum in sorted({row["stratum"] for row in rows}):
        for mode in sorted({row["mode"] for row in rows}):
            group = [row for row in rows if row["stratum"] == stratum and row["mode"] == mode]
            summary[f"{stratum}/{mode}"] = {
                "assets": len(group), "natural_eos": sum(row["generation"]["success"] for row in group),
                "edge_f1_all_rows": float(np.mean([row["generation"]["edge_f1_or_zero"] for row in group])),
                "teacher_ce": float(np.mean([row["teacher"]["all"]["ce"] for row in group])),
                "coordinate_ce": float(np.mean([row["teacher"]["coordinate"]["ce"] for row in group])),
            }
    return summary


def require_formal_started(controller_path, rank_path):
    controller = json.loads(controller_path.read_text())
    rank = json.loads(rank_path.read_text())
    if controller.get("status") not in {"running", "completed"} or controller.get("plan", {}).get("plan") != "formal":
        raise ValueError("old-model GPU diagnostics may start only after the formal new-model run")
    if int(rank.get("samples", 0)) < 1 or rank.get("initialization", {}).get("old_method_trained_weights_loaded") is not False:
        raise ValueError("formal-start evidence must contain completed fresh-model training updates")
    return {"controller_sha256": sha256(controller_path), "rank_sha256": sha256(rank_path),
            "formal_label": controller["plan"]["label"], "observed_formal_samples": rank["samples"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", default="all")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--formal-controller", type=Path)
    parser.add_argument("--formal-rank", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = json.loads(args.plan.read_text())
    runtime = plan["runtime"]
    os.environ.update(EVOWEAVE_UNIRIG_ROOT=runtime["unirig_root"], EVOWEAVE_OPT_CONFIG_ROOT=runtime["opt_config_root"],
                      RIGWEAVE_DISABLE_OPEN3D="1", RIGWEAVE_DISABLE_LIGHTNING_IMPORT="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), runtime["unirig_root"]]
    if sys.platform == "win32":
        sys.modules.setdefault("resource", ModuleType("resource"))
    import torch
    import transformers
    from eval_dynamic_rig_ce import CHECKPOINT_DEFAULTS, _build_dynamic_model
    from train_dynamic_rig import build_tokenizer, move_batch
    from rigweave.dynamic_rig.data import DynamicRigManifestDataset, dynamic_rig_collate
    torch.set_num_threads(2)
    corpus = Path(plan["corpus"])
    cases_path, gauge_path = corpus / "cases.json", corpus / plan["gauges_file"]
    if sha256(cases_path) != plan["cases_sha256"] or sha256(gauge_path) != plan["gauges_sha256"]:
        raise ValueError("diagnostic case/gauge receipt changed")
    cases = json.loads(cases_path.read_text())["cases"]
    gauges = json.loads(gauge_path.read_text())
    if not gauges.get("complete") or gauges["cases_sha256"] != plan["cases_sha256"]:
        raise ValueError("geometry audit is incomplete or unpaired")
    gauge_by_index = {row["case_index"]: row for row in gauges["rows"]}
    indices = list(range(len(cases))) if args.indices == "all" else [int(x) for x in args.indices.split(",")]
    if len(set(indices)) != len(indices) or not set(indices).issubset(range(len(cases))):
        raise ValueError("invalid or duplicated case indices")
    tokenizer = build_tokenizer(Path(runtime["tokenizer_config"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output.with_suffix(".manifest.jsonl")
    manifest_path.write_text("".join(json.dumps({"path": str(corpus / case["local_relative_path"])}) + "\n" for case in cases))
    dataset = DynamicRigManifestDataset(manifest_path, tokenizer, frame_count=8, random_query=False)
    report = {"complete": False, "started_at": datetime.now(timezone.utc).isoformat(), "plan": plan, "plan_sha256": sha256(args.plan),
              "source": {"git_head": subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"], text=True).strip(),
                         "script_sha256": sha256(__file__), "torch": torch.__version__, "transformers": transformers.__version__}, "rows": []}
    if args.prepare_only:
        for index in indices:
            case, gauge = cases[index], gauge_by_index[index]
            if sha256(corpus / case["local_relative_path"]) != case["npz_sha256"]:
                raise ValueError("diagnostic NPZ changed")
            sample = fixed_sample(dataset, index, case, gauge)
            report["rows"].append({"case_index": index, "asset_id": case["asset_id"], "target_tokens": len(sample.input_ids),
                                   "query_target_maxabs": float(sample.target_joints.abs().max())})
        report.update(complete=True, prepared_only=True, cuda_initialized=torch.cuda.is_initialized())
        if report["cuda_initialized"]:
            raise RuntimeError("CPU input preparation unexpectedly initialized CUDA")
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"cpu_inputs_verified": len(indices), "cuda_initialized": False}), flush=True)
        return
    if args.formal_controller is None or args.formal_rank is None:
        raise ValueError("formal-start evidence is required before old-model inference")
    report["formal_start_evidence"] = require_formal_started(args.formal_controller, args.formal_rank)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info()
    if free < plan["minimum_free_gib"] * 2**30:
        raise RuntimeError(f"insufficient free GPU memory for local diagnostic: {free / 2**30:.3f} GiB")
    torch.cuda.set_per_process_memory_fraction(plan["allocator_fraction"])
    checkpoint = Path(plan["checkpoint"])
    if sha256(checkpoint) != plan["checkpoint_sha256"]:
        raise ValueError("old-model checkpoint digest differs")
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    model_args = SimpleNamespace(**{**CHECKPOINT_DEFAULTS, **payload["args"], "checkpoint": checkpoint,
        "model_config": Path(runtime["model_config"]), "tokenizer_config": Path(runtime["tokenizer_config"]),
        "unirig_checkpoint": Path(runtime["upstream_checkpoint"])})
    del payload
    if model_args.motion_evidence_fusion != "bias" or model_args.motion_evidence_heads != 8:
        raise ValueError("this diagnostic is for the completed original bias_h8 model")
    original_load = torch.nn.Module.load_state_dict
    def strict_load(module, *load_args, **load_kwargs):
        result = original_load(module, *load_args, **load_kwargs)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"checkpoint state mismatch: {result}")
        return result
    with patch.object(torch.nn.Module, "load_state_dict", strict_load):
        model = _build_dynamic_model(model_args, tokenizer, device)
    model.requires_grad_(False).eval()
    report["gpu"] = {"name": torch.cuda.get_device_name(device), "total_gib": total / 2**30, "precision": "BF16 autocast"}
    with torch.no_grad():
        for index in indices:
            case, gauge = cases[index], gauge_by_index[index]
            if sha256(corpus / case["local_relative_path"]) != case["npz_sha256"]:
                raise ValueError("diagnostic NPZ changed")
            sample = fixed_sample(dataset, index, case, gauge)
            original = move_batch(dynamic_rig_collate([sample], tokenizer.pad), device)
            torch.manual_seed(plan["sampling_seed"] + index)
            refs = model.sample_references(original)
            for mode in plan["modes"]:
                batch = transform_batch(original, gauge, mode)
                observed_points = []
                def observe(_module, inputs):
                    observed_points.append({"dense": point_stats(inputs[0]), "anchors": point_stats(inputs[2])})
                hook = model.conditioner.surface_tokenizer.register_forward_pre_hook(observe)
                tick = time.monotonic()
                torch.cuda.reset_peak_memory_stats()
                try:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        condition = model.build_condition(batch, refs=refs)
                finally:
                    hook.remove()
                if len(observed_points) != 8 or not torch.isfinite(condition).all():
                    raise RuntimeError("actual encoder input capture or condition finiteness failed")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    teacher = teacher_metrics(model, condition, batch)
                    generated = generation_metrics(model, tokenizer, batch, condition, plan["max_new_tokens"])
                row = {**case, "mode": mode, "teacher": teacher, "generation": generated,
                       "sampled_encoder_points": observed_points, "wholemesh": point_stats(batch["frame_vertices"]),
                       "query_target": point_stats(batch["target_joints"]), "target_ids": sample.input_ids.tolist(),
                       "target_joints": sample.target_joints.tolist(), "target_parents": sample.target_parents.tolist(),
                       "seconds": time.monotonic() - tick, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
                report["rows"].append(row)
                args.output.write_text(json.dumps(report) + "\n")
                print(json.dumps({"case": index, "mode": mode, "ce": teacher["all"]["ce"],
                                  "success": generated["success"], "f1": generated["edge_f1_or_zero"]}), flush=True)
                del batch, condition
            del original, refs, sample
    report.update(complete=True, finished_at=datetime.now(timezone.utc).isoformat(), summary=summarize(report["rows"]))
    args.output.write_text(json.dumps(report) + "\n")
    print(json.dumps({"complete": True, "rows": len(report["rows"])}), flush=True)


if __name__ == "__main__":
    main()
