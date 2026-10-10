"""Run one fresh step and one strict-resume smoke, never a full training run."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
import tempfile
import time
from types import SimpleNamespace

import torch
from torch.utils.data.distributed import DistributedSampler

from motion_experiment_runtime import guard, load_config, source_metadata, verify_manifests, write_json
from run_motion_experiment import build_plan, training_environment
from rigweave.dynamic_rig.frame_batch_sampler import VariableFrameBatchSampler
from rigweave.dynamic_rig.frame_budget_accounting import FrameBudgetProgress


def validate_settings(raw):
    required = {"frame_budget", "frame_batch_cap", "max_steps", "first_stop_after_steps",
                "target_stop_step", "sample_overshoot"}
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError(f"frame_budget_smoke requires exactly {sorted(required)}")
    if any(type(value) is not int for value in raw.values()):
        raise ValueError("all smoke settings must be integers")
    if raw["frame_budget"] < 24 or raw["frame_batch_cap"] < 1:
        raise ValueError("explicit calibrated frame_budget >= 24 and frame_batch_cap >= 1 are required")
    for name, expected in (("max_steps", 4), ("first_stop_after_steps", 1),
                           ("target_stop_step", 3), ("sample_overshoot", 1)):
        if raw[name] != expected:
            raise ValueError(f"this bounded smoke requires {name}={expected}")
    return dict(raw)


def predict_schedule(expected, train_rows, world_size=2):
    if type(train_rows) is not int or train_rows < 1 or world_size != 2:
        raise ValueError("prediction requires the full nonempty manifest and exactly two ranks")
    if expected["grad_accum_steps"] != 8:
        raise ValueError("the smoke must retain grad_accum_steps=8")
    bases = [DistributedSampler(range(train_rows), num_replicas=world_size, rank=rank, shuffle=True)
             for rank in range(world_size)]
    samplers = [VariableFrameBatchSampler(base, expected["batch_size"], min_frames=expected["frames_min"],
                max_frames=expected["frames"], seed=expected["seed"], frame_budget=expected["frame_budget"],
                max_batch_size=expected["frame_batch_cap"]) for base in bases]
    steps, window, histogram = [], [], {}
    epoch = samples_seen = frames_seen = microbatches_seen = 0
    while len(steps) < expected["max_steps"]:
        for base, sampler in zip(bases, samplers):
            base.set_epoch(epoch)
            sampler.set_epoch(epoch)
        counts = [len(sampler) for sampler in samplers]
        if counts[0] < 1 or len(set(counts)) != 1:
            raise RuntimeError("DDP sampler lengths differ or are empty")
        iterators = [iter(sampler) for sampler in samplers]
        for batch_index in range(counts[0]):
            rows = [next(iterator) for iterator in iterators]
            sizes, frames = [len(row) for row in rows], [row[0].frames for row in rows]
            if len(set(sizes)) != 1 or len(set(frames)) != 1:
                raise RuntimeError("DDP frame-budget schedules disagree")
            batch, count = sizes[0], frames[0]
            window.append({"epoch": epoch, "batch_in_epoch": batch_index, "frames": count,
                           "batch_per_rank": batch, "rank_batches": sizes})
            if len(window) != expected["grad_accum_steps"]:
                continue
            step_samples = world_size * sum(row["batch_per_rank"] for row in window)
            step_frames = world_size * sum(row["batch_per_rank"] * row["frames"] for row in window)
            samples_seen += step_samples
            frames_seen += step_frames
            microbatches_seen += len(window)
            for row in window:
                key = str(row["frames"])
                histogram[key] = histogram.get(key, 0) + world_size * row["batch_per_rank"]
            next_epoch, next_batch = (epoch + 1, 0) if batch_index + 1 == counts[0] else (epoch, batch_index + 1)
            steps.append({"step": len(steps) + 1, "epoch": epoch, "microbatches": list(window),
                          "frames_in_step": [row["frames"] for row in window],
                          "micro_batch_sizes_in_step": [row["batch_per_rank"] for row in window],
                          "sample_seen": samples_seen, "input_frames_seen": frames_seen,
                          "samples_in_step": step_samples, "input_frames_in_step": step_frames,
                          "consumed_microbatches": microbatches_seen, "next_epoch": next_epoch,
                          "next_batch_in_epoch": next_batch, "samples_by_frame_count": dict(histogram),
                          "rank_samples_seen": [samples_seen // world_size] * world_size,
                          "rank_input_frames_seen": [frames_seen // world_size] * world_size})
            window.clear()
            if len(steps) == expected["max_steps"]:
                break
        epoch += 1
    return {"world_size": world_size, "grad_accum_steps": expected["grad_accum_steps"],
            "train_rows": train_rows, "rows_per_rank": len(bases[0]), "steps": steps,
            "scope": "Actual sampler on manifest row indices only; no assets or GT are loaded. Microbatch counters are per rank; exposure counters are global and include DDP padding."}


def build_smoke_plan(config, output):
    settings = validate_settings(config["frame_budget_smoke"])
    base = build_plan(config, "full", "bias_h8")
    if config["full_training"]["world_size"] != 2 or len(base["devices"]) != 2 or len(set(base["devices"])) != 2:
        raise ValueError("smoke requires exactly the two recorded devices and world_size=2")
    expected = dict(base["expected"])
    locked = {"limit_train": 0, "grad_accum_steps": 8, "frames": 24, "frames_min": 2,
              "train_random_query": True, "target_active_skin_only": False, "motion_checkpointing": True,
              "use_motion_features": False, "use_time_embedding": False, "freeze_ar": False,
              "freeze_conditioner": False, "train_surface_tokenizer": True, "scheduler": "onecycle",
              "amp_dtype": "bf16", "init_dynamic_encoder_checkpoint": None,
              "init_surface_tokenizer_from_dynamic": False, "motion_evidence_fusion": "bias", "motion_evidence_heads": 8}
    require_fields(expected, locked, "unchanged full trainable/data recipe")
    runtime = config["runtime"]
    job = PurePosixPath(runtime["job_root"])
    directory = PurePosixPath(runtime["output_root"]) / "frame_budget_smoke/bias_h8"
    expected.update(output_dir=str(directory), frame_budget=settings["frame_budget"],
                    frame_batch_cap=settings["frame_batch_cap"], max_steps=settings["max_steps"],
                    no_save_optimizer=False, log_every=1, val_every=0, save_every=0,
                    resume_checkpoint=None, init_checkpoint=None)
    train_rows = config["baseline"]["args"]["train_rows"]
    prediction = predict_schedule(expected, train_rows)
    threshold = prediction["steps"][2]["sample_seen"] - settings["sample_overshoot"]
    if not prediction["steps"][1]["sample_seen"] < threshold < prediction["steps"][2]["sample_seen"]:
        raise ValueError("sample threshold must be crossed, not reached early, on the complete third step")
    expected.update(max_samples=threshold, sample_milestones=str(threshold))
    phases = []
    for name, stop, resume in (("fresh", 1, None), ("resume", 0, str(directory / "checkpoint_last.pt"))):
        key = f"frame_budget_smoke_{name}"
        expected_path = str(job / "results" / f"{key}_expected_args.json")
        phase_args = {**expected, "stop_after_steps": stop, "resume_checkpoint": resume, "expected_args": expected_path}
        phases.append({"phase": name, "command": base["command"], "devices": list(base["devices"]),
                       "output": str(directory), "expected": phase_args, "expected_path": expected_path,
                       "log": str(job / "logs" / f"{key}.log"), "expected_final_step": 1 if name == "fresh" else 3})
    return {"stage": "frame_budget_smoke", "candidate": "bias_h8", "settings": settings,
            "output": str(directory), "result": str(output), "max_samples": threshold,
            "sample_milestones": [threshold], "prediction": prediction,
            "optimizer_schedule": predict_optimizer_schedule(expected), "phases": phases}


def predict_optimizer_schedule(expected):
    names = ("motion", "ar", "surface")
    groups = [{"params": [torch.nn.Parameter(torch.zeros(1))], "lr": expected[f"lr_{name}"], "name": name} for name in names]
    optimizer = torch.optim.AdamW(groups, weight_decay=expected["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[group["lr"] for group in groups], total_steps=expected["max_steps"],
        pct_start=expected["onecycle_pct_start"], anneal_strategy="cos", div_factor=expected["onecycle_div_factor"],
        final_div_factor=expected["onecycle_final_div_factor"],
    )
    rows = []
    for step in range(expected["max_steps"] + 1):
        rows.append({"step": step, "lrs": {group["name"]: group["lr"] for group in optimizer.param_groups},
                     "betas": [list(group["betas"]) for group in optimizer.param_groups]})
        if step < expected["max_steps"]:
            optimizer.step()
            scheduler.step()
    return rows


def require_fields(actual, expected, label):
    differences = {key: {"expected": value, "actual": actual.get(key)} for key, value in expected.items()
                   if key not in actual or actual[key] != value}
    if differences:
        raise ValueError(f"{label} mismatch: {differences}")


def recorded_args(phase, train_rows):
    return {**phase["expected"], "effective_batch": 0, "train_rows": train_rows,
            "sample_milestones_parsed": [phase["expected"]["max_samples"]]}


def require_exact_args(actual, expected, label):
    if set(actual) != set(expected):
        raise ValueError(f"{label} argument names differ: missing={sorted(set(expected) - set(actual))}, extra={sorted(set(actual) - set(expected))}")
    require_fields(actual, expected, label)


def smoke_environment(config, phase):
    environment = training_environment(config, phase)
    # The real launcher consumes EVOWEAVE_RESUME_CHECKPOINT, not the RIGWEAVE alias.
    environment.update(EVOWEAVE_INIT_CHECKPOINT="", EVOWEAVE_RESUME_CHECKPOINT=phase["expected"]["resume_checkpoint"] or "")
    if environment["RIGWEAVE_NPROC"] != "2":
        raise ValueError("launcher environment changed the required two-rank setup")
    return environment


def trainer_contract(path, environment):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    body = []
    for node in main.body:
        if isinstance(node, ast.Global):
            continue
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "args" for target in node.targets):
            break
        parser_assignment = (isinstance(node, ast.Assign) and len(node.targets) == 1
                             and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "parser"
                             and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "argparse.ArgumentParser")
        parser_call = (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                       and ast.unparse(node.value.func) in {"parser.add_argument", "parser.set_defaults"})
        if not (parser_assignment or parser_call):
            raise RuntimeError("trainer parser construction changed; refusing to execute other trainer setup")
        body.append(node)
    else:
        raise RuntimeError("trainer parse_args boundary was not found")
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name in {"json_safe", "resume_contract_differences"}]
    # These are Linux launcher arguments even when the CPU checks run on Windows.
    scope = {"argparse": argparse, "Path": PurePosixPath, "os": SimpleNamespace(environ=environment), "json": json}
    # Only the real argparse construction and pure resume contract execute, never trainer imports/setup.
    code = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *body, *helpers], type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), scope)
    return scope["parser"], scope["resume_contract_differences"]


def verify_launcher_arguments(config, phase, environment):
    marker = b"\x00FRAME_BUDGET_SMOKE_ARGV\x00"
    with tempfile.TemporaryDirectory(prefix="frame_smoke_argv_") as directory:
        stub = Path(directory) / "torchrun"
        stub.write_text("#!/usr/bin/env bash\nprintf '\\0FRAME_BUDGET_SMOKE_ARGV\\0'\nprintf '%s\\0' \"$@\"\n", encoding="utf-8")
        stub.chmod(0o700)
        capture_env = {**environment, "PATH": directory + os.pathsep + environment["PATH"],
                       "RIGWEAVE_SKIP_PREFLIGHT": "1", "RIGWEAVE_SKIP_DATALOADER_CHECK": "1",
                       "CUDA_VISIBLE_DEVICES": "", "RIGWEAVE_REQUIRE_CUDA": "0"}
        captured = subprocess.run(phase["command"], cwd=config["runtime"]["repo"], env=capture_env,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    if captured.stdout.count(marker) != 1:
        raise RuntimeError("CPU-only launcher capture did not reach the isolated torchrun stub exactly once")
    argv = [os.fsdecode(value) for value in captured.stdout.split(marker)[1].split(b"\x00")[:-1]]
    script_index = argv.index("rigweave/scripts/train_dynamic_rig.py")
    if argv[:script_index] != ["--standalone", "--nproc_per_node", "2"]:
        raise ValueError("captured launcher is not the original standalone two-rank torchrun")
    trainer = Path(config["runtime"]["repo"]) / "model_training/rigweave/scripts/train_dynamic_rig.py"
    parser, _ = trainer_contract(trainer, environment)
    actual = json.loads(json.dumps(vars(parser.parse_args(argv[script_index + 1:])), default=str))
    require_exact_args(actual, phase["expected"], f"{phase['phase']} actual launcher/argparse")
    return {"argument_count": len(actual), "two_rank_launcher": True, "complete_exact_match": True}


def validate_prepared_state(state, config, config_path):
    active = state.get("active_operation", {})
    require_fields(active, {"experiment": config["experiment"], "stage": "frame_budget_smoke",
                           "candidate": "bias_h8", "status": "prepared"}, "committed prepared operation")
    if Path(active["config"]).resolve() != Path(config_path).resolve():
        raise ValueError("smoke config differs from the committed prepared operation")
    if "train" not in state.get("allowed_operations", []) or "train" in state.get("blocked_operations", []):
        raise ValueError("current state does not authorize the train guard")
    return {"state_id": state["state_id"], "stage": active["stage"], "candidate": active["candidate"]}


def reject_collisions(plan, config_path, repo):
    directory = Path(plan["output"]).resolve()
    paths = [Path(plan["result"])]
    for phase in plan["phases"]:
        paths += [Path(phase["log"]), Path(phase["expected_path"])]
    if Path(plan["result"]).suffix.lower() != ".json":
        raise ValueError("the controller output must be a JSON report")
    json_paths = [path for path in paths if path.suffix.lower() == ".json"]
    paths += [path.with_suffix(path.suffix + ".tmp") for path in json_paths]
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("report/log/expected artifact paths collide")
    if directory.exists() or Path(plan["output"]).is_symlink():
        raise FileExistsError(f"refusing to reuse a smoke training directory: {directory}")
    if any(path.is_symlink() for path in paths):
        raise FileExistsError("refusing a pre-existing artifact symlink")
    for path in resolved:
        if path == Path(config_path).resolve() or path.is_relative_to(Path(repo).resolve()) or path.is_relative_to(directory):
            raise ValueError(f"controller artifact overlaps source/config/trainer output: {path}")
        if path.exists():
            raise FileExistsError(f"refusing to overwrite a smoke artifact: {path}")
        if any(other != path and path.is_relative_to(other) for other in resolved):
            raise ValueError("one artifact path is a parent of another")


def expected_counter_fields(predicted, train_rows):
    return {"sample_seen": predicted["sample_seen"], "optimizer_samples_seen": predicted["sample_seen"],
            "input_frames_seen": predicted["input_frames_seen"], "samples_in_step": predicted["samples_in_step"],
            "effective_batch": predicted["samples_in_step"], "input_frames_in_step": predicted["input_frames_in_step"],
            "consumed_microbatches": predicted["consumed_microbatches"], "train_rows": train_rows,
            "epoch_equivalent": predicted["sample_seen"] / train_rows, "loss_normalization": "global_target_token_weight"}


def validate_log_rows(rows, plan, phase):
    train_rows, final_step = plan["prediction"]["train_rows"], phase["expected_final_step"]
    predictions = plan["prediction"]["steps"]
    configs = [row for row in rows if row.get("event") == "run_config"]
    if len(configs) != 1:
        raise ValueError("expected exactly one fresh run_config event")
    require_fields(configs[0], {"world_size": 2, "batch_mode": "frame_budget", "effective_batch": None,
                               "loss_normalization": "global_target_token_weight", "train_rows": train_rows,
                               "sample_milestones": plan["sample_milestones"]}, "run_config")
    require_exact_args(configs[0]["args"], recorded_args(plan["phases"][0], train_rows), "run_config args")
    training = [row for row in rows if "loss" in row]
    if [row.get("step") for row in training] != list(range(1, final_step + 1)):
        raise ValueError("training log has missing, duplicated or unexpected optimizer steps")
    tokens_seen = 0.0
    compact = []
    for row, predicted in zip(training, predictions):
        require_fields(row, {**expected_counter_fields(predicted, train_rows), "epoch": predicted["epoch"],
                             "frames_in_step": predicted["frames_in_step"], "grad_accum": 8,
                             "micro_batch_sizes_in_step": predicted["micro_batch_sizes_in_step"],
                             "micro_batch_per_gpu": None}, f"step {predicted['step']} counters/schedule")
        for name in ("loss", "ce", "gradient_norm_before_clip", "motion_adapter_gradient_norm_after_clip",
                     "motion_adapter_parameter_norm", "seconds", "gpu_peak_gb"):
            if not isinstance(row.get(name), (int, float)) or not math.isfinite(row[name]) or row[name] < 0:
                raise ValueError(f"missing/non-finite training metric: {name}")
        if any(row[name] <= 0 for name in ("gradient_norm_before_clip", "motion_adapter_gradient_norm_after_clip", "motion_adapter_parameter_norm")):
            raise ValueError("expected nonzero full-model/adapter gradients and updated bias")
        if set(row.get("lrs", {})) != {"motion", "ar", "surface"} or any(not math.isfinite(value) or value <= 0 for value in row["lrs"].values()):
            raise ValueError("missing or invalid active-group learning rates")
        expected_lrs = plan["optimizer_schedule"][predicted["step"]]["lrs"]
        if any(not math.isclose(row["lrs"][name], value, rel_tol=1e-12, abs_tol=1e-15) for name, value in expected_lrs.items()):
            raise ValueError("logged learning rates disagree with the unchanged four-step OneCycle schedule")
        token_weight = row.get("target_token_weight_in_step", 0)
        if not math.isfinite(token_weight) or token_weight <= 0:
            raise ValueError("invalid target-token weight")
        tokens_seen += token_weight
        if not math.isclose(row.get("target_token_weight_seen", -1), tokens_seen, rel_tol=1e-12, abs_tol=1e-6):
            raise ValueError("cumulative target-token weight disagrees with logged complete steps")
        compact.append({name: row[name] for name in ("step", "loss", "ce", "gradient_norm_before_clip",
                       "motion_adapter_gradient_norm_after_clip", "motion_adapter_parameter_norm", "sample_seen",
                       "input_frames_seen", "target_token_weight_in_step", "target_token_weight_seen", "lrs")})
    resumes = [row for row in rows if row.get("event") == "resume"]
    milestones = [row for row in rows if row.get("event") == "sample_milestone_checkpoint"]
    if phase["phase"] == "fresh":
        if resumes or milestones:
            raise ValueError("fresh step1 must not resume or reach the final sample milestone")
    else:
        if len(resumes) != 1 or len(milestones) != 1:
            raise ValueError("resume smoke needs exactly one resume and one sample milestone event")
        first, last = predictions[0], predictions[2]
        require_fields(resumes[0], {**expected_counter_fields(first, train_rows), "step": 1,
                                   "resume_checkpoint": phase["expected"]["resume_checkpoint"],
                                   "epoch": first["next_epoch"], "skipped_batches_in_epoch": first["next_batch_in_epoch"]}, "resume event/cursor")
        if not math.isclose(resumes[0]["target_token_weight_seen"], compact[0]["target_token_weight_seen"], rel_tol=1e-12, abs_tol=1e-6):
            raise ValueError("resume token counter differs from step1")
        require_fields(milestones[0], {**expected_counter_fields(last, train_rows), "step": 3,
                                      "sample_milestone": plan["max_samples"], "include_optimizer": True,
                                      "path": str(PurePosixPath(plan["output"]) / f"checkpoint_sample_{plan['max_samples']}.pt")}, "actual sample milestone")
        if not predictions[1]["sample_seen"] < plan["max_samples"] < last["sample_seen"] or last["sample_seen"] - plan["max_samples"] != 1:
            raise ValueError("the smoke did not stop at the intended actual-sample crossing")
    return {"training_steps": compact, "resume_event": resumes[0] if resumes else None,
            "milestone_event": milestones[0] if milestones else None,
            "target_token_weight_seen": tokens_seen, "last_token_weight": training[-1]["target_token_weight_in_step"]}


def inspect_checkpoint(path, plan, phase, log_evidence, resume_phase=None):
    path = Path(path)
    directory = Path(plan["output"]).resolve()
    allowed = {"checkpoint_last.pt", f"checkpoint_sample_{plan['max_samples']}.pt"}
    if path.resolve().parent != directory or path.name not in allowed:
        raise ValueError("checkpoint inspection is restricted to this smoke's own output files")
    if torch.cuda.is_initialized():
        raise RuntimeError("checkpoint inspection must remain CPU-only")
    payload = torch.load(path, mmap=True, map_location="cpu", weights_only=False)
    step, train_rows = phase["expected_final_step"], plan["prediction"]["train_rows"]
    predicted = plan["prediction"]["steps"][step - 1]
    require_exact_args(payload["args"], recorded_args(phase, train_rows), "checkpoint args")
    require_fields(payload, {"step": step, "sample_seen": predicted["sample_seen"],
                            "input_frames_seen": predicted["input_frames_seen"], "train_rows": train_rows,
                            "effective_batch": None, "epoch_equivalent": predicted["sample_seen"] / train_rows}, "checkpoint header")
    progress = FrameBudgetProgress.restore(payload["batch_accounting"], step=step, grad_accum_steps=8)
    require_fields(progress.checkpoint(), {"samples_seen": predicted["sample_seen"], "input_frames_seen": predicted["input_frames_seen"],
                                          "microbatches_seen": predicted["consumed_microbatches"], "next_epoch": predicted["next_epoch"],
                                          "next_batch_in_epoch": predicted["next_batch_in_epoch"], "last_samples": predicted["samples_in_step"],
                                          "last_frames": predicted["input_frames_in_step"], "samples_by_frame_count": predicted["samples_by_frame_count"]}, "checkpoint actual accounting/cursor")
    for name, actual in (("target_token_weight_seen", progress.target_token_weight_seen), ("last_token_weight", progress.last_token_weight)):
        if not math.isclose(actual, log_evidence[name], rel_tol=1e-12, abs_tol=1e-6):
            raise ValueError(f"checkpoint/log disagree on {name}")
    optimizer, scheduler = payload["optimizer"], payload["scheduler"]
    groups, states = optimizer["param_groups"], optimizer["state"]
    if [group.get("name") for group in groups] != ["motion", "ar", "surface"] or [len(group["params"]) for group in groups] != [304, 388, 196]:
        raise ValueError("optimizer does not contain the complete original three trainable groups")
    parameter_ids = [value for group in groups for value in group["params"]]
    if len(set(parameter_ids)) != 888 or set(parameter_ids) != set(states):
        raise ValueError("optimizer state IDs do not cover all 888 parameters exactly once")
    for value in states.values():
        moment, variance = value["exp_avg"], value["exp_avg_sq"]
        if float(value["step"]) != step or moment.shape != variance.shape or moment.numel() < 1:
            raise ValueError("Adam step/moment metadata is inconsistent")
        if moment.device.type != "cpu" or variance.device.type != "cpu" or not moment.is_floating_point() or not variance.is_floating_point():
            raise ValueError("expected CPU floating-point Adam state metadata")
    require_fields(scheduler, {"total_steps": 4, "last_epoch": step, "_step_count": step + 1}, "scheduler")
    if scheduler.get("_last_lr") != [group["lr"] for group in groups]:
        raise ValueError("scheduler/optimizer learning rates disagree")
    expected_optimizer = plan["optimizer_schedule"][step]
    for index, group in enumerate(groups):
        if not math.isfinite(group["lr"]) or group["lr"] <= 0 or group["weight_decay"] != phase["expected"]["weight_decay"]:
            raise ValueError("optimizer group hyperparameters disagree with the recipe")
        if (not math.isclose(group["lr"], expected_optimizer["lrs"][group["name"]], rel_tol=1e-12, abs_tol=1e-15)
                or list(group["betas"]) != expected_optimizer["betas"][index]):
            raise ValueError("checkpoint optimizer LR/betas disagree with the predicted OneCycle step")
    bias = [(name, value) for name, value in payload["model"].items() if name.endswith(".motion_injection.bias_weights")]
    if len(bias) != 12 or any(tuple(value.shape) != (8, 3) or not bool(torch.isfinite(value).all()) or float(value.norm()) <= 0 for _, value in bias):
        raise ValueError("missing, non-finite or unupdated twelve-layer bias parameters")
    if resume_phase is not None:
        trainer = Path(__file__).with_name("train_dynamic_rig.py")
        _, resume_contract = trainer_contract(trainer, {})
        differences = resume_contract(payload, SimpleNamespace(**recorded_args(resume_phase, train_rows)), effective_batch=0, train_rows=train_rows)
        if differences:
            raise ValueError(f"strict resume contract failed before second launch: {differences}")
    result = {"path": str(path), "bytes": path.stat().st_size, "step": step, "sample_seen": predicted["sample_seen"],
              "batch_accounting": progress.checkpoint(), "optimizer_group_counts": [len(group["params"]) for group in groups],
              "optimizer_states": len(states), "all_adam_steps": step,
              "scheduler": {name: scheduler[name] for name in ("total_steps", "last_epoch", "_step_count", "_last_lr")},
              "bias_parameter_norms": {name: float(value.norm()) for name, value in bias},
              "strict_resume_contract_passed": True if resume_phase is not None else None,
              "inspection": "CPU mmap metadata plus 288 bias coefficients only; no full model/Adam tensor scan or checkpoint writes."}
    del payload, optimizer, states, bias, groups
    gc.collect()
    return result


def execute_phase(config, phase, environment):
    path = Path(phase["log"])
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with path.open("x", encoding="utf-8") as handle:
        handle.write("COMMAND " + json.dumps(phase["command"]) + "\n")
        handle.flush()
        completed = subprocess.run(phase["command"], cwd=config["runtime"]["repo"], env=environment,
                                   stdout=handle, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{phase['phase']} launcher exited {completed.returncode}; no retry; see {path}")
    return {"child_exit": completed.returncode, "seconds": time.monotonic() - start, "log": str(path)}


def run_two_phases(phases, execute, inspect):
    if [phase["phase"] for phase in phases] != ["fresh", "resume"]:
        raise ValueError("the smoke may execute exactly one fresh phase followed by one resume phase")
    results = []
    for phase in phases:
        execution = execute(phase)
        evidence = inspect(phase)
        results.append({"phase": phase["phase"], **execution, **evidence})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    plan = build_smoke_plan(config, args.output)
    root, job = Path(config["runtime"]["repo"]), Path(config["runtime"]["job_root"])
    if root.resolve() != Path(__file__).resolve().parents[3]:
        raise RuntimeError("smoke must run from the recorded source checkout")
    import fcntl
    # The outer controller holds the cross-stage .operation.lock while invoking this script.
    with (job / ".frame_budget_smoke.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads((root / "model_training/state/current.json").read_text(encoding="utf-8"))
        state_evidence = validate_prepared_state(state, config, args.config)
        source, manifests = source_metadata(config), verify_manifests(config)
        train_path = plan["phases"][0]["expected"]["train_manifest"]
        train_manifest = [row for row in manifests if Path(row["path"]).resolve() == Path(train_path).resolve()]
        if len(train_manifest) != 1 or train_manifest[0]["rows"] != plan["prediction"]["train_rows"]:
            raise ValueError("prediction does not use the full verified training manifest row count")
        reject_collisions(plan, args.config, root)
        guard(config, "train")
        environments = {phase["phase"]: smoke_environment(config, phase) for phase in plan["phases"]}
        contracts = {phase["phase"]: verify_launcher_arguments(config, phase, environments[phase["phase"]]) for phase in plan["phases"]}
        report = {"complete": False, "status": "prepared", "plan": plan, "source": source, "manifests": manifests,
                  "state": state_evidence, "launcher_contracts": contracts, "phases": [],
                  "started_at": datetime.now(timezone.utc).isoformat(),
                  "limits": ["Counts, sampler cursor and strict state resumption only; no bitwise model-equivalence claim.",
                             "Recomputed motion evidence has measured numerical variation; validation is disabled to isolate this smoke and avoid best-checkpoint copies.",
                             "Fresh official AR/surface initialization and a fresh motion encoder; the finished 80k checkpoint is never loaded or modified.",
                             "Only checkpoint_last is saved at step1; it is inspected then used for resume, not copied. One final sample milestone plus checkpoint_last are retained.",
                             "Passing this smoke does not authorize a new full training run."]}
        for phase in plan["phases"]:
            expected_path = Path(phase["expected_path"])
            expected_path.parent.mkdir(parents=True, exist_ok=True)
            with expected_path.open("x", encoding="utf-8") as handle:
                json.dump(phase["expected"], handle, indent=2)
                handle.write("\n")
        write_json(args.output, report)

        def execute(phase):
            validate_prepared_state(json.loads((root / "model_training/state/current.json").read_text(encoding="utf-8")), config, args.config)
            if source_metadata(config) != source or verify_manifests(config) != manifests:
                raise RuntimeError("source/manifests changed between the two smoke launches")
            guard(config, "train")
            query = ["nvidia-smi", "-i", ",".join(map(str, phase["devices"])), "--query-compute-apps=pid", "--format=csv,noheader"]
            if subprocess.check_output(query, text=True).strip():
                raise RuntimeError("a recorded GPU has an active compute process; refusing this launch without retry")
            report.update(status="running", active_phase=phase["phase"])
            write_json(args.output, report)
            return execute_phase(config, phase, environments[phase["phase"]])

        def inspect(phase):
            directory = Path(plan["output"])
            args_path = directory / ("args.json" if phase["phase"] == "fresh" else "args_resume_step_1.json")
            require_exact_args(json.loads(args_path.read_text(encoding="utf-8")), recorded_args(phase, plan["prediction"]["train_rows"]), "saved args")
            metrics_path = directory / "logs/train.log"
            rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            logs = validate_log_rows(rows, plan, phase)
            resume_phase = plan["phases"][1] if phase["phase"] == "fresh" else None
            headers = [inspect_checkpoint(directory / "checkpoint_last.pt", plan, phase, logs, resume_phase)]
            names = {"checkpoint_last.pt"}
            if phase["phase"] == "resume":
                milestone = f"checkpoint_sample_{plan['max_samples']}.pt"
                names.add(milestone)
                headers.append(inspect_checkpoint(directory / milestone, plan, phase, logs))
            if {path.name for path in directory.glob("*.pt")} != names:
                raise ValueError("unexpected checkpoint multiplicity in this smoke output")
            evidence = {"log_evidence": logs, "checkpoint_headers": headers, "metrics_log": str(metrics_path)}
            report["phases"].append({"phase": phase["phase"], "child_exit": 0, **evidence})
            write_json(args.output, report)
            return evidence

        start = time.monotonic()
        try:
            report["phases"] = run_two_phases(plan["phases"], execute, inspect)
            report.update(complete=True, status="completed", actual_sample_termination_verified=True,
                          final_step=3, sample_overshoot=1, no_automatic_full_run=True)
        except BaseException as error:
            report.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            report.update(seconds=time.monotonic() - start, finished_at=datetime.now(timezone.utc).isoformat())
            write_json(args.output, report)
    print("COMPLETE", json.dumps({"result": str(args.output), "step": 3, "max_samples": plan["max_samples"], "overshoot": 1}), flush=True)


if __name__ == "__main__":
    main()
