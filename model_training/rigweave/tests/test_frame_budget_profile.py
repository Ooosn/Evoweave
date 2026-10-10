from __future__ import annotations

import ast
from contextlib import nullcontext
from dataclasses import dataclass, replace
import gc
import hashlib
import math
from pathlib import Path
import time
import unittest
from unittest.mock import Mock
import weakref

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/profile_frame_budget.py"


def helpers():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name != "main"]
    scope = {"torch": torch, "math": math, "Path": Path, "replace": replace,
             "hashlib": hashlib, "time": time, "CHECKPOINT_DEFAULTS": {"input_space_policy": "mesh_query_bbox"}}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SCRIPT), "exec"), scope)
    preflight = SCRIPT.with_name("preflight_motion_evidence.py")
    difference = [node for node in ast.parse(preflight.read_text(encoding="utf-8")).body
                  if isinstance(node, ast.FunctionDef) and node.name == "difference"]
    exec(compile(ast.Module(body=difference, type_ignores=[]), str(preflight), "exec"), scope)
    return scope


def plan():
    return {"checkpoint": "/reference/checkpoint_sample_80000.pt", "validation_rows": 32,
            "frame_budget": 72, "max_batch_cap": 12, "memory_fraction": 0.93,
            "acceptance_fraction": 0.90,
            "cases": [[24, 3], [18, 4], [14, 5], [12, 6], [10, 7], [9, 8],
                      [8, 9], [7, 10], [6, 12], [4, 12], [2, 12]]}


def measured(frames, batch, peak=89):
    return {"frames": frames, "batch": batch, "status": "measured", "backward_passes": 2,
            "optimizer_steps": 1,
            "peak_allocated_bytes": peak, "finite_loss": True, "finite_gradients": True, "finite_updates": True}


@dataclass
class Sample:
    path: str
    frame_vertices: torch.Tensor
    vertex_normals: torch.Tensor
    face_normals: torch.Tensor
    faces: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    target_joints: torch.Tensor
    target_parents: torch.Tensor
    selected_frames: torch.Tensor
    query_center: torch.Tensor
    query_scale: torch.Tensor
    joint_count: int = 3
    source_joint_count: int = 3


def sample(index=0, tokens=12, vertices=8):
    return Sample(
        path=f"asset_{index}.npz", frame_vertices=torch.arange(24 * vertices * 3).reshape(24, vertices, 3).float(),
        vertex_normals=torch.ones(24, vertices, 3), face_normals=torch.ones(24, 2, 3), faces=torch.ones(2, 3).long(),
        input_ids=torch.arange(tokens), attention_mask=torch.ones(tokens), target_joints=torch.arange(9).reshape(3, 3).float(),
        target_parents=torch.tensor([-1, 0, 1]), selected_frames=torch.arange(24) + 100,
        query_center=torch.tensor([2.0, 3.0, 4.0]), query_scale=torch.tensor(2.5),
    )


class TinyRoutes(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conditioner = torch.nn.Module()
        encoder = self.conditioner.motion_encoder = torch.nn.Module()
        encoder.use_time_embedding = False
        encoder.use_motion_features = False
        encoder.gradient_checkpointing = False
        encoder.time_embed = torch.nn.Parameter(torch.ones(2), requires_grad=False)
        encoder.motion_feature_mlp = torch.nn.Linear(1, 1).requires_grad_(False)
        encoder.active = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(2)) for _ in range(304)])
        self.transformer = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(2)) for _ in range(388)])
        self.conditioner.surface_tokenizer = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(2)) for _ in range(196)])

    def forward(self, batch):
        return {"loss": sum(parameter.square().sum() for parameter in self.parameters() if parameter.requires_grad)}


class CpuTorch:
    def __init__(self):
        self.cuda = Mock()
        for name in ("memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved"):
            getattr(self.cuda, name).return_value = 1024

    def __getattr__(self, name):
        return getattr(torch, name)

    def device(self, name):
        if name != "cuda:0":
            raise AssertionError(name)
        return torch.device("cpu")

    def autocast(self, device, *, dtype):
        if device != "cuda" or dtype != torch.bfloat16:
            raise AssertionError((device, dtype))
        return nullcontext()


class FrameBudgetProfileTest(unittest.TestCase):
    def setUp(self):
        self.scope = helpers()

    def test_plan_accepts_only_bounded_explicit_ascending_cases(self):
        self.assertEqual(self.scope["validate_plan"](plan()), plan())
        changes = [
            ("validation_rows", 0), ("validation_rows", 33), ("validation_rows", True),
            ("frame_budget", 23), ("frame_budget", 72.0), ("max_batch_cap", 0),
            ("memory_fraction", 0.94), ("memory_fraction", float("nan")),
            ("acceptance_fraction", 0.91), ("acceptance_fraction", 0), ("acceptance_fraction", True),
            ("checkpoint", "/reference/checkpoint_last.pt"), ("checkpoint", ""),
            ("cases", []), ("cases", [[24, 4]]), ("cases", [[1, 1]]), ("cases", [[25, 1]]),
            ("cases", [[2, 13]]), ("cases", [[24, True]]), ("cases", [[24, 3, 1]]),
            ("cases", [[24, 3], [24, 3]]), ("cases", [[18, 4], [24, 3]]),
        ]
        for key, value in changes:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.scope["validate_plan"]({**plan(), key: value})

    def test_output_cannot_overwrite_inputs_or_write_a_checkpoint(self):
        validate = self.scope["validate_output"]
        validate(Path("profile_report.json"), [Path("experiment.json")])
        for output, inputs in ((Path("checkpoint_sample_80000.pt"), []),
                               (Path("experiment.json"), [Path("experiment.json")]),
                               (Path("report.json"), [Path("report.json.tmp")])):
            with self.subTest(output=output), self.assertRaises(ValueError):
                validate(output, inputs)

    def test_checkpoint_recipe_requires_final_original_routes(self):
        saved = {"motion_evidence_fusion": "bias", "motion_evidence_heads": 8, "motion_depth": 12,
                 "frames": 24, "motion_checkpointing": True, "use_time_embedding": False,
                 "use_motion_features": False, "freeze_ar": False, "freeze_conditioner": False,
                 "train_surface_tokenizer": True, "target_active_skin_only": False,
                 "train_manifest": "train.jsonl", "val_manifest": "valid.jsonl", "weight_decay": 0.04,
                 "lr_motion": 1e-4, "lr_ar": 1e-4, "lr_surface": 1e-4}
        groups = [{"params": list(range(count)), "name": name, "lr": 2e-6, "weight_decay": 0.04}
                  for name, count in zip(("motion", "ar", "surface"), (304, 388, 196))]
        payload = {"args": saved, "step": 1667, "sample_seen": 80016,
                   "optimizer": {"param_groups": groups, "state": dict.fromkeys(range(888))}}
        config = {"baseline": {"args": {"train_manifest": "train.jsonl", "val_manifest": "valid.jsonl"}}}
        settings, recipe = self.scope["checkpoint_recipe"](payload, config)
        self.assertEqual(settings["input_space_policy"], "mesh_query_bbox")
        self.assertEqual([row["lr"] for row in recipe], [1e-4] * 3)
        self.assertEqual([row["checkpoint_final_lr_not_used"] for row in recipe], [2e-6] * 3)
        for key, value in (("step", 1666), ("sample_seen", 80000)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.scope["checkpoint_recipe"]({**payload, key: value}, config)
        for key, value in (("use_time_embedding", True), ("freeze_ar", True), ("train_surface_tokenizer", False),
                           ("target_active_skin_only", True), ("lr_motion", float("nan")), ("val_manifest", "other.jsonl")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.scope["checkpoint_recipe"]({**payload, "args": {**saved, key: value}}, config)
        groups[0]["params"].pop()
        with self.assertRaises(ValueError):
            self.scope["checkpoint_recipe"](payload, config)

    def test_selection_reads_at_most_32_samples_and_retains_only_extremes(self):
        class Dataset:
            def __init__(self):
                self.calls = []
                self.live = weakref.WeakValueDictionary()

            def __len__(self):
                return 40

            def __getitem__(self, index):
                self.calls.append(index)
                value = sample(index, tokens=100 if index in (27, 39) else 12,
                               vertices=30 if index in (7, 39) else 8)
                self.live[index] = value
                return value

        dataset = Dataset()
        selected, report = self.scope["select_stress_samples"](dataset, 32)
        gc.collect()
        self.assertEqual(dataset.calls, list(range(32)))
        self.assertEqual(set(dataset.live), {7, 27})
        self.assertEqual([value.path for value in selected], ["asset_27.npz", "asset_7.npz"])
        self.assertEqual(report["unique_retained_samples"], 2)
        self.assertEqual(report["inspected_rows"], 32)
        self.assertEqual(report["selected"]["longest_input"]["input_tokens"], 100)
        self.assertEqual(report["selected"]["largest_mesh"]["vertices"], 30)

    def test_selection_ties_short_and_invalid_datasets(self):
        selected, report = self.scope["select_stress_samples"]([sample(0), sample(1)], 32)
        self.assertIs(selected[0], selected[1])
        self.assertEqual(report["unique_retained_samples"], 1)
        self.assertEqual(report["inspected_rows"], 2)
        for dataset, limit in (([], 32), ([sample()], 33), ([sample()], 0)):
            with self.assertRaises(ValueError):
                self.scope["select_stress_samples"](dataset, limit)
        original = sample()
        for invalid in (replace(original, frame_vertices=original.frame_vertices[:2]),
                        replace(original, attention_mask=torch.zeros_like(original.attention_mask)),
                        replace(original, target_parents=original.target_parents[:2])):
            with self.assertRaises(ValueError):
                self.scope["sample_summary"](invalid, 0)

    def test_frame_slicing_and_batch_repetition_preserve_full_gt_and_query(self):
        original = sample()
        frames_before = original.frame_vertices.clone()
        for frames in (2, 4, 24):
            sliced = self.scope["slice_sample"](original, frames)
            self.assertEqual(sliced.frame_vertices.shape[0], frames)
            self.assertEqual(sliced.vertex_normals.shape[0], frames)
            self.assertEqual(sliced.face_normals.shape[0], frames)
            self.assertTrue(torch.equal(sliced.frame_vertices, original.frame_vertices[:frames]))
            self.assertTrue(torch.equal(sliced.selected_frames, original.selected_frames[:frames]))
            for name in ("input_ids", "target_joints", "target_parents", "query_center", "query_scale", "faces"):
                self.assertIs(getattr(sliced, name), getattr(original, name))
        self.assertTrue(torch.equal(original.frame_vertices, frames_before))
        samples = [sample(0), sample(1)]
        for index in (0, 1):
            batch = self.scope["stress_batch"](samples, 2, 1, index, lambda rows: rows)
            self.assertEqual(batch[0].path, samples[index].path)
        batch = self.scope["stress_batch"](samples, 6, 3, 0, lambda rows: rows)
        self.assertEqual([row.path for row in batch], ["asset_0.npz", "asset_1.npz", "asset_0.npz"])
        for frames in (1, 25, True):
            with self.assertRaises(ValueError):
                self.scope["slice_sample"](original, frames)

    def test_approval_requires_threshold_two_backward_passes_and_all_finite(self):
        classify = self.scope["classify_candidate"]
        self.assertTrue(classify(measured(24, 3, peak=90), 100, 0.90)["approved"])
        self.assertFalse(classify(measured(24, 3, peak=91), 100, 0.90)["approved"])
        changes = [("finite_loss", False), ("finite_gradients", False), ("finite_updates", False),
                   ("status", "oom"), ("backward_passes", 1), ("peak_allocated_bytes", float("nan")),
                   ("optimizer_steps", 0), ("peak_allocated_bytes", -1), ("peak_allocated_bytes", True)]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                self.assertFalse(classify({**measured(24, 3), key: value}, 100, 0.90)["approved"])

    def test_summary_does_not_extrapolate_untested_caps_or_frames(self):
        current_plan = plan()
        rows = [{**measured(frames, batch), "approved": True} for frames, batch in current_plan["cases"]]
        summary = self.scope["summarize_cases"](current_plan, rows)
        self.assertEqual(summary["largest_approved_measured_batch"], 12)
        self.assertEqual(summary["measured_global_cap"], 12)
        self.assertIsNone(summary["full_t2_t24_global_cap"])
        self.assertIn(23, summary["unmeasured_frames"])
        partial = self.scope["summarize_cases"](current_plan, rows[:4])
        self.assertEqual(partial["largest_approved_measured_batch"], 6)
        self.assertIsNone(partial["measured_global_cap"])
        self.assertFalse(partial["all_candidates_approved"])
        self.assertEqual(partial["unattempted_cases"], current_plan["cases"][4:])
        rows[4]["approved"] = False
        self.assertIsNone(self.scope["summarize_cases"](current_plan, rows)["measured_global_cap"])

    def test_first_oom_is_unsafe_measurement_and_stops_without_retry(self):
        cleanup, record = Mock(), Mock()
        self.scope["memory_snapshot"] = lambda: {"peak_allocated_bytes": 92, "peak_reserved_bytes": 93}
        measure = Mock(side_effect=[measured(24, 3), torch.OutOfMemoryError("planned test OOM"), measured(14, 5)])
        rows, reason = self.scope["run_candidates"](plan(), measure, cleanup, 100, record)
        self.assertEqual(measure.call_count, 2)
        self.assertEqual(cleanup.call_count, 2)
        self.assertEqual(record.call_count, 2)
        self.assertEqual(reason, "first_oom_no_retry")
        self.assertTrue(rows[0]["approved"])
        self.assertEqual(rows[1]["status"], "oom")
        self.assertEqual(rows[1]["decision"], "unsafe_oom")
        self.assertFalse(rows[1]["approved"])
        self.assertIn("planned test OOM", rows[1]["error"])

    def test_non_oom_errors_propagate(self):
        for error in (ValueError("bad sample"), RuntimeError("CUDA error, not an OOM type"), FloatingPointError("nan")):
            with self.subTest(error=type(error)), self.assertRaises(type(error)):
                self.scope["run_candidates"](plan(), Mock(side_effect=error), Mock(), 100, Mock())

    def test_trainable_groups_keep_disabled_branches_frozen(self):
        model = TinyRoutes()
        recipe = [{"name": name, "lr": 1e-4, "weight_decay": 0.04} for name in ("motion", "ar", "surface")]
        groups = self.scope["trainable_groups"](model, recipe)
        self.assertEqual([len(group["params"]) for group in groups], [304, 388, 196])
        self.assertTrue(model.conditioner.motion_encoder.gradient_checkpointing)
        self.assertTrue(model.training)
        self.assertFalse(model.conditioner.motion_encoder.time_embed.requires_grad)
        self.assertFalse(any(p.requires_grad for p in model.conditioner.motion_encoder.motion_feature_mlp.parameters()))
        model.conditioner.motion_encoder.time_embed.requires_grad_(True)
        with self.assertRaises(RuntimeError):
            self.scope["trainable_groups"](model, recipe)

    def test_cpu_model_exercises_two_backward_passes_with_resident_adam_and_gradients(self):
        model = TinyRoutes()
        recipe = [{"name": name, "lr": 1e-4, "weight_decay": 0.04} for name in ("motion", "ar", "surface")]
        groups = self.scope["trainable_groups"](model, recipe)
        optimizer = torch.optim.AdamW(groups, weight_decay=0.04)
        cpu = CpuTorch()
        self.scope.update(torch=cpu, move_batch=lambda batch, device: batch)
        samples = [sample(0), sample(1)]
        warmup = self.scope["measure_step"](model, optimizer, samples, lambda rows: rows, 2, 1, warmup=True)
        self.assertEqual(warmup["backward_passes"], 2)
        self.assertEqual(len(optimizer.state), 888)
        row = self.scope["measure_step"](model, optimizer, samples, lambda rows: rows, 24, 3)
        self.assertEqual([item["gradients_present"] for item in row["before_pass"]], [0, 888])
        self.assertEqual([item["adam_states_present"] for item in row["before_pass"]], [888, 888])
        self.assertEqual(row["backward_passes"], 2)
        self.assertEqual(row["optimizer_steps"], 1)
        self.assertTrue(row["finite_updates"])
        self.assertEqual(cpu.cuda.reset_peak_memory_stats.call_count, 2)
        self.assertTrue(all(update["max_abs"] > 0 for update in row["group_updates"].values()))
        self.assertFalse(model.conditioner.motion_encoder.time_embed.requires_grad)

    def test_source_guards_precede_cuda_and_no_checkpoint_save_or_compile(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
        cuda_lines = [node.lineno for node in calls if ast.unparse(node.func).startswith("torch.cuda.")]
        for name in ("guard", "source_metadata", "verify_manifests"):
            line = next(node.lineno for node in calls if ast.unparse(node.func) == name)
            self.assertLess(line, min(cuda_lines))
        guard_call = next(node for node in calls if ast.unparse(node.func) == "guard")
        self.assertEqual(ast.literal_eval(guard_call.args[1]), "preflight")
        all_calls = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
        self.assertNotIn("torch.save", all_calls)
        self.assertNotIn("torch.compile", all_calls)
        handlers = [node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)]
        self.assertEqual([ast.unparse(node.type) for node in handlers], ["torch.OutOfMemoryError"])


if __name__ == "__main__":
    unittest.main()
