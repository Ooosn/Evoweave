"""Small CPU fixtures for the original checkpoint's matched-panel adapter."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_original_baseline_matched.py"
SPEC = importlib.util.spec_from_file_location("original_baseline_matched_under_test", SCRIPT)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def geometry(identity="asset"):
    vertices = torch.arange(96, dtype=torch.float32).reshape(1, 8, 4, 3) / 100
    targets = torch.arange(72, dtype=torch.float32).reshape(1, 8, 3, 3) / 100
    return {
        "frame_vertices": vertices, "vertex_normals": vertices + 1,
        "faces": torch.tensor([[[0, 1, 2], [1, 2, 3]]]),
        "face_normals": torch.arange(48, dtype=torch.float32).reshape(1, 8, 2, 3),
        "frame_target_joints": targets, "target_joints": targets[:, 0].clone(),
        "target_parents": torch.tensor([[-1, 0, 1]]), "joint_count": torch.tensor([3]),
        "joint_mask": torch.ones(1, 3, dtype=torch.bool), "frame_mask": torch.ones(1, 8, dtype=torch.bool),
        "vertex_count": torch.tensor([4]), "face_count": torch.tensor([2]),
        "selected_frames": torch.arange(8).reshape(1, 8),
        "frame_numbers": (100 + torch.arange(8)).reshape(1, 8),
        "frame_mesh_bbox_side": torch.ones(1, 8),
        "query_center": torch.tensor([[3., 4., 5.]]), "query_scale": torch.tensor([2.]),
        "root_rt_rotations": torch.eye(3).repeat(1, 8, 1, 1),
        "root_rt_translations": torch.arange(24, dtype=torch.float32).reshape(1, 8, 3),
        "root_rt_augmentation": torch.eye(3)[None], "root_rt_root_index": torch.tensor([0]),
        "root_rt_relative_nonrigid": torch.tensor([False]),
        "asset_id": identity, "path": ["/immutable/asset.npz"], "cls": ["articulationxl"],
        "sampling_seed": 123, "heldout_raw_ids": [8, 9], "frame_partition": "seen_eval",
    }


def successful_rows(identity="a", gt_count=3):
    return [{"asset_id": identity, "frame_index": slot, "success": True,
             "has_eos": True, "parse_ok": True, "status": "eos", "gt_joint_count": gt_count,
             "original_edge_f1": .8, "reference_B65_f1": .6,
             "exact_count": True, "exact_parent_list": True, "structure_signature": [-1, 0, 1]}
            for slot in evaluation.SLOTS]


class MatchedBaselineTests(unittest.TestCase):
    def test_all_query_permutations_preserve_window_bbox_rt_and_input(self):
        original = geometry()
        before = evaluation.geometry_digest(original)
        for slot in evaluation.SLOTS:
            query, order = evaluation.permute_query(original, slot)
            self.assertEqual(order, [slot] + [i for i in range(8) if i != slot])
            self.assertEqual(query["selected_frames"][0].tolist(), order)
            self.assertTrue(torch.equal(query["target_joints"], original["frame_target_joints"][:, slot]))
            for key in ("query_center", "query_scale", "target_parents", "root_rt_augmentation"):
                self.assertTrue(torch.equal(query[key], original[key]), key)
            for key in evaluation.FRAME_FIELDS:
                self.assertTrue(torch.equal(query[key], original[key][:, order]), key)
            query["frame_vertices"].add_(100)
            self.assertEqual(evaluation.geometry_digest(original), before)

    def test_every_required_geometry_field_fails_closed_when_missing(self):
        for key in evaluation.REQUIRED_FIELDS:
            with self.subTest(key=key):
                batch = geometry()
                del batch[key]
                with self.assertRaisesRegex(ValueError, "missing fields"):
                    evaluation.permute_query(batch, 2)

    def test_bbox_parent_normals_and_rt_drift_are_rejected(self):
        original = geometry()
        for key in ("query_center", "query_scale", "target_parents", "vertex_normals",
                    "root_rt_translations", "root_rt_augmentation", "frame_numbers", "frame_mesh_bbox_side"):
            with self.subTest(key=key):
                query, _ = evaluation.permute_query(original, 4)
                query[key] = query[key].clone()
                query[key].view(-1)[-1] += 1
                with self.assertRaises(ValueError):
                    evaluation.verify_permutation(original, query, 4)

    def test_invalid_geometry_or_unapproved_slot_fails(self):
        for change in (lambda b: b["selected_frames"].fill_(0),
                       lambda b: b["target_joints"].add_(.1),
                       lambda b: b["frame_mask"].fill_(False),
                       lambda b: b["frame_target_joints"].fill_(float("nan"))):
            batch = geometry()
            change(batch)
            with self.assertRaises(ValueError):
                evaluation.permute_query(batch, 2)
        for slot in (-1, 1, 8, True):
            with self.assertRaises(ValueError):
                evaluation.permute_query(geometry(), slot)

    def test_reference_checks_all_eight_frames_and_tokens_without_mutation(self):
        batch = geometry()
        targets = [[256, 260, frame, 257] for frame in range(8)]
        reference = {("valid", "asset", frame): {"raw_frame_index": frame,
            "target_tokens": tokens, "original_edge_f1": .5} for frame, tokens in enumerate(targets)}
        before = deepcopy(reference)
        evaluation.verify_reference(batch, targets, reference)
        self.assertEqual(reference, before)
        for field, value in (("raw_frame_index", 99), ("target_tokens", [1]), ("original_edge_f1", float("nan"))):
            changed = deepcopy(reference)
            changed[("valid", "asset", 7)][field] = value
            with self.assertRaises(ValueError):
                evaluation.verify_reference(batch, targets, changed)
        del reference[("valid", "asset", 1)]
        with self.assertRaisesRegex(ValueError, "missing"):
            evaluation.verify_reference(batch, targets, reference)

    def test_coverage_rejects_duplicates_missing_and_extra_slots(self):
        rows = successful_rows()
        evaluation.validate_coverage(rows, ["a"])
        for changed in (rows[:-1], rows + [rows[0]], rows[:-1] + [rows[0]],
                        rows[:-1] + [{**rows[-1], "frame_index": 1}]):
            with self.assertRaisesRegex(ValueError, "views"):
                evaluation.validate_coverage(changed, ["a"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluation.validate_coverage(rows, ["a", "a"])

    def test_failure_zero_macro_average_strata_and_consistency_denominators(self):
        rows = successful_rows("a", 52) + successful_rows("b", 3)
        rows[0].update(success=False, has_eos=False, parse_ok=False, status="max_tokens",
                       original_edge_f1=0., exact_count=False, exact_parent_list=False,
                       structure_signature=None)
        result = evaluation.summarize(rows, ["a", "b"])
        self.assertAlmostEqual(result["gt52"]["original_edge_f1"], .6)
        self.assertAlmostEqual(result["non52"]["original_edge_f1"], .8)
        self.assertAlmostEqual(result["all"]["original_edge_f1"], .7)
        self.assertEqual(result["all"]["failed_generations"], 1)
        self.assertEqual(result["all"]["parse_failures"], 1)
        self.assertEqual(result["all"]["eos_failures"], 1)
        self.assertEqual(result["all"]["all_four_same_parent_and_count"], .5)
        self.assertEqual(result["all"]["pairwise_same_parent_and_count"], .75)
        self.assertEqual(result["gt52"]["pairwise_same_parent_and_count"], .5)
        rows[0]["original_edge_f1"] = .1
        with self.assertRaisesRegex(ValueError, "failed generations"):
            evaluation.summarize(rows, ["a", "b"])

    def test_equal_counts_with_different_parents_are_not_consistent(self):
        rows = successful_rows()
        rows[0]["structure_signature"] = [-1, 0, 0]
        result = evaluation.summarize(rows, ["a"])["all"]
        self.assertEqual(result["all_four_same_parent_and_count"], 0.)
        self.assertEqual(result["pairwise_same_parent_and_count"], .5)

    def test_context_budget_counts_condition_prefix_and_new_tokens(self):
        report = evaluation.context_budget(3076, 2048)
        self.assertEqual(report["reserved_positions"], 3074)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            evaluation.context_budget(3073, 2048)
        for value in (0, -1, True, 3076.):
            with self.assertRaises(ValueError):
                evaluation.context_budget(value, 2048)

    def test_strict_restore_checks_every_tensor_and_keeps_source_state(self):
        model = torch.nn.Linear(2, 3)
        state = {key: torch.ones_like(value) for key, value in model.state_dict().items()}
        before = {key: value.clone() for key, value in state.items()}
        with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")):
            report = evaluation.strict_restore(model, state)
        self.assertTrue(report["strict"])
        self.assertTrue(report["all_tensors_equal"])
        self.assertEqual(report["model_parameters"], 9)
        for key in state:
            self.assertTrue(torch.equal(state[key], before[key]))
        for changed in ({"weight": state["weight"]}, {**state, "unknown": torch.zeros(1)},
                        {**state, "weight": torch.zeros(1)}):
            with self.assertRaises(RuntimeError):
                evaluation.strict_restore(model, changed)

    def test_cpu_yaml_override_has_exactly_one_semantic_difference(self):
        config = {"llm": {"_attn_implementation": "flash_attention_2", "nested": {"a": [1, 2]}},
                  "mesh_encoder": {"depth": 12}, "unchanged": True}
        with tempfile.TemporaryDirectory() as directory:
            source, derived = Path(directory) / "original.yaml", Path(directory) / "cpu.yaml"
            source.write_text(yaml.safe_dump(config))
            original = source.read_bytes()
            report = evaluation.derive_cpu_yaml(source, derived)
            result = yaml.safe_load(derived.read_text())
            self.assertEqual(result["llm"].pop("_attn_implementation"), "sdpa")
            result["llm"]["_attn_implementation"] = "flash_attention_2"
            self.assertEqual(result, config)
            self.assertEqual(source.read_bytes(), original)
            self.assertTrue(report["all_other_fields_recursively_equal"])
            self.assertEqual(report["original"]["sha256"], evaluation.sha256(source))
            with self.assertRaises(FileExistsError):
                evaluation.derive_cpu_yaml(source, derived)

    def test_existing_report_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            evaluation.write_json(path, {"value": 1})
            with self.assertRaises(FileExistsError):
                evaluation.write_json(path, {"value": 2})
            self.assertEqual(json.loads(path.read_text()), {"value": 1})

    def test_local_sdpa_is_explicit_and_does_not_affect_original_flash_route(self):
        for flash in (True, False):
            self.assertEqual(evaluation.construction_backend({}, "audit", flash)["backend"], "sdpa")
        self.assertEqual(evaluation.construction_backend({}, "smoke", True)["backend"], "flash_attention_2")
        self.assertEqual(evaluation.construction_backend({"attention_backend": "sdpa"}, "evaluate", False,
                                                        platform="nt")["backend"], "sdpa")
        for runtime, flash, platform in (({}, False, "nt"), ({"attention_backend": "sdpa"}, True, "nt"),
                                          ({"attention_backend": "sdpa"}, False, "posix"),
                                          ({"attention_backend": "unknown"}, False, "nt")):
            with self.assertRaisesRegex(ValueError, "backend"):
                evaluation.construction_backend(runtime, "smoke", flash, platform)

    def test_relocated_asset_paths_keep_saved_args_and_nonpath_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            saved = {"model_config": "/remote/model.yaml", "tokenizer_config": "/remote/tokenizer.yaml",
                     "unirig_checkpoint": "/remote/upstream.ckpt", "frames": 24, "motion_depth": 12}
            before = deepcopy(saved)
            runtime = {key: str(Path(directory) / name) for key, name in
                       (("model_config", "model.yaml"), ("tokenizer_config", "tokenizer.yaml"),
                        ("unirig_checkpoint", "upstream.ckpt"))}
            settings, report = evaluation.checkpoint_settings(saved, {"motion_depth": 6}, runtime, "weights.pt")
            self.assertEqual(settings["frames"], 8)
            self.assertEqual(settings["motion_depth"], 12)
            self.assertEqual(saved, before)
            for key in ("model_config", "tokenizer_config"):
                self.assertEqual(settings[key], Path(runtime[key]))
                self.assertEqual(report[key], {"saved": saved[key], "effective": runtime[key]})
            self.assertIsNone(settings["unirig_checkpoint"])
            self.assertFalse(report["unirig_checkpoint"]["loaded"])

    def test_artifact_map_verifies_exact_bytes_and_only_relocates_panel_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = ["/remote/train.jsonl", "/remote/valid.jsonl", "/remote/selection.json",
                     "/remote/rank_0.json", "/remote/rank_1.json", "/remote/summary.json"]
            artifacts = {}
            for index, original in enumerate(paths):
                local = root / f"{index}.json"
                local.write_text(json.dumps({"path": "/canonical/source.npz", "id": index}))
                artifacts[original] = {"path": str(local), "sha256": evaluation.sha256(local),
                                       "bytes": local.stat().st_size}
            specifications = [{"path": original, "sha256": artifacts[original]["sha256"], "unchanged": True}
                              for original in paths]
            panel = {"data": {"manifests": dict(zip(("train", "valid"), specifications[:2])),
                              "surface_seed": 20260529},
                     "reference_evaluation": {"files": dict(zip(("selection", "rank_0", "rank_1", "summary"), specifications[2:]))},
                     "unrelated": {"path": "/keep/this/path"}}
            original = deepcopy(panel)
            verified = {}
            relocated = evaluation.relocate_panel_config(panel, artifacts, verified)
            self.assertEqual(panel, original)
            self.assertEqual(relocated["unrelated"], panel["unrelated"])
            self.assertEqual(relocated["data"]["surface_seed"], 20260529)
            recovered = deepcopy(relocated)
            for before, after in zip(specifications, [*recovered["data"]["manifests"].values(),
                                                      *recovered["reference_evaluation"]["files"].values()]):
                self.assertEqual(evaluation.sha256(after["path"]), before["sha256"])
                self.assertIn("/canonical/source.npz", Path(after["path"]).read_text())
                after["path"] = before["path"]
            self.assertEqual(recovered, original)
            self.assertEqual(len(verified), 6)
            for altered in ({**artifacts[paths[0]], "bytes": 1},
                            {**artifacts[paths[0]], "sha256": "f" * 64}):
                with self.assertRaisesRegex(ValueError, "bytes/hash"):
                    evaluation.local_artifact(paths[0], {paths[0]: altered}, {})
            with self.assertRaisesRegex(ValueError, "lacks"):
                evaluation.local_artifact("/missing", artifacts, {})

    def test_memory_limit_uses_local_contract_and_rejects_conflicts(self):
        self.assertEqual(evaluation.memory_fraction({"gpu_memory_fraction_limit": .45}), .45)
        for runtime in ({"gpu_memory_fraction_limit": .9}, {"gpu_memory_fraction_limit": 0},
                        {"gpu_memory_fraction_limit": True},
                        {"gpu_memory_fraction_limit": .45, "cuda_memory_fraction": .85}):
            with self.assertRaises(ValueError):
                evaluation.memory_fraction(runtime)

    def test_external_artifact_manifest_expands_only_a_deepcopy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "download_manifest.json"
            mapping = {"/remote/asset.npz": {"path": str(Path(directory) / "asset.npz"),
                                            "sha256": "a" * 64, "bytes": 12}}
            raw = json.dumps({"complete": True, "local_artifacts": mapping}, indent=4).encode() + b"\n"
            path.write_bytes(raw)
            config = {"local_artifacts_manifest": {"path": str(path), "sha256": evaluation.sha256(path)},
                      "runtime": {"unchanged": [1, 2]}}
            before = deepcopy(config)
            expanded, identity = evaluation.expand_local_artifacts(config)
            self.assertEqual(expanded["local_artifacts"], mapping)
            self.assertEqual(identity["sha256"], evaluation.sha256(path))
            self.assertEqual(identity["bytes"], len(raw))
            expanded["runtime"]["unchanged"].append(3)
            expanded["local_artifacts"].clear()
            self.assertEqual(config, before)
            self.assertEqual(path.read_bytes(), raw)

    def test_external_artifact_manifest_rejects_tamper_and_incomplete_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "download_manifest.json"
            path.write_text(json.dumps({"complete": True, "local_artifacts": {}}))
            config = {"local_artifacts_manifest": {"path": str(path), "sha256": evaluation.sha256(path)}}
            path.write_text(json.dumps({"complete": True, "local_artifacts": {"tampered": {}}}))
            with self.assertRaisesRegex(ValueError, "raw SHA256"):
                evaluation.expand_local_artifacts(config)
            for payload in ({"complete": False, "local_artifacts": {}},
                            {"complete": 1, "local_artifacts": {}},
                            {"complete": True, "local_artifacts": []},
                            {"complete": True}, []):
                path.write_text(json.dumps(payload))
                config["local_artifacts_manifest"]["sha256"] = evaluation.sha256(path)
                with self.assertRaisesRegex(ValueError, "complete.*dictionary"):
                    evaluation.expand_local_artifacts(config)

    def test_inline_and_external_artifact_forms_are_always_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "cannot both"):
            evaluation.expand_local_artifacts({"local_artifacts": {},
                "local_artifacts_manifest": {"path": "not-opened", "sha256": "a" * 64}})

    def test_dataset_resolver_keeps_original_npz_and_rt_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            originals = ("/original/asset.npz", "/original/root_rt.npz")
            artifacts = {}
            for index, original in enumerate(originals):
                local = Path(directory) / f"{index}.npz"
                local.write_bytes(f"exact-original-{index}".encode())
                artifacts[original] = {"path": str(local), "sha256": evaluation.sha256(local),
                                       "bytes": local.stat().st_size}
            rows = [{"asset_id": "a", "path": originals[0],
                     "root_rt": {"cache_path": originals[1], "index": 3, "padding": 1.1}}]
            before = deepcopy(rows)
            calls = []
            def fixed_dataset(actual_rows, panel, frames, *, mode, file_resolver):
                self.assertIs(actual_rows, rows)
                self.assertEqual(frames, 8)
                self.assertEqual(mode, "seen_eval")
                row = actual_rows[0]
                calls.append(file_resolver(row["path"]))
                calls.append(file_resolver(row["root_rt"]["cache_path"]))
                return SimpleNamespace(path=row["path"], resolved_path=calls[0])
            verified = {}
            module = SimpleNamespace(fixed_dataset=fixed_dataset)
            with patch.dict(sys.modules, {"virtual_query_xyz_runtime": module}):
                result = evaluation.matched_dataset(rows, {"panel": "unchanged"},
                                                    {"local_artifacts": artifacts}, verified)
            self.assertEqual(rows, before)
            self.assertEqual(result.path, originals[0])
            self.assertEqual(result.resolved_path, str(Path(artifacts[originals[0]]["path"]).resolve()))
            self.assertEqual(set(verified), set(originals))
            for original in originals:
                self.assertEqual(verified[original]["original_path"], original)
                self.assertEqual(verified[original]["sha256"], artifacts[original]["sha256"])
                self.assertIn("mtime_ns", verified[original])

    def test_resolved_artifact_mtime_drift_fails_cached_open_and_final_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.npz"
            path.write_bytes(b"data")
            original = "/original/data.npz"
            artifacts = {original: {"path": str(path), "bytes": 4, "sha256": evaluation.sha256(path)}}
            verified = {}
            evaluation.local_artifact(original, artifacts, verified)
            record = verified[original]
            os.utime(path, ns=(record["mtime_ns"], record["mtime_ns"] + 1000000000))
            with self.assertRaisesRegex(RuntimeError, "mtime changed"):
                evaluation.local_artifact(original, artifacts, verified)
            with self.assertRaisesRegex(RuntimeError, "mtime changed"):
                evaluation.assert_files_unchanged({str(path): record}, record)

    def test_final_provenance_hash_detects_same_size_change_with_restored_mtime(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.npz"
            path.write_bytes(b"data")
            record = evaluation.file_identity(path)
            path.write_bytes(b"evil")
            os.utime(path, ns=(record["mtime_ns"], record["mtime_ns"]))
            with self.assertRaisesRegex(RuntimeError, "changed during evaluation"):
                evaluation.assert_files_unchanged({str(path): record}, record)

    def test_audit_orchestration_never_calls_cuda_or_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "reference.json"
            source.write_text("{}")
            spec = {"path": str(source), "sha256": evaluation.sha256(source)}
            panel = {"evaluation": {"input_frames": 8, "frame_indices": list(evaluation.SLOTS),
                                    "valid_assets": 16, "seed": 20261022, "max_new_tokens": 2048},
                     "data": {"normalization": "input_sequence_mesh_bbox", "global_rt_removal": True,
                              "coordinate_clipping": False, "manifests": {"train": spec, "valid": spec}},
                     "reference_evaluation": {"files": {name: spec for name in ("selection", "rank_0", "rank_1", "summary")}}}
            panel_path = root / "panel.json"
            panel_path.write_text(json.dumps(panel))
            checkpoint = root / "tiny_checkpoint_fixture"
            checkpoint.write_bytes(b"immutable fixture")
            config = {"family": evaluation.FAMILY,
                      "panel_config": {"path": str(panel_path), "sha256": evaluation.sha256(panel_path)},
                      "checkpoint": {"path": str(checkpoint), "sha256": evaluation.CHECKPOINT_SHA,
                                     "bytes": evaluation.CHECKPOINT_BYTES, "step": 1667, "sample_seen": 80016},
                      "runtime": {"repo": str(evaluation.REPO), "python": sys.executable,
                                  "unirig_root": str(root), "opt_config_root": str(root),
                                  "extra_pythonpath": str(root)},
                      "evaluation": {"input_frames": 8, "frame_slots": list(evaluation.SLOTS),
                                     "valid_assets": 16, "seed": 20261022, "max_new_tokens": 2048},
                      "reference_B65": panel["reference_evaluation"]}
            original_config = deepcopy(config)
            evaluation.validate_config(config, panel)
            self.assertEqual(config, original_config)
            for field, value in (("valid_assets", 15), ("input_frames", 24), ("frame_slots", [0, 1, 2, 3])):
                changed = deepcopy(config)
                changed["evaluation"][field] = value
                with self.assertRaises(ValueError):
                    evaluation.validate_config(changed, panel)
            config_path, output = root / "config.json", root / "audit"
            artifact_manifest = root / "download_manifest.json"
            artifact_manifest.write_text(json.dumps({"complete": True, "local_artifacts": {
                str(source): {"path": str(source), "sha256": evaluation.sha256(source),
                              "bytes": source.stat().st_size}}}))
            config["local_artifacts_manifest"] = {"path": str(artifact_manifest),
                                                   "sha256": evaluation.sha256(artifact_manifest)}
            config["runtime"]["device_index"] = 0
            config_path.write_text(json.dumps(config))
            raw_config_bytes = config_path.read_bytes()
            identities = [f"asset{i}" for i in range(16)]
            audits = [{"asset_id": identity} for identity in identities]
            metadata = {"checkpoint": evaluation.file_identity(checkpoint), "sources": {},
                        "asset_ids": identities, "audited_assets": audits, "panel": {"fixture": True},
                        "strict_restoration": {"strict": True}, "context_budget": evaluation.context_budget(3076, 2048)}
            work = [(index, [], audit) for index, audit in enumerate(audits)]
            prepared = (None, None, None, None, work, metadata)
            def verify_stage(args, guarded_config):
                self.assertEqual(guarded_config, config)
                self.assertNotIn("local_artifacts", guarded_config)
                return "fixture-revision"
            guard = SimpleNamespace(verify_stage=verify_stage)
            argv = [str(SCRIPT), "--config", str(config_path), "--output", str(output), "--mode", "audit"]
            with patch.dict(sys.modules, {"original_baseline_eval_guard": guard}), patch.object(sys, "argv", argv), \
                    patch.object(sys, "path", list(sys.path)), \
                    patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0"}), \
                    patch.object(evaluation, "prepare_cpu", return_value=prepared), \
                    patch.object(evaluation, "generate_rows", side_effect=AssertionError("generation forbidden")), \
                    patch.object(torch.cuda, "set_device", side_effect=AssertionError("CUDA forbidden")), \
                    patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")):
                evaluation.main()
            result = json.loads((output / "summary.json").read_text())
            self.assertTrue(result["complete"])
            self.assertEqual(result["generated_views"], 0)
            self.assertEqual(result["audited_assets"], 16)
            self.assertFalse(result["quality_approval"])
            self.assertEqual(checkpoint.read_bytes(), b"immutable fixture")
            self.assertEqual(config_path.read_bytes(), raw_config_bytes)
            report = json.loads((output / "rank_0.json").read_text())
            self.assertEqual(report["input_files"][str(config_path.resolve())]["sha256"],
                             evaluation.sha256(config_path))
            self.assertEqual(report["input_files"][str(artifact_manifest.resolve())]["sha256"],
                             config["local_artifacts_manifest"]["sha256"])

    def test_generation_scores_continuous_gt_at_original_slot(self):
        batch = geometry()
        gt = batch["frame_target_joints"][0, 4].numpy().copy()
        tokenizer = SimpleNamespace(bos=256, eos=257, cls_name_to_token=lambda _: 260,
            detokenize=lambda _: SimpleNamespace(joints=gt, parents=[None, 0, 1]))
        calls = []
        def metric(joints, parents, target, target_parents):
            calls.append(target.copy())
            np.testing.assert_array_equal(target, gt)
            self.assertEqual(target_parents, [-1, 0, 1])
            return {"edge_f1": .75}
        order = [4, 0, 1, 2, 3, 5, 6, 7]
        row = evaluation.score_generation([256, 260, 1, 257], tokenizer, batch, 4, order, .6, metric)
        self.assertTrue(row["success"])
        self.assertEqual(row["original_edge_f1"], .75)
        self.assertEqual(row["raw_frame_index"], 4)
        self.assertEqual(row["permuted_raw_frames"], order)
        self.assertEqual(row["structure_signature"], [-1, 0, 1])
        row = evaluation.score_generation([256, 260, 1], tokenizer, batch, 4, order, .6, metric)
        self.assertFalse(row["success"])
        self.assertTrue(row["parse_ok"])
        self.assertEqual(row["original_edge_f1"], 0.)
        self.assertIsNone(row["structure_signature"])
        self.assertEqual(len(calls), 1)
        tokenizer.detokenize = lambda _: (_ for _ in ()).throw(ValueError("bad parse"))
        row = evaluation.score_generation([256, 260, 257], tokenizer, batch, 4, order, .6, metric)
        self.assertEqual(row["status"], "parse_failure")
        self.assertEqual(row["original_edge_f1"], 0.)

    def test_guard_rejection_precedes_output_creation_and_stage_work(self):
        with tempfile.TemporaryDirectory() as directory:
            config, output = Path(directory) / "config.json", Path(directory) / "output"
            config.write_text(json.dumps({"family": evaluation.FAMILY}))
            guard = SimpleNamespace(verify_stage=lambda *_: (_ for _ in ()).throw(RuntimeError("not authorized")))
            argv = [str(SCRIPT), "--config", str(config), "--output", str(output), "--mode", "audit"]
            with patch.dict(sys.modules, {"original_baseline_eval_guard": guard}), patch.object(sys, "argv", argv), \
                    patch.object(evaluation, "prepare_cpu", side_effect=AssertionError("stage work forbidden")), \
                    patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")):
                with self.assertRaisesRegex(RuntimeError, "not authorized"):
                    evaluation.main()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
