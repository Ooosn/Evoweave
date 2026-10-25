import copy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BASE / "src"), str(BASE / "scripts")]
from diagnose_id_xyz_checkpoint import gt_structure_free_xyz, replay_with_native_evaluator, structural_divergence, summarize, tensor_difference, tensor_identity, verify_geometry
from query_diagnostics_guard import CPUS, FAMILY, SCOPE, execution_cpu_binding, validate_ssh_exception
from rigweave.dynamic_rig.id_xyz_diagnostics import classify_gt_roles, first_divergence_summary, teacher_forced_diagnostics
from rigweave.dynamic_rig.id_xyz_tokens import IdXYZTokens


def codec():
    return IdXYZTokens(267, tuple(range(256)), bos=257, eos=258, pad=259,
        branch=256, class_token=266, max_joints=16, max_sequence_tokens=128)


def target():
    return codec().encode([-1, 0, 1, 0], np.zeros((1, 4, 3), np.float32))["input_ids"][0].tolist()


class Decoder:
    def __init__(self, invalid=False):
        self.embedding = torch.nn.Embedding(codec().vocab_size, 4)
        self.calls = 0
        self.invalid = invalid

    def get_input_embeddings(self):
        return self.embedding

    def __call__(self, **kwargs):
        self.calls += 1
        scores = torch.zeros(1, 1, codec().vocab_size)
        scores[0, 0, 130] = 5
        if self.invalid:
            scores[0, 0, 0] = float("nan")
        return SimpleNamespace(logits=scores, past_key_values=("cached",))


class OracleTests(unittest.TestCase):
    def test_tensor_repetition_reports_bitwise_and_numeric_difference(self):
        first = torch.tensor([1., 2., 3.])
        second = first.clone()
        self.assertTrue(tensor_difference(first, second)["equal"])
        self.assertEqual(tensor_identity(first), tensor_identity(second))
        second[-1] += .25
        difference = tensor_difference(first, second)
        self.assertFalse(difference["equal"])
        self.assertEqual(difference["max_abs"], .25)
        with self.assertRaises(ValueError):
            tensor_difference(first, second.double())

    def test_native_replay_captures_exact_condition_without_replacing_output(self):
        class Encoder(torch.nn.Module):
            def forward(self, x):
                return x[:, 0, 2:], x
        encoder = Encoder()
        model = SimpleNamespace(motion_encoder=encoder,
            config=SimpleNamespace(register_tokens=1, query_tokens=4, dim=4))
        values = torch.arange(48.).reshape(1, 2, 6, 4)
        def evaluate(model, *args, **kwargs):
            result = model.motion_encoder(values)
            self.assertIs(result[1], values)
            return [{"frame_index": slot} for slot in (0, 2, 4, 6)]
        with patch.dict(sys.modules, {"id_xyz_evaluation": SimpleNamespace(evaluate_item=evaluate)}):
            condition, rows = replay_with_native_evaluator(model, None, codec(), None, None, None, "valid")
        self.assertTrue(torch.equal(condition, values[:, :, 2:]))
        self.assertEqual(set(rows), {0, 2, 4, 6})
        self.assertFalse(encoder._forward_hooks)

    def test_native_replay_removes_hook_after_failure(self):
        encoder = torch.nn.Identity()
        model = SimpleNamespace(motion_encoder=encoder, config=SimpleNamespace())
        def fail(*args, **kwargs):
            raise ValueError("native failure")
        with patch.dict(sys.modules, {"id_xyz_evaluation": SimpleNamespace(evaluate_item=fail)}):
            with self.assertRaisesRegex(ValueError, "native failure"):
                replay_with_native_evaluator(model, None, codec(), None, None, None, "valid")
        self.assertFalse(encoder._forward_hooks)

    def test_oracle_forces_only_structure_and_predicts_coordinates(self):
        ids, native = target(), codec()
        roles = classify_gt_roles(native, ids)
        decoder = Decoder()
        output = gt_structure_free_xyz(SimpleNamespace(xyz_decoder=decoder), native,
            torch.zeros(1, 3, 4), ids, roles)
        for index in range(2, len(ids)):
            self.assertEqual(output[index], 130 if roles[index] in ("x", "y", "z") else ids[index])
        self.assertEqual(native.decode(output)["parents"].tolist(), native.decode(ids)["parents"].tolist())
        self.assertEqual(decoder.calls, len(ids) - 2)
        self.assertTrue(structural_divergence(native, output, ids)["equal"])
        self.assertEqual(first_divergence_summary(native, output, ids)["role"], "x")

    def test_oracle_rejects_nonfinite_logits(self):
        ids = target()
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            gt_structure_free_xyz(SimpleNamespace(xyz_decoder=Decoder(True)), codec(),
                torch.zeros(1, 3, 4), ids, classify_gt_roles(codec(), ids))

    def test_structure_divergence_ignores_xyz_but_detects_branch(self):
        chain = codec().encode([-1, 0, 1, 2], np.zeros((1, 4, 3), np.float32))["input_ids"][0].tolist()
        value = structural_divergence(codec(), chain, target())
        self.assertFalse(value["equal"])
        self.assertEqual(value["role"], "branch")

    def test_summary_separates_nontrivial_choices(self):
        ids = target()
        scores = torch.full((len(ids)-1, codec().vocab_size), -5.)
        for i, token in enumerate(ids[1:]):
            scores[i, token] = 5.
        teacher = teacher_forced_diagnostics(codec(), ids, scores, legal_choice=True)
        row = {"group": "valid", "asset_id": "a", "gt_joint_count": 4,
            "original_edge_f1": .7, "oracle_original_edge_f1": .8,
            "GT_roundtrip_original_edge_f1": .9, "free_first_divergence": {"role": "x"},
            "free_structure_divergence": {"role": None}, "saved_generation_tokens_equal": True,
            "teacher": teacher, "other_pose_teacher": teacher}
        result = summarize([row])["valid"]["non52"]
        self.assertEqual(result["roles"]["class"]["nontrivial_choices"], 0)
        self.assertIsNone(result["roles"]["class"]["nontrivial_legal_accuracy"])
        self.assertEqual(result["roles"]["new_joint_id"]["nontrivial_legal_accuracy"], 1)
        self.assertTrue(result["oracle_not_model_quality"])
        for status in ("generated_has_extra_tokens", "prefix_mismatch", "uncomparable_prefix"):
            changed = copy.deepcopy(row)
            changed["free_first_divergence"] = {"status": status, "role": None}
            counts = summarize([changed])["valid"]["all"]["first_full_mismatch_roles"]
            self.assertEqual(counts, {status: 1})
        row["free_first_divergence"] = {"status": "match", "role": None}
        self.assertEqual(summarize([row])["valid"]["all"]["first_full_mismatch_roles"], {"equal": 1})


class GeometryTests(unittest.TestCase):
    def fixture(self):
        ids = target()
        geometry = {"asset_id": ["a"], "selected_frames": torch.tensor([[5, 3, 9, 1, 2, 7, 6, 4]]),
            "joint_count": torch.tensor([4]), "target_parents": torch.tensor([[-1, 0, 1, 0]]),
            "frame_target_joints": torch.zeros(1, 8, 4, 3)}
        targets = {"input_ids": torch.tensor([[ids] * 8]),
            "attention_mask": torch.ones(1, 8, len(ids), dtype=torch.bool)}
        reference = {("valid", "a", slot): {"selected_raw_frames": [5, 3, 9, 1, 2, 7, 6, 4],
            "target_tokens": ids.copy(), "gt_joint_count": 4, "original_target_parents": [-1, 0, 1, 0],
            "original_target_joints": [[0., 0., 0.]] * 4} for slot in (0, 2, 4, 6)}
        return geometry, targets, reference

    def test_exact_original_gt_check(self):
        geometry, targets, reference = self.fixture()
        self.assertEqual(list(verify_geometry(geometry, targets, "valid", reference)), [0, 2, 4, 6])

    def test_source_frame_coordinate_and_token_drift_rejected(self):
        for key, replacement in (("selected_raw_frames", [0] * 8), ("target_tokens", [1]),
                ("original_target_joints", [[1., 0., 0.]] * 4)):
            geometry, targets, reference = self.fixture()
            reference["valid", "a", 2][key] = replacement
            with self.assertRaises(ValueError):
                verify_geometry(geometry, targets, "valid", reference)


class GuardTests(unittest.TestCase):
    def fixture(self):
        exception = {"scope": SCOPE, "explicit_human_authorized": True,
            "kernel_scheduler_membership": False, "allocation": 130309880, "host": "pcg01i",
            "physical_gpus": [5, 6], "cpu_slots": 1, "execution_cpu_budget": 4, "cpu_affinity": CPUS,
            "rss_limit_bytes": 68719476736, "gpu_memory_fraction_limit": .85,
            "keeper_pid": 1090013, "authorization": "human explicitly authorized inference diagnostics"}
        state = {"active_operation": {"query_decoder_diagnostics": {"label": SCOPE,
            "explicit_human_authorized": True, "ssh_exception": exception}}}
        runtime = {**exception, "execution_mode": "explicit_ssh_exception", "require_scheduler_membership": False}
        return state, {"family": FAMILY, "runtime": runtime}

    def test_exact_exception_and_cpu_budget(self):
        state, config = self.fixture()
        exception = validate_ssh_exception(state, config)
        self.assertEqual(execution_cpu_binding(config, exception, set(range(100))), set(CPUS))

    def test_changed_or_unauthorized_resources_rejected(self):
        for key, value in (("cpu_slots", 4), ("execution_cpu_budget", 8),
                           ("cpu_affinity", [0, 1, 2, 3]), ("allocation", 1)):
            state, config = self.fixture()
            config["runtime"][key] = value
            with self.assertRaises(RuntimeError):
                validate_ssh_exception(state, config)
        state, config = self.fixture()
        state["active_operation"]["query_decoder_diagnostics"]["explicit_human_authorized"] = False
        with self.assertRaises(RuntimeError):
            validate_ssh_exception(state, config)


if __name__ == "__main__":
    unittest.main()
