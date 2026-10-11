from __future__ import annotations

from copy import deepcopy
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent / "src"))
import probe_inactive_motion_use as probe


class InactiveMotionUseTest(unittest.TestCase):
    def states(self):
        values = torch.arange(1, 148, dtype=torch.float32).square().reshape(1, 7, 7, 3)
        return values / values.sum(-1, keepdim=True)

    def plan(self):
        return dict(checkpoint="base.pt", source_cache="cache.pt", source_evaluation="saved.json",
                    source_cache_sha256="a" * 64, source_evaluation_sha256="b" * 64,
                    seed=11, permutation_seed=17, max_new_tokens=1400, prefix_tokens=64,
                    views=list(probe.VIEWS), controls=list(probe.CONTROLS))

    def test_controls_are_normalized_and_leave_cache_unchanged(self):
        states = self.states()
        original = states.clone()
        for control in ("base", "unknown", "permuted"):
            changed, _ = probe.intervene_states(states, control, 17, 3)
            torch.testing.assert_close(changed.sum(-1), torch.ones_like(changed[..., 0]))
            self.assertTrue(torch.equal(states, original))
            self.assertNotEqual(states.data_ptr(), changed.data_ptr())
            if control == "unknown":
                self.assertTrue(torch.equal(changed[..., 0], torch.ones_like(changed[..., 0])))
                self.assertEqual(changed[..., 1:].count_nonzero(), 0)
            elif control == "base":
                self.assertTrue(torch.equal(changed, states))

    def test_permutation_uses_both_axes_same_across_views_and_local_rng(self):
        states = self.states()
        before = torch.get_rng_state().clone()
        a, info = probe.intervene_states(states, "permuted", 17, 3)
        b, other = probe.intervene_states(states.flip(-1), "permuted", 17, 3)
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        self.assertEqual(info["anchor_order"], other["anchor_order"])
        order = torch.tensor(info["anchor_order"])
        self.assertTrue(torch.equal(a, states[:, order][:, :, order]))
        self.assertTrue(torch.equal(b, a.flip(-1)))
        self.assertFalse(torch.equal(a, states))
        _, different = probe.intervene_states(states, "permuted", 17, 4)
        self.assertNotEqual(info["anchor_order"], different["anchor_order"])

    def test_bad_channel_states_and_unknown_controls_fail(self):
        for bad in (torch.zeros(1, 2, 2, 3), torch.ones(1, 2, 3, 3),
                    torch.full((1, 2, 2, 3), float("nan")), torch.ones(1, 2, 2, 2)):
            with self.assertRaises(ValueError):
                probe.intervene_states(bad, "unknown", 17, 3)
        with self.assertRaises(ValueError):
            probe.intervene_states(self.states(), "actual", 17, 3)

    def test_forward_changes_original_evidence_not_relation_branch(self):
        captured = []
        class Encoder(torch.nn.Module):
            def forward(self, tokens, *, query_points, motion_evidence):
                captured.append((tokens, query_points, motion_evidence))
                return tokens[:, 0] + motion_evidence.states[..., 1].sum(-1, keepdim=True)
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(()), requires_grad=False)
                self.conditioner = SimpleNamespace(motion_encoder=Encoder())
            def _ar_losses(self, cond, batch):
                return {"ce_loss": cond.square().mean()}
        entry = {"index": 3, "input": {"tokens": torch.ones(1, 2, 7, 4),
                 "query_points": torch.arange(42.).reshape(1, 2, 7, 3), "states": self.states(),
                 "anchor_features": torch.arange(21.).reshape(1, 7, 3),
                 "valid_groups": torch.tensor([[True, False, True, True, True, True, True]]),
                 "rng": {"cpu": torch.get_rng_state().clone(), "cuda": []}},
                 "batch": {"target_joints": torch.ones(1, 5, 3), "target_parents": torch.arange(5)}}
        original = deepcopy(entry)
        versions = probe.tensor_versions(entry)
        model = Model()
        fake_train = SimpleNamespace(move_batch=lambda batch, device: dict(batch))
        with patch.dict(sys.modules, {"train_dynamic_rig": fake_train}):
            base, _, _, _ = probe.cached_forward(model, entry, "base", 17)
            for control in probe.CONTROLS:
                cond, losses, batch, _ = probe.cached_forward(model, entry, control, 17)
                self.assertTrue(cond.requires_grad)
                self.assertTrue(torch.isfinite(losses["ce_loss"]))
                self.assertTrue(torch.equal(batch["target_joints"], original["batch"]["target_joints"]))
                self.assertFalse(torch.equal(cond, base))
        self.assertEqual(probe.tensor_versions(entry), versions)
        for tokens, points, evidence in captured:
            self.assertTrue(tokens.requires_grad)
            self.assertTrue(torch.equal(tokens, original["input"]["tokens"]))
            self.assertTrue(torch.equal(points, original["input"]["query_points"]))
            self.assertTrue(torch.equal(evidence.anchor_features, original["input"]["anchor_features"]))
            self.assertTrue(torch.equal(evidence.valid_groups, original["input"]["valid_groups"]))
        self.assertTrue(torch.equal(entry["input"]["states"], original["input"]["states"]))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))

    def test_plan_is_bounded_and_pins_inputs(self):
        plan = self.plan()
        original = deepcopy(plan)
        probe.validate_plan(plan)
        self.assertEqual(plan, original)
        for change in ({"controls": ["actual"]}, {"views": ["weak2"]}, {"max_new_tokens": 64},
                       {"prefix_tokens": 0}, {"source_cache_sha256": ""}, {"seed": True}, {"checkpoint": ""}):
            with self.assertRaises(ValueError):
                probe.validate_plan({**plan, **change})

    def test_original_cache_rows_and_full_targets_must_match(self):
        plan = self.plan()
        observation = {"valid": [False, True], "strata": {"quiet_strict": [False, True]}}
        entries = []
        for index in range(16):
            for view in probe.VIEWS:
                row = {"index": index, "view": view, "asset_id": str(index), "ce": 1.0,
                       "generation": {"generated_ids": [1, 2, 3]}}
                entries.append({"index": index, "view": view, "asset_id": str(index),
                    "joint_count": 2, "baseline": row, "observation": deepcopy(observation),
                    "batch": {"joint_count": torch.tensor([2]), "input_ids": torch.tensor([[1, 2]]),
                        "target_joints": torch.zeros(1, 2, 3), "target_parents": torch.tensor([[-1, 0]]),
                        "query_center": torch.zeros(1, 3), "query_scale": torch.ones(1)},
                    "input": {"query_points": torch.zeros(1, 2 if view == "weak2" else 8, 7, 3)}})
        prior = {"complete": True, "changed_base_parameters": [], "plan": plan,
                 "baseline": {"valid": [deepcopy(e["baseline"]) for e in entries]},
                 "observations": [{"index": e["index"], "view": e["view"], **deepcopy(observation)} for e in entries],
                 "selected": {"valid": [{"index": i, "asset_id": str(i), "joint_count": 2} for i in range(16)]}}
        self.assertEqual(len(probe.validate_inputs(entries, prior, plan)), 48)
        with self.assertRaises(ValueError):
            probe.validate_inputs(entries[:-1], prior, plan)
        with self.assertRaises(ValueError):
            probe.validate_inputs(entries, {**prior, "complete": False}, plan)
        with self.assertRaises(ValueError):
            probe.validate_inputs(entries, {**prior, "changed_base_parameters": ["weight"]}, plan)
        altered = deepcopy(entries)
        altered[0]["baseline"]["ce"] = 2.0
        with self.assertRaises(ValueError):
            probe.validate_inputs(altered, prior, plan)
        altered = deepcopy(entries)
        altered[1]["batch"]["target_joints"][0, 1, 0] = 0.1
        with self.assertRaises(ValueError):
            probe.validate_inputs(altered, prior, plan)
        altered = deepcopy(entries)
        altered[0]["observation"]["strata"]["quiet_strict"][1] = False
        with self.assertRaises(ValueError):
            probe.validate_inputs(altered, prior, plan)

    def test_parity_fails_closed(self):
        probe.require_parity(2e-5, {"max_abs": 0})
        for ce, delta, prefix in ((3e-5, 0, True), (float("nan"), 0, True),
                                  (0, 1e-9, True), (0, float("nan"), True), (0, 0, False)):
            with self.assertRaises(RuntimeError):
                probe.require_parity(ce, {"max_abs": delta}, same_prefix=prefix)

    def test_prefix_needs_full_budget_or_original_eos(self):
        ids = list(range(100))
        self.assertTrue(probe.prefix_matches(ids[:66], ids, 64))
        self.assertFalse(probe.prefix_matches(ids[:2], ids, 64))
        self.assertFalse(probe.prefix_matches([], ids, 64))
        self.assertTrue(probe.prefix_matches(ids[:20], ids[:20], 64))
        entries = [{"index": i, "joint_count": n, "view": v}
                   for i, n in enumerate((20, 7, 147)) for v in probe.VIEWS]
        selected = probe.prefix_entries(entries)
        self.assertEqual(len(selected), 4)
        self.assertEqual({e["index"] for e in selected}, {1, 2})
        self.assertNotIn("static8", {e["view"] for e in selected})

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            probe.reserve_output(path)
            contents = path.read_bytes()
            with self.assertRaises(FileExistsError):
                probe.reserve_output(path)
            self.assertEqual(path.read_bytes(), contents)

    def test_hash_progress_is_flushed_before_reading(self):
        output = io.StringIO()
        def read_hash(path):
            self.assertIn("HASH_START source_cache", output.getvalue())
            return "a" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.pt"
            path.touch()
            with redirect_stdout(output), patch.object(probe, "file_hash", side_effect=read_hash):
                self.assertEqual(probe.logged_hash(path, "source_cache"), "a" * 64)
        self.assertIn("HASH_DONE source_cache", output.getvalue())


if __name__ == "__main__":
    unittest.main()
