from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rigweave.dynamic_rig.frame_budget_accounting import (
    FrameBudgetProgress, normalize_token_gradients, target_token_weight,
)


def trainer_helpers():
    path = Path(__file__).resolve().parents[1] / "scripts/train_dynamic_rig.py"
    tree = ast.parse(path.read_text())
    names = {"json_safe", "accounting_fields", "resume_contract_differences", "save_checkpoint"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = {"torch": torch, "argparse": argparse, "Any": Any, "Path": Path, "json": json,
             "unwrap_model": lambda model: model, "trim_host_allocator": lambda: None}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), scope)
    return scope


class FrameBudgetAccountingTest(unittest.TestCase):
    def test_target_weight_ignores_padding_and_first_input(self):
        batch = {"input_ids": torch.tensor([[101, 1, 2, 9], [101, 2, 9, 99]]),
                 "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])}
        self.assertEqual(target_token_weight(batch, 9, 1), 5)
        self.assertEqual(target_token_weight(batch, 9, 2), 7)
        self.assertEqual(target_token_weight(batch, 9, 0), 3)
        with self.assertRaises(ValueError):
            target_token_weight(batch, 9, -1)

    def test_unequal_microbatches_and_ranks_match_global_token_mean(self):
        data = torch.arange(14, dtype=torch.float64).reshape(7, 2) / 13
        labels = torch.tensor([0, 1, 2, 1, 0, 2, 2])
        token_weights = torch.where(labels == 2, 2.0, 1.0).double()
        initial = torch.tensor([[0.2, -0.1, 0.3], [0.1, 0.4, -0.2]], dtype=torch.float64)
        reference = torch.nn.Parameter(initial.clone())
        losses = torch.nn.functional.cross_entropy(data @ reference, labels, reduction="none")
        ((losses * token_weights).sum() / token_weights.sum()).backward()
        local_gradients = []
        for partitions in ([[0, 1], [2]], [[3], [4, 5, 6]]):
            parameter = torch.nn.Parameter(initial.clone())
            for indices in partitions:
                loss = torch.nn.functional.cross_entropy(data[indices] @ parameter, labels[indices], reduction="none")
                (loss * token_weights[indices]).sum().backward()
            local_gradients.append(parameter.grad.clone())
        combined = torch.nn.Parameter(initial.clone())
        combined.grad = sum(local_gradients) / 2
        normalize_token_gradients([combined], 2, float(token_weights.sum()))
        torch.testing.assert_close(combined.grad, reference.grad, rtol=1e-12, atol=1e-12)

    def test_progress_roundtrip_uses_actual_samples_frames_and_cursor(self):
        progress = FrameBudgetProgress()
        progress.record_step(world_size=2, batches=[6, 3], frames=[8, 24], global_samples=18,
                             global_frames=240, global_token_weight=350, next_epoch=1, next_batch_in_epoch=0)
        restored = FrameBudgetProgress.restore(progress.checkpoint(), step=1, grad_accum_steps=2)
        self.assertEqual(restored.next_epoch, 1)
        self.assertEqual(restored.next_batch_in_epoch, 0)
        self.assertEqual(restored.fields(100)["sample_seen"], 18)
        restored.record_step(world_size=2, batches=[4, 4], frames=[16, 16], global_samples=16,
                             global_frames=256, global_token_weight=300, next_epoch=1, next_batch_in_epoch=2)
        self.assertEqual(restored.samples_seen, 34)
        self.assertEqual(restored.input_frames_seen, 496)
        self.assertEqual(restored.samples_by_frame_count, {"8": 12, "24": 6, "16": 16})

    def test_rejects_mismatched_rank_counts_or_resume_cursor(self):
        progress = FrameBudgetProgress()
        with self.assertRaises(ValueError):
            progress.record_step(world_size=2, batches=[3], frames=[24], global_samples=7,
                                 global_frames=144, global_token_weight=100, next_epoch=0, next_batch_in_epoch=1)
        progress.record_step(world_size=2, batches=[3], frames=[24], global_samples=6,
                             global_frames=144, global_token_weight=100, next_epoch=0, next_batch_in_epoch=1)
        with self.assertRaises(ValueError):
            FrameBudgetProgress.restore(progress.checkpoint(), step=2, grad_accum_steps=1)
        broken = progress.checkpoint()
        broken["samples_seen"] += 1
        with self.assertRaises(ValueError):
            FrameBudgetProgress.restore(broken, step=1, grad_accum_steps=1)

    def test_checkpoint_metadata_and_resume_contract(self):
        helpers = trainer_helpers()
        progress = FrameBudgetProgress()
        progress.record_step(world_size=2, batches=[6, 3], frames=[8, 24], global_samples=18,
                             global_frames=240, global_token_weight=350, next_epoch=1, next_batch_in_epoch=0)
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        args = argparse.Namespace(effective_batch=0, train_rows=100, frame_budget=72,
                                  frame_batch_cap=6, max_samples=0, scheduler="none")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            helpers["save_checkpoint"](path, model, optimizer, 1, args, batch_accounting=progress)
            payload = torch.load(path, weights_only=False)
        self.assertEqual(payload["sample_seen"], 18)
        self.assertEqual(payload["input_frames_seen"], 240)
        self.assertIsNone(payload["effective_batch"])
        self.assertEqual(payload["batch_accounting"]["next_epoch"], 1)
        self.assertEqual(helpers["resume_contract_differences"](payload, args, effective_batch=0, train_rows=100), [])
        missing = deepcopy(payload)
        del missing["batch_accounting"]
        self.assertTrue(helpers["resume_contract_differences"](missing, args, effective_batch=0, train_rows=100))
        args.frame_batch_cap = 8
        self.assertTrue(helpers["resume_contract_differences"](payload, args, effective_batch=0, train_rows=100))

    def test_legacy_accounting_and_missing_dynamic_counts(self):
        account = trainer_helpers()["accounting_fields"]
        self.assertEqual(account(105, 48, 15920)["sample_seen"], 5040)
        with self.assertRaises(ValueError):
            account(1, 0, 15920)


if __name__ == "__main__":
    unittest.main()
