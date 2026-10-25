"""CPU-only tests for pure ID+XYZ diagnostics."""

import os
from pathlib import Path
import sys
import unittest

import torch

os.environ.setdefault("RIGWEAVE_DISABLE_OPEN3D", "1")
os.environ.setdefault("RIGWEAVE_DISABLE_LIGHTNING_IMPORT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rigweave.dynamic_rig.id_xyz_diagnostics import (  # noqa: E402
    ROLE_NAMES,
    classify_gt_roles,
    first_divergence_summary,
    teacher_forced_diagnostics,
)
from rigweave.dynamic_rig.id_xyz_tokens import IdXYZTokens  # noqa: E402


def codec() -> IdXYZTokens:
    return IdXYZTokens(271, tuple(range(256)), bos=257, eos=258, pad=259,
                       branch=256, class_token=270, max_joints=16,
                       max_sequence_tokens=128)


def stream(parents, value=0.0):
    result = codec().encode(parents, torch.full((1, len(parents), 3), value))
    return result["input_ids"][0].tolist()


def known_logits(tokens, *, wrong_row=None, wrong_token=None):
    result = torch.full((len(tokens) - 1, codec().vocab_size), -7.0)
    for row, token in enumerate(tokens[1:]):
        result[row, token] = 7.0
    if wrong_row is not None:
        target = tokens[wrong_row + 1]
        result[wrong_row, target] = -7.0
        result[wrong_row, wrong_token if wrong_token is not None else 0] = 7.0
    return result


class IdXYZDiagnosticsTests(unittest.TestCase):
    def test_roles_cover_branching_stream_and_per_role_metrics(self):
        tokens = stream([-1, 0, 1, 0])
        roles = classify_gt_roles(codec(), tokens)
        self.assertEqual(set(role for role in roles if role), set(ROLE_NAMES))
        self.assertEqual({role: roles.count(role) for role in ROLE_NAMES},
                         {"class": 1, "new_joint_id": 4, "parent_id": 1,
                          "branch": 1, "eos": 1, "x": 4, "y": 4, "z": 4})
        result = teacher_forced_diagnostics(codec(), tokens, known_logits(tokens), legal_choice=True)
        self.assertEqual(result["full_vocab"]["count"], len(tokens) - 1)
        self.assertEqual(result["full_vocab"]["top1_correct"], len(tokens) - 1)
        self.assertEqual(result["per_role"]["branch"]["full_vocab_top1_correct"], 1)
        self.assertEqual(result["per_role"]["parent_id"]["full_vocab_count"], 1)
        self.assertGreater(result["legal_choice"]["nontrivial_choice_count"], 0)

    def test_shifted_alignment_catches_one_target_only(self):
        tokens = stream([-1, 0, 1])
        wrong_row = 2  # logits[2] predicts full_ids[3], not full_ids[2].
        result = teacher_forced_diagnostics(codec(), tokens,
                                            known_logits(tokens, wrong_row=wrong_row))
        self.assertEqual(result["full_vocab"]["top1_correct"], len(tokens) - 2)
        role = result["target_roles"][wrong_row + 1]
        self.assertEqual(result["per_role"][role]["full_vocab_top1_correct"],
                         result["per_role"][role]["full_vocab_count"] - 1)

    def test_legal_choice_metrics_distinguish_illegal_full_vocab_argmax(self):
        tokens = stream([-1, 0, 1, 0])
        row = tokens.index(tokens[6]) - 1  # after J0 XYZ, a nontrivial choice site.
        prefix = tokens[: row + 1]
        target = tokens[row + 1]
        allowed = codec().allowed_tokens(prefix)
        self.assertGreater(len(allowed), 1)
        logits = known_logits(tokens)
        logits[row].fill_(-2.0)
        logits[row, target] = 0.0
        logits[row, allowed[0] if allowed[0] != target else allowed[1]] = -1.0
        logits[row, 0] = 5.0  # illegal, so only full-vocabulary top-1 is wrong.
        result = teacher_forced_diagnostics(codec(), tokens, logits, legal_choice=True)
        self.assertEqual(result["full_vocab"]["top1_correct"], len(tokens) - 2)
        self.assertEqual(result["legal_choice"]["top1_correct"], len(tokens) - 1)
        self.assertGreater(result["full_vocab"]["nll_sum"], result["legal_choice"]["nll_sum"])

    def test_forced_and_nontrivial_native_choice_counts_are_separate(self):
        tokens = stream([-1, 0, 1])
        result = teacher_forced_diagnostics(codec(), tokens, known_logits(tokens), legal_choice=True)
        class_legal = result["per_role"]["class"]["legal_choice"]
        self.assertEqual(class_legal["count"], 1)
        self.assertEqual(class_legal["nontrivial_choice_count"], 0)
        self.assertGreater(result["legal_choice"]["nontrivial_choice_count"], 0)

    def test_declared_padding_is_shape_checked_and_ignored(self):
        tokens = stream([-1, 0]) + [codec().pad, codec().pad]
        logits = known_logits(tokens)
        result = teacher_forced_diagnostics(codec(), tokens, logits, ignore_pad=True)
        self.assertEqual(result["ignored_pad_count"], 2)
        self.assertEqual(result["target_count"], len(tokens) - 3)
        with self.assertRaisesRegex(ValueError, "ignore_pad"):
            teacher_forced_diagnostics(codec(), tokens, logits)

    def test_logits_shapes_dtype_and_finiteness_fail_closed(self):
        tokens = stream([-1])
        good = known_logits(tokens)
        cases = [good[:-1], good.unsqueeze(0), good.to(torch.int64)]
        for logits in cases:
            with self.subTest(shape=tuple(logits.shape), dtype=logits.dtype), self.assertRaises(ValueError):
                teacher_forced_diagnostics(codec(), tokens, logits)
        bad = good.clone()
        bad[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            teacher_forced_diagnostics(codec(), tokens, bad)

    def test_invalid_gt_stream_is_rejected_by_native_codec(self):
        tokens = stream([-1, 0])
        for malformed in (tokens[:-1], tokens + [codec().eos],
                          tokens[:3] + [codec().pad] + tokens[3:]):
            with self.subTest(malformed=malformed), self.assertRaisesRegex(ValueError, "GT|PAD"):
                classify_gt_roles(codec(), malformed)

    def test_first_divergence_is_positional_and_reports_gt_role(self):
        tokens = stream([-1, 0, 1, 0])
        exact = first_divergence_summary(codec(), tokens, tokens)
        self.assertEqual(exact["status"], "match")
        mismatch = list(tokens)
        branch = tokens.index(codec().branch)
        mismatch[branch] = tokens[branch + 1]
        result = first_divergence_summary(codec(), mismatch, tokens)
        self.assertEqual((result["status"], result["index"], result["role"]),
                         ("first_mismatch", branch, "branch"))
        inserted = tokens[:branch] + [0] + tokens[branch:]
        result = first_divergence_summary(codec(), inserted, tokens)
        self.assertEqual((result["index"], result["role"]), (branch, "branch"))
        self.assertTrue(result["positional_only"])

    def test_first_divergence_handles_eos_and_length_differences(self):
        tokens = stream([-1, 0])
        short = tokens[:-1]
        result = first_divergence_summary(codec(), short, tokens)
        self.assertEqual((result["status"], result["role"]), ("generated_ended_early", "eos"))
        long = tokens + [codec().eos]
        result = first_divergence_summary(codec(), long, tokens)
        self.assertEqual((result["status"], result["role"]), ("generated_has_extra_tokens", None))


if __name__ == "__main__":
    unittest.main()
