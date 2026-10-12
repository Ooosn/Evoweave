"""Namespace-isolated CPU tests for virtual query target preparation."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import types
import unittest

import numpy as np
import torch


ROOT = Path(__file__).parents[1]
CODEC_PATH = ROOT / "src" / "rigweave" / "dynamic_rig" / "aux_skeleton_tokens.py"
ADAPTER_PATH = ROOT / "src" / "rigweave" / "dynamic_rig" / "virtual_query_xyz_data.py"


@dataclass
class TokenizeInput:
    bones: np.ndarray
    tails: np.ndarray
    branch: np.ndarray
    is_leaf: np.ndarray
    no_skin: object
    cls: str
    parts_bias: dict


class RecordingTokenizer:
    bos = 60
    eos = 61
    pad = 62
    num_discrete = 32
    token_id_branch = 32
    vocab_size = 64

    def __init__(self) -> None:
        self.calls: list[TokenizeInput] = []

    def cls_name_to_token(self, name: str) -> int:
        return {"articulationxl": 50, "other": 51}[name]

    def tokenize(self, token_input: TokenizeInput) -> np.ndarray:
        self.calls.append(token_input)
        tokens = [self.bos, self.cls_name_to_token(token_input.cls)]
        joints = token_input.bones[:, 3:]
        for index, point in enumerate(joints):
            if bool(token_input.branch[index]):
                tokens.append(self.token_id_branch)
            tokens.extend(int(np.clip(np.rint(float(value) * 8.0 + 16.0), 0, self.num_discrete - 1)) for value in point)
        tokens.append(self.eos)
        return np.asarray(tokens, dtype=np.int64)


def install_isolated_modules() -> tuple[types.ModuleType, types.ModuleType, list[tuple[str, object]]]:
    """Install only the package names needed by the relative adapter import."""

    for name in ("rigweave", "rigweave.dynamic_rig", "src", "src.tokenizer"):
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    spec_module = types.ModuleType("src.tokenizer.spec")
    spec_module.TokenizeInput = TokenizeInput
    sys.modules["src.tokenizer.spec"] = spec_module

    helper_calls: list[tuple[str, object]] = []
    data_module = types.ModuleType("rigweave.dynamic_rig.data")

    def derive_tokenizer_tails(joints: np.ndarray, parents: list[int | None]) -> np.ndarray:
        helper_calls.append(("tails", np.asarray(joints).copy()))
        points = np.asarray(joints, dtype=np.float32)
        children: list[list[int]] = [[] for _ in range(int(points.shape[0]))]
        for child, parent in enumerate(parents):
            if parent is not None:
                children[int(parent)].append(int(child))
        tails = points.copy()
        for index, kids in enumerate(children):
            if kids:
                tails[index] = points[int(kids[0])]
                continue
            parent = parents[index]
            if parent is None:
                tails[index] = points[index]
                continue
            direction = points[index] - points[int(parent)]
            if float(np.linalg.norm(direction)) <= 1.0e-8:
                tails[index] = points[index]
            else:
                tails[index] = points[index] + direction
        return tails.astype(np.float32, copy=False)

    def make_token_input(
        joints: np.ndarray,
        parents: list[int | None],
        cls: str,
        tokenizer: RecordingTokenizer,
        tails: np.ndarray | None,
    ) -> np.ndarray:
        """The existing data._make_token_input body, with no coordinate edits."""

        from src.tokenizer.spec import TokenizeInput as InjectedTokenizeInput

        parent_joints = []
        branch = []
        child_count = np.zeros((joints.shape[0],), dtype=np.int64)
        for index, parent in enumerate(parents):
            if parent is not None:
                child_count[parent] += 1
            parent_joints.append(joints[index] if parent is None else joints[parent])
            branch.append(False if index == 0 else parent != index - 1)
        bones = np.concatenate([np.asarray(parent_joints, dtype=np.float32), joints.astype(np.float32)], axis=-1)
        if tails is None:
            tails = derive_tokenizer_tails(joints, parents)
        else:
            tails = np.asarray(tails, dtype=np.float32)
        if tails.shape != joints.shape:
            raise ValueError("tails shape does not match joints shape")
        token_input = InjectedTokenizeInput(
            bones=bones,
            tails=tails,
            branch=np.asarray(branch, dtype=bool),
            is_leaf=child_count == 0,
            no_skin=None,
            cls=cls,
            parts_bias={},
        )
        helper_calls.append(("token_input", token_input))
        return tokenizer.tokenize(token_input)

    data_module._derive_tokenizer_tails = derive_tokenizer_tails
    data_module._make_token_input = make_token_input
    sys.modules[data_module.__name__] = data_module

    codec_spec = importlib.util.spec_from_file_location("rigweave.dynamic_rig.aux_skeleton_tokens", CODEC_PATH)
    if codec_spec is None or codec_spec.loader is None:
        raise RuntimeError(f"cannot load {CODEC_PATH}")
    codec_module = importlib.util.module_from_spec(codec_spec)
    sys.modules[codec_spec.name] = codec_module
    codec_spec.loader.exec_module(codec_module)

    adapter_spec = importlib.util.spec_from_file_location("rigweave.dynamic_rig.virtual_query_xyz_data", ADAPTER_PATH)
    if adapter_spec is None or adapter_spec.loader is None:
        raise RuntimeError(f"cannot load {ADAPTER_PATH}")
    adapter_module = importlib.util.module_from_spec(adapter_spec)
    sys.modules[adapter_spec.name] = adapter_module
    adapter_spec.loader.exec_module(adapter_module)
    return codec_module, adapter_module, helper_calls


CODEC, ADAPTER, HELPER_CALLS = install_isolated_modules()
AuxSkeletonTokens = CODEC.AuxSkeletonTokens
build_xyz_aux_targets = ADAPTER.build_xyz_aux_targets


def base_targets() -> dict[str, torch.Tensor]:
    xyz = torch.tensor(
        [[
            [[0.00, 0.00, 0.00], [0.20, 0.00, 0.00], [0.20, 0.15, 0.00]],
            [[0.10, 0.00, 0.00], [0.30, 0.00, 0.00], [0.30, 0.15, 0.00]],
        ]],
        dtype=torch.float32,
    )
    return {
        "frame_target_joints": xyz,
        "target_parents": torch.tensor([[-1, 0, 0]], dtype=torch.long),
        "joint_mask": torch.ones((1, 3), dtype=torch.bool),
        "frame_mask": torch.ones((1, 2), dtype=torch.bool),
    }


class VirtualQueryXYZDataTests(unittest.TestCase):
    def setUp(self) -> None:
        HELPER_CALLS.clear()
        self.codec = AuxSkeletonTokens()

    def build(self, targets: dict, tokenizer: RecordingTokenizer | None = None, **kwargs):
        tokenizer = RecordingTokenizer() if tokenizer is None else tokenizer
        output = build_xyz_aux_targets(targets, tokenizer, self.codec, device="cpu", **kwargs)
        return output, tokenizer

    def test_reuses_existing_helpers_and_pads_complete_main_and_aux_labels(self) -> None:
        output, tokenizer = self.build(base_targets(), max_main_tokens=2048)
        self.assertEqual(len(tokenizer.calls), 2)
        self.assertEqual([kind for kind, _ in HELPER_CALLS], ["tails", "token_input", "tails", "token_input"])
        for call in tokenizer.calls:
            self.assertEqual(call.cls, "articulationxl")
            self.assertEqual(call.bones.shape, (3, 6))
            self.assertEqual(call.tails.shape, (3, 3))

        main_width = int(output["main_lengths"].max())
        self.assertEqual(tuple(output["main_input_ids"].shape), (1, 2, main_width))
        self.assertEqual(tuple(output["main_attention_mask"].shape), (1, 2, main_width))
        self.assertLess(main_width, 2048)
        self.assertTrue(output["main_attention_mask"].dtype == torch.bool)
        for frame in range(2):
            length = int(output["main_lengths"][0, frame])
            self.assertEqual(int(output["main_attention_mask"][0, frame].sum()), length)
            self.assertEqual(int(output["main_input_ids"][0, frame, 0]), tokenizer.bos)
            self.assertEqual(int(output["main_input_ids"][0, frame, 1]), tokenizer.cls_name_to_token("articulationxl"))
            self.assertEqual(int(output["main_input_ids"][0, frame, length - 1]), tokenizer.eos)
            self.assertTrue(torch.all(output["main_input_ids"][0, frame, length:] == tokenizer.pad))

        self.assertEqual(output["aux_input_ids"].ndim, 2)
        self.assertEqual(output["aux_attention_mask"].ndim, 2)
        self.assertEqual(output["aux_labels"].ndim, 2)
        valid = int(output["aux_attention_mask"][0].sum())
        self.assertEqual(int(output["aux_input_ids"][0, 0]), self.codec.bos)
        self.assertEqual(int(output["aux_labels"][0, valid - 1]), self.codec.eos)
        self.assertTrue(torch.all(output["aux_labels"][0, valid:] == -100))
        self.assertTrue(torch.all(output["aux_input_ids"][0, valid:] == self.codec.pad))
        self.assertEqual(output["aux_dfs_order"].tolist(), [[0, 1, 2]])

    def test_float32_ulp_boundary_matches_direct_original_helpers(self) -> None:
        rng = np.random.default_rng(20261013)
        xyz = rng.uniform(-0.5, 0.5, size=(1, 3, 3, 3)).astype(np.float32)
        xyz[0, 0, 0, 0] = np.nextafter(np.float32(0.25), np.float32(1.0))
        xyz[0, 1, 1, 1] = np.nextafter(np.float32(-0.25), np.float32(-1.0))
        targets = {
            "frame_target_joints": torch.from_numpy(xyz),
            "target_parents": torch.tensor([[-1, 0, 0]], dtype=torch.long),
            "joint_mask": torch.ones((1, 3), dtype=torch.bool),
            "frame_mask": torch.ones((1, 3), dtype=torch.bool),
        }
        output, _ = self.build(targets, max_main_tokens=2048)
        captured = [payload for kind, payload in HELPER_CALLS if kind == "token_input"]
        self.assertEqual(len(captured), 3)

        direct_tokenizer = RecordingTokenizer()
        parents = [None, 0, 0]
        for frame, joints in enumerate(xyz[0]):
            self.assertEqual(joints.dtype, np.float32)
            tails = ADAPTER._derive_tokenizer_tails(joints, parents)
            direct_tokens = ADAPTER._make_token_input(
                joints, parents, "articulationxl", direct_tokenizer, tails=tails
            )
            expected = direct_tokenizer.calls[-1]
            actual = captured[frame]
            np.testing.assert_array_equal(actual.bones, expected.bones)
            np.testing.assert_array_equal(actual.tails, expected.tails)
            np.testing.assert_array_equal(actual.branch, expected.branch)
            np.testing.assert_array_equal(actual.is_leaf, expected.is_leaf)
            length = int(output["main_lengths"][0, frame])
            np.testing.assert_array_equal(output["main_input_ids"][0, frame, :length].numpy(), direct_tokens)

    def test_perturbing_one_frame_changes_only_that_main_sequence(self) -> None:
        original, _ = self.build(base_targets())
        changed_targets = base_targets()
        changed_targets["frame_target_joints"][0, 1, 2, 1] = 0.75
        changed, _ = self.build(changed_targets)
        torch.testing.assert_close(original["main_input_ids"][:, 0], changed["main_input_ids"][:, 0])
        self.assertFalse(torch.equal(original["main_input_ids"][:, 1], changed["main_input_ids"][:, 1]))
        for key in ("aux_input_ids", "aux_attention_mask", "aux_labels", "aux_dfs_order", "aux_source_to_dfs"):
            torch.testing.assert_close(original[key], changed[key])

    def test_nonquery_coordinate_bounds_and_padded_frame_masks_fail_closed(self) -> None:
        bad = base_targets()
        bad["frame_target_joints"][0, 1, 0, 0] = 1.2
        with self.assertRaisesRegex(ValueError, r"outside \[-1,1\]"):
            self.build(bad)

        bad_mask = base_targets()
        bad_mask["frame_mask"][0, 1] = False
        with self.assertRaisesRegex(ValueError, "frame_mask"):
            self.build(bad_mask)

    def test_main_capacity_and_vocab_contract_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_main_tokens"):
            self.build(base_targets(), max_main_tokens=3)

        bad_tokenizer = RecordingTokenizer()
        bad_tokenizer.vocab_size = 10
        with self.assertRaisesRegex(ValueError, "outside tokenizer.vocab_size"):
            self.build(base_targets(), tokenizer=bad_tokenizer)

    def test_raw_medium_batch_list_keeps_assets_and_aux_is_per_asset(self) -> None:
        first = base_targets()
        second_xyz = torch.tensor(
            [[
                [[-0.10, 0.00, 0.00], [0.10, 0.00, 0.00]],
                [[-0.05, 0.00, 0.00], [0.15, 0.00, 0.00]],
            ]], dtype=torch.float32,
        )
        second = {
            "frame_target_joints": second_xyz,
            "target_parents": torch.tensor([[-1, 0]], dtype=torch.long),
            "joint_mask": torch.ones((1, 2), dtype=torch.bool),
            "frame_mask": torch.ones((1, 2), dtype=torch.bool),
            "cls": ["other"],
        }
        output, tokenizer = self.build([first, second])
        self.assertEqual(tuple(output["main_input_ids"].shape[:2]), (2, 2))
        self.assertEqual(output["aux_input_ids"].shape[0], 2)
        self.assertEqual(output["aux_attention_mask"].ndim, 2)
        self.assertEqual(tuple(output["aux_dfs_order"].shape), (2, 3))
        self.assertEqual(output["aux_joint_counts"].tolist(), [3, 2])
        self.assertEqual(output["classnames"], ["articulationxl", "other"])
        self.assertEqual(len(tokenizer.calls), 4)
        main_width = int(output["main_lengths"].max())
        self.assertEqual(output["main_input_ids"].shape[-1], main_width)
        self.assertLess(main_width, 2048)
        short_length = int(output["main_lengths"][1, 0])
        self.assertTrue(torch.all(output["main_input_ids"][1, 0, short_length:] == tokenizer.pad))


if __name__ == "__main__":
    unittest.main()
