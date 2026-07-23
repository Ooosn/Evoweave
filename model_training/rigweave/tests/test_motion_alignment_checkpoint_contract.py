from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import torch


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))
if sys.platform == "win32" and "resource" not in sys.modules:
    resource_stub = types.ModuleType("resource")
    resource_stub.RUSAGE_SELF = 0
    resource_stub.getrusage = lambda _: types.SimpleNamespace(ru_maxrss=0)
    sys.modules["resource"] = resource_stub

from eval_dynamic_rig_ce import apply_checkpoint_eval_defaults  # noqa: E402


def _namespace(checkpoint: Path) -> argparse.Namespace:
    return argparse.Namespace(checkpoint=checkpoint)


def test_eval_restores_query_rigid_policy_from_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "query_rigid.pt"
    torch.save({"args": {"motion_alignment_policy": "query_rigid"}}, checkpoint)
    args = _namespace(checkpoint)
    apply_checkpoint_eval_defaults(args)
    assert args.motion_alignment_policy == "query_rigid"


def test_legacy_checkpoint_defaults_to_no_motion_alignment(tmp_path: Path) -> None:
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"args": {}}, checkpoint)
    args = _namespace(checkpoint)
    apply_checkpoint_eval_defaults(args)
    assert args.motion_alignment_policy == "none"
