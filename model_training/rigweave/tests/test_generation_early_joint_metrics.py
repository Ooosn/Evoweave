from __future__ import annotations

from pathlib import Path
import sys
import types
from types import SimpleNamespace

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if sys.platform == "win32" and "resource" not in sys.modules:
    resource_stub = types.ModuleType("resource")
    resource_stub.RUSAGE_SELF = 0
    resource_stub.getrusage = lambda _: SimpleNamespace(ru_maxrss=0)
    sys.modules["resource"] = resource_stub

from eval_dynamic_rig_generation import _output_metrics  # noqa: E402


def test_output_metrics_records_root_and_first_serialized_child_error() -> None:
    target = SimpleNamespace(
        joints=np.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32),
        parents=[None, 0, 1],
    )
    prediction = SimpleNamespace(
        joints=np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 2.0], [0.0, 2.0, 0.0]], dtype=np.float32),
        parents=[None, 0, 1],
    )

    metrics = _output_metrics(prediction, target, (-1.0, 1.0))

    assert metrics["root_l2"] == 1.0
    assert metrics["joint1_l2"] == 2.0


def test_output_metrics_omits_unavailable_early_joint_errors() -> None:
    target = SimpleNamespace(
        joints=np.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        parents=[None, 0],
    )
    prediction = SimpleNamespace(
        joints=np.asarray([[0.5, 0.0, 0.0]], dtype=np.float32),
        parents=[None],
    )

    metrics = _output_metrics(prediction, target, (-1.0, 1.0))

    assert metrics["root_l2"] == 0.5
    assert "joint1_l2" not in metrics
