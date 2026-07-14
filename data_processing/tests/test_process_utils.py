from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from process_utils import run_process_group  # noqa: E402


@unittest.skipUnless(os.name == "posix", "POSIX process-group behavior")
class ProcessUtilsTest(unittest.TestCase):
    def test_timeout_kills_descendant_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pid_path = Path(tmp) / "child.pid"
            parent_script = (
                "import pathlib, subprocess, sys, time; "
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
                f"pathlib.Path({str(pid_path)!r}).write_text(str(child.pid)); "
                "time.sleep(60)"
            )

            with self.assertRaises(subprocess.TimeoutExpired):
                run_process_group(
                    [sys.executable, "-c", parent_script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=1.0,
                )

            child_pid = int(pid_path.read_text(encoding="utf-8"))
            deadline = time.monotonic() + 5.0
            while Path(f"/proc/{child_pid}").exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertFalse(Path(f"/proc/{child_pid}").exists())


if __name__ == "__main__":
    unittest.main()
