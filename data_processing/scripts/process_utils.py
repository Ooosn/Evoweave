"""Subprocess helpers that do not leave descendant processes behind."""

from __future__ import annotations

import os
import signal
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, Any


def _terminate_process_group(process: subprocess.Popen[Any], grace_sec: float = 5.0) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return

        try:
            process.wait(timeout=grace_sec)
        except subprocess.TimeoutExpired:
            pass

        # The group can outlive its leader, so check the group even if wait() returned.
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        os.killpg(process.pid, signal.SIGKILL)
        return

    # Windows has no killpg equivalent. taskkill /T targets the exact process tree.
    subprocess.run(
        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def run_process_group(
    cmd: Sequence[str | os.PathLike[str]],
    *,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    stdout: int | IO[Any] | None = None,
    stderr: int | IO[Any] | None = None,
    text: bool = False,
    timeout: float | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[Any]:
    """Run a command in its own process group and clean the full group on exit."""
    popen_kwargs: dict[str, Any] = {
        "cwd": None if cwd is None else str(Path(cwd)),
        "env": env,
        "stdout": stdout,
        "stderr": stderr,
        "text": text,
    }
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    else:
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    process = subprocess.Popen(list(cmd), **popen_kwargs)
    try:
        captured_stdout, captured_stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(process)
        captured_stdout, captured_stderr = process.communicate()
        exc.stdout = captured_stdout
        exc.stderr = captured_stderr
        raise
    except BaseException:
        _terminate_process_group(process)
        process.communicate()
        raise

    completed = subprocess.CompletedProcess(
        args=list(cmd),
        returncode=int(process.returncode),
        stdout=captured_stdout,
        stderr=captured_stderr,
    )
    if check:
        completed.check_returncode()
    return completed
