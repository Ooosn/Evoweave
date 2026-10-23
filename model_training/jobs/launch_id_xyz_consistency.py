#!/usr/bin/env python3
"""Run the ID/XYZ experiment through the existing bounded resource supervisor."""
from launch_legacy_single_target import main


if __name__ == "__main__":
    raise SystemExit(main(
        trainer_module_name="id_xyz_launch_guard",
        exception_env_name="EVOWEAVE_ID_XYZ_SSH_EXCEPTION",
        trainer_script="train_id_xyz_consistency.py",
        modes=("audit", "preflight", "capacity", "train"), cpu_modes=("audit",)))
