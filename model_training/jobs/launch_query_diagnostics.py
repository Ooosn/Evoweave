#!/usr/bin/env python3
"""Inference-only diagnostics through the existing bounded HGC supervisor."""
from launch_legacy_single_target import main


if __name__ == "__main__":
    raise SystemExit(main(trainer_module_name="query_diagnostics_guard",
        exception_env_name="EVOWEAVE_QUERY_DIAGNOSTICS_SSH_EXCEPTION",
        trainer_script="diagnose_id_xyz_checkpoint.py",
        modes=("audit", "smoke", "diagnose"), cpu_modes=("audit",)))
