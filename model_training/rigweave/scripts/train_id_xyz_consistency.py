#!/usr/bin/env python3
"""Guarded entry for the real-frame joint ID/XYZ experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import traceback

from id_xyz_launch_guard import ROOT, verify_launch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("audit", "preflight", "capacity", "train"), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    source = verify_launch(args, config)
    # All computation imports follow verified provenance, binding and stage authorization.
    sys.path.insert(0, str(ROOT / "model_training/analysis"))
    import torch
    import torch.distributed as dist
    from id_xyz_runtime import build_model, read_entries, sha256, validate_config, write_json
    from id_xyz_stages import audit, gather, read_reviewed, run_stage, validate_receipt
    from structure_xyz_fill_runtime import verify_data_selection

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    claimed = False
    try:
        validate_config(config)
        entries = read_entries(config)
        verify_data_selection(config, entries)
        if args.mode == "audit":
            if world != 1 or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
                raise RuntimeError("CPU audit requires one process and hidden CUDA")
            args.output.mkdir(parents=True, exist_ok=False)
            claimed = True
            result = audit(args, config, entries, source)
        else:
            if (world != 2 or torch.cuda.device_count() != world
                    or world != config["runtime"]["world_size"]):
                raise RuntimeError("GPU execution requires exactly the approved two devices")
            device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
            torch.cuda.set_device(device)
            torch.cuda.set_per_process_memory_fraction(config["runtime"]["gpu_memory_fraction_limit"], device)
            dist.init_process_group("nccl", timeout=timedelta(minutes=60))
            if rank == 0:
                args.output.mkdir(parents=True, exist_ok=False)
            dist.barrier()
            claimed = True
            reviewed = read_reviewed(config["safety"]["data_audit"])
            validate_receipt(config, reviewed)
            model, tokenizer, codec, initial = build_model(config, device)
            if initial["initial_model_sha256"] != reviewed["initialization"]["initial_model_sha256"]:
                raise RuntimeError("GPU weights differ from the CPU-audited official initialization")
            fingerprints = gather(initial["initial_model_sha256"], world)
            if len(set(fingerprints)) != 1:
                raise RuntimeError("rank initializations differ")
            initial.update(code_commit=source, config_sha256=sha256(args.config),
                           torch_version=torch.__version__, device=torch.cuda.get_device_name(device))
            write_json(args.output / f"initialization_rank_{rank}.json", initial)
            result = run_stage(model, tokenizer, codec, initial, entries, config, args,
                               device, rank, world, source, reviewed)
        result.update(code_commit=source, config_sha256=sha256(args.config),
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.output / f"result_rank_{rank}.json", result)
        if rank == 0:
            write_json(args.output / "result.json", result)
        print("RESULT " + json.dumps({k: v for k, v in result.items()
              if k in ("passed", "complete", "mode", "asset_exposures", "step")}), flush=True)
    except Exception as error:
        if claimed:
            write_json(args.output / f"failure_rank_{rank}.json", {
                "passed": False, "complete": False, "mode": args.mode, "rank": rank,
                "error": repr(error), "traceback": traceback.format_exc(),
                "failed_utc": datetime.now(timezone.utc).isoformat()})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
