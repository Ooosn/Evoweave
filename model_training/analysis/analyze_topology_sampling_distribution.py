#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
RIGWEAVE_SRC = REPO / "rigweave" / "src"
if str(RIGWEAVE_SRC) not in sys.path:
    sys.path.insert(0, str(RIGWEAVE_SRC))

from rigweave.dynamic_rig import (  # noqa: E402
    PuppeteerDynamicRigDataset,
    TopologyFamilyMixtureSampler,
    load_parent_topology_signatures,
)


def parse_alphas(value: str) -> tuple[float, ...]:
    alphas = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not alphas:
        raise ValueError("at least one mixture alpha is required")
    if any(not 0.0 <= alpha <= 1.0 for alpha in alphas):
        raise ValueError(f"mixture alphas must be in [0, 1], got {alphas}")
    return alphas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit exact parent-topology families and proposed mixture distributions."
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-joints", type=int, default=101)
    parser.add_argument("--scan-workers", type=int, default=8)
    parser.add_argument("--alphas", type=str, default="0,0.5,0.75,1")
    parser.add_argument("--sample-exposures", type=int, default=14400)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()

    dataset = PuppeteerDynamicRigDataset(
        args.train_manifest,
        max_joints=args.max_joints,
    )
    signatures = load_parent_topology_signatures(
        dataset.paths,
        num_workers=args.scan_workers,
    )
    reports: dict[str, object] = {}
    for alpha in parse_alphas(args.alphas):
        sampler = TopologyFamilyMixtureSampler(
            signatures,
            mixture_alpha=alpha,
            seed=args.seed,
        )
        report = sampler.report()
        report["expected_draws_by_family_frequency"] = {
            label: float(probability) * int(args.sample_exposures)
            for label, probability in report[
                "expected_row_mass_by_family_frequency"
            ].items()
        }
        reports[f"{alpha:.6g}"] = report

    payload = {
        "train_manifest": str(args.train_manifest),
        "max_joints": int(args.max_joints),
        "raw_rows": int(dataset.raw_rows),
        "rows_after_max_joint_filter": len(dataset),
        "filtered_over_max_joints": int(dataset.filtered_over_max_joints),
        "sample_exposures": int(args.sample_exposures),
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
