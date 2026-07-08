#!/usr/bin/env python3
"""Audit a Puppeteer skeleton-decoder checkpoint before Evoweave integration.

This script does not load Evoweave data and does not train. It checks whether a
local Puppeteer checkout and checkpoint expose the expected skeleton decoder
state needed for a future `Puppeteer decoder backbone + Evoweave motion
condition prefix` model variant.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def load_state_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return dict(load_file(str(path), device="cpu"))
    import torch

    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "module"):
            value = obj.get(key)
            if isinstance(value, dict):
                return value
        return obj
    raise TypeError(f"unsupported checkpoint object type: {type(obj)!r}")


def first_existing(root: Path, relative_paths: tuple[str, ...]) -> Path:
    hits = [root / rel for rel in relative_paths if (root / rel).exists()]
    if len(hits) != 1:
        raise FileNotFoundError(
            f"expected exactly one of {relative_paths} under {root}, found {[str(p) for p in hits]}"
        )
    return hits[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--puppeteer-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config-json", type=Path, default=None)
    args = parser.parse_args()

    puppeteer_root = args.puppeteer_root.resolve()
    skeleton_root = puppeteer_root / "skeleton"
    model_root = skeleton_root / "skeleton_models"
    skeleton_opt = first_existing(model_root, ("skeleton_opt.py",))
    skeletongen = first_existing(model_root, ("skeletongen.py",))

    sys.path.insert(0, str(skeleton_root))
    import skeleton_models.skeleton_opt as skeleton_opt_module  # noqa: F401
    from skeleton_models.skeleton_opt import SkeletonOPTConfig, SkeletonOPT

    state = load_state_dict(args.checkpoint.resolve())
    keys = sorted(str(k) for k in state.keys())
    groups = Counter()
    for key in keys:
        if "transformer.model.decoder.layers." in key:
            groups["decoder_layers"] += 1
        elif "transformer.model.decoder.embed_tokens" in key:
            groups["embed_tokens"] += 1
        elif "transformer.model.decoder.embed_positions" in key:
            groups["embed_positions"] += 1
        elif "transformer.model.decoder.token_embed_positions" in key:
            groups["token_embed_positions"] += 1
        elif "transformer.model.decoder.cond_embed" in key:
            groups["cond_embed"] += 1
        elif "transformer.lm_head" in key or key.endswith("lm_head.weight"):
            groups["lm_head"] += 1
        elif "cond_proj" in key or "cond_head_proj" in key:
            groups["condition_projection"] += 1
        else:
            groups["other"] += 1

    if groups["decoder_layers"] <= 0:
        raise RuntimeError("checkpoint has no transformer.model.decoder.layers.* keys")

    config_payload = None
    if args.config_json is not None:
        config_path = args.config_json.resolve()
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        config = SkeletonOPTConfig(**config_payload)
        model = SkeletonOPT(config)
        model_keys = set(model.state_dict().keys())
        direct_hits = sum(1 for key in keys if key in model_keys)
    else:
        direct_hits = None

    report = {
        "puppeteer_root": str(puppeteer_root),
        "skeleton_opt": str(skeleton_opt),
        "skeletongen": str(skeletongen),
        "checkpoint": str(args.checkpoint.resolve()),
        "state_key_count": len(keys),
        "key_groups": dict(sorted(groups.items())),
        "config_json": str(args.config_json.resolve()) if args.config_json else None,
        "direct_state_dict_key_hits_with_config": direct_hits,
        "feasible_backbone_signal": groups["decoder_layers"] > 0,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
