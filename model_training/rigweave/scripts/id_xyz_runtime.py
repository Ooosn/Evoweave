"""Initialization, current-data preparation and matched sampling for ID/XYZ AR."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import subprocess

import numpy as np
import torch
from torch import nn

from rigweave.dynamic_rig.id_xyz_consistency import IdXYZConfig, IdXYZRig, extend_decoder_vocabulary
from rigweave.dynamic_rig.id_xyz_tokens import IdXYZTokens
from rigweave.dynamic_rig.surface_tokenizer import FixedQuerySurfaceTokenizer
from rigweave.dynamic_rig.virtual_frame_data import prepare_virtual_surface
from virtual_query_xyz_runtime import fixed_dataset, read_entries, sha256, write_json
from virtual_fusion_normal_runtime import MatchedFrameStream
from virtual_frame_runtime import exposure_learning_rate
from probe_virtual_query_lambda import parameter_fingerprint


FAMILY = "real_frame_id_xyz_consistency_v1"
TRAIN_SHA = "a17b6be96fb33a32a1bee7117b07789da6261c6b5693104f3d2cce07c09bab88"
VALID_SHA = "fd2be7b5885833946e68dc186300a820757ae652c487749eec1b049d381e230a"


def source_hashes():
    root = Path(__file__).resolve().parents[3]
    paths = [
        "src/rigweave/dynamic_rig/id_xyz_consistency.py",
        "src/rigweave/dynamic_rig/id_xyz_tokens.py",
        "src/rigweave/dynamic_rig/id_xyz_losses.py",
        "src/rigweave/dynamic_rig/aux_skeleton_tokens.py",
        "src/rigweave/dynamic_rig/motion_encoder.py",
        "src/rigweave/dynamic_rig/motion_evidence.py",
        "src/rigweave/dynamic_rig/structure_xyz_fill_data.py",
        "src/rigweave/dynamic_rig/virtual_frame_data.py",
        "src/rigweave/dynamic_rig/shared_structure_medium_data.py",
        "src/rigweave/dynamic_rig/root_rt.py", "src/rigweave/dynamic_rig/data.py",
        "src/rigweave/dynamic_rig/sampling.py", "src/rigweave/dynamic_rig/frame_batch_sampler.py",
        "src/rigweave/dynamic_rig/surface_tokenizer.py", "src/rigweave/dynamic_rig/skeleton_topology_metrics.py",
        "scripts/virtual_query_xyz_runtime.py", "scripts/virtual_fusion_normal_runtime.py",
        "scripts/virtual_frame_runtime.py",
        "scripts/structure_xyz_fill_runtime.py", "scripts/structure_xyz_fill_evaluation.py",
        "scripts/id_xyz_runtime.py", "scripts/id_xyz_evaluation.py",
        "scripts/train_id_xyz_consistency.py", "scripts/id_xyz_stages.py",
    ]
    return {path: sha256(root / "model_training/rigweave" / path) for path in paths}


def validate_config(config):
    if config.get("family") != FAMILY:
        raise ValueError("wrong experiment family")
    if config["architecture"] != asdict(IdXYZConfig()):
        raise ValueError("the reviewed real-frame architecture changed")
    if config["model_contract"] != {
        "virtual_frame": False, "frame_fusion": False, "auxiliary_decoder": False,
        "auxiliary_loss": False, "all_real_frames_supervised": True,
        "frame_role_policy": "all_real", "main_sequence": "explicit_DFS_ID_and_XYZ",
        "pruning": False, "global_rotation_augmentation": False,
    }:
        raise ValueError("virtual/auxiliary/pruning or target policy changed")
    data = config["data"]
    for name, expected in {
        "frames_min": 2, "frames_max": 24, "heldout_frames": 8,
        "normalization": "input_sequence_mesh_bbox", "global_rt_removal": True,
        "coordinate_clipping": False, "max_main_tokens": 2048,
    }.items():
        if data.get(name) != expected:
            raise ValueError(f"current data contract changed: {name}")
    for split, rows, digest in (("train", 5000, TRAIN_SHA), ("valid", 967, VALID_SHA)):
        spec = data["manifests"][split]
        if spec["rows"] != rows or spec["sha256"] != digest:
            raise ValueError("fixed data membership changed")
    if config["codec"] != {"max_joints": 256, "max_sequence_tokens": 2048}:
        raise ValueError("ID/XYZ grammar limits changed")
    consistency = config["consistency"]
    if (consistency["kind"] != "leave_one_out_detached_soft_target_cross_entropy"
            or consistency["temperature"] != 1
            or consistency["positions"] != "GT_structure_ID_parent_BRANCH_EOS"
            or consistency["weight"] not in (0., 1.)):
        raise ValueError("the paired structural consistency experiment changed")
    plan = config["training"]
    if (plan["target_assets"] != 20000 or plan["effective_assets_per_update"] != 48
            or plan["checkpoint_assets"] != [5000, 10000, 15000, 20000]
            or plan["evaluation_assets"] != [0, 5000, 10000, 15000, 20000]):
        raise ValueError("the first-stage exposure/assessment budget changed")
    for frame in range(2, 25):
        batch = plan["frame_policy"].get(str(frame))
        if type(batch) is not int or batch < 1 or 24 % batch:
            raise ValueError("physical batches must divide each rank's 24-asset update")
        ceiling = 6 if frame <= 6 else 12 if frame <= 12 else 24
        if batch != plan["frame_policy"].get(str(ceiling)):
            raise ValueError("each physical batch tier must match its measured upper frame bound")
    optim = config["optimization"]
    if optim != {"peak_lr": 1e-4, "weight_decay": 0., "clip_grad_norm": 1.,
                 "warmup_fraction": .1, "div_factor": 5., "final_div_factor": 10.,
                 "schedule_samples": 20000}:
        raise ValueError("the matched optimizer schedule changed")


def reset_random(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def load_tokenizer(config):
    from train_dynamic_rig import build_tokenizer
    runtime = config["runtime"]
    if sha256(runtime["tokenizer_config"]) != runtime["tokenizer_config_sha256"]:
        raise ValueError("original coordinate tokenizer changed")
    tokenizer = build_tokenizer(Path(runtime["tokenizer_config"]))
    if tokenizer.num_discrete != 256 or list(tokenizer.continuous_range) != [-1, 1]:
        raise ValueError("original coordinate bins/range changed")
    codec = IdXYZTokens(
        legacy_vocab_size=tokenizer.vocab_size, coordinate_ids=tuple(range(256)),
        bos=tokenizer.bos, eos=tokenizer.eos, pad=tokenizer.pad,
        branch=tokenizer.token_id_branch,
        class_token=tokenizer.cls_name_to_token("articulationxl"), **config["codec"])
    return tokenizer, codec


def build_model(config, device):
    import yaml
    from transformers import OPTConfig, OPTForCausalLM
    from src.model.michelangelo.get_model import get_encoder_simplified

    initial, runtime = config["initialization"], config["runtime"]
    upstream = runtime["unirig_root"]
    source = subprocess.check_output(["git", "-C", upstream, "rev-parse", "HEAD"], text=True).strip()
    if (source != initial["upstream_source_commit"]
            or subprocess.check_output(["git", "-C", upstream, "status", "--porcelain"], text=True).strip()):
        raise ValueError("official upstream source is not the pinned clean revision")
    if (initial["checkpoint"] != initial["official_checkpoint"]
            or sha256(initial["checkpoint"]) != initial["sha256"]
            or sha256(initial["model_config"]) != initial["model_config_sha256"]):
        raise ValueError("official checkpoint or model configuration changed")
    reset_random(initial["seed"])
    tokenizer, codec = load_tokenizer(config)
    specification = yaml.safe_load(Path(initial["model_config"]).read_text())
    lm_args = dict(specification["llm"])
    lm_args.pop("pretrained_model_name_or_path")
    lm_args["vocab_size"] = tokenizer.vocab_size
    if torch.device(device).type == "cpu":
        # CPU audit constructs and fingerprints weights without running attention.
        lm_args["_attn_implementation"] = "eager"
    lm_config = OPTConfig.from_pretrained(runtime["opt_config_root"], local_files_only=True, **lm_args)
    lm_config.torch_dtype = torch.float32
    decoder = OPTForCausalLM(lm_config)
    mesh_args = dict(specification["mesh_encoder"])
    if mesh_args.pop("__target__") != "michelangelo_encoder":
        raise ValueError("unexpected surface encoder")
    mesh = get_encoder_simplified(**mesh_args)
    projection = nn.Linear(mesh.width, lm_config.hidden_size)
    payload = torch.load(initial["checkpoint"], map_location="cpu", mmap=True, weights_only=False)
    state = {key.removeprefix("model."): value for key, value in payload["state_dict"].items()}
    loaded = {}
    for prefix, module in (("transformer.", decoder), ("mesh_encoder.", mesh), ("output_proj.", projection)):
        selected = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        module.load_state_dict(selected, strict=True)
        loaded[prefix] = {"strict": True, "tensors": len(selected),
                          "elements": sum(value.numel() for value in selected.values())}
    del payload, state, selected
    extension = extend_decoder_vocabulary(decoder, codec.vocab_size, seed=initial["id_seed"])
    for module in decoder.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.
        for key in ("dropout", "attention_dropout", "activation_dropout", "layerdrop"):
            if isinstance(getattr(module, key, None), (int, float)):
                setattr(module, key, 0.)
    decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reset_random(initial["seed"])
    model = IdXYZRig(FixedQuerySurfaceTokenizer(mesh, projection), decoder, IdXYZConfig(**config["architecture"]))
    fingerprint = parameter_fingerprint(model)
    report = {"source": source, "official_checkpoint_sha256": initial["sha256"],
              "load_report": loaded, "vocabulary_extension": extension,
              "initial_model_sha256": fingerprint, "trained_experiment_weights_loaded": False,
              "architecture": model.configuration(), "codec": codec.configuration(),
              "groups": {name: sum(p.numel() for p in getattr(model, name).parameters() if p.requires_grad)
                         for name in ("surface_tokenizer", "motion_encoder", "xyz_decoder")}}
    model.to(device)
    return model, tokenizer, codec, report


def build_targets(geometry, codec, device):
    sizes = geometry["joint_count"].cpu().tolist()
    rows = [codec.encode(geometry["target_parents"][i, :count].detach().cpu().numpy(),
                         geometry["frame_target_joints"][i, :, :count].detach().cpu().numpy())
            for i, count in enumerate(sizes)]
    batch, frames = len(rows), geometry["frame_target_joints"].shape[1]
    length = max(row["input_ids"].shape[1] for row in rows)
    ids = torch.full((batch, frames, length), codec.pad, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids, dtype=torch.bool)
    structural = torch.zeros_like(attention)
    coordinate = torch.zeros_like(attention)
    for index, row in enumerate(rows):
        width = row["input_ids"].shape[1]
        ids[index, :, :width] = torch.as_tensor(row["input_ids"], device=device)
        attention[index, :, :width] = True
        structural[index, :, :width] = torch.as_tensor(row["structure_mask"], device=device)
        coordinate[index, :, :width] = (ids[index, :, :width] >= 0) & (ids[index, :, :width] < 256)
    return {"input_ids": ids, "attention_mask": attention, "structure_mask": structural,
            "coordinate_mask": coordinate, "frame_mask": geometry["frame_mask"],
            "lengths": [row["input_ids"].shape[1] for row in rows],
            "dfs_orders": [row["dfs_order"].tolist() for row in rows]}


def prepare_batch(dataset, requests, config, codec, device):
    surface, geometry = prepare_virtual_surface([dataset[request] for request in requests], device,
                                                **config["surface_sampling"])
    targets = build_targets(geometry, codec, device)
    return surface, targets, geometry


def make_stream(config, rank, world, state=None):
    return MatchedFrameStream(
        range(5000), rank=rank, world_size=world, seed=config["data"]["stream_seed"],
        target_assets=config["training"]["target_assets"], effective_assets=48,
        frame_policy=config["training"]["frame_policy"], manifest_sha256=TRAIN_SHA,
        state=state, update_boundaries=config["training"]["checkpoint_assets"])


def optimizer_for(model, config):
    groups = [{"name": name, "params": [p for p in getattr(model, name).parameters() if p.requires_grad]}
              for name in ("surface_tokenizer", "motion_encoder", "xyz_decoder")]
    ids = [id(p) for group in groups for p in group["params"]]
    if len(ids) != len(set(ids)):
        raise ValueError("optimizer groups alias parameters")
    return torch.optim.AdamW(groups, lr=config["optimization"]["peak_lr"], betas=(.95, .999),
                              weight_decay=config["optimization"]["weight_decay"])


def set_learning_rate(optimizer, samples, config):
    p = config["optimization"]
    lr, beta1 = exposure_learning_rate(samples, p["schedule_samples"], peak=p["peak_lr"],
        warmup_fraction=p["warmup_fraction"], div_factor=p["div_factor"], final_div_factor=p["final_div_factor"])
    for group in optimizer.param_groups:
        group["lr"], group["betas"] = lr, (beta1, .999)
    return lr, beta1


def gradient_groups(model):
    report = {}
    for name in ("surface_tokenizer", "motion_encoder", "xyz_decoder"):
        gradients = [p.grad.detach().float() for p in getattr(model, name).parameters()
                     if p.requires_grad and p.grad is not None]
        if not gradients or any(not bool(torch.isfinite(g).all()) for g in gradients):
            raise RuntimeError(f"missing or nonfinite {name} gradients")
        norm = sum(float(g.square().sum()) for g in gradients) ** .5
        if norm <= 0:
            raise RuntimeError(f"zero {name} gradient")
        report[name] = norm
    return report


def training_contract(config):
    contract = {key: config[key] for key in
            ("family", "architecture", "model_contract", "codec", "surface_sampling", "data",
             "initialization", "optimization", "consistency", "training", "evaluation",
             "final_evaluation", "reference_evaluation")}
    contract["tokenizer"] = {key: config["runtime"][key] for key in
                             ("tokenizer_config", "tokenizer_config_sha256")}
    return contract


def paired_contract_hash(config):
    contract = training_contract(config)
    contract["consistency"] = {k: v for k, v in contract["consistency"].items() if k != "weight"}
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
