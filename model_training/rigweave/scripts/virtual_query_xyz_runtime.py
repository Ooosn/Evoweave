"""Pinned original XYZ decoder initialization for the approved virtual-query route."""
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

from rigweave.dynamic_rig.aux_skeleton_tokens import AuxSkeletonTokens
from rigweave.dynamic_rig.surface_tokenizer import FixedQuerySurfaceTokenizer
from rigweave.dynamic_rig.virtual_query_xyz import VirtualQueryXYZConfig, VirtualQueryXYZRig
from rigweave.dynamic_rig.virtual_frame_data import VirtualFrameDataset, read_virtual_manifest, prepare_virtual_surface
from rigweave.dynamic_rig.virtual_query_xyz_data import build_xyz_aux_targets


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def build_model(config, device, *, load_official_weights=True, initialization_seed=None):
    import yaml
    from transformers import OPTConfig, OPTForCausalLM
    from src.model.michelangelo.get_model import get_encoder_simplified
    from train_dynamic_rig import build_tokenizer

    if type(load_official_weights) is not bool:
        raise ValueError('load_official_weights must be a boolean')
    initial, runtime = config['initialization'], config['runtime']
    seed = initial['seed'] if initialization_seed is None else initialization_seed
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError('invalid model initialization seed')
    upstream = runtime['unirig_root']
    revision = subprocess.check_output(['git', '-C', upstream, 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', upstream, 'status', '--porcelain'], text=True).strip()
    if revision != initial['upstream_source_commit'] or dirty:
        raise ValueError('official upstream implementation is not the declared clean revision')
    if Path(initial['checkpoint']).resolve() != Path(initial['official_checkpoint']).resolve():
        raise ValueError('only the official upstream checkpoint may initialize this route')
    if load_official_weights and sha256(initial['checkpoint']) != initial['sha256']:
        raise ValueError('official initialization checkpoint hash changed')
    if sha256(initial['model_config']) != initial['model_config_sha256']:
        raise ValueError('published original XYZ model configuration hash changed')
    if sha256(runtime['tokenizer_config']) != runtime['tokenizer_config_sha256']:
        raise ValueError('published original XYZ tokenizer configuration hash changed')
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    tokenizer = build_tokenizer(Path(runtime['tokenizer_config']))
    upstream_config = yaml.safe_load(Path(initial['model_config']).read_text())
    if load_official_weights:
        payload = torch.load(initial['checkpoint'], map_location='cpu', mmap=True, weights_only=False)
        state = {key.removeprefix('model.'): value for key, value in payload['state_dict'].items()}
    lm_args = dict(upstream_config['llm'])
    lm_args.pop('pretrained_model_name_or_path')
    lm_args['vocab_size'] = tokenizer.vocab_size
    if torch.device(device).type == 'cpu':
        # Weight-only CPU audits do not execute the CUDA attention backend.
        lm_args['_attn_implementation'] = 'eager'
    lm_config = OPTConfig.from_pretrained(runtime['opt_config_root'], local_files_only=True, **lm_args)
    lm_config.torch_dtype = torch.float32
    main = OPTForCausalLM(lm_config)
    surface_args = dict(upstream_config['mesh_encoder'])
    if surface_args.pop('__target__') != 'michelangelo_encoder':
        raise ValueError('unexpected upstream surface encoder')
    if not load_official_weights and surface_args.get('pretrained_path') is not None:
        raise ValueError('random initialization cannot use an encoder checkpoint')
    surface = get_encoder_simplified(**surface_args)
    projection = nn.Linear(surface.width, lm_config.hidden_size)
    report, loaded = {}, set()
    if load_official_weights:
        for prefix, module in [('transformer.', main), ('mesh_encoder.', surface), ('output_proj.', projection)]:
            selected = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            module.load_state_dict(selected, strict=True)
            loaded.update(k for k in state if k.startswith(prefix))
            report[prefix] = {'strict': True, 'tensors': len(selected),
                              'parameters': sum(v.numel() for v in selected.values())}
        if 'transformer.lm_head.weight' not in loaded or 'transformer.model.decoder.embed_tokens.weight' not in loaded:
            raise ValueError('full XYZ input embedding and output head must both load')
        report['unused_upstream_keys'] = sorted(set(state)-loaded)
        del payload, state, selected
    else:
        report = {'random_constructor_initialization': True, 'checkpoint_tensors_loaded': 0}
    for module in main.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.
        for name in ('dropout', 'attention_dropout', 'activation_dropout', 'layerdrop'):
            if isinstance(getattr(module, name, None), (int, float)):
                setattr(module, name, 0.)
    main.requires_grad_(True)
    main.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    architecture = VirtualQueryXYZConfig(**config['architecture'])
    if architecture.query_tokens != 1024 or architecture.motion_depth != 12:
        raise ValueError('real runtime must use the approved 1024-slot, 12-block architecture')
    codec = AuxSkeletonTokens(**config['auxiliary_codec'])
    torch.manual_seed(seed)
    model = VirtualQueryXYZRig(FixedQuerySurfaceTokenizer(surface, projection), main, architecture,
                               auxiliary_vocab_size=codec.vocab_size, auxiliary_pad=codec.pad)
    model.to(device)
    initialization = {'source_commit': revision, 'checkpoint_sha256': initial['sha256'] if load_official_weights else None,
                      'model_config_sha256': sha256(initial['model_config']),
                      'tokenizer_config_sha256': sha256(runtime['tokenizer_config']),
                      'seed': seed, 'load_report': report, 'architecture': asdict(architecture),
                      'groups': {name: sum(p.numel() for p in getattr(model, name).parameters() if p.requires_grad)
                                 for name in ('surface_tokenizer', 'motion_encoder', 'frame_fusion', 'xyz_decoder', 'auxiliary')},
                      'old_method_trained_weights_loaded': False,
                      'xyz_embedding_trainable': model.xyz_decoder.get_input_embeddings().weight.requires_grad,
                      'xyz_head_trainable': model.xyz_decoder.get_output_embeddings().weight.requires_grad}
    return model, tokenizer, codec, initialization


def read_entries(config):
    result = {}
    for split, spec in config['data']['manifests'].items():
        rows = read_virtual_manifest(spec['path'], expected_count=spec['rows'], expected_sha256=spec['sha256'])
        if any(r.get('root_rt', {}).get('version') != 'retained_root_rigid_v1' for r in rows):
            raise ValueError('this route requires the retained-root RT descriptor on every row')
        result[split] = rows
    return result


def fixed_dataset(rows, config, frames, mode='seen_eval', *, file_resolver=None):
    data = config['data']
    return VirtualFrameDataset(rows, frame_seed=data['frame_seed'], sampling_seed=data['surface_seed'],
        frames=frames, heldout_frames=data['heldout_frames'], mode=mode,
        motion_fps_ratio=.75, minimum_random_frames=1, motion_vertex_samples=512,
        file_resolver=file_resolver)


def prepare_batch(batch, config, tokenizer, codec, device):
    surface, geometry = prepare_virtual_surface([batch], device, **config['surface_sampling'])
    target = build_xyz_aux_targets(geometry, tokenizer, codec, device=device,
                                  max_main_tokens=config['data']['max_main_tokens'])
    return surface, target, geometry
