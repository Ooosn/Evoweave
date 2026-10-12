#!/usr/bin/env python3
"""Fixed-data reference controls, never replacements for the approved model."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'model_training/rigweave/src'))
from rigweave.dynamic_rig.motion_encoder import AnchorWiseAlternatingMotionEncoder
from rigweave.dynamic_rig.motion_evidence import grouped_motion_evidence
from rigweave.dynamic_rig.virtual_query_xyz import VirtualQueryXYZRig


class ReferenceConditionRig(nn.Module):
    """Reuse the actual surface, legacy encoder and verified original XYZ loss."""
    def __init__(self, base, route):
        super().__init__()
        if route not in ('static_surface', 'legacy_query_first', 'virtual_final_real'):
            raise ValueError('unrecognized reference control')
        self.base = base
        self.route = route

    @property
    def config(self):
        return self.base.config

    @property
    def surface_tokenizer(self):
        return self.base.surface_tokenizer

    @property
    def motion_encoder(self):
        return self.base.motion_encoder

    @property
    def frame_fusion(self):
        return self.base.frame_fusion

    @property
    def condition_norm(self):
        return self.base.condition_norm

    @property
    def xyz_decoder(self):
        return self.base.xyz_decoder

    @property
    def auxiliary(self):
        return self.base.auxiliary

    def encode(self, surface):
        if self.route == 'virtual_final_real':
            return self.base.encode(surface)
        frames = surface['query_points'].shape[1]
        tokens = []
        for frame in range(frames):
            args = tuple(surface[key][:, frame] for key in
                         ('dense_points', 'dense_normals', 'query_points', 'query_normals'))
            if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
                tokens.append(checkpoint(self.surface_tokenizer, *args, use_reentrant=False))
            else:
                tokens.append(self.surface_tokenizer(*args))
        tokens = torch.stack(tokens, 1)
        if self.route == 'static_surface':
            return tokens, None
        conditions = []
        for frame in range(frames):
            order = [frame]+[other for other in range(frames) if other != frame]
            evidence = None
            if self.config.motion_evidence_fusion != 'off':
                # The old grouping convention uses the query pose as frame 0.
                evidence = grouped_motion_evidence(surface['dense_points'][:, order].detach(),
                                                   surface['query_indices'])
            conditions.append(AnchorWiseAlternatingMotionEncoder.forward(self.motion_encoder,
                tokens[:, order], query_points=surface['query_points'][:, order], motion_evidence=evidence))
        return torch.stack(conditions, 1), None

    def conditions(self, final_frames, virtual):
        return final_frames

    def main_logits(self, condition, input_ids, attention_mask):
        return self.base.main_logits(condition, input_ids, attention_mask)

    def forward_encoded(self, final_frames, virtual, targets, *, auxiliary_weight):
        if auxiliary_weight != 0:
            raise ValueError('reference control has no auxiliary structure branch')
        return VirtualQueryXYZRig.forward_encoded(self, final_frames, virtual, targets, auxiliary_weight=0.)

    def forward(self, surface, targets, *, auxiliary_weight):
        return self.forward_encoded(*self.encode(surface), targets, auxiliary_weight=auxiliary_weight)

    def generate_condition(self, *args, **kwargs):
        return self.base.generate_condition(*args, **kwargs)


def declared_reference_run(operation, args, config_digest):
    matches = [run for run in operation.get('runs', [operation])
               if Path(run['config']).resolve() == args.config.resolve()]
    if len(matches) != 1:
        raise ValueError('reference configuration is not uniquely declared')
    run = matches[0]
    if run['config_sha256'] != config_digest:
        raise ValueError('wrong reference configuration')
    if Path(run['output']).resolve() != args.output.resolve():
        raise ValueError('wrong reference output')
    return run


def verify(args):
    from virtual_query_xyz_runtime import sha256
    state = json.loads((ROOT/'model_training/state/current.json').read_text())
    op = state['active_operation']
    if state['architecture_contract']['id'] != 'virtual_query_xyz_aux_ids_v1':
        raise ValueError('wrong current route')
    if not op.get('gpu_work_allowed') or op.get('formal_training_allowed') or op['mode'] != 'reference_fit':
        raise ValueError('reference fit is not the authorized bounded operation')
    declared_reference_run(op, args, sha256(args.config))
    subprocess.run([sys.executable, str(ROOT/'model_training/tools/agent_work_guard.py'), 'check',
        '--operation', 'train', '--entrypoint', str(Path(__file__).resolve())], check=True)
    if subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
        raise ValueError('reference control requires clean published source')
    subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', op['code_commit'], 'HEAD'], check=True)
    subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet', op['code_commit'], 'HEAD', '--',
        'model_training/rigweave', 'model_training/tools'], check=True)
    return subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()


def main():
    import probe_virtual_query_xyz as probe
    from virtual_query_xyz_runtime import build_model, write_json
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    source = verify(args)
    route = config['reference_route']
    if config['probe']['auxiliary_weight'] != 0 or config['architecture'].get('frame_residual_scale', 0) != 0:
        raise ValueError('reference must not enable the auxiliary loss or residual candidate')
    args.output.mkdir(parents=True, exist_ok=False)
    args.mode = 'fit'
    def builder(cfg, device):
        model, tokenizer, codec, initial = build_model(cfg, device)
        initial['reference_route'] = route
        initial['approved_virtual_route'] = False
        return ReferenceConditionRig(model, route), tokenizer, codec, initial
    def require_reference_gradients(model, norms, weight):
        required = ['surface_tokenizer', 'xyz_decoder']
        if route in ('legacy_query_first', 'virtual_final_real'):
            required.append('motion_encoder')
        if any(norms[name] <= 0 for name in required):
            raise RuntimeError('missing required reference gradient: '+json.dumps(norms))
        if route == 'virtual_final_real':
            gradient = model.motion_encoder.virtual_slots.grad
            if gradient is None or not bool(torch.isfinite(gradient).all()) or float(gradient.norm()) <= 0:
                raise RuntimeError('module-isolation control must retain virtual-slot learning through the encoder')
        for module in (model.xyz_decoder.get_input_embeddings(), model.xyz_decoder.get_output_embeddings()):
            if module.weight.grad is None or not torch.isfinite(module.weight.grad).all():
                raise RuntimeError('missing or nonfinite original XYZ embedding/head gradient')
    try:
        with patch.object(probe, 'build_model', builder), patch.object(probe, 'require_gradients', require_reference_gradients):
            result = probe.run(args, config, source)
        write_json(args.output/'result.json', {**result, 'reference_route': route, 'approved_virtual_route': False,
            'acceptance_meaning': 'reference_control_fitting_only_not_candidate_acceptance',
            'execution_complete': True, 'finished_utc': datetime.now(timezone.utc).isoformat(), 'code_commit': source})
    except Exception:
        write_json(args.output/'result.json', {'accepted': False, 'execution_complete': False,
            'reference_route': route, 'formal_training_allowed': False,
            'traceback': traceback.format_exc(), 'code_commit': source})
        raise


if __name__ == '__main__':
    main()
