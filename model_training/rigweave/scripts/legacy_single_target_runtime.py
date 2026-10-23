"""Shared data and evaluation contracts for the real-query single-target run."""
from __future__ import annotations

from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from virtual_query_xyz_runtime import build_model as build_virtual_base
from virtual_query_xyz_runtime import fixed_dataset, read_entries, sha256, write_json
from virtual_fusion_normal_runtime import MatchedFrameStream
from virtual_frame_runtime import exposure_learning_rate
from probe_virtual_query_lambda import parameter_fingerprint
from rigweave.dynamic_rig.virtual_query_xyz import VirtualQueryXYZRig


FAMILY = 'legacy_real_query_single_target_v1'
BASE_INITIAL_SHA256 = 'a89b29c903abffe2eead47b5411edb8588b599fe874ce490c990ce07225d3635'
TRAIN_SHA256 = 'a17b6be96fb33a32a1bee7117b07789da6261c6b5693104f3d2cce07c09bab88'
VALID_SHA256 = 'fd2be7b5885833946e68dc186300a820757ae652c487749eec1b049d381e230a'
EXCLUDED_PARTS = ('motion_encoder.virtual_slots', 'frame_fusion.', 'condition_norm.', 'auxiliary.')


def validate_config(config):
    if config.get('family') != FAMILY or config.get('target_policy') != 'sampled_random_query_at_slot0':
        raise ValueError('only the sampled real query may be the training target')
    data, plan, optim = (config[k] for k in ('data', 'training', 'optimization'))
    expected_data = {'frames_min': 2, 'frames_max': 24, 'heldout_frames': 8,
                     'normalization': 'input_sequence_mesh_bbox', 'global_rt_removal': True,
                     'coordinate_clipping': False, 'frame_seed': 20261013,
                     'surface_seed': 20260529, 'stream_seed': 20261013, 'max_main_tokens': 2048}
    if any(data.get(k) != v for k, v in expected_data.items()):
        raise ValueError('the current dynamic sampling/RT/normalization contract changed')
    for split, rows, digest in (('train', 5000, TRAIN_SHA256), ('valid', 967, VALID_SHA256)):
        spec = data['manifests'][split]
        if spec['rows'] != rows or spec['sha256'] != digest:
            raise ValueError('the fixed5k comparison data changed')
    if config.get('pruning') != {'enabled': False} or config.get('global_rotation_augmentation') is not False:
        raise ValueError('single-target restoration must not inherit extra augmentation')
    milestones = list(range(5000, 100001, 5000))
    if (plan.get('target_assets') != 100000 or plan.get('effective_assets_per_update') != 48
            or plan.get('checkpoint_assets') != milestones or plan.get('evaluation_assets') != milestones):
        raise ValueError('the100k asset budget and exact5k checkpoints are fixed')
    for frames in range(2, 25):
        batch = plan['frame_policy'].get(str(frames))
        if type(batch) is not int or batch < 1 or 24 % batch:
            raise ValueError('physical microbatches must divide the24asset per-rank update')
    expected_optim = {'schedule': 'b65_lineage_replay', 'peak_lr': 1e-4, 'warmup_fraction': .1,
                      'div_factor': 5, 'final_div_factor': 10, 'weight_decay': 0,
                      'schedule_samples': 20000, 'clip_grad_norm': 1, 'auxiliary_weight': 0,
                      'constant_after20000': 1e-4, 'constant_after60000': 3e-5}
    if optim != expected_optim:
        raise ValueError('replay the bestB lineage without a new100k cycle or auxiliary loss')
    expected_arch = {'dim': 1024, 'heads': 8, 'query_tokens': 1024, 'register_tokens': 96,
                     'motion_depth': 12, 'fusion_depth': 1, 'auxiliary_depth': 2,
                     'auxiliary_max_tokens': 1024, 'max_frames': 24, 'dropout': 0,
                     'motion_evidence_fusion': 'bias', 'motion_evidence_heads': 8,
                     'gradient_checkpointing': True, 'temporal_chunk_size': 4096,
                     'fusion_mode': 'real_queries_virtual'}
    if config['architecture'] != expected_arch:
        raise ValueError('construction must preserve exact common B initialization')
    if config['surface_sampling'] != {'surface_samples': 65536, 'vertex_samples': 8192, 'query_tokens': 1024}:
        raise ValueError('surface sampling differs from B')
    if config['evaluation'] != {'train_assets': 32, 'valid_assets': 128, 'seed': 20261014,
                                'frames': 8, 'generated_frame_indices': list(range(8)),
                                'max_new_tokens': 2048, 'initial': True}:
        raise ValueError('evaluation must retain the original1536view comparison panel')
    if (config['runtime']['world_size'] != 2 or config['runtime']['physical_gpus'] != [5, 6]
            or config['runtime']['allocation'] != 130309880 or config['runtime']['host'] != 'pcg01i'):
        raise ValueError('only the existing two granted H100s are authorized')
    if (config['runtime'].get('execution_mode') != 'explicit_ssh_exception'
            or config['runtime'].get('require_scheduler_membership') is not False
            or config['runtime'].get('gpu_memory_fraction_limit') != .85):
        raise ValueError('this run requires its explicitly published, resource-bounded SSH exception')
    if config['capacity']['allocated_fraction_limit'] != .85:
        raise ValueError('capacity must retain at least15percent GPU headroom')
    return milestones


def make_stream(config, rank, world=2):
    if world != 2:
        raise ValueError('the exposure stream is defined for exactly two ranks')
    return MatchedFrameStream(range(5000), rank=rank, world_size=world,
        seed=config['data']['stream_seed'], target_assets=100000, effective_assets=48,
        frame_policy=config['training']['frame_policy'], manifest_sha256=TRAIN_SHA256,
        update_boundaries=config['training']['checkpoint_assets'])


def set_learning_rate(optimizer, samples, config):
    if type(samples) is not int or not 0 <= samples < 100000:
        raise ValueError('learning-rate position must be an actual pre-update asset count')
    p = config['optimization']
    if samples < 20000:
        lr, beta1 = exposure_learning_rate(samples, 20000, peak=p['peak_lr'],
            warmup_fraction=p['warmup_fraction'], div_factor=p['div_factor'],
            final_div_factor=p['final_div_factor'])
        for group in optimizer.param_groups:
            group['betas'] = (beta1, group['betas'][1])
    else:
        lr = p['constant_after20000'] if samples < 60000 else p['constant_after60000']
        betas = [group['betas'] for group in optimizer.param_groups]
        if not betas or any(value != betas[0] for value in betas):
            raise ValueError('continuation must retain the last actual common Adam betas')
        beta1 = betas[0][0]
    for group in optimizer.param_groups:
        group['lr'] = lr
    return lr, beta1


def build_model(config, device):
    from rigweave.dynamic_rig.legacy_single_target import LegacySingleTargetRig
    base, tokenizer, codec, initial = build_virtual_base(config, torch.device('cpu'))
    full = parameter_fingerprint(base)
    if full != BASE_INITIAL_SHA256:
        raise ValueError('fresh official B construction differs from its pinned initial tensors')
    expected = {k: v for k, v in base.state_dict().items() if not any(p in k for p in EXCLUDED_PARTS)}
    common = parameter_fingerprint(base, exclude_parts=EXCLUDED_PARTS)
    model = LegacySingleTargetRig.from_virtual_base(base)
    actual = model.state_dict()
    if actual.keys() != expected.keys() or any(
            actual[k].dtype != v.dtype or actual[k].shape != v.shape or not torch.equal(actual[k], v)
            for k, v in expected.items()):
        raise ValueError('single-target extraction changed common initial tensors')
    restored = parameter_fingerprint(model)
    if restored != common:
        raise ValueError('single-target common parameter fingerprint changed')
    initial.update(family=FAMILY, full_B_initial_state_sha256=full,
        common_initial_state_sha256=common, actual_initial_state_sha256=restored,
        exact_common_tensors=len(actual), excluded_parts=list(EXCLUDED_PARTS),
        virtual_or_auxiliary_modules_retained=False, old_method_trained_weights_loaded=False,
        groups={name: sum(p.numel() for p in getattr(model, name).parameters() if p.requires_grad)
                for name in ('surface_tokenizer', 'motion_encoder', 'xyz_decoder')})
    del actual, expected, base
    gc.collect()
    model.to(device)
    return model, tokenizer, codec, initial


def train_target_indices(surface):
    points = surface['query_points']
    if points.ndim != 4 or not 2 <= points.shape[1] <= 24:
        raise ValueError('training requires the sampled2..24pose sequence')
    return torch.zeros(points.shape[0], dtype=torch.long, device=points.device)


def verify_reference(config, entries):
    from train_virtual_fusion_normal import selected_evaluation_rows, verify_selection
    verify_selection(config, entries)
    refs = config['reference_evaluation']['files']
    for spec in refs.values():
        if sha256(spec['path']) != spec['sha256']:
            raise ValueError('the bestB reference artifact changed')
    selected = json.loads(Path(refs['selection']['path']).read_text())
    chosen = selected_evaluation_rows(entries, config)
    if selected != chosen:
        raise ValueError('the evaluation panel differs from bestB')
    reports = [json.loads(Path(refs[f'rank_{rank}']['path']).read_text()) for rank in (0, 1)]
    return {group: {(r['asset_id'], r['frame_index']): r
                   for report in reports for r in report['groups'][group]['rows']}
            for group in chosen}


class EvaluationAdapter(nn.Module):
    """Use the unchanged original scorer; all-target decoding is evaluation only."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config

    def encode(self, surface):
        tokens = self.model.encode_surface(surface)
        conditions = []
        for frame in range(tokens.shape[1]):
            target = torch.full((tokens.shape[0],), frame, dtype=torch.long, device=tokens.device)
            conditions.append(self.model.condition_from_tokens(surface, tokens, target))
        return torch.stack(conditions, dim=1), None

    def conditions(self, frames, virtual):
        if virtual is not None:
            raise ValueError('the real-query evaluation has no virtual representation')
        return frames

    def main_logits(self, *args, **kwargs):
        return self.model.main_logits(*args, **kwargs)

    def forward_encoded(self, frames, virtual, targets, *, auxiliary_weight):
        if auxiliary_weight != 0:
            raise ValueError('auxiliary loss is absent from this baseline')
        return VirtualQueryXYZRig.forward_encoded(self, frames, virtual, targets, auxiliary_weight=0)

    def generate_condition(self, *args, **kwargs):
        return self.model.generate_condition(*args, **kwargs)


@torch.no_grad()
def run_evaluation(model, tokenizer, codec, entries, config, args, device, samples, rank, world, reference):
    import train_virtual_fusion_normal as normal
    import train_virtual_query_xyz as core
    adapter = EvaluationAdapter(model)
    incoming_mode = model.training
    chosen = normal.selected_evaluation_rows(entries, config)
    root = args.output / f'evaluation_{samples:06d}'
    root.mkdir(exist_ok=True)
    report = {'complete': False, 'samples': samples, 'rank': rank, 'groups': {}}
    for group, rows in chosen.items():
        local = rows[rank::world]
        dataset = fixed_dataset(local, config, 8, mode='heldout_eval' if group == 'train_heldout' else 'seen_eval')
        truths = {}

        def items():
            for index in range(len(dataset)):
                surface, target, geometry = core.prepare(dataset, [index], config, tokenizer, codec, device)
                selected = geometry['selected_frames'][0].cpu().tolist()
                reserved = geometry['heldout_raw_ids'][0]
                if ((group == 'train_heldout' and set(selected) != set(reserved))
                        or (group != 'train_heldout' and set(selected) & set(reserved))):
                    raise ValueError('evaluation crossed the fixed frame partition')
                size = int(geometry['joint_count'][0])
                truths[geometry['asset_id'][0]] = {
                    'count': size, 'parents': geometry['target_parents'][0, :size].cpu().tolist(),
                    'joints': geometry['frame_target_joints'][0, :, :size].cpu().tolist()}
                yield surface, target, geometry

        directory = root / f'{group}_rank_{rank}'
        directory.mkdir(exist_ok=False)
        eval_config = {**config, 'probe': {'auxiliary_weight': 0, 'max_new_tokens': 2048,
                                         'evaluate_auxiliary': False}}
        result = core.evaluate(adapter, items(), tokenizer, eval_config, directory, samples,
                               frame_indices=list(range(8)))
        normal.validate_evaluation_rows(result['rows'], local, 8)
        for row in result['rows']:
            base = reference[group][row['asset_id'], row['frame_index']]
            if row['raw_frame_index'] != base['raw_frame_index'] or row['target_tokens'] != base['target_tokens']:
                raise ValueError('actual target pose/tokenization differs from bestB')
            truth = truths[row['asset_id']]
            row.update(original_target_joint_count=truth['count'], original_target_parents=truth['parents'],
                       original_target_joints=truth['joints'][row['frame_index']],
                       bestB_original_edge_f1=base['original_edge_f1'])
        report['groups'][group] = result
        write_json(root / f'rank_{rank}.json', report)
    report['complete'] = True
    write_json(root / f'rank_{rank}.json', report)
    if world > 1:
        dist.barrier()
    if rank == 0:
        ranks = [json.loads((root / f'rank_{r}.json').read_text()) for r in range(world)]
        summaries, strata = {}, {}
        for group, selected in chosen.items():
            rows = [row for part in ranks for row in part['groups'][group]['rows']]
            normal.validate_evaluation_rows(rows, selected, 8)
            summaries[group] = core.evaluation_summary(rows)
            strata[group] = {}
            for name, subset in (('GT52', [r for r in rows if r['original_target_joint_count'] == 52]),
                                 ('non52', [r for r in rows if r['original_target_joint_count'] != 52])):
                strata[group][name] = {'views': len(subset), **core.evaluation_summary(subset)} if subset else {'views': 0}
        write_json(root / 'summary.json', {'complete': True, 'samples': samples, 'summary': summaries,
            'strata': strata, 'frames_per_asset': 8, 'evaluation_assets': {k: len(v) for k, v in chosen.items()},
            'failed_generations_scored_zero': True, 'bestB_target_identity_verified': True,
            'quality_accepted': False, 'full_validation': False})
    if world > 1:
        dist.barrier()
    model.train(incoming_mode)


def training_contract(config):
    return {k: deepcopy(config[k]) for k in ('family', 'target_policy', 'data', 'architecture',
        'surface_sampling', 'initialization', 'optimization', 'pruning', 'global_rotation_augmentation',
        'training', 'evaluation', 'reference_evaluation', 'auxiliary_codec', 'normal_match')}


def contract_sha256(config):
    return hashlib.sha256(json.dumps(training_contract(config), sort_keys=True, separators=(',', ':')).encode()).hexdigest()
