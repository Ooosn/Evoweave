"""First-stage feedback checks; no launcher, optimizer, or GPU admission path."""
from collections import defaultdict
from contextlib import nullcontext
import math
import statistics

import numpy as np
import torch

from rigweave.dynamic_rig.skeleton_topology_metrics import _topology_metrics
from rigweave.dynamic_rig.virtual_query_xyz import original_xyz_vocabulary_processor


def backbone_state(payload, template, *, layout):
    """Extract the complete V- backbone without loading the stage2 head."""
    state = payload['model']
    if layout == 'density':
        result = dict(state)
    elif layout == 'anchor':
        if not any(name.startswith('head.') for name in state):
            raise ValueError('anchor checkpoint has no stage2 head')
        if any(not name.startswith(('backbone.', 'head.')) for name in state):
            raise ValueError('unrecognized anchor state namespace')
        result = {name.removeprefix('backbone.'): value for name, value in state.items()
                  if name.startswith('backbone.')}
    else:
        raise ValueError('checkpoint layout must be explicitly density or anchor')
    if set(result) != set(template) or not result:
        raise ValueError('incomplete or unexpected backbone keys')
    for name, value in result.items():
        expected = template[name]
        if (not isinstance(value, torch.Tensor) or value.shape != expected.shape
                or value.dtype != expected.dtype):
            raise ValueError('backbone tensor contract differs: '+name)
    return result


def compare_backbone_weights(donor, current, *, require_unchanged_nonshared=False):
    """Compare all parameters AND buffers; shared changes alone are not quality."""
    current = backbone_state({'model': current}, donor, layout='density')
    groups = {'shared': [], 'nonshared': []}
    changed = {'shared': [], 'nonshared': []}
    for name, before in donor.items():
        group = ('shared' if name.startswith(('base.surface_tokenizer.', 'base.motion_encoder.'))
                 else 'nonshared')
        groups[group].append(name)
        after = current[name]
        left = before.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        right = after.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        if not torch.equal(left, right):
            changed[group].append(name)
    if not all(groups.values()):
        raise ValueError('expected shared and nonshared backbone tensors')
    if require_unchanged_nonshared and changed['nonshared']:
        raise ValueError('stage2-only changed nonshared backbone tensors: '+repr(changed['nonshared']))
    return {group: {'tensors': len(groups[group]), 'changed': sorted(changed[group]),
                    'all_unchanged': not changed[group]} for group in groups}


@torch.inference_mode()
def generate_asset(model, surface, tokenizer, *, frame_indices, max_new_tokens=2048, precision='bf16'):
    """Only mesh input and the original BOS/class prompt reach generation."""
    if getattr(model, 'virtual_enabled', None) is not False:
        raise ValueError('feedback comparison requires the established V- backbone')
    if set(surface) != {'dense_points', 'dense_normals', 'query_points', 'query_normals', 'query_indices'}:
        raise ValueError('generation accepts only the canonical mesh-surface fields')
    points = surface['query_points']
    if points.ndim != 4 or points.shape[0] != 1:
        raise ValueError('generation requires one multi-frame asset')
    frames = list(frame_indices)
    if (not frames or len(set(frames)) != len(frames)
            or any(type(f) is not int or not 0 <= f < points.shape[1] for f in frames)):
        raise ValueError('invalid or duplicate generation frame slots')
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError('token budget must be a positive integer')
    if precision not in ('bf16', 'fp32'):
        raise ValueError('generation precision must be explicitly bf16 or fp32')
    was_training = model.training
    tf32 = torch.backends.cuda.matmul.allow_tf32
    model.eval()
    try:
        if points.is_cuda and precision == 'fp32':
            torch.backends.cuda.matmul.allow_tf32 = False
        context = (torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision == 'bf16')
                   if points.is_cuda else nullcontext())
        with context:
            real, virtual = model.encode(surface)
            conditions = model.conditions(real, virtual)
            if conditions.shape[:2] != points.shape[:2]:
                raise ValueError('condition batch/frame identity differs')
            output = {}
            for frame in frames:
                start = torch.tensor([tokenizer.bos, tokenizer.cls_name_to_token('articulationxl')],
                                     device=points.device, dtype=torch.long)
                output[frame] = model.generate_condition(conditions[:, frame], start,
                    eos_token=tokenizer.eos, max_new_tokens=max_new_tokens,
                    logits_processor=original_xyz_vocabulary_processor(tokenizer, start))
        return output
    finally:
        if points.is_cuda:
            torch.backends.cuda.matmul.allow_tf32 = tf32
        model.train(was_training)


def _skeleton(joints, parents):
    xyz = np.asarray(joints)
    if (not np.issubdtype(xyz.dtype, np.floating) or xyz.ndim != 2
            or xyz.shape[1:] != (3,) or not len(xyz) or not np.isfinite(xyz).all()):
        raise ValueError('nonfinite, empty or malformed skeleton coordinates')
    parents = [None if p is None else int(p) for p in parents]
    if len(parents) != len(xyz):
        raise ValueError('joint/parent count differs')
    if any(p is not None and (p < -1 or p >= len(xyz) or p == child)
           for child, p in enumerate(parents)):
        raise ValueError('parent index outside skeleton or self edge')
    return xyz, parents


def score_generation(predicted, tokenizer, truth, parents, diagonal):
    """Score original GT, with conditional geometry and zero failure quality."""
    truth, parents = _skeleton(truth, parents)
    diagonal = float(diagonal)
    if not math.isfinite(diagonal) or diagonal <= 0:
        raise ValueError('target mesh bounding-box diagonal must be positive')
    record = {'success': False, 'status': predicted['status'], 'original_edge_f1': 0.,
              'exact_count': False, 'gt_joint_count': len(truth), 'predicted_joint_count': None,
              'absolute_joint_count_error': None, 'original_normalized_j2j': None,
              'target_mesh_diagonal': diagonal, 'generated_tokens': predicted['tokens']}
    if not predicted['success']:
        return record
    if predicted['status'] != 'eos' or not predicted['tokens'] or predicted['tokens'][-1] != tokenizer.eos:
        raise ValueError('successful generation must end with EOS')
    try:
        parsed = tokenizer.detokenize(np.asarray(predicted['tokens'], dtype=np.int64))
        parsed_parents = parsed.parents if getattr(parsed, 'parents', None) is not None else parsed._get_parents()
        xyz, parsed_parents = _skeleton(parsed.joints, parsed_parents)
    except Exception as exc:
        record.update(status='invalid_eos_sequence', error=repr(exc))
        return record
    # Set success only after decoding AND all metrics complete without error.
    topology = _topology_metrics(xyz, parsed_parents, truth, parents)
    distances = np.linalg.norm(xyz[:, None]-truth[None], axis=-1)
    j2j = float(.5*(distances.min(0).mean()+distances.min(1).mean())/diagonal)
    if not math.isfinite(j2j) or not math.isfinite(topology['edge_f1']):
        raise ValueError('nonfinite generation metric')
    record.update(success=True, original_edge_f1=float(topology['edge_f1']),
        exact_count=len(xyz) == len(truth), predicted_joint_count=len(xyz),
        absolute_joint_count_error=abs(len(xyz)-len(truth)), original_normalized_j2j=j2j,
        predicted_joints=xyz.tolist(), predicted_parents=parsed_parents)
    return record


def summarize_rows(rows):
    """Equal-weight assets; report the denominator for success-only geometry."""
    if not rows:
        raise ValueError('empty generation evaluation')
    keys = [(r['asset_id'], r['raw_frame_index']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate asset/raw-frame rows')
    assets = defaultdict(list)
    for row in rows:
        if not row['success'] and (row['original_edge_f1'] != 0 or row['exact_count']
                or row['original_normalized_j2j'] is not None):
            raise ValueError('failed generation has nonzero quality or a geometry score')
        if row['success'] and (row['original_normalized_j2j'] is None
                or not math.isfinite(row['original_normalized_j2j'])
                or row['original_normalized_j2j'] < 0):
            raise ValueError('successful generation lacks valid geometry')
        if not math.isfinite(row['original_edge_f1']) or not 0 <= row['original_edge_f1'] <= 1:
            raise ValueError('invalid topology score')
        assets[row['asset_id']].append(row)
    result = {metric: statistics.mean(statistics.mean(r[metric] for r in group)
              for group in assets.values()) for metric in ('success', 'original_edge_f1', 'exact_count')}
    geometry = [[r['original_normalized_j2j'] for r in group if r['success']] for group in assets.values()]
    values = [statistics.mean(group) for group in geometry if group]
    result.update(assets=len(assets), views=len(rows), failures=sum(not r['success'] for r in rows),
        original_normalized_j2j_successful_views=statistics.mean(values) if values else None,
        geometry_scored_assets=len(values), geometry_scored_views=sum(map(len, geometry)),
        geometry_scope='Original-GT symmetric nearest-joint distance / target mesh bbox diagonal; '
            'successful views only, averaged within asset then across assets with successes; no alignment.',
        quality_scope='All views including failed generation, averaged within asset then equally across assets.')
    return result
