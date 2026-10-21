"""Explicit initialization, matched data and sampling for XYZ token-fill trials."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import subprocess

import numpy as np
import torch

from rigweave.dynamic_rig.aux_skeleton_tokens import AuxSkeletonTokens
from rigweave.dynamic_rig.frame_batch_sampler import FrameSampleRequest
from rigweave.dynamic_rig.structure_xyz_fill import (
    StructureXYZFillConfig, StructureXYZFillRig, initialize_fill_embeddings,
)
from rigweave.dynamic_rig.structure_xyz_fill_data import build_structure_fill_targets, decode_coordinate_tokens
from rigweave.dynamic_rig.virtual_frame_data import prepare_virtual_surface, collate_virtual_targets
from rigweave.dynamic_rig.virtual_query_xyz_data import build_xyz_aux_targets
from fusion_evaluation import tree_fingerprint
from probe_virtual_query_lambda import parameter_fingerprint
from virtual_fusion_normal_runtime import MatchedFrameStream
from virtual_query_xyz_runtime import build_model as build_old_model, fixed_dataset, read_entries, sha256


TRAIN5000_SHA = 'a17b6be96fb33a32a1bee7117b07789da6261c6b5693104f3d2cce07c09bab88'
VALID_SHA = 'fd2be7b5885833946e68dc186300a820757ae652c487749eec1b049d381e230a'
FAMILY = 'structure_xyz_token_fill_v1'
SOURCE_ROOT = Path(__file__).resolve().parents[3]


def data_source_hashes():
    names = ('structure_xyz_fill_data.py', 'leaf_pruning.py', 'aux_skeleton_tokens.py',
             'virtual_query_xyz_data.py', 'virtual_frame_data.py', 'shared_structure_medium_data.py',
             'root_rt.py', 'data.py', 'sampling.py')
    return {name: sha256(SOURCE_ROOT / 'model_training/rigweave/src/rigweave/dynamic_rig' / name) for name in names}


def model_source_hashes():
    paths = [f'model_training/rigweave/src/rigweave/dynamic_rig/{name}' for name in
             ('structure_xyz_fill.py', 'virtual_query_xyz.py', 'motion_encoder.py',
              'surface_tokenizer.py', 'motion_evidence.py', 'skeleton_topology_metrics.py')]
    paths += [f'model_training/rigweave/scripts/{name}' for name in
              ('structure_xyz_fill_runtime.py', 'train_structure_xyz_fill.py', 'structure_xyz_fill_evaluation.py')]
    return {'data': data_source_hashes(), 'model': {path: sha256(SOURCE_ROOT / path) for path in paths}}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def validate_config(config):
    if config.get('purpose') != FAMILY:
        raise ValueError('configuration is not an explicit structure/XYZ-fill experiment')
    data = config['data']
    expected = {'train': (5000, TRAIN5000_SHA), 'valid': (967, VALID_SHA)}
    for split, (count, digest) in expected.items():
        spec = data['manifests'][split]
        if spec['rows'] != count or spec['sha256'] != digest:
            raise ValueError('new trials must retain the exact accepted 5k/validation membership')
    if (data['normalization'] != 'input_sequence_mesh_bbox' or data['global_rt_removal'] is not True
            or data['coordinate_clipping'] is not False or data['heldout_frames'] != 8
            or (data['frames_min'], data['frames_max']) != (2, 24)):
        raise ValueError('normalization, root RT or frame partition differs from accepted data')
    if config['quantization'] != {'bins': 256, 'continuous_range': [-1., 1.]}:
        raise ValueError('coordinate tokens must retain the exact original 256-bin chart')
    fill = config['fill_architecture']
    if (fill['fill_depth'], fill['fill_max_tokens'], fill['fill_ffn_dim'], fill['coordinate_bins']) != (4, 3076, 4096, 256):
        raise ValueError('first-version fill architecture differs from the agreed baseline')
    if (config['architecture']['dim'], config['architecture']['heads'], config['architecture']['query_tokens'],
            config['architecture']['motion_depth'], config['architecture']['auxiliary_depth']) != (1024, 8, 1024, 12, 2):
        raise ValueError('shared upstream or structure decoder differs from the declared architecture')
    if config['pruning']['batch_probability'] != .2 or config['pruning']['max_fraction'] != .2:
        raise ValueError('first-version pruning must use an 80/20 gate and 0..20% leaf deletion')
    if config['fill_initialization']['kind'] not in {'random', 'structure_and_position'}:
        raise ValueError('unknown selective fill initialization')
    if type(config['fill_initialization']['seed']) is not int:
        raise ValueError('fill initialization requires an integer seed')
    evaluation = config['evaluation']
    if (evaluation['input_frames'] != 8 or evaluation['frame_indices'] != [0, 2, 4, 6]
            or evaluation['train_assets'] != 8 or evaluation['valid_assets'] != 16):
        raise ValueError('early evaluation must use the fixed small matched panel')
    for key in ('shared_lr', 'structure_lr', 'fill_lr', 'structure_weight', 'clip_grad_norm'):
        value = config['optimization'][key]
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'invalid optimization field {key}')
    optimization = config['optimization']
    if (len(optimization['betas']) != 2 or any(not np.isfinite(x) or not 0 < x < 1 for x in optimization['betas'])
            or not np.isfinite(optimization['weight_decay']) or optimization['weight_decay'] < 0
            or type(optimization['warmup_assets']) is not int or optimization['warmup_assets'] < 0
            or not 0 < optimization['warmup_start_factor'] <= 1):
        raise ValueError('invalid AdamW or warmup configuration')
    if config['optimization']['schedule'] != 'linear_warmup_then_constant':
        raise ValueError('short trials must not silently change their learning-rate horizon')
    plan = config['training']
    if plan['effective_assets_per_update'] != 48:
        raise ValueError('comparisons retain 48 actual assets per optimizer update')
    milestones = plan['evaluation_assets']
    if (not milestones or milestones != sorted(set(milestones))
            or any(type(x) is not int or x <= 0 for x in milestones)
            or milestones[-1] != plan['target_assets'] or plan['checkpoint_assets'] != milestones):
        raise ValueError('explicit ordered evaluation/checkpoint milestones must end at the budget')
    world = config['runtime']['world_size']
    if world not in (1, 2) or len(config['runtime']['physical_gpus']) != world:
        raise ValueError('bounded execution needs one or two explicitly granted devices')
    make_stream(config, list(range(5000)), world=world)
    return milestones


def load_tokenizer(config):
    from train_dynamic_rig import build_tokenizer
    runtime = config['runtime']
    if (subprocess.check_output(['git', '-C', runtime['unirig_root'], 'rev-parse', 'HEAD'], text=True).strip()
            != config['initialization']['upstream_source_commit'] or
            subprocess.check_output(['git', '-C', runtime['unirig_root'], 'status', '--porcelain'], text=True).strip()):
        raise ValueError('coordinate tokenizer source is not the pinned clean vendor revision')
    if sha256(runtime['tokenizer_config']) != runtime['tokenizer_config_sha256']:
        raise ValueError('the pinned coordinate tokenizer configuration changed')
    tokenizer = build_tokenizer(Path(runtime['tokenizer_config']))
    if tokenizer.num_discrete != 256 or tuple(tokenizer.continuous_range) != (-1., 1.):
        raise ValueError('actual tokenizer differs from the declared coordinate convention')
    return tokenizer


def build_fill_model(config, device):
    """Warm-start only declared common modules; construct an independent new XYZ backend."""
    warm = config['common_initialization']
    if warm['kind'] != 'accepted_b65_shared_and_structure':
        raise ValueError('first trials require their declared common B65 starting point')
    if sha256(warm['checkpoint']) != warm['sha256']:
        raise ValueError('accepted common B65 checkpoint fingerprint changed')
    base, tokenizer, codec, official = build_old_model(config, torch.device('cpu'))
    payload = torch.load(warm['checkpoint'], map_location='cpu', mmap=True, weights_only=False)
    if (payload['schema'] != 'virtual_query_xyz_aux_ids_v1' or payload['asset_exposures'] != 65000
            or payload['architecture'] != asdict(base.config)):
        raise ValueError('common checkpoint does not match the declared B65 architecture and exposure')
    base.load_state_dict(payload['model'], strict=True)
    inherited = {'asset_exposures': payload['asset_exposures'], 'updates': payload['step'],
                 'supervised_frame_exposures': payload['supervised_frame_exposures'], 'source': payload['source']}
    del payload
    position = base.xyz_decoder.model.decoder.embed_positions
    # OPT has an internal row offset; its former mesh prefix occupied real positions too.
    text_position_offset = int(position.offset) + base.config.query_tokens
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config['fill_initialization']['seed'])
        model = StructureXYZFillRig(
            base.surface_tokenizer, StructureXYZFillConfig(upstream=base.config, **config['fill_architecture']),
            codec, motion_encoder=base.motion_encoder, structure_decoder=base.auxiliary,
        )
    kind = config['fill_initialization']['kind']
    copied = initialize_fill_embeddings(
        model, structure_embedding=base.auxiliary.embedding.weight if kind == 'structure_and_position' else None,
        position_embedding=position.weight if kind == 'structure_and_position' else None,
        position_row_offset=text_position_offset if kind == 'structure_and_position' else 0,
        source_name=f'B65_{kind}',
    )
    report = {'common_checkpoint': warm, 'inherited_training': inherited,
              'optimizer_restored': False, 'new_training_counters_start_at_zero': True,
              'not_a_from_scratch_comparison': True, 'fill_initialization': copied,
              'fill_seed': config['fill_initialization']['seed'], 'official_loading': official,
              'architecture': model.configuration(),
              'common_parameters_sha256': parameter_fingerprint(model, exclude_parts=('fill_decoder.',)),
              'fresh_fill_attention_ffn_head_sha256': parameter_fingerprint(
                  model.fill_decoder, exclude_parts=('embedding.', 'position.')),
              'full_initial_state_sha256': parameter_fingerprint(model),
              'parameter_groups': {name: sum(p.numel() for p in getattr(model, name).parameters())
                                   for name in ('surface_tokenizer', 'motion_encoder', 'structure_decoder', 'fill_decoder')}}
    del base, position
    model.to(device)
    return model, tokenizer, codec, report


def make_stream(config, dataset, *, rank=0, world=1, state=None, extend=False):
    plan, data = config['training'], config['data']
    if extend:
        if state is None:
            raise ValueError('stream extension requires a completed saved stream')
        return MatchedFrameStream.extend_completed(dataset, state, target_assets=plan['target_assets'],
                                                   update_boundaries=plan['evaluation_assets'])
    return MatchedFrameStream(
        dataset, rank=rank, world_size=world, seed=data['stream_seed'], target_assets=plan['target_assets'],
        effective_assets=plan['effective_assets_per_update'], frame_policy=plan['frame_policy'],
        manifest_sha256=data['manifests']['train']['sha256'], min_frames=data['frames_min'],
        max_frames=data['frames_max'], state=state, update_boundaries=plan['evaluation_assets'],
    )


def restore_sampling_state(config, dataset, stream_state, *, asset_exposures,
                           supervised_frame_exposures, step, rank=0, world=1):
    for name, value in (('asset_exposures', asset_exposures), ('supervised_frame_exposures', supervised_frame_exposures),
                        ('step', step)):
        if type(value) is not int or value < 1:
            raise ValueError(f'saved {name} must be a positive integer')
    old_budget = stream_state['contract']['target_assets']
    target = config['training']['target_assets']
    if target < old_budget:
        raise ValueError('a resume may not shrink or silently replace the existing exposure plan')
    stream = make_stream(config, dataset, rank=rank, world=world, state=stream_state, extend=target > old_budget)
    if stream.rank != rank or stream.world_size != world or stream.cursor * world != asset_exposures:
        raise ValueError('saved stream cursor/rank disagrees with the asset exposure counter')
    if step > len(stream.global_ends) or int(stream.global_ends[step - 1]) != asset_exposures:
        raise ValueError('saved optimizer update is not the exact exposure-plan boundary')
    expected_frames = int(np.dot(stream.update_sizes[:step].astype(np.int64), stream.frame_counts[:step]))
    if expected_frames != supervised_frame_exposures:
        raise ValueError('saved supervised-frame counter differs from the exact exposure plan')
    return stream


def pruning_choices(joint_counts, requests, config, update_index, *, enabled):
    counts = [int(value) for value in joint_counts]
    if len(counts) != len(requests) or any(value < 1 for value in counts):
        raise ValueError('pruning counts and asset visits do not align')
    policy = config['pruning']
    if type(update_index) is not int or update_index < 0:
        raise ValueError('pruning requires a nonnegative global update index')
    gate = bool(enabled and np.random.default_rng(np.random.SeedSequence(
        [policy['seed'], update_index, 813])).random() < policy['batch_probability'])
    remove_counts, seeds, fractions = [], [], []
    for count, request in zip(counts, requests):
        if not gate:
            remove_counts.append(0)
            seeds.append(None)
            fractions.append(0.)
            continue
        if not isinstance(request, FrameSampleRequest):
            raise ValueError('training pruning needs explicit, reproducible asset visit seeds')
        random = np.random.default_rng(np.random.SeedSequence([
            policy['seed'], request.sampling_seed & 0xffffffff, request.sampling_seed >> 32, 991]))
        fraction = float(random.uniform(0., policy['max_fraction']))
        fractions.append(fraction)
        remove_counts.append(int(np.floor(fraction * (count - 1))))
        seeds.append(int(random.integers(0, 2**63 - 1)))
    return {'batch_gate': gate, 'remove_counts': remove_counts, 'prune_seeds': seeds, 'fractions': fractions}


def prepare_fill_batch(dataset, requests, config, codec, device, *, update_index=0, pruning=False):
    batches = [dataset[request] for request in requests]
    surface, geometry = prepare_virtual_surface(batches, device, **config['surface_sampling'])
    decision = pruning_choices(geometry['joint_count'].cpu().tolist(), requests, config, update_index, enabled=pruning)
    target = build_structure_fill_targets(
        geometry, codec, device=device, max_fill_tokens=config['fill_architecture']['fill_max_tokens'],
        **config['quantization'], remove_counts=decision['remove_counts'], prune_seeds=decision['prune_seeds'],
    )
    return surface, target, geometry, decision


def verify_data_selection(config, entries):
    spec = config['normal_match']['selection']
    if sha256(spec['path']) != spec['sha256']:
        raise ValueError('the unchanged 5k subset selection receipt changed')
    receipt = json.loads(Path(spec['path']).read_text())
    if (receipt['status'] != 'passed' or receipt['derived_manifest'] != config['data']['manifests']['train']
            or not receipt['rows_unchanged']['full_json_rows_preserved']
            or not receipt['rows_unchanged']['root_rt_preserved_for_selected_rows']
            or receipt['actual_frame_count']['minimum_actual_frame_count'] < 32):
        raise ValueError('training subset lacks its accepted identity/RT/frame certificate')
    for field in ('asset_id', 'path'):
        train, valid = ({row[field] for row in entries[key]} for key in ('train', 'valid'))
        if len(train) != 5000 or len(valid) != 967 or train & valid:
            raise ValueError('training and validation membership overlaps or duplicates')


def load_small_reference_panel(config, entries):
    reference, evaluation = config['reference_evaluation'], config['evaluation']
    for spec in reference['files'].values():
        if sha256(spec['path']) != spec['sha256']:
            raise ValueError('accepted B65 evaluation reference changed')
    selection = json.loads(Path(reference['files']['selection']['path']).read_text())
    count_by_group = {'train_seen': 32, 'train_heldout': 32, 'valid': 128}
    if any(len(selection[group]) != count for group, count in count_by_group.items()):
        raise ValueError('reference is not the original accepted 32/32/128 panel')
    lookup = {split: {row['asset_id']: row for row in values} for split, values in entries.items()}
    panel = {}
    for number, (group, size) in enumerate((('train_seen', evaluation['train_assets']), ('valid', evaluation['valid_assets']))):
        source = selection[group]
        chosen = sorted(np.random.default_rng(np.random.SeedSequence([evaluation['seed'], number])).permutation(len(source))[:size])
        split = 'train' if group == 'train_seen' else 'valid'
        panel[group] = []
        for index in chosen:
            row = source[int(index)]
            actual = lookup[split][row['asset_id']]
            if actual['path'] != row['path'] or actual['root_rt'] != row['root_rt']:
                raise ValueError('evaluation asset differs from the immutable manifest row')
            panel[group].append(actual)
    panel['train_heldout'] = panel['train_seen']
    records = {}
    for key in ('rank_0', 'rank_1'):
        payload = json.loads(Path(reference['files'][key]['path']).read_text())
        if not payload['complete'] or payload['samples'] != 65000:
            raise ValueError('reference evaluation is incomplete or from a different checkpoint')
        for group in count_by_group:
            for row in payload['groups'][group]['rows']:
                identity = (group, row['asset_id'], row['frame_index'])
                if identity in records:
                    raise ValueError('reference evaluation contains duplicated views')
                records[identity] = row
    if len(records) != 1536:
        raise ValueError('accepted reference does not cover all original views')
    subset = {}
    for group, rows in panel.items():
        for row in rows:
            for frame in evaluation['frame_indices']:
                key = (group, row['asset_id'], frame)
                if key not in records:
                    raise ValueError('reference lacks a requested small-panel view')
                subset[key] = records[key]
    manifest = {'groups': {group: [{'asset_id': row['asset_id'], 'path': row['path']} for row in rows]
                           for group, rows in panel.items()},
                'input_frames': evaluation['input_frames'], 'scored_frame_indices': evaluation['frame_indices'],
                'seed': evaluation['seed'], 'reference_checkpoint_assets': 65000,
                'reference_file_hashes': {key: spec['sha256'] for key, spec in reference['files'].items()},
                'reference_original_edge_f1': {
                    group: float(np.mean([row['original_edge_f1'] for key, row in subset.items() if key[0] == group]))
                    for group in panel}}
    manifest['panel_sha256'] = tree_fingerprint(manifest)
    return panel, subset, manifest


def audit_new_target_contract(config, entries, codec, tokenizer):
    """Audit the new static token interface, not rescreen or alter accepted data."""
    static, cache = {}, {}
    for split, rows in entries.items():
        checked = []
        for index, row in enumerate(rows):
            with np.load(row['path'], allow_pickle=False) as raw:
                parents = np.asarray(raw['target_parents'])
            if parents.ndim != 1 or not np.issubdtype(parents.dtype, np.signedinteger):
                raise ValueError(f'invalid source parent array for {row["asset_id"]}')
            identity = parents.astype('<i8', copy=False).tobytes()
            if identity not in cache:
                encoded = codec.encode(parents)
                inverse = encoded.source_to_dfs
                expected = np.array([-1 if parents[source] < 0 else inverse[parents[source]]
                                     for source in encoded.dfs_order], dtype=np.int64)
                if not np.array_equal(codec.decode(encoded.tokens), expected):
                    raise ValueError('new structure codec changed a source edge')
                fill_length = int(encoded.tokens.size + 3 * parents.size)
                if fill_length > config['fill_architecture']['fill_max_tokens']:
                    raise ValueError('actual expanded target exceeds fill position capacity')
                cache[identity] = {'joints': int(parents.size), 'structure_tokens': int(encoded.tokens.size),
                                   'fill_tokens': fill_length}
            checked.append({'asset_id': row['asset_id'], 'manifest_index': index,
                            'source_parents_sha256': hashlib.sha256(identity).hexdigest(), **cache[identity]})
        static[split] = checked
    longest = sorted(static['train'], key=lambda row: (-row['fill_tokens'], row['manifest_index']))[:3]
    largest_structure = sorted(static['train'], key=lambda row: (-row['structure_tokens'], row['manifest_index']))[:3]
    smallest = min(static['train'], key=lambda row: (row['joints'], row['manifest_index']))
    old_spec = config['capacity']['selection']
    if sha256(old_spec['path']) != old_spec['sha256']:
        raise ValueError('the accepted geometry stress selection changed')
    old_selection = json.loads(Path(old_spec['path']).read_text())
    if old_selection['manifest'] != config['data']['manifests']['train']:
        raise ValueError('geometry stress selection refers to another dataset')
    geometry_by_frames = old_selection['geometry_by_frames']
    groups = {'longest_fill': [row['manifest_index'] for row in longest],
              'longest_structure': [row['manifest_index'] for row in largest_structure],
              **{f'geometry_t{frames}': [geometry_by_frames[str(frames)]['manifest_row_index']]
                 for frames in range(2, 25)}}
    for frames in range(2, 25):
        metadata = geometry_by_frames[str(frames)]
        row = entries['train'][metadata['manifest_row_index']]
        if row['asset_id'] != metadata['asset_id'] or row['path'] != metadata['path']:
            raise ValueError('geometry stress index does not identify the declared original asset')
    indices = sorted(set(groups['longest_fill'] + groups['longest_structure'] +
                         [smallest['manifest_index']] + [groups[f'geometry_t{t}'][0] for t in (6, 12, 24)]))
    dataset = fixed_dataset(entries['train'], config, 2, mode='train')
    loader_checks = []
    for index in indices:
        for frames in (2, 24):
            request = FrameSampleRequest(index, frames, (config['data']['stream_seed'] << 32) + index + frames)
            raw = dataset[request]
            geometry = collate_virtual_targets([raw], 'cpu')
            target = build_structure_fill_targets(geometry, codec, **config['quantization'],
                                                  max_fill_tokens=config['fill_architecture']['fill_max_tokens'])
            original = build_xyz_aux_targets(geometry, tokenizer, codec, max_main_tokens=config['data']['max_main_tokens'])
            count = int(target['original_joint_counts'][0])
            source_order = target['fill_source_indices'][0, :count].numpy()
            for frame in range(frames):
                ids = original['main_input_ids'][0, frame][original['main_attention_mask'][0, frame]].numpy()
                parsed = tokenizer.detokenize(ids)
                decoded = decode_coordinate_tokens(target['xyz_labels'][0, frame, :count].numpy(), **config['quantization'])
                if (parsed.joints.shape != (count, 3) or
                        not np.array_equal(decoded, np.asarray(parsed.joints, dtype=np.float32)[source_order])):
                    raise ValueError('XYZ fill labels differ from the exact original coordinate token protocol')
            removed = int(np.floor(.2 * (count - 1)))
            pruned = build_structure_fill_targets(geometry, codec, **config['quantization'],
                max_fill_tokens=config['fill_architecture']['fill_max_tokens'], remove_counts=[removed], prune_seeds=[index + 4401])
            if (not torch.equal(target['structure_labels'], pruned['structure_labels']) or
                    int(pruned['fill_joint_counts'][0]) != count - removed):
                raise ValueError('XYZ pruning changed structure supervision or joint count')
            loader_checks.append({'asset_id': raw['asset_id'], 'manifest_index': index, 'frames': frames,
                                  'selected_raw_frames': geometry['selected_frames'][0].tolist(),
                                  'max_abs_coordinate': float(geometry['frame_target_joints'].abs().max()),
                                  'fill_tokens': int(target['fill_lengths'][0]), 'pruned_joints': removed,
                                  'official_coordinate_token_parity': True})
    return {'passed': True, 'membership_changed': False, 'data_rescreened': False,
            'new_static_target_contract': static, 'unique_source_parent_arrays': len(cache),
            'max_fill_tokens': {split: max(row['fill_tokens'] for row in rows) for split, rows in static.items()},
            'max_structure_tokens': {split: max(row['structure_tokens'] for row in rows) for split, rows in static.items()},
            'max_joints': {split: max(row['joints'] for row in rows) for split, rows in static.items()},
            'stress_groups': groups, 'real_loader_checks': loader_checks,
            'manifests': config['data']['manifests'], 'quantization': config['quantization'],
            'architecture': config['fill_architecture'], 'auxiliary_codec': config['auxiliary_codec'],
            'target_source_sha256': data_source_hashes()}


def optimizer_for(model, config):
    plan = config['optimization']
    groups = [
        {'params': list(model.surface_tokenizer.parameters()) + list(model.motion_encoder.parameters()),
         'name': 'shared', 'lr': plan['shared_lr'], 'peak_lr': plan['shared_lr']},
        {'params': list(model.structure_decoder.parameters()), 'name': 'structure',
         'lr': plan['structure_lr'], 'peak_lr': plan['structure_lr']},
        {'params': list(model.fill_decoder.parameters()), 'name': 'fill',
         'lr': plan['fill_lr'], 'peak_lr': plan['fill_lr']},
    ]
    identifiers = [id(parameter) for group in groups for parameter in group['params']]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('optimizer groups alias shared parameters')
    return torch.optim.AdamW(groups, betas=tuple(plan['betas']), weight_decay=plan['weight_decay'])


def set_learning_rate(optimizer, assets, config):
    plan = config['optimization']
    warmup = plan['warmup_assets']
    factor = (plan['warmup_start_factor'] + (1. - plan['warmup_start_factor']) * min(1., assets / warmup)
              if warmup > 0 else 1.)
    for group in optimizer.param_groups:
        group['lr'] = group['peak_lr'] * factor
    return {group['name']: group['lr'] for group in optimizer.param_groups}


def gradient_groups(model):
    result = {}
    for name in ('surface_tokenizer', 'motion_encoder', 'structure_decoder', 'fill_decoder'):
        grads = [parameter.grad.detach().float().norm() for parameter in getattr(model, name).parameters()
                 if parameter.requires_grad and parameter.grad is not None]
        value = float(torch.stack(grads).norm()) if grads else 0.
        if not np.isfinite(value) or value <= 0:
            raise RuntimeError(f'missing or nonfinite training signal in {name}')
        result[name] = value
    return result


def reset_random(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def matched_trial_fingerprint(config):
    contract = {key: config[key] for key in (
        'purpose', 'data', 'surface_sampling', 'auxiliary_codec', 'architecture', 'fill_architecture',
        'quantization', 'common_initialization', 'optimization', 'pruning', 'evaluation', 'reference_evaluation')}
    contract['fill_seed'] = config['fill_initialization']['seed']
    contract['budget'] = {key: config['training'][key] for key in (
        'target_assets', 'effective_assets_per_update', 'evaluation_assets', 'checkpoint_assets', 'frame_policy')}
    contract['world_size'] = config['runtime']['world_size']
    return tree_fingerprint(contract)


__all__ = ['FAMILY', 'validate_config', 'build_fill_model', 'make_stream', 'pruning_choices',
           'prepare_fill_batch', 'verify_data_selection', 'load_small_reference_panel', 'write_json',
           'optimizer_for', 'set_learning_rate', 'gradient_groups', 'load_tokenizer']
