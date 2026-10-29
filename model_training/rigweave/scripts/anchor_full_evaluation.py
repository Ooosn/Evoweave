"""Fixed-input FP32 validation of both capabilities during full-data training."""
import gc
import hashlib
import json
from pathlib import Path
import tarfile

import torch
import torch.distributed as dist

from anchor_full_data import CachedFiles
from anchor_full_protocol import scientific_sha, sha256
from anchor_joint_runtime import joint_metrics
from reference_flow_protocol import sample_seed
from reference_flow_runtime import mirror, summarize
from stage1_feedback import generate_asset, score_generation, summarize_rows
from stage1_surface_repair_runtime import canonical_identity
from fusion_evaluation import tree_fingerprint
from virtual_pair_runtime import evaluation_rows, gather, shared_deadline_due
from virtual_query_xyz_runtime import build_model, fixed_dataset, write_json
from rigweave.dynamic_rig.reference_skeleton_xyz import reference_slots
from rigweave.dynamic_rig.virtual_frame_data import prepare_virtual_surface
from rigweave.dynamic_rig.virtual_path_ablation import VirtualPathAblationRig


def canonical_input(dataset, index, base, config, mode, device):
    row = dataset.entries[index]
    identity = {'asset_id': row['asset_id'], 'path': row['path'], 'root_rt': row['root_rt'],
        'data': base['data'], 'surface_sampling': base['surface_sampling'], 'mode': mode, 'frames': 8}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = Path(config['data_cache']['root'])/'evaluation_inputs'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/(key+'.pt')
    sidecar = path.with_suffix('.json')
    if path.exists():
        spec = json.loads(sidecar.read_text())
        if spec['identity'] != identity or spec['path'] != str(path) or path.is_symlink() or sha256(path) != spec['sha256']:
            raise RuntimeError('immutable evaluation input changed')
        record = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    else:
        batch = dataset[index]
        surface, geometry = prepare_virtual_surface([batch], device, **base['surface_sampling'])
        count = int(geometry['joint_mask'][0].sum())
        vertices = batch['frame_vertices'][0]
        record = {'asset_id': batch['asset_id'], 'surface': {k: v.detach().cpu() for k, v in surface.items()},
            'truth': geometry['frame_target_joints'][0, :, :count].cpu(),
            'parents': geometry['target_parents'][0, :count].cpu(),
            'raw_frames': geometry['selected_frames'][0].tolist(),
            'diagonals': torch.linalg.vector_norm(vertices.amax(1)-vertices.amin(1), dim=-1).cpu(),
            'normalization_sha256': tree_fingerprint({k: geometry[k] for k in ('query_center', 'query_scale')}),
            'surface_sha256': tree_fingerprint(surface), 'heldout_raw_ids': batch['heldout_raw_ids'],
            'frame_partition': mode}
        raw, heldout = set(record['raw_frames']), set(record['heldout_raw_ids'])
        if (mode == 'heldout_eval' and not raw <= heldout) or (mode != 'heldout_eval' and raw & heldout):
            raise RuntimeError('canonical evaluation frame partition leaked')
        temporary = path.with_suffix('.pt.tmp')
        torch.save(record, temporary)
        temporary.replace(path)
        path.chmod(0o444)
        spec = {'path': str(path), 'sha256': sha256(path), 'identity': identity}
        write_json(sidecar, spec)
    if tree_fingerprint(record['surface']) != record['surface_sha256']:
        raise RuntimeError('canonical evaluation surface tensor digest differs')
    return record, spec


def archive_inputs(specs, output, config, *, final):
    label = 'full' if final else 'panel'
    receipt = output/f'canonical_{output.name}_{label}_archive.json'
    if receipt.exists():
        return json.loads(receipt.read_text())
    unique = {spec['path']: spec for spec in specs}
    archive = output/f'canonical_{output.name}_{label}{len(unique)}.tar'
    temporary = archive.with_suffix('.tar.tmp')
    with tarfile.open(temporary, 'w') as handle:
        for path in sorted(unique):
            item = Path(path)
            handle.add(item, arcname='inputs/'+item.name, recursive=False)
            handle.add(item.with_suffix('.json'), arcname='inputs/'+item.with_suffix('.json').name, recursive=False)
    with tarfile.open(temporary, 'r') as handle:
        for path, spec in unique.items():
            stream = handle.extractfile('inputs/'+Path(path).name)
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(8*1024**2), b''):
                digest.update(block)
            if digest.hexdigest() != spec['sha256']:
                raise RuntimeError('canonical input archive verification failed')
    temporary.replace(archive)
    value = {**mirror(archive, archive.name, config), 'canonical_assets': len(unique),
        'all_input_members_verified': True}
    write_json(receipt, value)
    mirror(receipt, receipt.name, config)
    return value


def eager_backbone(model, base, device):
    original, tokenizer, _, _ = build_model(base, torch.device('cpu'), load_official_weights=False)
    backbone = VirtualPathAblationRig(original, virtual_enabled=False)
    if backbone.xyz_decoder.config._attn_implementation != 'eager':
        raise RuntimeError('FP32 validation requires eager decoder attention')
    backbone.load_state_dict({name: value.detach().cpu() for name, value in model.backbone.state_dict().items()}, strict=True)
    return backbone.to(device).eval(), tokenizer


@torch.inference_mode()
def evaluate(model, base, config, entries, output, samples, device, rank, world,
             *, final=False, include_ar=False):
    if shared_deadline_due(config, device, world):
        return {'complete': False, 'stopped_for_deadline': True}
    tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    was_training = model.training
    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    backbone, tokenizer = eager_backbone(model, base, device)
    resolver = CachedFiles(config, base)
    panel = evaluation_rows(base, entries)
    panel_ids = {row['asset_id'] for row in panel}
    directory = output / (f'final_{samples:06d}' if final else f'evaluation_{samples:06d}')
    if rank == 0:
        directory.mkdir(parents=True, exist_ok=False)
    dist.barrier(device_ids=[device.index])
    selected = entries['valid'] if final else panel
    splits = [('valid', selected, 'seen_eval')]
    if final:
        train = sorted(entries['train'], key=lambda r: sample_seed(config['evaluation']['seed'], r['asset_id'], 'train-eval'))[:32]
        splits += [('train_eligible', train, 'seen_eval'), ('train_heldout', train, 'heldout_eval')]
    rows, ar_rows, examples, input_specs = [], [], [], []
    stopped = False
    try:
        for split, selected, mode in splits:
            dataset = fixed_dataset(selected, base, 8, mode=mode, file_resolver=resolver)
            # Pair iterations even for the odd-sized full validation split.
            for offset in range((len(selected) + world - 1) // world):
                if shared_deadline_due(config, device, world):
                    stopped = True
                    break
                index = offset * world + rank
                if index >= len(selected):
                    continue
                record, input_spec = canonical_input(dataset, index, base, config, mode, device)
                input_specs.append(input_spec)
                surface = {k: v.to(device) for k, v in record['surface'].items()}
                features, _ = backbone.encode(surface)
                target, parents = record['truth'].to(device), record['parents'].to(device)
                mask = torch.ones((1, len(parents)), dtype=torch.bool, device=device)
                raw, asset = record['raw_frames'], record['asset_id']
                for ref in config['evaluation']['reference_slots']:
                    reference = torch.tensor([ref], device=device)
                    prediction = model.head(features, surface['query_points'], surface['query_normals'],
                        target[ref:ref+1], parents[None], mask, reference)[0]
                    if not torch.isfinite(prediction).all() or not torch.equal(prediction[ref], target[ref].float()):
                        raise RuntimeError('reference skeleton changed during XYZ inference')
                    for method, values in [('head', prediction), ('copy', target[ref:ref+1].expand_as(target))]:
                        for frame in range(8):
                            if frame == ref:
                                continue
                            rows.append({'split': split, 'asset_id': asset,
                                'selection_panel': asset in panel_ids,
                                'reference_kind': 'GT', 'reference_slot': ref, 'reference_raw_frame': raw[ref],
                                'target_slot': frame, 'target_raw_frame': raw[frame], 'method': method,
                                'reference_success': True, 'joint_count': len(parents),
                                'surface_sha256': record['surface_sha256'],
                                **joint_metrics(values[frame], target[frame], parents.tolist(), float(record['diagonals'][frame]))})
                    if split == 'valid' and len(examples) < 4:
                        examples.append({'asset_id': asset, 'reference_slot': ref,
                            'prediction': prediction.cpu(), 'truth': target.cpu(), 'parents': parents.cpu(),
                            'points': surface['query_points'][0].cpu(), 'raw_frames': raw})
                if include_ar and split == 'valid' and asset in panel_ids:
                    generated = generate_asset(backbone, surface, tokenizer,
                        frame_indices=config['evaluation']['ar_frame_indices'],
                        max_new_tokens=config['evaluation']['max_new_tokens'], precision='fp32')
                    for frame, prediction in generated.items():
                        ar_rows.append({**canonical_identity(record, frame),
                            **score_generation(prediction, tokenizer, record['truth'][frame].numpy(),
                                record['parents'].tolist(), float(record['diagonals'][frame]))})
                del surface, record, features, target, parents, mask, prediction
                if offset % 16 == 0:
                    write_json(directory/f'progress_rank_{rank}.json', {'split': split,
                        'processed_local': offset+1, 'xyz_rows': len(rows), 'AR_rows': len(ar_rows)})
            if stopped:
                break
        all_rows = sum(gather(rows, world), [])
        all_ar = sum(gather(ar_rows, world), [])
        all_inputs = sum(gather(input_specs, world), [])
        expected_assets = 967 if final else 128
        head_rows = [row for row in all_rows if row['split'] == 'valid' and row['method'] == 'head']
        if not stopped and (len(head_rows) != expected_assets*14
                or len({r['asset_id'] for r in head_rows}) != expected_assets
                or (include_ar and len(all_ar) != 128*8)):
            raise RuntimeError('full-run validation coverage differs')
        summary = {'complete': not stopped, 'stopped_for_deadline': stopped,
            'samples': samples, 'final': final, 'precision': 'fp32_noTF32', 'decoder_attention': 'eager',
            'scientific_sha256': scientific_sha(config), 'groups': summarize(all_rows),
            'stage1': summarize_rows(all_ar) if all_ar else None,
            'stage1_decoding': 'greedy_fast_validation_not_official_beam_inference',
            'stage1_GT_reference_or_teacher_prefix': False,
            'stage2_reference': 'GT_only', 'data_cache_index_sha256': resolver.index_sha256}
        path = directory/f'examples_rank_{rank}.pt'
        torch.save(examples, path)
        mirror(path, directory.name+'/'+path.name, config)
        if rank == 0:
            if not stopped:
                summary['canonical_input_archive'] = archive_inputs(all_inputs, output, config, final=final)
            for name, value in [('rows.json', all_rows), ('stage1_rows.json', all_ar), ('summary.json', summary)]:
                path = directory/name
                write_json(path, value)
                mirror(path, directory.name+'/'+name, config)
        dist.barrier(device_ids=[device.index])
        return summary
    finally:
        del backbone, tokenizer
        model.train(was_training)
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
        gc.collect()
        torch.cuda.empty_cache()
