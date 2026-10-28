"""Bounded diagnostics of inputs, repeated inference and checkpoint reconstruction."""
import hashlib
import time

import torch
import torch.distributed as dist

from anchor_replay_protocol import validate_replay


def tensor_digest(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str((tuple(value.shape), value.dtype)).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def difference(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError('replay tensor shape/dtype differs')
    delta = (left.double()-right.double()).abs()
    return {'bitwise_equal': tensor_digest(left) == tensor_digest(right),
            'max_abs': float(delta.max()), 'mean_abs': float(delta.mean())}


@torch.no_grad()
def run(config, output, source, device, rank, world):
    from anchor_integration_protocol import regime_config, scientific_sha, sha256
    from anchor_joint_runtime import head_digest, load_model, joint_metrics
    from reference_flow_runtime import gather, read_all
    from virtual_pair_runtime import shared_deadline_due
    from virtual_query_xyz_runtime import fixed_dataset, write_json
    from rigweave.dynamic_rig.virtual_frame_data import prepare_virtual_surface
    from anchor_integration_runtime import release

    spec = validate_replay(config, 'capacity')
    started = time.monotonic()
    def deadline():
        if shared_deadline_due(config, device, world) or time.monotonic()-started > spec['max_seconds']:
            raise RuntimeError('bounded replay deadline reached')
    entry = spec['examples'][rank]
    if sha256(entry['path']) != entry['sha256']:
        raise ValueError('retained example file changed')
    examples = torch.load(entry['path'], map_location='cpu', weights_only=True)
    expected = [r for r in examples if r['asset_id'] == spec['case_ids'][rank] and r['reference_slot'] == 0]
    if len(expected) != 1:
        raise ValueError('fixed case missing or duplicated')
    expected = expected[0]
    run_config = regime_config(config, spec['regime'])
    base, entries = read_all(run_config)
    selected = [r for r in entries['valid'] if r['asset_id'] == spec['case_ids'][rank]]
    if len(selected) != 1:
        raise ValueError('fixed validation asset missing or duplicated')
    dataset = fixed_dataset(selected, base, 8, mode='seen_eval')
    records, predictions = [], {}
    for load in range(spec['model_loads']):
        deadline()
        model, optimizer, _, _, _, _, _, samples, frames, step = load_model(
            run_config, device, rank, world, spec['checkpoint'])
        if (samples, frames, step) != (80000, 1043152, 1680):
            raise ValueError('checkpoint counters differ')
        # Optimizer restoration matches the production loader, but no step is permitted.
        def forbidden_step(*args, **kwargs):
            raise RuntimeError('optimizer update forbidden in replay')
        optimizer.step = forbidden_step
        model.eval()
        initial_head = head_digest(model.head)
        initial_backbone = head_digest(model.backbone)
        for repeat in range(spec['repeats_per_load']):
            deadline()
            batch = dataset[0]
            surface, geometry = prepare_virtual_surface([batch], device, **base['surface_sampling'])
            if geometry['selected_frames'][0].tolist() != expected['raw_frames']:
                raise ValueError('fixed-case raw frames differ')
            if not torch.equal(geometry['frame_target_joints'][0].cpu(), expected['truth']):
                raise ValueError('fixed-case original GT differs')
            if not torch.equal(geometry['target_parents'][0].cpu(), expected['parents']):
                raise ValueError('fixed-case parents differ')
            input_hashes = {key: tensor_digest(value) for key, value in surface.items()}
            query_comparison = difference(surface['query_points'][0].cpu(), expected['points'])
            vertices = batch['frame_vertices'][0]
            diagonal = torch.linalg.vector_norm(vertices.amax(1)-vertices.amin(1), dim=-1)
            parents, truth = geometry['target_parents'][0], geometry['frame_target_joints'][0]
            for precision in spec['precisions']:
                deadline()
                torch.cuda.reset_peak_memory_stats(device)
                rng = torch.cuda.get_rng_state(device).clone()
                tf32 = torch.backends.cuda.matmul.allow_tf32
                try:
                    if precision == 'fp32':
                        torch.backends.cuda.matmul.allow_tf32 = False
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision == 'bf16'):
                        features, _ = model.backbone.encode(surface)
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision == 'bf16'):
                        predicted = model.head(features, surface['query_points'], surface['query_normals'],
                            truth[0:1], parents[None], geometry['joint_mask'], torch.tensor([0], device=device))[0]
                finally:
                    torch.backends.cuda.matmul.allow_tf32 = tf32
                if not bool(torch.isfinite(predicted).all()):
                    raise ValueError('nonfinite replay prediction')
                predicted = predicted.cpu()
                key = f'{load}:{repeat}:{precision}'
                predictions[key] = predicted
                records.append({'load': load, 'repeat': repeat, 'precision': precision,
                    'input_sha256': input_hashes, 'feature_sha256': tensor_digest(features),
                    'prediction_sha256': tensor_digest(predicted),
                    'head_sha256': initial_head, 'backbone_sha256': initial_backbone,
                    'CUDA_RNG_unchanged': torch.equal(rng, torch.cuda.get_rng_state(device)),
                    'matmul_TF32': tf32 if precision == 'bf16' else False,
                    'query_points_vs_retained': query_comparison,
                    'prediction_vs_retained': difference(predicted, expected['prediction']),
                    'target7_metrics': joint_metrics(predicted[7], truth[7].cpu(), parents.cpu().tolist(), float(diagonal[7])),
                    'max_reserved_bytes': torch.cuda.max_memory_reserved(device)})
                write_json(output/f'progress_rank_{rank}.json', {'optimizer_updates': 0, 'records': records})
                del features, predicted
            del batch, surface, geometry, vertices, truth, parents
        if initial_head != head_digest(model.head) or initial_backbone != head_digest(model.backbone):
            raise RuntimeError('inference mutated model state')
        del model, optimizer
        release()
        dist.barrier(device_ids=[device.index])
    torch.save({'asset_id': selected[0]['asset_id'], 'raw_frames': expected['raw_frames'],
                'predictions': predictions, 'retained_prediction': expected['prediction']},
               output/f'predictions_rank_{rank}.pt')
    local = {'rank': rank, 'asset_id': selected[0]['asset_id'], 'records': records,
        'repeat_same_load': {p: difference(predictions[f'0:0:{p}'], predictions[f'0:1:{p}']) for p in spec['precisions']},
        'reload_same_case': {p: difference(predictions[f'0:0:{p}'], predictions[f'1:0:{p}']) for p in spec['precisions']},
        'optimizer_updates': 0, 'seconds': time.monotonic()-started}
    ranks = gather(local, world)
    return {'complete': True, 'passed': True, 'profile': spec['profile'], 'source': source,
        'scientific_sha256': scientific_sha(config), 'ranks': ranks,
        'optimizer_updates': 0, 'not_model_quality': True,
        'acceptance_scope': 'Diagnostic execution only; interpret input and precision/reload differences separately.'}
