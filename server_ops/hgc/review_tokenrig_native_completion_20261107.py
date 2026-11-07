"""Read-only model audit and completion receipts for the finished native run."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path('/work/wangyy/tokenrig_alignment_20261105')
OUTPUT = ROOT/'outputs_native_train'
TRAIN = OUTPUT/'train'
DURABLE = Path('/home/wangyy/evorig/outputs/tokenrig_alignment_20261105/native_full')
CODE = ROOT/'code_native_train'
SOURCE = '2f50732ee08e98d609c38bc2f73ec3657e5c7022'
CONFIG_SHA = 'e4fece4995c54dab87318bc13ab2280c47670fe1e920a371c7e63626fe159abb'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for data in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(data)
    return digest.hexdigest()


def balanced(rows, field):
    grouped = defaultdict(list)
    for row in rows:
        value = row[field]
        if not math.isfinite(value):
            raise RuntimeError('nonfinite validation metric')
        grouped[row['asset_id']].append(value)
    return sum(sum(v)/len(v) for v in grouped.values())/len(grouped)


def verify_groups(rows, summary):
    for group in summary['groups']:
        selected = [r for r in rows if r['split'] == group['split'] and r['method'] == group['method']]
        if (len(selected) != group['rows'] or len({r['asset_id'] for r in selected}) != group['assets']
                or any(r['reference_kind'] != 'GT' for r in selected)):
            raise RuntimeError('GT-reference validation coverage differs')
        for metric in ('normalized_mpjpe', 'edge_f1', 'normalized_j2j', 'pck_005'):
            if abs(balanced(selected, metric)-group[metric]) > 1e-12:
                raise RuntimeError('independent metric aggregation differs: '+metric)


def verify_copies(spec, *, fresh=False):
    result = {'sha256': spec['sha256'], 'bytes': spec['bytes'], 'fresh_SHA256': fresh}
    for key in ('path', 'durable_path'):
        path = Path(spec[key])
        if path.is_symlink() or not path.is_file() or path.stat().st_size != spec['bytes']:
            raise RuntimeError('required artifact copy is missing or changed: '+str(path))
        if fresh and sha(path) != spec['sha256']:
            raise RuntimeError('required artifact copy failed SHA256: '+str(path))
        result[key] = str(path)
    return result


def main():
    permitted = os.sched_getaffinity(0) & {6, 7, 9, 10, 11, 12, 14, 15, 61, 63, 64, 65, 66, 67, 68, 69}
    if not permitted:
        raise RuntimeError('no approved bounded CPU for completion review')
    os.sched_setaffinity(0, {min(permitted)})
    os.nice(19)
    review_path = OUTPUT/'primary_completion_review.json'
    if review_path.exists():
        raise RuntimeError('a primary completion review already exists; inspect, do not overwrite')
    if (subprocess.check_output(['git', '-C', str(CODE), 'rev-parse', 'HEAD'], text=True).strip() != SOURCE
            or subprocess.check_output(['git', '-C', str(CODE), 'status', '--porcelain'], text=True).strip()
            or sha(CODE/'model_training/experiments/tokenrig_native_full_20261105.json') != CONFIG_SHA):
        raise RuntimeError('deployed source/config changed during training')
    sys.path.insert(0, str(CODE/'model_training/rigweave/scripts'))
    from anchor_full_protocol import source_hashes
    audit = read(ROOT/'outputs_native_capacity/audit/result.json')
    config = read(CODE/'model_training/experiments/tokenrig_native_full_20261105.json')
    resources = read(OUTPUT/'train_resources.json')
    if (not resources['complete'] or not resources['passed'] or resources['exit_code'] != 0
            or resources['source'] != SOURCE or not resources['kernel_scheduler_membership']
            or not resources['payload_cleanup_complete']
            or any(v > 78 * 1024 for v in resources['peak_gpu_memory_mib'].values())):
        raise RuntimeError('native exit/resources/cleanup did not pass')
    results = [read(TRAIN/f'result_rank_{rank}.json') for rank in (0, 1)]
    traces = []
    for rank, result in enumerate(results):
        with (TRAIN/f'updates_rank_{rank}.jsonl').open() as handle:
            rows = [json.loads(line) for line in handle]
        actual = (sum(r['assets_in_update'] for r in rows), len(rows), sum(r['frames_in_update'] for r in rows))
        expected = (audit['plan']['assets'], audit['plan']['updates'], audit['plan']['input_frame_visits'])
        if (actual != expected or actual != (result['samples'], result['optimizer_updates'], result['input_frame_visits'])
                or actual[:2] != (300000, 6270) or not result['complete'] or not result['passed']
                or result['stopped_for_deadline'] or result['source_sha256'] != source_hashes()
                or any(not all(math.isfinite(r[k]) for k in
                    ('ar_loss', 'xyz_loss', 'bone_loss', 'gradient_norm')) for r in rows)):
            raise RuntimeError('training counts, source hashes or finite-update contract differs')
        traces.append({'rank': rank, 'visits': actual[0], 'updates': actual[1],
            'input_frame_visits': actual[2], 'all_logged_losses_and_gradients_finite': True})
    curve = []
    first_inputs = None
    identity = ('raw_frame_index', 'normalization_sha256', 'surface_sha256',
        'original_GT_sha256', 'target_mesh_diagonal', 'gt_joint_count')
    for n in range(10000, 300001, 10000):
        directory = TRAIN/f'evaluation_{n:06d}'
        summary = read(directory/'summary.json')
        if (not summary['complete'] or summary['samples'] != n or summary['precision'] != 'fp32_noTF32'
                or summary['decoder_attention'] != 'eager' or summary['stage2_reference'] != 'GT_only'):
            raise RuntimeError('a scheduled development validation did not complete')
        verify_groups(read(directory/'rows.json'), summary)
        ar = summary['stage1']
        if (ar is not None) != (n % 50000 == 0):
            raise RuntimeError('AR validation cadence differs')
        if ar is not None:
            rows = read(directory/'stage1_rows.json')
            inputs = {(r['asset_id'], r['frame_index']): tuple(r[k] for k in identity) for r in rows}
            if (len(inputs) != len(rows) or len(rows) != 1024 or len({r['asset_id'] for r in rows}) != 128
                    or any(not r['success'] and r['original_edge_f1'] != 0 for r in rows)
                    or abs(balanced(rows, 'original_edge_f1')-ar['original_edge_f1']) > 1e-12
                    or (first_inputs is not None and inputs != first_inputs)):
                raise RuntimeError('AR metrics or canonical input/GT identities differ')
            first_inputs = inputs
        head = next(g for g in summary['groups'] if g['split'] == 'valid' and g['method'] == 'head')
        curve.append({'visits': n, 'stage1_edge_F1': None if ar is None else ar['original_edge_f1'],
            'GT_reference_MPJPE': head['normalized_mpjpe'], 'GT_reference_edge_F1': head['edge_f1']})
    final = read(TRAIN/'final_300000/summary.json')
    verify_groups(read(TRAIN/'final_300000/rows.json'), final)
    full_head = next(g for g in final['groups'] if g['split'] == 'valid' and g['method'] == 'head')
    if not final['complete'] or (full_head['assets'], full_head['rows']) != (967, 13538):
        raise RuntimeError('required full967 GT-reference validation is incomplete')
    archives = []
    for spec, count in ((final['canonical_input_archive'], 1031),
            (read(TRAIN/'evaluation_300000/summary.json')['canonical_input_archive'], 128)):
        if spec['canonical_assets'] != count or not spec['all_input_members_verified']:
            raise RuntimeError('canonical input archive coverage was not verified')
        archives.append({**verify_copies(spec), 'canonical_assets': count, 'all_members_verified_by_native_saver': True})
    artifacts = []
    for n in (100000, 200000, 290000, 300000):
        spec = read(TRAIN/f'checkpoint_{n:06d}.json')
        if not spec['both_copies_verified'] or not spec['optimizer_and_rank_RNG_retained']:
            raise RuntimeError('full recovery lacks native save verification')
        artifacts.append({'kind': 'full_recovery', 'visits': n, **verify_copies(spec, fresh=n == 300000)})
        print('RECOVERY_VERIFIED', n, flush=True)
    for n in config['artifacts']['retention']['weight_milestones']:
        spec = read(TRAIN/f'weights_{n:06d}.json')
        if not spec['complete'] or not spec['model_only'] or not spec['tensor_roundtrip_verified']:
            raise RuntimeError('model-only export lacks its native tensor verification')
        artifacts.append({'kind': 'model_only', 'visits': n, **verify_copies(spec, fresh=n in (250000, 300000))})
        print('WEIGHTS_VERIFIED', n, flush=True)
    payloads = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if str(CODE) in command and any(x in command for x in
                ('launch_virtual_pair.py', 'torch.distributed.run', 'train_anchor_full.py')):
            payloads.append(int(entry.name))
    if payloads:
        raise RuntimeError('owned training payload still exists: '+str(payloads))
    grant = subprocess.check_output(['qstat', '-j', '131351736'], text=True)
    if 'h_rt=345600' not in grant or 'pcg01i:16' not in grant:
        raise RuntimeError('retained native allocation could not be confirmed')
    report = {'review_complete': True, 'training_and_prescribed_validation_complete': True,
        'reviewed_utc': datetime.now(timezone.utc).isoformat(), 'training_finished_utc': results[0]['finished_utc'],
        'source': SOURCE, 'config_sha256': CONFIG_SHA, 'ranks': traces, 'development_curve': curve,
        'final967_GT_reference': full_head, 'best_development': read(TRAIN/'best_state.json')['best'],
        'retained_artifacts': artifacts, 'canonical_input_archives': archives,
        'resource_receipt_sha256': sha(OUTPUT/'train_resources.json'),
        'peak_gpu_memory_mib': resources['peak_gpu_memory_mib'], 'supervisor_exit_code': 0,
        'native_kernel_provenance_verified': True, 'owned_payload_cleanup_complete': True,
        'live_source_unchanged': True, 'allocation_released': False, 'working_model330k_unchanged': True,
        'native_allocation': {'job': 131351736, 'host': 'pcg01i', 'physical_gpus': [3, 4], 'CPU_slots': 16,
            'accepted_h_rt_seconds': 345600, 'retained_grant_snapshot': grant},
        'independent_quality_accepted': False, 'formal_predicted_reference_comparison_not_performed': True,
        'review_execution': {'CUDA_used': False, 'CPU_budget': 1,
            'affinity': sorted(os.sched_getaffinity(0)), 'kernel_cgroups': Path('/proc/self/cgroup').read_text()}}
    encoded = (json.dumps(report, indent=2)+'\n').encode()
    for path in (review_path, DURABLE/'primary_completion_review.json'):
        with path.open('xb') as handle:
            handle.write(encoded)
    archive = DURABLE/'completion_receipts_20261107.tar.gz'
    with tarfile.open(archive, 'x:gz') as bundle:
        for path in (review_path, OUTPUT/'train_resources.json', TRAIN/'result_rank_0.json',
                TRAIN/'result_rank_1.json', TRAIN/'best_state.json', TRAIN/'evaluation_250000/summary.json',
                TRAIN/'evaluation_300000/summary.json', TRAIN/'final_300000/summary.json'):
            bundle.add(path, arcname=str(path.relative_to(OUTPUT)), recursive=False)
    print(json.dumps({'review_path': str(review_path), 'durable_review': str(DURABLE/'primary_completion_review.json'),
        'review_sha256': sha(review_path), 'receipt_archive': str(archive), 'receipt_archive_sha256': sha(archive),
        'ranks': traces, 'final967': full_head, 'last_development': curve[-1],
        'best_stage1_visits': report['best_development']['stage1']['samples']}),flush=True)


if __name__ == '__main__':
    main()
