"""Bind Qwen formal training to its completed stress tests and live CPU audit."""
import csv
from concurrent.futures import ThreadPoolExecutor
import io
import json
import math
from pathlib import Path
import tarfile

from anchor_full_protocol import scientific_sha, sha256, source_hashes, validate_config


def stage_cache_reflink(config, base):
    import fcntl
    from anchor_full_data import verify_ssd
    from virtual_query_xyz_runtime import write_json

    spec = config['data_cache']
    old, new = Path(spec['reflink_source']).resolve(), Path(spec['root']).resolve()
    if old != Path('/work/wangyy/anchor_full_20261030_data') or old == new:
        raise RuntimeError('unexpected TokenRig cache source')
    index = json.loads((old/'index.json').read_text())
    if not index['complete'] or index['manifests'] != base['data']['manifests']:
        raise RuntimeError('cache and formal manifests differ')
    new.mkdir(exist_ok=False)
    storage = verify_ssd(new)
    if old.stat().st_dev != new.stat().st_dev:
        raise RuntimeError('cache reflinks must remain on the verified SSD')

    def clone(item):
        source = Path(item['cached_path']).resolve(strict=True)
        target = new/source.relative_to(old)
        if source.stat().st_size != item['bytes']:
            raise RuntimeError('source cache size differs')
        target.parent.mkdir(exist_ok=True, parents=True)
        with source.open('rb') as read, target.open('xb') as write:
            fcntl.ioctl(write.fileno(), 0x40049409, read.fileno())  # Linux FICLONE
        if target.stat().st_size != item['bytes']:
            raise RuntimeError('reflink length differs')
        target.chmod(0o444)
        return {**item, 'cached_path': str(target)}

    files = index['files']
    with ThreadPoolExecutor(max_workers=4) as pool:
        index['files'] = dict(zip(files, pool.map(clone, files.values()), strict=True))
    index['storage'] = storage
    write_json(spec['index'], index)
    return {'sha256': sha256(spec['index']), 'files': len(files), 'copy_on_write': True}


def verify_capacity(config):
    if config['tokenrig_decoder'].get('conditioning'):
        return verify_native_capacity(config)
    expected = {'default': ('output', 6, 24, 1, 12),
                'frames12': ('frames12', 12, 24, 1, 4),
                'frames24': ('frames24', 24, 12, 2, 4)}
    decoder = config['tokenrig_decoder']
    peaks = {}
    for spec in config['execution']['tokenrig_capacity_archives']:
        if sha256(spec['path']) != spec['sha256']:
            raise RuntimeError('TokenRig capacity archive digest differs')
        with tarfile.open(spec['path'], 'r:gz') as archive:
            for name in spec['profiles']:
                folder, frames, batch, accum, count = expected[name]
                if name in peaks or config['training']['frame_policy'][str(frames)] != batch:
                    raise RuntimeError('capacity profile differs from the formal batch policy')
                with archive.extractfile(folder+'/nvml.csv') as handle:
                    rows = list(csv.reader(io.TextIOWrapper(handle)))
                observed = {row[-2].strip() for row in rows}
                peak = max(int(row[-1]) for row in rows)
                if observed != set(config['runtime']['gpu_uuids']) or not 0 < peak < 78000:
                    raise RuntimeError('capacity GPU identity or memory differs')
                peaks[name] = peak
                for rank in (0, 1):
                    with archive.extractfile(f'{folder}/result_rank_{rank}.json') as handle:
                        result = json.load(handle)
                    init = result['initialization']
                    cases = result['cases']
                    if (result['complete'] is not True or len(cases) != count
                            or not init['strict_weights_loaded'] or init['width'] != 896
                            or init['skeleton_vocab_size'] != 267
                            or any(init[key] != decoder[key] for key in ('checkpoint_sha256',
                                'qwen_config_sha256', 'tokenizer_source_sha256'))):
                        raise RuntimeError('capacity initialization or completion differs')
                    selected = [case for case in cases if case['frames'] == frames]
                    if ({(case['stress_group'], case['step']) for case in selected} != {
                        (group, step) for group in ('largest_joints', 'largest_geometry') for step in (1, 2)}
                            or len(selected) != 4 or any(case['batch_per_rank'] != batch
                            or case['accumulation'] != accum or case['assets_in_update'] != 48
                            or not all(math.isfinite(case[key]) for key in ('ar_loss', 'xyz_loss',
                                'bone_loss', 'gradient_norm')) for case in selected)):
                        raise RuntimeError('capacity lacks four verified stress updates per profile')
                    if name == 'default' and not result['teacher_prefix']['passed']:
                        raise RuntimeError('released decoder lacks a cached-prefix parity check')
    if set(peaks) != set(expected):
        raise RuntimeError('capacity evidence did not cover the full frame policy')
    return {'complete': True, 'passed': True, 'peak_nvml_mib_by_profile': peaks}


def verify_native_capacity(config):
    spec = config['execution']['capacity']
    if (sha256(spec['path']) != spec['sha256']
            or sha256(spec['resources_path']) != spec['resources_sha256']):
        raise RuntimeError('native TokenRig capacity evidence changed')
    result = json.loads(Path(spec['path']).read_text())
    resources = json.loads(Path(spec['resources_path']).read_text())
    audited = verify_audit(config)
    if (result.get('complete') is not True or result.get('passed') is not True
            or result.get('scientific_sha256') != scientific_sha(config)
            or result.get('source_sha256') != source_hashes()
            or result.get('initialization') != audited['initialization']
            or result.get('profile_weights_not_training_initialization') is not True
            or resources.get('complete') is not True or resources.get('passed') is not True
            or resources.get('mode') != 'capacity' or resources.get('exit_code') != 0
            or resources.get('source') != result.get('source')
            or resources.get('allocation') != config['runtime']['allocation']
            or not resources.get('kernel_scheduler_membership')
            or resources.get('payload_cleanup_complete') is not True):
        raise RuntimeError('native capacity is not bound to the exact source and recipe')
    peaks = resources.get('peak_gpu_memory_mib', {})
    if (set(peaks) != {str(i) for i in config['runtime']['physical_gpus']}
            or any(not 0 < value <= 78 * 1024 for value in peaks.values())):
        raise RuntimeError('native capacity exceeds the whole-device memory ceiling')
    base = validate_config(config)
    steps = config['capacity']['optimizer_steps_per_case']
    expected = {(case['frames'], case['batch_per_rank'], case['stress_group'], step)
        for case in base['capacity']['cases'] for step in range(1, steps + 1)}
    ranks = result.get('ranks', [])
    if len(ranks) != 2 or [r['rank'] for r in ranks] != [0, 1]:
        raise RuntimeError('native capacity needs both approved ranks')
    for rank in ranks:
        cases = rank['cases']
        extreme = [case for case in cases if case['profile'] == 'full_manifest_extreme']
        if (len(cases) != len(expected) + 1 or len(extreme) != len(expected)
                or {(c['frames'], c['batch_per_rank'], c['stress_group'], c['optimizer_step'])
                    for c in extreme} != expected
                or any(c['assets_in_update'] != 48 for c in extreme)
                or any(not all(math.isfinite(c[key]) for key in
                    ('ar_loss', 'xyz_loss', 'bone_loss', 'gradient_norm')) for c in cases)
                or rank.get('resume_roundtrip_exact') is not True
                or len(rank['representative_updates']) != config['capacity']['representative_updates']
                or rank['FP32_inference']['teacher_prefix'].get('passed') is not True
                or rank['FP32_inference'].get('XYZ_finite') is not True):
            raise RuntimeError('native capacity lacks finite stress updates and exact recovery')
    checkpoint = result['resume_checkpoint']
    if (checkpoint.get('complete') is not True or checkpoint.get('both_copies_verified') is not True
            or checkpoint.get('optimizer_and_rank_RNG_retained') is not True):
        raise RuntimeError('native capacity recovery was not durably verified')
    return result


def verify_audit(config):
    spec = config['execution']['CPU_audit']
    audit = json.loads(Path(spec['path']).read_text())
    resources = json.loads(Path(spec['resources_path']).read_text())
    if (audit['complete'] is not True or audit['passed'] is not True or audit['cuda_initialized']
            or audit['scientific_sha256'] != scientific_sha(config)
            or audit['source_sha256'] != source_hashes()
            or audit['source'] != resources['source']
            or audit['plan']['assets'] != config['training']['additional_assets']
            or resources['complete'] is not True or resources['passed'] is not True
            or resources['mode'] != 'audit' or resources['exit_code'] != 0
            or resources['allocation'] != config['runtime']['allocation']
            or not resources['kernel_scheduler_membership'] or not resources['payload_cleanup_complete']):
        raise RuntimeError('live TokenRig CPU audit did not pass for this exact source and recipe')
    return audit
