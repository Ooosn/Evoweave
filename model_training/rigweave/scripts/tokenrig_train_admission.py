"""Bind Qwen formal training to its completed stress tests and live CPU audit."""
import csv
import io
import json
import math
from pathlib import Path
import tarfile

from anchor_full_protocol import scientific_sha, sha256, source_hashes


def verify_capacity(config):
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
