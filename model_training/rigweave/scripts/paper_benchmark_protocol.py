"""Explicit inference-only admission for the first paper comparison."""
import hashlib
import json
from pathlib import Path
import __future__
import importlib.abc
import importlib.machinery
import sys

SCOPE = 'paper_benchmark_20261102_v1'
ENTRYPOINT = 'model_training/rigweave/scripts/evaluate_paper_benchmark.py'
ADAPTER_REVISION = 'official_puppeteer_unit_cube_skin_eager_bf16_20261102_v2'


def puppeteer_normalization(points):
    """Match the authors' normalize_to_unit_cube(scale_factor=0.9995)."""
    import numpy as np
    points = np.asarray(points, dtype=np.float32)
    center = (points.min(0) + points.max(0)) * .5
    extent = np.abs(points - center).max()
    if not np.isfinite(points).all() or extent <= 0:
        raise ValueError('invalid Puppeteer input surface')
    scale = .9995 / extent
    normalized = (points - center) * scale
    pc_center = (normalized.min(0) + normalized.max(0)) * .5
    pc_scale = (normalized.max(0) - normalized.min(0)).max() + 1e-5
    return normalized, center, float(scale), pc_center, float(pc_scale)


def defer_official_annotations(root):
    """Postpone legacy type hints in memory; do not edit the pinned source."""
    root = Path(root).resolve()

    class Loader(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            return compile(self.get_source(fullname), self.path, 'exec',
                flags=__future__.annotations.compiler_flag, dont_inherit=True)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if (spec and spec.origin and isinstance(spec.loader, importlib.machinery.SourceFileLoader)
                    and Path(spec.origin).resolve().is_relative_to(root)):
                spec.loader = Loader(fullname, spec.origin)
                return spec
            return None

    finder = Finder()
    sys.meta_path.insert(0, finder)
    return finder


def scientific_sha(config):
    value = {k: v for k, v in config.items() if k not in ('audit_result', 'runtime')}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def shared_gpu_policy(config):
    policy = config['runtime'].get('paper_shared_inference')
    expected = {'explicit_human_authorized': True, 'scope': SCOPE,
        'own_memory_fraction': .30, 'minimum_free_mib': 32768,
        'preserve_existing_processes': True}
    if (config.get('release_extension')=='skintokens_skeleton_prefix_20261102_v1'
            and config['runtime'].get('paper_completed_evoweave')):
        expected.update(own_memory_fraction=.10,minimum_free_mib=20480)
    if config.get('family') != SCOPE or policy != expected:
        raise ValueError('shared GPU use is restricted to the authorized paper inference')
    return policy


def shared_gpu_admission(config):
    import subprocess
    policy = shared_gpu_policy(config)
    rows = subprocess.check_output(['nvidia-smi','-i',','.join(config['runtime']['gpu_uuids']),
        '--query-gpu=uuid,memory.free,memory.total','--format=csv,noheader,nounits'],text=True,timeout=30)
    parsed = [[v.strip() for v in line.split(',')] for line in rows.splitlines()]
    if ([r[0] for r in parsed] != config['runtime']['gpu_uuids']
            or any(int(r[1]) < policy['minimum_free_mib']
                or int(r[2])-int(r[1])+int(r[2])*policy['own_memory_fraction'] > int(r[2])*.85 for r in parsed)):
        raise RuntimeError('shared inference lacks recorded free memory and85%whole-card headroom')
    return {'memory_free_mib':int(parsed[0][1]),'memory_total_mib':int(parsed[0][2]),'policy':policy}


def validate_config(config, mode):
    if mode not in ('audit', 'capacity') or config.get('family') != SCOPE:
        raise ValueError('paper comparison is inference-only')
    expected = {'assets': 24, 'input_frames': 8, 'reference_slot': 0,
                'selection_seed': 20261102, 'optimizer_updates': 0,
                'payload_limit_seconds': 10800,
                'methods': ['evoweave330k', 'unirig_official', 'puppeteer_official'],
                'retry_failed_generation': False, 'use_GT_reference': False,
                'case_selection': 'hash_order_excluding_development128'}
    if config.get('release_extension') == 'skintokens_skeleton_prefix_20261102_v1':
        expected['methods'].append('skintokens_official')
        if not config.get('inherit_prepared_inputs') or not config['runtime'].get('paper_completed_evoweave'):
            raise ValueError('baseline extension must reuse prepared inputs and completed predictions')
    if config.get('evaluation') != expected:
        raise ValueError('unreviewed paper comparison protocol')
    revision = config.get('baseline_adapter_revision')
    if revision is not None:
        if revision != ADAPTER_REVISION or not config['runtime'].get('paper_completed_baselines'):
            raise ValueError('adapter repair must bind and retain the completed UniRig generations')
    runtime = config['runtime']
    if (runtime['physical_gpus'] != [4] or runtime['execution_cpu_budget'] != 8
            or runtime['world_size'] != 1 or runtime['require_scheduler_membership'] is not True
            or runtime['allocation'] != 131170166):
        raise ValueError('paper comparison escaped the retained native subset')
    spec = config['working_model']
    if spec['sha256'] != 'abaa87aa7b377f17093850981024a4d63529449f68f082e7c6502ea7a5571018':
        raise ValueError('paper evaluation must retain the selected330k model')
    if mode == 'capacity' and not config.get('audit_result'):
        raise ValueError('CPU audit must be recorded before GPU inference')


def validate_sources(config):
    import subprocess
    result = {}
    for name, spec in config['baseline_sources'].items():
        root = Path(spec['path'])
        revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip()
        if revision != spec['commit'] or dirty:
            raise ValueError('baseline source is not the exact clean pinned revision: ' + name)
        result[name] = revision
    return result
