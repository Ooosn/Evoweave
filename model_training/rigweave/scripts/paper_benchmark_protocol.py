"""Explicit inference-only admission for the first paper comparison."""
import hashlib
import json
from pathlib import Path

SCOPE = 'paper_benchmark_20261102_v1'
ENTRYPOINT = 'model_training/rigweave/scripts/evaluate_paper_benchmark.py'


def scientific_sha(config):
    value = {k: v for k, v in config.items() if k not in ('audit_result', 'runtime')}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_config(config, mode):
    if mode not in ('audit', 'capacity') or config.get('family') != SCOPE:
        raise ValueError('paper comparison is inference-only')
    expected = {'assets': 24, 'input_frames': 8, 'reference_slot': 0,
                'selection_seed': 20261102, 'optimizer_updates': 0,
                'payload_limit_seconds': 10800,
                'methods': ['evoweave330k', 'unirig_official', 'puppeteer_official'],
                'retry_failed_generation': False, 'use_GT_reference': False,
                'case_selection': 'hash_order_excluding_development128'}
    if config.get('evaluation') != expected:
        raise ValueError('unreviewed paper comparison protocol')
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
