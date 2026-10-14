#!/usr/bin/env python3
"""Matched legacy-query-first control for the isolated medium comparison."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import traceback
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'model_training/rigweave/src'))
from fusion_evaluation import tree_fingerprint
from probe_virtual_query_reference import ReferenceConditionRig, declared_reference_run
from virtual_query_xyz_runtime import sha256, write_json

CODE_ROOT = Path('/work/wangyy/virtual_fusion_20261013/code')
OUTPUT_ROOT = Path('/home/wangyy/evorig/outputs/virtual_fusion_20261013')
ENTRYPOINT = 'model_training/rigweave/scripts/probe_virtual_fusion_reference.py'
ROUTE = 'legacy_query_first'
BASELINE_CONFIG = {
    'path': (CODE_ROOT / 'model_training/experiments/fusion_a_medium_v1.json').as_posix(),
    'sha256': 'e3ab38819f1594b4b8e381ab56caf7c37b5bce26226ec68706701526f754dfe5',
}
EXPECTED_FINGERPRINTS = {
    'full_base_initial_state_sha256': 'a89b29c903abffe2eead47b5411edb8588b599fe874ce490c990ce07225d3635',
    'inputs_sha256': '2b37990348c01a77986369e655276acad1953509f32f563a2e9221eb95bee0fa',
    'training_order_sha256': 'bcf38c1e159180b0ead429af15bae55f7ced52df5594f9eb39eb1e0833327f4a',
}
ALLOWED_EXCEPTIONS = {
    'probe.paired_output': 'removed; no completed-peer wait',
    'probe.auxiliary_weight': 0,
    'probe.evaluate_auxiliary': False,
    'comparison.gradient_measurement_updates': [],
    'reference_route': ROUTE,
    'reference_match': 'added pinned baseline and expected fingerprints',
}


def validate_config(config):
    expected_match = {'baseline_config': BASELINE_CONFIG, **EXPECTED_FINGERPRINTS}
    if config.get('reference_route') != ROUTE or config.get('reference_match') != expected_match:
        raise ValueError('reference route or pinned reference_match differs')
    baseline_path = Path(BASELINE_CONFIG['path'])
    baseline_bytes = baseline_path.read_bytes()
    if hashlib.sha256(baseline_bytes).hexdigest() != BASELINE_CONFIG['sha256']:
        raise ValueError('pinned A baseline configuration bytes changed')
    expected = json.loads(baseline_bytes)
    del expected['probe']['paired_output']
    weight = config.get('probe', {}).get('auxiliary_weight')
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight != 0:
        raise ValueError('auxiliary weight must be numeric zero')
    expected['probe']['auxiliary_weight'] = weight
    expected['probe']['evaluate_auxiliary'] = False
    expected['comparison']['gradient_measurement_updates'] = []
    expected['reference_route'] = ROUTE
    expected['reference_match'] = expected_match
    if json.dumps(config, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError('reference configuration has drift outside the declared A exceptions')
    return deepcopy(expected_match)


def declared_operation(state, args, config_digest):
    operation = state['active_operation']
    if state.get('architecture_contract', {}).get('id') != 'virtual_query_xyz_aux_ids_v1':
        raise ValueError('wrong architecture authority for the reference comparison')
    if (operation.get('mode') != 'reference_compare' or operation.get('gpu_work_allowed') is not True
            or operation.get('formal_training_allowed') is not False or operation.get('complete') is not False):
        raise ValueError('an incomplete, bounded reference_compare operation must be explicitly authorized')
    entrypoint = operation.get('required_entrypoint')
    if not entrypoint or (ROOT / entrypoint).resolve() != (ROOT / ENTRYPOINT).resolve():
        raise ValueError('the reference operation must require this exact entrypoint')
    if args.config.resolve().parent != (CODE_ROOT / 'model_training/experiments').resolve():
        raise ValueError('reference configuration must remain in the isolated fusion checkout')
    if args.output.resolve().parent != OUTPUT_ROOT.resolve():
        raise ValueError('reference output must remain in the isolated fusion output root')
    declared = declared_reference_run(operation, args, config_digest)
    if declared.get('physical_gpu') != 5:
        raise ValueError('the bounded legacy reference requires the declared physical GPU5')
    return declared


def verify_operation(args):
    if ROOT.resolve() != CODE_ROOT.resolve():
        raise ValueError('reference execution requires the isolated HGC source checkout')
    state = json.loads((ROOT / 'model_training/state/current.json').read_text())
    declared_operation(state, args, sha256(args.config))
    operation = state['active_operation']
    subprocess.run([sys.executable, str(ROOT / 'model_training/tools/agent_work_guard.py'),
        'check', '--operation', 'train', '--entrypoint', str(Path(__file__).resolve())], check=True)
    if subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
        raise ValueError('reference comparison requires clean published source')
    subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', operation['code_commit'], 'HEAD'], check=True)
    subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet', operation['code_commit'], 'HEAD', '--',
        'model_training/rigweave', 'model_training/tools'], check=True)
    return subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()


def configure_device():
    if socket.gethostname().split('.')[0] != 'pcg02' or os.environ.get('CUDA_VISIBLE_DEVICES') != '5':
        raise ValueError('reference comparison requires pcg02 and physical CUDA_VISIBLE_DEVICES=5')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('reference comparison requires exactly one visible CUDA device')
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.95, 0)
    return torch.device('cuda:0')


def verify_reference_fingerprints(expected, actual):
    if set(expected) != set(EXPECTED_FINGERPRINTS) or set(actual) != set(expected):
        raise ValueError('reference fingerprint fields are incomplete')
    mismatches = {key: {'expected': expected[key], 'actual': actual[key]}
                  for key in expected if expected[key] != actual[key]}
    if mismatches:
        raise ValueError('reference fingerprint mismatch before optimizer updates: ' + json.dumps(mismatches))


def require_reference_gradients(model, norms, weight):
    if weight != 0:
        raise ValueError('the legacy reference has no auxiliary gradient branch')
    for name in ('surface_tokenizer', 'motion_encoder', 'xyz_decoder'):
        norm = float(norms.get(name, 0.))
        if not math.isfinite(norm) or norm <= 0:
            raise RuntimeError('missing, nonfinite or zero legacy reference gradient: ' + name)
    for name, module in (('embedding', model.xyz_decoder.get_input_embeddings()),
                         ('head', model.xyz_decoder.get_output_embeddings())):
        gradient = module.weight.grad
        norm = float(gradient.float().norm()) if gradient is not None else 0.
        if (gradient is None or not bool(torch.isfinite(gradient).all())
                or not math.isfinite(norm) or norm <= 0):
            raise RuntimeError('missing, nonfinite or zero original XYZ ' + name + ' gradient')


def run_reference(config, output, device, source, *, config_path):
    import probe_virtual_query_lambda as comparison

    match = validate_config(config)
    config = deepcopy(config)
    config['_path'] = str(config_path)
    groups, index = comparison.load_cache(config)
    if {name: len(rows) for name, rows in groups.items()} != {'train': 64, 'train_heldout': 8, 'valid': 32}:
        raise ValueError('the reference requires the same 104-entry raw cache')
    items = groups['train'] + groups['train_heldout'] + groups['valid']
    budget = comparison.comparison_plan(config, len(groups['train']))
    order = comparison.training_order(len(groups['train']), budget['actual_asset_exposures'],
                                     config['comparison']['order_seed'])
    order_sha = hashlib.sha256(json.dumps(order, separators=(',', ':')).encode()).hexdigest()
    base, tokenizer, codec, initial = comparison.build_model(config, device)
    base_sha = comparison.parameter_fingerprint(base)
    actual = {'full_base_initial_state_sha256': base_sha, 'inputs_sha256': tree_fingerprint(items),
              'training_order_sha256': order_sha}
    expected = {key: match[key] for key in EXPECTED_FINGERPRINTS}
    verify_reference_fingerprints(expected, actual)
    if (initial.get('checkpoint_sha256') != config['initialization']['sha256']
            or initial.get('old_method_trained_weights_loaded') is not False
            or not all(initial.get('load_report', {}).get(key, {}).get('strict') is True
                       for key in ('transformer.', 'mesh_encoder.', 'output_proj.'))):
        raise ValueError('reference official initialization loader was not verified')
    initial.update(full_base_initial_state_sha256=base_sha, reference_route=ROUTE,
                   approved_virtual_route=False)
    model = ReferenceConditionRig(base, ROUTE)
    matched_path = output / 'matched_reference.json'
    write_json(matched_path, {'stage': 'reference_compare_pre_update_match', 'passed': True,
        'optimizer_updates': 0, 'reference_route': ROUTE, 'code_commit': source,
        'baseline_config': match['baseline_config'], 'expected': expected, 'actual': actual,
        'allowed_exceptions': ALLOWED_EXCEPTIONS, 'budget': budget,
        'common_cache_index_sha256': config['comparison']['cache']['sha256'],
        'official_loader': initial['load_report'], 'peer_wait_required': False,
        'auxiliary_applicable': False, 'formal_training_allowed': False})
    # Keep the common runner and its optimizer/evaluation loop intact; reuse the
    # already verified cache and base model rather than loading either twice.
    with patch.object(comparison, 'load_cache', return_value=(groups, index)), \
            patch.object(comparison, 'build_model', return_value=(model, tokenizer, codec, initial)), \
            patch.object(comparison, 'require_gradients', require_reference_gradients):
        result = comparison.run_comparison(config, output, device, source)
    if result['training_order_sha256'] != actual['training_order_sha256']:
        raise RuntimeError('common runner training order differs from the pre-update reference match')
    return {**result, 'stage': 'fusion_reference_comparison', 'reference_route': ROUTE,
            'reference_only': True, 'approved_virtual_route': False, 'accepted': False,
            'accepted_meaning': 'matched_legacy_reference_only_not_candidate_or_formal_acceptance',
            'formal_training_allowed': False, 'auxiliary_applicable': False,
            'auxiliary_metrics_meaning': 'not_applicable; common trace zero placeholders are not auxiliary scores',
            'matched_reference_sha256': sha256(matched_path),
            'full_base_initial_state_sha256': base_sha, 'inputs_sha256': actual['inputs_sha256'],
            'next_action': 'primary_matched_reference_review_required_no_automatic_continuation'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    source = verify_operation(args)
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result = run_reference(config, args.output, configure_device(), source, config_path=args.config)
        write_json(args.output / 'result.json', {**result, 'execution_complete': True,
            'finished_utc': datetime.now(timezone.utc).isoformat(), 'code_commit': source})
    except Exception:
        write_json(args.output / 'result.json', {'stage': 'fusion_reference_comparison',
            'accepted': False, 'execution_complete': False, 'reference_route': ROUTE,
            'reference_only': True, 'formal_training_allowed': False, 'auxiliary_applicable': False,
            'traceback': traceback.format_exc(), 'code_commit': source})
        raise


if __name__ == '__main__':
    main()
