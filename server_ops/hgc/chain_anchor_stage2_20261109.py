"""Run the admitted Stage 2 CPU audit, both capacity trials and matched training."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE/'model_training/rigweave/scripts'))
from anchor_stage2_protocol import (ACCEPTED_WHOLE_GPU_MIB, ARMS, ENTRYPOINT, MILESTONES, SCOPE,
                                    capacity_stage, frame_policy, sha256, validate_config)

STATE = CODE/'model_training/state/current.json'
CONFIG = CODE/'model_training/experiments'/f'{SCOPE}.json'
REF = 'refs/heads/codex/anchor-stage2-5k-chunk4-20261109'
RECOVERY_REF = 'refs/heads/codex/anchor-stage2-frozen-eval-20261110'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
    temp.replace(path)


def git(*args):
    return subprocess.check_output(['git', '-C', str(CODE), *args], text=True, timeout=120).strip()


def publish(state, message, ref):
    state['updated_at'] = now()
    write(STATE, state)
    git('-c', 'user.name=Evoweave native runner', '-c', 'user.email=evoweave-runner@localhost',
        'commit', '--only', '-m', message, '--', str(STATE.relative_to(CODE)))
    head = git('rev-parse', 'HEAD')
    git('push', 'origin', 'HEAD:'+ref)
    if git('ls-remote', 'origin', ref).split() != [head, ref] or git('status', '--porcelain'):
        raise RuntimeError('private state publication did not verify')
    return head


def spec(path):
    return {'path': str(path), 'sha256': sha256(path)}


def stage(state, config, config_path, ref, mode, name):
    op = state['active_operation']
    root = Path(config['runtime']['output_root'])
    out, resources = root/name, root/f'{name}_resources.json'
    durable = Path(config['artifacts']['durable_root'])
    receipts = durable/'receipts'/name
    if out.exists() or resources.exists() or receipts.exists():
        raise RuntimeError('refuse to overwrite a retained attempt')
    receipts.mkdir(parents=True)
    config_sha = sha256(config_path)
    op.update(status=f'{name}_admitted', gpu_work_allowed=mode != 'audit',
              formal_training_allowed=mode == 'train',
              runs=[{'mode': mode, 'config': str(config_path), 'config_sha256': config_sha,
                     'output': str(out), 'approved': True}])
    state.update(state_id=f'{config_path.stem}_{name}_admitted',
                 allowed_operations=['inspect', 'report', 'preflight'] + (['train'] if mode == 'train' else []),
                 blocked_operations=['submit', 'matched_eval'] + ([] if mode == 'train' else ['train']),
                 next_required_result=f'Complete and verify the pinned {name} stage.')
    head = publish(state, f'Admit standalone Stage 2 {name}', ref)
    write(receipts/'admission.json', {'source': head, 'operation': op})
    write(durable/'status.json', {'complete': False, 'stage': name, 'mode': mode, 'source': head,
                                 'started_utc': now(), 'output': str(out)})
    subprocess.run([sys.executable, '-B', str(CODE/'model_training/tools/agent_work_guard.py'), 'begin'],
                   cwd=CODE, check=True, stdout=subprocess.DEVNULL)
    with (receipts/'stage.log').open('xb') as log:
        run = subprocess.run([sys.executable, '-B', str(CODE/'model_training/jobs/launch_virtual_pair.py'),
            '--mode', mode, '--config', str(config_path), '--output', str(out)], cwd=CODE,
            stdout=log, stderr=subprocess.STDOUT)
    for path in (resources, out/'result.json', *out.glob('failure_rank_*.json'),
                 *out.glob('result_rank_*.json'), *out.glob('progress_rank_*.json')):
        if path.is_file():
            shutil.copyfile(path, receipts/path.name)
    receipt_path = receipts/resources.name
    if not receipt_path.is_file():
        raise RuntimeError(f'{name} failed before resource admission; see {receipts}/stage.log')
    resource = read(receipt_path)
    if (not resource['complete'] or resource['source'] != head or not resource['kernel_scheduler_membership']
            or not resource.get('payload_cleanup_complete')):
        raise RuntimeError(f'{name} did not exit with confirmed native payload cleanup')
    result_path = receipts/'result.json'
    if run.returncode or not resource['passed'] or not result_path.is_file():
        failures = [read(p) for p in receipts.glob('failure_rank_*.json')]
        memory_failure = any(row['cuda_oom'] for row in failures) or (
            resource.get('error') == "RuntimeError('the training process tree exceeded64GiB RAM')")
        if mode == 'capacity' and memory_failure:
            outcome = {'passed': False, 'reason': 'insufficient capacity', 'failures': failures,
                       'resources': spec(receipt_path)}
            write(receipts/'outcome.json', outcome)
            return outcome
        raise RuntimeError(f'{name} failed; see {receipts}/stage.log')
    result = read(result_path)
    if (not result['complete'] or not result['passed'] or result['source'] != head
            or result['config_sha256'] != config_sha):
        raise RuntimeError(f'{name} result is not bound to its admission')
    if mode != 'audit':
        for rank in (0, 1):
            other = read(receipts/f'result_rank_{rank}.json')
            keys = set(result) - {'rank', 'finished_utc', 'elapsed_seconds', 'validation'}
            if other['rank'] != rank or any(result[k] != other[k] for k in keys):
                raise RuntimeError(f'{name} has unequal rank completion')
    peaks = resource.get('peak_gpu_memory_mib', {})
    if mode == 'capacity' and (set(peaks) != {'3', '4'} or not min(peaks.values())):
        raise RuntimeError('missing live two-device peak')
    # Leave three GiB below the unchanged 78 GiB whole-device stop line.
    accepted = mode != 'capacity' or max(peaks.values()) <= ACCEPTED_WHOLE_GPU_MIB
    outcome = {'passed': accepted, 'result': spec(result_path), 'resources': spec(receipt_path),
               'peak_gpu_memory_mib': peaks}
    if not accepted:
        outcome['reason'] = 'insufficient whole-device headroom'
    write(receipts/'outcome.json', outcome)
    return outcome


def check_training(result, planned):
    if (any(result[k] != planned[k] for k in ('samples', 'updates', 'input_frame_visits', 'plan_sha256'))
            or [r['samples'] for r in result['validation']] != [0, *MILESTONES]
            or [r['additional_asset_exposures'] for r in result['checkpoints']] != [48, *MILESTONES]
            or not all(r['both_copies_verified'] for r in result['checkpoints'])):
        raise RuntimeError('formal run did not complete the exact sampling/checkpoint/evaluation plan')
    for value in result['validation']:
        row = next(g for g in value['groups'] if g['method'] == 'head')
        if (row['assets'], row['rows'], row['normalized_mpjpe_scored_assets']) != (128, 1792, 128):
            raise RuntimeError('GT reference evaluation coverage changed')


def training_inputs(root):
    identities = []
    for rank in (0, 1):
        h = hashlib.sha256()
        with (root/f'updates_rank_{rank}.jsonl').open(encoding='utf-8') as f:
            for step, line in enumerate(f, 1):
                row = json.loads(line)
                if row['step'] != step:
                    raise ValueError('training trace has a missing update')
                h.update(json.dumps(row['microbatches'], sort_keys=True).encode())
        identities.append(h.hexdigest())
    return identities


def run(state, config, config_path, ref):
    from virtual_pair_guard import deadline_due, require_job_membership, verify_allocation

    op = state['active_operation']
    runtime = config['runtime']
    require_job_membership(Path('/proc/self/cgroup').read_text(), runtime['allocation'])
    verify_allocation(config, subprocess.check_output(['qstat', '-j', str(runtime['allocation'])],
                                                      text=True, timeout=30))
    if deadline_due(config):
        raise RuntimeError('native allocation reached its save-stop line')
    audited = stage(state, config, config_path, ref, 'audit', 'audit')
    audit = read(Path(audited['result']['path']))
    if audit['cuda_initialized'] or audit['optimizer_updates']:
        raise RuntimeError('Stage 2 CPU audit must not use CUDA or update a model')
    if 'recovery' in config:
        if audit['matched_target_rows'] != 1792:
            raise RuntimeError('frozen panel must match all retained development targets')
        op['stage_evidence']['recovery'] = audited['result']
    else:
        if audit['train_assets'] != 5000:
            raise RuntimeError('Stage 2 CPU audit did not verify the fixed train inputs')
        op.update(CPU_audit_accepted=True, stage_evidence={'audit': audited['result']}, capacity_attempts={})
        for size in (24, 12):
            results = {arm: stage(state, config, config_path, ref, 'capacity', capacity_stage(arm, size)) for arm in ARMS}
            op['capacity_attempts'][str(size)] = results
            if all(v['passed'] for v in results.values()):
                op.update(capacity_accepted=True, capacity_selection={'long_batch_per_rank': size,
                    'frame_policy': frame_policy(size), 'results': {k: v['result'] for k, v in results.items()},
                    'resources': {k: v['resources'] for k, v in results.items()}})
                break
        else:
            raise RuntimeError('neither user-authorized shared batch passed both arms')
    durable = Path(config['artifacts']['durable_root'])
    op['training_results'] = {}
    curves, reference = {}, None
    initial = read(Path(op['stage_evidence']['audit']['path']))['initialization']
    for arm in ARMS:
        trained = stage(state, config, config_path, ref, 'train', arm)
        result = read(Path(trained['result']['path']))
        check_training(result, audit['plan'])
        if result['initialization'] != initial[arm]:
            raise RuntimeError('formal initialization changed after profiling')
        identity = training_inputs(durable/arm)
        valid = {v['input_panel_sha256'] for v in result['validation']}
        if (len(valid) != 1 or ('recovery' in config and valid != {audit['input_panel_sha256']})
                or (reference is not None and reference != (identity, valid))):
            raise RuntimeError('actual training inputs or validation identities differ')
        reference = (identity, valid)
        op['training_results'][arm] = trained
        curves[arm] = [{'samples': value['samples'], 'seconds': value['evaluation_seconds'],
                        **next(g for g in value['groups'] if g['method'] == 'head')}
                       for value in result['validation']]
    result_path = durable/'comparison.json'
    write(result_path, {'complete': True, 'passed': True, 'arms': ARMS, 'samples_per_arm': 100000,
        'updates_per_arm': audit['plan']['updates'], 'long_batch_per_rank': op['capacity_selection']['long_batch_per_rank'],
        'input_frame_visits_per_arm': audit['plan']['input_frame_visits'],
        'training_input_sha256_by_rank': reference[0], 'input_panel_sha256': next(iter(reference[1])),
        'validation': curves, 'reference': 'GT_only', 'working_model_changed': False})
    return spec(result_path)


def close(state, config, config_path, ref, evidence, error=None):
    op = state['active_operation']
    status = 'complete' if not error else 'failed_or_stopped'
    op.update(complete=True, status=status, formal_training_allowed=False, gpu_work_allowed=False,
              completion=evidence, error=error)
    for row in op['runs']:
        row['approved'] = False
    state.update(state_id=f'{config_path.stem}_{status}', allowed_operations=['inspect', 'report'],
                 blocked_operations=['submit', 'preflight', 'train', 'matched_eval'],
                 next_required_result='Review matched Stage 2 evidence; no automatic model change.')
    head = publish(state, f'Record standalone Stage 2 {status}', ref)
    write(Path(config['artifacts']['durable_root'])/'status.json', {'complete': True, 'passed': not error,
          'source': head, 'evidence': evidence, 'error': error, 'finished_utc': now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    config_path = parser.parse_args().config.resolve()
    state, config = read(STATE), read(config_path)
    validate_config(config)
    ref = RECOVERY_REF if 'recovery' in config else REF
    op = state['active_operation']
    head = git('rev-parse', 'HEAD')
    if (CODE != Path(config['runtime']['repo']) or git('status', '--porcelain')
            or config_path.parent != CONFIG.parent
            or git('ls-remote', 'origin', ref).split() != [head, ref]
            or op['experiment'] != config_path.stem or op['explicit_human_authorized'] is not True
            or op['required_entrypoint'] != ENTRYPOINT or op['complete']
            or op['config_sha256'] != sha256(config_path)
            or (Path(config['artifacts']['durable_root'])/'status.json').exists()):
        raise RuntimeError('new private checkout and exact Stage 2 authorization required')
    try:
        evidence = run(state, config, config_path, ref)
    except BaseException as exc:
        close(state, config, config_path, ref, None, str(exc))
        raise
    close(state, config, config_path, ref, evidence)


if __name__ == '__main__':
    main()
