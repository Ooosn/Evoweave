"""Close the static baseline and execute the three native Stage 1 controls."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from review_skintokens_stage1_20261109 import (
    IDENTITY, POINTS, compare, digest, identities, panel, read, review_training, stage_receipt,
)


CODE = Path(__file__).resolve().parents[2]
STATE = CODE/'model_training/state/current.json'
sys.path.insert(0, str(CODE/'model_training/rigweave/scripts'))
from skintokens_stage1_protocol import ARMS, DURABLE, ENTRYPOINT, REPO, SCOPE, STATIC_CONFIG, recipe

RUNNER = Path('/home/wangyy/jobs/evoweave_h100_20261107_145343/runner')
STATIC = 'skintokens_static_5k_20261109'
BASELINE_REVIEW = Path(DURABLE)/STATIC/'receipts/completion_review.json'
STATUS = Path(DURABLE)/SCOPE/'status.json'
GIT_URL = 'https://github.com/Ooosn/Evoweave.git'
GIT_REF = 'refs/heads/codex/skintokens-stage1-5k-20261109'
STATE_KEY = 'skintokens_stage1_followup_20261109'
PY = sys.executable


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, data, *, new=False):
    raw = json.dumps(data, indent=2, ensure_ascii=True, allow_nan=False)+'\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if new and path.exists():
        if read(path) != data:
            raise ValueError('refuse to replace retained evidence: '+str(path))
        return
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('x', encoding='utf-8') as handle:
        handle.write(raw)
    temporary.replace(path)


def git(*args):
    return subprocess.check_output(['git', '-C', str(CODE), *args], text=True, timeout=120).strip()


def require_static_done():
    name = STATIC+'_train.sh'
    if (not (RUNNER/'done'/name).is_file() or (RUNNER/'failed'/name).exists()
            or (RUNNER/'running'/name).exists()):
        raise RuntimeError('native runner has not completed the baseline successfully')


def review_static():
    require_static_done()
    state = read(STATE)
    op = state['active_operation']
    config = read(STATIC_CONFIG)
    if (op['experiment'] != STATIC or op['config_sha256'] != digest(STATIC_CONFIG)
            or not op['verified_start']['kernel_scheduler_membership']):
        raise ValueError('static baseline is not the pinned live operation')
    result = review_training(STATIC_CONFIG, config, op['verified_start']['source'])
    audit_file = Path(op['output_root'])/'audit/result.json'
    if digest(audit_file) != op['stage_evidence']['audit']['result_sha256']:
        raise ValueError('baseline CPU audit changed')
    identity = {r['asset_id']: tuple(r[k] for k in IDENTITY) for r in read(audit_file)['fixed_validation_inputs']}
    _, initial = panel(Path(result['retained_panel_root'])/'evaluation_000000', 0)
    if identity != identities(initial):
        raise ValueError('baseline validation input identity differs from the admitted panel')
    write(BASELINE_REVIEW, result, new=True)
    print(json.dumps({'path': str(BASELINE_REVIEW), 'sha256': digest(BASELINE_REVIEW),
        'visits': result['samples'], 'updates': result['updates'], 'curve': result['curve'],
        'peak_gpu_memory_mib': result['resources']['peak_gpu_memory_mib']}))


def record_static(local_review, expected_sha):
    if digest(local_review) != expected_sha:
        raise ValueError('downloaded baseline review is not the verified artifact')
    state, result = read(STATE), read(local_review)
    op = state['active_operation']
    if (op['experiment'] != STATIC or op['complete'] or STATE_KEY in state
            or not result['complete'] or not result['passed'] or result['arm'] != 'static'
            or result['resources']['source'] != op['verified_start']['source']
            or (result['samples'], result['updates']) != (30000, 627)):
        raise ValueError('baseline completion is not bound to the active run')
    evidence = {'path': BASELINE_REVIEW.as_posix(), 'sha256': expected_sha}
    close(state, 'skintokens_static_5k_20261109_complete', evidence)
    state['active_operation'].update(samples=30000, optimizer_updates=627,
        latest_checkpoint=result['checkpoints'][-1], quality_accepted=False)
    state[STATE_KEY] = {'authorization': 'User directed analysis and launch after the baseline completes.',
        'explicit_human_authorized': True, 'ordered_arms': list(ARMS), 'static_review': evidence,
        'completed': {}, 'automatic_retry_allowed': False, 'working_model_change_allowed': False}
    state['next_required_result'] = 'Sequentially audit, capacity-check and train the three Stage 1-only controls.'
    write(STATE, state)


def close(state, status, review):
    op = state['active_operation']
    op.update(status=status, complete=True, gpu_work_allowed=False, formal_training_allowed=False,
              completion_review=review)
    for row in op['runs']:
        row['approved'] = False
    state.update(state_id=status, updated_at=now(), allowed_operations=['inspect', 'report'],
        blocked_operations=['submit', 'train', 'preflight', 'matched_eval'])


def publish(message, *paths):
    files = [str(path.relative_to(CODE)) for path in paths]
    git('add', '--', *files)
    git('-c', 'user.name=Evoweave native runner', '-c', 'user.email=evoweave-runner@localhost',
        'commit', '--only', '-m', message, '--', *files)
    head = git('rev-parse', 'HEAD')
    git('push', 'origin', 'HEAD:'+GIT_REF)
    if git('ls-remote', 'origin', GIT_REF).split() != [head, GIT_REF] or git('status', '--porcelain'):
        raise RuntimeError('private state publication did not verify')
    return head


def stage(config, path, mode, state, source, preflights):
    root = Path(config['runtime']['output_root'])
    receipts = Path(config['artifacts']['durable_root'])/'receipts'
    receipts.mkdir(parents=True, exist_ok=True)
    out = root/mode
    log = receipts/f'{mode}.log'
    if out.exists() or log.exists() or (root/f'{mode}_resources.json').exists():
        raise RuntimeError('refuse to overwrite an earlier attempt')
    arm = config['arm']
    op = {'label': SCOPE, 'experiment': SCOPE, 'arm': arm, 'complete': False,
        'status': 'exact_'+mode+'_admitted', 'code_commit': source,
        'explicit_human_authorized': True, 'authorization': state[STATE_KEY]['authorization'],
        'required_entrypoint': ENTRYPOINT, 'operation_entrypoints': {'preflight': ENTRYPOINT, 'train': ENTRYPOINT},
        'formal_training_allowed': mode == 'train', 'gpu_work_allowed': mode != 'audit',
        'CPU_audit_accepted': 'audit' in preflights, 'capacity_accepted': 'capacity' in preflights,
        'stage_evidence': preflights.copy(), 'training_manifest': config['train_manifest'],
        'initialization': {'checkpoint': config['tokenrig_decoder']['checkpoint'],
                           'sha256': config['tokenrig_decoder']['checkpoint_sha256']},
        'resource_binding': {k:config['runtime'][k] for k in
            ('allocation', 'host', 'physical_gpus', 'cpu_slots', 'stop_before_utc')},
        'validation': config['evaluation'], 'output_root': str(root), 'durable_root': str(receipts.parent),
        'acceptance_test': 'Strict release, matched fixed inputs, no stage 2; bounded finite capacity then 30000 visits/627 updates and all durable checkpoints.',
        'automatic_retry_allowed': False, 'working_model_change_allowed': False,
        'runs': [{'mode': mode, 'approved': True, 'config': str(path), 'config_sha256': digest(path),
                  'output': str(out)}]}
    state.update(active_operation=op, state_id=f'{SCOPE}_{arm}_{mode}', updated_at=now(),
        allowed_operations=['inspect','report','preflight'] + (['train'] if mode == 'train' else []),
        blocked_operations=['submit','matched_eval'] + ([] if mode == 'train' else ['train']),
        next_required_result='Verify this exact native stage before admitting the next one.')
    status = {'complete': False, 'arm': arm, 'mode': mode, 'source': None,
        'config': str(path), 'config_sha256': digest(path), 'output': str(out), 'admitting_utc': now(),
        'finished_arms': list(state[STATE_KEY]['completed']),
        'remaining_arms': list(ARMS[ARMS.index(arm)+1:])}
    write(STATUS, status)
    write(STATE, state)
    head = publish(f'Admit SkinTokens {arm} {mode}', path, STATE)
    write(receipts/f'{mode}_state.json', state, new=True)
    write(STATUS, {**status, 'source': head, 'started_utc': now()})
    subprocess.run([PY,'-B',str(CODE/'model_training/tools/agent_work_guard.py'),'begin'],
                   cwd=CODE, check=True, stdout=subprocess.DEVNULL)
    try:
        with log.open('xb') as handle:
            subprocess.run([PY,'-B',str(CODE/'model_training/jobs/launch_virtual_pair.py'),
                '--mode',mode,'--config',str(path),'--output',str(out)], cwd=CODE,
                check=True, stdout=handle, stderr=subprocess.STDOUT)
    finally:
        for file in (root/f'{mode}_resources.json', out/'result.json'):
            if file.is_file():
                shutil.copyfile(file, receipts/f'{mode}_{file.name}' if file.name == 'result.json'
                                else receipts/file.name)
    result, receipt = stage_receipt(path, config, mode, head)
    return result, receipt


def comparison(baseline, completed):
    roots = {'static': baseline['retained_panel_root']}
    for arm, info in completed.items():
        roots[arm] = read(info['path'])['retained_panel_root']
    result = {}
    for samples in POINTS:
        pairs = {arm: panel(Path(root)/f'evaluation_{samples:06d}', samples) for arm, root in roots.items()}
        contrasts = [('static', arm) for arm in ARMS] + [
            ('frame_local', 'temporal_multi'), ('temporal_multi', 'temporal_query')]
        result[str(samples)] = {'metrics': {arm: p[0]['stage1'] for arm,p in pairs.items()},
            'contrasts': {f'{a}_to_{b}': compare(pairs[a][1], pairs[b][1]) for a,b in contrasts}}
    return {'complete': True, 'passed': True, 'stage2_enabled': False, 'reference_views': 128,
        'exposures_per_arm': 30000, 'comparison': result, 'working_model_changed': False}


def run():
    from virtual_pair_guard import deadline_due, require_job_membership, verify_allocation

    require_static_done()
    state = read(STATE)
    followup = state[STATE_KEY]
    code = git('rev-parse', 'HEAD')
    static = followup['static_review']
    if (CODE != Path(REPO) or STATUS.exists() or git('status', '--porcelain')
            or git('config', '--get', 'remote.origin.url') != GIT_URL
            or git('ls-remote', 'origin', GIT_REF).split() != [code, GIT_REF]
            or not state['active_operation']['complete']
            or followup['explicit_human_authorized'] is not True
            or followup['ordered_arms'] != list(ARMS) or followup['completed']
            or static['path'] != str(BASELINE_REVIEW) or digest(BASELINE_REVIEW) != static['sha256']):
        raise RuntimeError('follow-up source/authorization or completed baseline evidence differs')
    baseline = read(BASELINE_REVIEW)
    config = recipe(ARMS[0])
    runtime = config['runtime']
    require_job_membership(Path('/proc/self/cgroup').read_text(), runtime['allocation'])
    verify_allocation(config, subprocess.check_output(['qstat','-j',str(runtime['allocation'])],
                                                     text=True, timeout=30))
    if deadline_due(config):
        raise RuntimeError('native grant reached the save-stop time')
    _, static_panel = panel(Path(baseline['retained_panel_root'])/'evaluation_000000', 0)
    matched_plan = matched_initial = matched_inputs = None
    for arm in ARMS:
        config = recipe(arm)
        path = CODE/'model_training/experiments'/f'{SCOPE}_{arm}.json'
        if path.exists():
            raise RuntimeError('refuse to reuse a generated arm config')
        write(path, config, new=True)
        audit, audited = stage(config, path, 'audit', state, code, {})
        initial, plan = audit['initialization'], audit['plan']
        if (audit['cuda_initialized'] or audit['optimizer_updates'] != 0 or audit['train_assets'] != 5000
                or {r['asset_id']:tuple(r[k] for k in IDENTITY) for r in audit['fixed_validation_inputs']}
                   != identities(static_panel)):
            raise ValueError('CPU audit does not match the fixed static reference panel')
        common_plan = {k:plan[k] for k in ('samples','updates','input_frame_visits','plan_sha256')}
        if matched_plan is not None and (common_plan != matched_plan or initial != matched_initial):
            raise ValueError('dynamic arm initialization or exposure stream changed')
        matched_plan, matched_initial = common_plan, initial
        capacity, bounded = stage(config, path, 'capacity', state, code, {'audit': audited})
        if (capacity['optimizer_updates'] != 0 or capacity['initialization'] != initial
                or len(capacity['profiles_by_rank']) != 2
                or any(len(p) != 6 for p in capacity['profiles_by_rank'])):
            raise ValueError('two-rank no-update capacity evidence is incomplete')
        train, trained = stage(config, path, 'train', state, code, {'audit': audited, 'capacity': bounded})
        if (train['initialization'] != initial or any(train[k] != plan[k] for k in
                ('samples','updates','input_frame_visits','supervised_frame_visits','plan_sha256'))):
            raise ValueError('formal training did not realize the exact audit')
        report = review_training(path, config, trained['source'])
        if (report['ordered_assets_sha256'] != baseline['ordered_assets_sha256']
                or matched_inputs is not None and report['ordered_inputs_sha256'] != matched_inputs):
            raise ValueError('actual asset order or multi-frame inputs changed')
        matched_inputs = report['ordered_inputs_sha256']
        location = Path(config['artifacts']['durable_root'])/'receipts/completion_review.json'
        write(location, report, new=True)
        evidence = {'path': str(location), 'sha256': digest(location)}
        followup['completed'][arm] = evidence
        close(state, f'{SCOPE}_{arm}_complete', evidence)
        write(STATE, state)
        publish(f'Record SkinTokens {arm} completion', STATE)
    result = comparison(baseline, followup['completed'])
    location = STATUS.with_name('comparison.json')
    write(location, result, new=True)
    followup['comparison'] = {'path': str(location), 'sha256': digest(location)}
    close(state, f'{SCOPE}_complete', followup['comparison'])
    state['next_required_result'] = 'Review paired Stage 1 evidence; no model promotion authorized.'
    write(STATE, state)
    head = publish('Record matched SkinTokens Stage 1 comparisons', STATE)
    write(STATUS, {'complete': True, 'passed': True, 'source': head, 'finished_utc': now(),
                   'finished_arms': list(ARMS), 'comparison': followup['comparison']})


def record_failure(exc):
    if not STATUS.exists() or read(STATUS)['complete']:
        return
    current = read(STATUS)
    out = Path(current['output'])
    resource = out.with_name(out.name+'_resources.json')
    state = read(STATE)
    evidence = {'complete': True, 'passed': False, 'error': repr(exc), 'failed_utc': now(),
        'payload_cleanup_confirmed': (read(resource).get('payload_cleanup_complete') is True
                                      if resource.is_file() else current['source'] is None),
        'resources': str(resource), 'latest_full_checkpoint': None}
    for path in sorted(out.glob('checkpoint_*.json'), reverse=True):
        value = read(path)
        if (value.get('both_copies_verified') and value.get('optimizer_and_rank_RNG_retained')
                and Path(value['durable_path']).is_file()):
            evidence['latest_full_checkpoint'] = value
            break
    current.update(evidence)
    write(STATUS, current)
    close(state, f'{SCOPE}_failed', current)
    state['next_required_result'] = 'Review the retained failure; no automatic retry or new allocation.'
    write(STATE, state)
    publish('Record SkinTokens Stage 1 follow-up failure', STATE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('review-static', 'record-static', 'run'))
    parser.add_argument('--review', type=Path)
    parser.add_argument('--sha256')
    args = parser.parse_args()
    if args.action == 'review-static':
        review_static()
    elif args.action == 'record-static':
        record_static(args.review, args.sha256)
    else:
        try:
            run()
        except Exception as error:
            record_failure(error)
            raise
