"""Record completed alignment evidence and prepare a separate fresh native run."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT/'model_training'
RUN = '/work/wangyy/tokenrig_alignment_20261105'
DURABLE = '/home/wangyy/evorig/outputs/tokenrig_alignment_20261105'
ENTRY = 'model_training/rigweave/scripts/train_anchor_full.py'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')


def archive_json(archive, name):
    data = archive.extractfile(name).read()
    return json.loads(data), digest(data)


def record_probe(state, receipt):
    op = state['active_operation']
    if op['experiment'] != 'tokenrig_alignment_probe_20261105' or op['complete']:
        raise RuntimeError('the completed identity probe must still be the active operation')
    with tarfile.open(receipt) as archive:
        result, result_sha = archive_json(archive, 'outputs_probe_r5/capacity/result.json')
        resources, _ = archive_json(archive, 'outputs_probe_r5/capacity_resources.json')
        rows, rows_sha = archive_json(archive, 'outputs_probe_r5/capacity/rows.json')
    if (result.get('complete') is not True or result.get('passed') is not True
            or result['optimizer_updates'] != 0 or result['matched_input_views'] != 12
            or resources.get('passed') is not True or resources.get('exit_code') != 0
            or resources.get('payload_cleanup_complete') is not True or len(rows) != 36):
        raise RuntimeError('the identity probe has not completed cleanly')
    static = {r['asset_id']: r for r in rows if r['variant'] == 'aligned_static'}
    dynamic = {r['asset_id']: r for r in rows if r['variant'] == 'aligned_dynamic'}
    if static.keys() != dynamic.keys() or len(static) != 12:
        raise RuntimeError('the two routes do not cover the same twelve inputs')
    # Compare all recorded fields except route labels and timing.
    ignored = {'variant', 'seconds', 'condition_tokens'}
    for asset in static:
        a = {k: v for k, v in static[asset].items() if k not in ignored}
        b = {k: v for k, v in dynamic[asset].items() if k not in ignored}
        if a != b:
            raise RuntimeError('initial dynamic predictions differ from the aligned static route')
    evidence = {'source': result['source'], 'path': RUN+'/outputs_probe_r5/capacity/result.json',
        'sha256': result_sha, 'rows_sha256': rows_sha,
        'durable_archive': DURABLE+'/probe_r5_receipts.tar.gz',
        'archive_sha256': digest(receipt.read_bytes()), 'cases': 12, 'views_per_case': 1,
        'precision': result['precision'], 'decoding': result['stage1_decoding'],
        'optimizer_updates': 0, 'same_input_fingerprints_verified': True,
        'static_dynamic_rows_identical': True, 'summaries': result['summaries'],
        'seconds': result['seconds'], 'not_full_validation': True, 'stage2_evaluated': False}
    op.update(complete=True, status='zero_update_alignment_verified', review=evidence,
        payload_cleanup_complete=True)
    state.setdefault('alignment_evidence', {})['identity_native512_probe'] = evidence
    state['next_required_result'] = 'Run independent native joint-update capacity and exact full recovery checks before fresh training.'
    save(MODEL/'state/archive/tokenrig_identity_probe_20261105.json', state)


def prepare_capacity(state, code_commit):
    old = state['active_operation']
    if old.get('status') != 'zero_update_alignment_verified' or not old['complete']:
        raise RuntimeError('record the actual zero-update result before a new operation')
    path = MODEL/'experiments/tokenrig_native_capacity_20261105.json'
    if path.exists():
        raise RuntimeError('native capacity has already been prepared')
    config = json.loads((MODEL/'experiments/tokenrig_full_20261104.json').read_text())
    probe = json.loads((MODEL/'experiments/tokenrig_alignment_probe_20261105_r5.json').read_text())
    config['tokenrig_decoder'] = probe['tokenrig_decoder']
    config['tokenrig_decoder'].pop('eager', None)
    config['backbone']['initialization'] = 'released_TokenRig_mesh_projection_Qwen'
    config['runtime'].update(repo=RUN+'/code_native_capacity', output_root=RUN+'/outputs_native_capacity')
    config['artifacts']['durable_root'] = DURABLE+'/native_full'
    config['execution'] = {'phase': 'tokenrig_native_full300k', 'CPU_audit': {
        'path': RUN+'/outputs_native_capacity/audit/result.json',
        'resources_path': RUN+'/outputs_native_capacity/audit_resources.json'}, 'capacity': None}
    save(path, config)
    config_sha = digest(path.read_bytes())
    state['last_completed_alignment_operation'] = old
    state['active_operation'] = {'label': config['family'], 'experiment': 'tokenrig_native_full300k_20261105',
        'complete': False, 'explicit_human_authorized': True,
        'authorization': '2026-11-05 user authorized pausing the hybrid run, fixing pretrained alignment, and starting fresh if initial quality recovers. Zero-update native512 dynamic12-view edgeF1 is0.7798844100010204.',
        'required_entrypoint': ENTRY, 'operation_entrypoints': {'preflight': ENTRY, 'train': ENTRY},
        'code_commit': code_commit, 'formal_training_allowed': False, 'gpu_work_allowed': True,
        'CPU_audit_accepted': False, 'capacity_accepted': False, 'automatic_retry_allowed': False,
        'working_model_unchanged': True, 'old_run_resume': state['last_operation']['pause_review'],
        'resource_policy': 'Native131351736;GPU3/4;16CPU;78GiB whole-device limit; retain allocation and unrelated tasks.',
        'acceptance_test': 'Independent extreme joint updates at6/12/24frames, FP32 cached-prefix parity, exact model/Adam/both-rank RNG/sampler disk recovery, durable backup; no diagnostic or stress weights in formal initialization.',
        'runs': [{'stage': 'native_'+mode, 'mode': mode, 'approved': True,
            'config': config['runtime']['repo']+'/model_training/experiments/'+path.name,
            'config_sha256': config_sha, 'output': config['runtime']['output_root']+'/'+mode}
            for mode in ('audit', 'capacity')]}
    state.update(state_id='tokenrig_native_full300k_20261105',
        allowed_operations=['inspect', 'report', 'preflight'], blocked_operations=['train', 'submit', 'matched_eval'],
        human_context='model_training/docs/TOKENRIG_ALIGNMENT_20261105.md',
        next_required_result='Complete the independent native CPU/capacity stage; review finite gradients and exact recovery before admitting a fresh300k run.')
    print('capacity_config_sha256', config_sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('record-probe', 'prepare-capacity'))
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--code-commit')
    args = parser.parse_args()
    path = MODEL/'state/current.json'
    state = json.loads(path.read_text(encoding='utf-8'))
    if args.mode == 'record-probe':
        record_probe(state, args.receipt)
    else:
        if not args.code_commit or len(args.code_commit) != 40:
            raise ValueError('prepare needs an exact implementation commit')
        prepare_capacity(state, args.code_commit)
    state['updated_at'] = datetime.now(timezone.utc).isoformat()
    save(path, state)


if __name__ == '__main__':
    main()
