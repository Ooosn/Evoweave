"""Record the user-approved Qwen full run without modifying the OPT recipe."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ENTRY = 'model_training/rigweave/scripts/train_anchor_full.py'
RUN = '/work/wangyy/tokenrig_full_20261104'
HOME = '/home/wangyy'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-commit', required=True)
    parser.add_argument('--restage-before-launch', action='store_true')
    args = parser.parse_args()
    model = ROOT/'model_training'
    state_path = model/'state/current.json'
    config_path = model/'experiments/tokenrig_full_20261104.json'
    archive = model/'state/archive/current_before_tokenrig_full_20261104.json'
    if args.restage_before_launch:
        if json.loads(state_path.read_bytes())['state_id'] != 'tokenrig_full300k_20261104':
            raise RuntimeError('restaging applies only to this planned operation')
        old_state = archive.read_bytes()
    elif config_path.exists() or archive.exists():
        raise RuntimeError('TokenRig operation has already been prepared')
    else:
        old_state = state_path.read_bytes()
    state = json.loads(old_state)
    if not state['active_operation']['complete']:
        raise RuntimeError('another model operation is still active')
    config = json.loads((model/'experiments/anchor_full_20261030_v1.json').read_text())
    config['tokenrig_decoder'] = json.loads((model/'experiments/tokenrig_decoder_20261104.json').read_text())
    for key in ('source', 'checkpoint', 'qwen_config_root'):
        config['tokenrig_decoder'][key] = config['tokenrig_decoder'][key].replace(
            '/work/wangyy/paper_baselines_20261102', RUN+'/deps')
    config['runtime'].update(repo=RUN+'/code', allocation=131351736, hostname='pcg01i',
        output_root=RUN+'/outputs', stop_before_utc='2026-11-07T16:51:44Z',
        cpu_affinity=[6,7,9,10,11,12,14,15,61,63,64,65,66,67,68,69],
        gpu_memory_fraction_limit=79872/81559)
    config['runtime'].pop('keeper_pid')
    config['runtime']['extra_pythonpath'] = (RUN+'/deps/runtime_skin:'
        + config['runtime']['extra_pythonpath'])
    config['data_cache'].update(root=RUN+'/data', index=RUN+'/data/index.json',
        reflink_source='/work/wangyy/anchor_full_20261030_data')
    config['training']['frame_policy'] = {str(t): 24 if t <= 12 else 12 for t in range(2, 25)}
    config['artifacts']['durable_root'] = HOME+'/evorig/outputs/tokenrig_full_20261104'
    config['execution'] = {
        'phase': 'tokenrig_full300k',
        'CPU_audit': {'path': RUN+'/outputs/audit/result.json', 'resources_path': RUN+'/outputs/audit_resources.json'},
        'tokenrig_capacity_archives': [
            {'path': HOME+'/evoweave-tokenrig-8a6f9f3-capacity-receipts.tar.gz',
             'sha256': '7b882ee3a15c9dbf538838b85e96dfff8ffe39cb1d8f9b838985bdd6f0a95bef',
             'profiles': ['default']},
            {'path': HOME+'/evoweave-tokenrig-01d5b17-batch-receipts.tar.gz',
             'sha256': '1a543c00a57d50a12a21f8666c578a00aaa728b0ef36afe2eb8de29aa1841a76',
             'profiles': ['frames12', 'frames24']}]}
    encoded = (json.dumps(config, indent=2)+'\n').encode()
    config_sha = hashlib.sha256(encoded).hexdigest()
    config_remote = RUN+'/code/model_training/experiments/'+config_path.name
    state.update(state_id='tokenrig_full300k_20261104', updated_at=datetime.now(timezone.utc).isoformat(),
        allowed_operations=['inspect','report','preflight','train'], blocked_operations=['submit','matched_eval'],
        human_context='model_training/docs/TOKENRIG_FULL_20261104.md',
        next_required_result='Pass guarded CPU audit, then verify actual TokenRig full training startup and monitor to 300000 visits.')
    base = json.loads((model/'experiments/anchor_full_backbone_20261030_v1.json').read_text())
    state['active_operation'] = {
        'label': config['family'], 'experiment': 'tokenrig_full300k_20261104',
        'complete': False, 'explicit_human_authorized': True,
        'authorization': '2026-11-04: User authorized fresh TokenRig joint training to 300000 visits at unchanged learning rates and tested enlarged physical batches.',
        'required_entrypoint': ENTRY, 'operation_entrypoints': {'preflight': ENTRY, 'train': ENTRY},
        'code_commit': args.code_commit, 'formal_training_allowed': True, 'gpu_work_allowed': True,
        'CPU_audit_accepted': False, 'live_tokenrig_audit_required': True, 'capacity_accepted': True,
        'automatic_retry_allowed': False, 'working_model_unchanged': True,
        'manifest': base['data']['manifests'], 'mesh_initialization': base['initialization'],
        'decoder_initialization': config['tokenrig_decoder'],
        'capacity_archives': config['execution']['tokenrig_capacity_archives'],
        'resource_policy': 'Native HGC 131351736, GPU3/4, 16 CPUs, 78 GiB sampled whole-device ceiling. Preserve allocation and unrelated tasks.',
        'acceptance_test': 'Exact CPU audit required before GPU launch; strict pretrained weights, 300000 visits, finite joint updates and backed-up recovery points.',
        'runs': [{'stage': 'tokenrig_full_'+mode, 'mode': mode, 'approved': True,
                  'config': config_remote, 'config_sha256': config_sha, 'output': RUN+'/outputs/'+mode}
                 for mode in ('audit', 'train')]}
    if not args.restage_before_launch:
        archive.write_bytes(old_state)
    config_path.write_bytes(encoded)
    state_path.write_bytes((json.dumps(state, indent=2, ensure_ascii=False)+'\n').encode())
    print(config_sha)


if __name__ == '__main__':
    main()
