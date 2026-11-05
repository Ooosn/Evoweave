"""Pin the explicitly authorized zero-update pretrained-interface comparison."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = '/work/wangyy/tokenrig_alignment_20261105'
ENTRY = 'model_training/rigweave/scripts/probe_tokenrig_alignment.py'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-commit', required=True)
    args = parser.parse_args()
    model = ROOT/'model_training'
    path = model/'state/current.json'
    previous = json.loads(path.read_text(encoding='utf-8'))
    if (previous['active_operation']['status'] != 'paused_by_user_recovery_retained'
            or not previous['active_operation']['complete']):
        raise RuntimeError('record and verify the interrupted run before a new operation')
    archive = model/'state/archive/tokenrig_full250k_paused_20261105.json'
    target = model/'experiments/tokenrig_alignment_probe_20261105.json'
    if archive.exists() or target.exists():
        raise RuntimeError('the alignment comparison has already been prepared')
    config = json.loads((model/'experiments/tokenrig_full_20261104.json').read_text())
    config['runtime'].update(repo=RUN+'/code_probe', output_root=RUN+'/outputs_probe',
        extra_pythonpath='/work/wangyy/tokenrig_full_20261104/deps/runtime_skin:/home/wangyy/evorig/runtime/puppeteer_py310_hgc')
    config['tokenrig_decoder']['source'] = RUN+'/deps/SkinTokens'
    config['tokenrig_decoder']['eager'] = True
    config['tokenrig_decoder']['conditioning'] = {'kind': 'released_mesh_projection_v1',
        'source_commit': '273b691d35989d71cd17ff2895fdc735097b92d1'}
    config['backbone']['initialization'] = 'released_TokenRig_mesh_projection_Qwen'
    config['artifacts']['durable_root'] = '/home/wangyy/evorig/outputs/tokenrig_alignment_20261105/probe'
    config['execution'] = {'phase': 'tokenrig_alignment_probe', 'CPU_audit': {
        'path': RUN+'/outputs_probe/audit/result.json',
        'resources_path': RUN+'/outputs_probe/audit_resources.json'}}
    config['alignment_probe'] = {
        'cases': [5, 15, 23, 24, 30, 44, 54, 61, 70, 95, 105, 108], 'frames': [0],
        'variants': ['hybrid_dynamic', 'native_static', 'aligned_static', 'aligned_dynamic'],
        'input_reference_rows': '/work/wangyy/tokenrig_full_20261104/outputs/train/evaluation_250000/stage1_rows.json',
        'normalization': 'unchanged_selected_input_union_mesh_bbox', 'optimizer_updates': 0,
        'scope': '12 fixed validation assets, first frame of the same eight-frame input; diagnostic, not full validation or formal15beam'}
    encoded = (json.dumps(config, indent=2)+'\n').encode()
    digest = hashlib.sha256(encoded).hexdigest()
    archive.write_bytes(path.read_bytes())
    target.write_bytes(encoded)
    state = previous
    state['last_operation'] = previous['active_operation']
    state['active_operation'] = {
        'label': config['family'], 'experiment': 'tokenrig_alignment_probe_20261105',
        'complete': False, 'explicit_human_authorized': True,
        'authorization': '2026-11-05: User paused the previous run and requested a fresh test after fixing pretrained loading/alignment; assess zero-update quality before a new long run.',
        'required_entrypoint': ENTRY, 'operation_entrypoints': {'preflight': ENTRY, 'train': ENTRY},
        'code_commit': args.code_commit, 'gpu_work_allowed': True, 'formal_training_allowed': False,
        'CPU_audit_accepted': False, 'capacity_accepted': False, 'automatic_retry_allowed': False,
        'working_model_unchanged': True, 'optimizer_updates_allowed': 0,
        'resource_policy': 'Retain native131351736, GPU3/4,16CPU; zero-update probe only, no extra allocation.',
        'old_run_resume': state['last_operation']['pause_review'],
        'acceptance_test': 'Strict complete native mesh/projection/Qwen weights; lifted projection parity; same input fingerprints across four routes; report actual initial quality without assuming improvement.',
        'runs': [{'stage': 'alignment_'+mode, 'mode': mode, 'approved': True,
            'config': RUN+'/code_probe/model_training/experiments/'+target.name,
            'config_sha256': digest, 'output': RUN+'/outputs_probe/'+mode} for mode in ('audit', 'capacity')]}
    state.update(state_id='tokenrig_alignment_probe_20261105',
        updated_at=datetime.now(timezone.utc).isoformat(),
        allowed_operations=['inspect', 'report', 'preflight'], blocked_operations=['train', 'submit', 'matched_eval'],
        human_context='model_training/docs/TOKENRIG_ALIGNMENT_20261105.md',
        next_required_result='Audit and execute the pinned no-update interface comparison; inspect actual dynamic quality before admitting a new training run. The old250k recovery and OPT330k remain unchanged.')
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False)+'\n', encoding='utf-8', newline='\n')
    print(digest)


if __name__ == '__main__':
    main()
