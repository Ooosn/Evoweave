"""Fixed final-30k case diagnosis, separate from the rejected shortlist."""
import json
from pathlib import Path

from reference_flow_protocol import sha256

RUN = '/work/wangyy/virtual_query_matched_5k_20261025'
SPEC = {
    'candidate_additional_visits': 30000,
    'posthoc_case_diagnostic': True,
    'promote_model': False,
    'previous_full105_replay_unresolved': True,
    'source_training': {
        'result': {'path': RUN+'/outputs/train_v30/result.json',
            'sha256': 'a38d2fa9114c9d49e3334aefef4356323f70f27d4893be0d09243e4ef8b77e86'},
        'resources': {'path': RUN+'/outputs/train_v30_resources.json',
            'sha256': 'd5e059981376c9c13215619a408f5901a20a499f7aaae68442a79597b419170b'}},
    'baseline': {
        'result': {'path': RUN+'/outputs/capacity_v28/result.json',
            'sha256': 'fa2268f03518276c7aab5a5b545b977b99c9d71a297e721f3407b02bb12dad3a'},
        'resources': {'path': RUN+'/outputs/capacity_v28_resources.json',
            'sha256': '08d32ea7fc2f26ad056a46a71eded720f8ab6a56a2fd6e6b78da0a23cb75a748'}}}


def validate_visual(config):
    if config['hardcase_visual'] != SPEC or 'inference_diagnosis' in config:
        raise ValueError('unreviewed hardcase visual comparison or resumed replay diagnosis')
    runtime = config['runtime']
    if (runtime['allocation'] != 131170166 or runtime['keeper_pid'] != 2268188
            or runtime.get('allocation_directory') != RUN+'/allocation_binding_native_20261031'
            or runtime['stop_before_utc'] != '2026-11-01T11:40:04Z'):
        raise ValueError('hardcase visual allocation differs')


def verify_evidence(config, records):
    from stage1_official_inference import reference_seed
    from stage1_surface_repair_runtime import canonical_identity
    validate_visual(config)
    results = {}
    for label in ('source_training', 'baseline'):
        values = {}
        for name, spec in config['hardcase_visual'][label].items():
            if sha256(spec['path']) != spec['sha256']:
                raise ValueError('retained hardcase visual evidence changed')
            values[name] = json.loads(Path(spec['path']).read_text())
        result, resource = values['result'], values['resources']
        if (not result['complete'] or not result['passed'] or not resource['complete']
                or not resource['passed'] or resource['exit_code'] != 0
                or not resource['payload_cleanup_complete']):
            raise ValueError('source inference/training did not complete cleanly')
        results[label] = result
    training, baseline = results['source_training'], results['baseline']
    if training['samples'] != 30000 or training['optimizer_updates'] != 628 or baseline['optimizer_updates'] != 0:
        raise ValueError('wrong final endpoint or nonzero baseline updates')
    policy = baseline['policy']
    if any(policy[k] != config['official_smoke'][k] for k in ('case_indices', 'generation_kwargs', 'base_seed')):
        raise ValueError('baseline official policy differs')
    references = {}
    for artifact in baseline['artifacts']:
        spec = artifact['references']
        if sha256(spec['path']) != spec['sha256']:
            raise ValueError('baseline reference artifact changed')
        references.update(json.loads(Path(spec['path']).read_text()))
    if set(references) != {r['asset_id'] for r in records}:
        raise ValueError('baseline fixed cases differ')
    for record in records:
        row = references[record['asset_id']]
        if any(row.get(k) != value for k, value in canonical_identity(record, 0).items()):
            raise ValueError('baseline canonical mesh/GT/frame/normalization differs')
        if row['seed'] != reference_seed(record['asset_id'], record['raw_frames'][0], policy['base_seed']):
            raise ValueError('baseline random seed differs')
    return {'assets': len(records), 'references_valid': sum(r['success'] for r in references.values()),
        'same_inputs_and_official_seed': True, 'baseline_recomputed': False,
        'full105_replay_unresolved': True, 'model_promotion_allowed': False}
