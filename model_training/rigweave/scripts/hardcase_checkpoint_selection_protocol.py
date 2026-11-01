"""Two frozen intermediate checkpoints; same official inputs and no new optimization."""
import json,math
from pathlib import Path
from reference_flow_protocol import sha256
from hardcase_visual_protocol import RUN,inherited_native_capacity,verify_stage_deployment

MANIFEST='model_training/selections/hardcase_330k400k_20261102.json'
MANIFEST_SHA='34653d80c4e0bd138b72bebf2f01df8d2018afb8bd26198589a9d23a821de151'

def selection_manifest(config):
    from predicted_reference_protocol import bound_config
    spec=config['checkpoint_selection']
    if (set(spec)!={'kind','total_visits','manifest'} or spec['kind']!='hardcase_checkpoint_selection_20261102'
            or spec['total_visits'] not in (380000,390000)
            or spec['manifest']!={'relative_path':MANIFEST,'sha256':MANIFEST_SHA}):
        raise ValueError('checkpoint selection was not frozen before inference')
    return bound_config(spec['manifest'])

def validate_selection(config):
    manifest=selection_manifest(config); runtime=config['runtime']; total=config['checkpoint_selection']['total_visits']
    expected=manifest['checkpoint_candidates'][str(total)]
    if ('hardcase_visual' in config or 'inference_diagnosis' in config
            or config['checkpoints']!={'stage1':expected,'stage2':expected}
            or config['source_deployment']!={'receipt':RUN+'/source_deployments/hardcase_checkpoint_selection_20261102.json'}
            or config['official_smoke'].get('inherited_official_capacity')!=manifest['baseline330k']
            or runtime['allocation']!=131170166 or runtime['keeper_pid']!=2268188
            or runtime.get('allocation_directory')!=RUN+'/allocation_binding_native_20261031'
            or runtime['required_h_rt_seconds']!=345600 or runtime['stop_before_utc']!='2026-11-04T11:40:04Z'):
        raise ValueError('selection checkpoint, source or retained native grant differs')
    return expected

def retained_pair(spec):
    values={}
    for name,item in spec.items():
        if sha256(item['path'])!=item['sha256']: raise ValueError('frozen selection evidence changed')
        values[name]=json.loads(Path(item['path']).read_text())
    result,resource=values['result'],values['resources']
    if not (result['complete'] and result['passed'] and resource['complete'] and resource['passed']
            and resource['exit_code']==0 and resource['payload_cleanup_complete']):
        raise ValueError('retained selection source did not finish cleanly')
    return result

def verify_selection_evidence(config,records):
    from stage1_official_inference import reference_seed
    from stage1_surface_repair_runtime import canonical_identity
    validate_selection(config); manifest=selection_manifest(config)
    for label in ('valid_receipt','development_receipt'):
        item=manifest[label]
        if sha256(item['path'])!=item['sha256']: raise ValueError('existing matched validation changed')
    valid=json.loads(Path(manifest['valid_receipt']['path']).read_text())
    if not valid['complete'] or not valid['development_stage1_and_GT_stage2_inputs_identical']:
        raise ValueError('validation inputs were not matched')
    training=retained_pair(manifest['source_training'])
    if training['samples']!=100000 or training['optimizer_updates']!=2091: raise ValueError('continuation source differs')
    counts={}
    for label in ('baseline300k','baseline330k','already_reviewed400k'):
        result=retained_pair(manifest[label]); policy=config['official_smoke']
        if result['optimizer_updates']!=0 or any(result['policy'][k]!=policy[k] for k in ('case_indices','generation_kwargs','base_seed')):
            raise ValueError('retained official inference policy differs')
        refs={}
        for artifact in result['artifacts']:
            item=artifact['references']
            if sha256(item['path'])!=item['sha256']: raise ValueError('retained canonical references changed')
            refs.update(json.loads(Path(item['path']).read_text()))
        if set(refs)!={r['asset_id'] for r in records}: raise ValueError('fixed12cases differ')
        for record in records:
            row=refs[record['asset_id']]
            if any(row.get(k)!=v for k,v in canonical_identity(record,0).items()) or row['seed']!=reference_seed(record['asset_id'],record['raw_frames'][0],20261029):
                raise ValueError('canonical input, GT, normalization or seed differs')
        counts[label]=sum(row['success'] for row in refs.values())
    return {'assets':12,'same_inputs_and_official_seed':True,'references_valid':counts['baseline300k'],
        'previous330k_references_valid':counts['baseline330k'],'retained400k_references_valid':counts['already_reviewed400k'],
        'candidate_total_visits':config['checkpoint_selection']['total_visits'],'baseline_recomputed':False,
        'previous330k_recomputed':False,'retained400k_recomputed':False,
        'candidates_selected_only_from_existing_valid':True,'full105_replay_unresolved':True,
        'population_quality_accepted':False}

def inherited_selection_capacity(config):
    validate_selection(config)
    return inherited_native_capacity(config,selection_manifest(config)['baseline330k'],verify_stage_deployment(config))

def assess_candidate(baseline,candidate,case_j2j_losses,reference_valid,background_loss,rule):
    """Apply the predeclared practical-benefit rule, never tune it after results."""
    f1_gain=candidate['edge_f1']-baseline['edge_f1']
    ratio=candidate['normalized_j2j_successful']/baseline['normalized_j2j_successful'] if candidate['normalized_j2j_successful'] is not None else math.inf
    f1_route=f1_gain>=rule['min_target_edge_f1_gain'] and ratio<=1+rule['max_relative_j2j_worsening_with_f1_gain']
    geometry_route=ratio<=1-rule['min_relative_j2j_gain'] and f1_gain>=-rule['max_target_edge_f1_loss_with_j2j_gain']
    gates={'no_new_failures':reference_valid==12 and candidate['failures']==0,
        'complete_case_geometry':len(case_j2j_losses)==12 and all(v is not None and math.isfinite(v) for v in case_j2j_losses),
        'no_large_case_regression':all(v is not None and v<=rule['max_single_case_absolute_j2j_loss'] for v in case_j2j_losses),
        'background_valid_preserved':background_loss<=rule['max_valid_background_f1_loss'],
        'meaningful_paired_gain':f1_route or geometry_route}
    return {'qualifies':all(gates.values()),'gates':gates,'target_edge_f1_gain':f1_gain,
        'relative_j2j_change':ratio-1 if math.isfinite(ratio) else None,
        'F1_gain_route':f1_route,'geometry_gain_route':geometry_route}
