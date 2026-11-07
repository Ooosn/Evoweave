"""Verify a finished continuation and freeze exactly two new evaluation candidates."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path('/work/wangyy/tokenrig_alignment_20261105')
CODE = ROOT/'code_native_cont400k_20261107'
OUT = ROOT/'outputs_native_cont400k_20261107'
DURABLE = Path('/home/wangyy/evorig/outputs/tokenrig_alignment_20261105/native_cont400k_20261107')


def read(path):
    return json.loads(Path(path).read_text())


def checked_pair(spec, *, fresh=False):
    from tokenrig_comparison_matrix import digest
    for key in ('path','durable_path'):
        path = Path(spec[key])
        if not path.is_file() or path.is_symlink() or path.stat().st_size != spec['bytes']:
            raise ValueError('a complete checkpoint copy is missing or changed: '+str(path))
        if fresh and digest(path) != spec['sha256']:
            raise ValueError('checkpoint SHA256 differs: '+str(path))


def target_identity(rows):
    keys = {}
    for row in rows:
        if row['split'] != 'valid' or row['method'] != 'head':
            continue
        key = row['asset_id'],row['reference_slot'],row['target_slot']
        if row['reference_kind'] != 'GT' or key in keys:
            raise ValueError('GT-reference target pairing differs')
        keys[key] = tuple(row[k] for k in ('reference_raw_frame','target_raw_frame','surface_sha256','joint_count'))
    if len(keys) != 1792:
        raise ValueError('development GT pair denominator differs')
    return keys


def archive_receipts(paths):
    from tokenrig_comparison_matrix import digest
    archive = DURABLE/'receipts/training_completion.tar.gz'
    with tarfile.open(archive, 'x:gz') as bundle:
        for name,path in paths.items():
            bundle.add(path, arcname=name, recursive=False)
    with tarfile.open(archive, 'r:gz') as bundle:
        for name,path in paths.items():
            if hashlib.sha256(bundle.extractfile(name).read()).hexdigest() != digest(path):
                raise ValueError('durable training evidence archive changed')
    return {'path':str(archive),'sha256':digest(archive),'bytes':archive.stat().st_size}


def main():
    allowed = os.sched_getaffinity(0) & {6,7,9,10,11,12,14,15,61,63,64,65,66,67,68,69}
    if not allowed:
        raise RuntimeError('no permitted CPU for the bounded review')
    os.sched_setaffinity(0,{min(allowed)})
    os.nice(19)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
                'RIGWEAVE_DISABLE_OPEN3D','RIGWEAVE_DISABLE_LIGHTNING_IMPORT'):
        os.environ[key] = '1'
    sys.path[:0] = [str(CODE/'model_training/rigweave/src'),str(CODE/'model_training/rigweave/scripts')]
    import torch
    from anchor_full_data import make_stream
    from anchor_full_protocol import scientific_sha, source_hashes
    from reference_flow_runtime import read_all
    from tokenrig_comparison_matrix import balanced,digest
    from tokenrig_native_continuation import (MILESTONES,PARENT_ASSETS,PARENT_FRAMES,PARENT_STEPS,
        PARENT_ROOT,checked_json,verify_audit,verify_panel)
    from virtual_pair_runtime import next_update_batches
    torch.set_num_threads(1)
    config_path = CODE/'model_training/experiments/tokenrig_native_cont400k_20261107.json'
    config = read(config_path)
    audit = verify_audit(config)
    res = read(OUT/'train_resources.json')
    result = [read(OUT/'train'/f'result_rank_{rank}.json') for rank in (0,1)]
    source = subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip()
    sources = source_hashes()
    scisha = scientific_sha(config)
    runtime = config['runtime']
    if (subprocess.check_output(['git','-C',str(CODE),'status','--porcelain'],text=True).strip()
            or not res['complete'] or not res['passed'] or res['exit_code'] != 0 or res['source'] != source
            or res['allocation'] != runtime['allocation'] or not res['kernel_scheduler_membership']
            or res['cpu_affinity'] != runtime['cpu_affinity']
            or res['cuda_visible_devices'] != ','.join(runtime['gpu_uuids'])
            or not res['payload_cleanup_complete'] or set(res['peak_gpu_memory_mib']) != {'3','4'}
            or any(not 0 < v <= 78*1024 for v in res['peak_gpu_memory_mib'].values())):
        raise ValueError('native continuation failed source/resource/cleanup admission')
    base,entries = read_all(config)
    trace_hashes = []
    boundary_steps = {}
    evidence = {'train_resources.json': OUT/'train_resources.json',
        'config.json': config_path, 'audit.json': OUT/'audit/result.json',
        'audit_resources.json': OUT/'audit_resources.json',
        'train.log': DURABLE/'receipts/train.log'}
    for rank,saved in enumerate(result):
        if (not saved['complete'] or not saved['passed'] or saved['stopped_for_deadline']
                or saved['samples'] != 400000 or saved['optimizer_updates'] != 8360
                or saved['input_frame_visits'] != audit['plan']['input_frame_visits']
                or saved['source'] != source or saved['source_sha256'] != sources
                or saved['scientific_sha256'] != scisha
                or saved['initialization'] != audit['initialization']):
            raise ValueError('rank completion or exact restored optimizer identity differs')
        for kind in ('restoration','result'):
            name = f'{kind}_rank_{rank}.json'
            evidence[name] = OUT/'train'/name
        restore = read(evidence[f'restoration_rank_{rank}.json'])
        if (not restore['rank_RNG_restored'] or restore['rank'] != rank
                or restore['optimizer_sha256'] != audit['initialization']['optimizer_sha256']
                or restore['model_sha256'] != audit['initialization']['model_sha256']):
            raise ValueError('actual rank restoration differs from the CPU audit')
        stream = make_stream(base,rank,2,restore['sampler'])
        trace = OUT/'train'/f'updates_rank_{rank}.jsonl'
        evidence[trace.name] = trace
        samples,frames,step = PARENT_ASSETS,PARENT_FRAMES,PARENT_STEPS
        with trace.open() as handle:
            for line in handle:
                row = json.loads(line)
                goal,batches = next_update_batches(stream,samples,2)
                samples += goal
                frames += goal*batches[0][0].frames
                step += 1
                if samples in MILESTONES:
                    if boundary_steps.setdefault(samples,step) != step:
                        raise ValueError('ranks disagree on checkpoint update')
                if ((row['step'],row['additional_asset_exposures'],row['input_frame_exposures'],
                     row['assets_in_update'],row['code_commit'],row['global_backbone_asset_exposures'],
                     row['frames_in_update'],row['learning_rate_asset_position'])
                        != (step,samples,frames,goal,source,samples,goal*batches[0][0].frames,PARENT_ASSETS)
                        or len(row['microbatches']) != len(batches)
                        or not all(math.isfinite(row[k]) for k in ('ar_loss','xyz_loss','bone_loss','gradient_norm'))
                        or not all(math.isclose(row[k],v,rel_tol=0,abs_tol=1e-15)
                            for k,v in (('lr',2e-6),('head_lr',6e-6),('beta1',.95)))):
                    raise ValueError('actual training trace differs from the fixed continuation')
                for actual,planned in zip(row['microbatches'],batches):
                    if (actual['manifest_indices'] != [r.item_index for r in planned]
                            or actual['asset_ids'] != [entries['train'][r.item_index]['asset_id'] for r in planned]
                            or actual['sampling_seeds'] != [r.sampling_seed for r in planned]
                            or actual['frames'] != planned[0].frames):
                        raise ValueError('an actual asset, frame count, or seed differs from the suffix plan')
        if (samples,frames,step) != (400000,audit['plan']['input_frame_visits'],8360):
            raise ValueError('actual final continuation counters differ')
        trace_hashes.append(digest(trace))
    parent_directory = Path(PARENT_ROOT)/'evaluation_300000'
    parent_summary = checked_json({'path':str(parent_directory/'summary.json'),
        'sha256':'724aecd1989d19a88d20685acea83a4567b8d01b3654971504ab69996b145bf7'})
    parent_rows = read(parent_directory/'rows.json')
    parent_ar = checked_json(config['native_continuation']['parent_stage1_rows'])
    target_keys = target_identity(parent_rows)
    parent_head = [r for r in parent_rows if r['split']=='valid' and r['method']=='head']
    parent_dev = {'visits':PARENT_ASSETS, 'stage1_edge_F1':balanced(parent_ar,'original_edge_f1'),
        'GT_MPJPE':balanced(parent_head,'normalized_mpjpe'), 'GT_edge_F1':balanced(parent_head,'edge_f1')}
    parent_group = next(g for g in parent_summary['groups'] if g['split']=='valid' and g['method']=='head')
    if any(abs(parent_dev[k]-v) > 1e-12 for k,v in (
            ('stage1_edge_F1',parent_summary['stage1']['original_edge_f1']),
            ('GT_MPJPE',parent_group['normalized_mpjpe']),('GT_edge_F1',parent_group['edge_f1']))):
        raise ValueError('retained 300k development metrics differ')
    curve = []
    for n in MILESTONES:
        directory = OUT/'train'/f'evaluation_{n:06d}'
        for name in ('summary.json','stage1_rows.json','rows.json','examples_rank_0.pt','examples_rank_1.pt'):
            relative = directory.name+'/'+name
            if digest(directory/name) != digest(DURABLE/relative):
                raise ValueError('durable development evaluation differs: '+relative)
        summary,ar,rows = [read(directory/name) for name in ('summary.json','stage1_rows.json','rows.json')]
        verify_panel(config,OUT/'train',n)
        if (not summary['complete'] or summary['samples'] != n or summary['stage2_reference'] != 'GT_only'
                or summary['scientific_sha256'] != scisha
                or summary['stage1_decoding'] != 'greedy_fast_validation_not_official_beam_inference'
                or summary['stage1_GT_reference_or_teacher_prefix']
                or summary['precision'] != 'fp32_noTF32' or summary['decoder_attention'] != 'eager'
                or target_identity(rows) != target_keys
                or any(not r['success'] and r['original_edge_f1'] != 0 for r in ar)):
            raise ValueError('matched development protocol did not pass')
        heads = [r for r in rows if r['split']=='valid' and r['method']=='head']
        f1,mpjpe,gt_f1 = balanced(ar,'original_edge_f1'),balanced(heads,'normalized_mpjpe'),balanced(heads,'edge_f1')
        group = next(g for g in summary['groups'] if g['split']=='valid' and g['method']=='head')
        if (abs(f1-summary['stage1']['original_edge_f1']) > 1e-12
                or abs(mpjpe-group['normalized_mpjpe']) > 1e-12 or abs(gt_f1-group['edge_f1']) > 1e-12):
            raise ValueError('independent reaggregation differs')
        spec = read(OUT/'train'/f'checkpoint_{n:06d}.json')
        durable_spec = read(DURABLE/f'checkpoint_{n:06d}.json')
        if (spec != durable_spec or spec['samples'] != n or not spec['complete']
                or spec['scientific_sha256'] != scisha
                or spec['step'] != boundary_steps[n]
                or Path(spec['path']) != OUT/'train'/f'checkpoint_{n:06d}.pt'
                or Path(spec['durable_path']) != DURABLE/f'checkpoint_{n:06d}.pt'
                or not spec['both_copies_verified'] or not spec['optimizer_and_rank_RNG_retained']
                or spec['code_commit'] != source):
            raise ValueError('a complete verified recovery receipt differs')
        checked_pair(spec)
        evidence[f'checkpoint_{n:06d}.json'] = directory.parent/f'checkpoint_{n:06d}.json'
        curve.append({'visits':n,'stage1_edge_F1':f1,'GT_MPJPE':mpjpe,'GT_edge_F1':gt_f1,
            'checkpoint':spec,'development_summary_sha256':digest(directory/'summary.json')})
    ranked = sorted(curve,key=lambda r:(-r['stage1_edge_F1'],r['GT_MPJPE'],r['visits']))
    for n in sorted({400000,*[r['visits'] for r in ranked[:2]]}):
        checked_pair(next(r['checkpoint'] for r in curve if r['visits']==n),fresh=True)
    candidates = [{'label':f"native{r['visits']//1000}k",'visits':r['visits'],
        'path':r['checkpoint']['path'],'sha256':r['checkpoint']['sha256']} for r in ranked[:2]]
    review = {'complete':True,'passed':True,'source':source,'config_sha256':digest(config_path),
        'resources_sha256':digest(OUT/'train_resources.json'),'rank_trace_sha256':trace_hashes,
        'assets':400000,'new_asset_visits':100000,'optimizer_updates':8360,'new_optimizer_updates':2090,
        'input_frame_visits':audit['plan']['input_frame_visits'],'all_logged_losses_gradients_finite':True,
        'all_actual_indices_assets_and_sampling_seeds_match':True,'native_payload_cleanup_complete':True,
        'all_ten_recoveries_retained':True,'development_assets':128,'development_stage1_views':1024,
        'development_GT_pairs':1792,'same_parent_panel_input_and_GT':True,
        'parent_development':parent_dev,'curve':curve,
        'rule':'descending_greedy_F1_then_lower_GT_MPJPE_then_earlier_visits',
        'selected_candidates':candidates,'formal_results_not_used_for_selection':True,
        'peak_gpu_memory_mib':res['peak_gpu_memory_mib'],'working_model_unchanged':True,
        'quality_accepted':False,'CUDA_used':torch.cuda.is_initialized()}
    if review['CUDA_used']:
        raise ValueError('completion review must not initialize CUDA')
    review['durable_training_receipts'] = archive_receipts(evidence)
    data = (json.dumps(review,indent=2)+'\n').encode()
    for path in (OUT/'screening.json',DURABLE/'screening.json'):
        with path.open('xb') as handle:
            handle.write(data)
    print(json.dumps({'screening_sha256':hashlib.sha256(data).hexdigest(),
        'selected_candidates':candidates,'curve':[{k:v for k,v in r.items() if k!='checkpoint'} for r in curve]}))


if __name__ == '__main__':
    main()
