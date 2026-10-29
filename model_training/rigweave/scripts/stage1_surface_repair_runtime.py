"""Reuse110 exact-input assets; evaluate18 assets on one immutable tensor cache."""
import copy
import gc
import json
from pathlib import Path
import subprocess
import time

import torch
import torch.distributed as dist

from ar_only_runtime import analyze_evaluations, load_evaluation_model
from fusion_evaluation import tree_fingerprint
from reference_flow_protocol import load_backbone_config
from reference_flow_runtime import mirror, read_all
from stage1_feedback import backbone_state, generate_asset, score_generation, summarize_rows
from stage1_surface_repair_protocol import (LABELS, PAIR_FIELDS, ROOT, merge_rows, parent_config,
    plan_inputs, scientific_sha, sha256)
from virtual_pair_runtime import evaluation_rows, gather, shared_deadline_due
from virtual_query_xyz_runtime import fixed_dataset, write_json
from rigweave.dynamic_rig.virtual_frame_data import prepare_virtual_surface


def read_previous(config):
    summaries,rows = {},{}
    for label in LABELS:
        spec = config['prior_evaluations'][label]
        for item in (spec['summary'],*spec['rows'],*spec['examples']):
            if sha256(item['path']) != item['sha256']:
                raise ValueError('retained parent evaluation changed')
        summaries[label] = json.loads(Path(spec['summary']['path']).read_text())
        if (not summaries[label]['complete'] or not summaries[label]['no_GT_reference_or_teacher_prefix']
                or summaries[label]['identity']['checkpoint_sha256'] != config['checkpoints'][label]['sha256']):
            raise ValueError('parent evaluation checkpoint identity differs')
        rows[label] = []
        for rank,item in enumerate(spec['rows']):
            if item['sha256'] != summaries[label]['row_artifacts'][rank]['sha256']:
                raise ValueError('parent summary row binding differs')
            rows[label].extend(json.loads(line) for line in Path(item['path']).read_text().splitlines())
        if summarize_rows(rows[label]) != summaries[label]['summary']:
            raise ValueError('parent summary recomputation differs')
    plan = plan_inputs(summaries,rows)
    if plan['affected_asset_ids'] != config['affected_asset_ids'] or len(plan['retained_asset_ids']) != 110:
        raise ValueError('repair asset scope changed')
    return summaries,rows,plan


def audit(config, output, source):
    summaries,rows,plan = read_previous(config)
    failure = {}
    for name,spec in config['parent_failure'].items():
        if sha256(spec['path']) != spec['sha256']:
            raise ValueError('original failure evidence changed')
        failure[name] = json.loads(Path(spec['path']).read_text())
    if (not failure['resources']['complete'] or failure['resources']['exit_code'] != 1
            or not failure['resources']['payload_cleanup_complete'] or failure['resources'].get('error')
            or failure['resources'].get('cleaned_payload_pids')
            or failure['failure']['error'] != "ValueError('paired inputs differ: surface_sha256')"):
        raise ValueError('parent failed outside the bounded final input gate')
    original = parent_config(config)
    base,_ = read_all(original)
    for spec in config['checkpoints'].values():
        if sha256(spec['path']) != spec['sha256']:
            raise ValueError('immutable evaluation checkpoint changed')
    inherited = config['capacity']['inherited']
    if sha256(inherited['path']) != inherited['sha256']:
        raise ValueError('inherited FP32 capacity changed')
    capacity = json.loads(Path(inherited['path']).read_text())
    resources = json.loads(Path(inherited['resources_path']).read_text())
    if not (capacity['passed'] and capacity['complete'] and resources['passed'] and resources['payload_cleanup_complete']):
        raise ValueError('inherited capacity was not cleanly accepted')
    for rank in capacity['ranks']:
        profile = rank['full_decode']
        if (profile['precision'],profile['input_frames'],profile['new_tokens']) != ('fp32',24,2048):
            raise ValueError('full FP32 decoder capacity differs')
    paths = ['model_training/rigweave/src','model_training/rigweave/scripts/ar_only_runtime.py',
        'model_training/rigweave/scripts/stage1_feedback.py','model_training/rigweave/scripts/virtual_query_xyz_runtime.py',
        'model_training/rigweave/scripts/virtual_pair_runtime.py']
    subprocess.run(['git','-C',str(ROOT),'diff','--quiet',config['prior_source'],source,'--',*paths],check=True)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU repair audit initialized CUDA')
    result = {'complete':True,'passed':True,'source':source,'scientific_sha256':scientific_sha(config),
        'plan':plan,'inherited_capacity':inherited,'unchanged_model_and_decoder_source':True,
        'optimizer_updates':0,'recomputed_views':1584,'retained_views_per_checkpoint':880,
        'cuda_initialized':False,'quality_accepted':False}
    write_json(output/'result.json',result)
    return result


def canonical_record(batch, surface, geometry):
    count = int(geometry['joint_mask'][0].sum())
    vertices = batch['frame_vertices'][0]
    result = {'asset_id':batch['asset_id'],'surface':{k:v.detach().cpu() for k,v in surface.items()},
        'truth':geometry['frame_target_joints'][0,:,:count].cpu(),
        'parents':geometry['target_parents'][0,:count].cpu(),
        'raw_frames':geometry['selected_frames'][0].tolist(),
        'diagonals':torch.linalg.vector_norm(vertices.amax(1)-vertices.amin(1),dim=-1).cpu(),
        'normalization_sha256':tree_fingerprint({k:geometry[k] for k in ('query_center','query_scale')}),
        'surface_sha256':tree_fingerprint(surface)}
    if tree_fingerprint(result['surface']) != result['surface_sha256']:
        raise ValueError('surface CPU transfer changed bits')
    if set(result['raw_frames']) & set(batch['heldout_raw_ids']):
        raise ValueError('repair entered reserved raw frames')
    return result


def canonical_identity(record, frame):
    return {'asset_id':record['asset_id'],'frame_index':frame,'raw_frame_index':record['raw_frames'][frame],
        'normalization_sha256':record['normalization_sha256'],'surface_sha256':record['surface_sha256'],
        'original_GT_sha256':tree_fingerprint({'joints':record['truth'][frame],'parents':record['parents']}),
        'target_mesh_diagonal':float(record['diagonals'][frame]),'gt_joint_count':len(record['parents'])}


def validate_canonical(record, original):
    if tree_fingerprint(record['surface']) != record['surface_sha256']:
        raise ValueError('canonical surface changed')
    rows = {r['frame_index']:r for r in original if r['asset_id'] == record['asset_id']}
    if set(rows) != set(range(8)):
        raise ValueError('canonical asset lacks8parent views')
    for frame in range(8):
        identity = canonical_identity(record,frame)
        if any(identity[field] != rows[frame][field] for field in PAIR_FIELDS):
            raise ValueError('canonical rawframe/GT/normalization differs')


def run(config, output, source, device, rank, world):
    spec = config['audit_result']
    if not spec or sha256(spec['path']) != spec['sha256']:
        raise ValueError('repair CPU audit is not bound')
    accepted = json.loads(Path(spec['path']).read_text())
    if not accepted['passed'] or accepted['scientific_sha256'] != scientific_sha(config):
        raise ValueError('repair CPU audit differs')
    summaries,rows,plan = read_previous(config)
    original = parent_config(config)
    base,entries = read_all(original)
    selected = evaluation_rows(base,entries)
    dataset = fixed_dataset(selected,base,8,mode='seen_eval')
    affected = set(config['affected_asset_ids'])
    cached,manifest = {},[]
    cache_dir = output/'canonical'
    cache_dir.mkdir(exist_ok=True)
    for index in range(rank,len(selected),world):
        if shared_deadline_due(config,device,world):
            raise RuntimeError('repair cache reached allocation deadline')
        if selected[index]['asset_id'] not in affected:
            continue
        batch = dataset[index]
        surface,geometry = prepare_virtual_surface([batch],device,**base['surface_sampling'])
        record = canonical_record(batch,surface,geometry)
        validate_canonical(record,rows['donor'])
        path = cache_dir/f'asset_{index:03d}.pt'
        if path.exists():
            raise RuntimeError('refuse to replace canonical input')
        torch.save(record,path)
        saved = torch.load(path,map_location='cpu',weights_only=True)
        validate_canonical(saved,rows['donor'])
        cached[record['asset_id']] = saved
        manifest.append({'asset_id':record['asset_id'],'surface_sha256':record['surface_sha256'],
                         **mirror(path,'canonical/'+path.name,config)})
        del surface,geometry,record,batch
    all_manifest = sorted(sum(gather(manifest,world),[]),key=lambda x:x['asset_id'])
    if {x['asset_id'] for x in all_manifest} != affected or len(all_manifest) != 18:
        raise ValueError('canonical cache coverage differs')
    cache_identity = tree_fingerprint(all_manifest)
    if rank == 0:
        write_json(output/'canonical_manifest.json',all_manifest)
    model,tokenizer = load_evaluation_model(original,device,config['checkpoints']['donor'])
    evaluations = {}
    for label in LABELS:
        checkpoint = config['checkpoints'][label]
        if rank == 0 and sha256(checkpoint['path']) != checkpoint['sha256']:
            raise ValueError('repair checkpoint changed')
        dist.barrier(device_ids=[device.index])
        payload = torch.load(checkpoint['path'],map_location='cpu',mmap=True,weights_only=False)
        model.load_state_dict(backbone_state(payload,model.state_dict(),layout=checkpoint['layout']),strict=True)
        del payload
        directory = output/'stage1'/label
        directory.mkdir(parents=True,exist_ok=True)
        replacement,examples_by_asset = [],{}
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        for index in range(rank,len(selected),world):
            if shared_deadline_due(config,device,world):
                raise RuntimeError('repair evaluation reached allocation deadline')
            asset = selected[index]['asset_id']
            if asset not in affected:
                continue
            record = cached[asset]
            validate_canonical(record,rows[label])
            surface = {k:v.to(device) for k,v in record['surface'].items()}
            if tree_fingerprint(surface) != record['surface_sha256']:
                raise ValueError('device transfer changed canonical surface')
            generated = generate_asset(model,surface,tokenizer,frame_indices=list(range(8)),
                max_new_tokens=2048,precision='fp32')
            asset_rows = []
            for frame in range(8):
                value = score_generation(generated[frame],tokenizer,record['truth'][frame].numpy(),
                    record['parents'].tolist(),float(record['diagonals'][frame]))
                value.update(canonical_identity(record,frame))
                replacement.append(value)
                asset_rows.append(value)
            examples_by_asset[asset] = {'asset_id':asset,'raw_frames':record['raw_frames'],
                'points':record['surface']['query_points'][0],'truth':record['truth'],
                'parents':record['parents'],'rows':asset_rows}
            write_json(directory/f'progress_rank_{rank}.json',{'recomputed_local_assets':len(examples_by_asset),
                'total_local_affected':len(cached),'label':label,'seconds':time.monotonic()-started})
            del surface,generated
        parent_spec = config['prior_evaluations'][label]
        local_rows = [json.loads(line) for line in Path(parent_spec['rows'][rank]['path']).read_text().splitlines()]
        merged = merge_rows(local_rows,replacement,affected)
        row_path = directory/f'rows_rank_{rank}.jsonl'
        with row_path.open('x') as handle:
            for row in merged:
                handle.write(json.dumps(row)+'\n')
        examples = torch.load(parent_spec['examples'][rank]['path'],map_location='cpu',weights_only=True)
        examples = [examples_by_asset.get(e['asset_id'],e) for e in examples]
        torch.save(examples,directory/f'examples_rank_{rank}.pt')
        artifacts = gather({'rank':rank,**mirror(row_path,f'stage1/{label}/'+row_path.name,config)},world)
        mirror(directory/f'examples_rank_{rank}.pt',f'stage1/{label}/examples_rank_{rank}.pt',config)
        all_rows = sum(gather(merged,world),[])
        identity = copy.deepcopy(summaries[label]['identity'])
        identity['input_repair'] = {'canonical_manifest_sha256':cache_identity,
            'repair_scientific_sha256':scientific_sha(config),'source':source,
            'policy':config['repair'],'parent_source':config['prior_source']}
        report = {'complete':True,'label':label,'identity':identity,'summary':summarize_rows(all_rows),
            'row_artifacts':artifacts,'no_GT_reference_or_teacher_prefix':True,'optimizer_updates':0,
            'recomputed_assets':18,'bitwise_retained_assets':110,
            'max_reserved_bytes_per_rank':gather(torch.cuda.max_memory_reserved(device),world)}
        evaluations[label] = report
        if rank == 0:
            write_json(directory/'summary.json',report)
            mirror(directory/'summary.json',f'stage1/{label}/summary.json',config)
        gc.collect()
        torch.cuda.empty_cache()
    comparisons = analyze_evaluations(evaluations) if rank == 0 else None
    comparisons = gather(comparisons,world)[0]
    if rank == 0:
        write_json(output/'stage1_comparisons.json',comparisons)
        mirror(output/'stage1_comparisons.json','stage1_comparisons.json',config)
    return {'complete':True,'passed':True,'source':source,'scientific_sha256':scientific_sha(config),
        'optimizer_updates':0,'canonical_manifest':all_manifest,'plan':plan,'evaluations':evaluations,
        'paired_analysis':comparisons,'parent_failure_preserved':True,'quality_accepted':False}
