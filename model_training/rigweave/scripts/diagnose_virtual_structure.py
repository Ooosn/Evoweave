"""Read-only signal/routing/readout audit of actual new-model checkpoints."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from motion_experiment_runtime import guard, load_config, source_metadata, write_json
from train_virtual_frame import dataset_for, evaluation_rows, load_entries, prepare
from virtual_frame_runtime import build_virtual_model, file_sha256
from rigweave.dynamic_rig.virtual_frame_rig import coordinate_regression_loss


def difference(actual, expected):
    delta = (actual.float()-expected.float()).abs()
    return {'meanabs':float(delta.mean()), 'maxabs':float(delta.max())}


def contrast(value, axis):
    value = value.float()
    rms = value.square().mean().sqrt()
    spread = (value-value.mean(axis,keepdim=True)).square().mean().sqrt()
    return {'rms':float(rms),'contrast_rms':float(spread),'contrast_over_rms':float(spread/rms.clamp_min(1e-12))}


def coordinates(xyz, targets):
    from eval_dynamic_rig_generation import _topology_metrics
    count = int(targets['joint_count'][0])
    gt = targets['frame_target_joints'][0,:,:count].cpu().numpy()
    prediction = xyz[0,:,:count].float().cpu().numpy()
    parents = targets['target_parents'][0,:count].tolist()
    return {'xyz_l1':float(coordinate_regression_loss(xyz,targets['frame_target_joints'],targets['joint_mask'],targets['frame_mask']).mean()),
            'query_F1':_topology_metrics(prediction[0],parents,gt[0],parents)['edge_f1'],
            'temporal_contrast':contrast(xyz,1)}


def attention_summary(module, query, memory):
    with torch.autocast(query.device.type,enabled=False):
        dim, heads = query.shape[-1], module.num_heads
        width = dim//heads
        weight, bias = module.in_proj_weight.float(), module.in_proj_bias.float()
        q = F.linear(query.float(),weight[:dim],bias[:dim]).reshape(query.shape[0],-1,heads,width).transpose(1,2)
        k = F.linear(memory.float(),weight[dim:2*dim],bias[dim:2*dim]).reshape(memory.shape[0],-1,heads,width).transpose(1,2)
        probability = torch.softmax(q@k.transpose(-1,-2)/(width**.5),dim=-1)
        entropy = -(probability*probability.clamp_min(1e-30).log()).sum(-1)/np.log(memory.shape[1])
    return {'fp32_reconstruction_not_a_replacement_forward':True,'normalized_entropy_mean':float(entropy.mean()),
            'max_probability_mean':float(probability.max(-1).values.mean()),
            'joint_attention_contrast_rms':float((probability-probability.mean(2,keepdim=True)).square().mean().sqrt())}


def readout_capacity(features, refreshed, targets, device):
    gt = targets['frame_target_joints'].cpu().numpy().reshape(-1,3).astype(np.float64)
    count = int(targets['joint_count'][0])
    x = features.bfloat16().float().cpu().numpy().reshape(-1,features.shape[-1]).astype(np.float64)
    again = refreshed.bfloat16().float().reshape(-1,features.shape[-1])
    results = {}
    for scope, size in (('query',count),('all_frames',len(gt))):
        matrix = np.concatenate((x[:size],np.ones((size,1))),axis=1)
        u,s,v = np.linalg.svd(matrix,full_matrices=False)
        fits = {}
        for cutoff in (1e-3,1e-6):
            keep = s > s[0]*cutoff
            weight = v[keep].T@((u[:,keep].T@gt[:size])/s[keep,None])
            w = torch.tensor(weight[:-1].T,device=device,dtype=torch.float32)
            b = torch.tensor(weight[-1],device=device,dtype=torch.float32)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                fitted = F.linear(features.reshape(-1,features.shape[-1])[:size],w,b).float()
                repeated = F.linear(again[:size],w,b).float()
            target = torch.tensor(gt[:size],device=device,dtype=torch.float32)
            fits[str(cutoff)] = {'retained_rank':int(keep.sum()),'FP64_fit_l1':float(np.abs(matrix@weight-gt[:size]).mean()),
                                 'actual_BF16_linear_l1':float((fitted-target).abs().mean()),
                                 'refreshed_forward_BF16_linear_l1':float((repeated-target).abs().mean()),
                                 'weight_norm':float(np.linalg.norm(weight[:-1]))}
        results[scope] = {'rows':size,'columns':matrix.shape[1],'singular_max':float(s[0]),'singular_min':float(s[-1]),
                          'condition_number':float(s[0]/max(s[-1],1e-30)),'fits':fits}
    return results


def cohort(model, config, plan, spec, device, rank, path):
    rows=[]
    report={'complete':False,'source':source_metadata(config),'model':spec,'rows':rows,
            'no_optimizer_or_parameter_updates':True,'GT_parent_conditioning_only_not_free_generation':True,
            'started_at':datetime.now(timezone.utc).isoformat()}
    selected=evaluation_rows(load_entries(config),config)
    for split,values in selected.items():
        dataset=dataset_for(values,config,mode='seen_eval')
        for index in range(rank,len(values),len(plan['models'])):
            surface,targets=prepare(dataset,[index],config,device)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                anchors,memory=model.encode(surface)
                teacher=model.forward_encoded(anchors,memory,surface['query_points'],targets,xyz_weight=.5)
            h,points,mask=teacher['joint_features'],surface['query_points'],targets['joint_mask']
            results={'normal':coordinates(teacher['joints'],targets)}
            alternatives={'zero_learned_anchors':(torch.zeros_like(anchors),points),
                          'zero_explicit_points':(anchors,torch.zeros_like(points)),
                          'break_feature_point_pairing':(anchors,points.flip(2)),
                          'spatial_mean_anchors':(anchors.mean(2,keepdim=True).expand_as(anchors),points)}
            for name,(a,p) in alternatives.items():
                with torch.autocast('cuda',dtype=torch.bfloat16):xyz=model.positions(h,mask,a,p)
                results[name]=coordinates(xyz,targets)
            rows.append({'split':split,'asset_id':values[index]['asset_id'],'manifest_index':values[index]['manifest_index'],
                         'selected_frames':targets['selected_frames'].cpu().tolist(),'metrics':results})
            write_json(path,report)
            print('COHORT '+json.dumps({'rank':rank,'completed':len(rows),'split':split,
                                       'normal_xyz_l1':results['normal']['xyz_l1']}),flush=True)
            del surface,targets,anchors,memory,teacher,h,points,mask,alternatives,xyz
    report.update(complete=True,finished_at=datetime.now(timezone.utc).isoformat())
    write_json(path,report)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    args = parser.parse_args()
    plan = load_config(args.config)
    config = load_config(plan['virtual_config'])
    guard(config,'preflight')
    state = json.loads((Path(config['runtime']['repo'])/'model_training/state/current.json').read_text())
    if state['active_operation']['label'] != plan['label'] or state['active_operation']['config_sha256'] != file_sha256(args.config):
        raise ValueError('unrecorded structural diagnostic')
    rank = int(os.environ['LOCAL_RANK'])
    spec = plan['models'][rank]
    device = torch.device('cuda',rank)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.75,device)
    output = Path(plan['output'])
    output.mkdir(parents=True,exist_ok=True)
    path = output/(spec['name']+'.json')
    if path.exists():raise FileExistsError(path)
    if file_sha256(spec['checkpoint']) != spec['sha256']:raise ValueError('checkpoint changed')
    model, initialization = build_virtual_model(config,'self_cross_2',device)
    payload = torch.load(spec['checkpoint'],map_location='cpu',mmap=True,weights_only=False)
    model.load_state_dict(payload['model'],strict=True)
    del payload
    model.requires_grad_(False).eval()
    if plan.get('mode')=='cohort':
        cohort(model,config,plan,spec,device,rank,path)
        return
    selected = evaluation_rows(load_entries(config),config)['train']
    dataset = dataset_for(selected,config,mode='seen_eval')
    indices = [next(i for i,row in enumerate(selected) if row['manifest_index']==j) for j in plan['manifest_indices']]
    surface, targets = prepare(dataset,[indices[0]],config,device)
    if surface['query_points'].shape[:2] != (1,8):raise ValueError('this diagnostic requires the recorded single-asset eight-frame view')
    report = {'complete':False,'source':source_metadata(config),'model':spec,'initialization':initialization,
              'started_at':datetime.now(timezone.utc).isoformat(),'asset':targets['asset_id'][0],
              'selected_frames':targets['selected_frames'].cpu().tolist(),'no_optimizer_or_parameter_updates':True}
    write_json(path,report)
    captures, attention, layer_signals = {}, {}, {}
    hooks = [model.positions.xyz_head.register_forward_pre_hook(lambda m,a:captures.update(readout=a[0].detach()))]
    for i,layer in enumerate(model.positions.layers):
        hooks.append(layer.register_forward_hook(lambda m,a,out,i=i:layer_signals.update({str(i):{'joints':contrast(out,1),'frames':contrast(out.reshape(1,8,-1,out.shape[-1]),1)}})))
        hooks.append(layer.cross_attention.register_forward_pre_hook(lambda m,a,i=i:attention.update({str(i):attention_summary(m,a[0],a[1])})))
    with torch.autocast('cuda',dtype=torch.bfloat16):
        anchors,memory = model.encode(surface)
        teacher = model.forward_encoded(anchors,memory,surface['query_points'],targets,xyz_weight=.5)
    z = captures['readout'].clone()
    for hook in hooks:hook.remove()
    h, baseline = teacher['joint_features'],teacher['joints']
    report['baseline'] = coordinates(baseline,targets)
    report['signals'] = {'H_joint':contrast(h,1),'anchor_spatial':contrast(anchors,2),'anchor_temporal':contrast(anchors,1),
                          'GT_temporal':contrast(targets['frame_target_joints'],1), 'layers':layer_signals,'attention':attention,
                          'final_joint':contrast(z,1),'final_temporal':contrast(z.reshape(1,8,-1,z.shape[-1]),1)}
    ablations = {}
    points, mask = surface['query_points'],targets['joint_mask']
    frames = torch.tensor([0,7,6,5,4,3,2,1],device=device)
    joints = torch.arange(h.shape[1]-1,-1,-1,device=device)
    queries = torch.arange(anchors.shape[2]-1,-1,-1,device=device)
    conditions = {
        'repeat_same_position_inputs':(h,anchors,points,baseline),
        'joint_permutation':(h[:,joints],anchors,points,baseline[:,:,joints]),
        'frame_permutation':(h,anchors[:,frames],points[:,frames],baseline[:,frames]),
        'paired_anchor_permutation':(h,anchors[:,:,queries],points[:,:,queries],baseline),
        'same_H_for_all_joints':(h.mean(1,keepdim=True).expand_as(h),anchors,points,None),
        'repeat_query_frame_memory':(h,anchors[:,:1].expand_as(anchors),points[:,:1].expand_as(points),None),
        'zero_learned_anchors':(h,torch.zeros_like(anchors),points,None),
        'zero_explicit_points':(h,anchors,torch.zeros_like(points),None),
        'break_feature_point_pairing':(h,anchors,points[:,:,queries],None),
    }
    for name,(joint,anchor,point,expected) in conditions.items():
        with torch.autocast('cuda',dtype=torch.bfloat16):xyz=model.positions(joint,mask,anchor,point)
        record={'changed_from_baseline':difference(xyz,baseline),'metrics_against_original_GT':coordinates(xyz,targets),
                'joint_spread':contrast(xyz,2)}
        if expected is not None:record['routing_equivariance_error']=difference(xyz,expected)
        ablations[name]=record
    other_surface,_ = prepare(dataset,[indices[1]],config,device)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        other_anchors,_=model.encode(other_surface)
        wrong=model.positions(h,mask,other_anchors,other_surface['query_points'])
    ablations['different_asset_anchors_and_points']={'changed_from_baseline':difference(wrong,baseline),'metrics_against_original_GT':coordinates(wrong,targets)}
    report['ablations']=ablations
    del other_surface,other_anchors,conditions
    reordered={k:v[:,frames] if v.ndim==4 else v for k,v in surface.items()}
    with torch.autocast('cuda',dtype=torch.bfloat16):permuted=model(reordered,targets,xyz_weight=.5)
    report['whole_nonquery_frame_permutation']={'XYZ':difference(permuted['joints'],baseline[:,frames]),
                                                'H':difference(permuted['joint_features'],h)}
    captures.clear()
    hook=model.positions.xyz_head.register_forward_pre_hook(lambda m,a:captures.update(readout=a[0].detach()))
    with torch.autocast('cuda',dtype=torch.bfloat16):repeated=model(surface,targets,xyz_weight=.5)
    hook.remove()
    report['whole_same_input_repeat']={'XYZ':difference(repeated['joints'],baseline),
                                        'H':difference(repeated['joint_features'],h),'readout':difference(captures['readout'],z)}
    report['linear_readout_capacity']=readout_capacity(z,captures['readout'],targets,device)
    report['current_XYZ_head_weight_norm']=float(model.positions.xyz_head.weight.norm())
    torch.save({'H':h.cpu(),'anchors':anchors.cpu(),'readout':z.cpu(),'refreshed_readout':captures['readout'].cpu(),
                'points':points.cpu(),'target':targets['frame_target_joints'].cpu(),'parents':targets['target_parents'].cpu()},
               output/(spec['name']+'_signals.pt'))
    report.update(complete=True,finished_at=datetime.now(timezone.utc).isoformat(),peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30)
    write_json(path,report)
    print('STRUCTURE_PROBE '+json.dumps({'model':spec['name'],'baseline':report['baseline'],'signals':report['signals'],
                                       'linear_readout_capacity':report['linear_readout_capacity']}),flush=True)


if __name__=='__main__':main()
