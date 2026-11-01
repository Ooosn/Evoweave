"""Native, zero-update comparison on a frozen validation subset."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from paper_benchmark_protocol import scientific_sha, validate_config, validate_sources, defer_official_annotations
from virtual_pair_guard import ROOT, verify_launch, deadline_due


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def validate_tree(xyz, parents):
    xyz = np.asarray(xyz, dtype=np.float32)
    parents = [-1 if p is None else int(p) for p in parents]
    if xyz.shape != (len(parents), 3) or not len(parents) or not np.isfinite(xyz).all():
        raise ValueError('invalid predicted joint coordinates')
    if sum(p < 0 for p in parents) != 1:
        raise ValueError('prediction does not contain exactly one root')
    for start in range(len(parents)):
        seen, node = set(), start
        while node >= 0:
            if node >= len(parents) or node in seen:
                raise ValueError('prediction contains a cycle or an invalid parent')
            seen.add(node)
            node = parents[node]
    return xyz, parents


def metrics(record, prediction, parents, *, frame=0):
    from rigweave.dynamic_rig.skeleton_topology_metrics import _topology_metrics
    xyz, parents = validate_tree(prediction, parents)
    truth = record['truth'][frame].numpy()
    truth_parents = record['parents'].tolist()
    pairwise = np.linalg.norm(xyz[:, None] - truth[None], axis=-1)
    topology = _topology_metrics(xyz, parents, truth, truth_parents)
    return {'asset_id': record['asset_id'], 'frame': frame, 'success': True,
            'edge_f1': float(topology['edge_f1']), 'predicted_joints': xyz.tolist(),
            'predicted_parents': parents, 'predicted_joint_count': len(parents),
            'gt_joint_count': len(truth_parents), 'exact_count': len(parents) == len(truth_parents),
            'j2j': float(.5 * (pairwise.min(0).mean() + pairwise.min(1).mean()) / float(record['diagonals'][frame]))}


def failure(record, error, *, frame=0):
    return {'asset_id': record['asset_id'], 'frame': frame, 'success': False,
            'edge_f1': 0., 'j2j': None, 'exact_count': False,
            'gt_joint_count': len(record['parents']), 'error': repr(error)}


def summarize(rows):
    valid = [r for r in rows if r['success']]
    return {'rows': len(rows), 'failures': len(rows) - len(valid),
            'edge_f1_all': float(np.mean([r['edge_f1'] for r in rows])),
            'exact_count_all': float(np.mean([r['exact_count'] for r in rows])),
            'j2j_success_only': float(np.mean([r['j2j'] for r in valid])) if valid else None,
            'geometry_denominator': len(valid)}


def select_rows(base, entries, config):
    from virtual_pair_runtime import evaluation_rows
    excluded = {r['asset_id'] for r in evaluation_rows(base, entries)}
    if len(excluded) != 128:
        raise ValueError('the excluded development panel changed')
    candidates = [r for r in entries['valid'] if r['asset_id'] not in excluded]
    seed = str(config['evaluation']['selection_seed']) + ':'
    selected = sorted(candidates, key=lambda r: hashlib.sha256((seed + r['asset_id']).encode()).hexdigest())[:24]
    if len(selected) != 24 or len({r['asset_id'] for r in selected}) != 24:
        raise ValueError('paper subset is incomplete')
    return selected


def audit(config, output, source):
    from reference_flow_runtime import read_all
    from predicted_reference_protocol import bound_config
    from virtual_query_xyz_runtime import fixed_dataset
    from rigweave.dynamic_rig.virtual_frame_data import prepare_virtual_surface
    from stage1_surface_repair_runtime import canonical_record
    original = bound_config(config['parent_config'])
    output.mkdir(exist_ok=False)
    hashes = {}
    for name, spec in {'working_model': config['working_model'], **config['baseline_weights']}.items():
        actual = digest(spec['path'])
        if spec.get('sha256') and spec['sha256'] != actual:
            raise ValueError('weight digest differs: ' + name)
        hashes[name] = actual
    sources = validate_sources(config)
    auxiliary={}
    if config['runtime'].get('paper_skin_qwen_config'):
        path=str(Path(config['runtime']['paper_skin_qwen_config'])/'config.json')
        auxiliary[path]=digest(path)
    cache = output / 'canonical.pt'
    inherited=config.get('inherit_prepared_inputs')
    if inherited:
        if digest(inherited['path'])!=inherited['sha256']:
            raise ValueError('original prepared-input receipt changed')
        prior=json.loads(Path(inherited['path']).read_text())
        resources=json.loads(Path(inherited['resources_path']).read_text())
        prior_root=Path(inherited['path']).parent
        if (not prior['passed'] or not prior['complete'] or prior['assets']!=24
                or not resources['passed'] or not resources['payload_cleanup_complete']
                or digest(prior_root/'canonical.pt')!=prior['canonical_sha256']
                or digest(prior_root/'selection.json')!=prior['selection_sha256']
                or prior['weight_sha256']['working_model']!=config['working_model']['sha256']):
            raise ValueError('prepared-input inheritance failed')
        cache.hardlink_to(prior_root/'canonical.pt')
        selection=json.loads((prior_root/'selection.json').read_text())
    else:
        base, entries = read_all(original)
        selected = select_rows(base, entries, config)
        dataset = fixed_dataset(selected, base, 8, mode='seen_eval')
        records = []
        for i in range(len(selected)):
            batch = dataset[i]
            surface, geometry = prepare_virtual_surface([batch], torch.device('cpu'), **base['surface_sampling'])
            record = canonical_record(batch, surface, geometry)
            record['vertices'] = batch['frame_vertices'][0].cpu()
            records.append(record)
            write(output / 'progress.json', {'assets': i + 1, 'total': len(selected)})
        torch.save(records, cache)
        selection = [{'asset_id': r['asset_id'], 'raw_frames': r['raw_frames'],
                      'surface_sha256': r['surface_sha256'], 'normalization_sha256': r['normalization_sha256'],
                      'joints': len(r['parents'])} for r in records]
    write(output / 'selection.json', selection)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU preparation initialized CUDA')
    result = {'complete': True, 'passed': True, 'source': source,
              'scientific_sha256': scientific_sha(config), 'optimizer_updates': 0,
              'cuda_initialized': False, 'assets': len(selection),
              'prepared_inputs_inherited_without_resampling': inherited,
              'selection_sha256': digest(output / 'selection.json'),
              'canonical_sha256': digest(cache), 'baseline_revisions': sources,
              'weight_sha256': hashes,
              'auxiliary_sha256': auxiliary,
              'evaluation_scope': 'nonselection_validation_subset_not_an_independent_test_set',
              'preprocessing': 'source_skeleton_metadata_root_RT_then_selected_input_mesh_normalization'}
    write(output / 'result.json', result)


def record_progress(output, method, rows):
    write(output / (method + '_rows.json'), rows)
    write(output / 'progress.json', {'method': method, **summarize(rows)})


def baseline(config, records, output, method):
    """Use isolated processes because the official repositories both own src.*."""
    runtime = config['runtime']
    source = config['baseline_sources'][method]
    sys.path.insert(0, source['path'])
    os.chdir(source['path'])
    checkpoint = config['baseline_weights'][method]['path']
    device = torch.device('cuda', 0)
    torch.cuda.set_per_process_memory_fraction(.30, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    trace = {}
    if method == 'unirig_official':
        defer_official_annotations(source['path'])
        from omegaconf import OmegaConf
        from src.model.parse import get_model
        from src.tokenizer.parse import get_tokenizer
        from src.tokenizer.spec import TokenizerConfig
        model_config = OmegaConf.load('configs/model/unirig_ar_350m_1024_81920_float32.yaml')
        model_config.llm.pretrained_model_name_or_path = runtime['opt_config_root']
        tokenizer = get_tokenizer(TokenizerConfig.parse(OmegaConf.load(
            'configs/tokenizer/tokenizer_parts_articulationxl_256.yaml')))
        model = get_model(**model_config, tokenizer=tokenizer)
        payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
        state = {k.removeprefix('model.'): v for k, v in payload['state_dict'].items()}
        model.load_state_dict(state, strict=True)
        del payload, state
        model.to(device).eval()
    elif method == 'puppeteer_official':
        from types import SimpleNamespace
        os.chdir(Path(source['path']) / 'skeleton')
        sys.path.insert(0, str(Path.cwd()))
        from skeleton_models import skeletongen
        from functools import partial
        # The released complete state strictly restores the encoder too.
        # Avoid downloading its redundant, immediately overwritten initialization.
        skeletongen.load_model = partial(skeletongen.load_model, ckpt_path=None)
        model = skeletongen.SkeletonGPT(SimpleNamespace(joint_token=True, n_discrete_size=128,
            n_max_bones=100, llm=runtime['opt_config_root'], seq_shuffle=True,
            num_beams=1, batchsize_per_gpu=1))
        payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
        model.load_state_dict(payload['model'], strict=True)
        del payload
        model.to(device).half().eval()
    else:
        from copy import deepcopy
        from src.model.tokenrig import TokenRig
        from transformers import StoppingCriteria, StoppingCriteriaList
        payload=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
        cfg=deepcopy(payload['hyper_parameters']['model_config'])
        cfg['pretrained_vae']=config['baseline_weights']['skintokens_vae']['path']
        cfg['llm']['pretrained_model_name_or_path']=runtime['paper_skin_qwen_config']
        del payload
        model=TokenRig.load_from_system_checkpoint(checkpoint,model_config=cfg,strict=True).to(device).eval()
        tokenizer=model.tokenizer
        class SkeletonEnd(StoppingCriteria):
            def __call__(self,input_ids,scores,**kwargs):
                return input_ids[:,-1]==tokenizer.eos
        skeleton_stop=StoppingCriteriaList([SkeletonEnd()])
    original_generate = model.transformer.generate
    def capture_generate(*args, **kwargs):
        ids = original_generate(*args, **kwargs)
        trace['ids'] = ids.detach().cpu().tolist()
        return ids
    model.transformer.generate = capture_generate
    rows = []
    started = time.monotonic()
    for record in records:
        if deadline_due(config) or time.monotonic() - started > 4800:
            raise RuntimeError('paper baseline reached its explicit deadline')
        random.seed(20261102)
        np.random.seed(20261102)
        torch.manual_seed(20261102)
        t0 = time.monotonic()
        try:
            points = record['surface']['dense_points'][0, 0].float()
            normals = record['surface']['dense_normals'][0, 0].float()
            trace.clear()
            if method in ('unirig_official','skintokens_official'):
                center = .5 * (points.amin(0) + points.amax(0))
                scale = 2. / (points.amax(0) - points.amin(0)).amax()
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    extra={'only_ids':True,'stopping_criteria':skeleton_stop} if method=='skintokens_official' else {}
                    decoded = model.generate(vertices=((points-center)*scale).to(device),
                        normals=normals.to(device), cls='articulation' if method=='skintokens_official' else 'articulationxl',
                        max_new_tokens=2048, num_beams=1, do_sample=False,**extra)
                if not trace['ids'][0] or trace['ids'][0][-1] != tokenizer.eos:
                    raise ValueError('max_tokens_without_EOS')
                if method=='skintokens_official':
                    decoded=tokenizer.detokenize(ids=decoded.output_ids.detach().cpu().numpy())
                parents = decoded.parents if decoded.parents is not None else decoded._get_parents()
                xyz = np.asarray(decoded.joints, dtype=np.float32) / float(scale) + center.numpy()
            else:
                points, normals = points[:8192], normals[:8192]
                center = .5 * (points.amin(0) + points.amax(0))
                scale = .9995 / (points.amax(0) - points.amin(0)).amax()
                normalized = (points-center)*scale
                pc_center = .5 * (normalized.amin(0) + normalized.amax(0))
                pc_scale = (normalized.amax(0)-normalized.amin(0)).amax() + 1e-5
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                    model.generate({'pc_normal': torch.cat((normalized,normals), -1)[None].to(device).half()})
                ids = trace['ids'][0]
                if 1 not in ids:
                    raise ValueError('max_tokens_without_EOS')
                if ids and ids[0] == 0:
                    ids = ids[1:]
                ids = ids[:ids.index(1)]
                if not ids or len(ids) % 4:
                    raise ValueError('invalid_joint_token_sequence')
                tokens = np.asarray(ids, dtype=np.int64).reshape(-1, 4) - 3
                if (tokens[:, :3] < 0).any() or (tokens[:, :3] >= 128).any():
                    raise ValueError('invalid_coordinate_token')
                xyz = (tokens[:, :3].astype(np.float32)/128. - .5)
                xyz = (xyz * float(pc_scale) + pc_center.numpy()) / float(scale) + center.numpy()
                parents = (tokens[:, 3] - 1).tolist()
            row = metrics(record, xyz, parents)
        except (ValueError, IndexError, TypeError, AssertionError) as error:
            row = failure(record, error)
        row.update(method=method, generation_seconds=time.monotonic()-t0,
                   raw_generated_ids=trace.get('ids'), input_surface_sha256=record['surface_sha256'])
        rows.append(row)
        record_progress(output, method, rows)
    write(output / (method + '_summary.json'), summarize(rows))


def run(config, output, source, config_path):
    from predicted_reference_protocol import bound_config
    from ar_only_runtime import load_evaluation_model
    from stage1_feedback import generate_asset, score_generation
    from predicted_reference_runtime import head_prediction, score_output
    from rigweave.dynamic_rig.anchor_joint_binding import AnchorJointBinding, AnchorJointConfig
    spec = config['audit_result']
    if digest(spec['path']) != spec['sha256']:
        raise ValueError('paper CPU result changed')
    audit_result = json.loads(Path(spec['path']).read_text())
    if not audit_result['passed'] or audit_result['scientific_sha256'] != scientific_sha(config):
        raise ValueError('CPU scientific identity differs')
    cache = Path(spec['path']).parent / 'canonical.pt'
    if digest(cache) != audit_result['canonical_sha256']:
        raise ValueError('canonical paper inputs changed')
    records = torch.load(cache, map_location='cpu', weights_only=True)
    validate_sources(config)
    for name, weight in {'working_model': config['working_model'], **config['baseline_weights']}.items():
        if digest(weight['path']) != audit_result['weight_sha256'][name]:
            raise ValueError('audited weights changed before inference: ' + name)
    original = bound_config(config['parent_config'])
    original['evaluation']['decoder_attention'] = 'eager'
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.30, device)
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    model, tokenizer = load_evaluation_model(original, device, config['working_model'])
    head = AnchorJointBinding(AnchorJointConfig(**original['architecture']), mode='reference_transport')
    payload = torch.load(config['working_model']['path'], map_location='cpu', mmap=True, weights_only=False)
    head.load_state_dict({k.removeprefix('head.'): v for k, v in payload['model'].items() if k.startswith('head.')}, strict=True)
    del payload
    head.to(device).eval()
    rows, targets = [], []
    for record in records:
        if deadline_due(config):
            raise RuntimeError('native save-stop reached')
        surface = {k: v.to(device) for k, v in record['surface'].items()}
        t0 = time.monotonic()
        with torch.inference_mode():
            generated = generate_asset(model, surface, tokenizer, frame_indices=[0], precision='fp32')
            raw = score_generation(generated[0], tokenizer, record['truth'][0].numpy(),
                record['parents'].tolist(), float(record['diagonals'][0]))
            if raw['success']:
                try:
                    row = metrics(record, raw['predicted_joints'], raw['predicted_parents'])
                except (ValueError, IndexError, TypeError, AssertionError) as error:
                    row = failure(record, error)
            else:
                row = failure(record, raw['status'])
            if row['success']:
                dynamic_error = None
                try:
                    features, _ = model.encode(surface)
                    xyz = torch.tensor(row['predicted_joints'], device=device, dtype=torch.float32)
                    prediction = head_prediction(head, features, surface, xyz, row['predicted_parents'], 0)
                except (ValueError, IndexError, TypeError, AssertionError) as error:
                    dynamic_error = repr(error)
                for frame in range(1, 8):
                    try:
                        if dynamic_error:
                            raise ValueError(dynamic_error)
                        target = metrics(record, prediction[frame].cpu().numpy(), row['predicted_parents'], frame=frame)
                    except (ValueError, IndexError, TypeError, AssertionError) as error:
                        target = failure(record,error,frame=frame)
                    targets.append(dict(target, method='evoweave330k_head'))
                    targets.append(dict(metrics(record, row['predicted_joints'], row['predicted_parents'], frame=frame),
                                        method='evoweave330k_reference_copy'))
            else:
                for frame in range(1, 8):
                    for method in ('evoweave330k_head','evoweave330k_reference_copy'):
                        targets.append(dict(failure(record,row.get('error',raw['status']),frame=frame),method=method))
        row.update(method='evoweave330k', generation_seconds=time.monotonic()-t0,
                   raw_generated_ids=generated[0]['tokens'], input_surface_sha256=record['surface_sha256'])
        rows.append(row)
        record_progress(output, 'evoweave330k', rows)
        write(output / 'dynamic_rows.json', targets)
        del surface
    del model, head
    gc.collect()
    torch.cuda.empty_cache()
    env = os.environ.copy()
    for method in ('unirig_official','puppeteer_official'):
        cmd = [sys.executable, '-B', str(Path(__file__).resolve()), '--config', str(config_path),
               '--output', str(output), '--mode', 'capacity', '--baseline', method]
        with (output / (method+'.log')).open('x') as log:
            subprocess.run(cmd, env=env, check=True, stdout=log, stderr=subprocess.STDOUT,
                           timeout=config['evaluation']['payload_limit_seconds'])
    summaries = {'evoweave330k': summarize(rows)}
    for method in ('unirig_official','puppeteer_official'):
        summaries[method] = json.loads((output / (method + '_summary.json')).read_text())
    for method in ('evoweave330k_head','evoweave330k_reference_copy'):
        summaries[method] = summarize([r for r in targets if r['method']==method])
    write(output / 'result.json', {'complete': True, 'passed': True, 'source': source,
        'scientific_sha256': scientific_sha(config), 'optimizer_updates': 0,
        'assets': len(records), 'summaries': summaries, 'seconds': time.monotonic()-started,
        'scope': '24_hash_selected_nonselection_validation_assets',
        'GT_skeleton_given_to_predictors': False, 'greedy_all_methods': True,
        'Puppeteer_output': 'raw_joint_tokens_without_geometric_postprocessing',
        'Puppeteer_initialization': 'strict_complete_released_state_including_point_encoder_no_redundant_auxiliary_checkpoint',
        'UniRig_python310_compatibility': 'deferred_source_annotations_only_no_file_edits',
        'baseline_input': 'same_canonical_reference_surface_points_and_normals',
        'sample_counts': {'unirig_official': 'retained_dense_surface', 'puppeteer_official': 8192},
        'root_RT_from_dataset_metadata': True,
        'pretrained_data_overlap_with_baselines_not_excluded': True,
        'shared_GPU_timing_not_an_isolated_speed_benchmark': True})


def complete_baselines(config, output, source, config_path):
    """Retain completed generations; finish only the unstarted official baselines."""
    receipt=config['audit_result']
    if digest(receipt['path'])!=receipt['sha256']:
        raise ValueError('prepared input receipt changed')
    accepted=json.loads(Path(receipt['path']).read_text())
    if not accepted['passed'] or accepted['scientific_sha256']!=scientific_sha(config):
        raise ValueError('prepared input science differs')
    cache=Path(receipt['path']).parent/'canonical.pt'
    if digest(cache)!=accepted['canonical_sha256']:
        raise ValueError('prepared canonical data changed')
    records=torch.load(cache,map_location='cpu',weights_only=True)
    expected={r['asset_id']:r['surface_sha256'] for r in records}
    retained=config['runtime']['paper_completed_evoweave']
    old_resources=json.loads(Path(retained['resources_path']).read_text())
    if digest(retained['resources_path'])!=retained['resources_sha256']:
        raise ValueError('original partial resource evidence changed')
    if not old_resources['complete'] or not old_resources['payload_cleanup_complete'] or old_resources['exit_code']!=1:
        raise ValueError('partial comparison did not exit cleanly after its adapter error')
    saved={}
    for name,spec in retained['files'].items():
        if digest(spec['path'])!=spec['sha256']:
            raise ValueError('retained completed prediction changed')
        saved[name]=json.loads(Path(spec['path']).read_text())
    rows=saved['evoweave330k_rows.json']
    targets=saved['dynamic_rows.json']
    target_keys={(r['asset_id'],r['frame'],r['method']) for r in targets}
    wanted={(asset,frame,method) for asset in expected for frame in range(1,8)
        for method in ('evoweave330k_head','evoweave330k_reference_copy')}
    if (len(rows)!=24 or {r['asset_id']:r['input_surface_sha256'] for r in rows}!=expected
            or len(targets)!=336 or target_keys!=wanted):
        raise ValueError('retained generations do not cover the fixed24cohort')
    validate_sources(config)
    for name,spec in {'working_model':config['working_model'],**config['baseline_weights']}.items():
        if digest(spec['path'])!=accepted['weight_sha256'][name]:
            raise ValueError('audited model weights changed')
    if any(digest(path)!=value for path,value in accepted.get('auxiliary_sha256',{}).items()):
        raise ValueError('audited auxiliary model configuration changed')
    output.mkdir(exist_ok=False)
    for name,value in saved.items():write(output/name,value)
    failures={}
    for method in config['evaluation']['methods'][1:]:
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--config',str(config_path),
            '--output',str(output),'--mode','capacity','--baseline',method]
        env=os.environ.copy()
        if method=='skintokens_official':
            env['PYTHONPATH']=config['runtime']['paper_skin_pythonpath']
            env.pop('USE_FLASH3',None)
        with (output/(method+'.log')).open('x') as log:
            child=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,
                timeout=config['evaluation']['payload_limit_seconds'])
        if child.returncode:
            failures[method]={'exit_code':child.returncode,'log':str(output/(method+'.log'))}
    summaries={'evoweave330k':summarize(rows)}
    for method in config['evaluation']['methods'][1:]:
        path=output/(method+'_summary.json')
        if path.exists():summaries[method]=json.loads(path.read_text())
    for method in ('evoweave330k_head','evoweave330k_reference_copy'):
        summaries[method]=summarize([r for r in targets if r['method']==method])
    result={'complete':True,'passed':not failures,'source':source,'scientific_sha256':scientific_sha(config),
        'optimizer_updates':0,'assets':24,'summaries':summaries,'setup_failures':failures,
        'completed_Evoweave_not_regenerated':True,'retained_Evoweave':retained,
        'GT_skeleton_given_to_predictors':False,'greedy_all_methods':True,
        'Puppeteer_output':'raw_joint_tokens_without_geometric_postprocessing',
        'SkinTokens_output':'released_GRPO1400_skeleton_prefix_only_no_skinning_or_GT_count',
        'root_RT_from_dataset_metadata':True,'shared_GPU_timing_not_an_isolated_speed_benchmark':True}
    write(output/'result.json',result)
    if failures:raise RuntimeError('one or more official baseline setups failed; other results retained')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('audit','capacity'), required=True)
    parser.add_argument('--baseline', choices=('unirig_official','puppeteer_official','skintokens_official'))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config, args.mode)
    source = verify_launch(args, config)
    torch.set_num_threads(1)
    if args.mode == 'audit':
        audit(config, args.output, source)
        return
    try:
        if args.baseline:
            audit_path = Path(config['audit_result']['path'])
            records = torch.load(audit_path.parent/'canonical.pt', map_location='cpu', weights_only=True)
            baseline(config, records, args.output, args.baseline)
        elif config['runtime'].get('paper_completed_evoweave'):
            complete_baselines(config,args.output,source,args.config.resolve())
        else:
            run(config, args.output, source, args.config.resolve())
    except Exception as error:
        if args.output.exists():
            name='failure_'+args.baseline+'.json' if args.baseline else 'failure.json'
            write(args.output/name, {'complete': False, 'passed': False,
                'error': repr(error), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
