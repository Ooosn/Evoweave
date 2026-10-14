"""CPU execution of the actual normal-trend contracts, callbacks and training loop."""
from contextlib import nullcontext, redirect_stdout
from copy import deepcopy
import io
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from test_virtual_query_training import functions, namespace as core_namespace, ROOT
from virtual_fusion_normal_runtime import MatchedFrameStream
from rigweave.dynamic_rig.frame_batch_sampler import FrameSampleRequest


def namespace():
    core = SimpleNamespace(**core_namespace())
    result = {'deepcopy': deepcopy, 'np': np, 'Path': Path, 'json': json, 'torch': torch,
              'core': core, 'MatchedFrameStream': MatchedFrameStream,
              'PURPOSE': 'normal5000_trend', 'ENTRYPOINT': 'model_training/rigweave/scripts/train_virtual_fusion_normal.py',
              'ROUTES': {'legacy': 'virtual_queries_real', 'a': 'virtual_queries_real', 'b': 'real_queries_virtual',
                         'c': 'relative_position', 'd': 'token_film'},
              'MILESTONES': [5000, 10000, 15000, 20000], 'hashlib': hashlib,
              'BASE_INITIAL_SHA256': 'a89b29c903abffe2eead47b5411edb8588b599fe874ce490c990ce07225d3635',
              'SOURCE_HASHES': {'train': '653f496d6610caee5d1ba6aa8dc26b2d656665c31aad05aead93be97a07f3b27',
                                'valid': 'fd2be7b5885833946e68dc186300a820757ae652c487749eec1b049d381e230a'},
              'sha256': core.sha256, 'write_json': core.write_json, 'auxiliary_summary': core.auxiliary_summary}
    functions('probe_virtual_query_lambda.py', ['parameter_fingerprint'], result)
    functions('train_virtual_fusion_normal.py', ['validate_config', 'make_stream', 'comparison_contract',
        'initialization_fingerprints',
        'verify_selection', 'selected_evaluation_rows', 'record_samples', 'validate_evaluation_rows',
        'run_evaluation', 'validate_capacity', 'declared_operation', 'capacity_result'], result)
    return result


def config(route='a'):
    ns = namespace()
    value = json.loads((ROOT / 'model_training/experiments/fusion_a_medium_v1.json').read_text())
    value.pop('comparison')
    value.pop('probe')
    value.update(purpose='normal5000_trend', route=route)
    value['architecture']['fusion_mode'] = ns['ROUTES'][route]
    if route == 'legacy':
        value['reference_route'] = 'legacy_query_first'
    value['runtime']['physical_gpus'] = [5, 7]
    value['data'].update(frames_min=2, frames_max=24, stream_seed=20261013)
    value['data']['manifests']['train'] = {'path': '/subset.jsonl', 'rows': 5000, 'sha256': '0' * 64}
    value['optimization'].update(schedule_samples=20000, clip_grad_norm=1, auxiliary_weight=0 if route == 'legacy' else .1)
    value['training'] = {'target_assets': 20000, 'effective_assets_per_update': 48,
        'checkpoint_assets': [5000, 10000, 15000, 20000], 'evaluation_assets': [5000, 10000, 15000, 20000],
        'frame_policy': {str(t): 4 for t in range(2, 25)}}
    value['normal_match'] = {'base_initial_state_sha256': ns['BASE_INITIAL_SHA256'],
                             'selection': {'path': '/selection.json', 'sha256': '1' * 64}}
    value['evaluation'] = {'train_assets': 32, 'valid_assets': 128, 'seed': 20261014, 'frames': 8,
        'generated_frame_indices': list(range(8)), 'max_new_tokens': 2048, 'initial': True}
    value['capacity'] = {'selection': {'path': '/capacity.json', 'sha256': '2' * 64}}
    return value


class NormalTrainingTests(unittest.TestCase):
    def test_exact_budget_and_route_contracts(self):
        ns = namespace()
        contracts, plans = [], []
        for route in ('legacy', 'a', 'b', 'c', 'd'):
            plan = config(route)
            self.assertEqual(ns['validate_config'](plan, 'train', 2), [5000, 10000, 15000, 20000])
            contracts.append(ns['comparison_contract'](plan))
            plans.append(ns['make_stream'](plan, 0, 2).plan_sha256)
            ns['validate_config'](plan, 'profile', 1)
        self.assertTrue(all(contract == contracts[0] for contract in contracts))
        self.assertEqual(len(set(plans)), 1)
        counts = []
        position = 0
        while position < 20000:
            size = ns['core'].update_asset_target(position, 20000, 48, [5000, 10000, 15000, 20000], 2) * 2
            counts.append(size)
            position += size
        self.assertEqual(counts, ns['make_stream'](config(), 0, 2).update_sizes.tolist())
        self.assertEqual(len(counts), 420)

    def test_five_routes_keep_identical_common_initial_tensors(self):
        from test_virtual_query_fusion import make_model
        ns = namespace()
        expected = ns['initialization_fingerprints'](make_model(), 'a')
        for route, mode in ns['ROUTES'].items():
            model = make_model(mode)
            actual = ns['initialization_fingerprints'](model, route)
            self.assertEqual(actual['shared_initial_state_sha256'], expected['shared_initial_state_sha256'])
            if route in {'c', 'd'}:
                self.assertNotEqual(actual['full_base_initial_state_sha256'], expected['full_base_initial_state_sha256'])
            with torch.no_grad():
                model.motion_encoder.norm.weight.add_(1)
            changed = ns['initialization_fingerprints'](model, route)
            self.assertNotEqual(changed['shared_initial_state_sha256'], expected['shared_initial_state_sha256'])

    def test_rejects_old_budget_fixed_frames_cache_and_unmatched_settings(self):
        ns = namespace()
        for section, key, value in [('training', 'target_assets', 80016), ('data', 'frames_max', 4),
            ('data', 'heldout_frames', 4), ('optimization', 'peak_lr', .0002),
            ('optimization', 'auxiliary_weight', .3), ('evaluation', 'generated_frame_indices', [0]),
            ('architecture', 'fusion_mode', 'real_queries_virtual')]:
            plan = config()
            plan[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                ns['validate_config'](plan, 'train', 2)
        plan = config()
        plan['comparison'] = {'cache': 'old-four-poses'}
        with self.assertRaises(ValueError):
            ns['validate_config'](plan, 'train', 2)
        with self.assertRaises(ValueError):
            ns['validate_config'](config(), 'train', 1)

    def test_batch_policy_is_not_a_scientific_sampling_difference(self):
        ns, left, right = namespace(), config(), config()
        right['training']['frame_policy'] = {str(t): 12 for t in range(2, 25)}
        self.assertEqual(ns['comparison_contract'](left), ns['comparison_contract'](right))
        self.assertEqual(ns['make_stream'](left, 1, 2).plan_sha256, ns['make_stream'](right, 1, 2).plan_sha256)
        right['data']['stream_seed'] += 1
        self.assertNotEqual(ns['comparison_contract'](left), ns['comparison_contract'](right))

    def test_only_the_exact_incomplete_published_operation_may_run(self):
        ns, plan = namespace(), config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps(plan))
            args = SimpleNamespace(config=path, output=Path(directory) / 'output', mode='profile')
            run = {'config': str(path), 'config_sha256': ns['sha256'](path), 'output': str(args.output),
                   'mode': 'profile', 'approved': True, 'physical_gpus': [5, 7]}
            operation = {'mode': 'normal5000', 'gpu_work_allowed': True, 'formal_training_allowed': False,
                         'complete': False, 'required_entrypoint': ns['ENTRYPOINT'], 'runs': [run]}
            state = {'architecture_contract': {'id': 'virtual_query_xyz_aux_ids_v1'}, 'active_operation': operation}
            ns['declared_operation'](state, args, plan)
            for key, value in [('complete', True), ('gpu_work_allowed', False), ('formal_training_allowed', True),
                               ('mode', 'reference_compare'), ('required_entrypoint', 'wrong.py')]:
                changed = deepcopy(state)
                changed['active_operation'][key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    ns['declared_operation'](changed, args, plan)
            for key, value in [('approved', False), ('config_sha256', 'bad'), ('output', '/other'),
                               ('physical_gpus', [7]), ('mode', 'train')]:
                changed = deepcopy(state)
                changed['active_operation']['runs'][0][key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    ns['declared_operation'](changed, args, plan)

    def test_training_pose_record_rejects_leak_and_duplicates(self):
        ns = namespace()
        geometry = {'asset_id': ['a'], 'selected_frames': torch.tensor([[2, 3]]),
                    'heldout_raw_ids': [list(range(10, 18))]}
        requests = [FrameSampleRequest(4, 2, 1234)]
        self.assertEqual(ns['record_samples'](geometry, requests)['sampling_seeds'], [1234])
        for frames in ([[2, 2]], [[2, 10]], [[2, 3, 4]]):
            geometry['selected_frames'] = torch.tensor(frames)
            with self.assertRaises(ValueError):
                ns['record_samples'](geometry, requests)

    def test_capacity_needs_both_ranks_and_every_route_specific_stress_case(self):
        ns = namespace()
        for route in ('legacy', 'a', 'b', 'c', 'd'):
            plan = config(route)
            receipt = {'accepted': True, 'world_size': 2, 'capacity_contract': ns['core'].capacity_contract(plan),
                'approved_cases': [{'frames': t, 'batch_per_rank': 4, 'stress_group': group,
                    'ranks_approved': [0, 1], 'accumulation_microbatches': 2}
                    for t in range(2, 25) for group in (['longest_main', f'geometry_t{t}'] +
                                                       ([] if route == 'legacy' else ['largest_auxiliary']))]}
            ns['validate_capacity'](plan, receipt)
            receipt['world_size'] = 1
            with self.assertRaises(ValueError):
                ns['validate_capacity'](plan, receipt)
            receipt['world_size'] = 2
            receipt['approved_cases'].pop()
            with self.assertRaisesRegex(ValueError, 'unmeasured'):
                ns['validate_capacity'](plan, receipt)

    def test_panel_is_deterministic_disjoint_and_uses_full_reserved_eight(self):
        ns = namespace()
        entries = {split: [{'asset_id': f'{split}{i}'} for i in range(count)]
                   for split, count in [('train', 5000), ('valid', 967)]}
        chosen = ns['selected_evaluation_rows'](entries, config())
        self.assertEqual(chosen, ns['selected_evaluation_rows'](entries, config('legacy')))
        self.assertEqual(chosen['train_seen'], chosen['train_heldout'])
        self.assertEqual([len(chosen[key]) for key in ('train_seen', 'train_heldout', 'valid')], [32, 32, 128])
        rows = [{'asset_id': row['asset_id'], 'frame_index': t, 'raw_frame_index': t + 10}
                for row in chosen['train_heldout'] for t in range(8)]
        ns['validate_evaluation_rows'](rows, chosen['train_heldout'], 8)
        with self.assertRaises(ValueError):
            ns['validate_evaluation_rows'](rows[:-1], chosen['train_heldout'], 8)
        rows[1]['raw_frame_index'] = rows[0]['raw_frame_index']
        with self.assertRaises(ValueError):
            ns['validate_evaluation_rows'](rows, chosen['train_heldout'], 8)

    def test_capacity_buckets_require_same_batch_and_identical_stress_assets(self):
        ns, plan = namespace(), config()
        plan['training']['frame_policy'] = {str(t): 24 if t <= 4 else 8 if t <= 12 else 4
                                            for t in range(2, 25)}
        cases = []
        for frames, batch in ((4, 24), (12, 8), (24, 4)):
            for group in ('longest_main', f'geometry_t{frames}', 'largest_auxiliary'):
                aliases = [f'geometry_t{t}' for t in range(2, 25)] if group.startswith('geometry_t') else [group]
                cases.append({'frames': frames, 'batch_per_rank': batch, 'stress_group': group,
                    'covered_stress_groups': aliases, 'ranks_approved': [0, 1], 'accumulation_microbatches': 2})
        receipt = {'accepted': True, 'world_size': 2, 'capacity_contract': ns['core'].capacity_contract(plan),
                   'approved_cases': cases}
        ns['validate_capacity'](plan, receipt)
        for change in ('frame', 'batch', 'membership'):
            altered = deepcopy(receipt)
            if change == 'frame':
                altered['approved_cases'][0]['frames'] = 3
            elif change == 'batch':
                altered['approved_cases'][0]['batch_per_rank'] = 23
            else:
                altered['approved_cases'][1]['covered_stress_groups'] = ['geometry_t4']
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'unmeasured'):
                ns['validate_capacity'](plan, altered)

    def test_capacity_aliases_are_derived_from_the_pinned_selection(self):
        ns, plan = namespace(), config()
        a, b = {'asset_id': 'a', 'path': '/a.npz'}, {'asset_id': 'b', 'path': '/b.npz'}
        selection = {'groups': {'longest_main': [a], 'largest_auxiliary': [b]},
                     'geometry_by_frames': {'2': a, '3': b, '24': a}}
        case = {'frames': 24, 'batch_per_rank': 4, 'stress_group': 'geometry_t24', 'accepted': True,
                'measurements': [{'peak_allocated_bytes': 10}]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'selection.json'
            path.write_text(json.dumps(selection))
            plan['capacity']['selection'] = {'path': str(path), 'sha256': ns['sha256'](path)}
            for rank in (0, 1):
                (Path(folder) / f'capacity_rank_{rank}.json').write_text(json.dumps({'accepted': True, 'cases': [case]}))
            result = ns['capacity_result'](SimpleNamespace(output=Path(folder)), plan, 2)
            self.assertEqual(result['approved_cases'][0]['covered_stress_groups'],
                             ['longest_main', 'geometry_t2', 'geometry_t24'])
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'selection changed'):
                ns['capacity_result'](SimpleNamespace(output=Path(folder)), plan, 2)

    def test_actual_evaluation_callback_keeps_legacy_auxiliary_not_applicable(self):
        for route in ('legacy', 'a'):
            ns, plan = namespace(), config(route)
            chosen = {group: [{'asset_id': group + str(i)} for i in range(2)]
                      for group in ('train_seen', 'train_heldout', 'valid')}
            ns['selected_evaluation_rows'] = lambda *a: chosen
            ns['fixed_dataset'] = lambda rows, c, f, mode: (rows, mode)
            class Dataset:
                def __init__(self, rows, mode):
                    self.rows, self.mode = rows, mode
                def __len__(self):
                    return len(self.rows)
            ns['fixed_dataset'] = lambda rows, c, f, mode: Dataset(rows, mode)
            def prepare(dataset, indices, *args):
                raw = range(10, 18) if dataset.mode == 'heldout_eval' else range(8)
                return None, None, {'asset_id': [dataset.rows[indices[0]]['asset_id']],
                                   'selected_frames': torch.tensor([list(raw)]), 'heldout_raw_ids': [list(range(10, 18))]}
            def evaluate(model, items, tokenizer, eval_config, output, step, frame_indices):
                self.assertEqual(eval_config['probe']['evaluate_auxiliary'], route != 'legacy')
                geometries = [item[2] for item in items]
                rows = [{'asset_id': item['asset_id'][0], 'frame_index': t,
                         'raw_frame_index': int(item['selected_frames'][0, t]), 'success': True,
                         'quantized_edge_f1': 1., 'original_edge_f1': 1., 'exact_count': True,
                         'main_ce': 1., 'main_accuracy': .5, 'quantized_j2j': 0.}
                        for item in geometries for t in frame_indices]
                result = {'rows': rows}
                if route != 'legacy':
                    result['auxiliary_rows'] = [{'asset_id': item['asset_id'][0], 'natural_eos': True,
                        'success': True, 'exact_count': True, 'exact_parents': True, 'exact_tokens': True}
                        for item in geometries]
                return result
            ns['core'].prepare, ns['core'].evaluate = prepare, evaluate
            ns['dist'] = SimpleNamespace(barrier=lambda: None)
            with tempfile.TemporaryDirectory() as directory:
                args = SimpleNamespace(output=Path(directory))
                for rank in (1, 0):
                    ns['run_evaluation'](nn.Linear(1, 1), None, None, {}, plan, args, 'cpu', 20000, rank, 2)
                result = json.loads((Path(directory) / 'evaluation_020000/summary.json').read_text())
            self.assertEqual(result['summary']['valid']['views'], 16)
            self.assertFalse(result['full_validation'])
            self.assertEqual('auxiliary_summary' in result, route != 'legacy')

    def test_actual_core_loop_uses_passed_stream_boundaries_gradient_and_record_callbacks(self):
        ns = core_namespace()
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(1.))
            def forward(self, surface, target, auxiliary_weight):
                per_asset = (self.weight * surface).square()
                return {'loss_per_asset': per_asset, 'loss': per_asset.mean(), 'main_loss': per_asset.mean(),
                        'auxiliary_loss': per_asset.new_tensor(0.), 'main_accuracy': per_asset.new_tensor(1.)}
        stream = MatchedFrameStream(range(5), rank=0, world_size=1, seed=1, target_assets=20,
            effective_assets=8, update_boundaries=[10, 20], frame_policy={str(t): 3 for t in range(2, 25)},
            manifest_sha256='0' * 64)
        checkpoints, evaluations, gradients, recorded = [], [], [], []
        def prepare(dataset, requests, *args):
            return torch.tensor([float(r.item_index + 1) for r in requests]), None, requests
        def recorder(geometry, requests):
            recorded.extend(requests)
            return {'asset_ids': [str(r.item_index) for r in requests]}
        ns.update(fixed_dataset=lambda *a, **k: [], optimizer_for=lambda m, c: torch.optim.SGD(m.parameters(), lr=.001),
            wrap_model=lambda m, *a: m, prepare=prepare, set_exposure_lr=lambda *a: (.001, .9),
            grad_norms=lambda *a: {}, save_checkpoint=lambda *a: checkpoints.append(a[5]))
        plan = config('legacy')
        plan['training'].update(target_assets=20, effective_assets_per_update=8, evaluation_assets=[10, 20])
        plan['optimization']['clip_grad_norm'] = 1e9
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
             patch.object(torch, 'autocast', lambda *a, **k: nullcontext()), \
             patch.object(torch.cuda, 'reset_peak_memory_stats'), patch.object(torch.cuda, 'synchronize'), \
             patch.object(torch.cuda, 'max_memory_allocated', return_value=0), \
             patch.object(torch.cuda, 'max_memory_reserved', return_value=0):
            ns['run_training'](Model(), None, None, {}, {'train': []}, plan,
                SimpleNamespace(output=Path(directory)), 'cpu', 0, 1, 'test', stream=stream, milestones=[10, 20],
                evaluation_callback=lambda *a: evaluations.append(a[7]),
                gradient_check=lambda *a: gradients.append(True), sample_recorder=recorder)
            trace = [json.loads(line) for line in (Path(directory) / 'trace_rank_0.jsonl').read_text().splitlines()]
        self.assertEqual([row['asset_exposures'] for row in trace], [8, 10, 18, 20])
        self.assertEqual(checkpoints, [10, 20])
        self.assertEqual(evaluations, [10, 20])
        self.assertEqual(gradients, [True])
        self.assertEqual(len(recorded), 20)
        self.assertEqual(trace[-1]['supervised_frame_exposures'], sum(r.frames for r in recorded))


if __name__ == '__main__':
    unittest.main()
