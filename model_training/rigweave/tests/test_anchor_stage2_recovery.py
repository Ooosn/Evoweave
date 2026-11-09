import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'model_training/rigweave/scripts'))
from anchor_full_data import make_stream
from anchor_stage2_protocol import frame_policy, predecessor_config, validate_config
from anchor_stage2_recovery import restore
from fusion_evaluation import tree_fingerprint
from reference_flow_protocol import scientific_sha
from virtual_pair_runtime import next_update_batches, parameter_names


def config():
    return json.loads((ROOT/'model_training/experiments/anchor_stage2_frozen_eval_20261110.json').read_text())


def test_eval_recovery_does_not_allow_training_recipe_changes():
    recipe = config()
    validate_config(recipe)
    for key, field, value in (('optimization', 'peak_lr', 0.001),
                              ('data', 'stream_seed', 1),
                              ('training', 'bone_weight', 0.)):
        changed = copy.deepcopy(recipe)
        changed[key][field] = value
        with pytest.raises(ValueError, match='exact training recipe'):
            validate_config(changed)


def test_restore_exact_optimizer_and_stream_for_next_update(tmp_path):
    recipe = config()
    base = {**recipe, 'training': {**recipe['training'], 'frame_policy': frame_policy(24)}}
    streams = [make_stream(base, rank, 2) for rank in range(2)]
    samples = frames = step = 0
    while samples < 10000:
        size, batches = next_update_batches(streams[0], samples, 2)
        next_update_batches(streams[1], samples, 2)
        frames += 2 * sum(r.frames for b in batches for r in b)
        samples += size
        step += 1
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    old = predecessor_config(recipe)
    old['training'].update(frame_policy=frame_policy(24), arm='st12', backbone_exposure_offset=0)
    old['artifacts']['durable_root'] += '/st12'
    initial = {'arm': 'st12'}
    admitted = {'initialization': {'st12': initial}, 'plan': {'plan_sha256': streams[0].plan_sha256,
        'milestones': {str(samples): {'input_frame_visits': frames, 'updates': step}}}}
    path = tmp_path/'checkpoint.pt'
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'configuration': old, 'scientific_sha256': scientific_sha(old), 'initialization': initial,
        'code_commit': recipe['recovery']['source'], 'optimizer_parameter_names': parameter_names(model),
        'additional_asset_exposures': samples, 'global_backbone_asset_exposures': samples,
        'input_frame_exposures': frames, 'step': step,
        'rank_states': [{'rank': rank, 'stream': stream.state_dict()} for rank, stream in enumerate(streams)]}, path)
    proof = {'checkpoints': [{'path': str(path)}], 'adam_states': len(optimizer.state)}
    for rank in range(2):
        resumed = torch.nn.Linear(2, 2)
        adam = torch.optim.AdamW(resumed.parameters())
        with patch('anchor_stage2_recovery.apply_rng') as rng:
            stream, *counters = restore(recipe, base, proof, admitted, resumed, adam, rank)
            assert rng.call_args.args[0]['rank'] == rank
        assert counters == [10000, 130240, 209]
        assert tree_fingerprint(model.state_dict()) == tree_fingerprint(resumed.state_dict())
        assert tree_fingerprint(optimizer.state_dict()) == tree_fingerprint(adam.state_dict())
        assert next_update_batches(stream, samples, 2) == next_update_batches(streams[rank], samples, 2)
