"""Matched exposure, readout and supervision tests without external weights."""
from copy import deepcopy
import json

import pytest
import torch

from skintokens_stage1_protocol import ARMS, ROOT, STATIC_CONFIG, recipe, validate_config
from test_virtual_path_ablation import make_base, make_surface, make_targets
from train_skintokens_stage1 import Stage1Skeleton, plan_for, stream_for
from train_skintokens_static import make_stream


def test_matched_plans_and_supervision():
    base = json.loads((ROOT/'model_training/experiments/anchor_full_backbone_20261030_v1.json').read_text())
    static = json.loads(STATIC_CONFIG.read_text())
    configs = [recipe(arm) for arm in ARMS]
    for config in configs:
        validate_config(config)
        assert stream_for(config, 0).order.tolist() == make_stream(static, 0).order.tolist()
        assert config['evaluation'] == static['evaluation']
    a, b, c = [plan_for(config, base) for config in configs]
    assert a == b
    assert a['samples'] == c['samples'] == 30000
    assert a['updates'] == c['updates'] == 627
    assert a['plan_sha256'] == c['plan_sha256']
    assert a['input_frame_visits'] == c['input_frame_visits']
    assert a['supervised_frame_visits'] > c['supervised_frame_visits'] == 30000
    configs[0]['stage2_enabled'] = True
    with pytest.raises(ValueError):
        validate_config(configs[0])


def test_frame_local_isolation_and_query_readout():
    torch.set_num_threads(1)
    initial = make_base()
    surface = make_surface()
    changed = {k:v.clone() for k,v in surface.items()}
    changed['query_points'][:, 1] += 4.
    params = [Stage1Skeleton(deepcopy(initial), arm).eval() for arm in ARMS]
    assert all(model.trainable_parameter_names == params[0].trainable_parameter_names for model in params)
    for model, arm in zip(params, ARMS):
        before, virtual = model.encode(surface)
        after, _ = model.encode(changed)
        assert virtual is None
        assert before.shape[1] == (1 if arm == 'temporal_query' else 3)
        assert torch.equal(before[:, 0], after[:, 0]) == (arm == 'frame_local')
        assert all(not p.requires_grad for p in model.auxiliary.parameters())
        assert not any(name.startswith('head.') for name in model.state_dict())


def test_selected_native_ce_matches_existing_multi_target_loss():
    torch.set_num_threads(1)
    model = Stage1Skeleton(make_base(), 'temporal_multi').eval()
    surface, targets = make_surface(), make_targets()
    slots = torch.tensor([[0, 2], [1, 0]])
    valid = torch.tensor([[True, True], [True, False]])
    rows = torch.arange(2)[:, None]
    ids = targets['main_input_ids'][rows, slots]
    masks = targets['main_attention_mask'][rows, slots]
    actual, accuracy = model(surface, ids, masks, slots, valid)
    encoded, virtual = model.encode(surface)
    expected = model.forward_encoded(encoded, virtual, targets, auxiliary_weight=0.,
        target_frame_indices=slots, target_frame_mask=valid)
    torch.testing.assert_close(actual, expected['loss_per_asset'])
    torch.testing.assert_close(accuracy.mean(), expected['main_accuracy'])
    actual.mean().backward()
    assert model.motion_encoder.blocks[0].anchor_temporal.self_attn.in_proj_weight.grad is not None
    single = Stage1Skeleton(make_base(), 'temporal_query').eval()
    with pytest.raises(ValueError, match='first real query'):
        single(surface, ids[:, :1], masks[:, :1], torch.ones(2, 1, dtype=torch.long), valid[:, :1])
