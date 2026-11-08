"""Tests for the single-pose Stage 1 training contract."""
from pathlib import Path
import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from skintokens_static_protocol import validate_config
from train_skintokens_static import SinglePoseSkeleton, make_stream
from rigweave.dynamic_rig.virtual_frame_data import VirtualFrameDataset
from rigweave.dynamic_rig.frame_batch_sampler import FrameSampleRequest


def config():
    path = Path(__file__).resolve().parents[2] / 'experiments/skintokens_static_5k_20261109.json'
    return json.loads(path.read_text())


def test_exact_static_plan_with_one_frame_per_visit():
    recipe = config()
    validate_config(recipe)
    a, b = (make_stream(recipe, rank) for rank in (0, 1))
    realized = []
    for i in range(627):
        block = a.take() + b.take()
        assert all(r.frames == 1 for r in block)
        assert len(block) == (16 if i % 209 == 208 else 48)
        realized.extend(r.item_index for r in block)
    assert realized == list(a.order)
    assert a.cursor == b.cursor == 15000
    assert all(len(set(realized[i:i+5000])) == 5000 for i in range(0, 30000, 5000))
    changed = copy.deepcopy(recipe)
    changed['stage2_enabled'] = True
    with pytest.raises(ValueError):
        validate_config(changed)


def test_single_pose_uses_same_query_draw():
    entry = {'path': '/data/a.npz', 'asset_id': 'asset'}
    dataset = VirtualFrameDataset([entry])

    def materialize(row, choose, **_):
        ids = choose(np.random.default_rng(7).normal(size=(32, 12, 3)))
        return {'frame_vertices': torch.ones(1, len(ids), 12, 3), 'selected_frames': ids}

    with patch('rigweave.dynamic_rig.virtual_frame_data.load_medium_batch', materialize):
        static = dataset[FrameSampleRequest(0, 1, 33)]
        dynamic = dataset[FrameSampleRequest(0, 8, 33)]
    assert static['selected_frames'].tolist() == dynamic['selected_frames'][:1].tolist()
    assert not set(static['selected_frames']) & set(static['heldout_raw_ids'])


class Geometry(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3, 512)

    def encode_latents(self, pc, feats):
        return None, self.layer((pc + feats).mean(1, keepdim=True)).expand(-1, 512, -1), 512, None


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(9, 8)
        self.output = nn.Linear(8, 9)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, **_):
        return SimpleNamespace(logits=self.output(inputs_embeds.cumsum(1)))


def test_only_native_skeleton_modules_train():
    model = SinglePoseSkeleton(Geometry(), nn.Linear(512, 8), Decoder())
    points = torch.randn(2, 12, 3)
    ids = torch.tensor([[0, 1, 2, 3], [0, 1, 3, 4]])
    mask = torch.tensor([[True, True, True, True], [True, True, True, False]])
    losses, accuracy = model(points, points, ids, mask)
    assert losses.shape == accuracy.shape == (2,)
    losses.mean().backward()
    assert set(dict(model.named_children())) == {'mesh_encoder', 'decoder_condition_projection', 'xyz_decoder'}
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(ValueError):
        model.encode(points[:, None], points[:, None])
