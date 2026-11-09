"""CPU tests for fixed input identity and exported skeleton scoring."""
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skintokens_original_test_20261109 import distances, glb_skeleton, source_key


def test_origin_key_matches_repo_path_across_revisions():
    assert source_key('https://github.com/A/B/blob/aaa/folder/a b.glb') == source_key(
        'https://raw.githubusercontent.com/a/b/bbb/folder/a%20b.glb')
    assert source_key('https://sketchfab.com/3d-models/model-0123456789abcdef0123456789abcdef') == (
        'sketchfab','0123456789abcdef0123456789abcdef')


def test_rig_distances_use_common_bbox_scale():
    a = np.array([[0.,0.,0.],[1.,0.,0.]])
    edges = np.array([[0,1]])
    same = distances(a,edges,a,edges,2.)
    assert same['J2J_percent'] == same['B2B_percent'] == 0.
    assert same['J2B_percent'] == pytest.approx(.25)  # The first endpoint is omitted by the reference sampling.
    assert all(value == pytest.approx(10., abs=.01) for value in
               distances(a,edges,a+[0,.2,0],edges,2.).values())


def test_glb_joint_heads_include_ancestor_transforms(tmp_path):
    doc = {'nodes':[{'translation':[3.,0.,0.], 'children':[1]},
                    {'translation':[0.,1.,0.], 'children':[2]},
                    {'translation':[0.,0.,2.]}],
           'skins':[{'joints':[1,2]}]}
    raw = json.dumps(doc).encode()
    raw += b' ' * ((-len(raw)) % 4)
    file = tmp_path/'skeleton.glb'
    file.write_bytes(struct.pack('<4sIII4s',b'glTF',2,20+len(raw),len(raw),b'JSON')+raw)
    joints, edges = glb_skeleton(file)
    np.testing.assert_array_equal(joints, [[3.,1.,0.],[3.,1.,2.]])
    np.testing.assert_array_equal(edges, [[0,1]])
