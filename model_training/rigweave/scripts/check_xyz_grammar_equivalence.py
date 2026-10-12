#!/usr/bin/env python3
"""Compare the lightweight adapter to the pinned upstream class on real GT prefixes."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

root=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(root/'model_training/rigweave/src'))
from train_dynamic_rig import build_tokenizer
from rigweave.dynamic_rig.aux_skeleton_tokens import AuxSkeletonTokens
from rigweave.dynamic_rig.shared_structure_medium_data import load_medium_batch
from rigweave.dynamic_rig.virtual_query_xyz_data import build_xyz_aux_targets
from rigweave.dynamic_rig.virtual_query_xyz import original_xyz_vocabulary_processor

p=argparse.ArgumentParser()
p.add_argument('--config',type=Path,required=True)
p.add_argument('--selection',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
config=json.loads(a.config.read_text())
selected=json.loads(a.selection.read_text())[0]['path']
with Path(config['data']['manifests']['train']['path']).open() as f:
    row=next(r for line in f if (r:=json.loads(line))['path']==selected)
batch=load_medium_batch(row,np.asarray([9,7],dtype=np.int64))
tokenizer=build_tokenizer(Path(config['runtime']['tokenizer_config']))
target=build_xyz_aux_targets([batch],tokenizer,AuxSkeletonTokens(**config['auxiliary_codec']))
source=Path(config['runtime']['unirig_root'])/'src/model/unirig_ar.py'
tree=ast.parse(source.read_text())
node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='VocabSwitchingLogitsProcessor')
scope={'torch':torch,'LogitsProcessor':object,'TokenizerSpec':object,'LongTensor':torch.LongTensor,'FloatTensor':torch.FloatTensor}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
checks=0
sequences=[]
for frame in range(2):
    ids=target['main_input_ids'][0,frame][target['main_attention_mask'][0,frame]]
    start=ids[:2]
    official=scope['VocabSwitchingLogitsProcessor'](tokenizer,start)
    adapter=original_xyz_vocabulary_processor(tokenizer,start)
    for stop in range(2,len(ids)):
        prefix=ids[2:stop][None]
        scores=torch.arange(tokenizer.vocab_size,dtype=torch.float32)[None]*.01
        assert torch.equal(official(prefix,scores.clone()),adapter(prefix,scores.clone()))
        checks+=1
    sequences.append(ids.tolist())
result={'status':'passed','prefix_checks':checks,'selected_asset':row['asset_id'],
        'distinct_frame_targets':len({tuple(s) for s in sequences}),
        'main_token_lengths':[len(s) for s in sequences],
        'upstream_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'changed_upstream_files':False}
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
