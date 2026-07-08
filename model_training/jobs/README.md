# Model Training Jobs

This folder stores model-training job scripts.

## Immediate Baseline

The current clean first baseline is:

```text
run_rootless_flat_unirig_motion_baseline_20260706.sh
```

It consumes the finalized rootless-v3 train/valid manifests, sets
`RIGWEAVE_TARGET_ROOT_POLICY=legacy` and `RIGWEAVE_TARGET_START_POLICY=joint0`,
uses the rootless `target_joints_rootspace` / `target_parents` fields through
the existing flat UniRig tokenizer path, and full-finetunes the UniRig AR
decoder, dynamic conditioner, and surface tokenizer.

Default manifests:

```text
/ssdwork/liuhaohan/evorig/evoweave_rebuild_rootless_v3_20260706/quality_distributions/rootless_bbox_consistency/final_manifests/train_manifest.jsonl
/ssdwork/liuhaohan/evorig/evoweave_rebuild_rootless_v3_20260706/quality_distributions/rootless_bbox_consistency/final_manifests/valid_manifest.jsonl
```

The launcher reads the manifest entries. It must not rescan the NPZ directory.
The baseline launcher uses LR `1e-4`, AdamW weight decay `0.04`, OneCycle
scheduling, and effective batch `48`.

It is the old-style flat decoder control run on the current cleaned data
contract.

The run starts from the official UniRig skeleton checkpoint, not from the old
Evoweave 80k dynamic checkpoint. Supplying `JOB_INIT_CHECKPOINT` changes the
experiment into a continuation/probe variant and should not be labeled as this
clean baseline.

## Puppeteer Variant

The Puppeteer decoder-backbone launcher is:

```text
run_rootless_puppeteer_motion_baseline_20260707.sh
```

It uses a separate training entry point:

```text
rigweave/scripts/train_puppeteer_dynamic_rig.py
```

This route must not call `rigweave/scripts/run_dynamic_ar_train.sh`, because
the target representation and decoder family are different from the UniRig AR
route.

The target is Puppeteer joint-token form:

```text
x, y, z, parent_index
```

The data contract is the current rootless NPZ: `target_joints_rootspace` plus
`target_parents`, with joint 0 as the single rootless skeleton root. Tail rows
are not generated target tokens.

A formal Puppeteer run requires:

```text
PUPPETEER_CHECKPOINT=/path/to/puppeteer_skeleton_w_diverse_pose.pth
```

For code-smoke only, not for baseline claims:

```text
RANDOM_INIT_SMOKE=1 TINY_RANDOM_DECODER=1 JOB_NPROC=1 JOB_MAX_STEPS=20 \
  bash jobs/run_rootless_puppeteer_motion_baseline_20260707.sh
```

The Puppeteer baseline trains all modules at the baseline LR scale rather than
using a conservative decoder finetune LR:

```text
decoder lr: 1e-4
target_aware_pos_embed lr: 1e-4
surface/motion/prefix lr: 1e-4
weight_decay: 0.04
scheduler: onecycle
target_coord_scale: 0.25
formal A100 run: nproc=4, micro_batch=3, grad_accum=4
effective_batch: 48
```

`target_coord_scale` is strict affine scaling into Puppeteer's `[-0.5, 0.5]`
coordinate token range. It is not clipping. Smaller scales are required because
some accepted rootless-v3 skeletons extend beyond the query mesh bbox.

The Puppeteer route is a separate wrapper. It uses ordinary DDP by default;
enable `--ddp-find-unused-parameters` only for diagnostics if a future branch
creates unused trainable parameters.

For a single-GPU development check, use:

```text
JOB_NPROC=1 JOB_BATCH_SIZE=3 JOB_GRAD_ACCUM=16
```

For the formal four-GPU Westlake run, keep micro batch unchanged and preserve
the same effective batch:

```text
JOB_NPROC=4 JOB_BATCH_SIZE=3 JOB_GRAD_ACCUM=4
```

The preferred formal training resource group is Westlake
`huangxiangru` (`groupId=c965c1ec-1ad4-43b6-a1d3-3bccad2667ba`) on 80GB A100
nodes such as `gvna17`. Do not submit this baseline to 40GB A100 nodes.
