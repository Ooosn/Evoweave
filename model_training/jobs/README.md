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

## Rejected Static Cross-Attention Residual

The old diagnostic launcher is:

```text
run_rootless_flat_static_motion_residual_20260723.sh
```

Do not submit this route for formal training. Its 5k diagnostic learned nearly
uniform attention over all 1024 dynamic keys: normalized attention entropy was
`0.999902`, the effective key count was `1023.3/1024`, and `99.9999%` of the
residual energy was a token-shared vector. Normal versus zero motion changed
the fused condition by only `0.000008` of the static-condition RMS. It therefore
implemented a global bias rather than anchor-specific motion.

The static side also calls UniRig on zero-padded collated mesh vertices without
using `vertex_count`. That makes its input depend on the largest mesh in the
batch. Results from this route are retained only as causal evidence.

## Anchor-Aligned Motion-Residual Candidate

The only active residual preflight is:

```text
run_rootless_flat_anchor_motion_residual_20260723.sh
```

For anchor `q`, this route combines only the frame-0 surface token at `q` with
the motion encoder output at the same `q`; it has no `Q x Q` fusion attention.
The final residual layer is zero initialized, so step 0 is exactly the
trackable frame-0 condition. The surface tokenizer, motion encoder, AR decoder,
and residual fuser remain trainable.

Before any short training, the untrained official UniRig decoder must be
evaluated on the same fixed rows with two conditions: official static UniRig
sampling and trackable frame-0 sampling. The candidate is invalid if the
trackable condition does not preserve usable static generation. If that check
passes, the 5k diagnostic keeps the formal `1667`-step OneCycle horizon and
uses `--stop-after-samples 5000`; it must not compress OneCycle into 105 steps.

This launcher is intentionally not a general experiment wrapper. It rejects
dynamic init/resume checkpoints, checks the exact final manifests and GPU
count, requires a clean immutable Git commit, preserves effective batch 48,
and runs `audit_static_motion_residual_contract.py` before torchrun. Stack,
explicit-tree, action, branch-prior, oracle-prefix, condition-refresh, and all
recovery losses are disabled. A formal run is allowed only after matched
small-scale free-generation and condition-use evaluation; passing the startup
audit alone is not a quality claim.

## Joint-Token AR Variant

The joint-token AR launcher is:

```text
run_rootless_puppeteer_motion_baseline_20260707.sh
```

The Westlake four-GPU wrapper for the same route is:

```text
westlake_rootless_puppeteer_motion_fullft_20260708.sh
```

It defaults to random initialization. Set `JOB_PUPPETEER_CHECKPOINT` only for a
deliberate pretrained-initialized variant.

It uses a separate training entry point:

```text
rigweave/scripts/train_puppeteer_dynamic_rig.py
```

This route must not call `rigweave/scripts/run_dynamic_ar_train.sh`, because
the target representation and decoder family are different from the UniRig AR
route. It may initialize its decoder from Puppeteer weights, but the model
contract is Evoweave-owned and must also work from random initialization.

The target is joint-token form:

```text
x, y, z, parent_index
```

The data contract is the current rootless NPZ: `target_joints_rootspace` plus
`target_parents`, with joint 0 as the single rootless skeleton root. Tail rows
are not generated target tokens.

A pretrained-initialized run uses:

```text
PUPPETEER_CHECKPOINT=/path/to/puppeteer_skeleton_w_diverse_pose.pth
```

For from-scratch training:

```text
RANDOM_INIT=1 bash jobs/run_rootless_puppeteer_motion_baseline_20260707.sh
```

When launched from this repository, the job script resolves `MODEL_ROOT` from
its own location and uses the bundled third-party references:

```text
${MODEL_ROOT}/third_party_references/UniRig
${MODEL_ROOT}/third_party_references/UniRig_hf
${MODEL_ROOT}/third_party_references/Puppeteer
```

Do not point this launcher at the older standalone model-training copy unless
that is the explicit experiment being run.

The Puppeteer baseline trains all modules at the baseline LR scale rather than
using a conservative decoder finetune LR:

```text
decoder lr: 1e-4
joint_slot_embedding lr: 1e-4
surface/motion/prefix lr: 1e-4
weight_decay: 0.04
scheduler: onecycle
target_coord_scale: 0.25
formal A100 run: nproc=4, micro_batch=3, grad_accum=4
effective_batch: 48
```

The formal condition/decoder contract is:

```text
query_tokens: 1024
cond_length: 1024
condition_projection: identity
decoder_norm_style: pre
joint_slot_embedding: enabled
train_random_query: enabled
scheduler: onecycle
```

The identity route preserves all 1024 ordered surface-anchor tokens. The old
learned-query `cross_attention` route compressed them to 257 anonymous slots
and collapsed mesh/pose separation in controlled retraining. The old inherited
post-LN decoder also failed to learn from random initialization. The training
entry point enforces the approved contract by default; disable the check only
inside an explicitly named diagnostic run.

`target_coord_scale` is strict affine scaling into Puppeteer's `[-0.5, 0.5]`
coordinate token range. It is not clipping. Smaller scales are required because
some accepted rootless-v3 skeletons extend beyond the query mesh bbox.

The joint-token route is a separate wrapper. It uses ordinary DDP by default;
enable `--ddp-find-unused-parameters` only for diagnostics if a future branch
creates unused trainable parameters.

Before formal training, run a one-GPU preflight with batch size 1 and
`RIGWEAVE_PREFLIGHT_CONTRACT_SANITY=1`. This verifies that teacher-forcing
logits and manual generation logits match for the same prefix; if it fails by a
large margin, the training/generation contract is wrong and no training job
should be submitted. The default bf16 FlashAttention tolerance is `3e-2`; a
one-token forced/prefix shift is much larger, about `4e-1` max logit diff in the
current diagnostic.

For a single-GPU development check, use:

```text
JOB_NPROC=1 JOB_BATCH_SIZE=3 JOB_GRAD_ACCUM=16
```

For the formal four-GPU Westlake run, keep micro batch unchanged and preserve
the same effective batch:

```text
JOB_NPROC=4 JOB_BATCH_SIZE=3 JOB_GRAD_ACCUM=4
```

Resource selection is operational policy rather than part of the model
contract. Follow the current Westlake/HGC workflow instructions and do not
encode resource-group fallbacks in this launcher.

The same launcher can run in an already allocated HGC shell by overriding the
environment bootstrap without changing the model profile:

```text
JOB_CONDA_SH=/home/wangyy/miniconda3/etc/profile.d/conda.sh
JOB_CONDA_ENV=mygs
JOB_OPT_CONFIG_ROOT=/home/wangyy/evorig/model_assets/hf_configs/facebook-opt-350m
JOB_EXTRA_PYTHONPATH=/home/wangyy/evorig/runtime/puppeteer_py310_hgc
```

Set `RIGWEAVE_NO_SAVE_OPTIMIZER=1` for checkpoint-only training runs where
resumption with optimizer state is not required.
