# Current Model Context

This document records the current model-side state before the next baseline run.
It is intentionally separate from future research ideas.

## High-Level Task

Evoweave learns to generate a complete canonical skeleton tree from a dynamic
mesh sequence. The current model path is motion-aware and autoregressive:

```text
rootless dynamic NPZ -> dynamic dataset -> motion/mesh conditioning -> AR skeleton decoder
```

The training target is the rootless skeleton produced by the data-processing
module. The model should not assume a synthetic root target.

## Current Verified Status

Use this section as the source of truth before discussing next model work.

- The active data is the finalized rootless-v3 manifest/NPZ contract.
- The flat UniRig rootless-v3 baseline has already been trained and evaluated.
  It is not a pending baseline.
- The current model module has two active baselines:
  - flat UniRig on rootless-v3 data;
  - Evoweave joint-token AR decoder on rootless-v3 data, optionally initialized
    from Puppeteer decoder weights.
- The joint-token route is our own model contract. It must pass fixed-prefix
  teacher-forcing/generation logit alignment and from-scratch sanity checks;
  Puppeteer weights are optional initialization only.

## Current Code Snapshot

The current module code lives under:

```text
D:\evoweave\model_training\rigweave
```

It was copied from:

```text
D:\evoweave\remote_repo_patch\rigweave
```

This is a snapshot to stabilize the model branch. Future edits should happen
inside `model_training/rigweave`, not in `remote_repo_patch`.

## Main Components

Important scripts:

- `rigweave/scripts/run_dynamic_ar_train.sh`: shell wrapper and environment
  passthrough.
- `rigweave/scripts/train_dynamic_rig.py`: distributed training entry point.
- `rigweave/scripts/eval_dynamic_rig_generation.py`: free-generation
  evaluation.
- `rigweave/scripts/eval_dynamic_rig_ce.py`: cross-entropy / teacher-forcing
  evaluation.
- `rigweave/scripts/train_puppeteer_dynamic_rig.py`: joint-token AR training
  entry point. Despite the historical filename, this is not allowed to depend
  on Puppeteer pretraining for correctness.
- `rigweave/scripts/audit_skeleton_token_self_consistency.py`: tokenizer /
  representation self-consistency audit.

Important source files:

- `rigweave/src/rigweave/dynamic_rig/data.py`: dynamic rig dataset and target
  extraction for model training.
- `rigweave/src/rigweave/dynamic_rig/unirig_wrapper.py`: model wrapper,
  conditioning, and flat UniRig decoder integration.
- `rigweave/src/rigweave/dynamic_rig/skeleton_contract.py`: skeleton contract
  helpers copied with the model snapshot.

Old raw-data and canonical-NPZ scripts from the source snapshot were not kept as
training entry points. Data-processing ownership is now centralized in
`D:\evoweave\data_processing`.

## Active Baseline Questions

The current model-side questions are:

- how the flat UniRig baseline behaves on the remaining rootless-v3 failure
  cases;
- how the joint-token AR baseline compares under the same rootless-v3 data
  contract;
- whether the model condition should use motion evidence more explicitly after
  the two baselines are measured cleanly.

## Boundary With Data Processing

The data-processing module owns:

- head/tail row-skeleton extraction;
- rootless target rewrite;
- component pruning;
- quality histograms and data gates.

This model module owns:

- how the rootless target is tokenized for training;
- how motion and mesh condition the model;
- how the decoder rolls out skeleton tokens;
- how generation failures are diagnosed and evaluated.

If a model-side issue appears to be caused by data, first document the evidence
and compare against the data contract instead of editing the data module from
this branch.
