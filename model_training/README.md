# Evoweave Model Training

This is the canonical model, training, evaluation, and model-side diagnosis
module.

Read these files before model work:

1. `model_training/state/current.json`
2. `model_training/docs/CURRENT_MODEL_CONTEXT.md`
3. `model_training/docs/EVIDENCE_AWARE_MOTION_CONDITIONING.md`
4. `DATASET_SOURCE_OF_TRUTH.md`

Run the context guard before inspecting, implementing, evaluating, or training:

```bash
python model_training/tools/agent_work_guard.py begin
```

## Current Scope

The only current baseline is the Westlake flat UniRig model trained on the
final rootless-v3 manifests. The active research task is a local relative-motion
evidence path supervised by skin relations.

Puppeteer, stack-close, sibling perturbation, condition refresh, anchor/static
residuals, centered-motion, query-rigid, oracle-prefix, and explicit-tree work
are historical or rejected routes. They are not active baselines and must not
be resumed from their checkpoints.

HGC checkpoints are not part of the current Westlake experiment chain.

## Data Boundary

This module only consumes the finalized rootless-v3 train/valid manifests. It
must not scan the NPZ directory directly or modify data-processing code.

The model contract is:

```text
rootless-v3 dynamic NPZ
-> random query pose with matching mesh and skeleton target
-> 1024 ordered query-aligned surface anchors
-> autoregressive rootless skeleton generation
```

Targets contain neither a synthetic root nor generated tail tokens.

## Current Execution Boundary

The first task is an evidence-only held-out analysis. It uses dynamic vertices,
surface neighborhoods, and skin weights and does not load a skeleton checkpoint.

Only after that evidence passes may code add a separate motion-evidence memory
to the flat UniRig decoder. Full training and task submission remain blocked
until the development preflight passes. One formal two-A100 opportunity remains.

## Historical Documents

Other documents under `model_training/docs/` record prior experiments and causal
evidence. They are not current instructions unless the current state explicitly
links to them.
