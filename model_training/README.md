# Evoweave Model Training

Dataset revision (2026-10-13): use the immutable root-RT-cache HGC
manifests in [dataset authority](../DATASET_SOURCE_OF_TRUTH.md), containing
18,002 train / 967 validation assets. This revision adds 18 whole-asset
exclusions; the previous 81 remain excluded. Complete manifest rows, including
`root_rt`, are required by the RT-aware flat and virtual-frame data loaders.
They remove rigid root motion, preserve nonrigid residuals, and use one
selected-input mesh-union normalization with padding 1.1. Legacy Puppeteer and
the old shared wrapper fail closed on these manifests. Historical experiment
configs retain their original manifest identities for reproducibility.
Data preprocessing passed full screening, 48 CPU tests and 80 real-loader
checks; no model architecture was changed or training launched by this task.
The root cache depends on source skeleton metadata, not a mesh-only RT estimator.
Read `state/current.json` for operation status; historical baselines below
are not permission to restart an old-data job.

This is the canonical Evoweave model-side module. It owns model architecture,
training scripts, checkpoint workflows, and generation/evaluation code.

Data processing is owned by `D:\evoweave\data_processing`. This module consumes
the finalized rootless-v3 manifests and NPZ files; it should not edit
data-processing code.

## Current Scope

The current model module has two active baselines:

- flat UniRig on rootless-v3 data;
- Evoweave joint-token AR decoder on rootless-v3 data, optionally initialized
  from Puppeteer decoder weights.

Do not treat historical experiments as current context unless a new task
explicitly asks for them.

## Directory Layout

```text
model_training/
  README.md
  docs/
    CURRENT_MODEL_CONTEXT.md
    EVIDENCE_AWARE_MOTION_CONDITIONING.md
    FLAT_UNIRIG_HITMAX_CAUSAL_DIAGNOSIS_20260719.md
    PUPPETEER_CONDITION_COLLAPSE_DIAGNOSIS_20260715.md
    PUPPETEER_DECODER_BACKBONE_FEASIBILITY_20260706.md
  jobs/
    run_rootless_flat_unirig_motion_baseline_20260706.sh
    run_rootless_puppeteer_motion_baseline_20260707.sh
  rigweave/
    docs/
    scripts/
    src/
  third_party_references/
    Puppeteer/
```

## Current Code Source

`model_training/rigweave` was copied from:

```text
D:\evoweave\remote_repo_patch\rigweave
```

It is the current baseline code snapshot for this module. Future model edits
should happen inside `model_training/rigweave`.

Important entry points:

- `model_training/rigweave/scripts/run_dynamic_ar_train.sh`
- `model_training/rigweave/scripts/train_dynamic_rig.py`
- `model_training/rigweave/scripts/eval_dynamic_rig_generation.py`
- `model_training/rigweave/scripts/eval_dynamic_rig_ce.py`
- `model_training/rigweave/scripts/audit_skeleton_token_self_consistency.py`
- `model_training/rigweave/scripts/train_puppeteer_dynamic_rig.py`

## Baselines

UniRig flat baseline:

```text
model_training/jobs/run_rootless_flat_unirig_motion_baseline_20260706.sh
model_training/jobs/westlake_rootless_flat_unirig_motion_fullft_20260707.sh
```

Joint-token AR baseline:

```text
model_training/jobs/run_rootless_puppeteer_motion_baseline_20260707.sh
model_training/jobs/westlake_rootless_puppeteer_motion_fullft_20260708.sh
```

Both baselines use the finalized rootless-v3 train/valid manifests. The
joint-token route has its own decoder/tokenizer contract, with explicit
`(x, y, z, parent_index)` tokens and learned joint-slot embeddings. Puppeteer
weights are optional initialization, not a model-design requirement, and this
route must also pass from-scratch contract sanity checks.

The failed joint-token runs had two verified configuration defects: the
randomly initialized 24-layer decoder inherited Puppeteer's post-LN setting,
and a learned-query projector compressed 1024 ordered condition tokens into
257 anonymous slots. The approved baseline contract is now pre-LN,
identity-1024 condition transfer, joint-slot embedding, random query poses,
and OneCycle scheduling. The trainer enforces this contract before model
construction. Read `docs/PUPPETEER_CONDITION_COLLAPSE_DIAGNOSIS_20260715.md`
for the controlled retraining evidence and remaining generation checks.

The current flat UniRig hitmax mechanism is documented in
`docs/FLAT_UNIRIG_HITMAX_CAUSAL_DIAGNOSIS_20260719.md`. Its conclusion is based
on fixed-prefix condition interventions, repetition analysis, bounded prefix
repairs, full training-length statistics, and surface-sampling replay. Do not
treat max-token termination or forced EOS as a model-quality fix.
