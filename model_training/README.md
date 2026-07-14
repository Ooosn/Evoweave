# Evoweave Model Training

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

The joint-token route currently has a verified failure mode: end-to-end
conditioner training with pure teacher forcing can collapse pose conditioning
and learn pose from the GT prefix instead. Read
`docs/PUPPETEER_CONDITION_COLLAPSE_DIAGNOSIS_20260715.md` before submitting a
full Puppeteer run. The existing full-finetune job is not approved until its
training objective preserves query-pose information.
