# Evoweave Workspace Map

This workspace is organized by module. Each module owns its own code, runbook,
contracts, and validation notes. New work should be added inside the appropriate
module instead of creating temporary scripts in the workspace root.

## Canonical Modules

### `data_processing/`

Canonical raw-data to training-NPZ pipeline.

Owns:

- raw source manifests;
- Blender Pass1 export from preserved originals;
- row-skeleton head/tail contract;
- sequence dedupe;
- Pass1 readability precheck;
- rootless NPZ rewrite before formal hard quality screening;
- rootless target strict hard screening;
- final histogram screening.

Read first:

- `DATASET_SOURCE_OF_TRUTH.md`
- `data_processing/README.md`
- `data_processing/docs/DATA_PIPELINE_CONTRACT.md`
- `data_processing/docs/CURRENT_DATASET.md`

### `server_ops/`

Server access, environment notes, job templates, and operational scripts. This
folder should not own dataset logic or model code.

### `model_training/`

Canonical model, training, checkpoint, and model-side diagnosis module.

Owns:

- dynamic rig model training/evaluation scripts;
- autoregressive skeleton generation diagnostics;
- tokenizer/representation experiments that affect training;
- hitmax causal analysis and generation traces;
- baseline run plans and model-side research directions.

Read first:

- `model_training/README.md`
- `model_training/docs/CURRENT_MODEL_CONTEXT.md`
- `model_training/docs/FLAT_UNIRIG_HITMAX_CAUSAL_DIAGNOSIS_20260719.md`
- `model_training/docs/EVIDENCE_AWARE_MOTION_CONDITIONING.md`

### `model_training_oldline_baidu_20260624/`

Isolated Baidu old-line model/training code snapshot.

Owns:

- old-line reproduction code from the 2026-06-24 Baidu handoff;
- historical June 28 probe job scripts;
- historical June 30 hitmax trace artifacts.

Use this folder for old baseline reproduction. Do not copy old-line fallback,
root/tail, checkpoint-continuation, or compatibility behavior into
`model_training/`.

Read first:

- `model_training_oldline_baidu_20260624/MODULE_BOUNDARY.md`

### `reports/`

Analysis outputs and one-off investigation reports. Reports are evidence, not
canonical pipeline code.

### `visuals/`

Visualization outputs and assets. Visualization scripts should eventually move
into a dedicated visualization module if they become reusable tools.

## Non-Canonical / External Context

These folders are not canonical module boundaries:

- `remote_current/`: checked-out remote project snapshot for reference.
- `server_outputs/`: copied server outputs.
- `server_live_20260703/`, `server_patch/`, `server_pull/`,
  `remote_repo_patch/`: operational or patch-transfer context.
- `rebuild_reports_20260704/`: historical rebuild reports.

Do not import pipeline logic from these folders unless it is moved into a
canonical module and the decision is documented.

## Workspace Rules

- Do not put new temporary scripts in `D:\evoweave`.
- The current training dataset is defined by `DATASET_SOURCE_OF_TRUTH.md`.
  Never infer current data from report folder names such as `latest_*`.
- Do not use old official NPZ archives as data-processing inputs; they are
  allowed only as external oracle/comparison data.
- Data-processing quality gates must live in the formal data pipeline, not in
  post-hoc quarantine scripts.
- The histogram stage is the final data-screening layer.
- Model training is owned by `model_training/`. Tokenizer work that changes
  training semantics should be documented there until it becomes its own module.
- Evaluation and visualization should each get their own module directory before
  they become active workstreams.
