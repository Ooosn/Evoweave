# Evoweave Agent Operating Contract

This file is an execution contract, not background documentation. Every agent
working in this repository must follow it before making claims or running work.

## Mandatory Session Start

At the start of a task, after context compaction, and after resuming an old
terminal:

1. Read `PROJECT_MAP.md` and `DATASET_SOURCE_OF_TRUTH.md`.
2. For model work, run:

   ```text
   python model_training/tools/agent_work_guard.py begin
   ```

3. Inspect the live Git HEAD, worktree, server process, and output artifact.
4. Do not answer a status question from conversation memory.

`begin` writes a short-lived ignored receipt tied to the current state hash,
context hash, and Git HEAD. A stale or missing receipt means the agent has not
loaded the current state.

## Status Reports

Before reporting model status, run:

```text
python model_training/tools/agent_work_guard.py check --operation report
```

A status report must separate:

- verified artifact or metric;
- interpretation;
- unknown or missing comparison.

Diagnostics are not model-quality progress. Training logs are not generation
quality. A metric without a matched baseline must not be described as good.

## GPU Work Gate

Before any training or fine-tuning command:

1. Update `model_training/state/current.json` with one `active_operation` that
   records the exact input manifest, source checkpoint, code commit, output
   directory, resource request, command/config, and acceptance test.
2. Commit and push that state change before allocating or using GPUs.
3. Run `begin` again on the server checkout.
4. Run `check --operation train` or `check --operation submit`.

If the operation is not explicitly allowed by current state, the guard must
fail and the GPU command must not run. Do not bypass the guard by calling the
Python trainer directly.

After completion, interruption, or failure, immediately update the same state
entry with result, artifacts, and accepted/rejected status, then commit and
push. Never start a second experiment while the first is unrecorded.

## Model Module Rules

- `model_training/state/current.json` is the machine-readable current state.
- `model_training/docs/CURRENT_MODEL_CONTEXT.md` explains that state for humans.
- Detailed historical documents are evidence, not active state.
- Existing accepted reference checkpoints must be evaluated before retraining.
- No fallback code, hidden compatibility path, or unrecorded smoke experiment.
- Do not change the data contract from the model module.

If state and live artifacts disagree, stop. Record the disagreement as unknown
and reconcile it before continuing.
