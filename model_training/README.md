# Evoweave Model Training

This module owns model architecture, training, generation, and evaluation.
Data production is owned by `data_processing/`.

## Research Entry Points

| Layer | Document | Purpose |
| --- | --- | --- |
| Project status | [Current model context](docs/CURRENT_MODEL_CONTEXT.md) | Research objective, current task, results, and open questions |
| Method | [Original flat UniRig baseline](docs/BASELINE_FLAT_UNIRIG.md) | Model identity, source entry points, historical recipe, and verified runs |
| Experiments | [Result index](docs/CURRENT_MODEL_CONTEXT.md#研究结果索引) | Questions, measurements, conclusions, and detailed evidence |
| Data | [Dataset source of truth](../DATASET_SOURCE_OF_TRUTH.md) | Current membership, root-RT alignment, and normalization |
| Execution | [Machine state](state/current.json) | Exact operation permissions, configurations, and receipts |

The current task is a 20,000-exposure real-frame ID/XYZ autoregressive trial
on the fixed 5,000-asset subset, with dynamic 2..24 frames and unchanged root-RT
preprocessing. Every real frame is supervised by the same full decoder;
structural consistency is evaluated under ground-truth prefixes. See the
[protocol](docs/ID_XYZ_CONSISTENCY_20261024.md). The original single-query
baseline comparison is deferred, not redefined as this experiment.

## Code Entry Points

- Current training: [train_id_xyz_consistency.py](rigweave/scripts/train_id_xyz_consistency.py).
- Current launcher: [launch_id_xyz_consistency.py](jobs/launch_id_xyz_consistency.py).
- Original baseline training: [train_dynamic_rig.py](rigweave/scripts/train_dynamic_rig.py).
- Generation: [eval_dynamic_rig_generation.py](rigweave/scripts/eval_dynamic_rig_generation.py).
- Teacher-forced evaluation: [eval_dynamic_rig_ce.py](rigweave/scripts/eval_dynamic_rig_ce.py).
- Model implementation: `rigweave/src/rigweave/dynamic_rig/`.
- Experiment recipes: `experiments/`; measured summaries: `analysis/`.

Historical experiment documents describe their own datasets and budgets.
Current execution permissions are maintained only in `state/current.json`.
