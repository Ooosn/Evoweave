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

The matched virtual-module on/off comparison is complete at20,000exposures per
arm, with four full passes over the fixed5,000assets. Both arms retain real-frame
temporal interaction: its benefit is established and is not being tested again.
Original-GT edge F1 is0.69050on versus0.71108off; the paired95%interval for the
difference includes zero. No incremental virtual-module benefit was observed.
The100,000-exposure LRhorizon was preserved, not completed. No further compute
is authorized. See the [result and protocol](docs/VIRTUAL_QUERY_MATCHED_5K_20261025.md#completed-20k-result).

Keep `state/current.json` limited to current permissions, contracts and results.
Historical runs and source-transition details are [archived](state/archive/current_before_compaction_20261026.json);
archive permissions are never active.

## Code Entry Points

- Current training: [train_virtual_pair.py](rigweave/scripts/train_virtual_pair.py).
- Current launcher: [launch_virtual_pair.py](jobs/launch_virtual_pair.py).
- Original baseline training: [train_dynamic_rig.py](rigweave/scripts/train_dynamic_rig.py).
- Generation: [eval_dynamic_rig_generation.py](rigweave/scripts/eval_dynamic_rig_generation.py).
- Teacher-forced evaluation: [eval_dynamic_rig_ce.py](rigweave/scripts/eval_dynamic_rig_ce.py).
- Model implementation: `rigweave/src/rigweave/dynamic_rig/`.
- Experiment recipes: `experiments/`; measured summaries: `analysis/`.

Historical experiment documents describe their own datasets and budgets.
Current execution permissions are maintained only in `state/current.json`.
