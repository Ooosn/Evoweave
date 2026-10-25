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

The current task is a matched virtual-module on/off pair using the legacy codec,
the fixed 5,000-asset subset, dynamic 2..24 frames and unchanged root-RT
preprocessing. Every real frame is supervised. Both arms start from identical
official-initialized state under the same 100,000-exposure learning-rate horizon.
The currently authorized continuation stops both at 20,000 exposures for a matched
evaluation; V+ resumes its 5,000 checkpoint and V- starts at zero. The new grant
has two H100s and 16 CPU slots, with bounded ordered CPU-data prefetch configured;
auxiliary and cross-frame consistency losses are both zero. No small-data fitting
substitutes for this comparison. See the
[protocol](docs/VIRTUAL_QUERY_MATCHED_5K_20261025.md). The completed ID/XYZ run and
original single-query baseline evaluation remain separate historical results.

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
