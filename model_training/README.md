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

The single-target/all-frame comparison is complete with unchanged multi-frame
input, no virtual path and the same20kasset budget. F1 is0.66660single versus
0.71108all-frame; the paired-asset95%interval for single-minus-all is entirely
negative. The single-frame baseline is lower in this comparison; the proposed reference
skeleton/XYZ design is not yet tested. No further compute is automatically
released. See the [single-query result](docs/SINGLE_QUERY_SUPERVISION_20261026.md#completed-result).

The fixed random two-target20k point is also complete: F1 is0.69958, improving
single-target0.66660 with9.0%more update time and saving22.8%time versus all-frame.
Two-minus-all F1 is-0.01150 with95%interval[-0.03152,+0.00832]; this is not proof
of equivalence. All three share inputs and initialization. No further run is
authorized. See the [three-arm result](docs/TWO_QUERY_SUPERVISION_20261026.md#completed-result).

The subsequent three-frame and random10%-30% arms also completed fresh20k runs.
Three-frame F1 is0.70624 with116.7update minutes; random-fraction F1 is0.69944
with113.9minutes. Three has a higher point estimate than two at nearly the same
measured update cost, but the difference95%CI[-0.00696,+0.02154]includes zero.
Fractional supervision has fewer failures but lower exact joint-count accuracy
(39.65%versus two46.39%). All five arms retain the same input/init/20k contract.
Eight new full checkpoints have verified durable mirrors; all execution gates
are closed. See the [five-arm result](docs/QUERY_DENSITY_SUPERVISION_20261026.md#completed-result).

The prior matched virtual-module on/off comparison is complete at20,000exposures per
arm, with four full passes over the fixed5,000assets. Both arms retain real-frame
temporal interaction: its benefit is established and is not being tested again.
Original-GT edge F1 is0.69050on versus0.71108off; the paired95%interval for the
difference includes zero. No incremental virtual-module benefit was observed.
The100,000-exposure LRhorizon was preserved, not completed. Its permission does
not authorize an extension. See the [result and protocol](docs/VIRTUAL_QUERY_MATCHED_5K_20261025.md#completed-20k-result).

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
