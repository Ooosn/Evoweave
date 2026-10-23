# Original Flat UniRig Baseline

## Method Identity

The original baseline conditions a complete UniRig skeleton decoder on a
single real query pose plus other observed poses of the same asset. Each asset
visit supervises one query skeleton. This is the baseline trained on Westlake
and then reproduced on HGC, before the later all-target and A/B experiments.

1. Select one random eligible query frame and place it in input slot 0.
2. Sample surface points and form 1,024 query-aligned anchor tokens.
3. Encode the selected poses with the spatial/temporal motion encoder.
4. Pass the query's 1,024 condition tokens to the full UniRig autoregressive
   decoder; train against that query's complete posed skeleton sequence.

The model uses 12 motion blocks, 8 attention heads and 96 auxiliary register
tokens. Registers supplement the anchor tokens; they are not the AR condition
sequence. The historical implementation trains surface, motion and AR modules.

## Source

The existing launch chain is:

```text
model_training/jobs/run_rootless_flat_unirig_motion_baseline_20260706.sh
  -> model_training/rigweave/scripts/run_dynamic_ar_train.sh
  -> model_training/rigweave/scripts/train_dynamic_rig.py
```

| Component | Existing implementation |
| --- | --- |
| Surface tokens | [surface_tokenizer.py](../rigweave/src/rigweave/dynamic_rig/surface_tokenizer.py), FixedQuerySurfaceTokenizer |
| Motion encoding | [motion_encoder.py](../rigweave/src/rigweave/dynamic_rig/motion_encoder.py), AnchorWiseAlternatingMotionEncoder |
| Conditioning | [model.py](../rigweave/src/rigweave/dynamic_rig/model.py), DynamicRigConditioner |
| UniRig AR integration | [unirig_wrapper.py](../rigweave/src/rigweave/dynamic_rig/unirig_wrapper.py), DynamicRigUniRigAR |
| Query selection and targets | [data.py](../rigweave/src/rigweave/dynamic_rig/data.py), query in selected slot 0 |
| Dynamic frame count | [train_dynamic_rig.py](../rigweave/scripts/train_dynamic_rig.py), --frames-min and --frames |

## Established Runs

| Run | Data and change | Measured result | Interpretation |
| --- | --- | --- | --- |
| July HGC reproduction | 15,920 train / 856 valid; fixed T=24 | 1,667 updates; 80,016 nominal exposures; sample80000 checkpoint verified present on October 24 | Original trained baseline recovered |
| October 10 bias_h8 | Same historical manifests; T=2..24 and 8-head motion-evidence bias | Completed 1,667 updates at source fb27ce9; final inline validation CE 1.159892 at step 1,600 | Dynamic-frame training completed; full-budget generation superiority remains unestablished |
| October 13 root-RT integration | Expanded, screened 18,002 / 967 manifests; inverse root RT and selected-input mesh-union normalization | Consumer source 6ea3be2; 48 CPU tests and 80 real-loader checks | Data and loader integration verified; this publication itself launched no training |

The October 10 run precedes the October 13 RT publication. Its completion
establishes dynamic-frame training, not a completed run combining dynamic
frames, RT removal and expanded data. The user recalls that combined version
working; the matching completed-run record remains to be located.

## Historical Training Recipe

The fixed historical recipe uses two H100s, batch 3 per GPU, accumulation 8,
and nominal global batch 48. It uses BF16, motion checkpointing, AdamW weight
decay 0.04 and gradient clipping at 1. The OneCycle schedule lasts 1,667 updates,
with peak LR 1e-4 for all three groups, warmup fraction 0.1, division factor 5,
and final division factor 10.

For the October 10 comparison, 80,016 is the nominal step-based exposure
count; the partial-minibatch loader processed 79,996 actual examples. Keep
those counts distinct when comparing training budgets.

Fresh training initializes surface/AR from official UniRig weights and creates
a new motion encoder. The October 10 full run did not resume the July final
checkpoint or a short-screen checkpoint. Its seed was 20260529. Historical
initialization RNG was not saved, so bitwise recreation of July motion weights
is not established.

The exact historical arguments, asset paths and candidate settings are in
[motion_evidence_base_compare_20261010.json](../experiments/motion_evidence_base_compare_20261010.json).
That file's baseline block also records a historical crash-resume checkpoint;
it describes that old run, not initialization for the requested fresh run.

## Evidence Locations

- July HGC run:
  `/home/wangyy/evorig/outputs/flat_unirig_hgc2h100_matched80k_20260717`.
  `checkpoint_sample_80000.pt` was verified present, 7,995,123,574 bytes.
- Original Westlake run identity:
  `rootless_flat_unirig_motion_fullft_20260707_hxr4gpu`;
  its historical path is retained in `state/current.json: reference_runs.flat_unirig`.
- October 10 run:
  `/home/wangyy/evorig/outputs/motion_evidence_base_compare_20261010/full/bias_h8`.
  Completion and checkpoint inspection receipts are under
  `/home/wangyy/jobs/motion_compare_hgc_20261010/results/`:
  `full_bias_h8_completion.json` and
  `full_bias_h8_final_checkpoint_inspection.json`.
  Completion receipts were verified; the final weights were absent from their
  recorded run paths on October 24, and their current location is unresolved.
- Scientific comparisons and diagnostics:
  [motion-evidence results](MOTION_EVIDENCE_COMPARISON_20261010.md).
- Current preprocessing and immutable manifests:
  [dataset source of truth](../../DATASET_SOURCE_OF_TRUTH.md).

## Measured Generation Quality

The July 18 matched heldout-52 evaluation uses the same assets, query frames,
24 selected frames, normalization and GT joint counts for all checkpoints.

| Checkpoint | Parsed | Hit token limit | Joint-count MAE | J2J | J2B | B2B | Topology F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Flat UniRig sample80000 | 42/52 | 10 | 6.0952 | 0.038961 | 0.027517 | 0.022120 | 0.639410 |
| Puppeteer sample80000 | 52/52 | 0 | 43.5192 | 0.053189 | 0.044258 | 0.038557 | 0.202308 |
| Puppeteer best_val | 52/52 | 0 | 44.0000 | 0.054615 | 0.046626 | 0.040644 | 0.197214 |
| Puppeteer final | 52/52 | 0 | 43.6154 | 0.052854 | 0.043918 | 0.038224 | 0.199103 |

Flat UniRig's geometry and F1 above cover only its 42 parsed cases. J2J, J2B
and B2B are joint-to-joint, joint-to-bone and bone-to-bone distance measures;
smaller is better. MAE is mean absolute joint-count error. Parsing success
means the output can be decoded, not that its skeleton is correct.
Heldout-52 deliberately includes 36 low-joint-count cases (4..10 joints);
it is a stress panel, not a representative validation sample.

| Heldout-52 stratum | Flat parsed | Flat successful-case F1 | Puppeteer sample80000 F1 |
| --- | ---: | ---: | ---: |
| 16 training assets, 4..10 joints | 11/16 | 0.486790 | 0.019744 |
| 16 training assets, 22..92 joints | 15/16 | 0.951028 | 0.599009 |
| 20 validation assets, 4..10 joints | 16/20 | 0.452194 | 0.030999 |

A separate seed-101, 1,400-token-limit evaluation found:

| Panel | Parsed | Successful-case F1 | All-case F1, failures = 0 |
| --- | ---: | ---: | ---: |
| Fixed random 60 training assets | 56/60 | 0.8472 | 0.7907 |
| valid-common60 | 54/60 | 0.7935 | 0.7141 |
| valid-common60: 52-joint stratum | 20/20 | 0.9490 | 0.9490 |
| valid-common60: 20..75 joints, excluding 52 | 18/20 | 0.7609 | 0.6848 |
| valid-common60: 76..101 joints | 16/20 | 0.6357 | 0.5086 |

The baseline learns useful skeletons, especially the frequent 52-joint family;
termination and rare-topology generalization remain limitations. The
valid-common60 panel is stratified, so 0.7141 is its score, not a full-dataset
estimate. Evidence is under
`/home/wangyy/evorig/outputs/matched_heldout52_20260718/`, including
`flat_distribution_audit_20260718/flat_train_random60_seed101.json` and
`flat_valid_common60_seed101.json`. Causal diagnoses are in
[flat UniRig hitmax](FLAT_UNIRIG_HITMAX_CAUSAL_DIAGNOSIS_20260719.md) and
[Puppeteer long-tail analysis](PUPPETEER_TOPOLOGY_LONG_TAIL_DIAGNOSIS_20260718.md).

Two historical interpretation checks matter: Puppeteer F1 0.751/0.796 came
from four training rows, while its eight-row validation scores were
0.117/0.159. Also, on 16 low-joint cases, correct rather than cross-asset
conditioning reduced coordinate NLL from 4.4735 to 2.6749 and helped all 16.
That establishes use of the condition, not generalization to arbitrary rigs.

## Requested Fresh Run

Use the existing single-query implementation with dynamic T=2..24, current
root-RT preprocessing and the fixed 5,000-asset subset. The requested budget is
100,000 actual asset visits, equivalent to 20 nominal passes over that subset,
on two H100s. One visit supervises one query skeleton.

This changes data membership and budget relative to the historical recipe.
The exact restored launch configuration and schedule still need to be bound
to the recovered implementation. The later A/B reference is a comparison
result, not the source of this baseline's identity or historical recipe.
