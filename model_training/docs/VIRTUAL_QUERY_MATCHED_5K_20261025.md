# Matched Virtual-Module Training On Fixed 5k Assets

## Purpose And Authorization

The human wants to distinguish a benefit of virtual queries from decoder/training
confounds, not test whether pose information is used. On October25 the human
authorized fresh training with experiment-design discretion, explicitly requiring
the existing5000-asset set and rejecting small-data fitting as a research substitute.
This stage keeps the legacy codec fixed and compares virtual-module on/off.
The ID-codec factor is deferred, not silently reinterpreted as this pair.

## Missing Historical B

The stop receipt from October21 explicitly retained B70 at
`normal_b_lr3e5_80000_v1/checkpoint_asset_070000.pt`,8438115504bytes,70000exposures;
its payload hash had not been audited. B65's recorded SHA256 was
`1555619a677f13bcb2e044c2b7e72cf623483490ab0edc0eac24cb24b8ead7f`.
The older October22 state called B65's absence intentional cleanup, but this does
not prove that deletion of the retained B70 was requested or correctly performed.
No deletion/move/archive receipt explaining B70's disappearance was found.
Fresh15:13JST inventory found no B60/B65/B70 payload in their original trees,
related work roots or identified backup locations. Other existing models and
the Structure/XYZ archives are not B substitutes. Global absence and the exact
cause of loss remain unproven. This run does not delete any historical artifact.

Evidence: local stop receipt under
`D:/evoweave/server_ops/outputs/virtual_fusion_20261013/b80_20261021/user_stop_20261021_v1/stop_receipt_v1.json`;
new inventory under
`D:/evoweave/server_ops/outputs/virtual_query_matched_5k_20261025/preparation/inventory.log`.

## Controlled Comparison

- V+: original B real-query/virtual-key-value computation, including virtual
  feedback in all temporal blocks. Auxiliary loss is0, so this is not an exact
  reproduction of historical B's auxiliary-weight0.1 training recipe.
- V-: no virtual frame in temporal attention and no final virtual cross-attention
  residual. Keep real-frame temporal attention, common FFN residuals, final norm,
  surface encoder and original full decoder unchanged.
- Both are constructed from the same official pretrained checkpoint and same
  random seed. All initial state tensors match. Unused virtual/cross/auxiliary
  parameters remain dormant in the state for identity checks, but receive no
  gradients and are excluded from Adam. Report active parameter counts; this
  tests removal of the module, not a parameter-count-matched substitute.
- Use the same legacy256-bin codec, all-real-frame targets, per-frame sequence
  mean CE then asset mean, auxiliary loss0 and cross-frame consistency loss0.
  No new pruning, coordinate clipping, pose intervention or small-fit stage.

## Data And Budget

Both arms use all5000 immutable selected training assets, SHA256
`a17b6be96fb33a32a1bee7117b07789da6261c6b5693104f3d2cce07c09bab88`.
Keep retained-root RT, selected-input sequence-union mesh normalization,
dynamic2..24frames and8reserved poses. The967validation pool is disjoint.

Each arm has100000asset exposures, about20complete passes, with effective
batch48 and2100updates (exact5k boundaries give a final short update per pass).
Both use one100k-horizon OneCycle schedule from the start: peak1e-4, warmup10%,
div_factor5, final_div_factor10, AdamW weight_decay0, gradient clip1.
There are no retrospective learning-rate horizon extensions.

One pair configuration controls both arms. Identical deterministic full-permutation
asset order, global frame-count choices, raw-pose seeds and surface-sampling seeds
are recorded. Run at most one model on the2GPUs at a time, alternating5k-exposure
segments and preserving each arm's model/Adam/RNG/sampler state. Physical batches
are24/12/6 per rank for T<=6/T<=12/T<=24 after real capacity acceptance.

The same128validation assets and8scored input frames per asset are evaluated at
25k/50k/75k/100k. Use original continuous GT F1, failures as0, count errors and
non52/gt52 strata. No oracle or teacher-forced token accuracy substitutes for
generated skeleton quality. Compare arms at equal exposures; incomplete or
unequal-progress scores do not establish architecture superiority. One paired
seed is a first causal comparison under this recipe, not universal proof.

## Admission And Retention

CPU checks restore the actual official weights, verify full common initialization,
all5000manifest identities/RT selection evidence, evaluation membership and the
matched sampling plan. CPU tests cover V+ equivalence, V- path independence,
remaining temporal coupling, all-frame objective and exact model/Adam/sampler
checkpoint restoration. Real capacity checks use the existing full-5k stress
selection, not a small-data fitting experiment; their updated weights are discarded.
Primary must accept CPU and both-arm GPU capacity before training is released.

Allocation130309880, GPU5/6 onpcg01i, has1scheduler slot. The explicit SSH
execution exception enforces4total CPUs[58,63,64,65],64GiB RAM and85%VRAM.
No other process or GPU is touched. The grant expires17:35JST; the training
handoff deadline is17:10JST, leaving25minutes for checkpoint copy/cleanup.
Allocation expiry does not mean the100k-per-arm objective is complete.

Every5k checkpoint and deadline handoff includes full model, Adam, named optimizer
parameters, every rank's RNG and sampler, progress, source and configuration.
Write atomically on the verified NVMe `/work` storage, then copy each large file
to `/home/wangyy/evorig/outputs/virtual_query_matched_5k_20261025/<arm>/` and verify
SHA256 before publishing its receipt. Preserve every checkpoint; no automatic
cleanup, deletion, overwrite, release, or extra GPU allocation is authorized.
Dataset small files are not copied to HDD. Pending final evaluation prevents a
completed-training claim even when both arms have reached100k.

## Current Stage

Implementation and16CPU tests passed. Actual CPU audit completed exit0 at15:40:31JST,
without initializing CUDA. Both arms have exactly identical initial state,
fingerprint `5637def54e52b339063ea4338c432755c876f08a5520f96e2b45d73570331a3c`.
Active parameters are675917600(V+) and670666528(V-). All5000training identities,
967validation identities and the128-asset evaluation selection were verified.
Primary accepted this audit and released only both-arm GPU capacity. Formal
training remains blocked pending its measured acceptance.
