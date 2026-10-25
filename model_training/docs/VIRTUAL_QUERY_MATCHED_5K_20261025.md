# Matched Virtual-Module Training On Fixed 5k Assets

## Authorized 20k Continuation

The latest human request authorizes a new two-H100 allocation and a first matched
comparison at20000asset exposures per arm. This operational cap does not change
the fixed5000training identities, the original100000-exposure OneCycle horizon,
or the deterministic sampling plan. V-off begins from the accepted official
initialization; V+continues its retained5000checkpoint including Adam/RNG/cursor.
Do not automatically continue past20k. Completion of this milestone is distinct
from completion of the originally planned100k horizon.

The latest CPU default is8slots per H100, hence16total for this pair. Before
submission the selected queue had71allocatable CPU slots and4available H100s.
Job130494177 started19:33:29JST onOctober25 with16slots, H100devices3/4 and64GiB
total host-memory reservation (mem_req4G per slot). Its keeper is a real scheduler
child in both devices/cpuset cgroups; no SSH-membership exception is used. The
24hour grant endsOctober26 at19:33:29JST; checkpoint handoff deadline is19:03JST.
All ranks/helpers will be bound to16selected distinct physical cores within the
job's allowed CPUs, with numerical-library threads kept at1.

CPU input work previously ran serially inside each GPU rank. Ordered loading now
uses6CPUworker threads per rank and at most2queued microbatches; it parallelizes
the unchanged NPZ/root-RT/frame-selection/normalization/normals loader. GPU surface
sampling and target construction remain in the owning rank, with the original
request-specific seeds. Prefetch uses a separate planner; checkpoints contain
only the consumed training cursor. CPU audit compares all loaded fields exactly
on the existing stress groups and records timing without treating cache-biased
microbenchmarks as production throughput. Real trace data-wait and step timing
are recorded after admission.

The20k evaluation uses the unchanged128validation assets and8views each. Partial
attempts are preserved and retried in new directories. Completion binds hashed
row artifacts to the exact checkpoint, source and selection. The final paired
report checks identical raw frames, continuous GT, normalization and target
tokens, and reports V+minusV- F1 with asset-level bootstrap intervals. Those
intervals describe validation-asset uncertainty for one training seed, not
variation across independently trained seeds. The original GPU capacity receipt
may be inherited only through an exact, primary-reviewed source transition;
model/data math and optimizer settings are unchanged. No new formal compute is
released before fresh CPU equivalence and scheduler admission pass.

CPUaudit_v7 passed at20:35JST, with unchanged initial fingerprint and sampling
plan,36bitwise-equal CPUinput comparisons, unchanged RNG and all6fulltwo-rank
queue stress cases accepted. Peak process RSS19172925440bytes (17.9GiB), below
the48GiBdata threshold that leaves16GiB for the previously measured model/runtime.
The65focused CPUregressions pass. Primary accepted the exact source transition
for the old GPUcapacity receipt and V+checkpoint, then released onlytrain_v2.
This acceptance is execution integrity, not a quality or throughput conclusion.

The first command-job checks stopped before Torch because qstat was absent from
PATH and host GPUindices were not job-local indices. The scoped launcher now
loads official scheduler settings and queries/binds the granted UUIDs. Actual
job inventory showed logical0/1map to host3/4; unauthorized device opens remain
denied. No device controls were changed and no SSHcompute fallback was used.

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

The two-arm capacity stage subsequently completed exit0 at16:10:24JST. Each arm
passed6stress configurations with2optimizer steps and2accumulation microbatches
per case on both ranks. All expected gradient routes were finite/nonzero; disabled
virtual parameters had no gradients. Peak reserved memory was66137882624bytes(V+)
and63789072384bytes(V-); peak process-tree RAM11829301248bytes. The original
profile weights were discarded and payload cleanup completed. Primary accepted
execution and released only the exact fresh train_v1 configuration. This is not
evidence of model quality or of a virtual-module benefit.

Formal train_v1 has started from the accepted official-initialized state under
producer commit `8feea86c24be80543befca1c374c1b57ae6c93c4`, supervisor922127 and
launcher922156. Primary verified the first3real updates on both ranks at16:18JST:
48assets/update,144exposures,1920supervised frames, exact shared sampling plan and
initialization, finite nonzero gradient routes. At16:22:20JST both ranks had
completed10updates,480exposures and6720frames on V+; V- had not started. This is
the full5000-asset stream, not the capacity cohort. No model-quality comparison
is available at this unequal progress. The100k endpoint per arm is the primary
comparison; intermediate matched-budget curves do not replace it.

The running producer checkout is kept unchanged. Startup evidence is under
`D:/evoweave/server_ops/outputs/virtual_query_matched_5k_20261025/startup_v1/`.
A bounded read-only watcher collects the terminal stage and verifies checkpoint
receipts/file presence before alerting primary. Authority to request the same
two H100s after the current grant expires is still pending; no extra allocation
has been requested.

## Allocation Handoff

The stage exited0 at17:10:45JST after V+reached5000exposures,105updates and66928
supervised frames. Primary checked both complete training traces:2500asset
requests per rank, exactly5000distinct manifest indices0..4999, each seen once.
V- is still0 and has no checkpoint. This is one full pass over the required5k
training set, not completion of the100k-per-arm comparison. No quality evaluation
or architecture conclusion is claimed at this unequal progress.

The full V+model/Adam/all-rank RNG/sampler checkpoint is8223925514bytes at both:

```text
/work/wangyy/virtual_query_matched_5k_20261025/outputs/train_v1/virtual_on/checkpoint_005000.pt
/home/wangyy/evorig/outputs/virtual_query_matched_5k_20261025/virtual_on/checkpoint_005000.pt
```

Producer-verified SHA256:
`2edb0f6f487c090caebdd8716a9505b103eca8ab8dd0398882b26bb0f0cb6727`.
Independent terminal inspection confirmed both files' existence, equal byte
counts and matching JSON receipts; it did not download or rehash the payloads.
Model fingerprint is
`5a272a6e5f7a610cd5168c04b5935248d1f77e747bcf4643a86370182cf5c8ed`.
Both ranks agree on progress, their payloads are gone, GPU5/6use0MiB, and keeper
1090013/the current allocation remain. Resource and verification receipts are
under the local `runs/train_v1` and `monitor_train_v1` artifact directories.

Current launch permissions are closed. The objective remains incomplete. Future
authorized continuation should start the less-trained V- arm from the identical
official initialization and later restore V+including optimizer/RNG/sampler,
without changing the recipe or100k horizon. New-allocation authorization is
pending; no replacement GPUs or automatic resume were initiated.
