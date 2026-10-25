# Frozen Query And Decoder Diagnostics

The human explicitly requested beginning the proposed staged investigation.
This first stage is inference only: no optimizer, backward, checkpoint rewrite,
loss adjustment or fresh training. Completed100k training remains preserved.

**Status: frozen ID100k diagnostics completed and execution accepted.** The full
paired stage finished exit0 at14:24:40JST on October25. B virtual-query ablation
remains blocked by missing weights; this is not completion of that causal test.

## Checkpoint Availability

B65 was previously deleted. A fresh13:04JST read found no weights in its original
normal-B output tree or the known virtual-fusion work/preparation paths; the local
virtual-fusion artifact tree also has no `.pt` copy. Its saved predictions are
not a model. Do not substitute originalJuly80k, ID100k or structure/XYZ-fill for
a B65 architecture ablation. B query-path ablation remains unavailable until
weights are recovered or a separately designed matched model is trained.

ID100k is present at
`/work/wangyy/id_xyz_consistency_20261024/outputs/consistency_100000_cpu4_v1/checkpoint_100000.pt`.
SHA256 `ada70ea7b97b4bdbcbd699b1698696faa0eef0298df09e07637e687eea3a29c8`;
completed model fingerprint `b546677a66e26d321bea1432c4307f7d612cdade2cd5c8d683af08c96835dd23`.
All model/data/runtime imports use the immutable trained `cpu4_code` at `e1ed5bd`.
Only the new diagnostic metrics and entrypoint come from the isolated new checkout.

## Measurements

- Original matched small panel:8 training assets with seen and heldout windows,
  plus16 validation assets;8 real inputs and scored slots0/2/4/6,128 views total.
- Strictly restore every saved tensor and verify the completed model fingerprint.
  Verify raw frame IDs, original continuous GT, parents and target tokens against
  saved ID100k records. After the measured native-repeatability audit below,
  historical-token agreement is recorded rather than required; a second decode
  of each identical captured condition must still reproduce its native tokens.
- Under the correct GT prefix, separate class/new-ID/parent-ID/BRANCH/EOS/X/Y/Z
  NLL and accuracy. Preserve full-vocabulary metrics. Separately report native
  grammar-constrained decisions and sites with more than one legal choice.
- Repeat teacher-forced scoring with another pose's condition from the same asset,
  keeping the target unchanged. This is a pose-dependence diagnostic, not a quality
  score or proof of bone-semantic correspondence.
- Report first full-token divergence and first structure-only divergence. These
  are first positional mismatches, not semantic alignments after insertions.
- Force the entire GT structural sequence, including joint count and EOS, while
  predicting only XYZ from its own generated coordinate prefix. Score resulting
  geometry against original GT, but label it as an oracle diagnostic, never model
  generation quality. Quantized-GT roundtrip is a reference, not an asserted bound.
- Role aggregates are token-weighted and retain denominators; macro view/asset
  quality has four equally weighted views per group asset. Non52 is a joint-count
  stratum and is not automatically synonymous with rare topology.

## Execution Gates

CPU audit covers all32 group-assets without CUDA. Primary accepts the actual
audit before releasing2 validation assets/8 views on GPU. Primary accepts the
smoke before releasing the128-view run. Any input/weight/source/fixed-condition replay mismatch
stops execution and is preserved as a failure receipt; no fallback initialization.

Reuse allocation130309880, GPU5/6, keeper1090013. Actual scheduler grant is1slot;
explicit SSH execution is bounded to4 total CPUs `[58,63,64,65]`,64GiB RAM and
85% GPU memory under the existing supervisor. Do not touch other processes,
release the allocation or modify the trained checkout. Stop before16:45JST.
17 CPU diagnostic/guard/oracle tests passed before real CPU audit.

## CPU Acceptance

The actual CPU audit completed exit0 at13:23:49JST: all896 saved state tensors
and the final model fingerprint matched; all32 group-assets/128 scored views
matched original frame IDs, continuous GT, parents and target tokens. CUDA was
not initialized; peak process-tree RAM3457019904bytes, cleanup complete.
Summary SHA256 `d08c8c254e5bad9924b37253abe44a68d56f174935b20dff10468274d138c304`.

Post-audit review fixed only GPU-result aggregation of anomalous sequence
endings, which must not be counted as exact matches. The CPU restoration/data
audit and model/forward/oracle paths are unchanged;17 CPU tests passed again.
Primary released only the8-view GPU smoke. Full128-view execution still needs
independent smoke acceptance; no weight updates are authorized.

The first GPU smoke stopped on replay mismatch at the first validation view on
both ranks (first token differences at positions49/53). Weights/source and GT
identity checks passed, but no diagnostic metric is accepted from this run.
Exit1 and payload cleanup were recorded. The retry invokes the frozen native
`evaluate_item` with its original no-grad/autocast scope and leaves parameter
requires-grad flags unchanged. A read-only output hook captures exactly the
native encoder condition for additional measurements; it cannot replace outputs
and is removed even after failure.19CPU tests pass, including this capture and
cleanup. The token-equality gate is unchanged; full execution remains blocked.

Native-path smoke2 also failed historical replay (first mismatches49/61;
12/6 positions differ in the first two equal-length outputs). Both failures
cleaned up their own payloads; no metrics or weight updates were accepted.
The frozen motion-evidence code uses CUDA index_add reductions, a possible
source of numerical variation, but that is not yet a demonstrated cause.

The next approved stage is only a two-asset repeatability audit: materialize
surface inputs twice and compare exact hashes; hold one materialization fixed
for three native evaluations; compute motion evidence three times on identical
points; decode the identical first captured condition three times. Record
condition/motion tensor deltas, token equality and per-repeat F1. No deterministic
mode, alternate kernel, model repair, new sampling policy or weight change is
introduced. Full diagnostics remain blocked pending interpretation.

## Measured Native Variation And Paired Protocol

The repetition audit completed exit0 with all saved weights/fingerprint unchanged.
Both assets' five sampled-surface tensors matched bitwise on repeated sampling.
On the same fixed input, motion states differed by at most1.79e-6 and encoder
conditions by at most0.005562. Native output tokens varied; the8-view mean F1
was0.857208/0.854363/0.866831 across three repeats. Decoding an identical captured
condition three times was exact on both assets and matched that native pass.
Motion evidence varies too, but this is not proof that it is the unique cause.

The original historical bitwise-token gate was therefore inappropriate for
this native pipeline. This is an explicit evidence-backed protocol correction,
not a silent tolerance or a change to model arithmetic. Version2 preserves
runtime, frozen model source, trained configuration, weights and reference data
exactly. It runs3 native encoder/evaluation repetitions on each fixed sampled
input. Within each repetition, free generation, teacher scoring, wrong-pose
scoring and the GT-structure oracle share the captured native condition.
Every view additionally repeats decoder generation on that identical condition
and requires exact token equality and unchanged condition bytes.

Report native F1 by repeat, historical F1 separately, historical-token agreement,
and paired oracle differences. Oracle results remain diagnostic only. Coverage
is8 unique smoke views/24 repeated rollouts, then128 unique full views/384
rollouts. The repeatability receipt is hash-pinned and primary acceptance is a
launch prerequisite.21 CPU tests pass. Only the revised smoke is released.

The revised smoke completed exit0 at13:58:16JST. Both ranks strictly restored
all896 tensors and retained the exact completed model fingerprint. Coverage
was8 unique views/24 repetitions, with every fixed-condition decoder replay
exact, no optimizer/backward, unchanged checkpoint stats, and full cleanup.
Peak process-tree RAM was8553304064bytes. Primary accepted execution and pairing,
not model quality, then released the128-unique-view/384-repetition stage with
unchanged computation and protocol. Oracle F1 need not exceed unconstrained F1;
this two-asset result does not establish a general mechanism.

## Full Paired Results

There are24 distinct assets:8 training assets measured in seen and heldout-pose
windows, plus16 validation assets. These form32 group-assets,128 unique scored
views and384 rollouts with3 native repetitions. Both ranks strictly restored896
saved tensors, retained the exact completed model fingerprint and unchanged
checkpoint stats, and matched all GT/frame records. All384 same-condition
decoder replays were exact. No optimizer or backward ran. Peak process-tree RAM
was8568463360bytes; exit0 and payload cleanup were verified. At14:28:02JST the
supervisor was absent, GPU5/6 each used0MiB, and the original allocation/keeper
remained active. No GPU release or further run was requested.

Full artifacts are under
`D:/evoweave/server_ops/outputs/query_utility_diagnostics_20261025/runs/diagnose_v2`.
Summary SHA256 `9ee3e2a588eb4131dd52d0b36474ca01e3d30832bc44c1f576dbc86edefb1168`;
resources SHA256 `3de6a696ca91ae22a46024b5532c892e3737fee37deed443ce7229cbb2d341ab`.
Final local analysis is
`D:/evoweave/server_ops/outputs/query_utility_diagnostics_20261025/analysis_diagnose_v2_final.json`,
SHA256 `26a40b76e5da788768b8aeaa1e523faa5d56901963cd1011b999d5e521215a36`.
Its15unit tests passed and the canonical CPU runner exited0. Primary independently
checked coverage, replay flags,181exact-structure invariants and all9subgroup F1
means against producer summaries (absolute tolerance1e-12).

### GT-Prefix Diagnostics

These are full-vocabulary, token-weighted top1 accuracies. The entire previous
token history is GT. XYZ accuracy means exactly the correct quantization bin,
not spatial F1 or a geometric tolerance. Parent-ID sites occur after BRANCH;
ordinary chain parents are implicit in this codec.

| Group | Assets | Parent ID | BRANCH | EOS | XYZ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training seen | 8 | 99.78% | 100.00% | 100.00% | 55.12% |
| Training heldout poses | 8, same identities | 99.55% | 100.00% | 100.00% | 51.44% |
| Validation | 16 | 71.91% | 90.74% | 65.63% | 45.77% |
| Validation non52 | 11, subset | 56.59% | 85.68% | 50.00% | 36.03% |

Grammar filtering is not a cure: validation non52 parent-ID accuracy among
nontrivial legal choices is57.64%, and EOS is53.03%. Validation gt52 structural
roles are100% under GT prefix, but this count-defined stratum is not automatically
an exact-topology-frequency class. New-ID accuracy is100% on the sampled training
assets,93.83% on validation and90.53% on validation non52; predictable serial IDs
must not dominate an aggregate structural-accuracy interpretation.

Token-weighted XYZ NLL for matched versus another pose's condition is1.165->3.604
on training-seen,1.245->3.719 on heldout poses,2.016->3.860 on validation and
2.629->4.181 on validation non52, in nats per token (lower is better).
The target prefix and target coordinates are unchanged. Thus the current
real-frame-only model uses pose-specific condition information under GT prefixes.
This does not measure virtual-query usefulness or cross-pose aggregation benefit.

### Free Generation And Structure Intervention

| Group | Native F1 | GT-structure/free-XYZ F1 | Paired change | Asset-bootstrap95% interval |
| --- | ---: | ---: | ---: | --- |
| Training seen | 0.79856 | 0.83557 | +0.03701 | [-0.00755, +0.08884] |
| Training heldout poses | 0.78615 | 0.77447 | -0.01168 | [-0.05929, +0.02820] |
| Validation | 0.63499 | 0.54187 | -0.09312 | [-0.15587, -0.03458] |
| Validation non52 | 0.50548 | 0.36373 | -0.14174 | [-0.21303, -0.07020] |
| Validation gt52 | 0.91994 | 0.93379 | +0.01385 | [0.00000, +0.03861] |

Each asset first averages its four views and three paired repetitions; bootstrap
resamples assets10000times with seed20261025 and percentile endpoints. Frames and
repetitions are not independent samples. This selected diagnostic panel and its
small subgroups support exploratory conclusions, not population-wide proof.
In non52 validation,10assets decline,1is unchanged and0improve under intervention.

All181 rollouts whose freely generated structure already equals GT have exactly
equal oracle/native tokens and F1, including69validation rollouts. Therefore this
intervention is a no-op on the observed exact-structure cases. When it changes
structure, it also changes subsequent coordinate prefixes. Correct parent IDs
and joint counts alone do not ensure compatible XYZ. The result rejects the
simple sufficiency claim that repairing structure alone must improve this frozen
model; it does not establish that GT structure is harmful, nor uniquely locate
the remaining cause in the encoder or decoder. This oracle is not an upper bound.

Every first full-token mismatch is XYZ (384/384), but coordinates are numerous
and a one-bin difference already counts: this is not causal proof that every
later structure error originated in coordinates. Exact free structure is53/96
seen,59/96heldout and69/192validation; non52 validation is18/132 (13.64%). There
are no generation failures, but1/384 outputs ends with EOS as the only legal
capacity choice. Parsing/termination is not a quality pass.

Validation native F1 across the3repeats is0.635008/0.632575/0.637402. Historical
single-pass F1 remains0.639080 and is not overwritten; only21.88% of repeated
validation token streams equal that historical stream exactly. Quantized-GT
roundtrip F1 is0.93315 for validation and0.90276 for non52, a separate reference
against original continuous GT, not a claimed ceiling.

## Interpretation And Next Discriminating Test

The sampled training structures are learned under GT prefixes, including on
heldout poses, while unseen-asset parent/EOS prediction is substantially weaker
even with correct history. Coordinate prediction is imperfect even on training
assets and free rollout further changes structure. Evidence points to both
unseen-asset generalization and coupled coordinate/structure rollout limitations,
not merely an inability to learn serial IDs. More training might help, but this
frozen diagnostic cannot establish that additional steps alone will fix them.

The virtual-query question remains unanswered because B65 weights are absent.
A proposed next causal comparison is a matched legacy-codec pair with/without
virtual queries, initialized from the same official pretrained weights with the
same current data, all-frame targets, optimization budget and selection rule.
Hold auxiliary and cross-frame-consistency losses at0 in that first pair to
isolate the virtual module; match/record any unavoidable capacity differences.
Then add the ID-codec pair to obtain a2x2 test if needed. These are proposals,
not authorized new training or a claim to reproduce historical B training.
