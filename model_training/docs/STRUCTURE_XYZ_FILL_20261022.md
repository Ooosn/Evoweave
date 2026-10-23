# Structure-Conditioned XYZ Token Filling

## Question And Outcome

Does generating one explicit skeleton structure and filling its coordinates
from each pose's mesh improve multi-pose skeleton reconstruction?

The selected random-position model completed 160,000 asset exposures, 3,355
updates and 2,068,624 supervised-frame exposures. Execution completed; final
model quality remains unaccepted. At 80k, the small matched validation panel
showed faster early learning but no demonstrated higher quality ceiling than
the earlier A recipe. Reliable non52 structure generation remained a limitation.

## Method

- Shared surface encoding and the 1,024-anchor virtual/query motion network
  supply both heads. Both losses train shared features; final anchors retain
  their gradient path.
- A two-layer explicit-ID AR decoder generates one
  `BOS J0 ... BRANCH parent Jnew ... EOS` structure under active-ancestor grammar.
- XYZ consumes that discrete structure, with three unknown coordinate slots
  after every new joint. Parent references add no slots. The structure
  decoder's hidden states and logits are not XYZ inputs.
- Each coordinate slot has an axis-mask, joint-ID and sequence-position
  embedding. Four pre-LN blocks use width 1,024, eight heads, FFN 4,096,
  GELU and zero dropout. Each block applies bidirectional self-attention,
  cross-attention to that frame's final mesh anchors plus 3D position encoding,
  then an FFN.
- One shared 256-class head predicts all XYZ slots in parallel. Topology stays
  fixed during filling; module parameters are shared across frames. GT
  coordinates serve as labels. There is no iterative hard-token refill or
  cross-branch alignment loss.
- Expanded-sequence capacity is 3,076; mesh memory does not consume it.
  The 256-joint grammar has worst-case expanded length 1,534. Actual accepted
  train/valid maxima were 1,210/1,012.

The original 256-bin convention uses rounded scaled coordinates and legal-bin
saturation after checking finite continuous coordinates, with bin-center
offset +0.5 on decode. Continuous GT remains the scoring target. Normalization
uses retained-root RT and the selected-input mesh-union bbox with padding 1.1.

## Initialization And Training

Only compatible official UniRig `mesh_encoder` and `output_proj` weights are
pretrained. Motion and structure modules use fresh seed 20261013; XYZ uses
20261022. Old A/B weights and optimizers are not dependencies.

Two matched initializers were tested: R uses random XYZ sequence positions;
P copies official OPT rows 1026:3078 into its first 2,052 rows, leaving the
remaining rows fresh. All other shared and fresh tensors are identical.
The comparison progressed through 512 and 2,048 exposures per arm.

| At 2,048 exposures | Validation free F1 | GT-structure XYZ CE | GT-structure XYZ L1 |
| --- | ---: | ---: | ---: |
| R: random positions | 0.039343 | 4.892949 | 0.219939 |
| P: copied positions | 0.047666 | 4.903097 | 0.230727 |

R was selected because copied positions had no consistent benefit: P had
higher small-panel free F1, while R had lower GT-conditioned L1 across all
three groups and better heldout-pose F1. This is a simplicity decision, not
statistical superiority. Both still mainly emitted the 52-joint template.

The selected recipe uses effective batch 48, surface LR 3e-5, other LRs 1e-4,
fresh AdamW betas (0.9,0.999), weight decay 0, clipping 1, and a 192-exposure
warmup from 20% LR followed by constant LR. Through 80k, loss is the sum of
per-asset mean coordinate CE and per-asset mean structure CE. The 80k-to-160k
continuation changes only the structure coefficient from 1 to 4 and expands
evaluation; model, LR, sampling, normalization and pruning stay unchanged.

### Terminal Pruning

Each update has an 80% unchanged / 20% pruning gate, shared across ranks and
microbatches. For each asset visit on a pruned update, draw a fraction in
0..20%, then delete floor(fraction*(N-1)) joints by repeated uniform sampling
from the current nonroot-leaf pool. Compact IDs once and retain original GT
source-index mappings. All poses in the visit use the same deletion set.
Full structure CE stays unchanged; only XYZ's structure input, slots and
associated labels are pruned.

## Data And Evaluation

The immutable training subset has 5,000 assets, SHA-256
`a17b6be96fb33a32a1bee7117b07789da6261c6b5693104f3d2cce07c09bab88`.
The validation pool has 967 assets, SHA-256
`fd2be7b5885833946e68dc186300a820757ae652c487749eec1b049d381e230a`.
Complete root-RT descriptors and eight heldout frames per training asset are
preserved.

Small-panel evaluation uses eight train-seen contexts, eight train-heldout
contexts and 16 validation assets. Each has eight input frames; positions
0,2,4,6 are scored, yielding 128 views total and 64 validation views. Historical
B65 reference F1 is 0.873763 / 0.845508 / 0.681973 on these three groups.

The larger panel uses 32 train-seen, 32 train-heldout and 128 validation
contexts, with eight scored frames each: 1,536 views, including 1,024 validation
views. It is still a subset of the 967-asset validation pool.

All comparisons preserve raw frame identity and continuous GT. The original
spatial edge-F1 scoring is shared with the earlier model; failures score zero.
GT-structure localization and free structure-plus-XYZ generation are reported
separately. Exposure counts mean asset visits, not optimizer updates.

## Results And Interpretation

| Stage | Validation result | Interpretation |
| --- | --- | --- |
| Two-asset, 32-update fit | Both 21/54-joint structures correct; XYZ CE down 77.8%..97.8%, L1 down 96.0%..99.2% | Bounded trainability established; the 54-joint geometry remained imperfect |
| Selected R, 4,096 / 6,144 / 8,192 exposures | Small-panel F1 0.10714 / 0.15203 / 0.17719 | Learning progressed; structure remained template-dominated |
| 20k | Small-panel F1 0.333817; old A 0.219859 on the same panel; non52 count correct 0/11 | Faster early progress for the whole recipe, not isolated architecture causality |
| 55k / 80k | Small-panel peak 0.580167 / final 0.559265 | The final checkpoint was not the best checkpoint |
| Old A, same panel through 80k | Peak 0.579980 at 65k; final 0.538555 | No established higher ceiling; at 80k, asset differences give 7 new wins, 8 old wins, 1 within 0.01 |
| 80k structure strata | Non52 F1 0.411897, count correct 2/11; five 52-joint assets F1 0.883476 | Natural termination on every context did not imply reliable structure prediction |
| Re-evaluated 80k | Larger-panel validation F1 0.640624; projected small-panel 0.558888 | Different cohorts, not a gain from further training |
| 160k | Completed 3,355 updates and 2,068,624 supervised frames | Final large-panel trajectory still requires scientific review; completion is not quality acceptance |

The 21-joint fit's quantized-GT reference F1 was 0.70 versus continuous GT
self-F1 1.0 under the unchanged spatial matcher. This illustrates metric
sensitivity to quantization, not a mathematical upper bound.

The 8,192-exposure condition audit found weak, rather than absent, sensitivity
to the correct asset condition and no consistent validation benefit. Some
saved training anchors did not replay bitwise; GT/frame identities were
unchanged, and no specific sampler cause was established.

### Structure-Loss Balance

On 32 fixed T8 training cases (10 with 52 joints, 22 non52), the median
structure/XYZ shared-gradient norm ratio was 0.1314 overall and 0.1683 on
non52; the non52 motion-encoder ratio was 0.2350. Weight 4 changes those raw
ratios to 0.5258, 0.6734 and 0.9400. This motivated the controlled continuation;
it does not establish optimality or mean a fourfold learning rate.
These are raw gradients, not Adam-preconditioned updates. Compare the two raw
CE terms across the change, not the differently weighted total loss.

## Reproducibility And Evidence

- [Model/trainability and quantization checks](../analysis/structure_xyz_fill_preflight_v2_review.json).
- [Matched 512](../analysis/structure_xyz_fill_matched512_v3_review.json) and
  [2,048 initializer comparison](../analysis/structure_xyz_fill_matched2048_v3_review.json).
- [8,192 condition audit](STRUCTURE_XYZ_CONDITION_8192_20261022.md).
- [80k scientific review](../analysis/structure_xyz_fill_formal80000_completion_review_v1.json).
- [Loss-balance measurement](../analysis/structure_xyz_fill_loss_balance80000_review_v1.json).
- [160k recipe](../experiments/structure_xyz_fill_random_160000_v1.json).
- Final result:
  `/work/wangyy/structure_xyz_fill_20261022/outputs/formal_random_160000_v1/result.json`,
  SHA-256 `d200706c995a1f3643c627e8902e7a74874d8cd15b35878ce46758066fee57e7`.
  The terminal record is `state/current.json: last_operation`.
