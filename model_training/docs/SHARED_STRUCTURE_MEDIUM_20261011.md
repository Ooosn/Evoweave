# Medium Shared-Skeleton Expansion

## Scope And Hypothesis

The user authorized a medium-scale expansion, not formal full-dataset training.
The product remains the generated reusable skeleton sequence `(parents, H)`.
Mesh localization provides supervision and a use interface; it is not the
primary research objective. Separating structure from posed XYZ removes XYZ
tokens from the autoregressive structure prefix, but does not itself establish
motion-independent joint identity or novelty relative to prior work.

Keep the implemented 128-token structure memory, inherited AR body, new parent
embedding/head and two-layer target-mesh position decoder unchanged. Start
again from the immutable `bias_h8` sample80000 initialization, not from the
eight-asset trained skeleton checkpoint. New modules use seed 20261011, as in
the small probe. Geometry/motion feature parameters remain frozen for this
bounded experiment; this is not a resource estimate for full-model training.

## Data And Budget

Consume the current final manifests declared in `DATASET_SOURCE_OF_TRUTH.md`:
18,097 training and 971 validation rows. Use the existing source/joint-count-band
round-robin selector, seed 20261012, to choose 256 training and 64 validation
assets. Selected identities are disjoint. No NPZ-directory scan, dataset rewrite,
joint filtering, inactive-joint removal or new root is permitted.

The pinned CPU audit reports:

| Split | Assets | J Range | Ordered Topology Families | Base / Addition |
| --- | ---: | ---: | ---: | ---: |
| Train | 256 | 4-206 | 128 | 230 / 26 |
| Valid | 64 | 4-173 | 35 | 57 / 7 |

Twenty-two validation assets have an ordered parent stream absent from the
selected training assets. This is the dataset's ordered identity criterion,
not an unordered-tree-isomorphism count.

Training is capped at 4,096 optimizer updates, one asset per microbatch and
four microbatches per update. Shuffled complete asset passes give exactly
16,384 asset exposures, or 64 per training asset. Sample T uniformly from
2 through 24 per exposure and alternate randomly between two query charts.
Use AdamW/OneCycle with peak LR 1e-4 for new modules and 1e-5 for the inherited
AR body, weight decay 0.04, gradient clipping 1, and dropout zero. Save
checkpoints including optimizer, scheduler, sampler and RNG at updates
1,024, 2,048 and 4,096. No automatic continuation beyond this budget.

## Frame Separation

All selected NPZs contain 40 original frames. An immutable asset-specific seed
partitions raw frame IDs into 32 training-eligible and eight held-out frames
before any query or surface selection. This avoids inheriting the old loader's
FPS-first/random-tail selection bias.

Build two source banks from the 32 eligible frames. Each uses a different first
query mesh and its own mesh-only common coordinate chart and tracked surface
references. Neither can access a held-out frame. Select actual T from a bank
before recomputing the temporal encoder and original u/O/M evidence. Supervise
XYZ on all 32 eligible target meshes; every target is independently normalized
by its own mesh-only bbox. Both source and target inputs to optimization are
physically sliced, not merely masked after a forward pass.

The third source bank contains only the eight held-out frames, with its own
query mesh. It is evaluation-only. The primary reuse test generates S from
eligible source frames and binds that fixed S to held-out target meshes.
The third bank instead asks whether disjoint held-out evidence can recover the
same correct structure and usable joint identities. Do not conflate these two
conditions, or describe agreement between two wrong trees as success.

Important limits:

- The frame holdout applies to this new probe. The old initialization was
  trained on the base manifest and may have seen some base-asset poses; new
  base/addition strata are reported separately. No full-history pose holdout
  is claimed.
- Each asset contributes one recorded action. Disjoint frame IDs, even with a
  changed query pose, do not establish invariance across independent actions.
- Frame exclusion does not guarantee large pose differences or remove near
  duplicates; evaluate the measured geometry and do not equate frame novelty
  with motion novelty.

## Coordinates And Storage

Keep the strict 256-bin `[-1,1]` position vocabulary and mesh-only bbox padding
1.1. The 320-asset/40-frame CPU audit found unpadded local maxabs 1.07687545 on
train and 1.01399255 on valid. Thus no selected coordinate requires clipping or
changing the per-instance normalization to depend on GT. Retain the exact
inverse to raw NPZ coordinates. This does not certify the full dataset.

The source retains a common query chart for motion, while every target uses its
own chart. Canonical rootless-v3 removes auxiliary RT nodes, not global pose
transforms from the stored coordinates. No cross-frame XYZ-equality loss is used.

Cache independent per-frame surface features, never full-clip contextualized
features. Write one large shard per asset and load only the currently used asset
with memory mapping. The compute node's `/work` was verified as `/dev/md0`,
RAID0 over `nvme0n1p1` and `nvme1n1p1`, both nonrotational Samsung NVMe disks.
Scratch cache root is `/work/wangyy/shared_structure_medium_20261011_r1` on
pcg01. These caches are node-local and reconstructible from pinned inputs.
Checkpoints, evaluation artifacts and the cache index remain under the persistent
`/home/wangyy/evorig/outputs/shared_structure_medium_20261011_r1` root.

Use only physical GPU5 in existing qlogin129547842 on pcg01i, leaving GPU7 and
the allocation keepalive untouched. The recorded allocation reserves 64GiB host
memory. Reject GPU peak >=90% of available device memory or CPU peak >=90% of
64GiB. Recheck live allocation, exact device, SSD mount and capacity at launch.

## Evaluation And Acceptance

Compare initialized/final T8 on the same first32 training diagnostic assets and
all64 validation assets. The final first16 assets of each split also receive
T2, T24, alternate-query T8 and disjoint-heldout-source T8 evaluation. This gives
96 initialized and 224 final generations, not an unbounded full-training-set
generation sweep. The uniform generation cap is256, covering every selected
GT count; natural EOS and cap failure remain separate. No forced EOS or GT-count
stopping is allowed.

Persist each generated S and rebind it without source encoding or AR generation.
Measure ordered tree/count correctness, teacher-prefix losses, held-out geometry
and one-to-one located-edge coverage. Also measure the fraction of fixed GT
joint identities within0.05 in all eight held-out meshes. Wrong counts and
generation failures score zero for this all-case identity coverage; identity
MPJPE is reported separately only for count-matched outputs, with its denominator.
One-to-one per-frame coverage must not replace identity-preserving checks.

Distances are in padded target-mesh maximum-half-extent units. To express error
as a fraction of the mesh's longest bbox side, multiply by1.1/2. The0.05 matching
tolerance therefore corresponds to2.75% of that side. Asset-level summaries
weight assets equally. Teacher-forcing supplies true parents, not true H;
teacher-prefix localization is not an isolated test of the position head.

Zero-memory diagnostics cover the first16 assets of each split. Cross-source
agreement includes GT-tree correctness and same-target prediction differences.
Base/addition summaries are descriptive; their small subsets are not a reliable
standalone generalization benchmark.

Before cache/training, require CPU isolation tests and a real GPU preflight on
min/max-J train/valid assets, T2/T8/T24, both training query charts and held-out
source geometry. Test prefix parity, serialization/rebinding, positive XYZ-only
gradients into structure, all-component updates and a T24/J512/32target stress
case. Review each completed stage before preparing the next. Preserve failed
attempts rather than silently reusing their artifacts. Execution completion is
not method-quality acceptance or authorization for full retraining.
