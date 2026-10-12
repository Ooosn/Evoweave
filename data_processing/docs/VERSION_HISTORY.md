# Data Processing Version History

This file records contract-level data-processing changes. It is documentation
only; old code paths are not kept as fallback implementations.

## 2026-10-13: Retained Root RT Cache Overlay

- Publish immutable `retained_root_rigid_v1` R/t cache shards and full manifest
  descriptors without rewriting source NPZs. Remove rigid root motion only;
  preserve scale/shear, target topology, joints and weights.
- Train/valid now contain 18,002 / 967 assets. Eighteen additional whole assets
  are isolated: 8 aligned-window bounds failures, 9 numerical consistency
  failures and 1 hidden-source-frame case. Seventy-two of 73 nonrigid-root
  assets pass the standard algorithm. Earlier exclusions remain excluded.
- Use one selected-input mesh-union normalization with padding 1.1. Exhaustive
  single/pair bounds checks cover arbitrary stored-frame subsets before extra
  rotation augmentation; the online consumer uses matching float32 arithmetic.
- RT metadata is retained through flat/virtual-frame loading and inversion;
  incompatible older wrappers reject the new manifest rather than ignore it.
- Pass1 rejects selected meshes hidden at sampled frames, preventing stale
  evaluated geometry. Publication passed 48 CPU tests and 80 real-loader checks.
- Source receipts and current hashes are in `DATASET_SOURCE_OF_TRUTH.md`.

## 2026-10-12: All-Frame Normalized Joint Bounds

- The final bbox-consistency stage now rejects an entire asset if any retained
  target joint coordinate exceeds `[-1, 1]` after that frame's mesh-only
  normalization with padding `1.1`.
- All stored frames and all retained joints are checked, including heldout
  poses and inactive-skin joints. The existing bbox-diagonal cutoff remains.
- Rejection records include the extremum, frame/joint/axis, violation counts,
  and normalization parameters. Invalid geometry fails closed.
- No source/NPZ rewriting, label clipping, joint removal, or target-derived
  normalization is introduced. Publish accepted and quarantined manifest
  partitions as a new version; preserve historical inputs and checkpoints.

## 2026-07-14: Deterministic Blender Process and FBX Media Handling

- Formal Blender subprocesses run in separate process groups. Timeout and
  interruption terminate the complete process tree, so a failed attempt cannot
  leave an exporter running while the next attempt starts.
- FBX import disables Blender's external image search. Textures are not part of
  Pass1 NPZ output, and missing media references must not trigger recursive
  scans of unrelated shared storage.
- The affected Objaverse asset was compared against its official oracle NPZ;
  all shared geometry, animation, skinning, skeleton, transform, action, and
  selected-frame arrays were exactly equal.

## 2026-07-06: Rootless Dynamic NPZ v3

- Canonical schema is `rootless_dynamic_npz_v3`.
- Validators and final audit require v3; v1/v2 rootless schemas are not accepted
  by the current module.
- Root/RT detection uses the parent graph, not row 0.
- Unskinned top root/RT chains are removed from the training target until the
  first skinned training joint.
- Deleted root/RT transforms are stored only in `recorded_root_*` auxiliary
  fields for debugging or visualization.
- Training fields remain in the normalized source coordinate space:
  `root_positions` is zero, `root_rotations_*` is identity, and
  `frame_vertices_rootspace`/`target_joints_rootspace` are not transformed by
  deleted root/RT rows.
- Multiple raw graph roots are allowed only when root pruning leaves exactly one
  active rootless training skeleton entry. If pruning produces a forest, the
  sample is rejected.
- Bone tail endpoint coordinates are not written to rootless training NPZs.
  Real retained tail/end joint rows may remain when skinned or structurally
  required.

## 2026-07-07: Final BBox Consistency Screening

- The final skeleton/mesh extent screen is
  `build_rootless_bbox_consistency_distributions.py`.
- The score is symmetric:
  `min(skeleton_bbox_diag, mesh_bbox_diag) / max(skeleton_bbox_diag,
  mesh_bbox_diag)`.
- The default final cutoff is
  `target_joint_mesh_bbox_diag_consistency >= 0.4`.
- This replaces the old one-direction `joint_bbox / mesh_bbox` interpretation
  for final screening, so both under-covering skeletons and over-expanded
  skeletons are handled by the same metric.
- Final outputs live under
  `quality_distributions/rootless_bbox_consistency/`.
- Final screened manifests are train/valid only; original test rows are merged
  into train.

## 2026-07-07: Rootless Target Contract Before First Formal Reject

- Raw Pass1 screening is now `build_pass1_precheck_dataset.py` and is limited to
  readability, schema/shape, skin, parent graph, face, and coarse LBS sanity.
- Root/rootless policy is no longer decided on raw Pass1 fields.
- `build_rootless_dynamic_npz.py` runs immediately after Pass1 precheck
  combining, before formal hard quality screening.
- `build_rootless_target_strict_dataset.py` is the first formal hard-reject
  stage and operates on actual rootless v3 training target fields.
- The old raw hard-reject scripts were removed from the canonical module to
  avoid fallback or accidental use.
