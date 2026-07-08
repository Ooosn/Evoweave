# Data Processing Version History

This file records contract-level data-processing changes. It is documentation
only; old code paths are not kept as fallback implementations.

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
