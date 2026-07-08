# Evoweave Data Processing

This is the canonical Evoweave data-processing module. It rebuilds training NPZ
data from preserved original assets and owns the data contracts up to the
rootless training dataset and quality distributions.

Old official NPZ archives are not pipeline inputs. They may be used only as an
external oracle for debugging rebuild mismatches.

## Current Dataset Pointer

The current production/training dataset pointer is documented in
`docs/CURRENT_DATASET.md` and the workspace-level `DATASET_SOURCE_OF_TRUTH.md`.
Do not infer current data from report folder names such as `latest_*`.

## Module Boundary

This module owns:

- source manifest reconstruction from the transfer manifest;
- TexVerse source audit from original zip files;
- Objaverse-XL and TexVerse Blender Pass1 export;
- explicit row-skeleton head/tail handling;
- motion-FPS frame selection;
- dynamic sequence dedupe;
- Pass1 readability precheck before training-target rewriting;
- rootless training-target rewrite before the first formal hard quality reject;
- rootless target strict hard screening;
- rootless validation/audit;
- final histogram screening and final screened manifests.

This module does not own:

- model training;
- tokenizer ablations beyond strict roundtrip validation;
- visualization UI/tools;
- post-hoc quarantine pipelines;
- temporary probes;
- old NPZ-to-NPZ restoration as a formal data source.

## Directory Layout

```text
data_processing/
  README.md
  RUNBOOK_80K_REBUILD.md
  PACKAGE_MANIFEST.txt
  environment.yml
  requirements.txt
  run_rebuild_from_originals.sh
  configs/
    rebuild_paths.env.example
  docs/
    CURRENT_DATASET.md
    DATA_PIPELINE_CONTRACT.md
    VERSION_HISTORY.md
  scripts/
    build_rebuild_source_manifests.py
    texverse_quality_audit.py
    export_texverse_pass1_batch.py
    export_objaverse_xl_pass1_batch.py
    dedupe_dynamic_sequences.py
    build_pass1_precheck_dataset.py
    combine_dynamic_precheck_manifests.py
    build_rootless_dynamic_npz.py
    build_rootless_target_strict_dataset.py
    validate_rootless_dynamic_npz.py
    audit_rootless_final_dataset.py
    build_rootless_quality_score_distributions.py
    build_rootless_bbox_consistency_distributions.py
  src/
    rigweave/
```

## Main Entry Point

```bash
bash run_rebuild_from_originals.sh
```

Important environment variables:

- `EVOWEAVE_TRANSFER_ROOT`: preserved transfer/original-data root.
- `EVOWEAVE_TRANSFER_MANIFEST`: transfer manifest CSV.
- `EVOWEAVE_REBUILD_ROOT`: output root for rebuilt data.
- `EVOWEAVE_SEQUENCE_FRAMES`: sequence frame count, default `40`.
- `EVOWEAVE_MOTION_FPS_DESCRIPTOR_VERTICES`: FPS descriptor vertices, default
  `1024`.
- `EVOWEAVE_UNIRIG_ROOT`: UniRig dependency root for tokenizer validation.
- `EVOWEAVE_ROOTLESS_MIN_TARGET_JOINT_MESH_BBOX_CONSISTENCY`: final
  skeleton/mesh bbox consistency cutoff, default `0.4`.

## Pipeline Summary

1. Build per-source manifests from preserved originals.
2. Audit TexVerse source candidates inside original zips.
3. Export Pass1 dynamic NPZ from original assets with Blender.
4. Dedupe dynamic sequences.
5. Run Pass1 readability precheck only: schema, shape, finite values, skin, and
   LBS sanity. Raw Pass1 root policy is not decided here.
6. Combine accepted Pass1 precheck manifests across sources.
7. Rewrite accepted NPZs into rootless v3 training targets.
8. Run the first formal hard quality reject on rootless target fields.
9. Validate/audit rootless target strict data.
10. Run final dynamic quality histogram screening.
11. Run final skeleton/mesh bbox-consistency histogram screening and write final
    accepted/rejected manifests.

The detailed contract is in `docs/DATA_PIPELINE_CONTRACT.md`.

## Current Important Decisions

- Pass1 skeleton rows are explicit Blender bone heads. Bone tail coordinates are
  not added as joints.
- Real rows named `tail` or `end` are kept only when they are skinned or
  structurally required.
- Root detection uses the parent array, not row 0.
- Rootless rewrite is part of the target contract and happens before formal
  hard quality screening. It removes unskinned top root/RT chains until the
  first skinned training joint and does not add a synthetic root.
- Rootless forest after root removal is rejected.
- Deleted root/RT transforms are preserved only as `recorded_root_*` auxiliary
  metadata for debugging/visualization; they are not used to normalize training
  coordinates.
- Per-bone tail endpoint coordinates are not a training target. Pass1 may keep
  `rest_tails/rest_tails_raw` only as Blender-read audit fields, but rootless
  training NPZ v3 does not write `target_tails_rootspace`,
  `posed_tails_rootspace`, `rest_tails`, or `rest_tails_raw`.
- Tail/end joint rows are different from tail endpoint coordinates. Real
  tail/end rows are still kept when skinned or structurally required.
- Review visualizations should draw the orange joint rows and blue parent-child
  tree only; they should not draw Blender/FBX tail endpoint rays.
- Histogram screening is the final data-screening layer, not a temporary
  analysis layer attached to earlier gates.
- The final skeleton/mesh bbox consistency cutoff is
  `target_joint_mesh_bbox_diag_consistency >= 0.4` by default, where
  consistency is `min(joint_bbox_diag, mesh_bbox_diag) /
  max(joint_bbox_diag, mesh_bbox_diag)`.
- Final screened manifests are train/valid only. Original test rows are merged
  into train at the final bbox-consistency screening stage.

## Validation Expectations

A completed rebuild should include:

- per-source Pass1 manifests;
- dedupe manifests;
- Pass1 precheck accepted/rejected manifests and metrics;
- combined Pass1 precheck manifests;
- rootless derived cache `train/val/test` manifests and NPZs;
- rootless target strict accepted/rejected manifests, metrics, and accepted-only
  materialized NPZ root;
- `rootless_validation.json`;
- rootless final audit CSV;
- final dynamic quality histogram outputs;
- final skeleton/mesh bbox consistency histogram outputs.
- final bbox-consistency screened manifests under
  `quality_distributions/rootless_bbox_consistency/final_manifests/`.
- final training manifest:
  `quality_distributions/rootless_bbox_consistency/final_manifests/train_manifest.jsonl`.
- final validation manifest:
  `quality_distributions/rootless_bbox_consistency/final_manifests/valid_manifest.jsonl`.

Rootless validation should have zero invalid samples before the dataset is used
for training.
