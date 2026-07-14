# Evoweave Data Processing Contract

This folder is the standalone Evoweave raw-data processing module. Future agents
working on data rebuilds should treat this folder as the module boundary and
should not mix model training, tokenizer experiments, temporary probes, or old
NPZ restoration logic into this tree.

The formal input is preserved original data from the Baidu/transfer source tree.
Old official NPZ archives are allowed only as an external oracle for comparison;
they are not an input to the rebuild pipeline.

## Source Names

The rebuild manifest currently recognizes three `dataset_source` buckets:

- `texverse`
- `objaverse_xl`
- `objaverse_xl_more_anim`

`objaverse_xl_more_anim` is a historical Objaverse-XL animation sub-bucket, not a
separate conceptual dataset. Whether it is present in a run must be decided by
the transfer manifest's `dataset_source` column. If the transfer manifest has no
rows for this source, its generated source manifest should simply be empty.

## Pipeline Contract

1. `build_rebuild_source_manifests.py`

   Reads the transfer manifest and writes per-source rebuild manifests. This
   stage performs no data filtering. It only maps `asset_id`, `split`,
   `dataset_source`, and preserved original paths. Missing originals are fatal
   when `--fail-on-missing-original` is used.

2. `texverse_quality_audit.py`

   TexVerse Pass0 source audit. It expands each original zip and audits
   importable source candidates with Blender. It uses only the original zip
   contents and does not consult old NPZ archives.

3. `export_texverse_pass1_batch.py` / `export_objaverse_xl_pass1_batch.py`

   Blender exports raw dynamic Pass1 NPZ from original assets. Export failures
   are the first reject layer.

   A Blender process timeout is an execution failure, not a data-quality
   decision. The Objaverse batch runner repeats the exact same export command
   according to `EVOWEAVE_PASS1_TIMEOUT_RETRIES` before recording a final
   reject. A retry must not change the source asset, Blender options, target
   contract, or use an old NPZ; the manifest records `batch_attempts` and any
   transient timeout reason.

   Pass1 must use the row-skeleton contract:

   - joints are explicit Blender bone rows / heads;
   - bone tail coordinates are never added as new joints;
   - real rows named `tail` / `end` are kept when skinned or structurally
     required;
   - unweighted terminal tail/end rows may be dropped by the contract.
   - `rest_tails/rest_tails_raw` are Blender/FBX endpoint audit fields only at
     this layer. They are not skeleton nodes and are not training targets.

   Frame sampling uses motion-FPS:

   - dense candidate frames;
   - per-frame mesh center normalization;
   - up to `EVOWEAVE_MOTION_FPS_DESCRIPTOR_VERTICES` descriptor vertices;
   - FPS selects `EVOWEAVE_SEQUENCE_FRAMES` frames;
   - selected frame numbers are sorted before export.

   FPS is intended to maximize pose coverage inside the selected source/action.
   It is not required to reproduce old NPZ frame numbers.

4. `dedupe_dynamic_sequences.py`

   Removes duplicate dynamic sequences by motion fingerprints.

5. `build_pass1_precheck_dataset.py`

   Raw Pass1 readability precheck before target rewriting. This stage owns only
   file/schema/shape/finite value checks, skin row validity, parent graph
   readability, face validity, split-map assignment, duplicate asset-sequence
   handling, and coarse LBS reconstruction sanity.

   It does not make root/rootless policy decisions and it does not run
   training-target quality gates. Samples can have raw dummy roots, raw
   unskinned RT chains, or raw multi-root forms at this stage; those are handled
   by rootless target rewriting.

6. `combine_dynamic_precheck_manifests.py`

   Combines accepted Pass1 precheck outputs across source buckets and performs
   cross-source manifest checks.

7. `build_rootless_dynamic_npz.py`

   Target-contract rewrite stage before the first formal hard quality reject.
   This is where NPZs are rewritten for training. Current outputs use
   `rootless_dynamic_npz_v3`:

   - locate the raw root from the parent array, not blindly from row 0;
   - allow raw files to contain multiple `parent < 0` graph roots only when
     root pruning leaves exactly one active training skeleton entry;
   - remove unskinned top root/RT chains from the training target until the
     first skinned joint;
   - do not add a synthetic root;
   - reject samples whose rootless target becomes a forest;
   - reject unskinned top roots whose active children would become multiple
     rootless training entries;
   - keep only mesh connected components controlled by retained rootless joints;
   - store deleted root/RT auxiliary fields for visualization/debugging only.
   - do not write per-bone endpoint coordinates such as `target_tails_rootspace`,
     `posed_tails_rootspace`, `rest_tails`, or `rest_tails_raw` into rootless
     training NPZs.
   - record root fields such as `raw_root_indices`, `recorded_root_indices`,
     `recorded_root_positions`, and `recorded_root_rotations_*`.

   In v3, deleted root/RT rows are record-only.  Training fields named
   `frame_vertices_rootspace` and `target_joints_rootspace` stay in the
   normalized source coordinate space; `root_positions` is zero and
   `root_rotations_*` is identity for reconstruction compatibility.  The real
   deleted root transforms live only in `recorded_root_*` fields.

   The rootless training target is the joint tree:

   - `target_joints_rootspace`
   - `target_parents`
   - `target_skin_weights`
   - target metadata such as `target_has_skin`, `target_is_connector`, and
     `target_is_tail_or_end`

   `target_is_tail_or_end` is metadata for real retained joint rows whose names
   look like tail/end. It is not a per-bone endpoint coordinate.

8. `build_rootless_target_strict_dataset.py`

   First formal hard-reject layer on the actual rootless training target fields.
   It validates rootless contract semantics and applies the established dynamic
   quality hard gates to rootless target data, then materializes an accepted-only
   dataset root for validation, histograms, and training.

9. `validate_rootless_dynamic_npz.py` / `audit_rootless_final_dataset.py`

   Validate the rootless contract, rootspace reconstruction, skin conservation,
   LBS reconstruction, face validity, and audit-only metrics. These scripts can
   report hard/soft issues, but they are not the place to introduce new
   exploratory quality gates.

10. `build_rootless_quality_score_distributions.py`

   Runs the final dynamic quality histogram screening from rootless NPZs:
   motion coverage, motion amount, motion validity, bbox stability,
   edge stretch/collapse stability, spike cleanliness, geometry stability, and
   overall quality.

11. `build_rootless_bbox_consistency_distributions.py`

   Runs the final skeleton/mesh bbox consistency histogram screening from
   rootless NPZ target fields directly and writes the final accepted/rejected
   manifests:

   - `target_joint_mesh_bbox_diag_consistency`
   - `active_joint_mesh_bbox_diag_consistency`

   This stage is the final place for histogram-based data screening. Its checks
   should not be duplicated as earlier post-hoc gates. The score formula is
   `min(skeleton_bbox_diag, mesh_bbox_diag) / max(skeleton_bbox_diag,
   mesh_bbox_diag)`, so both under-covering skeletons and over-expanded
   skeletons score near zero. The default final cutoff is
   `target_joint_mesh_bbox_diag_consistency >= 0.4`.

   Final screened manifests are train/valid only:

   - original `train` rows stay in train;
   - original `test` rows are merged into train;
   - original `val` rows are written as `valid`.

## Tail Endpoint Policy

There are two different concepts that must not be mixed:

- A real joint row whose name contains `tail` or `end`. This is part of the
  parent tree if it survived the row-skeleton contract. It may be skinned or
  structurally required and can participate in training like any other joint.
- A Blender/FBX per-bone tail endpoint coordinate. This is a display/endpoint
  field attached to a bone row. It is often directionally unreliable and can
  produce distracting rays in visualizations.

The formal pipeline keeps the first concept and removes the second concept from
training-target semantics:

- Pass1 may retain endpoint coordinates only for audit/debugging.
- Rootless v3 training NPZs do not write endpoint-coordinate fields.
- Final screening metrics and validation use joint rows and parent-child tree
  edges only.
- Visual review plots must not draw endpoint rays by default.

## Reject Layers

The intended reject layers are:

- Pass1 Blender/source export failures.
- Dedup duplicate dynamic sequence removal.
- Pass1 precheck readability rejects.
- Rootless contract/build rejects.
- Rootless target strict hard checks and established hard quality gates.
- Rootless validation/audit reports for contract correctness.
- Final histogram screening after rootless rewrite.

## Outputs

The runnable entrypoint is:

```bash
bash run_rebuild_from_originals.sh
```

Important output paths under `EVOWEAVE_REBUILD_ROOT`:

- `00_source_manifests/`
- `texverse/pass1_motionfps_sequence*/`
- `objaverse_xl/pass1_motionfps_sequence*/`
- `objaverse_xl_more_anim/pass1_motionfps_sequence*/`
- `*/sequence_dedup/`
- `*/pass1_precheck/`
- `combined_pass1_precheck/`
- `rootless_clean/`
- `rootless_target_strict/`
- `rootless_quality/audit/`
- `quality_distributions/rootless_dynamic_scores/`
- `quality_distributions/rootless_bbox_consistency/`
- `quality_distributions/rootless_bbox_consistency/final_manifests/`
  - `train_manifest.jsonl`
  - `valid_manifest.jsonl`

## Not Included

This canonical module must not contain:

- old NPZ archive conversion as a formal data source;
- temporary probe scripts;
- post-hoc sample-removal quarantine scripts;
- model training or tokenizer ablation code;
- alternate Blender fallback execution paths.
