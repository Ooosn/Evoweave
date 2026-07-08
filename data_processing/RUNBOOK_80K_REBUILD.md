# 80k Rebuild Runbook

This package is the standalone Evoweave raw-data processing module. It rebuilds
NPZs from preserved original assets and stops at final histogram-screened
train/valid manifests.

Old official NPZ archives are not pipeline inputs. Use them only as an external
oracle when debugging mismatches.

## Required External Inputs

- Preserved original-data transfer tree with `full_transfer_manifest.csv`.
- Blender executable with import support for the source asset formats.
- UniRig checkout with
  `configs/tokenizer/tokenizer_parts_articulationxl_256.yaml`.
- Enough local scratch/output storage for Pass1 exports, rootless NPZs, logs,
  CSV metrics, and histogram PNGs.

## Environment

```bash
cd data_processing
conda env create -f environment.yml
conda activate evoweave-data
```

Install Blender and 7z outside this conda environment if they are not already on
the server:

```bash
sudo apt-get update
sudo apt-get install -y blender p7zip-full
```

If UniRig has additional tokenizer dependencies in its own environment file,
install those into the same environment before running the rebuild.

## Configure Paths

```bash
cp configs/rebuild_paths.env.example configs/rebuild_paths.env
vim configs/rebuild_paths.env
source configs/rebuild_paths.env
```

Check these variables before launching a full run:

```bash
echo "${EVOWEAVE_TRANSFER_ROOT}"
echo "${EVOWEAVE_TRANSFER_MANIFEST}"
echo "${EVOWEAVE_REBUILD_ROOT}"
echo "${EVOWEAVE_UNIRIG_ROOT}"
echo "${BLENDER}"
echo "${PYTHON}"
```

## Launch

```bash
bash run_rebuild_from_originals.sh 2>&1 | tee "${EVOWEAVE_REBUILD_ROOT}/run.log"
```

For a small path/environment check, set `EVOWEAVE_REBUILD_LIMIT` in the sourced
env file. For the full 80k rebuild, keep it at `0`.

## Pipeline Stages

1. Build source manifests from the transfer manifest.
2. Audit TexVerse zips and select source files.
3. Export raw dynamic Pass1 NPZs from original assets with Blender.
4. Dedupe dynamic sequences.
5. Run Pass1 readability precheck only.
6. Combine accepted Pass1 precheck manifests across sources.
7. Rewrite accepted samples into rootless v3 training NPZs.
8. Run the first formal hard reject on rootless target fields.
9. Validate and audit the rootless target strict dataset.
10. Build rootless dynamic quality histograms.
11. Build final skeleton/mesh bbox-consistency histograms and final manifests.

## Final Training Inputs

After a successful run, use:

```text
${EVOWEAVE_REBUILD_ROOT}/quality_distributions/rootless_bbox_consistency/final_manifests/train_manifest.jsonl
${EVOWEAVE_REBUILD_ROOT}/quality_distributions/rootless_bbox_consistency/final_manifests/valid_manifest.jsonl
```

The final split policy is train/valid only. Original test rows are merged into
train during final bbox-consistency screening.

The final skeleton/mesh bbox consistency default cutoff is:

```text
target_joint_mesh_bbox_diag_consistency >= 0.4
```

The score is:

```text
min(joint_bbox_diag, mesh_bbox_diag) / max(joint_bbox_diag, mesh_bbox_diag)
```

## Main Debug Files

- `${EVOWEAVE_REBUILD_ROOT}/README_REBUILD_OUTPUTS.txt`
- `${EVOWEAVE_REBUILD_ROOT}/00_source_manifests/`
- `${EVOWEAVE_REBUILD_ROOT}/*/pass1_motionfps_sequence40/manifest.jsonl`
- `${EVOWEAVE_REBUILD_ROOT}/*/pass1_precheck/summary.json`
- `${EVOWEAVE_REBUILD_ROOT}/combined_pass1_precheck/summary.json`
- `${EVOWEAVE_REBUILD_ROOT}/rootless_clean/rootless_summary.json`
- `${EVOWEAVE_REBUILD_ROOT}/rootless_target_strict/summary.json`
- `${EVOWEAVE_REBUILD_ROOT}/rootless_target_strict/rootless_validation.json`
- `${EVOWEAVE_REBUILD_ROOT}/rootless_quality/audit/rootless_final_metrics.csv`
- `${EVOWEAVE_REBUILD_ROOT}/quality_distributions/rootless_bbox_consistency/rootless_bbox_consistency_report.md`

## Contract Notes

- Pass1 joints are explicit Blender bone rows/heads.
- Blender/FBX per-bone tail endpoint coordinates are audit fields only and are
  not training joints.
- Real retained joint rows whose names contain `tail` or `end` are kept only
  when skinned or structurally required.
- Root detection uses the parent graph, not row 0.
- Rootless v3 removes unskinned top root/RT chains, does not add a synthetic
  root, rejects rootless forests, and keeps only mesh components controlled by
  retained rootless joints.
- Deleted roots are recorded in `recorded_root_*` fields only for
  visualization/debugging.
