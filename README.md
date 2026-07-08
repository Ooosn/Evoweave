# Evoweave

Evoweave is organized as separate code modules with GitHub as the source of
truth for code synchronization.

## Modules

- `data_processing/`: raw-data export, rootless skeleton cleanup, quality
  metrics, histogram generation, and final manifest construction.
- `model_training/`: model code, training launchers, evaluation scripts, and
  model-side documentation.
- `server_ops/`: reusable Westlake/platform helper scripts and lightweight job
  launch helpers.

## Sync Policy

- Core code moves through GitHub only.
- Server environments should clone or pull this repository before running new
  code.
- Datasets, NPZ files, checkpoints, logs, rendered reports, cached platform
  tokens, and generated outputs stay out of Git.
- Large third-party references and pretrained checkpoints are kept on local or
  server storage, not in this repository.

## Current Baselines

- UniRig flat autoregressive baseline on rootless-v3 data.
- Puppeteer decoder baseline on rootless-v3 data.

Training jobs should read the final train/valid manifests directly and must not
rescan NPZ directories.
