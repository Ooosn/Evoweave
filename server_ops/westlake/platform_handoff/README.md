# Clean GPU Platform Handoff

Purpose: let another Codex/chat window connect to and operate the GPU platform
for any project. This file intentionally contains no project-specific paths,
training code, datasets, or research notes.

## First Rule

Every session starts with:

```powershell
.\scripts\Get-GpuNodeTime.ps1
```

The development environment is short-lived, normally about 4 hours. If the
active environment has less than 30 minutes remaining, do not start long work.
Create/restart/resume a fresh environment first.

## Platform Defaults

```text
Base URL: https://172.16.78.10:32206
Account: liuhaohan
SSH alias: my-gpu-server
Approved image name: liuhaohan_w
Default debug GPU: 1 x NVIDIA-A100-SXM4-80GB
Default debug CPU: 16
Default memory field: 0
```

The scripts prompt for the platform password when needed and cache only the
platform token under `scripts/.cache/`.

## Mandatory Image Rule

All environment operations must use the approved image name `liuhaohan_w`.

- Do not use an old environment just because it exists.
- Do not use project-named images or dated experimental images unless the user
  explicitly approves it in the current conversation.
- Do not create a new image name or a new tag without explicit user approval.
- If packages or system dependencies are installed in a development
  environment, export/commit back into `liuhaohan_w` only, overwriting the
  approved tag selected for that run.
- If a helper command requires an image tag, first confirm the current approved
  tag for `liuhaohan_w`; do not reuse stale tags from this document.
- Before any create/restart/commit command, run a dry-run/plan and verify the
  payload image resolves to `liuhaohan_w`. Stop if it does not.
- For development-environment payloads, keep `memory` at `0`. Do not set it to
  node physical memory values such as 500, 755, or 1007 GB.

## Quick Start

Inspect environments and remaining time:

```powershell
.\scripts\Get-GpuNodeTime.ps1
.\scripts\Get-GpuDevEnv.ps1 list
```

If there is a suitable running environment, configure SSH:

```powershell
.\scripts\Get-GpuDevEnv.ps1 ssh-config --source-wp-id <wp-id> --confirm
ssh my-gpu-server
```

If there is no suitable running environment, create one:

```powershell
.\scripts\Get-GpuDevEnv.ps1 plan --image-name liuhaohan_w --image-tag <approved-tag> --gpu-count 1 --gpu-type NVIDIA-A100-SXM4-80GB --cpu 16 --memory 0
.\scripts\Get-GpuDevEnv.ps1 create --image-name liuhaohan_w --image-tag <approved-tag> --gpu-count 1 --gpu-type NVIDIA-A100-SXM4-80GB --cpu 16 --memory 0 --wait --confirm
.\scripts\Get-GpuDevEnv.ps1 ssh-config --source-wp-id <new-wp-id> --confirm
ssh my-gpu-server
```

If an environment is halted/expired and the image is already correct, prefer
reuse over creating many environment records:

```powershell
.\scripts\Get-GpuDevEnv.ps1 resume --source-wp-id <wp-id> --gpu-count 1 --gpu-type NVIDIA-A100-SXM4-80GB --cpu 16 --memory 0 --wait --confirm
.\scripts\Get-GpuDevEnv.ps1 ssh-config --source-wp-id <wp-id> --confirm
```

If `restart` fails with `IRESOURCE_CREATE_SERVICE_FAILED`, use terminate then
create:

```powershell
.\scripts\Get-GpuDevEnv.ps1 terminate --source-wp-id <old-wp-id> --wait --confirm
.\scripts\Get-GpuDevEnv.ps1 plan --image-name liuhaohan_w --image-tag <approved-tag> --gpu-count 1 --gpu-type NVIDIA-A100-SXM4-80GB --cpu 16 --memory 0
.\scripts\Get-GpuDevEnv.ps1 create --image-name liuhaohan_w --image-tag <approved-tag> --gpu-count 1 --gpu-type NVIDIA-A100-SXM4-80GB --cpu 16 --memory 0 --wait --confirm
.\scripts\Get-GpuDevEnv.ps1 ssh-config --source-wp-id <new-wp-id> --confirm
```

## Useful Read-Only Commands

```powershell
.\scripts\Get-GpuDevEnv.ps1 list
.\scripts\Get-GpuDevEnv.ps1 cards
.\scripts\Get-GpuDevEnv.ps1 resources
.\scripts\Get-GpuDevEnv.ps1 template
```

Before creating a GPU development environment or submitting a training task,
check the resource-group detail page or helper output for resource usage shown
as used/total. Choose a group whose requested card type has free cards. Do not
infer availability from the card name alone.

The card type must match the selected resource group. For example, a 40GB A100
request will be rejected if the payload still points at an 80GB development
group.

The helper now tries to match `groupId` / `groupName` to the requested
`--gpu-type` by reading the node resource table. Still inspect `plan` before
`create`: development environments may reject non-development resource group
labels even when the card type itself is available.

Some training groups enforce minimum resource sizes. If task creation reports a
minimum CPU or GPU count for a group, do not silently exceed the user's requested
GPU budget. Choose another compatible group or leave the task queued.

Preview a create payload without creating anything:

```powershell
.\scripts\Get-GpuDevEnv.ps1 plan --image-name liuhaohan_w --image-tag <approved-tag> --gpu-count 1 --gpu-type NVIDIA-A100-SXM4-80GB --cpu 16 --memory 0
```

## State-Changing Commands

All state-changing helper commands require `--confirm`:

```powershell
.\scripts\Get-GpuDevEnv.ps1 create ...
.\scripts\Get-GpuDevEnv.ps1 restart ...
.\scripts\Get-GpuDevEnv.ps1 resume ...
.\scripts\Get-GpuDevEnv.ps1 resource-change ...
.\scripts\Get-GpuDevEnv.ps1 pause ...
.\scripts\Get-GpuDevEnv.ps1 terminate ...
.\scripts\Get-GpuDevEnv.ps1 commit-image ...
```

Change GPU count/type on an existing environment:

```powershell
.\scripts\Get-GpuDevEnv.ps1 resource-change --source-wp-id <wp-id> --gpu-count 4 --gpu-type NVIDIA-A100-SXM4-80GB --memory 0 --wait --confirm
.\scripts\Get-GpuDevEnv.ps1 ssh-config --source-wp-id <wp-id> --confirm
```

## Image Rules

Use only the approved image name:

```text
liuhaohan_w
```

Commit/export an image only when the environment itself changed, for example:

- packages installed into `/opt/conda`, `/usr/local`, or system paths;
- apt packages;
- shell/Jupyter/SSH configuration;
- files under `/root` that must persist.

Usually do not save an image just because project code or data under mounted
storage changed.

Commit image back into `liuhaohan_w` only. Do not invent a new tag:

```powershell
.\scripts\Get-GpuDevEnv.ps1 commit-image --image-name liuhaohan_w --image-tag <approved-tag> --wait --confirm
```

Do not pass the pod name printed by `list` as `--pod-id`. Only use `--pod-id`
when you have the platform pod UUID.

## Where To Put Work On The Server

Prefer persistent mounted storage for new projects:

```text
/ssdwork/liuhaohan
/liuhaohan
```

Avoid relying on temporary container-only paths for important code, datasets,
checkpoints, or logs.

## Development Environment vs Training Task

Use development environments for:

- writing and debugging scripts;
- smoke tests;
- inspecting logs;
- short evaluations;
- preparing commands for long jobs.

Use platform training tasks for:

- long downloads;
- long data cleaning;
- full training;
- anything that should continue beyond the 4-hour development window.

### Training Task Command Rule

Do not put a multi-line shell program, long quoting chain, or several piped
commands directly into the platform `command` field. The platform may normalize
newlines and quoting before launching the task, which can silently turn a real
job into a short no-op or an incorrectly parsed command.

For any non-trivial training task:

1. Write the full workflow as a versioned `.sh` file on persistent storage.
2. Validate it first with `bash -n`, then with `DRY_RUN=1` if the script supports
   dry runs.
3. Submit only a short command to the training task, for example:

```bash
bash /ssdwork/liuhaohan/RTS/project/scripts/run_training.sh
```

Avoid commands like:

```bash
set -euo pipefail; RUN_ID=...; cmd1 2>&1 | tee ...; cmd2 2>&1 | tee ...
```

If a task fails, create a rescue `.sh` for the remaining work and submit that
script path. Do not rescue by hand-composing a long platform command.

## Manual SSH Fallback

If the platform UI gives a raw SSH host/port/password, use:

```powershell
.\scripts\Connect-GpuNode.ps1 -HostName 172.16.78.10 -Port <port> -User root
ssh my-gpu-server
```

Normally prefer `Get-GpuDevEnv.ps1 ssh-config`, because it fetches and decrypts
the current SSH info directly from the platform API.

## Troubleshooting

- SSH port changes after create/restart/resume. Always run `ssh-config`.
- If login asks for captcha, the script writes `scripts/.cache/captcha.png`.
  Open the image, then rerun with `-CaptchaCode <code>` or enter it when
  prompted.
- Delete `scripts/.cache/gpu_platform_token.json` to force a fresh login.
- `--wait-seconds` means no-change timeout. If status/progress changes, the
  script keeps waiting. `--max-seconds 0` means no hard cap.
- After reconnecting, run `hostname`, `pwd`, and `nvidia-smi` to verify that the
  SSH alias points at the expected pod.
