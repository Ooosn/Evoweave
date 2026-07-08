# Clean GPU Platform API Reference

This is a compact API/command reference for the helper scripts in this package.
It is platform-specific but project-neutral.

## Authentication

The frontend stores the login token as `Access_Token` and sends:

```text
X-Auth-Token: <token>
```

The helper scripts cache the token at:

```text
scripts/.cache/gpu_platform_token.json
```

Login endpoints:

```http
GET  /api/ibase/v1/system/secret
GET  /api/ibase/v1/captcha
POST /api/ibase/v1/login
```

The password is encrypted with SM2 before login. If captcha is returned, the
script writes `scripts/.cache/captcha.png`.

## Development Environment List

The list view is assembled from:

```http
GET /api/iresource/v1/work-platform/base
GET /api/iresource/v1/work-platform/status
GET /api/iresource/v1/work-platform/usage
```

Helper:

```powershell
.\scripts\Get-GpuDevEnv.ps1 list
```

## Resource Discovery

Useful endpoints:

```http
GET /api/iresource/v1/work-platform/config
GET /api/iresource/v1/work-platform/summary
GET /api/iresource/v1/user-quota
GET /api/iresource/v1/node
GET /api/iresource/v1/node-group?page=1&pageSize=100
GET /api/iresource/v1/node/vgpu-usage
```

Helpers:

```powershell
.\scripts\Get-GpuDevEnv.ps1 resources
.\scripts\Get-GpuDevEnv.ps1 cards
```

When choosing a GPU resource group, inspect group-level resource usage
(used/total) first. The group must have free cards of the requested type, and
the request's `acceleratorCardType` must match the selected group. Some training
groups enforce minimum CPU or GPU counts; do not exceed a user-specified GPU
budget to satisfy those minimums without explicit approval.

## Template And Dry Run

Read a reusable template from an existing environment:

```http
GET /api/iresource/v1/work-platform/{wpId}/rebuild
```

Helpers:

```powershell
.\scripts\Get-GpuDevEnv.ps1 template --source-wp-id <wp-id>
.\scripts\Get-GpuDevEnv.ps1 plan --source-wp-id <wp-id> --gpu-count 1 --gpu-type NVIDIA-A100-SXM4-80GB
```

`plan` prints a create payload and does not submit it.

## Create

Endpoint:

```http
POST /api/iresource/v1/work-platform/
```

Helper:

```powershell
.\scripts\Get-GpuDevEnv.ps1 plan --image-name liuhaohan_w --image-tag <approved-tag> --gpu-count 1 --gpu-type NVIDIA-A100-SXM4-80GB --cpu 16 --memory 0
.\scripts\Get-GpuDevEnv.ps1 create --image-name liuhaohan_w --image-tag <approved-tag> --gpu-count 1 --gpu-type NVIDIA-A100-SXM4-80GB --cpu 16 --memory 0 --wait --confirm
```

Mandatory image rule:

- All create/restart/reconfigure/commit operations must use image name
  `liuhaohan_w`.
- Do not use project-named, dated, or experimental images unless the user
  explicitly approves it in the current conversation.
- Do not create a new image name or new tag without explicit user approval.
- If the environment changes, commit/export back into `liuhaohan_w` only,
  overwriting the approved tag selected for that run.
- If a command requires a tag, confirm the current approved tag first; do not
  reuse stale tags from this document.
- Before state-changing image/environment operations, run `plan` and verify the
  payload image resolves to `liuhaohan_w`.
- Keep the payload `memory` field at `0`. Do not set it to node physical memory
  values such as 500, 755, or 1007 GB.

Common payload fields:

```json
{
  "wpName": "<generated or supplied name>",
  "frameWork": "pytorch",
  "groupId": "<resource group>",
  "wpPodNum": 1,
  "cpu": 16,
  "memory": 0,
  "acceleratorCard": 1,
  "acceleratorCardType": "NVIDIA-A100-SXM4-80GB",
  "image": "192.168.108.1:5000/pytorch/liuhaohan_w-liuhaohan:<approved-tag>",
  "imageType": "INNER_IMAGE",
  "shmSize": 4,
  "volumes": [],
  "ports": [],
  "env": null,
  "models": []
}
```

## Reuse, Restart, Reconfigure, Stop

Endpoints:

```http
PUT  /api/iresource/v1/work-platform/pause/{wpId}
GET  /api/iresource/v1/work-platform/restore/{wpId}
PUT  /api/iresource/v1/work-platform/restore/{wpId}
POST /api/iresource/v1/work-platform/{wpId}
POST /api/iresource/v1/work-platform/{wpId}/reconfig
DELETE or platform action via terminate helper
```

Helpers:

```powershell
.\scripts\Get-GpuDevEnv.ps1 restart --source-wp-id <wp-id> --gpu-count 1 --gpu-type NVIDIA-A100-SXM4-80GB --memory 0 --wait --confirm
.\scripts\Get-GpuDevEnv.ps1 resume --source-wp-id <wp-id> --gpu-count 1 --gpu-type NVIDIA-A100-SXM4-80GB --memory 0 --wait --confirm
.\scripts\Get-GpuDevEnv.ps1 resource-change --source-wp-id <wp-id> --gpu-count 4 --gpu-type NVIDIA-A100-SXM4-80GB --memory 0 --wait --confirm
.\scripts\Get-GpuDevEnv.ps1 pause --source-wp-id <wp-id> --wait --confirm
.\scripts\Get-GpuDevEnv.ps1 terminate --source-wp-id <wp-id> --wait --confirm
```

After any action that changes the pod or SSH port:

```powershell
.\scripts\Get-GpuDevEnv.ps1 ssh-config --source-wp-id <wp-id> --confirm
```

## SSH Config

The helper fetches encrypted SSH info from the platform and writes a local SSH
alias.

Relevant endpoint:

```http
GET /api/iresource/v1/work-platform/{wpId}/pod/{podId}/shell
```

Helper:

```powershell
.\scripts\Get-GpuDevEnv.ps1 ssh-config --source-wp-id <wp-id> --confirm
ssh my-gpu-server
```

It creates or reuses:

```text
~/.ssh/id_ed25519_my_gpu_server
~/.ssh/config entry: Host my-gpu-server
```

## Image Commit

Check image name/tag:

```http
GET or POST /api/iresource/v1/images/check
```

Submit image commit:

```http
POST /api/iresource/v1/work-platform/commit-image
```

Progress:

```http
GET /api/iresource/v1/images/progress
```

Helpers:

```powershell
.\scripts\Get-GpuDevEnv.ps1 commit-image --image-name liuhaohan_w --image-tag <approved-tag> --wait --confirm
.\scripts\Get-GpuDevEnv.ps1 image-progress --image-name liuhaohan_w --image-tag <approved-tag> --wait
```

Commit/export only back into `liuhaohan_w`. Do not create a new image name or
new tag unless explicitly approved by the user.

Omit `--pod-id` unless you have the platform pod UUID. The pod name shown by
`list` is not the same value.

## Wait Semantics

```text
--wait-seconds: no-change timeout
--max-seconds: optional hard cap; 0 means no hard cap
```

If status/progress changes, the helper resets the no-change timer.

## Packaged Scripts

```text
scripts/Get-GpuNodeTime.ps1
scripts/Get-GpuDevEnv.ps1
scripts/Connect-GpuNode.ps1
scripts/gpu_node_time.py
scripts/gpu_dev_env.py
```

The PowerShell wrappers set environment variables and call the Python scripts.
They install missing Python packages with `python -m pip install --user` when
needed.
