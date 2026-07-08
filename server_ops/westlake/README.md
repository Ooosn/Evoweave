# Westlake Server Ops Handoff

This folder was copied from the original Anigroom codebase at
`D:\petsgaussianhair`. It contains only server connection, GPU-platform, and
server-run operation files.

The other `D:\Documents\anigroom` folder was checked first, but it only contains
scripts/outputs/external code and no Westlake connection handoff.

## What Was Copied

- `platform_handoff/`: GPU platform helper docs and scripts.
- `platform_handoff/scripts/`: PowerShell/Python helpers for listing GPU dev
  environments, refreshing SSH config, and manual SSH setup.
- `platform_handoff/westlake_jobs_rts_reference/`: old Westlake job script
  examples.
- `scripts/server/`: Anigroom server-side run/preflight/deploy scripts.
- `scripts/generate_white_tiger_orientation_maps_server.sh`
- `scripts/verify_white_tiger_server_env.sh`
- `configs/white_tiger_stage1_formal.env`
- `configs/white_tiger_stage1_sharp_probe.env`
- `ssh_config_my_gpu_server.example`: the current local SSH alias block.

No SSH private key was copied. The key remains at:

```text
C:\Users\namew\.ssh\id_ed25519_my_gpu_server
```

## Known Server Facts From Anigroom

```text
SSH alias: my-gpu-server
Account: liuhaohan
Approved image name: liuhaohan_w
Persistent storage root: /ssdwork/liuhaohan
Anigroom remote project root: /ssdwork/liuhaohan/petsgaussianhair
Python used by Anigroom server scripts: /opt/conda/envs/gs/bin/python
Alternate Python env seen on server: /opt/conda/envs/evoweave/bin/python
```

The SSH host/port can change whenever the GPU dev environment is created,
resumed, restarted, or reconfigured. Treat `ssh_config_my_gpu_server.example` as
a snapshot only; refresh `~/.ssh/config` before connecting.

## Quick Connect

Run these from this folder:

```powershell
cd D:\evoweave\server_ops\westlake\platform_handoff
.\scripts\Get-GpuNodeTime.ps1
.\scripts\Get-GpuDevEnv.ps1 list
```

If a suitable environment is running, refresh the SSH alias:

```powershell
.\scripts\Get-GpuDevEnv.ps1 ssh-config --source-wp-id <wp-id> --confirm
ssh my-gpu-server
```

Verify the target after login:

```powershell
ssh my-gpu-server "hostname; pwd; nvidia-smi; ls -lah /ssdwork/liuhaohan"
```

If no suitable environment is running, follow
`platform_handoff/README.md`. The important rule from the old project is to use
image name `liuhaohan_w`, run `plan` before state-changing operations, and keep
the memory field at `0`.

## Manual SSH Fallback

If the platform UI gives a raw SSH host/port/password:

```powershell
cd D:\evoweave\server_ops\westlake\platform_handoff
.\scripts\Connect-GpuNode.ps1 -HostName 172.16.78.10 -Port <port> -User root
ssh my-gpu-server
```

Prefer `Get-GpuDevEnv.ps1 ssh-config` when possible because it fetches the
current SSH info from the platform API.

## Anigroom Upload And Run Pattern

The old project used:

```powershell
scp <local-file> my-gpu-server:/ssdwork/liuhaohan/petsgaussianhair/<target>
ssh my-gpu-server "cd /ssdwork/liuhaohan/petsgaussianhair && bash scripts/server/preflight_white_tiger_stage1.sh"
ssh my-gpu-server "cd /ssdwork/liuhaohan/petsgaussianhair && bash scripts/server/run_white_tiger_stage1.sh"
```

To adapt these scripts for `evoweave`, change or override these variables:

```bash
PROJECT_ROOT=/ssdwork/liuhaohan/evoweave
PYTHON=/opt/conda/envs/gs/bin/python
CONFIG_PATH=/ssdwork/liuhaohan/evoweave/configs/<your-env-file>.env
OUTPUT_DIR=/ssdwork/liuhaohan/evoweave/outputs/<run-id>
```

The copied `scripts/server/*.sh` files still contain the original Anigroom
defaults. Keep them as references until the evoweave server layout is decided.

## Pull Results Back

Example:

```powershell
New-Item -ItemType Directory -Force D:\evoweave\server_pull | Out-Null
scp -r my-gpu-server:/ssdwork/liuhaohan/petsgaussianhair/outputs/<run-id> D:\evoweave\server_pull\
```

For large folders, use `rsync` from Git Bash/WSL if available:

```bash
rsync -av --progress my-gpu-server:/ssdwork/liuhaohan/petsgaussianhair/outputs/<run-id>/ /d/evoweave/server_pull/<run-id>/
```

## Training Task Rule

Do not paste long multi-line shell programs into the platform command field.
The Anigroom handoff used this pattern:

1. Put the real workflow in a `.sh` file on `/ssdwork/liuhaohan/...`.
2. Validate with `bash -n`.
3. Submit only a short command such as:

```bash
bash /ssdwork/liuhaohan/jobs/<job-id>/run_a.sh
```

See `platform_handoff/westlake_jobs_rts_reference/` for the old examples.

