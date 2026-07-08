param(
  [string]$HostName,
  [int]$Port,
  [string]$User = "root",
  [string]$HostAlias = "my-gpu-server",
  [string]$KeyPath = "$env:USERPROFILE\.ssh\id_ed25519_my_gpu_server"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not $HostName) {
  $HostName = Read-Host "HostName/IP"
}

if (-not $Port) {
  $Port = [int](Read-Host "SSH port")
}

$securePassword = Read-Host "SSH password" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
try {
  $plainPassword = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
} finally {
  [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
}

$sshDir = Join-Path $env:USERPROFILE ".ssh"
New-Item -ItemType Directory -Force $sshDir | Out-Null

if (-not (Test-Path $KeyPath)) {
  $keygenCommand = 'ssh-keygen -t ed25519 -f "' + $KeyPath + '" -N "" -C codex-my-gpu-server'
  cmd.exe /c $keygenCommand
}

$pubKeyPath = "$KeyPath.pub"
if (-not (Test-Path $pubKeyPath)) {
  throw "Public key not found: $pubKeyPath"
}

$pubKey = (Get-Content -Raw $pubKeyPath).Trim()

python -c "import paramiko" 2>$null
if ($LASTEXITCODE -ne 0) {
  python -m pip install --user paramiko
}

$env:GPU_NODE_HOSTNAME = $HostName
$env:GPU_NODE_PORT = [string]$Port
$env:GPU_NODE_USER = $User
$env:GPU_NODE_PASSWORD = $plainPassword
$env:GPU_NODE_PUBKEY = $pubKey

try {
  @'
import os
import posixpath
import paramiko

host = os.environ["GPU_NODE_HOSTNAME"]
port = int(os.environ["GPU_NODE_PORT"])
user = os.environ["GPU_NODE_USER"]
password = os.environ["GPU_NODE_PASSWORD"]
pubkey = os.environ["GPU_NODE_PUBKEY"].strip()

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(
    hostname=host,
    port=port,
    username=user,
    password=password,
    look_for_keys=False,
    allow_agent=False,
    timeout=20,
)

stdin, stdout, stderr = client.exec_command('printf %s "$HOME"')
home = stdout.read().decode("utf-8", errors="replace").strip()
if not home:
    home = "/root" if user == "root" else "/home/" + user

sftp = client.open_sftp()
try:
    ssh_dir = posixpath.join(home, ".ssh")
    auth_keys = posixpath.join(ssh_dir, "authorized_keys")
    try:
        sftp.mkdir(ssh_dir, mode=0o700)
    except OSError:
        pass
    try:
        existing = sftp.file(auth_keys, "r").read().decode("utf-8", errors="replace")
    except OSError:
        existing = ""
    if pubkey not in existing:
        with sftp.file(auth_keys, "a") as f:
            if existing and not existing.endswith("\n"):
                f.write("\n")
            f.write(pubkey + "\n")
    sftp.chmod(ssh_dir, 0o700)
    sftp.chmod(auth_keys, 0o600)
finally:
    sftp.close()

stdin, stdout, stderr = client.exec_command("hostname; whoami; pwd")
print(stdout.read().decode("utf-8", errors="replace"), end="")
err = stderr.read().decode("utf-8", errors="replace")
if err:
    print(err, end="")
client.close()
'@ | python -
} finally {
  Remove-Item Env:\GPU_NODE_HOSTNAME -ErrorAction SilentlyContinue
  Remove-Item Env:\GPU_NODE_PORT -ErrorAction SilentlyContinue
  Remove-Item Env:\GPU_NODE_USER -ErrorAction SilentlyContinue
  Remove-Item Env:\GPU_NODE_PASSWORD -ErrorAction SilentlyContinue
  Remove-Item Env:\GPU_NODE_PUBKEY -ErrorAction SilentlyContinue
  $plainPassword = $null
}

$configPath = Join-Path $sshDir "config"
if (Test-Path $configPath) {
  $lines = @(Get-Content $configPath)
} else {
  $lines = @()
}

$filtered = New-Object System.Collections.Generic.List[string]
$skip = $false
foreach ($line in $lines) {
  if ($line -match '^\s*Host\s+(.+?)\s*$') {
    $hostNames = $Matches[1] -split '\s+'
    $skip = $hostNames -contains $HostAlias
  }
  if (-not $skip) {
    $filtered.Add($line)
  }
}

$keyPathForConfig = $KeyPath.Replace("\", "/")
$block = @(
  "Host $HostAlias",
  "  HostName $HostName",
  "  User $User",
  "  Port $Port",
  "  IdentityFile $keyPathForConfig",
  "  IdentitiesOnly yes",
  "  PreferredAuthentications publickey,password",
  "  StrictHostKeyChecking accept-new",
  "  ServerAliveInterval 60",
  "  ServerAliveCountMax 3"
)

while ($filtered.Count -gt 0 -and [string]::IsNullOrWhiteSpace($filtered[$filtered.Count - 1])) {
  $filtered.RemoveAt($filtered.Count - 1)
}

if ($filtered.Count -gt 0) {
  $filtered.Add("")
}
$filtered.AddRange($block)
Set-Content -Path $configPath -Value $filtered -Encoding UTF8

ssh-keygen -R "[$HostName]:$Port" 2>$null | Out-Null
ssh-keygen -R "$HostName" 2>$null | Out-Null

Write-Host ""
Write-Host "Testing key login through SSH alias '$HostAlias'..."
ssh -o BatchMode=yes -o ConnectTimeout=10 $HostAlias "hostname; whoami; pwd"

Write-Host ""
Write-Host "Ready. Use: ssh $HostAlias"
