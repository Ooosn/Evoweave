param(
  [string]$BaseUrl = "https://172.16.78.10:32206",
  [string]$Account = "liuhaohan",
  [securestring]$Password,
  [string]$CaptchaCode,
  [string]$SshAlias = "my-gpu-server"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$plainPassword = $null
if ($Password) {
  $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($Password)
  try {
    $plainPassword = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
  } finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
  }
}

python -c "import requests, gmssl" 2>$null
if ($LASTEXITCODE -ne 0) {
  python -m pip install --user requests gmssl
}

$cacheDir = Join-Path $PSScriptRoot ".cache"
New-Item -ItemType Directory -Force $cacheDir | Out-Null

$env:GPU_PLATFORM_BASE_URL = $BaseUrl
$env:GPU_PLATFORM_ACCOUNT = $Account
$env:GPU_PLATFORM_CAPTCHA_PATH = (Join-Path $cacheDir "captcha.png")
$env:GPU_NODE_SSH_ALIAS = $SshAlias
if ($plainPassword) {
  $env:GPU_PLATFORM_PASSWORD = $plainPassword
}
if ($CaptchaCode) {
  $env:GPU_PLATFORM_CAPTCHA = $CaptchaCode
}

try {
  python (Join-Path $PSScriptRoot "gpu_node_time.py")
} finally {
  Remove-Item Env:\GPU_PLATFORM_BASE_URL -ErrorAction SilentlyContinue
  Remove-Item Env:\GPU_PLATFORM_ACCOUNT -ErrorAction SilentlyContinue
  Remove-Item Env:\GPU_PLATFORM_PASSWORD -ErrorAction SilentlyContinue
  Remove-Item Env:\GPU_PLATFORM_CAPTCHA_PATH -ErrorAction SilentlyContinue
  Remove-Item Env:\GPU_NODE_SSH_ALIAS -ErrorAction SilentlyContinue
  Remove-Item Env:\GPU_PLATFORM_CAPTCHA -ErrorAction SilentlyContinue
  $plainPassword = $null
}
