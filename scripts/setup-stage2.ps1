[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
. (Join-Path $PSScriptRoot "stage2-common.ps1")
$ctx = Get-Stage2Context $root
New-Item -ItemType Directory -Force $ctx.StateDir, $ctx.LogDir | Out-Null

if (-not $ctx.PythonRoot) {
    & (Join-Path $PSScriptRoot "setup-stage1.ps1")
    $ctx = Get-Stage2Context $root
}
$python = Join-Path $ctx.PythonRoot ".venv\Scripts\python.exe"
$llama = Join-Path $ctx.AssetRoot ".tools\llama\llama-server.exe"
$model = Join-Path $ctx.AssetRoot ".tools\ollama\qwen2.5-3b-instruct-q4_k_m.gguf"
foreach ($required in @($python, $llama, $model, (Join-Path $root "apps\desktop"))) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Required Stage 2 dependency is missing: $required" }
}
Find-Stage2Tesseract $ctx | Out-Null
Set-Stage2RustEnvironment $ctx | Out-Null

if (-not (Test-Path (Join-Path $root ".env"))) {
    $sharedEnv = Join-Path $ctx.AssetRoot ".env"
    if (Test-Path -LiteralPath $sharedEnv) { Copy-Item -LiteralPath $sharedEnv -Destination (Join-Path $root ".env") }
}

$env:TEMP = Join-Path $ctx.StateDir "tmp"
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
Write-Host "Installing integrated Stage 2 backend dependencies..."
& $python -m pip install -e "${root}[dev,perception-onnx]"
if ($LASTEXITCODE -ne 0) { throw "Stage 2 Python dependency installation failed." }
Write-Host "Installing desktop dependencies..."
Set-Stage2NodeEnvironment $ctx
& npm.cmd ci --prefix (Join-Path $root "apps\desktop")
if ($LASTEXITCODE -ne 0) { throw "Desktop dependency installation failed." }
Write-Host "Stage 2 setup is complete. Existing Qwen assets were reused from: $($ctx.AssetRoot)"
