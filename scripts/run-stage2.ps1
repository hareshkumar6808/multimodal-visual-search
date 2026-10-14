[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
. (Join-Path $PSScriptRoot "stage2-common.ps1")
$ctx = Get-Stage2Context $root
New-Item -ItemType Directory -Force $ctx.StateDir, $ctx.LogDir | Out-Null
& (Join-Path $PSScriptRoot "stop-stage2.ps1") -Quiet

foreach ($port in @(8765, 11434, 1420)) {
    $listeners = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($listeners) { throw "Port $port is already occupied by PID $($listeners[0].OwningProcess). Stop that process or the older project launcher first." }
}

$python = Join-Path $ctx.PythonRoot ".venv\Scripts\python.exe"
$llama = Join-Path $ctx.AssetRoot ".tools\llama\llama-server.exe"
$model = Join-Path $ctx.AssetRoot ".tools\ollama\qwen2.5-3b-instruct-q4_k_m.gguf"
foreach ($required in @($python, $llama, $model)) {
    if (-not (Test-Path $required)) { throw "Missing Stage 2 runtime dependency: $required. Run setup-stage2.ps1." }
}
$env:TESSERACT_CMD = Find-Stage2Tesseract $ctx
$env:LOCAL_BASE_URL = "http://127.0.0.1:11434/v1"
$env:LOCAL_MODEL = "qwen2.5-3b-instruct"
$env:LOCAL_SUPPORTS_VISION = "false"
Set-Stage2RustEnvironment $ctx | Out-Null
Set-Stage2NodeEnvironment $ctx

foreach ($name in @("backend.out.log", "backend.err.log", "local-provider.out.log", "local-provider.err.log", "desktop.out.log", "desktop.err.log")) {
    $path = Join-Path $ctx.LogDir $name
    if (Test-Path $path) { Clear-Content $path }
}

Write-Host "Starting local Qwen provider..."
$local = Start-Process $llama -WorkingDirectory $root -WindowStyle Hidden -PassThru `
    -ArgumentList @("-m", $model, "--host", "127.0.0.1", "--port", "11434", "--alias", $env:LOCAL_MODEL, "-c", "4096", "-ngl", "99") `
    -RedirectStandardOutput (Join-Path $ctx.LogDir "local-provider.out.log") `
    -RedirectStandardError (Join-Path $ctx.LogDir "local-provider.err.log")
Save-Stage2Process $ctx "local-provider" $local
Wait-Stage2Endpoint "local Qwen" "http://127.0.0.1:11434/health" 240

Write-Host "Starting Stage 2 backend..."
$backend = Start-Process $python -WorkingDirectory $root -WindowStyle Hidden -PassThru `
    -ArgumentList @("-m", "uvicorn", "services.orchestrator.main:app", "--host", "127.0.0.1", "--port", "8765") `
    -RedirectStandardOutput (Join-Path $ctx.LogDir "backend.out.log") `
    -RedirectStandardError (Join-Path $ctx.LogDir "backend.err.log")
Save-Stage2Process $ctx "backend" $backend
Wait-Stage2Endpoint "Stage 2 backend" "http://127.0.0.1:8765/api/health" 120

$providers = Invoke-RestMethod "http://127.0.0.1:8765/api/providers" -TimeoutSec 5
$ready = $providers | Where-Object { $_.configured -and $_.available } | Select-Object -First 1
if (-not $ready) { throw "Backend is healthy but no provider is ready." }
Write-Host "[ready] AI provider: $($ready.name)"

Write-Host "Starting native Stage 2 desktop..."
$desktopDir = Join-Path $root "apps\desktop"
$desktop = Start-Process "npm.cmd" -WorkingDirectory $desktopDir -WindowStyle Hidden -PassThru `
    -ArgumentList @("run", "tauri:dev") `
    -RedirectStandardOutput (Join-Path $ctx.LogDir "desktop.out.log") `
    -RedirectStandardError (Join-Path $ctx.LogDir "desktop.err.log")
Save-Stage2Process $ctx "desktop" $desktop

foreach ($attempt in 1..600) {
    $native = Get-Process -Name "multimodal-visual-search-desktop" -ErrorAction SilentlyContinue | Sort-Object StartTime -Descending | Select-Object -First 1
    if ($native) { Save-Stage2Process $ctx "native-desktop" $native; Write-Host "[ready] native desktop"; break }
    if ($desktop.HasExited) { throw "Tauri exited. See $($ctx.LogDir)\desktop.err.log" }
    if ($attempt -eq 600) { throw "Native desktop did not launch within five minutes." }
    Start-Sleep -Milliseconds 500
}
Write-Host "Stage 2 is ready. Logs: $($ctx.LogDir)"
