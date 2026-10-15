[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
. (Join-Path $PSScriptRoot "stage2-common.ps1")
$ctx = Get-Stage2Context $root
New-Item -ItemType Directory -Force $ctx.StateDir, $ctx.LogDir | Out-Null
& (Join-Path $PSScriptRoot "stop-stage2.ps1") -Quiet
Stop-StaleStage2Backend

foreach ($port in @(8768, 1420)) {
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
$env:PROVIDER_TIMEOUT_SECONDS = "10"
Set-Stage2RustEnvironment $ctx | Out-Null
Set-Stage2NodeEnvironment $ctx

foreach ($name in @("backend.out.log", "backend.err.log", "local-provider.out.log", "local-provider.err.log", "desktop.out.log", "desktop.err.log")) {
    $path = Join-Path $ctx.LogDir $name
    if (Test-Path $path) { Clear-Content $path }
}

& curl.exe --silent --fail --max-time 2 --output NUL "http://127.0.0.1:11434/health" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "[ready] existing local Qwen"
} else {
    $localListener = Get-NetTCPConnection -LocalPort 11434 -State Listen -ErrorAction SilentlyContinue
    if ($localListener) {
        throw "Port 11434 is occupied by a service that is not a healthy Qwen provider."
    }
    Write-Host "Starting local Qwen provider..."
    $local = Start-Process $llama -WorkingDirectory $root -WindowStyle Hidden -PassThru `
        -ArgumentList @("-m", $model, "--host", "127.0.0.1", "--port", "11434", "--alias", $env:LOCAL_MODEL, "-c", "4096", "-ngl", "99") `
        -RedirectStandardOutput (Join-Path $ctx.LogDir "local-provider.out.log") `
        -RedirectStandardError (Join-Path $ctx.LogDir "local-provider.err.log")
    Save-Stage2Process $ctx "local-provider" $local
    Wait-Stage2Endpoint "local Qwen" "http://127.0.0.1:11434/health" 240
}

Write-Host "Starting Stage 2 backend..."
$backend = Start-Process $python -WorkingDirectory $root -WindowStyle Hidden -PassThru `
    -ArgumentList @("-m", "uvicorn", "services.orchestrator.main:app", "--host", "127.0.0.1", "--port", "8768") `
    -RedirectStandardOutput (Join-Path $ctx.LogDir "backend.out.log") `
    -RedirectStandardError (Join-Path $ctx.LogDir "backend.err.log")
Save-Stage2Process $ctx "backend" $backend
Wait-Stage2Backend 120

$providersJson = & curl.exe --silent --fail --max-time 5 "http://127.0.0.1:8768/api/providers"
if ($LASTEXITCODE -ne 0) { throw "Backend provider status endpoint is unavailable." }
$providers = $providersJson | ConvertFrom-Json
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
