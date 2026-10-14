[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
. (Join-Path $PSScriptRoot "stage2-common.ps1")
$ctx = Get-Stage2Context $root
$statePath = Join-Path $ctx.StateDir "backend.json"
if (Test-Path $statePath) {
    $state = Get-Content -Raw $statePath | ConvertFrom-Json
    $process = Get-Process -Id ([int]$state.pid) -ErrorAction SilentlyContinue
    if ($process -and $process.StartTime.ToFileTimeUtc() -eq [long]$state.start_time) { & taskkill.exe /PID $process.Id /T /F 2>$null | Out-Null }
    Remove-Item $statePath -Force -ErrorAction SilentlyContinue
}
Wait-Stage2Endpoint "local Qwen" "http://127.0.0.1:11434/health" 5
$python = Join-Path $ctx.PythonRoot ".venv\Scripts\python.exe"
$env:TESSERACT_CMD = Find-Stage2Tesseract $ctx
$env:LOCAL_BASE_URL = "http://127.0.0.1:11434/v1"
$env:LOCAL_MODEL = "qwen2.5-3b-instruct"
$backend = Start-Process $python -WorkingDirectory $root -WindowStyle Hidden -PassThru `
    -ArgumentList @("-m", "uvicorn", "services.orchestrator.main:app", "--host", "127.0.0.1", "--port", "8765") `
    -RedirectStandardOutput (Join-Path $ctx.LogDir "backend.out.log") `
    -RedirectStandardError (Join-Path $ctx.LogDir "backend.err.log")
Save-Stage2Process $ctx "backend" $backend
Wait-Stage2Endpoint "Stage 2 backend" "http://127.0.0.1:8765/api/health" 120
