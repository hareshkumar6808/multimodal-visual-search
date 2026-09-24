[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$tesseract = Join-Path $root ".tools\Tesseract-OCR\tesseract.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Repository Python environment is missing: $python"
}
if (-not (Test-Path -LiteralPath $tesseract)) {
    throw "Repository Tesseract executable is missing: $tesseract"
}

$listeners = Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue
$ownerIds = @($listeners | Select-Object -ExpandProperty OwningProcess -Unique)
foreach ($ownerId in $ownerIds) {
    Write-Host "Stopping old backend on port 8765 (PID $ownerId)..."
    Stop-Process -Id $ownerId -Force -ErrorAction SilentlyContinue
}

foreach ($attempt in 1..20) {
    $stillListening = Get-NetTCPConnection `
        -LocalPort 8765 `
        -State Listen `
        -ErrorAction SilentlyContinue
    if (-not $stillListening) { break }
    if ($attempt -eq 20) {
        throw "Port 8765 is still occupied. Close the terminal running uvicorn and retry."
    }
    Start-Sleep -Milliseconds 250
}

$localHealth = $null
try {
    $localHealth = Invoke-RestMethod -Uri "http://127.0.0.1:11434/health" -TimeoutSec 3
} catch {
    throw "The local AI server is not running on port 11434. Run scripts\run-stage1.ps1 instead."
}

$env:TESSERACT_CMD = $tesseract
$env:LOCAL_BASE_URL = "http://127.0.0.1:11434/v1"
$env:LOCAL_MODEL = "qwen2.5-3b-instruct"
$env:LOCAL_SUPPORTS_VISION = "false"

Write-Host "Starting backend on http://127.0.0.1:8765"
Write-Host "Leave this terminal open. Press Ctrl+C when you want to stop the backend."
Set-Location -LiteralPath $root
& $python -m uvicorn services.orchestrator.main:app --host 127.0.0.1 --port 8765
