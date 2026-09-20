[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$stateDir = Join-Path $root ".tools\stage1"
$logDir = Join-Path $stateDir "logs"
New-Item -ItemType Directory -Force -Path $stateDir, $logDir | Out-Null

function Save-ProcessState {
    param([string]$Name, [System.Diagnostics.Process]$Process)
    $Process.Refresh()
    @{ pid = $Process.Id; start_time = $Process.StartTime.ToFileTimeUtc() } |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $stateDir "$Name.json")
}

function Wait-Endpoint {
    param([string]$Name, [string]$Uri, [int]$Attempts = 60)
    foreach ($attempt in 1..$Attempts) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $Uri -TimeoutSec 3
            if ($response.StatusCode -eq 200) {
                Write-Host "[ready] $Name"
                return
            }
        } catch {
            if ($attempt -eq $Attempts) { throw "$Name did not become ready: $($_.Exception.Message)" }
        }
        Start-Sleep -Milliseconds 500
    }
}

function Find-Tesseract {
    $command = Get-Command tesseract.exe -ErrorAction SilentlyContinue
    $candidates = @(
        $command.Source,
        (Join-Path $root ".tools\Tesseract-OCR\tesseract.exe"),
        "$env:ProgramFiles\Tesseract-OCR\tesseract.exe",
        "${env:ProgramFiles(x86)}\Tesseract-OCR\tesseract.exe"
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }
    if (-not $candidates) { throw "Tesseract was not found on PATH or in the supported local locations." }
    return (Resolve-Path -LiteralPath ($candidates | Select-Object -First 1)).Path
}

& (Join-Path $PSScriptRoot "stop-stage1.ps1") -Quiet

$llamaServer = Join-Path $root ".tools\llama\llama-server.exe"
$localModel = Join-Path $root ".tools\ollama\qwen2.5-3b-instruct-q4_k_m.gguf"
$python = Join-Path $root ".venv\Scripts\python.exe"
$rustBin = Join-Path $root ".tools\msys2\msys64\mingw64\bin"
$cargoHome = Join-Path $root ".tools\cargo"
$rustupHome = Join-Path $root ".tools\rustup"
$desktopDir = Join-Path $root "apps\desktop"
foreach ($required in @($llamaServer, $localModel, $python, $rustBin, $cargoHome, $rustupHome, $desktopDir)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Required Stage 1 dependency is missing: $required" }
}

$env:TESSERACT_CMD = Find-Tesseract
$env:LOCAL_BASE_URL = "http://127.0.0.1:11434/v1"
$env:LOCAL_MODEL = "qwen2.5-3b-instruct"
$env:LOCAL_SUPPORTS_VISION = "false"
$env:PATH = "$rustBin;$env:PATH"
$env:CARGO_HOME = $cargoHome
$env:RUSTUP_HOME = $rustupHome
$env:CARGO_SOURCE_LOCAL_SPARSE_REGISTRY = "sparse+https://index.crates.io/"
$env:CARGO_TARGET_DIR = Join-Path $env:TEMP "mvs-stage1-msys-target"
$env:RUSTFLAGS = "-C link-arg=-Wl,--exclude-all-symbols"

Write-Host "Starting local AI provider..."
$local = Start-Process -FilePath $llamaServer -WorkingDirectory $root -WindowStyle Hidden -PassThru `
    -ArgumentList @("-m", $localModel, "--host", "127.0.0.1", "--port", "11434", "--alias", $env:LOCAL_MODEL, "-c", "4096", "-ngl", "99") `
    -RedirectStandardOutput (Join-Path $logDir "local-provider.out.log") `
    -RedirectStandardError (Join-Path $logDir "local-provider.err.log")
Save-ProcessState "local-provider" $local
Wait-Endpoint "local AI provider" "http://127.0.0.1:11434/health"

Write-Host "Starting Stage 1 backend..."
$backend = Start-Process -FilePath $python -WorkingDirectory $root -WindowStyle Hidden -PassThru `
    -ArgumentList @("-m", "uvicorn", "services.orchestrator.main:app", "--host", "127.0.0.1", "--port", "8765") `
    -RedirectStandardOutput (Join-Path $logDir "backend.out.log") `
    -RedirectStandardError (Join-Path $logDir "backend.err.log")
Save-ProcessState "backend" $backend
Wait-Endpoint "backend" "http://127.0.0.1:8765/api/health"

$providers = Invoke-RestMethod -Uri "http://127.0.0.1:8765/api/providers" -TimeoutSec 5
$readyProvider = $providers | Where-Object { $_.configured -and $_.available } | Select-Object -First 1
if (-not $readyProvider) { throw "No configured and available AI provider was reported by the backend." }
Write-Host "[ready] AI provider: $($readyProvider.name)"

Write-Host "Starting native Tauri desktop..."
$desktop = Start-Process -FilePath "npm.cmd" -WorkingDirectory $desktopDir -WindowStyle Hidden -PassThru `
    -ArgumentList @("run", "tauri:dev") `
    -RedirectStandardOutput (Join-Path $logDir "desktop.out.log") `
    -RedirectStandardError (Join-Path $logDir "desktop.err.log")
Save-ProcessState "desktop" $desktop

$nativeReady = $false
foreach ($attempt in 1..120) {
    $native = Get-Process -Name "multimodal-visual-search-desktop" -ErrorAction SilentlyContinue |
        Sort-Object StartTime -Descending | Select-Object -First 1
    if ($native) {
        Save-ProcessState "native-desktop" $native
        $nativeReady = $true
        break
    }
    if ($desktop.HasExited) { throw "Tauri launcher exited. See $logDir\desktop.err.log" }
    Start-Sleep -Milliseconds 500
}
if (-not $nativeReady) { throw "The native Tauri process did not launch within 60 seconds." }

Write-Host ""
Write-Host "Stage 1 is ready."
Write-Host "Backend: http://127.0.0.1:8765"
Write-Host "AI: local qwen2.5-3b-instruct"
Write-Host "Logs: $logDir"
