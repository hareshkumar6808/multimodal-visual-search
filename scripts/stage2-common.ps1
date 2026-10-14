function Get-Stage2Context {
    param([string]$Root)
    $workspace = Split-Path -Parent $Root
    $assetRoots = @(
        $Root,
        (Join-Path $workspace "native-validation-repo")
    ) | Where-Object { Test-Path -LiteralPath (Join-Path $_ ".tools\llama\llama-server.exe") }
    $assetRoot = $assetRoots | Select-Object -First 1
    if (-not $assetRoot) { $assetRoot = $Root }
    $pythonRoots = @($Root, $assetRoot) | Where-Object {
        Test-Path -LiteralPath (Join-Path $_ ".venv\Scripts\python.exe")
    }
    $pythonRoot = $pythonRoots | Select-Object -First 1
    [pscustomobject]@{
        Root = $Root
        AssetRoot = $assetRoot
        PythonRoot = $pythonRoot
        StateDir = Join-Path $Root ".tools\stage2"
        LogDir = Join-Path $Root ".tools\stage2\logs"
    }
}

function Find-Stage2Tesseract {
    param($Context)
    $command = Get-Command tesseract.exe -ErrorAction SilentlyContinue
    $candidates = @(
        $command.Source,
        (Join-Path $Context.AssetRoot ".tools\Tesseract-OCR\tesseract.exe"),
        "$env:ProgramFiles\Tesseract-OCR\tesseract.exe",
        "${env:ProgramFiles(x86)}\Tesseract-OCR\tesseract.exe"
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }
    if (-not $candidates) { throw "Tesseract OCR was not found." }
    (Resolve-Path -LiteralPath ($candidates | Select-Object -First 1)).Path
}

function Save-Stage2Process {
    param($Context, [string]$Name, [System.Diagnostics.Process]$Process)
    $Process.Refresh()
    @{ pid = $Process.Id; start_time = $Process.StartTime.ToFileTimeUtc() } |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Context.StateDir "$Name.json")
}

function Wait-Stage2Endpoint {
    param([string]$Name, [string]$Uri, [int]$Attempts = 120)
    foreach ($attempt in 1..$Attempts) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $Uri -TimeoutSec 3
            if ($response.StatusCode -eq 200) { Write-Host "[ready] $Name"; return }
        } catch {
            if ($attempt -eq $Attempts) { throw "$Name did not become ready: $($_.Exception.Message)" }
        }
        Start-Sleep -Milliseconds 500
    }
}

function Stop-StaleStage2Backend {
    $listeners = @(Get-NetTCPConnection -LocalPort 8768 -State Listen -ErrorAction SilentlyContinue)
    if (-not $listeners) { return }

    try {
        $spec = Invoke-RestMethod "http://127.0.0.1:8768/openapi.json" -TimeoutSec 3
    } catch {
        throw "Port 8768 is occupied by an unknown service (PID $($listeners[0].OwningProcess))."
    }
    if ($spec.info.title -ne "Multimodal Visual Search Orchestrator") {
        throw "Port 8768 is occupied by another service (PID $($listeners[0].OwningProcess))."
    }

    $pids = $listeners | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($processId in $pids) {
        & taskkill.exe /PID $processId /T /F 2>$null | Out-Null
    }
    foreach ($attempt in 1..20) {
        if (-not (Get-NetTCPConnection -LocalPort 8768 -State Listen -ErrorAction SilentlyContinue)) { return }
        Start-Sleep -Milliseconds 250
    }
    throw "The older Visual Search backend on port 8768 could not be stopped."
}

function Wait-Stage2Backend {
    param([int]$Attempts = 120)
    foreach ($attempt in 1..$Attempts) {
        try {
            $health = Invoke-RestMethod "http://127.0.0.1:8768/api/health" -TimeoutSec 3
            $spec = Invoke-RestMethod "http://127.0.0.1:8768/openapi.json" -TimeoutSec 3
            if ($health.status -and $spec.paths.'/api/analyze/stream') {
                Write-Host "[ready] Stage 2 backend"
                return
            }
        } catch {
            if ($attempt -eq $Attempts) { throw "Stage 2 backend did not become ready: $($_.Exception.Message)" }
        }
        if ($attempt -eq $Attempts) { throw "Backend is running but does not provide the Stage 2 streaming API." }
        Start-Sleep -Milliseconds 500
    }
}

function Set-Stage2RustEnvironment {
    param($Context)
    $tools = Join-Path $Context.AssetRoot ".tools"
    $rustBin = Join-Path $tools "msys2\msys64\mingw64\bin"
    $cargoHome = Join-Path $tools "cargo"
    $rustupHome = Join-Path $tools "rustup"
    if ((Test-Path (Join-Path $rustBin "cargo.exe")) -and (Test-Path $cargoHome)) {
        $webViewLoader = Get-ChildItem (Join-Path $cargoHome "registry\src") -Filter "WebView2Loader.dll" -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match "[\\/]x64[\\/]WebView2Loader\.dll$" } |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($webViewLoader) { $env:PATH = "$($webViewLoader.DirectoryName);$rustBin;$env:PATH" }
        $env:CARGO_HOME = $cargoHome
        $env:RUSTUP_HOME = $rustupHome
        $env:RUSTUP_TOOLCHAIN = "stable-x86_64-pc-windows-gnu"
        $env:CARGO_SOURCE_LOCAL_SPARSE_REGISTRY = "sparse+https://index.crates.io/"
        $env:CARGO_TARGET_DIR = Join-Path $Context.StateDir "cargo-target"
        $env:RUSTFLAGS = "-C link-arg=-Wl,--exclude-all-symbols"
        return (Join-Path $rustBin "cargo.exe")
    }
    $cargo = Get-Command cargo.exe -ErrorAction SilentlyContinue
    if (-not $cargo) { throw "Rust/Cargo was not found." }
    $cargo.Source
}

function Set-Stage2NodeEnvironment {
    param($Context)
    $nodeTemp = Join-Path $Context.StateDir "node-temp"
    $npmCache = Join-Path $Context.StateDir "npm-cache"
    $shim = Join-Path $Context.StateDir "realpath-shim.cjs"
    New-Item -ItemType Directory -Force $nodeTemp, $npmCache | Out-Null
    @"
const fs = require('node:fs');
fs.realpathSync.native = fs.realpathSync;
fs.realpath.native = fs.realpath;
"@ | Set-Content -LiteralPath $shim -Encoding utf8
    $env:TEMP = $nodeTemp
    $env:TMP = $nodeTemp
    $env:npm_config_cache = $npmCache
    $env:NODE_OPTIONS = "--require=$shim"
}
