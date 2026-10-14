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
