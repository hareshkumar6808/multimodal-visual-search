[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$toolsDir = Join-Path $root ".tools"
$llamaDir = Join-Path $toolsDir "llama"
$modelDir = Join-Path $toolsDir "ollama"
$downloadDir = Join-Path $toolsDir "downloads"
$modelPath = Join-Path $modelDir "qwen2.5-3b-instruct-q4_k_m.gguf"
$modelHash = "626B4A6678B86442240E33DF819E00132D3BA7DDDFE1CDC4FBB18E0A9615C62D"
$llamaUrl = "https://github.com/ggml-org/llama.cpp/releases/download/b11063/llama-b11063-bin-win-cpu-x64.zip"
$modelUrl = "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf?download=true"

function Require-Command {
    param([string]$Name, [string]$InstallMessage)
    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $command) { throw "$Name was not found. $InstallMessage" }
    return $command.Source
}

function Download-File {
    param([string]$Uri, [string]$Destination)
    $partial = "$Destination.partial"
    Write-Host "Downloading $(Split-Path -Leaf $Destination)..."
    & curl.exe --fail --location --retry 3 --output $partial $Uri
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $Uri" }
    Move-Item -LiteralPath $partial -Destination $Destination -Force
}

if (-not [Environment]::Is64BitOperatingSystem) {
    throw "Stage 1 requires 64-bit Windows."
}

$pythonCommand = Get-Command py.exe -ErrorAction SilentlyContinue
if ($pythonCommand) {
    $pythonExecutable = $pythonCommand.Source
    $pythonPrefix = @("-3")
} else {
    $pythonExecutable = Require-Command "python.exe" "Install Python 3.11 or newer."
    $pythonPrefix = @()
}

Require-Command "node.exe" "Install Node.js 20 or newer." | Out-Null
Require-Command "npm.cmd" "Install Node.js 20 or newer." | Out-Null
Require-Command "cargo.exe" "Install Rust stable with rustup and the Visual C++ build tools." | Out-Null
Require-Command "curl.exe" "Use a supported Windows 10 or 11 installation." | Out-Null
$tesseractCommand = Get-Command tesseract.exe -ErrorAction SilentlyContinue
$tesseractCandidates = @(
    $tesseractCommand.Source,
    "$env:ProgramFiles\Tesseract-OCR\tesseract.exe",
    "${env:ProgramFiles(x86)}\Tesseract-OCR\tesseract.exe"
) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }
if (-not $tesseractCandidates) {
    throw "Tesseract OCR was not found. Install it on PATH or in the default Program Files location."
}

$driveName = [System.IO.Path]::GetPathRoot($root).Substring(0, 1)
$freeBytes = (Get-PSDrive -Name $driveName).Free
if ($freeBytes -lt 5GB) {
    throw "At least 5 GB of free disk space is required for setup and the local model."
}

New-Item -ItemType Directory -Force -Path $toolsDir, $llamaDir, $modelDir, $downloadDir | Out-Null

$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "Creating Python virtual environment..."
    & $pythonExecutable @pythonPrefix -m venv (Join-Path $root ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Python virtual environment creation failed." }
}

Write-Host "Installing backend dependencies..."
& $venvPython -m pip install -e "${root}[dev]"
if ($LASTEXITCODE -ne 0) { throw "Backend dependency installation failed." }

Write-Host "Installing desktop dependencies..."
& npm.cmd ci --prefix (Join-Path $root "apps\desktop")
if ($LASTEXITCODE -ne 0) { throw "Desktop dependency installation failed." }

$llamaServer = Join-Path $llamaDir "llama-server.exe"
if (-not (Test-Path -LiteralPath $llamaServer)) {
    $llamaArchive = Join-Path $downloadDir "llama-b11063-bin-win-cpu-x64.zip"
    if (-not (Test-Path -LiteralPath $llamaArchive)) {
        Download-File $llamaUrl $llamaArchive
    }
    Write-Host "Extracting llama.cpp..."
    $unpackDir = Join-Path $toolsDir "llama-unpack"
    $resolvedTools = [System.IO.Path]::GetFullPath($toolsDir).TrimEnd('\')
    $resolvedUnpack = [System.IO.Path]::GetFullPath($unpackDir)
    if (-not $resolvedUnpack.StartsWith("$resolvedTools\", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to replace an unpack directory outside .tools."
    }
    if (Test-Path -LiteralPath $resolvedUnpack) {
        Remove-Item -LiteralPath $resolvedUnpack -Recurse -Force
    }
    Expand-Archive -LiteralPath $llamaArchive -DestinationPath $resolvedUnpack -Force
    $extractedServer = Get-ChildItem -LiteralPath $resolvedUnpack -Recurse -File -Filter "llama-server.exe" |
        Select-Object -First 1
    if (-not $extractedServer) { throw "The llama.cpp archive did not contain llama-server.exe." }
    Copy-Item -Path (Join-Path $extractedServer.DirectoryName "*") -Destination $llamaDir -Recurse -Force
    Remove-Item -LiteralPath $resolvedUnpack -Recurse -Force
}

$downloadModel = $true
if (Test-Path -LiteralPath $modelPath) {
    $downloadModel = (Get-FileHash -Algorithm SHA256 -LiteralPath $modelPath).Hash -ne $modelHash
}
if ($downloadModel) {
    Write-Host "The local Qwen model is about 2.1 GB."
    Download-File $modelUrl $modelPath
}
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $modelPath).Hash -ne $modelHash) {
    throw "The downloaded model checksum is invalid. Delete $modelPath and rerun setup."
}

Write-Host ""
Write-Host "Stage 1 setup is complete."
Write-Host "Run: .\scripts\run-stage1.ps1"
