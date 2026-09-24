[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

& (Join-Path $PSScriptRoot "stop-stage1.ps1") -Quiet

$ports = @(8765, 11434, 1420)
$listeners = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
    Where-Object { $_.LocalPort -in $ports }
$ownerIds = @($listeners | Select-Object -ExpandProperty OwningProcess -Unique)

foreach ($ownerId in $ownerIds) {
    Write-Host "Stopping stale Stage 1 listener (PID $ownerId)..."
    Stop-Process -Id $ownerId -Force -ErrorAction Stop
}

Get-Process -Name "multimodal-visual-search-desktop" -ErrorAction SilentlyContinue |
    Stop-Process -Force -ErrorAction SilentlyContinue

foreach ($attempt in 1..20) {
    $remaining = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
        Where-Object { $_.LocalPort -in $ports }
    if (-not $remaining) { break }
    if ($attempt -eq 20) {
        $busyPorts = ($remaining.LocalPort | Sort-Object -Unique) -join ", "
        throw "Stage 1 ports are still occupied: $busyPorts"
    }
    Start-Sleep -Milliseconds 250
}

Set-Location -LiteralPath $root
& (Join-Path $PSScriptRoot "run-stage1.ps1")
