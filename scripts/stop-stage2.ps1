[CmdletBinding()]
param([switch]$Quiet)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
. (Join-Path $PSScriptRoot "stage2-common.ps1")
$ctx = Get-Stage2Context $root
if (-not (Test-Path $ctx.StateDir)) { if (-not $Quiet) { Write-Host "No Stage 2 state found." }; return }

foreach ($name in @("desktop", "native-desktop", "backend", "local-provider")) {
    $statePath = Join-Path $ctx.StateDir "$name.json"
    if (-not (Test-Path $statePath)) { continue }
    try {
        $state = Get-Content -Raw $statePath | ConvertFrom-Json
        $process = Get-Process -Id ([int]$state.pid) -ErrorAction SilentlyContinue
        if ($process -and $process.StartTime.ToFileTimeUtc() -eq [long]$state.start_time) {
            & taskkill.exe /PID $process.Id /T /F 2>$null | Out-Null
            if (-not $Quiet) { Write-Host "Stopped $name (PID $($process.Id))." }
        }
    } finally {
        Remove-Item -LiteralPath $statePath -Force -ErrorAction SilentlyContinue
    }
}
