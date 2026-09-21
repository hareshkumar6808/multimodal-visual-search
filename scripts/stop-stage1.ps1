[CmdletBinding()]
param([switch]$Quiet)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$stateDir = Join-Path $root ".tools\stage1"
if (-not (Test-Path -LiteralPath $stateDir)) {
    if (-not $Quiet) { Write-Host "No Stage 1 process state was found." }
    return
}

foreach ($name in @("desktop", "native-desktop", "backend", "local-provider")) {
    $statePath = Join-Path $stateDir "$name.json"
    if (-not (Test-Path -LiteralPath $statePath)) { continue }
    $removeState = $false
    try {
        $state = Get-Content -Raw -LiteralPath $statePath | ConvertFrom-Json
        $process = Get-Process -Id ([int]$state.pid) -ErrorAction SilentlyContinue
        if ($process -and $process.StartTime.ToFileTimeUtc() -eq [long]$state.start_time) {
            & taskkill.exe /PID $process.Id /T /F 2>$null | Out-Null
            if ($LASTEXITCODE -ne 0) { throw "taskkill failed with exit code $LASTEXITCODE" }
            if (-not $Quiet) { Write-Host "Stopped $name (PID $($process.Id))." }
        }
        $removeState = $true
    } catch {
        if (-not $Quiet) { Write-Warning "Could not stop $name safely: $($_.Exception.Message)" }
    } finally {
        if ($removeState) {
            Remove-Item -LiteralPath $statePath -Force -ErrorAction SilentlyContinue
        }
    }
}
