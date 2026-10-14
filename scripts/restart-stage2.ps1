[CmdletBinding()]
param()
$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "stop-stage2.ps1") -Quiet
& (Join-Path $PSScriptRoot "run-stage2.ps1")
