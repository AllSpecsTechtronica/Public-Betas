# Launch CV Ops (Qt + local API) from the Insight package root. Windows.
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [object[]]$Passthrough
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$InsightRoot = Join-Path $RepoRoot "Insight"
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Error "Missing venv at $RepoRoot\.venv. Run scripts\install_packages.ps1 first."
}
if (-not (Test-Path -LiteralPath (Join-Path $InsightRoot "insight_local"))) {
    Write-Error "Expected Insight tree at $InsightRoot"
}

$prevPyPath = [Environment]::GetEnvironmentVariable("PYTHONPATH", "Process")
if ([string]::IsNullOrWhiteSpace($prevPyPath)) {
    $env:PYTHONPATH = $RepoRoot
} else {
    $env:PYTHONPATH = "${RepoRoot};${prevPyPath}"
}
Set-Location -LiteralPath $InsightRoot
& $VenvPython -m insight_local.cvops @Passthrough
exit $LASTEXITCODE
