# Launch Insight/main.py (FastAPI + aiortc WebRTC stack). Windows.
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [object[]]$Passthrough
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$InsightRoot = Join-Path $RepoRoot "Insight"
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$MainPy = Join-Path $InsightRoot "main.py"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Error "Missing venv at $RepoRoot\.venv. Run scripts\install_packages.ps1 first."
}
if (-not (Test-Path -LiteralPath $MainPy)) {
    Write-Error "Missing $MainPy"
}

$prevPyPath = [Environment]::GetEnvironmentVariable("PYTHONPATH", "Process")
if ([string]::IsNullOrWhiteSpace($prevPyPath)) {
    $env:PYTHONPATH = $RepoRoot
} else {
    $env:PYTHONPATH = "${RepoRoot};${prevPyPath}"
}
Set-Location -LiteralPath $InsightRoot
& $VenvPython $MainPy @Passthrough
exit $LASTEXITCODE
