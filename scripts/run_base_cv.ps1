# Launch Base_Cv_program/main.py (PyQt5 modular CV). Windows. Uses .venv-base only.
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [object[]]$Passthrough
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$BaseRoot = Join-Path $RepoRoot "Base_Cv_program"
$VenvPython = Join-Path $RepoRoot ".venv-base\Scripts\python.exe"
$MainPy = Join-Path $BaseRoot "main.py"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Error "Missing venv at $RepoRoot\.venv-base. Run scripts\install_packages_base.ps1 first."
}
if (-not (Test-Path -LiteralPath $MainPy)) {
    Write-Error "Missing $MainPy"
}

Set-Location -LiteralPath $BaseRoot
& $VenvPython $MainPy @Passthrough
exit $LASTEXITCODE
