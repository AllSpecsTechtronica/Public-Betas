# Create/update .venv-base and install Base_Cv_program (PyQt5) dependencies. Windows.
param(
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"

function Resolve-BaseProgramPython {
    param([string]$Explicit)
    if (-not [string]::IsNullOrWhiteSpace($Explicit)) {
        return $Explicit.Trim()
    }
    $fromEnv = [Environment]::GetEnvironmentVariable("CVLAYER_BASE_PYTHON", "Process")
    if (-not [string]::IsNullOrWhiteSpace($fromEnv)) {
        return $fromEnv.Trim()
    }
    foreach ($name in @("python", "python3")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($null -ne $cmd) {
            return $cmd.Source
        }
    }
    $py = Get-Command "py" -ErrorAction SilentlyContinue
    if ($null -ne $py) {
        try {
            $exe = & py -3 -c "import sys; print(sys.executable)"
            if (-not [string]::IsNullOrWhiteSpace($exe)) {
                return $exe.Trim()
            }
        } catch {
            return $null
        }
    }
    return $null
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvDir = Join-Path $RepoRoot ".venv-base"
$ReqBase = Join-Path $RepoRoot "requirements-cvlayer-base.txt"

if (-not (Test-Path -LiteralPath $ReqBase)) {
    Write-Error "Missing requirements file: $ReqBase"
}

$pythonExe = Resolve-BaseProgramPython -Explicit $Python
if ([string]::IsNullOrWhiteSpace($pythonExe)) {
    Write-Error "Could not resolve Python. Install Python 3.10+, set CVLAYER_BASE_PYTHON, or pass -Python."
}

Write-Host "[cvLayer-base] Using Python: $pythonExe"
Write-Host "[cvLayer-base] Venv: $VenvDir"

if (-not (Test-Path -LiteralPath $VenvDir)) {
    & $pythonExe -m venv $VenvDir
}

$venvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Error "Venv python not found at $venvPython"
}

& $venvPython -m pip install --upgrade pip setuptools wheel
& $venvPython -m pip install -r $ReqBase

Write-Host "[cvLayer-base] Done. Use scripts\run_base_cv*.ps1 or activate: $VenvDir\Scripts\Activate.ps1"
