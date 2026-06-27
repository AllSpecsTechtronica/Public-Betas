# Create/update the repo virtualenv and install cvLayer Python dependencies (Windows / PowerShell).
param(
    [switch]$WithRag,
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvDir = Join-Path $RepoRoot ".venv"
$ReqCore = Join-Path $RepoRoot "requirements-cvlayer.txt"
$ReqRag = Join-Path $RepoRoot "requirements-cvlayer-rag.txt"

if (-not (Test-Path -LiteralPath $ReqCore)) {
    Write-Error "Missing requirements file: $ReqCore"
}

function Resolve-CvLayerPython {
    param([string]$Explicit)
    if (-not [string]::IsNullOrWhiteSpace($Explicit)) {
        return $Explicit.Trim()
    }
    $fromEnv = [Environment]::GetEnvironmentVariable("CVLAYER_PYTHON", "Process")
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

$pythonExe = Resolve-CvLayerPython -Explicit $Python
if ([string]::IsNullOrWhiteSpace($pythonExe)) {
    Write-Error "Could not resolve Python. Install Python 3.10+, set CVLAYER_PYTHON, or pass -Python."
}

Write-Host "[cvLayer] Using Python: $pythonExe"
Write-Host "[cvLayer] Venv: $VenvDir"

if (-not (Test-Path -LiteralPath $VenvDir)) {
    & $pythonExe -m venv $VenvDir
}

$venvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Error "Venv python not found at $venvPython"
}

& $venvPython -m pip install --upgrade pip setuptools wheel
& $venvPython -m pip install -r $ReqCore

if ($WithRag) {
    if (-not (Test-Path -LiteralPath $ReqRag)) {
        Write-Error "Missing requirements file: $ReqRag"
    }
    & $venvPython -m pip install -r $ReqRag
}

Write-Host "[cvLayer] Done. Use run scripts in scripts\ or activate: $VenvDir\Scripts\Activate.ps1"
