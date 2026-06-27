@echo off
setlocal
set "REPO=%~dp0.."
set "VENV_PY=%REPO%\.venv\Scripts\python.exe"
set "INSIGHT=%REPO%\Insight"
if not exist "%VENV_PY%" (
  echo Missing venv. Run scripts\install_packages.ps1 first.
  exit /b 1
)
set "PYTHONPATH=%REPO%"
cd /d "%INSIGHT%"
"%VENV_PY%" -m insight_local %*
