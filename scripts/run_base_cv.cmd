@echo off
setlocal
set "REPO=%~dp0.."
set "VENV_PY=%REPO%\.venv-base\Scripts\python.exe"
set "BASE=%REPO%\Base_Cv_program"
if not exist "%VENV_PY%" (
  echo Missing .venv-base. Run scripts\install_packages_base.ps1 first.
  exit /b 1
)
cd /d "%BASE%"
"%VENV_PY%" "%BASE%\main.py" %*
