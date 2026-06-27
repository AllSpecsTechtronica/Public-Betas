@echo off
REM Thin wrapper for the Base_Cv_program (PyQt5) venv installer.
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_packages_base.ps1" %*
