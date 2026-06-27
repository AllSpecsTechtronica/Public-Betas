@echo off
REM Thin wrapper: forwards to PowerShell installer.
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_packages.ps1" %*
