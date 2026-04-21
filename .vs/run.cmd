@echo off
REM =====================================================================
REM mathlib python launcher for pytest / bench / arbitrary scripts
REM  Injects PYTHONPATH=src and forwards all args to python.exe
REM =====================================================================
setlocal enableextensions

cd /d "%~dp0.."

set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
set "PYTHONIOENCODING=utf-8"

where python >nul 2>&1
if errorlevel 1 (
    echo [run.cmd] ERROR: python.exe not in PATH
    exit /b 1
)

python %*
exit /b %ERRORLEVEL%
