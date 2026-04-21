@echo off
REM =====================================================================
REM mathlib MSVC build wrapper for VS 2026 / VS Code / CLI
REM   1) vswhere.exe locates VS install (auto 2019/2022/2026)
REM   2) vcvars64.bat activated (skip if VSCMD_VER already set)
REM   3) DISTUTILS_USE_SDK=1 skips setuptools re-invocation of vcvarsall
REM   4) python setup.py build_ext --inplace --compiler=msvc -j 8
REM =====================================================================
setlocal enableextensions

set "VS_INSTALLER=C:\Program Files (x86)\Microsoft Visual Studio\Installer"

if not exist "%VS_INSTALLER%\vswhere.exe" (
    echo [build.cmd] ERROR: vswhere.exe not found at "%VS_INSTALLER%"
    exit /b 1
)

set "PATH=%VS_INSTALLER%;%PATH%"

set "VS_INSTALL="
for /f "usebackq delims=" %%i in (`"%VS_INSTALLER%\vswhere.exe" -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS_INSTALL=%%i"

if not defined VS_INSTALL (
    echo [build.cmd] ERROR: No Visual Studio with VC Tools x86.x64 found
    exit /b 1
)

set "VCVARS=%VS_INSTALL%\VC\Auxiliary\Build\vcvars64.bat"

if not exist "%VCVARS%" (
    echo [build.cmd] ERROR: vcvars64.bat not found at "%VCVARS%"
    exit /b 1
)

if not defined VSCMD_VER (
    call "%VCVARS%" >nul
    if errorlevel 1 (
        echo [build.cmd] ERROR: vcvars64.bat failed
        exit /b 1
    )
)

set "DISTUTILS_USE_SDK=1"
set "PYTHONIOENCODING=utf-8"

cd /d "%~dp0.."

where python >nul 2>&1
if errorlevel 1 (
    echo [build.cmd] ERROR: python.exe not in PATH
    exit /b 1
)

echo [build.cmd] VS install: %VS_INSTALL%
python -c "import sys; print('[build.cmd] Python:', sys.executable, sys.version.split()[0])"

python setup.py build_ext --inplace -j 8 --compiler=msvc %*
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo [build.cmd] BUILD FAILED exit %RC%
) else (
    echo [build.cmd] BUILD OK
)
exit /b %RC%
