@echo off
REM =====================================================================
REM mathlib MSVC build wrapper for VS 2026 / VS Code / CLI
REM   1) vswhere.exe locates VS install (auto 2019/2022/2026)
REM   2) Force x64 toolchain (override x86 Developer Command Prompt too)
REM   3) Verify Python is 64-bit (avoid pyatomic_msc.h sizeof mismatch)
REM   4) DISTUTILS_USE_SDK=1 skips setuptools re-invocation of vcvarsall
REM   5) python setup.py build_ext --inplace --compiler=msvc -j 8
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

REM Force x64 toolchain. If invoked from a 32-bit Developer Command Prompt
REM (VSCMD_ARG_TGT_ARCH=x86) the prior environment MUST be overridden —
REM otherwise cl.exe/link.exe resolve to HostX86\x86 and fail on
REM pyatomic_msc.h (sizeof(void*)==8 assertions) when building a
REM 64-bit Python extension.
set "NEED_VCVARS64=1"
if defined VSCMD_VER (
    if /I "%VSCMD_ARG_TGT_ARCH%"=="x64" if /I "%VSCMD_ARG_HOST_ARCH%"=="x64" set "NEED_VCVARS64=0"
)

if "%NEED_VCVARS64%"=="1" (
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

REM Verify Python is 64-bit. A 32-bit Python produces build/temp.win32-*
REM and cannot link against 64-bit MSVC libraries.
python -c "import struct,sys; sys.exit(0 if struct.calcsize('P')*8==64 else 1)"
if errorlevel 1 (
    echo [build.cmd] ERROR: Python is not 64-bit. Install/use a 64-bit Python 3.x.
    python -c "import sys,struct; print('  current:', sys.executable, 'bits=', struct.calcsize('P')*8)"
    exit /b 1
)

echo [build.cmd] VS install : %VS_INSTALL%
echo [build.cmd] Target arch: %VSCMD_ARG_TGT_ARCH%  Host arch: %VSCMD_ARG_HOST_ARCH%
python -c "import sys,struct; print('[build.cmd] Python    :', sys.executable, sys.version.split()[0], str(struct.calcsize('P')*8)+'-bit')"

python setup.py build_ext --inplace -j 8 --compiler=msvc %*
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo [build.cmd] BUILD FAILED exit %RC%
) else (
    echo [build.cmd] BUILD OK
)
exit /b %RC%
