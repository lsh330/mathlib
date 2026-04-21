@echo off
REM =====================================================================
REM mathlib build artifact cleaner
REM  Removes build/, dist/, *.pyd, Cython-generated .cpp/.c/.html/.obj
REM  (hand-written sources under laplace/cpp/ are preserved)
REM =====================================================================
setlocal enableextensions

cd /d "%~dp0.."

where python >nul 2>&1
if errorlevel 1 (
    echo [clean.cmd] ERROR: python.exe not in PATH
    exit /b 1
)

python -c "import pathlib, shutil, glob; root=pathlib.Path('.'); [shutil.rmtree(p, ignore_errors=True) for p in ['build','dist'] if (root/p).exists()]; [pathlib.Path(f).unlink() for pat in ('src/math_library/**/*.pyd','src/math_library/**/*.c','src/math_library/**/*.cpp','src/math_library/**/*.html','src/math_library/**/*.obj') for f in glob.glob(pat, recursive=True) if 'laplace/cpp/' not in f.replace(chr(92),'/')]; print('[clean.cmd] artifacts removed')"
exit /b %ERRORLEVEL%
