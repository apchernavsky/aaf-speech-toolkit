@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ------------------------------------------------------------
REM build_pyinstaller.bat
REM Build a self-contained Windows distribution via PyInstaller.
REM
REM Notes:
REM - For TensorFlow, prefer onedir (default). Onefile is not recommended.
REM - This script intentionally does NOT upgrade pip.
REM - SDK tools are bundled from .\sdk_bin (portable) and optionally from AAF_TOOLS_DIR.
REM - ffmpeg is optional; if present in .\sdk_bin or set via FFMPEG_EXE, it is bundled into bin\.
REM ------------------------------------------------------------

cd /d "%~dp0"

set "PYTHON=python"
set "ENTRY=aaf_pipeline.py"
set "DISTROOT=dist_pyinstaller.stage-%RANDOM%-%RANDOM%"
set "BUILDDIR=build_pyinstaller"
set "NAME=aaf_pipeline"

REM Toolkit root (portable defaults).
set "TOOLKIT_ROOT=%~dp0"
if "%TOOLKIT_ROOT:~-1%"=="\" set "TOOLKIT_ROOT=%TOOLKIT_ROOT:~0,-1%"

if not defined AAF_TOOLS_DIR set "AAF_TOOLS_DIR=%TOOLKIT_ROOT%\sdk_bin"
if not defined FFMPEG_EXE if exist "%TOOLKIT_ROOT%\sdk_bin\ffmpeg.exe" set "FFMPEG_EXE=%TOOLKIT_ROOT%\sdk_bin\ffmpeg.exe"

set "PIP_CONSTRAINT=%TOOLKIT_ROOT%\requirements-build-lock.txt"
set "LOCK_DIR=%TOOLKIT_ROOT%\.pyinstaller_build.lock"
mkdir "%LOCK_DIR%" 2>nul
if errorlevel 1 (
  echo ERROR: Build lock exists or cannot be created: %LOCK_DIR%
  echo        Another build may be running, or a previous build crashed.
  exit /b 2
)

echo.
echo === PyInstaller build (onedir) ===
echo Toolkit dir: %CD%
echo Entry: %ENTRY%
echo Dist: %DISTROOT%
echo Build: %BUILDDIR%
echo.

if not exist "%ENTRY%" (
  echo ERROR: entry not found: %ENTRY%
  rmdir "%LOCK_DIR%" >nul 2>nul
  exit /b 2
)

REM Create venv dedicated to build tooling.
if not exist ".venv-build\Scripts\python.exe" (
  echo Creating build venv: .venv-build
  "%PYTHON%" -m venv ".venv-build"
  if errorlevel 1 (
    rmdir "%LOCK_DIR%" >nul 2>nul
    exit /b 2
  )
)

set "VENV_PY=%TOOLKIT_ROOT%\.venv-build\Scripts\python.exe"

echo Installing build dependencies (pyinstaller)...
"%VENV_PY%" -m pip install pyinstaller
if errorlevel 1 (
  rmdir "%LOCK_DIR%" >nul 2>nul
  exit /b 2
)

REM tensorflow-hub 0.16 imports pkg_resources; setuptools 82+ no longer provides it.
echo Installing build compatibility dependency (setuptools with pkg_resources)...
"%VENV_PY%" -m pip install "setuptools<81"
if errorlevel 1 (
  rmdir "%LOCK_DIR%" >nul 2>nul
  exit /b 2
)

echo Installing runtime dependencies into build venv...
"%VENV_PY%" -m pip install -r ".\aaf_speech_filter\requirements.txt"
if errorlevel 1 (
  rmdir "%LOCK_DIR%" >nul 2>nul
  exit /b 2
)

REM Local project packages are bundled from the current source tree via PYTHONPATH/--paths below.
REM Do not rely on previously installed wheels/editables with the same 0.1.0 version.
"%VENV_PY%" -m pip uninstall -y aaf-speech-filter aaf-io >nul 2>nul

set "PYTHONPATH=%TOOLKIT_ROOT%;%TOOLKIT_ROOT%\aaf_io;%TOOLKIT_ROOT%\aaf_speech_filter;%PYTHONPATH%"

echo Preflight imports (build venv)...
"%VENV_PY%" -c "import pkg_resources; import tensorflow_hub; import tf_keras; import aaf_io; import aaf_gui; import aaf_workflow; import aaf_speech_filter; import aaf_speech_filter.sdk_removals; import aaf_speech_filter.timeline_sourceclips; import aaf_speech_filter.timeline_timing; import aaf_speech_filter.duplicate_filter; import aaf_speech_filter.aaf_mutation; import aaf_speech_filter.aaf_yamnet_lane_layout; import aaf_speech_filter.speech_yamnet"
if errorlevel 1 (
  echo ERROR: Preflight import failed in build venv.
  rmdir "%LOCK_DIR%" >nul 2>nul
  exit /b 2
)

REM Never reuse or delete an existing distribution staging directory.
if exist "%DISTROOT%" (
  echo ERROR: Staging directory already exists: %DISTROOT%
  rmdir "%LOCK_DIR%" >nul 2>nul
  exit /b 2
)

REM Hidden imports for TF Lite wrappers seen to break in frozen builds.
set "HI=--hidden-import=tensorflow.lite.python._pywrap_tensorflow_lite_metrics_wrapper"
set "HI=%HI% --hidden-import=tensorflow.lite.python._pywrap_tensorflow_interpreter_wrapper"
set "HI=%HI% --hidden-import=tensorflow.lite.python._pywrap_tensorflow_lite_calibration_wrapper"
set "HI=%HI% --hidden-import=tensorflow.lite.python._pywrap_analyzer_wrapper"
set "HI=%HI% --hidden-import=tensorflow.lite.python._pywrap_modify_model_interface"
set "HI=%HI% --hidden-import=tensorflow.lite.python._pywrap_string_util"
set "HI=%HI% --hidden-import=aaf_gui"
set "HI=%HI% --hidden-import=aaf_workflow"
set "HI=%HI% --hidden-import=aaf_speech_filter.sdk_removals"
set "HI=%HI% --hidden-import=aaf_speech_filter.timeline_sourceclips"
set "HI=%HI% --hidden-import=aaf_speech_filter.timeline_timing"
set "HI=%HI% --hidden-import=aaf_speech_filter.duplicate_filter"
set "HI=%HI% --hidden-import=aaf_speech_filter.aaf_mutation"
set "HI=%HI% --hidden-import=aaf_speech_filter.aaf_yamnet_lane_layout"

REM TensorFlow Hub/YAMNet need package data and metadata in frozen builds.
set "COLLECT=--collect-all tensorflow --collect-all tensorflow_hub --collect-all tf_keras --collect-all keras"

echo.
echo Building with PyInstaller...
"%VENV_PY%" -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --onedir ^
  --name "%NAME%" ^
  --distpath "%DISTROOT%" ^
  --workpath "%BUILDDIR%" ^
  --paths "%TOOLKIT_ROOT%" ^
  --paths "%TOOLKIT_ROOT%\aaf_io" ^
  --paths "%TOOLKIT_ROOT%\aaf_speech_filter" ^
  --noupx ^
  --windowed ^
  --add-data "aaf_tool_config.json;." ^
  %HI% ^
  %COLLECT% ^
  "%ENTRY%"

if errorlevel 1 (
  echo.
  echo ERROR: PyInstaller build failed.
  rmdir "%LOCK_DIR%" >nul 2>nul
  exit /b 1
)

set "APPDIST=%DISTROOT%\%NAME%"
if not exist "%APPDIST%" (
  echo ERROR: expected output folder not found: %APPDIST%
  rmdir "%LOCK_DIR%" >nul 2>nul
  exit /b 1
)

REM Copy failures, incomplete SDK/runtime payloads, and bounded smoke failures
REM must stop here; publish_distribution never sees a failed candidate.
"%VENV_PY%" scripts\stage_distribution_payload.py "%APPDIST%" --smoke
if errorlevel 1 (
  echo ERROR: Staged payload or processing smoke check failed.
  rmdir "%LOCK_DIR%" >nul 2>nul
  exit /b 1
)
"%VENV_PY%" scripts\publish_distribution.py "%DISTROOT%"
if errorlevel 1 (
  rmdir "%LOCK_DIR%" >nul 2>nul
  exit /b 1
)
set "APPDIST=dist_pyinstaller\%NAME%"
echo DONE.
echo - Output folder: %APPDIST%
echo - Run: %APPDIST%\%NAME%.exe  (GUI mode)
echo.
rmdir "%LOCK_DIR%" >nul 2>nul
exit /b 0
