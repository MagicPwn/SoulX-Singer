@echo off
cd /d "%~dp0"

set "ROOT=%~dp0"
set "PYTHON_HOME=%ROOT%_internal\python"

set "PATH=%PYTHON_HOME%;%PYTHON_HOME%\Scripts;%PATH%"
set "CONDA_PREFIX=%PYTHON_HOME%"
set PYTHONPATH=%ROOT%;%PYTHONPATH%
set HTTP_PROXY=
set HTTPS_PROXY=
set http_proxy=
set https_proxy=
set NO_PROXY=localhost,127.0.0.1,::1
set no_proxy=localhost,127.0.0.1,::1

echo SoulX-Singer MIDI Editor

python _internal\midi_editor_server.py

pause
