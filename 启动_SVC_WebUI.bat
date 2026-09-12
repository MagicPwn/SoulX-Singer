@echo off
cd /d "%~dp0"

set "ROOT=%~dp0"
set "PYTHON_HOME=%ROOT%_internal\python"

set "PATH=%PYTHON_HOME%;%PYTHON_HOME%\Library\mingw-w64\bin;%PYTHON_HOME%\Library\usr\bin;%PYTHON_HOME%\Library\bin;%PYTHON_HOME%\Scripts;%ROOT%_internal\ffmpeg\dist\bin;%PATH%"
set "CONDA_PREFIX=%PYTHON_HOME%"

set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_DATASETS_OFFLINE=1
set GRADIO_ANALYTICS_ENABLED=False
set HF_HUB_DISABLE_TELEMETRY=1
set PYTHONPATH=%ROOT%;%PYTHONPATH%
set HTTP_PROXY=
set HTTPS_PROXY=
set http_proxy=
set https_proxy=
set NO_PROXY=localhost,127.0.0.1,::1
set no_proxy=localhost,127.0.0.1,::1

python webui_svc.py --fp16

pause
