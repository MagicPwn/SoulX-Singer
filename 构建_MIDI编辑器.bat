@echo off
cd /d "%~dp0"

set "ROOT=%~dp0"
set "PATH=%ROOT%_internal\nodejs;%ROOT%_internal\python;%ROOT%_internal\python\Scripts;%PATH%"

echo Installing MIDI Editor dependencies...
cd /d "%ROOT%_internal\midi-editor"
call npm install
echo.
echo Building MIDI Editor...
call npm run build
echo.
echo Done! The editor is ready.
pause
