@echo off
setlocal

set "LOCALKNOWLEDGE_PORT=8765"

echo Stopping LocalKnowledge API on port %LOCALKNOWLEDGE_PORT%...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":%LOCALKNOWLEDGE_PORT%" ^| findstr "LISTENING"') do (
  taskkill /F /PID %%p >nul 2>nul
  echo Stopped PID: %%p
)

echo Done.
endlocal
