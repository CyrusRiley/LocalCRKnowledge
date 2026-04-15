@echo off
setlocal

cd /d "%~dp0"

set "QWEN_BASE_URL=http://127.0.0.1:8080/v1"
set "QWEN_MODEL=qwen2.5"
set "LK_SQLITE_JOURNAL_MODE=MEMORY"
set "LOCALKNOWLEDGE_PORT=8765"

set "MODEL_BAT=G:\LLM\start-Qwen2.5.bat"
if exist "%MODEL_BAT%" (
  echo [1/3] Starting local model server...
  start "Qwen2.5-Model" cmd /c call "%MODEL_BAT%"
) else (
  echo [1/3] Model startup script not found: %MODEL_BAT%
  echo       Only LocalKnowledge API will be started.
)

echo [2/3] Starting LocalKnowledge API...
timeout /t 2 /nobreak >nul

start "" "http://127.0.0.1:%LOCALKNOWLEDGE_PORT%"

echo [3/3] API running at http://127.0.0.1:%LOCALKNOWLEDGE_PORT%
python -m app.backend.api_server --host 127.0.0.1 --port %LOCALKNOWLEDGE_PORT%

endlocal
