@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_EXE=C:\Users\CyRi\.conda\envs\ABM_UrbanDesign\python.exe"
set "PYTHONNOUSERSITE=1"
set "QWEN_BASE_URL=http://127.0.0.1:8080/v1"
set "QWEN_MODEL=qwen2.5"
set "LK_SQLITE_JOURNAL_MODE=MEMORY"
set "LOCALKNOWLEDGE_PORT=8765"
set "QWEN_ANSWER_TIMEOUT_SECONDS=75"
set "QWEN_SYNTHESIS_TIMEOUT_SECONDS=75"
set "QWEN_SYNTHESIS_MAX_TOKENS=800"
set "LK_EMBEDDING_MODELS=zh=sentence_transformers,G:\LLM\embeddings\bge-base-zh-v1.5,0.55;multi=sentence_transformers,G:\LLM\embeddings\bge-m3,0.45;local=local,local-hash-ngram-384,0.25"
set "LK_EMBEDDING_ACTIVE=zh,multi,local"

if not exist "%PYTHON_EXE%" (
  echo Python interpreter not found:
  echo   %PYTHON_EXE%
  echo Please update PYTHON_EXE in start-localknowledge.bat.
  pause
  exit /b 1
)

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
"%PYTHON_EXE%" -m app.backend.api_server --host 127.0.0.1 --port %LOCALKNOWLEDGE_PORT%

endlocal
