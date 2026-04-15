@echo off
setlocal

cd /d "%~dp0"

set "QWEN_BASE_URL=http://127.0.0.1:8080/v1"
set "QWEN_MODEL=qwen2.5"
set "LK_SQLITE_JOURNAL_MODE=MEMORY"

set "MODEL_BAT=G:\LLM\start-Qwen2.5.bat"
if exist "%MODEL_BAT%" (
  echo [1/3] 正在启动本地模型服务...
  start "Qwen2.5-Model" "%MODEL_BAT%"
) else (
  echo [1/3] 未找到模型启动脚本: %MODEL_BAT%
  echo       将只启动 LocalKnowledge 前端/API 服务。
)

echo [2/3] 准备启动 LocalKnowledge API...
timeout /t 2 /nobreak >nul

start "" "http://127.0.0.1:8765"

echo [3/3] API 服务运行中: http://127.0.0.1:8765
python -m app.backend.api_server --host 127.0.0.1 --port 8765

endlocal
