@echo off
setlocal

echo 正在查找并关闭 8765 端口上的 LocalKnowledge 服务...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8765" ^| findstr "LISTENING"') do (
  taskkill /F /PID %%p >nul 2>nul
  echo 已关闭 PID: %%p
)

echo 完成。
endlocal
