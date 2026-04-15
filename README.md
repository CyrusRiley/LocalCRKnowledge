# LocalKnowledge

离线优先的本地科研笔记整理与检索工具。当前版本已实现 Python 后端闭环，并提供可直接试用的本地 Web 前端：导入零散文本，调用本地 Qwen2.5/llama.cpp OpenAI-compatible 接口整理为结构化 Markdown，写入 SQLite + FTS5，并支持增量导入、关键词检索、问题检索与 Markdown 导出。

## 当前状态

- 已实现：后端模块、SQLite schema、FTS5 检索、增量目录扫描、Markdown 导出、基础测试、可直接试用的前端交互页。
- 已预留：PDF 文本提取模块，安装 `pypdf` 后可尝试解析简单文本型 PDF。
- 暂未实现：Tauri 桌面壳。当前前端为本地 Web UI，优先验证业务流程与数据链路。

## 前端试用

启动本地 API + 前端服务：

```powershell
python -m app.backend.api_server --host 127.0.0.1 --port 8765
```

浏览器访问：

```text
http://127.0.0.1:8765
```

当前页面支持：

- 文本导入整理
- 上传 txt/md 批量导入
- 本地文件路径导入
- 目录增量更新
- 知识库列表/详情
- 关键词检索
- 问题检索并生成结构化回答
- Markdown 导出
- 长文本自动分段生成多条知识卡片（同一 source 关联多 note）

## 一键启动（推荐）

仓库根目录新增了两个脚本：

- `start-localknowledge.bat`：一键启动本地模型（若存在 `G:\LLM\start-Qwen2.5.bat`）并启动前端/API
- `stop-localknowledge.bat`：一键停止 8765 端口上的 LocalKnowledge 服务

直接双击或在终端执行：

```powershell
.\start-localknowledge.bat
```

## 本地模型配置

默认按 `llama serve` 的 OpenAI-compatible 接口调用：

```powershell
$env:QWEN_BASE_URL="http://127.0.0.1:8080/v1"
$env:QWEN_MODEL="qwen2.5"
```

如果你的 `llama serve` 只加载了一个模型，`model` 字段通常不会严格校验；如果服务端要求精确模型名，把 `QWEN_MODEL` 改成服务端实际名称即可。

其他常用环境变量：

```powershell
$env:LK_DB_PATH="data/db/knowledge.sqlite3"
$env:LK_SQLITE_JOURNAL_MODE="MEMORY"
$env:LK_DISABLE_LLM="1"
```

`LK_DISABLE_LLM=1` 用于本地验证主链路，会生成“待模型整理”的兜底条目，不代表真实整理效果。

## 常用命令

初始化数据库：

```powershell
python -m app.backend.main init-db
```

导入手动文本并导出 Markdown：

```powershell
python -m app.backend.main import-text --text "科研灵感：ABM 可用于分析慢跑路径偏好。" --direction "提炼为科研灵感条目" --export
```

导入单个 `.txt/.md` 文件：

```powershell
python -m app.backend.main import-file .\data\imports\note.md --direction "提炼为可用于写作的知识摘要"
```

增量扫描目录：

```powershell
python -m app.backend.main import-dir .\data\imports --direction "提炼为主题归纳条目"
```

关键词检索：

```powershell
python -m app.backend.main search ABM --limit 5
```

问题检索并生成结构化回答：

```powershell
python -m app.backend.main ask "我之前关于慢跑路径偏好的想法有哪些？" --limit 8 --export
```

本地无模型时验证：

```powershell
python -m app.backend.main --disable-llm import-text --text "科研灵感：ABM 可用于分析慢跑路径偏好。" --export
python -m app.backend.main --disable-llm search ABM
```

## 测试

```powershell
python -m unittest discover -s tests
```

## 目录说明

```text
app/backend/                 Python 核心逻辑
app/backend/api_server.py    前端 API 服务与静态页面托管
app/backend/database/        SQLite schema 与 repository
app/backend/importer/        文本/PDF 导入与目录扫描
app/backend/preprocess/      文本清洗、关键词预提取、分块
app/backend/llm/             Qwen 调用、prompt、JSON 解析
app/backend/formatter/       Markdown 模板生成
app/backend/retrieval/       搜索与回答生成
app/backend/updater/         增量更新
app/backend/export/          Markdown 导出
data/                        本地数据库、导出、日志和导入目录
tests/                       基础测试
app/frontend/                可直接试用的前端页面
```
