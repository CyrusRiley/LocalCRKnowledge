# LocalKnowledge

离线优先的本地科研笔记整理与检索工具。当前版本以 Python 后端和本地 Web 前端为主：导入零散文本或文件，调用本地 Qwen2.5/llama.cpp OpenAI-compatible 接口整理为结构化 Markdown，写入 SQLite + FTS5，并支持增量导入、知识单元拆分、关键词归一、知识关系整理、知识网络展示、检索问答和 Markdown 导出。

## 当前状态

- 已实现：后端主链路、SQLite schema、FTS5 检索、增量目录扫描、Markdown 导出、基础测试、本地 Web 前端。
- 已实现：长文本拆分为多条知识单元，保留同一 source 追溯关系，并生成 `knowledge_units`、`knowledge_groups`、关键词库和知识关系。
- 已实现：知识库快速整理与全局整理，前端以知识网络展示主题、概念、知识组、知识条和强语义关系。
- 已实现：检索问答默认使用摘录式回答，优先保证答案来自本地知识库内容。
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

- 粘贴文本导入与整理
- 上传 `.txt/.md` 文件批量导入
- 按本地文件路径导入
- 目录增量更新
- 整理过程进度提示
- 整理结果渲染预览与 Markdown 导出
- 知识网络展示
- 快速整理知识库
- 全局整理知识库，执行前会二次确认
- 入库历史查看
- 关键词检索
- 问题检索与摘录式结构化回答
- 回答结果导出 Markdown

## 一键启动

仓库根目录提供两个脚本：

- `start-localknowledge.bat`：启动本地模型脚本（若存在 `G:\LLM\start-Qwen2.5.bat`）并启动 LocalKnowledge 前端/API。
- `stop-localknowledge.bat`：停止 8765 端口上的 LocalKnowledge 服务。

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

## 数据与隐私

真实知识库默认存放在：

```text
data/db/knowledge.sqlite3
```

这些本地运行数据已被 `.gitignore` 排除：

- `data/db/`
- `data/exports/`
- `data/imports/`
- `data/logs/`

仓库中可提交的演示数据放在 `examples/`：

- `examples/fake_knowledge.sqlite3`
- `examples/fake_notes.md`

如需更换真实数据库位置，可以设置：

```powershell
$env:LK_DB_PATH="D:\your\private\path\knowledge.sqlite3"
```

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

问题检索并导出回答：

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
app/backend/analysis/        长文本结构分析与知识单元切分
app/backend/database/        SQLite schema 与 repository
app/backend/importer/        文本/PDF 导入与目录扫描
app/backend/preprocess/      文本清洗、关键词预提取、基础分块
app/backend/llm/             Qwen 调用、prompt、JSON 解析
app/backend/formatter/       Markdown 模板生成
app/backend/indexing/        FTS 索引维护
app/backend/keywords/        关键词归一、别名与关键词库
app/backend/maintenance/     迁移与回填脚本
app/backend/organizer/       知识库快速/全局整理
app/backend/relations/       知识关系构建
app/backend/retrieval/       搜索、知识网络与回答生成
app/backend/updater/         增量更新
app/backend/export/          Markdown 导出
app/frontend/                可直接试用的前端页面
data/                        本地数据库、导出、日志和导入目录
examples/                    可提交的 fake 示例知识库
tests/                       基础测试
```
