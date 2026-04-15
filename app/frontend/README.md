# Frontend Quick UI

当前目录包含一个可直接试用的本地前端页面，配合 `app.backend.api_server` 使用。

启动：

```powershell
python -m app.backend.api_server --host 127.0.0.1 --port 8765
```

浏览器访问：

```text
http://127.0.0.1:8765
```

页面已连通以下能力：

- 手动文本整理入库
- 上传 `.txt/.md` 批量导入
- 单文件路径导入
- 目录增量更新
- 知识库列表、详情查看与条目编辑保存
- 关键词检索与问题检索问答
- Markdown 导出
