from __future__ import annotations

from typing import Any

from app.backend.models import SourceRecord


def build_note_markdown(payload: dict[str, Any], source: SourceRecord) -> str:
    return f"""# {_text(payload.get("title"))}

## 内容类型
{_text(payload.get("note_type"))}

## 核心主题
{_bullet_list(payload.get("themes"))}

## 摘要
{_text(payload.get("summary"))}

## 关键要点
{_numbered_list(payload.get("key_points"))}

## 可用场景
{_bullet_list(payload.get("usage_scenarios"))}

## 我的进一步想法
{_text(payload.get("user_insights"))}

## 关键词
{_bullet_list(payload.get("keywords"))}

## 来源摘录
{_text(payload.get("source_excerpt"))}

## 来源信息
- source_id: {source.source_id}
- source_type: {source.source_type}
- file_path: {source.file_path or ""}
- imported_at: {source.imported_at}
""".strip() + "\n"


def _text(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else "（空）"


def _as_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _bullet_list(value: Any) -> str:
    items = _as_list(value)
    if not items:
        return "- （空）"
    return "\n".join(f"- {item}" for item in items)


def _numbered_list(value: Any) -> str:
    items = _as_list(value)
    if not items:
        return "1. （空）"
    return "\n".join(f"{index}. {item}" for index, item in enumerate(items, start=1))

