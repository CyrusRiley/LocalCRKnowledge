from __future__ import annotations

from typing import Any

from app.backend.models import SourceRecord


def _as_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def build_note_markdown(payload: dict[str, Any], source: SourceRecord) -> str:
    parts = [f"# {_clean_text(payload.get('title'))}"]
    _append_text_section(parts, "内容类型", payload.get("note_type"))
    _append_list_section(parts, "核心主题", payload.get("themes"), ordered=False)
    _append_text_section(parts, "摘要", payload.get("summary"))
    _append_text_section(parts, "保真整理", payload.get("faithful_content"))
    _append_list_section(parts, "关键要点", payload.get("key_points"), ordered=True)
    _append_list_section(parts, "可用场景", payload.get("usage_scenarios"), ordered=False)
    _append_text_section(parts, "我的进一步想法", payload.get("user_insights"))
    _append_list_section(parts, "关键词", payload.get("keywords"), ordered=False)
    _append_text_section(parts, "来源摘录", payload.get("source_excerpt"))
    parts.extend(
        [
            "## 来源信息",
            f"- source_id: {source.source_id}",
            f"- source_type: {source.source_type}",
            f"- file_path: {source.file_path or ''}",
            f"- imported_at: {source.imported_at}",
        ]
    )
    return "\n\n".join(part for part in parts if part.strip()).strip() + "\n"


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _append_text_section(parts: list[str], title: str, value: Any) -> None:
    text = _clean_text(value)
    if text:
        parts.append(f"## {title}\n{text}")


def _append_list_section(parts: list[str], title: str, value: Any, *, ordered: bool) -> None:
    items = _as_list(value)
    if not items:
        return
    if ordered:
        body = "\n".join(f"{index}. {item}" for index, item in enumerate(items, start=1))
    else:
        body = "\n".join(f"- {item}" for item in items)
    parts.append(f"## {title}\n{body}")
