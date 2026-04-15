from __future__ import annotations

import json
import re
from typing import Any


REQUIRED_NOTE_FIELDS = {
    "title": "",
    "note_type": "",
    "themes": [],
    "summary": "",
    "key_points": [],
    "usage_scenarios": [],
    "user_insights": "",
    "keywords": [],
    "source_excerpt": "",
}


def parse_note_json(content: str, *, fallback_title: str, source_text: str) -> dict[str, Any]:
    try:
        data = json.loads(_extract_json(content))
        if not isinstance(data, dict):
            raise ValueError("LLM response JSON is not an object")
    except Exception:  # noqa: BLE001
        data = {}
    return normalize_note_payload(data, fallback_title=fallback_title, source_text=source_text)


def parse_question_json(content: str, question: str) -> dict[str, Any]:
    try:
        data = json.loads(_extract_json(content))
        if not isinstance(data, dict):
            raise ValueError("LLM response JSON is not an object")
    except Exception:  # noqa: BLE001
        data = {}
    return {
        "keywords": _string_list(data.get("keywords")) or _fallback_terms(question),
        "themes": _string_list(data.get("themes")),
        "note_type": str(data.get("note_type") or ""),
        "intent": str(data.get("intent") or "question"),
    }


def normalize_note_payload(data: dict[str, Any], *, fallback_title: str, source_text: str) -> dict[str, Any]:
    normalized = dict(REQUIRED_NOTE_FIELDS)
    normalized.update({key: value for key, value in data.items() if key in REQUIRED_NOTE_FIELDS})
    normalized["title"] = str(normalized["title"] or fallback_title or "未命名条目")
    normalized["note_type"] = str(normalized["note_type"] or "未分类")
    normalized["themes"] = _string_list(normalized["themes"])
    normalized["key_points"] = _string_list(normalized["key_points"])
    normalized["usage_scenarios"] = _string_list(normalized["usage_scenarios"])
    normalized["keywords"] = _string_list(normalized["keywords"])
    normalized["summary"] = str(normalized["summary"] or _summary_fallback(source_text))
    normalized["user_insights"] = str(normalized["user_insights"] or "")
    normalized["source_excerpt"] = str(normalized["source_excerpt"] or source_text[:300])
    return normalized


def _extract_json(content: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)
    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        return content[start : end + 1]
    raise ValueError("No JSON object found")


def _string_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.replace("，", ",").split(",") if part.strip()]
    return [str(value).strip()]


def _summary_fallback(source_text: str) -> str:
    first = source_text.strip().split("\n", 1)[0] if source_text.strip() else ""
    return first[:160]


def _fallback_terms(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}|[\u4e00-\u9fff]{2,}", text)[:8]

