from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SourceRecord:
    source_id: str
    source_type: str
    file_path: str | None
    raw_text: str
    clean_text: str
    text_hash: str
    created_at: str
    imported_at: str
    status: str = "new"


@dataclass
class StructuredNote:
    note_id: str
    source_id: str
    title: str
    note_type: str
    themes: list[str] = field(default_factory=list)
    summary: str = ""
    faithful_content: str = ""
    key_points: list[str] = field(default_factory=list)
    usage_scenarios: list[str] = field(default_factory=list)
    user_insights: str = ""
    keywords: list[str] = field(default_factory=list)
    source_excerpt: str = ""
    markdown_content: str = ""
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_llm_payload(
        cls,
        *,
        note_id: str,
        source_id: str,
        payload: dict[str, Any],
        markdown_content: str,
        created_at: str,
        updated_at: str,
    ) -> "StructuredNote":
        return cls(
            note_id=note_id,
            source_id=source_id,
            title=str(payload.get("title") or "未命名条目"),
            note_type=str(payload.get("note_type") or "未分类"),
            themes=_string_list(payload.get("themes")),
            summary=str(payload.get("summary") or ""),
            faithful_content=str(
                payload.get("faithful_content")
                or payload.get("full_content")
                or payload.get("content")
                or payload.get("summary")
                or ""
            ),
            key_points=_string_list(payload.get("key_points")),
            usage_scenarios=_string_list(payload.get("usage_scenarios")),
            user_insights=str(payload.get("user_insights") or ""),
            keywords=_string_list(payload.get("keywords")),
            source_excerpt=str(payload.get("source_excerpt") or ""),
            markdown_content=markdown_content,
            created_at=created_at,
            updated_at=updated_at,
        )


@dataclass
class KnowledgeGroup:
    group_id: str
    source_id: str
    group_title: str
    group_type: str = ""
    summary: str = ""
    created_at: str = ""
    updated_at: str = ""


@dataclass
class KnowledgeUnit:
    unit_id: str
    group_id: str | None
    source_id: str
    note_id: str | None
    title: str
    content: str
    evidence: str = ""
    note_type: str = ""
    order_index: int = 0
    confidence: float = 0.7
    attributes: dict[str, Any] = field(default_factory=dict)
    parent_unit_id: str | None = None
    created_at: str = ""
    updated_at: str = ""


@dataclass
class KeywordTerm:
    term_id: str
    canonical_name: str
    normalized_name: str
    description: str = ""
    status: str = "active"
    created_at: str = ""
    updated_at: str = ""


@dataclass
class ChunkRecord:
    chunk_id: str
    note_id: str
    chunk_text: str
    chunk_order: int
    chunk_type: str
    keywords: list[str] = field(default_factory=list)


@dataclass
class PreprocessResult:
    clean_text: str
    title_candidate: str
    pre_keywords: list[str]
    chunks: list[str]


@dataclass
class SearchResult:
    note_id: str
    source_id: str
    title: str
    note_type: str
    summary: str
    markdown_content: str
    source_excerpt: str
    themes: list[str]
    keywords: list[str]
    file_path: str | None
    imported_at: str | None
    score: float
    snippet: str
    relevance_score: float = 0.0
    relevance_level: str = ""
    relevance_reason: str = ""
    relation_type: str = ""
    relation_strength: str = ""
    match_source: str = ""


def _string_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.replace("，", ",").split(",") if part.strip()]
    return [str(value).strip()]
