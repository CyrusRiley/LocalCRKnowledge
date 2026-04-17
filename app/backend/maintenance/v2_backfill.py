from __future__ import annotations

from logging import Logger
from uuid import uuid4

from app.backend.database.repository import KnowledgeRepository
from app.backend.keywords.normalizer import canonicalize_keywords
from app.backend.models import KnowledgeGroup, KnowledgeUnit
from app.backend.utils.time_utils import utc_now_iso


def backfill_v2_structures(repo: KnowledgeRepository, logger: Logger | None = None) -> dict:
    missing = repo.list_notes_without_units(limit=10000)
    if not missing:
        return {"groups": 0, "units": 0, "keywords": 0}

    group_by_source: dict[str, str] = {}
    groups = 0
    units = 0
    keyword_links = 0
    now = utc_now_iso()

    for note in missing:
        source_id = str(note.get("source_id") or "")
        if not source_id:
            continue
        group_id = group_by_source.get(source_id)
        if not group_id:
            existing_group = repo.get_first_group_for_source(source_id)
            if existing_group:
                group_id = existing_group["group_id"]
            else:
                group_id = str(uuid4())
                repo.upsert_knowledge_group(
                    KnowledgeGroup(
                        group_id=group_id,
                        source_id=source_id,
                        group_title=str(note.get("title") or "历史导入"),
                        group_type="legacy_backfill",
                        summary=str(note.get("summary") or "")[:300],
                        created_at=now,
                        updated_at=now,
                    )
                )
                groups += 1
            group_by_source[source_id] = group_id

        unit_id = str(uuid4())
        keywords, links = canonicalize_keywords(
            repo,
            [*(note.get("keywords") or []), *(note.get("themes") or [])],
            source="backfill",
        )
        repo.insert_knowledge_unit(
            KnowledgeUnit(
                unit_id=unit_id,
                group_id=group_id,
                source_id=source_id,
                note_id=str(note.get("note_id") or ""),
                title=str(note.get("title") or "未命名条目"),
                content=_unit_content(note),
                evidence=str(note.get("source_excerpt") or ""),
                note_type=str(note.get("note_type") or ""),
                order_index=0,
                confidence=0.6,
                attributes={
                    "themes": note.get("themes") or [],
                    "faithful_content": str(note.get("faithful_content") or "")[:1200],
                    "key_points": note.get("key_points") or [],
                    "usage_scenarios": note.get("usage_scenarios") or [],
                    "backfilled_keywords": keywords,
                },
                created_at=now,
                updated_at=now,
            )
        )
        repo.replace_unit_keywords(unit_id, links)
        units += 1
        keyword_links += len(links)

    if logger:
        logger.info("Backfilled v2 structures: groups=%s units=%s keyword_links=%s", groups, units, keyword_links)
    return {"groups": groups, "units": units, "keywords": keyword_links}


def _unit_content(note: dict) -> str:
    parts: list[str] = []
    if note.get("faithful_content"):
        parts.append(str(note["faithful_content"]))
    if note.get("summary"):
        parts.append(str(note["summary"]))
    for item in note.get("key_points") or []:
        parts.append(f"- {item}")
    if note.get("user_insights"):
        parts.append(str(note["user_insights"]))
    return "\n".join(parts).strip() or str(note.get("source_excerpt") or note.get("markdown_content") or "")
