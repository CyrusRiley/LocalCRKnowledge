from __future__ import annotations

import os
from logging import Logger
from uuid import uuid4

from app.backend.config import Settings
from app.backend.database.repository import KnowledgeRepository
from app.backend.formatter.markdown_builder import build_note_markdown
from app.backend.llm.client import LLMError, QwenClient
from app.backend.llm.parser import normalize_note_payload
from app.backend.models import ChunkRecord, SourceRecord, StructuredNote
from app.backend.preprocess.cleaner import build_chunk_contexts, extract_keywords, infer_title, preprocess_text
from app.backend.utils.time_utils import utc_now_iso


def process_source(
    source: SourceRecord,
    *,
    direction: str,
    repo: KnowledgeRepository,
    llm_client: QwenClient,
    settings: Settings,
    logger: Logger,
) -> dict:
    if not source.raw_text.strip():
        source.status = "failed"
        repo.upsert_source(source)
        return {"status": "failed", "reason": "empty_text", "source_id": source.source_id}

    existing_path = repo.get_source_by_path(source.file_path) if source.file_path else None
    if existing_path and existing_path.text_hash == source.text_hash:
        logger.info("Skipped unchanged file: %s", source.file_path)
        return {"status": "skipped", "reason": "unchanged", "source_id": existing_path.source_id}

    if existing_path and existing_path.text_hash != source.text_hash:
        source.source_id = existing_path.source_id
        source.created_at = existing_path.created_at
        source.status = "updated"
        repo.delete_notes_for_source(source.source_id)
    else:
        duplicate = repo.get_source_by_hash(source.text_hash)
        if duplicate:
            source.status = "duplicate"
            repo.upsert_source(source)
            logger.info("Skipped duplicate content: %s", source.file_path or source.source_id)
            return {"status": "duplicate", "source_id": source.source_id, "duplicate_of": duplicate.source_id}

    preprocess = preprocess_text(
        source.raw_text,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if len(preprocess.clean_text) < 3:
        source.clean_text = preprocess.clean_text
        source.status = "failed"
        repo.upsert_source(source)
        return {"status": "failed", "reason": "too_short", "source_id": source.source_id}

    source.clean_text = preprocess.clean_text
    repo.upsert_source(source)

    chunk_texts = preprocess.chunks if preprocess.chunks else [preprocess.clean_text]
    chunk_contexts = build_chunk_contexts(chunk_texts)
    llm_disabled = os.getenv("LK_DISABLE_LLM", "").lower() in {"1", "true", "yes"}
    llm_failed = False
    note_ids: list[str] = []
    note_markdowns: list[str] = []

    for ctx in chunk_contexts:
        chunk_keywords = extract_keywords(ctx.chunk_text) or preprocess.pre_keywords
        chunk_title = infer_title(ctx.chunk_text)
        if ctx.chunk_count > 1:
            chunk_title = f"{chunk_title}（片段{ctx.chunk_index + 1}/{ctx.chunk_count}）"

        if llm_disabled:
            llm_failed = True
            payload = fallback_payload(chunk_title, ctx.chunk_text, chunk_keywords)
        else:
            try:
                payload = llm_client.organize_text(
                    ctx.chunk_text,
                    direction=direction,
                    title_candidate=chunk_title,
                    pre_keywords=chunk_keywords,
                    prev_context=ctx.prev_context,
                    next_context=ctx.next_context,
                    chunk_index=ctx.chunk_index,
                    chunk_count=ctx.chunk_count,
                )
            except LLMError as exc:
                llm_failed = True
                logger.error("LLM organize failed for source %s chunk %s: %s", source.source_id, ctx.chunk_index, exc)
                payload = fallback_payload(chunk_title, ctx.chunk_text, chunk_keywords)

        markdown = build_note_markdown(payload, source)
        now = utc_now_iso()
        note_id = str(uuid4())
        note = StructuredNote.from_llm_payload(
            note_id=note_id,
            source_id=source.source_id,
            payload=payload,
            markdown_content=markdown,
            created_at=now,
            updated_at=now,
        )
        repo.insert_note(
            note,
            [
                ChunkRecord(
                    chunk_id=str(uuid4()),
                    note_id=note_id,
                    chunk_text=ctx.chunk_text,
                    chunk_order=ctx.chunk_index,
                    chunk_type="source_chunk",
                    keywords=chunk_keywords,
                )
            ],
        )
        note_ids.append(note_id)
        note_markdowns.append(markdown)

    repo.set_source_status(source.source_id, "failed" if llm_failed else source.status if source.status == "updated" else "processed")
    logger.info("Processed source %s into %s notes", source.source_id, len(note_ids))
    return {
        "status": "failed" if llm_failed else source.status if source.status == "updated" else "processed",
        "source_id": source.source_id,
        "note_id": note_ids[0] if note_ids else "",
        "note_ids": note_ids,
        "note_count": len(note_ids),
        "markdown": "\n\n---\n\n".join(note_markdowns),
        "llm_failed": llm_failed,
    }


def fallback_payload(title: str, clean_text: str, pre_keywords: list[str]) -> dict:
    return normalize_note_payload(
        {
            "title": title,
            "note_type": "待模型整理",
            "themes": [],
            "summary": clean_text[:160],
            "key_points": [line.strip("- ") for line in clean_text.splitlines() if line.strip()][:5],
            "usage_scenarios": [],
            "user_insights": "",
            "keywords": pre_keywords,
            "source_excerpt": clean_text[:300],
        },
        fallback_title=title,
        source_text=clean_text,
    )
