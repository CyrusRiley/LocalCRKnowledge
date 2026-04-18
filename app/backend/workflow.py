from __future__ import annotations

import os
from logging import Logger
from typing import Callable
from uuid import uuid4

from app.backend.analysis.document_analyzer import analyze_document
from app.backend.config import Settings
from app.backend.database.repository import KnowledgeRepository
from app.backend.embeddings.indexer import index_unit_embeddings
from app.backend.formatter.markdown_builder import build_note_markdown
from app.backend.keywords.normalizer import canonicalize_keywords
from app.backend.llm.client import LLMError, QwenClient
from app.backend.llm.parser import normalize_note_payload
from app.backend.models import ChunkRecord, KnowledgeGroup, KnowledgeUnit, SourceRecord, StructuredNote
from app.backend.preprocess.cleaner import build_chunk_contexts, extract_keywords, infer_title, preprocess_text
from app.backend.utils.time_utils import utc_now_iso

ProgressCallback = Callable[[dict], None]


def process_source(
    source: SourceRecord,
    *,
    direction: str,
    repo: KnowledgeRepository,
    llm_client: QwenClient,
    settings: Settings,
    logger: Logger,
    progress_cb: ProgressCallback | None = None,
) -> dict:
    def emit(stage: str, message: str, percent: int, **extra) -> None:
        if progress_cb:
            progress_cb(
                {
                    "status": "running",
                    "stage": stage,
                    "message": message,
                    "percent": max(0, min(100, int(percent))),
                    **extra,
                }
            )

    emit("start", "开始处理来源文本", 1)
    if not source.raw_text.strip():
        source.status = "failed"
        repo.upsert_source(source)
        return {"status": "failed", "reason": "empty_text", "source_id": source.source_id}

    existing_path = repo.get_source_by_path(source.file_path) if source.file_path else None
    if existing_path and existing_path.text_hash == source.text_hash:
        logger.info("Skipped unchanged file: %s", source.file_path)
        emit("skip", "文件未变化，跳过整理", 100)
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
            emit("skip", "内容重复，跳过入库", 100)
            return {"status": "duplicate", "source_id": source.source_id, "duplicate_of": duplicate.source_id}

    emit("preprocess", "正在清洗文本并进行初步切分", 5)
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
    emit("source", "正在写入来源记录", 10)
    repo.upsert_source(source)

    chunk_texts = preprocess.chunks if preprocess.chunks else [preprocess.clean_text]
    chunk_contexts = build_chunk_contexts(chunk_texts)
    llm_disabled = os.getenv("LK_DISABLE_LLM", "").lower() in {"1", "true", "yes"}
    llm_failed = False
    note_ids: list[str] = []
    note_markdowns: list[str] = []
    group_id = str(uuid4())
    now = utc_now_iso()
    repo.upsert_knowledge_group(
        KnowledgeGroup(
            group_id=group_id,
            source_id=source.source_id,
            group_title=preprocess.title_candidate,
            group_type=direction,
            summary=preprocess.clean_text[:300],
            created_at=now,
            updated_at=now,
        )
    )

    if not llm_disabled:
        structure_message = (
            "正在调用 Qwen 识别长文本结构"
            if len(preprocess.clean_text) > settings.chunk_size and not _chunks_look_structured(chunk_texts)
            else "正在确认文本结构"
        )
        emit("structure", structure_message, 14, total_chunks=len(chunk_texts))
        assisted_chunks = _try_llm_structure_assist(
            preprocess.clean_text,
            current_chunks=chunk_texts,
            llm_client=llm_client,
            settings=settings,
            logger=logger,
        )
        if assisted_chunks:
            chunk_texts = assisted_chunks
            chunk_contexts = build_chunk_contexts(chunk_texts)
        emit("structure", f"已确定 {len(chunk_contexts)} 个知识单元", 20, total_chunks=len(chunk_contexts))
    else:
        emit("structure", f"已确定 {len(chunk_contexts)} 个知识单元（模型已禁用）", 20, total_chunks=len(chunk_contexts))

    for ctx in chunk_contexts:
        total_chunks = max(1, ctx.chunk_count)
        before_percent = 20 + int((ctx.chunk_index / total_chunks) * 60)
        emit(
            "llm",
            f"正在调用 Qwen 整理第 {ctx.chunk_index + 1}/{ctx.chunk_count} 个知识单元",
            before_percent,
            current_chunk=ctx.chunk_index + 1,
            total_chunks=ctx.chunk_count,
        )
        chunk_keywords = extract_keywords(ctx.chunk_text) or preprocess.pre_keywords
        chunk_title = infer_title(ctx.chunk_text)
        if ctx.chunk_count > 1:
            chunk_title = f"{chunk_title}（片段{ctx.chunk_index + 1}/{ctx.chunk_count}）"

        chunk_llm_failed = False
        if llm_disabled:
            llm_failed = True
            chunk_llm_failed = True
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
                chunk_llm_failed = True
                logger.error("LLM organize failed for source %s chunk %s: %s", source.source_id, ctx.chunk_index, exc)
                payload = fallback_payload(chunk_title, ctx.chunk_text, chunk_keywords)

        emit(
            "keywords",
            f"正在归一关键词并写入第 {ctx.chunk_index + 1}/{ctx.chunk_count} 个知识单元",
            min(88, 20 + int(((ctx.chunk_index + 0.75) / total_chunks) * 60)),
            current_chunk=ctx.chunk_index + 1,
            total_chunks=ctx.chunk_count,
        )
        raw_keywords = [*payload.get("keywords", []), *payload.get("themes", []), *chunk_keywords]
        canonical_keywords, keyword_links = canonicalize_keywords(repo, raw_keywords, source="import")
        if not str(payload.get("faithful_content") or "").strip():
            payload["faithful_content"] = ctx.chunk_text[:1200]
        if canonical_keywords:
            payload["keywords"] = canonical_keywords[:12]
            if not payload.get("themes"):
                payload["themes"] = canonical_keywords[:4]

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
        unit_id = str(uuid4())
        unit = KnowledgeUnit(
            unit_id=unit_id,
            group_id=group_id,
            source_id=source.source_id,
            note_id=note_id,
            title=note.title,
            content=_unit_content(note),
            evidence=note.source_excerpt,
            note_type=note.note_type,
            order_index=ctx.chunk_index,
            confidence=0.55 if chunk_llm_failed else 0.78,
            attributes={
                "themes": note.themes,
                "faithful_content": note.faithful_content[:1200],
                "key_points": note.key_points,
                "usage_scenarios": note.usage_scenarios,
                "user_insights": note.user_insights,
                "raw_chunk_keywords": chunk_keywords,
            },
            created_at=now,
            updated_at=now,
        )
        repo.insert_knowledge_unit(unit)
        index_unit_embeddings(
            repo,
            {
                "unit_id": unit.unit_id,
                "note_id": unit.note_id,
                "title": unit.title,
                "content": unit.content,
                "evidence": unit.evidence,
                "note_summary": note.summary,
                "note_faithful_content": note.faithful_content,
                "themes": note.themes,
                "keywords": note.keywords,
            },
        )
        repo.replace_unit_keywords(unit_id, keyword_links)
        note_ids.append(note_id)
        note_markdowns.append(markdown)
        emit(
            "chunk_done",
            f"已完成第 {ctx.chunk_index + 1}/{ctx.chunk_count} 个知识单元",
            min(90, 20 + int(((ctx.chunk_index + 1) / total_chunks) * 60)),
            current_chunk=ctx.chunk_index + 1,
            total_chunks=ctx.chunk_count,
        )

    emit("finalize", "正在更新来源状态", 94, total_chunks=len(chunk_contexts))
    repo.set_source_status(source.source_id, "failed" if llm_failed else source.status if source.status == "updated" else "processed")
    logger.info("Processed source %s into %s notes", source.source_id, len(note_ids))
    emit("completed", f"来源处理完成，生成 {len(note_ids)} 条知识", 100, total_chunks=len(chunk_contexts))
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
            "faithful_content": clean_text[:1200],
            "key_points": [line.strip("- ") for line in clean_text.splitlines() if line.strip()][:5],
            "usage_scenarios": [],
            "user_insights": "",
            "keywords": pre_keywords,
            "source_excerpt": clean_text[:300],
        },
        fallback_title=title,
        source_text=clean_text,
    )


def _unit_content(note: StructuredNote) -> str:
    parts: list[str] = []
    if note.faithful_content:
        parts.append(note.faithful_content)
    if note.summary:
        parts.append(note.summary)
    if note.key_points:
        parts.extend(f"- {item}" for item in note.key_points)
    if note.user_insights:
        parts.append(note.user_insights)
    return "\n".join(parts).strip() or note.source_excerpt or note.markdown_content


def _try_llm_structure_assist(
    clean_text: str,
    *,
    current_chunks: list[str],
    llm_client: QwenClient,
    settings: Settings,
    logger: Logger,
) -> list[str]:
    if len(clean_text) <= settings.chunk_size:
        return []
    if len(current_chunks) >= 2 and _chunks_look_structured(current_chunks):
        return []
    try:
        markers = llm_client.detect_structure(clean_text, max_units=18)
    except LLMError as exc:
        logger.info("LLM structure detection skipped: %s", exc)
        return []

    sections = analyze_document(
        clean_text,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        llm_markers=markers,
    )
    assisted_chunks = [section.text for section in sections if section.text.strip()]
    if len(assisted_chunks) > len(current_chunks):
        logger.info("LLM structure detection split source into %s units", len(assisted_chunks))
        return assisted_chunks
    return []


def _chunks_look_structured(chunks: list[str]) -> bool:
    structured = 0
    for chunk in chunks:
        first = next((line.strip() for line in chunk.splitlines() if line.strip()), "")
        if first.startswith("#") or first[:2] in {"一、", "二、", "三、", "四、", "五、", "六、", "七、", "八、", "九、"}:
            structured += 1
        elif first[:2] in {"首先", "其次", "再次", "最后", "第一", "第二", "第三", "第四", "第五", "第六", "第七"}:
            structured += 1
    return structured >= max(1, len(chunks) // 2)
