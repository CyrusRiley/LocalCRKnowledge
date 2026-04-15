from __future__ import annotations

import os
import re
from logging import Logger

from app.backend.database.repository import KnowledgeRepository
from app.backend.llm.client import LLMError, QwenClient
from app.backend.models import SearchResult
from app.backend.preprocess.cleaner import extract_keywords


class SearchService:
    def __init__(self, repo: KnowledgeRepository, llm_client: QwenClient, logger: Logger):
        self.repo = repo
        self.llm_client = llm_client
        self.logger = logger

    def keyword_search(self, query: str, *, limit: int = 10) -> list[SearchResult]:
        self.logger.info("Keyword search: %s", query)
        return self.repo.search_notes(query, limit=limit)

    def theme_search(self, theme: str, *, limit: int = 10) -> list[dict]:
        self.logger.info("Theme search: %s", theme)
        return self.repo.list_notes(limit=limit, theme=theme)

    def question_search(self, question: str, *, limit: int = 10) -> tuple[dict, list[SearchResult]]:
        if os.getenv("LK_DISABLE_LLM", "").lower() in {"1", "true", "yes"}:
            parsed = self._fallback_parse(question)
        else:
            try:
                parsed = self.llm_client.parse_question(question)
            except LLMError as exc:
                self.logger.error("Question parsing failed, falling back to local keywords: %s", exc)
                parsed = self._fallback_parse(question)

        search_query = " ".join(parsed.get("keywords") or []) or question
        theme = (parsed.get("themes") or [None])[0]
        note_type = parsed.get("note_type") or None
        self.logger.info("Question search: %s -> %s", question, search_query)
        results = self.repo.search_notes(search_query, limit=limit, theme=theme, note_type=note_type)
        if not results and theme:
            results = self.repo.search_notes(search_query, limit=limit, note_type=note_type)
        if not results:
            results = self._retry_terms(parsed.get("keywords") or [], limit=limit, note_type=note_type)
        results = self._expand_with_adjacent_chunks(results, limit=limit)
        return parsed, results

    def _fallback_parse(self, question: str) -> dict:
        return {
            "keywords": _question_terms(question) or [question],
            "themes": [],
            "note_type": "",
            "intent": "question",
        }

    def _retry_terms(self, terms: list[str], *, limit: int, note_type: str | None) -> list[SearchResult]:
        seen: set[str] = set()
        merged: list[SearchResult] = []
        for term in sorted((term for term in terms if term), key=len):
            for item in self.repo.search_notes(term, limit=limit, note_type=note_type):
                if item.note_id not in seen:
                    seen.add(item.note_id)
                    merged.append(item)
                if len(merged) >= limit:
                    return merged
        return merged

    def _expand_with_adjacent_chunks(self, results: list[SearchResult], *, limit: int) -> list[SearchResult]:
        if not results:
            return results
        seed_ids = [item.note_id for item in results]
        neighbor_budget = max(2, min(6, limit // 2))
        neighbors = self.repo.get_adjacent_chunk_notes(seed_ids, distance=1, limit=neighbor_budget)
        seen = {item.note_id for item in results}
        merged = list(results)
        for item in neighbors:
            if item.note_id not in seen:
                merged.append(item)
                seen.add(item.note_id)
        return merged


def _question_terms(question: str) -> list[str]:
    terms = extract_keywords(question, limit=8)
    fragments = re.split(
        r"我之前|我以前|关于|有哪些|哪些|什么|如何|怎么|是否|的想法|想法|内容|可用于|用于|，|。|？|\?|、|\s+",
        question,
    )
    for fragment in fragments:
        cleaned = fragment.strip(" ：:；;,.!?")
        if 2 <= len(cleaned) <= 16 and cleaned not in terms:
            terms.append(cleaned)
    return terms[:8]
