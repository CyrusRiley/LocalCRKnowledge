from __future__ import annotations

import os
import re
from dataclasses import dataclass
from logging import Logger

from app.backend.database.repository import KnowledgeRepository
from app.backend.keywords.normalizer import expand_query_terms
from app.backend.llm.client import LLMError, QwenClient
from app.backend.models import SearchResult
from app.backend.preprocess.cleaner import extract_keywords


@dataclass
class RankedSource:
    name: str
    results: list[SearchResult]
    weight: float


class SearchService:
    def __init__(self, repo: KnowledgeRepository, llm_client: QwenClient, logger: Logger):
        self.repo = repo
        self.llm_client = llm_client
        self.logger = logger

    def keyword_search(self, query: str, *, limit: int = 10) -> list[SearchResult]:
        self.logger.info("Keyword search: %s", query)
        terms = self._expanded_terms(query, [])
        results, scores = self._merge_ranked(
            [
                RankedSource("fts_exact", self.repo.search_notes(query, limit=limit * 2), 1.0),
                RankedSource("fts_expanded", self.repo.search_notes(" ".join(terms), limit=limit * 2) if terms else [], 0.85),
                RankedSource("title", self.repo.search_notes_by_title(" ".join(terms or [query]), limit=limit * 2), 1.2),
                RankedSource("unit", self.repo.search_notes_by_unit_text(" ".join(terms or [query]), limit=limit * 2), 0.95),
                RankedSource("keyword", self.repo.search_notes_by_keyword_names(terms, limit=limit * 2) if terms else [], 1.1),
            ],
            limit=limit * 2,
        )
        results = self._expand_context(results, limit=limit * 2)
        return self._rerank(results, scores=scores, query_terms=terms or [query], limit=limit)

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

        parsed_keywords = parsed.get("keywords") or []
        parsed_themes = parsed.get("themes") or []
        expanded_terms = self._expanded_terms(question, [*parsed_keywords, *parsed_themes])
        search_query = " ".join(expanded_terms or parsed_keywords) or question
        theme = (parsed.get("themes") or [None])[0]
        note_type = parsed.get("note_type") or None
        self.logger.info("Question search: %s -> %s", question, search_query)
        results, scores = self._merge_ranked(
            [
                RankedSource("fts_question", self.repo.search_notes(question, limit=limit * 2, theme=theme, note_type=note_type), 1.0),
                RankedSource("fts_expanded", self.repo.search_notes(search_query, limit=limit * 2, theme=theme, note_type=note_type), 0.95),
                RankedSource("title", self.repo.search_notes_by_title(search_query, limit=limit * 2), 1.2),
                RankedSource("unit", self.repo.search_notes_by_unit_text(search_query, limit=limit * 2), 1.0),
                RankedSource("keyword", self.repo.search_notes_by_keyword_names(expanded_terms, limit=limit * 2) if expanded_terms else [], 1.1),
            ],
            limit=limit * 2,
        )
        if not results and theme:
            results = self.repo.search_notes(search_query, limit=limit, note_type=note_type)
        if not results:
            results = self._retry_terms(parsed.get("keywords") or [], limit=limit, note_type=note_type)
            if not results and note_type:
                results = self._retry_terms(parsed.get("keywords") or [], limit=limit, note_type=None)
            scores = {item.note_id: 0.3 for item in results}
        results = self._expand_context(results, limit=limit * 2)
        results = self._rerank(results, scores=scores, query_terms=expanded_terms or parsed_keywords or [question], limit=limit)
        parsed["expanded_terms"] = expanded_terms
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

    def _expand_context(self, results: list[SearchResult], *, limit: int) -> list[SearchResult]:
        if not results:
            return results
        seed_ids = [item.note_id for item in results]
        neighbor_budget = max(2, min(8, limit // 2))
        group_peers = self.repo.get_group_peer_notes(seed_ids, limit=neighbor_budget)
        relation_peers = self.repo.get_relation_peer_notes(seed_ids, limit=neighbor_budget)
        neighbors = self.repo.get_adjacent_chunk_notes(seed_ids, distance=1, limit=neighbor_budget)
        seen = {item.note_id for item in results}
        merged = list(results)
        for item in [*group_peers, *relation_peers, *neighbors]:
            if item.note_id not in seen:
                merged.append(item)
                seen.add(item.note_id)
            if len(merged) >= limit:
                break
        return merged

    def _merge_ranked(self, ranked_sources: list[RankedSource], *, limit: int) -> tuple[list[SearchResult], dict[str, float]]:
        scores: dict[str, float] = {}
        items: dict[str, SearchResult] = {}
        for source in ranked_sources:
            for rank, item in enumerate(source.results):
                items.setdefault(item.note_id, item)
                scores[item.note_id] = scores.get(item.note_id, 0.0) + source.weight / (60 + rank + 1)
        ordered = sorted(items.values(), key=lambda item: (-scores.get(item.note_id, 0.0), item.score, item.title))
        return ordered[:limit], scores

    def _rerank(
        self,
        results: list[SearchResult],
        *,
        scores: dict[str, float],
        query_terms: list[str],
        limit: int,
    ) -> list[SearchResult]:
        terms = [term.lower() for term in query_terms if term and len(term.strip()) >= 2]
        ranked = sorted(
            results,
            key=lambda item: (
                -self._rule_score(item, terms, scores.get(item.note_id, 0.0)),
                item.score,
                item.title,
            ),
        )
        return ranked[:limit]

    def _rule_score(self, item: SearchResult, terms: list[str], base_score: float) -> float:
        haystacks = {
            "title": item.title.lower(),
            "keywords": " ".join(item.keywords).lower(),
            "themes": " ".join(item.themes).lower(),
            "summary": item.summary.lower(),
            "snippet": item.snippet.lower(),
        }
        score = base_score
        for term in terms:
            if term in haystacks["title"]:
                score += 0.12
            if term in haystacks["keywords"]:
                score += 0.10
            if term in haystacks["themes"]:
                score += 0.08
            if term in haystacks["summary"]:
                score += 0.05
            if term in haystacks["snippet"]:
                score += 0.03
        if item.snippet.startswith("same_group:"):
            score += 0.018
        if item.snippet.startswith("relation:"):
            score += 0.015
        return score

    def _expanded_terms(self, query: str, extra_terms: list[str]) -> list[str]:
        base_terms = [query, *extra_terms, *extract_keywords(query, limit=8)]
        expanded = expand_query_terms(self.repo, base_terms)
        if expanded:
            return expanded[:16]
        seen: set[str] = set()
        result: list[str] = []
        for term in base_terms:
            term = str(term or "").strip()
            key = term.lower()
            if term and key not in seen:
                result.append(term)
                seen.add(key)
        return result[:16]


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
