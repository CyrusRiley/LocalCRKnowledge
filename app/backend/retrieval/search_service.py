from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from logging import Logger

from app.backend.database.repository import KnowledgeRepository
from app.backend.embeddings.indexer import rebuild_unit_embeddings
from app.backend.embeddings.providers import active_embedding_models, get_embedder
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
        vector_results = self._vector_search(query, limit=limit * 2)
        results, scores, reasons = self._merge_ranked(
            [
                RankedSource("fts_exact", self.repo.search_notes(query, limit=limit * 2), 1.0),
                RankedSource("fts_expanded", self.repo.search_notes(" ".join(terms), limit=limit * 2) if terms else [], 0.85),
                RankedSource("title", self.repo.search_notes_by_title(" ".join(terms or [query]), limit=limit * 2), 1.2),
                RankedSource("unit", self.repo.search_notes_by_unit_text(" ".join(terms or [query]), limit=limit * 2), 0.95),
                RankedSource("keyword", self.repo.search_notes_by_keyword_names(terms, limit=limit * 2) if terms else [], 1.1),
                RankedSource("vector", vector_results, 1.05),
            ],
            limit=limit * 2,
        )
        results, context_scores, context_reasons = self._expand_context(results, limit=limit * 2)
        scores.update({note_id: max(scores.get(note_id, 0.0), score) for note_id, score in context_scores.items()})
        for note_id, items in context_reasons.items():
            reasons.setdefault(note_id, []).extend(items)
        return self._rerank(results, scores=scores, reasons=reasons, query_terms=terms or [query], limit=limit)

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
        vector_results = self._vector_search(question, limit=limit * 3)
        theme = (parsed.get("themes") or [None])[0]
        note_type = parsed.get("note_type") or None
        self.logger.info("Question search: %s -> %s", question, search_query)
        results, scores, reasons = self._merge_ranked(
            [
                RankedSource("fts_question", self.repo.search_notes(question, limit=limit * 2, theme=theme, note_type=note_type), 1.0),
                RankedSource("fts_expanded", self.repo.search_notes(search_query, limit=limit * 2, theme=theme, note_type=note_type), 0.95),
                RankedSource("title", self.repo.search_notes_by_title(search_query, limit=limit * 2), 1.2),
                RankedSource("unit", self.repo.search_notes_by_unit_text(search_query, limit=limit * 2), 1.0),
                RankedSource("keyword", self.repo.search_notes_by_keyword_names(expanded_terms, limit=limit * 2) if expanded_terms else [], 1.1),
                RankedSource("vector", vector_results, 1.15),
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
            reasons = {item.note_id: ["关键词拆分重试命中"] for item in results}
        results, context_scores, context_reasons = self._expand_context(results, limit=limit * 2)
        scores.update({note_id: max(scores.get(note_id, 0.0), score) for note_id, score in context_scores.items()})
        for note_id, items in context_reasons.items():
            reasons.setdefault(note_id, []).extend(items)
        results = self._rerank(results, scores=scores, reasons=reasons, query_terms=expanded_terms or parsed_keywords or [question], limit=limit)
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

    def _expand_context(self, results: list[SearchResult], *, limit: int) -> tuple[list[SearchResult], dict[str, float], dict[str, list[str]]]:
        if not results:
            return results, {}, {}
        seed_ids = [item.note_id for item in results]
        neighbor_budget = max(2, min(8, limit // 2))
        group_peers = self.repo.get_group_peer_notes(seed_ids, limit=neighbor_budget)
        relation_peers = self.repo.get_relation_peer_notes(seed_ids, limit=neighbor_budget)
        neighbors = self.repo.get_adjacent_chunk_notes(seed_ids, distance=1, limit=neighbor_budget)
        seen = {item.note_id for item in results}
        merged = list(results)
        context_scores: dict[str, float] = {}
        context_reasons: dict[str, list[str]] = {}
        for item in [*group_peers, *relation_peers, *neighbors]:
            if item.note_id not in seen:
                merged.append(item)
                seen.add(item.note_id)
                score, reason = self._context_score_and_reason(item)
                context_scores[item.note_id] = score
                context_reasons[item.note_id] = [reason]
            if len(merged) >= limit:
                break
        return merged, context_scores, context_reasons

    def _merge_ranked(self, ranked_sources: list[RankedSource], *, limit: int) -> tuple[list[SearchResult], dict[str, float], dict[str, list[str]]]:
        scores: dict[str, float] = {}
        reasons: dict[str, list[str]] = {}
        items: dict[str, SearchResult] = {}
        for source in ranked_sources:
            for rank, item in enumerate(source.results):
                items.setdefault(item.note_id, item)
                scores[item.note_id] = scores.get(item.note_id, 0.0) + source.weight / (rank + 1)
                reasons.setdefault(item.note_id, []).append(_source_reason(source.name))
        ordered = sorted(items.values(), key=lambda item: (-scores.get(item.note_id, 0.0), item.score, item.title))
        return ordered[:limit], scores, reasons

    def _rerank(
        self,
        results: list[SearchResult],
        *,
        scores: dict[str, float],
        reasons: dict[str, list[str]],
        query_terms: list[str],
        limit: int,
    ) -> list[SearchResult]:
        terms = [term.lower() for term in query_terms if term and len(term.strip()) >= 2]
        scored: list[tuple[SearchResult, float, list[str]]] = []
        for item in results:
            item_score, item_reasons = self._rule_score(item, terms, scores.get(item.note_id, 0.0))
            all_reasons = _dedupe_reasons([*(reasons.get(item.note_id) or []), *item_reasons])
            scored.append((item, item_score, all_reasons))
        ranked = sorted(scored, key=lambda row: (-row[1], row[0].title))
        if not ranked:
            return []
        top_score = max(ranked[0][1], 0.0001)
        annotated: list[SearchResult] = []
        for index, (item, score, item_reasons) in enumerate(ranked[:limit]):
            level = _relevance_level(score, top_score, index)
            relation_type, relation_strength = _relation_from_snippet(item.snippet)
            annotated.append(
                replace(
                    item,
                    relevance_score=round(score, 4),
                    relevance_level=level,
                    relevance_reason="；".join(item_reasons[:5]),
                    relation_type=relation_type,
                    relation_strength=relation_strength,
                    match_source=_match_source(item.snippet),
                )
            )
        return annotated

    def _rule_score(self, item: SearchResult, terms: list[str], base_score: float) -> tuple[float, list[str]]:
        haystacks = {
            "title": item.title.lower(),
            "keywords": " ".join(item.keywords).lower(),
            "themes": " ".join(item.themes).lower(),
            "summary": item.summary.lower(),
            "snippet": item.snippet.lower(),
            "markdown": item.markdown_content.lower(),
        }
        score = base_score
        reasons: list[str] = []
        for term in terms:
            if term in haystacks["title"]:
                score += 0.45
                reasons.append(f"标题命中“{term}”")
            if term in haystacks["keywords"]:
                score += 0.38
                reasons.append(f"关键词命中“{term}”")
            if term in haystacks["themes"]:
                score += 0.30
                reasons.append(f"主题命中“{term}”")
            if term in haystacks["summary"]:
                score += 0.18
                reasons.append(f"摘要命中“{term}”")
            if term in haystacks["snippet"]:
                score += 0.12
                reasons.append(f"匹配片段命中“{term}”")
            if term in haystacks["markdown"]:
                score += 0.10
                reasons.append(f"完整内容命中“{term}”")
        if item.snippet.startswith("near_group:"):
            score += 0.28
            reasons.append("同一知识组相邻内容补充")
        if item.snippet.startswith("relation:"):
            relation_type, relation_strength = _relation_from_snippet(item.snippet)
            score += 0.55 if relation_strength == "strong" else 0.42
            reasons.append(f"知识关系网扩展：{relation_type or 'related'}")
        if item.snippet.startswith("vector:"):
            score += min(0.75, max(0.0, item.score)) * 0.9
            reasons.append("向量语义检索命中")
        return score, _dedupe_reasons(reasons)

    def _context_score_and_reason(self, item: SearchResult) -> tuple[float, str]:
        if item.snippet.startswith("relation:"):
            relation_type, relation_strength = _relation_from_snippet(item.snippet)
            return (0.75 if relation_strength == "strong" else 0.58), f"知识关系网扩展：{relation_type or 'related'}"
        if item.snippet.startswith("near_group:"):
            return 0.42, "同一知识组相邻内容补充"
        return 0.34, "同一来源相邻片段补充"

    def _vector_search(self, query: str, *, limit: int) -> list[SearchResult]:
        if os.getenv("LK_DISABLE_VECTOR_SEARCH", "").lower() in {"1", "true", "yes"}:
            return []
        try:
            configs = active_embedding_models(query)
            rebuild_unit_embeddings(self.repo, limit=10000, configs=configs)
            merged: dict[str, tuple[SearchResult, float]] = {}
            for config in configs:
                try:
                    embedder = get_embedder(config)
                    query_vector = embedder.embed(query)
                    model_results = self.repo.search_notes_by_vector(query_vector, model_name=embedder.model_name, limit=limit)
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning("Embedding model %s skipped: %s", config.key, exc)
                    continue
                for rank, item in enumerate(model_results, start=1):
                    rrf = config.weight / (60 + rank)
                    combined = (item.score * config.weight) + rrf
                    existing = merged.get(item.note_id)
                    snippet = f"vector:{config.key}:{item.snippet.removeprefix('vector:')}"
                    item = replace(item, score=combined, snippet=snippet)
                    if not existing or combined > existing[1]:
                        merged[item.note_id] = (item, combined)
            ordered = sorted(merged.values(), key=lambda pair: pair[1], reverse=True)
            return [item for item, _ in ordered[:limit]]
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Vector search skipped: %s", exc)
            return []

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


def _source_reason(source_name: str) -> str:
    labels = {
        "fts_exact": "全文检索直接命中",
        "fts_question": "问题原文命中",
        "fts_expanded": "扩展关键词命中",
        "title": "标题检索命中",
        "unit": "知识单元内容命中",
        "keyword": "关键词库命中",
        "vector": "向量语义检索命中",
    }
    return labels.get(source_name, source_name)


def _dedupe_reasons(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def _relevance_level(score: float, top_score: float, index: int) -> str:
    if index <= 2 or score >= top_score * 0.68:
        return "高相关"
    if score >= top_score * 0.38:
        return "中相关"
    return "补充相关"


def _relation_from_snippet(snippet: str) -> tuple[str, str]:
    if not snippet.startswith("relation:"):
        return "", ""
    parts = snippet.split(":", 3)
    relation_type = parts[1] if len(parts) > 1 else ""
    relation_strength = parts[2] if len(parts) > 2 else ""
    return relation_type, relation_strength


def _match_source(snippet: str) -> str:
    if snippet.startswith("relation:"):
        return "relation"
    if snippet.startswith("near_group:"):
        return "group"
    if snippet.startswith("unit:"):
        return "unit"
    if snippet.startswith("title:"):
        return "title"
    if snippet.startswith("keyword:"):
        return "keyword"
    if snippet.startswith("vector:"):
        return "vector"
    return "text"
