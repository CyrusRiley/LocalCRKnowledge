from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any
from uuid import uuid4

from app.backend.database.repository import KnowledgeRepository
from app.backend.utils.time_utils import utc_now_iso


GENERIC_SUFFIXES = (
    "技术",
    "方法",
    "方式",
    "机制",
    "模型",
    "系统",
    "体系",
    "理论",
    "研究",
    "分析",
    "工具",
    "策略",
)


def normalize_keyword_text(value: str) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("（", "(").replace("）", ")")
    text = re.sub(r"[\s_\-·•/／]+", "", text)
    return text


def canonical_stem(value: str) -> str:
    normalized = normalize_keyword_text(value)
    for suffix in GENERIC_SUFFIXES:
        if normalized.endswith(suffix) and len(normalized) > len(suffix) + 2:
            return normalized[: -len(suffix)]
    return normalized


def canonicalize_keywords(
    repo: KnowledgeRepository,
    raw_keywords: list[str],
    *,
    source: str = "auto",
) -> tuple[list[str], list[dict[str, Any]]]:
    seen_input: set[str] = set()
    canonical_names: list[str] = []
    links: list[dict[str, Any]] = []

    for raw in raw_keywords:
        text = _clean_keyword(raw)
        normalized = normalize_keyword_text(text)
        if not text or not normalized or normalized in seen_input:
            continue
        seen_input.add(normalized)

        term, matched_by, confidence = _find_or_create_term(repo, text, source=source)
        canonical = str(term["canonical_name"])
        if canonical not in canonical_names:
            canonical_names.append(canonical)
        links.append(
            {
                "term_id": term["term_id"],
                "canonical_name": canonical,
                "raw_keyword": text,
                "confidence": confidence,
                "matched_by": matched_by,
            }
        )
    return canonical_names, links


def expand_query_terms(repo: KnowledgeRepository, terms: list[str]) -> list[str]:
    expanded: list[str] = []
    seen: set[str] = set()
    all_terms = repo.list_keyword_terms()
    for term in terms:
        raw = _clean_keyword(term)
        if not raw:
            continue
        candidates = [raw]
        matched = _match_existing(all_terms, raw)
        if matched:
            candidates.append(str(matched["canonical_name"]))
            candidates.extend(str(matched.get("aliases") or "").split("||"))
        for candidate in candidates:
            candidate = candidate.strip()
            key = normalize_keyword_text(candidate)
            if candidate and key and key not in seen:
                expanded.append(candidate)
                seen.add(key)
    return expanded


def _find_or_create_term(repo: KnowledgeRepository, raw: str, *, source: str) -> tuple[dict[str, Any], str, float]:
    normalized = normalize_keyword_text(raw)
    stem = canonical_stem(raw)

    existing = repo.find_keyword_term(normalized) or repo.find_keyword_alias(normalized)
    if existing:
        return existing, "exact", 1.0

    if stem != normalized:
        stem_term = repo.find_keyword_term(stem) or repo.find_keyword_alias(stem)
        if stem_term:
            _ensure_alias(repo, stem_term["term_id"], raw, source=source, confidence=0.9)
            return stem_term, "suffix_alias", 0.9

    matched = _match_existing(repo.list_keyword_terms(), raw)
    if matched:
        _ensure_alias(repo, matched["term_id"], raw, source=source, confidence=0.82)
        return matched, "similar_alias", 0.82

    now = utc_now_iso()
    term_id = str(uuid4())
    canonical = _canonical_display(raw)
    repo.insert_keyword_term(
        term_id=term_id,
        canonical_name=canonical,
        normalized_name=stem or normalized,
        description="",
        created_at=now,
        updated_at=now,
    )
    term = repo.find_keyword_term(stem or normalized) or {
        "term_id": term_id,
        "canonical_name": canonical,
        "normalized_name": stem or normalized,
    }
    if normalized != stem:
        _ensure_alias(repo, term["term_id"], raw, source=source, confidence=0.86)
    return term, "created", 0.78


def _match_existing(terms: list[dict[str, Any]], raw: str) -> dict[str, Any] | None:
    normalized = normalize_keyword_text(raw)
    stem = canonical_stem(raw)
    best: tuple[float, dict[str, Any] | None] = (0.0, None)
    for term in terms:
        candidates = [str(term.get("canonical_name") or ""), str(term.get("normalized_name") or "")]
        candidates.extend(str(term.get("aliases") or "").split("||"))
        candidates.extend(str(term.get("normalized_aliases") or "").split("||"))
        for candidate in candidates:
            cand_norm = normalize_keyword_text(candidate)
            if not cand_norm:
                continue
            if normalized == cand_norm or stem == cand_norm:
                return term
            if (stem and cand_norm and (stem in cand_norm or cand_norm in stem)) and min(len(stem), len(cand_norm)) >= 3:
                return term
            score = SequenceMatcher(None, normalized, cand_norm).ratio()
            if score > best[0]:
                best = (score, term)
    if best[0] >= 0.88:
        return best[1]
    return None


def _ensure_alias(repo: KnowledgeRepository, term_id: str, alias: str, *, source: str, confidence: float) -> None:
    normalized = normalize_keyword_text(alias)
    if not normalized:
        return
    repo.insert_keyword_alias(
        alias_id=str(uuid4()),
        term_id=term_id,
        alias=alias,
        normalized_alias=normalized,
        source=source,
        confidence=confidence,
        created_at=utc_now_iso(),
    )


def _clean_keyword(value: str) -> str:
    text = str(value or "").strip(" \t\r\n，,;；。.!！?？、")
    return text[:40]


def _canonical_display(raw: str) -> str:
    text = _clean_keyword(raw)
    stem = canonical_stem(text)
    if stem != normalize_keyword_text(text):
        # Preserve Chinese display while dropping only common generic suffixes.
        for suffix in GENERIC_SUFFIXES:
            if text.endswith(suffix) and len(text) > len(suffix) + 2:
                return text[: -len(suffix)]
    return text

