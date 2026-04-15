from __future__ import annotations

import re
from dataclasses import dataclass

from app.backend.analysis.document_analyzer import split_text_into_units
from app.backend.models import PreprocessResult


NOISE_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[。！？!?；;]|(?<!\w)\.(?!\w))\s+")
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "关于",
    "可以",
    "需要",
    "一个",
    "以及",
}


@dataclass
class ChunkContext:
    chunk_text: str
    prev_context: str
    next_context: str
    chunk_index: int
    chunk_count: int


def preprocess_text(
    raw_text: str,
    *,
    chunk_size: int = 1800,
    chunk_overlap: int = 180,
) -> PreprocessResult:
    clean_text = normalize_text(raw_text)
    title_candidate = infer_title(clean_text)
    pre_keywords = extract_keywords(clean_text)
    chunks = chunk_text(clean_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return PreprocessResult(
        clean_text=clean_text,
        title_candidate=title_candidate,
        pre_keywords=pre_keywords,
        chunks=chunks,
    )


def normalize_text(raw_text: str) -> str:
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = NOISE_PATTERN.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def infer_title(clean_text: str, *, max_length: int = 48) -> str:
    for line in clean_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        markdown = re.match(r"^#{1,6}\s+(.+)$", stripped)
        if markdown:
            return markdown.group(1).strip()[:max_length]
        numbered = re.match(
            r"^(?:第[一二三四五六七八九十百]+[章节部分点]|[一二三四五六七八九十百]+[、.．]|[（(]?[0-9]+[）).、])\s*(.+)$",
            stripped,
        )
        if numbered and len(stripped) <= max_length + 16:
            return (numbered.group(1).strip() or stripped)[:max_length]

    keywords = extract_keywords(clean_text, limit=3)
    if keywords:
        return " / ".join(keywords)[:max_length]
    for line in clean_text.splitlines():
        cleaned = line.strip("#- 　\t")
        if cleaned:
            topic = re.split(r"[。！？!?；;，,]", cleaned, maxsplit=1)[0].strip()
            return (topic or cleaned)[:max_length]
    return "未命名条目"


def extract_keywords(clean_text: str, *, limit: int = 12) -> list[str]:
    counts: dict[str, int] = {}
    for match in WORD_PATTERN.findall(clean_text):
        token = match.strip().lower()
        if token in STOPWORDS:
            continue
        for candidate in _keyword_candidates(token):
            if candidate in STOPWORDS:
                continue
            counts[candidate] = counts.get(candidate, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
    return [token for token, _ in ranked[:limit]]


def _keyword_candidates(token: str) -> list[str]:
    if not re.fullmatch(r"[\u4e00-\u9fff]+", token):
        return [token]
    if 2 <= len(token) <= 8:
        return [token]
    cleaned = re.sub(
        r"(我之前|我以前|关于|有没有|有哪些|哪一些|哪些|什么|怎么|如何|是否|可以|用于|用来|内容|想法|意见|一个|以及|需要|进行)",
        " ",
        token,
    )
    candidates: list[str] = []
    for run in re.split(r"\s+", cleaned):
        if 2 <= len(run) <= 8:
            candidates.append(run)
        elif len(run) > 8:
            for size in (6, 5, 4, 3, 2):
                for start in range(0, len(run) - size + 1):
                    piece = run[start : start + size]
                    if not all(ch in "的是了和与有在中对及或" for ch in piece):
                        candidates.append(piece)
    return candidates or [token[:8]]


def chunk_text(clean_text: str, *, chunk_size: int = 1800, chunk_overlap: int = 180) -> list[str]:
    return split_text_into_units(clean_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def build_chunk_contexts(chunks: list[str], *, context_chars: int = 260) -> list[ChunkContext]:
    contexts: list[ChunkContext] = []
    total = len(chunks)
    for index, chunk in enumerate(chunks):
        prev_context = chunks[index - 1][-context_chars:].strip() if index > 0 else ""
        next_context = chunks[index + 1][:context_chars].strip() if index < total - 1 else ""
        contexts.append(
            ChunkContext(
                chunk_text=chunk,
                prev_context=prev_context,
                next_context=next_context,
                chunk_index=index,
                chunk_count=total,
            )
        )
    return contexts


def _split_long_paragraph(paragraph: str, chunk_size: int) -> list[str]:
    sentences = _split_sentences(paragraph)
    if not sentences:
        return [paragraph]
    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > chunk_size:
            if current:
                parts.append(" ".join(current).strip())
                current = []
                current_len = 0
            hard = _hard_split(sentence, chunk_size)
            parts.extend(hard)
            continue
        if current and current_len + 1 + len(sentence) > chunk_size:
            parts.append(" ".join(current).strip())
            current = [sentence]
            current_len = len(sentence)
        else:
            if current:
                current_len += 1 + len(sentence)
            else:
                current_len = len(sentence)
            current.append(sentence)
    if current:
        parts.append(" ".join(current).strip())
    return parts or [paragraph]


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    raw = SENTENCE_SPLIT_PATTERN.split(text)
    return [part.strip() for part in raw if part.strip()]


def _hard_split(text: str, chunk_size: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        parts.append(text[start:end].strip())
        start = end
    return [part for part in parts if part]


def _merge_tiny_chunks(chunks: list[str], *, min_chunk_size: int) -> list[str]:
    if not chunks:
        return []
    merged: list[str] = [chunks[0]]
    for chunk in chunks[1:]:
        if len(chunk) < min_chunk_size:
            merged[-1] = f"{merged[-1]}\n\n{chunk}".strip()
        else:
            merged.append(chunk)
    if len(merged) >= 2 and len(merged[-1]) < min_chunk_size:
        merged[-2] = f"{merged[-2]}\n\n{merged[-1]}".strip()
        merged.pop()
    return merged


def _apply_overlap(chunks: list[str], *, overlap: int) -> list[str]:
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    enriched = [chunks[0]]
    for index in range(1, len(chunks)):
        tail = chunks[index - 1][-overlap:].strip()
        if tail:
            enriched.append(f"{tail}\n\n{chunks[index]}".strip())
        else:
            enriched.append(chunks[index])
    return enriched
