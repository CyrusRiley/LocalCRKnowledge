from __future__ import annotations

import re
from dataclasses import dataclass

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
        line = line.strip("#- 　\t")
        if line:
            return line[:max_length]
    return "未命名条目"


def extract_keywords(clean_text: str, *, limit: int = 12) -> list[str]:
    counts: dict[str, int] = {}
    for match in WORD_PATTERN.findall(clean_text):
        token = match.strip().lower()
        if token in STOPWORDS:
            continue
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _ in ranked[:limit]]


def chunk_text(clean_text: str, *, chunk_size: int = 1800, chunk_overlap: int = 180) -> list[str]:
    if not clean_text:
        return []
    if len(clean_text) <= chunk_size:
        return [clean_text]

    paragraphs = [part.strip() for part in clean_text.split("\n\n") if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        units = [paragraph] if len(paragraph) <= chunk_size else _split_long_paragraph(paragraph, chunk_size)
        for unit in units:
            unit_len = len(unit)
            if current and current_len + 2 + unit_len > chunk_size:
                chunks.append("\n\n".join(current).strip())
                current = [unit]
                current_len = unit_len
            else:
                if current:
                    current_len += 2 + unit_len
                else:
                    current_len = unit_len
                current.append(unit)

    if current:
        chunks.append("\n\n".join(current).strip())

    merged = _merge_tiny_chunks(chunks, min_chunk_size=max(80, chunk_size // 4))
    return _apply_overlap(merged, overlap=max(0, min(chunk_overlap, chunk_size // 4)))


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
