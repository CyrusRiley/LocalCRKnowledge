from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import uuid4


@dataclass
class DocumentSection:
    section_id: str
    heading: str
    level: int
    text: str
    parent_id: str | None
    order_index: int
    strategy: str
    confidence: float


MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
NUMBERED_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"(?:第[一二三四五六七八九十百]+[章节部分点])|"
    r"(?:[一二三四五六七八九十百]+[、.．])|"
    r"(?:[（(][一二三四五六七八九十百0-9]+[）)])|"
    r"(?:(?:\d+(?:\.\d+){0,3}|[A-Za-z])[、.．)]))"
    r"\s*(.{0,80})$"
)
ENUMERATED_START_RE = re.compile(
    r"^\s*(?:"
    r"(?:首先|其次|再次|最后|第一|第二|第三|第四|第五|第六|第七|第八|第九|第十)[，,、：:]|"
    r"(?:\d+|[一二三四五六七八九十百]+)[、.．)]\s*|"
    r"[（(](?:\d+|[一二三四五六七八九十百]+)[）)]\s*)"
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])\s*")


def split_text_into_units(
    clean_text: str,
    *,
    chunk_size: int = 1800,
    chunk_overlap: int = 180,
) -> list[str]:
    sections = analyze_document(clean_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [section.text for section in sections if section.text.strip()]


def analyze_document(
    clean_text: str,
    *,
    chunk_size: int = 1800,
    chunk_overlap: int = 180,
    llm_markers: list[dict] | None = None,
) -> list[DocumentSection]:
    if not clean_text.strip():
        return []

    if llm_markers:
        marker_sections = sections_from_markers(clean_text, llm_markers)
        if len(marker_sections) >= 2:
            return _split_oversized_sections(marker_sections, chunk_size=chunk_size)

    for splitter in (_split_by_heading_lines, _split_by_enumerated_paragraphs):
        sections = splitter(clean_text)
        if len(sections) >= 2:
            return _split_oversized_sections(sections, chunk_size=chunk_size)

    fallback = _fallback_paragraph_chunks(clean_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [
        _section(
            heading=_topic_from_text(chunk),
            level=1,
            text=chunk,
            parent_id=None,
            order_index=index,
            strategy="paragraph_fallback",
            confidence=0.45,
        )
        for index, chunk in enumerate(fallback)
    ]


def sections_from_markers(clean_text: str, markers: list[dict]) -> list[DocumentSection]:
    spans: list[tuple[int, int, str]] = []
    search_from = 0
    for marker in markers:
        title = str(marker.get("title") or marker.get("heading") or "").strip()
        start_quote = _clean_quote(str(marker.get("start_quote") or ""))
        end_quote = _clean_quote(str(marker.get("end_quote") or ""))
        if not start_quote:
            continue
        start = clean_text.find(start_quote, search_from)
        if start < 0:
            start = clean_text.find(start_quote)
        if start < 0:
            continue
        if end_quote:
            end_pos = clean_text.find(end_quote, start)
            end = end_pos + len(end_quote) if end_pos >= 0 else start + len(start_quote)
        else:
            end = start + len(start_quote)
        if end <= start:
            continue
        spans.append((start, end, title))
        search_from = end

    if len(spans) < 2:
        return []

    spans.sort(key=lambda item: item[0])
    sections: list[DocumentSection] = []
    for index, (start, end, title) in enumerate(spans):
        next_start = spans[index + 1][0] if index + 1 < len(spans) else len(clean_text)
        text = clean_text[start:max(end, next_start)].strip()
        if not text:
            continue
        sections.append(
            _section(
                heading=title or _topic_from_text(text),
                level=1,
                text=text,
                parent_id=None,
                order_index=len(sections),
                strategy="llm_marker",
                confidence=0.72,
            )
        )
    return sections


def _split_by_heading_lines(clean_text: str) -> list[DocumentSection]:
    lines = clean_text.splitlines()
    starts: list[tuple[int, int, str]] = []
    stack: list[tuple[int, str]] = []
    parent_by_line: dict[int, str | None] = {}

    for index, line in enumerate(lines):
        heading = _match_heading(line)
        if not heading:
            continue
        level, title = heading
        while stack and stack[-1][0] >= level:
            stack.pop()
        parent_by_line[index] = stack[-1][1] if stack else None
        section_id = str(uuid4())
        stack.append((level, section_id))
        starts.append((index, level, title))

    if len(starts) < 2:
        return []

    sections: list[DocumentSection] = []
    if starts[0][0] > 0:
        preface = "\n".join(lines[: starts[0][0]]).strip()
        if preface:
            sections.append(
                _section(
                    heading=_topic_from_text(preface),
                    level=1,
                    text=preface,
                    parent_id=None,
                    order_index=len(sections),
                    strategy="heading_preface",
                    confidence=0.62,
                )
            )
    for order, (line_index, level, title) in enumerate(starts):
        next_line = starts[order + 1][0] if order + 1 < len(starts) else len(lines)
        text = "\n".join(lines[line_index:next_line]).strip()
        if not text:
            continue
        body = "\n".join(lines[line_index + 1 : next_line]).strip()
        next_level = starts[order + 1][1] if order + 1 < len(starts) else 0
        if not body and next_level > level:
            continue
        sections.append(
            _section(
                heading=title,
                level=level,
                text=text,
                parent_id=parent_by_line.get(line_index),
                order_index=len(sections),
                strategy="heading",
                confidence=0.9,
            )
        )
    return sections


def _split_by_enumerated_paragraphs(clean_text: str) -> list[DocumentSection]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", clean_text) if part.strip()]
    if len(paragraphs) < 2:
        return []

    sections: list[str] = []
    current: list[str] = []
    saw_marker = False
    for paragraph in paragraphs:
        is_start = bool(ENUMERATED_START_RE.match(paragraph))
        if is_start:
            saw_marker = True
            if current:
                sections.append("\n\n".join(current).strip())
            current = [paragraph]
        else:
            if current:
                current.append(paragraph)
            else:
                current = [paragraph]
    if current:
        sections.append("\n\n".join(current).strip())

    if not saw_marker or len(sections) < 2:
        return []

    return [
        _section(
            heading=_topic_from_text(text),
            level=1,
            text=text,
            parent_id=None,
            order_index=index,
            strategy="enumerated_paragraph",
            confidence=0.78,
        )
        for index, text in enumerate(sections)
    ]


def _split_oversized_sections(sections: list[DocumentSection], *, chunk_size: int) -> list[DocumentSection]:
    if chunk_size <= 0:
        return sections
    result: list[DocumentSection] = []
    max_size = max(chunk_size, int(chunk_size * 1.35))
    for section in sections:
        if len(section.text) <= max_size:
            section.order_index = len(result)
            result.append(section)
            continue
        parts = _fallback_paragraph_chunks(section.text, chunk_size=chunk_size, chunk_overlap=0)
        for part_index, part in enumerate(parts):
            heading = section.heading if part_index == 0 else f"{section.heading}（续 {part_index + 1}）"
            result.append(
                _section(
                    heading=heading,
                    level=section.level,
                    text=part,
                    parent_id=section.parent_id,
                    order_index=len(result),
                    strategy=f"{section.strategy}_oversized",
                    confidence=min(section.confidence, 0.65),
                )
            )
    return result


def _fallback_paragraph_chunks(clean_text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    if len(clean_text) <= chunk_size:
        return [clean_text]

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", clean_text) if part.strip()]
    if not paragraphs:
        paragraphs = [clean_text]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        units = [paragraph] if len(paragraph) <= chunk_size else _split_long_text(paragraph, chunk_size)
        for unit in units:
            if current and current_len + len(unit) + 2 > chunk_size:
                chunks.append("\n\n".join(current).strip())
                current = [unit]
                current_len = len(unit)
            else:
                current.append(unit)
                current_len += len(unit) + (2 if current_len else 0)
    if current:
        chunks.append("\n\n".join(current).strip())

    if chunk_overlap <= 0 or len(chunks) <= 1:
        return chunks
    overlap = max(0, min(chunk_overlap, chunk_size // 4))
    enriched = [chunks[0]]
    for index in range(1, len(chunks)):
        tail = chunks[index - 1][-overlap:].strip()
        enriched.append(f"{tail}\n\n{chunks[index]}".strip() if tail else chunks[index])
    return enriched


def _split_long_text(text: str, chunk_size: int) -> list[str]:
    sentences = [item.strip() for item in SENTENCE_SPLIT_RE.split(text) if item.strip()]
    if not sentences:
        return _hard_split(text, chunk_size)
    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        if len(sentence) > chunk_size:
            if current:
                parts.append("".join(current).strip())
                current = []
                current_len = 0
            parts.extend(_hard_split(sentence, chunk_size))
            continue
        if current and current_len + len(sentence) > chunk_size:
            parts.append("".join(current).strip())
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += len(sentence)
    if current:
        parts.append("".join(current).strip())
    return parts


def _hard_split(text: str, chunk_size: int) -> list[str]:
    return [text[index : index + chunk_size].strip() for index in range(0, len(text), chunk_size) if text[index : index + chunk_size].strip()]


def _match_heading(line: str) -> tuple[int, str] | None:
    stripped = line.strip()
    if not stripped:
        return None
    markdown = MARKDOWN_HEADING_RE.match(stripped)
    if markdown:
        return len(markdown.group(1)), markdown.group(2).strip()
    if len(stripped) > 96:
        return None
    numbered = NUMBERED_HEADING_RE.match(stripped)
    if numbered:
        title = numbered.group(1).strip() if numbered.group(1) else stripped
        return 2, title or stripped
    return None


def _topic_from_text(text: str, *, max_length: int = 40) -> str:
    for line in text.splitlines():
        heading = _match_heading(line)
        if heading:
            return heading[1][:max_length]
        cleaned = line.strip(" #*-　\t")
        if cleaned:
            cleaned = re.sub(r"^[（(]?\d+[）).、]\s*", "", cleaned)
            cleaned = re.sub(r"^[一二三四五六七八九十百]+[、.．]\s*", "", cleaned)
            if len(cleaned) <= max_length:
                return cleaned
            sentence = re.split(r"[。！？!?；;，,]", cleaned, maxsplit=1)[0].strip()
            return (sentence or cleaned)[:max_length]
    return "未命名条目"


def _clean_quote(value: str) -> str:
    return " ".join(value.split()) if "\n" not in value else "\n".join(line.strip() for line in value.splitlines()).strip()


def _section(
    *,
    heading: str,
    level: int,
    text: str,
    parent_id: str | None,
    order_index: int,
    strategy: str,
    confidence: float,
) -> DocumentSection:
    return DocumentSection(
        section_id=str(uuid4()),
        heading=heading.strip() or _topic_from_text(text),
        level=level,
        text=text.strip(),
        parent_id=parent_id,
        order_index=order_index,
        strategy=strategy,
        confidence=confidence,
    )
