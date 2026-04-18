from __future__ import annotations

import re
import os
from dataclasses import dataclass
from logging import Logger

from app.backend.llm.client import LLMError, QwenClient
from app.backend.models import SearchResult


@dataclass(frozen=True)
class AnswerBuildResult:
    content: str
    mode: str


def build_answer(
    question: str,
    results: list[SearchResult],
    *,
    llm_client: QwenClient,
    logger: Logger,
) -> str:
    return build_answer_result(question, results, llm_client=llm_client, logger=logger).content


def build_answer_result(
    question: str,
    results: list[SearchResult],
    *,
    llm_client: QwenClient,
    logger: Logger,
) -> AnswerBuildResult:
    # The local model's synthesized answers are often less faithful than direct
    # extractive summaries for this note-taking workflow, so answers always use
    # the stable extract mode. The llm_client/logger parameters stay in the
    # signature to keep the API boundary compatible with earlier callers.
    _ = (llm_client, logger)
    if not results:
        return AnswerBuildResult(
            content=f"""# {question}

## 简明回答
知识库中没有检索到足够相关的内容。

## 主要相关材料
（空）

## 材料不足或需要核对的地方
当前问题没有命中本地知识库中的明确材料。可以补充与“{question}”直接相关的笔记、文献摘录或论文修改意见后再检索。
""",
            mode="extract",
        )

    if os.getenv("LK_DISABLE_LLM", "").lower() not in {"1", "true", "yes"}:
        try:
            evidence = build_synthesis_evidence(results)
            synthesized = _clean_synthesis(llm_client.build_constrained_answer(question, evidence))
            if _valid_synthesis(synthesized):
                return AnswerBuildResult(content=model_supported_answer(question, results, synthesized), mode="model")
            logger.warning("Constrained answer was too short or empty; falling back to extract mode")
        except (LLMError, TimeoutError, OSError) as exc:
            logger.warning("Constrained answer generation failed, falling back to extract mode: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unexpected constrained answer error, falling back to extract mode: %s", exc)

    return AnswerBuildResult(content=fallback_answer(question, results), mode="extract")


def build_context(
    results: list[SearchResult],
    *,
    max_items: int = 8,
    markdown_chars: int = 0,
    total_chars: int = 3200,
) -> str:
    blocks = []
    used_chars = 0
    for index, item in enumerate(results[:max_items], start=1):
        markdown = (item.markdown_content or "")[:markdown_chars] if markdown_chars > 0 else ""
        markdown_block = f"\nMarkdown：\n{markdown}\n" if markdown else ""
        blocks.append(
            f"""## 条目 {index}: {item.title}
- note_id: {item.note_id}
- source_id: {item.source_id}
- note_type: {item.note_type}
- themes: {", ".join(item.themes)}
- keywords: {", ".join(item.keywords)}
- file_path: {item.file_path or ""}
- imported_at: {item.imported_at or ""}

摘要：
{item.summary}

匹配片段：
{item.snippet}
{markdown_block}"""
        )
        used_chars += len(blocks[-1])
        if used_chars >= total_chars:
            break
    return "\n\n".join(blocks)


def fallback_answer(question: str, results: list[SearchResult]) -> str:
    ranked, high, medium, supplemental = _ranked_groups(results)
    brief = _brief_answer(question, high, medium)
    return _build_answer_document(
        question,
        leading_heading="简明回答",
        leading_content=brief,
        ranked=ranked,
        high=high,
        medium=medium,
        supplemental=supplemental,
    )


def model_supported_answer(question: str, results: list[SearchResult], synthesized: str) -> str:
    ranked, high, medium, supplemental = _ranked_groups(results)
    return _build_answer_document(
        question,
        leading_heading="综合回答",
        leading_content=synthesized,
        ranked=ranked,
        high=high,
        medium=medium,
        supplemental=supplemental,
    )


def _build_answer_document(
    question: str,
    *,
    leading_heading: str,
    leading_content: str,
    ranked: list[SearchResult],
    high: list[SearchResult],
    medium: list[SearchResult],
    supplemental: list[SearchResult],
) -> str:
    material_sections = _material_sections(high, medium, supplemental)
    relation_section = _relation_section(ranked)
    theme_section = _theme_section(ranked)
    writing = _writing_section(high, medium)
    gaps = _gap_section(question, ranked)
    return f"""# {question}

## {leading_heading}
{leading_content}

## 主要相关材料
{material_sections}

## 关系链条与上下文
{relation_section}

## 按主题归纳
{theme_section}

## 可直接用于写作的段落
{writing or "材料不足，暂不生成。"}

## 材料不足或需要核对的地方
{gaps}
"""


def build_synthesis_evidence(results: list[SearchResult], *, max_items: int = 8, total_chars: int = 5200) -> str:
    ranked, high, medium, supplemental = _ranked_groups(results)
    selected = _dedupe_items([*high[:5], *medium[:2], *supplemental[:1]])[:max_items]
    blocks: list[str] = []
    used = 0
    for index, item in enumerate(selected, start=1):
        content = _clean_text(_primary_content(item), limit=760 if item.relevance_level == "高相关" else 420)
        evidence = _clean_text(item.source_excerpt or item.snippet, limit=220)
        block = f"""## 证据 {index}: {item.title}
检索说明（仅用于判断材料重要性，不属于研究事实）:
- 相关等级: {item.relevance_level or "相关"}
- 纳入原因: {item.relevance_reason or _fallback_reason(item)}
- 关系说明: {_relation_evidence_label(item)}

主题与关键词:
- 主题: {"、".join(item.themes[:4])}
- 关键词: {"、".join(item.keywords[:6])}

保真内容:
{content}

证据摘录:
{evidence}
"""
        used += len(block)
        if used > total_chars and blocks:
            break
        blocks.append(block)
    return "\n\n".join(blocks)


def _ranked_groups(results: list[SearchResult]) -> tuple[list[SearchResult], list[SearchResult], list[SearchResult], list[SearchResult]]:
    ranked = sorted(
        results,
        key=lambda item: (
            _level_rank(item.relevance_level),
            -float(item.relevance_score or 0.0),
            item.title,
        ),
    )
    high = [item for item in ranked if item.relevance_level == "高相关"] or ranked[: min(3, len(ranked))]
    medium = [item for item in ranked if item.relevance_level == "中相关"]
    supplemental = [item for item in ranked if item.relevance_level == "补充相关"]
    return ranked, high, medium, supplemental


def _brief_answer(question: str, high: list[SearchResult], medium: list[SearchResult]) -> str:
    focus = high[:3]
    if not focus:
        return "当前知识库只检索到零散补充材料，尚不足以形成稳定回答。"
    titles = "、".join(item.title for item in focus if item.title)
    first = _clean_text(_primary_content(focus[0]), limit=360)
    additions = [_clean_text(_primary_content(item), limit=180) for item in [*focus[1:3], *medium[:1]]]
    additions = [item for item in additions if item and item != first]
    if additions:
        return (
            f"围绕“{question}”，当前最相关的材料集中在：{titles}。"
            f"{first}\n\n补充来看，{' '.join(additions[:2])}"
        )
    return f"围绕“{question}”，当前最相关的材料集中在：{titles}。{first}"


def _material_sections(
    high: list[SearchResult],
    medium: list[SearchResult],
    supplemental: list[SearchResult],
) -> str:
    parts: list[str] = []
    parts.append("### 高相关\n" + _material_list(high[:5], detail_chars=520))
    if medium:
        parts.append("### 中相关\n" + _material_list(medium[:5], detail_chars=320))
    if supplemental:
        parts.append("### 补充相关\n" + _material_list(supplemental[:5], detail_chars=220))
    return "\n\n".join(parts)


def _material_list(items: list[SearchResult], *, detail_chars: int) -> str:
    if not items:
        return "（空）"
    blocks: list[str] = []
    for index, item in enumerate(items, start=1):
        reason = item.relevance_reason or _fallback_reason(item)
        detail = _clean_text(_primary_content(item), limit=detail_chars)
        evidence = _clean_text(item.source_excerpt or item.snippet, limit=180)
        relation = ""
        if item.relation_type:
            relation = f"\n   - 关系类型：{item.relation_type}（{item.relation_strength or 'unknown'}）"
        blocks.append(
            f"{index}. **{item.title or '未命名知识'}**（{item.relevance_level or '相关'}，相关度 {float(item.relevance_score or 0.0):.3f}）\n"
            f"   - 为什么相关：{reason}{relation}\n"
            f"   - 详细内容：{detail or '（无详细内容）'}\n"
            f"   - 证据摘录：{evidence or '（无摘录）'}"
        )
    return "\n".join(blocks)


def _relation_section(items: list[SearchResult]) -> str:
    relation_items = [item for item in items if item.match_source in {"relation", "group"} or item.relation_type]
    if not relation_items:
        return "本次命中主要来自文本、标题或关键词检索，未发现需要额外说明的强关系扩展。"
    lines = []
    for item in relation_items[:8]:
        if item.relation_type:
            lines.append(
                f"- **{item.title}**：由知识关系网纳入，关系类型为 `{item.relation_type}`，强度为 `{item.relation_strength or 'unknown'}`。"
            )
        elif item.match_source == "group":
            lines.append(f"- **{item.title}**：与直接命中的知识属于同一知识组或相邻片段，用于补足上下文。")
    return "\n".join(lines) or "未发现需要额外说明的关系链条。"


def _theme_section(items: list[SearchResult]) -> str:
    grouped: dict[str, list[SearchResult]] = {}
    for item in items:
        keys = item.themes or item.keywords or ["未分类主题"]
        for key in keys[:2]:
            grouped.setdefault(key, []).append(item)
    if not grouped:
        return "（空）"
    parts: list[str] = []
    for theme, theme_items in list(grouped.items())[:5]:
        unique = _dedupe_items(theme_items)[:4]
        joined = " ".join(_clean_text(_primary_content(item), limit=140) for item in unique)
        titles = "、".join(item.title for item in unique if item.title)
        parts.append(f"### {theme}\n相关条目：{titles or '未命名条目'}。\n{joined or '暂无可归纳内容。'}")
    return "\n\n".join(parts)


def _writing_section(high: list[SearchResult], medium: list[SearchResult]) -> str:
    candidates = _dedupe_items([*high[:3], *medium[:2]])
    if not candidates:
        return ""
    sentences = []
    for item in candidates:
        text = _clean_text(_primary_content(item), limit=260)
        if text:
            sentences.append(text)
    if not sentences:
        return ""
    return (
        "可以将现有材料组织为如下表述：\n\n"
        + " ".join(sentences)
    )


def _gap_section(question: str, items: list[SearchResult]) -> str:
    if len(items) < 3:
        return f"当前只命中 {len(items)} 条材料，回答可能覆盖不全。建议补充与“{question}”直接相关的原始笔记后再整理知识库。"
    relation_count = sum(1 for item in items if item.match_source == "relation" or item.relation_type)
    if relation_count == 0:
        return "本次回答主要依赖文本和关键词命中，尚未充分利用到强语义关系；可以在知识库页执行“全局整理”后再次检索。"
    return "以上回答仅基于当前已入库材料；如果需要用于正式写作，仍建议回到原始笔记或文献摘录中核对表述。"


def _primary_content(item: SearchResult) -> str:
    section = _markdown_section(item.markdown_content, "保真整理")
    if section:
        return section
    section = _markdown_section(item.markdown_content, "摘要")
    if section:
        return section
    return item.summary or item.source_excerpt or item.snippet or item.markdown_content


def _markdown_section(markdown: str, heading: str) -> str:
    if not markdown:
        return ""
    pattern = rf"(?ms)^##\s+{re.escape(heading)}\s*\n(.*?)(?=^##\s+|\Z)"
    match = re.search(pattern, markdown)
    return match.group(1).strip() if match else ""


def _clean_text(text: str, *, limit: int) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    clean = re.sub(r"^[-*]\s*", "", clean)
    if len(clean) <= limit:
        return clean
    clipped = clean[:limit].rstrip()
    sentence_end = max(clipped.rfind("。"), clipped.rfind("."), clipped.rfind("；"), clipped.rfind(";"))
    if sentence_end >= max(60, limit // 2):
        return clipped[: sentence_end + 1]
    return clipped + "..."


def _clean_synthesis(text: str) -> str:
    clean = str(text or "").strip()
    clean = re.sub(r"^```(?:markdown|md)?\s*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$", "", clean)
    clean = re.sub(r"^#+\s*综合回答\s*", "", clean, flags=re.IGNORECASE)
    clean = clean.strip()
    paragraphs = [re.sub(r"\s+", " ", part).strip() for part in re.split(r"\n\s*\n+", clean) if part.strip()]
    return "\n\n".join(paragraphs[:4])


def _valid_synthesis(text: str) -> bool:
    clean = re.sub(r"\s+", "", text or "")
    if len(clean) < 80:
        return False
    invalid_markers = (
        "材料不足",
        "无法回答",
        "没有足够",
        "未提供",
        "用户没有提供具体问题",
        "没有提供具体问题",
        "questionmarks",
        "typedaseriesofquestionmarks",
    )
    lowered = clean.lower()
    if any(marker in lowered for marker in invalid_markers):
        return False
    return True


def _level_rank(level: str) -> int:
    return {"高相关": 0, "中相关": 1, "补充相关": 2}.get(level, 1)


def _fallback_reason(item: SearchResult) -> str:
    if item.match_source == "relation":
        return f"通过知识关系网纳入：{item.relation_type or 'related'}"
    if item.match_source == "group":
        return "同一知识组或相邻片段补充"
    if item.match_source == "vector":
        return "通过向量语义检索纳入"
    if item.keywords:
        return "与关键词“" + "、".join(item.keywords[:3]) + "”相关"
    return "与问题存在文本匹配"


def _relation_evidence_label(item: SearchResult) -> str:
    if item.relation_type:
        strength = {"strong": "强", "medium": "中等", "weak": "弱"}.get(item.relation_strength, item.relation_strength or "未知")
        return f"由知识关系网纳入，关系为 {item.relation_type}，关系强度为{strength}"
    if item.match_source == "group":
        return "由同一知识组或相邻片段纳入，用于补足上下文"
    if item.match_source == "vector":
        return "由向量语义检索纳入，用于补充表达不同但语义相近的材料"
    return "直接检索命中"


def _dedupe_items(items: list[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    result: list[SearchResult] = []
    for item in items:
        if item.note_id not in seen:
            result.append(item)
            seen.add(item.note_id)
    return result
