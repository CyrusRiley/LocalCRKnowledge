from __future__ import annotations

import os
from logging import Logger

from app.backend.llm.client import LLMError, QwenClient
from app.backend.models import SearchResult


def build_answer(
    question: str,
    results: list[SearchResult],
    *,
    llm_client: QwenClient,
    logger: Logger,
) -> str:
    if not results:
        return f"""# 回答主题

## 相关结论
知识库中没有检索到足够相关的内容。

## 相关知识条目
（空）

## 可直接用于写作的内容
材料不足，暂不生成。

## 可以进一步展开的方向
- 补充与问题“{question}”相关的笔记或文献摘录。

## 相关来源
（空）
"""

    if os.getenv("LK_DISABLE_LLM", "").lower() in {"1", "true", "yes"}:
        return fallback_answer(question, results)

    context = build_context(results)
    try:
        return llm_client.build_answer(question, context)
    except LLMError as exc:
        logger.error("Answer generation failed, falling back to extractive answer: %s", exc)
        return fallback_answer(question, results)


def build_context(results: list[SearchResult]) -> str:
    blocks = []
    for index, item in enumerate(results, start=1):
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

Markdown：
{item.markdown_content[:2000]}
"""
        )
    return "\n\n".join(blocks)


def fallback_answer(question: str, results: list[SearchResult]) -> str:
    conclusions = "\n".join(f"- {item.summary or item.snippet}" for item in results[:5])
    entries = "\n".join(f"- {item.title}：{item.summary or item.snippet}" for item in results[:8])
    sources = "\n".join(
        f"- {item.source_id} / {item.title} / {item.file_path or ''}" for item in results[:8]
    )
    writing = "\n\n".join((item.summary or item.snippet).strip() for item in results[:3] if (item.summary or item.snippet).strip())
    return f"""# {question}

## 相关结论
{conclusions or "材料不足。"}

## 相关知识条目
{entries or "（空）"}

## 可直接用于写作的内容
{writing or "材料不足，暂不生成。"}

## 可以进一步展开的方向
- 继续补充更具体的原始笔记。
- 对当前命中条目的主题标签进行人工校正。

## 相关来源
{sources or "（空）"}
"""
