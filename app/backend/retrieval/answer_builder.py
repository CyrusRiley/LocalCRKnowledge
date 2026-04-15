from __future__ import annotations

from dataclasses import dataclass
from logging import Logger

from app.backend.llm.client import QwenClient
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
            content=f"""# 回答主题

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
""",
            mode="extract",
        )

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
