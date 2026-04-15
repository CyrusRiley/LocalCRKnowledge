from __future__ import annotations


NOTE_JSON_FIELDS = """{
  "title": "",
  "note_type": "",
  "themes": [],
  "summary": "",
  "key_points": [],
  "usage_scenarios": [],
  "user_insights": "",
  "keywords": [],
  "source_excerpt": ""
}"""


def organize_prompt(
    clean_text: str,
    direction: str,
    title_candidate: str,
    pre_keywords: list[str],
    *,
    prev_context: str = "",
    next_context: str = "",
    chunk_index: int | None = None,
    chunk_count: int | None = None,
) -> list[dict[str, str]]:
    system = (
        "你是一个本地科研笔记整理器。只把用户给出的原文整理成稳定 JSON，"
        "不要编造来源，不要输出 Markdown，不要输出解释。"
    )
    chunk_meta = ""
    if chunk_index is not None and chunk_count is not None:
        chunk_meta = f"\n片段位置：第 {chunk_index + 1}/{chunk_count} 段"
    context_block = ""
    if prev_context or next_context:
        context_block = f"""
上下文参考（只用于理解连续性，不可当作本段原文直接摘录）：
- 上文尾部参考：{prev_context or "（无）"}
- 下文开头参考：{next_context or "（无）"}
""".strip()
    user = f"""
请按固定 JSON 结构整理下面的科研笔记。

整理方向：{direction or "提炼为可用于写作的知识摘要"}
标题候选：{title_candidate}
初步关键词：{", ".join(pre_keywords)}
{chunk_meta}

输出要求：
1. 只能返回一个 JSON 对象。
2. 字段必须完整，字段名必须与模板一致。
3. themes、key_points、usage_scenarios、keywords 必须是字符串数组。
4. source_excerpt 必须来自“当前片段原文”，长度控制在 300 字以内。
5. 不确定的字段用空字符串或空数组，不要编造。

JSON 模板：
{NOTE_JSON_FIELDS}

{context_block}

当前片段原文：
{clean_text}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def question_parse_prompt(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "你是本地知识库检索助手。只返回 JSON，不要解释。",
        },
        {
            "role": "user",
            "content": f"""
请解析用户问题，生成检索用 JSON。

字段：
{{
  "keywords": [],
  "themes": [],
  "note_type": "",
  "intent": ""
}}

用户问题：{question}
""".strip(),
        },
    ]


def answer_prompt(question: str, context_markdown: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你是本地科研知识库回答生成器。必须只基于给定检索结果回答，"
                "不要引入外部知识。输出 Markdown。"
            ),
        },
        {
            "role": "user",
            "content": f"""
用户问题：
{question}

检索结果：
{context_markdown}

请按以下 Markdown 模板回答：

# 回答主题

## 相关结论

## 相关知识条目

## 可直接用于写作的内容

## 可以进一步展开的方向

## 相关来源

要求：
1. 必须基于检索结果，不要自由发挥。
2. 相关来源中写出 source_id / title / file_path。
3. 如果材料不足，明确说明不足。
""".strip(),
        },
    ]
