from __future__ import annotations


NOTE_JSON_FIELDS = """{
  "title": "",
  "note_type": "",
  "themes": [],
  "summary": "",
  "faithful_content": "",
  "key_points": [],
  "usage_scenarios": [],
  "user_insights": "",
  "keywords": [],
  "source_excerpt": ""
}"""


def structure_detect_prompt(clean_text: str, *, max_units: int = 16) -> list[dict[str, str]]:
    system = (
        "你是科研笔记的结构识别器。你的任务不是总结，也不是改写，"
        "只识别原文中相对完整的主题单元边界，并返回 JSON 数组。"
    )
    user = f"""
请识别下面文本中可以拆成知识条目的主题单元。

输出要求：
1. 只能返回 JSON 数组，不要 Markdown，不要解释。
2. 最多返回 {max_units} 个单元。
3. 每个元素必须包含 title、start_quote、end_quote、reason。
4. title 是主题短语，不要直接照抄第一句话，长度建议 6-24 个汉字。
5. start_quote 和 end_quote 必须是原文中真实连续出现的短摘录，用于程序定位边界。
6. 如果原文有“七个要点/若干阶段/多个小标题”，应按这些要点或小标题拆分。

JSON 示例：
[
  {{"title": "研究问题定位", "start_quote": "原文开头短摘录", "end_quote": "原文结尾短摘录", "reason": "该段集中讨论一个问题"}}
]

原文：
{clean_text}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


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
        "你是一个本地科研笔记整理器。只把用户给出的当前知识单元整理成稳定 JSON，"
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
5. title 必须是本知识单元的主题短语，不能直接使用第一句话；建议 6-24 个汉字。
6. keywords 优先使用简短、稳定、上位的概念词，例如“行为模拟”优先于“行为模拟技术”。
7. key_points 要合并相近表述，避免把同义句拆成重复要点。
8. 不确定的字段用空字符串或空数组，不要编造。
9. faithful_content 要保留当前知识单元的主要信息，用 1-3 段完整句子整理，不要只写短词或标签；它应比 summary 更完整，但不能超出当前片段原文含义。

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


def constrained_answer_prompt(question: str, evidence_markdown: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你是本地科研知识库的综合回答写作者。你的任务只是在给定证据内组织语言，"
                "不能引入外部知识，不能补充证据中没有的信息。"
            ),
        },
        {
            "role": "user",
            "content": f"""
请只基于下面的证据包，为用户问题生成 2-4 段连贯中文回答。

你必须回答的用户问题是：
{question}

证据包：
{evidence_markdown}

输出要求：
1. 只输出正文段落，不要标题，不要 Markdown 表格。
2. 必须综合高相关材料，不要逐条机械罗列。
3. 可以使用“首先、其次、此外、总体来看”等连接方式。
4. 每个关键判断都必须能在证据包中找到依据。
5. 不要写“根据材料1/材料2”这类编号引用。
6. 用户问题已经在上方给出，不能说“用户没有提供具体问题”，也不能要求用户重新提问。
7. 只要证据包中有“保真内容”或“证据摘录”，就必须基于这些内容给出回答；只有证据包为空或完全无关时，才说明材料不足。
8. 证据包里的“检索说明/纳入原因/关系说明”只用于判断材料重要性，不是研究事实，不能把相关分数、关系类型、关系强度当作论文内容。
9. 字数建议 350-800 字。

现在请直接回答这个问题：{question}
""".strip(),
        },
    ]
