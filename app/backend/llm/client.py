from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass

from app.backend.llm import prompts
from app.backend.llm.parser import parse_note_json, parse_question_json, parse_section_markers_json


class LLMError(RuntimeError):
    pass


@dataclass
class QwenClient:
    base_url: str
    model: str
    timeout_seconds: int = 120

    def organize_text(
        self,
        clean_text: str,
        *,
        direction: str,
        title_candidate: str,
        pre_keywords: list[str],
        prev_context: str = "",
        next_context: str = "",
        chunk_index: int | None = None,
        chunk_count: int | None = None,
    ) -> dict:
        content = self.chat(
            prompts.organize_prompt(
                clean_text,
                direction,
                title_candidate,
                pre_keywords,
                prev_context=prev_context,
                next_context=next_context,
                chunk_index=chunk_index,
                chunk_count=chunk_count,
            ),
            temperature=0.2,
            max_tokens=_env_int("QWEN_ORGANIZE_MAX_TOKENS", 1200),
        )
        return parse_note_json(content, fallback_title=title_candidate, source_text=clean_text)

    def parse_question(self, question: str) -> dict:
        content = self.chat(
            prompts.question_parse_prompt(question),
            temperature=0.1,
            max_tokens=_env_int("QWEN_PARSE_MAX_TOKENS", 256),
            timeout_seconds=_env_int("QWEN_PARSE_TIMEOUT_SECONDS", min(self.timeout_seconds, 20)),
        )
        return parse_question_json(content, question)

    def detect_structure(self, clean_text: str, *, max_units: int = 16) -> list[dict[str, str]]:
        content = self.chat(
            prompts.structure_detect_prompt(clean_text, max_units=max_units),
            temperature=0.0,
            max_tokens=_env_int("QWEN_STRUCTURE_MAX_TOKENS", 900),
        )
        return parse_section_markers_json(content)

    def build_answer(self, question: str, context_markdown: str) -> str:
        return self.chat(
            prompts.answer_prompt(question, context_markdown),
            temperature=0.2,
            max_tokens=_env_int("QWEN_ANSWER_MAX_TOKENS", 1200),
            timeout_seconds=_env_int("QWEN_ANSWER_TIMEOUT_SECONDS", min(self.timeout_seconds, 75)),
        )

    def build_constrained_answer(self, question: str, evidence_markdown: str) -> str:
        return self.chat(
            prompts.constrained_answer_prompt(question, evidence_markdown),
            temperature=0.15,
            max_tokens=_env_int("QWEN_SYNTHESIS_MAX_TOKENS", 800),
            timeout_seconds=_env_int("QWEN_SYNTHESIS_TIMEOUT_SECONDS", _env_int("QWEN_ANSWER_TIMEOUT_SECONDS", min(self.timeout_seconds, 75))),
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        timeout_seconds: int | None = None,
    ) -> str:
        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None and max_tokens > 0:
            payload["max_tokens"] = max_tokens
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds or self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

        try:
            data = json.loads(raw)
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise LLMError(f"Unexpected LLM response: {raw[:500]}") from exc


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default
