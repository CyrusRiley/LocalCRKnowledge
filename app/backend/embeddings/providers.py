from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.backend.embeddings.local_embedder import LocalHashEmbedder


class Embedder(Protocol):
    model_name: str

    def embed(self, text: str) -> list[float]:
        ...


@dataclass(frozen=True)
class EmbeddingModelConfig:
    key: str
    provider: str
    model_path: str
    model_name: str
    weight: float = 1.0


_EMBEDDER_CACHE: dict[str, Embedder] = {}


def configured_embedding_models() -> list[EmbeddingModelConfig]:
    raw = os.getenv(
        "LK_EMBEDDING_MODELS",
        "zh=sentence_transformers,BAAI/bge-base-zh-v1.5,0.55;"
        "multi=sentence_transformers,BAAI/bge-m3,0.45;"
        "local=local,local-hash-ngram-384,0.25",
    )
    configs: list[EmbeddingModelConfig] = []
    for part in [item.strip() for item in raw.split(";") if item.strip()]:
        if "=" not in part:
            continue
        key, spec = part.split("=", 1)
        pieces = [item.strip() for item in spec.split(",", 2)]
        provider = pieces[0] if pieces else "local"
        model_path = pieces[1] if len(pieces) > 1 else provider
        weight = _float(pieces[2], 1.0) if len(pieces) > 2 else 1.0
        model_name = _model_name(key.strip(), provider, model_path)
        configs.append(EmbeddingModelConfig(key=key.strip(), provider=provider, model_path=model_path, model_name=model_name, weight=weight))
    if not configs:
        configs.append(EmbeddingModelConfig("local", "local", "local-hash-ngram-384", "local-hash-ngram-384", 1.0))
    return configs


def active_embedding_models(query: str | None = None) -> list[EmbeddingModelConfig]:
    configs = configured_embedding_models()
    by_key = {item.key: item for item in configs}
    active = os.getenv("LK_EMBEDDING_ACTIVE", "local").strip()
    if active.lower() == "auto":
        keys = ["zh", "multi", "local"] if _mostly_cjk(query or "") else ["multi", "zh", "local"]
    else:
        keys = [item.strip() for item in re.split(r"[,;]", active) if item.strip()]
    selected = [by_key[key] for key in keys if key in by_key]
    if not selected and "local" in by_key:
        selected = [by_key["local"]]
    return selected or configs[:1]


def get_embedder(config: EmbeddingModelConfig) -> Embedder:
    cached = _EMBEDDER_CACHE.get(config.model_name)
    if cached:
        return cached
    if config.provider in {"local", "hash", "local_hash"}:
        embedder: Embedder = LocalHashEmbedder()
    elif config.provider in {"sentence_transformers", "sentence-transformers", "st"}:
        embedder = SentenceTransformerEmbedder(config.model_path, model_name=config.model_name)
    else:
        raise RuntimeError(f"Unsupported embedding provider: {config.provider}")
    _EMBEDDER_CACHE[config.model_name] = embedder
    return embedder


class SentenceTransformerEmbedder:
    def __init__(self, model_path: str, *, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("sentence-transformers is not installed") from exc
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_path, local_files_only=True)
        except TypeError:
            self.model = SentenceTransformer(model_path)

    def embed(self, text: str) -> list[float]:
        vector = self.model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
        return [float(value) for value in vector.tolist()]


def _model_name(key: str, provider: str, model_path: str) -> str:
    if provider in {"local", "hash", "local_hash"}:
        return "local-hash-ngram-384"
    name = Path(model_path).name or model_path
    return f"{provider}:{key}:{name}"


def _mostly_cjk(text: str) -> bool:
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return True
    cjk = sum(1 for ch in chars if "\u4e00" <= ch <= "\u9fff")
    return cjk / max(1, len(chars)) >= 0.25


def _float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
