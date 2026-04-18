from __future__ import annotations

import hashlib
import math
import re


class LocalHashEmbedder:
    """Small offline embedding fallback based on hashed word/character n-grams."""

    def __init__(self, *, dimensions: int = 384):
        self.dimensions = dimensions
        self.model_name = f"local-hash-ngram-{dimensions}"

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token, weight in _features(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            num = int.from_bytes(digest, "big", signed=False)
            index = num % self.dimensions
            sign = -1.0 if (num >> 9) & 1 else 1.0
            vector[index] += sign * weight
        return _normalize(vector)


def embed_text_hash(text: str) -> str:
    return hashlib.sha256(_normalize_text(text).encode("utf-8")).hexdigest()


def embedding_text(unit: dict) -> str:
    parts = [
        str(unit.get("title") or ""),
        str(unit.get("content") or ""),
        str(unit.get("evidence") or ""),
        str(unit.get("note_summary") or ""),
        str(unit.get("note_faithful_content") or ""),
        " ".join(unit.get("themes") or []),
        " ".join(unit.get("keywords") or []),
    ]
    return "\n".join(part for part in parts if part.strip())


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def _features(text: str) -> list[tuple[str, float]]:
    clean = _normalize_text(text)
    if not clean:
        return []
    features: list[tuple[str, float]] = []
    words = re.findall(r"[a-z][a-z0-9_-]{1,}|[0-9]+", clean)
    for word in words:
        features.append((f"w:{word}", 1.4))
    cjk_runs = re.findall(r"[\u4e00-\u9fff]+", clean)
    for run in cjk_runs:
        if len(run) <= 2:
            features.append((f"c:{run}", 1.2))
            continue
        for size, weight in ((2, 1.0), (3, 1.25), (4, 1.15)):
            for start in range(0, len(run) - size + 1):
                piece = run[start : start + size]
                if _useful_piece(piece):
                    features.append((f"c:{piece}", weight))
    return features


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _useful_piece(piece: str) -> bool:
    return not all(ch in "的是了和与有在中对及或" for ch in piece)


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return vector
    return [value / norm for value in vector]
