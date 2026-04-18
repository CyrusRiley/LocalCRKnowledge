from __future__ import annotations

from app.backend.database.repository import KnowledgeRepository
from app.backend.embeddings.local_embedder import embed_text_hash, embedding_text
from app.backend.embeddings.providers import EmbeddingModelConfig, Embedder, active_embedding_models, get_embedder
from app.backend.utils.time_utils import utc_now_iso


def index_unit_embedding(
    repo: KnowledgeRepository,
    unit: dict,
    *,
    config: EmbeddingModelConfig | None = None,
    embedder: Embedder | None = None,
) -> bool:
    config = config or active_embedding_models()[0]
    embedder = embedder or get_embedder(config)
    text = embedding_text(unit)
    if not text.strip():
        return False
    text_hash = embed_text_hash(text)
    if unit.get("embedding_model") == embedder.model_name and unit.get("embedding_text_hash") == text_hash:
        return False
    now = utc_now_iso()
    repo.upsert_unit_embedding(
        unit_id=str(unit["unit_id"]),
        note_id=str(unit.get("note_id") or "") or None,
        embedding_model=embedder.model_name,
        text_hash=text_hash,
        vector=embedder.embed(text),
        now=now,
    )
    return True


def index_unit_embeddings(
    repo: KnowledgeRepository,
    unit: dict,
    *,
    configs: list[EmbeddingModelConfig] | None = None,
) -> dict:
    configs = configs or active_embedding_models()
    stats = {"updated": 0, "skipped": 0, "failed": 0, "models": [], "errors": []}
    for config in configs:
        try:
            embedder = get_embedder(config)
            changed = index_unit_embedding(repo, unit, config=config, embedder=embedder)
            stats["models"].append(config.model_name)
            stats["updated" if changed else "skipped"] += 1
        except Exception as exc:  # noqa: BLE001
            stats["failed"] += 1
            _remember_error(stats, f"{config.key}/{config.model_name}: {_error_message(exc)}")
    return stats


def rebuild_unit_embeddings(
    repo: KnowledgeRepository,
    *,
    limit: int = 10000,
    configs: list[EmbeddingModelConfig] | None = None,
) -> dict:
    configs = configs or active_embedding_models()
    updated = 0
    skipped = 0
    failed = 0
    total_units = 0
    models: list[str] = []
    errors: list[str] = []
    for config in configs:
        models.append(config.model_name)
        units = repo.list_units_for_embedding(limit=limit, model_name=config.model_name)
        total_units = max(total_units, len(units))
        try:
            embedder = get_embedder(config)
        except Exception as exc:  # noqa: BLE001
            failed += len(units)
            _remember_error({"errors": errors}, f"{config.key}/{config.model_name}: {_error_message(exc)}")
            continue
        for unit in units:
            try:
                if index_unit_embedding(repo, unit, config=config, embedder=embedder):
                    updated += 1
                else:
                    skipped += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                _remember_error(
                    {"errors": errors},
                    f"{config.key}/{config.model_name} unit={unit.get('unit_id')}: {_error_message(exc)}",
                )
    return {"models": models, "updated": updated, "skipped": skipped, "failed": failed, "total": total_units, "errors": errors}


def _error_message(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _remember_error(stats: dict, message: str, *, limit: int = 8) -> None:
    errors = stats.setdefault("errors", [])
    if len(errors) < limit:
        errors.append(message)
