from __future__ import annotations

from app.backend.database.repository import KnowledgeRepository
from app.backend.models import ChunkRecord, StructuredNote


def index_note(repo: KnowledgeRepository, note: StructuredNote, chunks: list[ChunkRecord]) -> None:
    repo.insert_note(note, chunks)

