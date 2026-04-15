from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from app.backend.models import SourceRecord
from app.backend.utils.hash_utils import sha256_text
from app.backend.utils.time_utils import utc_now_iso


SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md"}


def build_manual_source(raw_text: str) -> SourceRecord:
    return build_source_record(raw_text=raw_text, source_type="manual", file_path=None)


def build_file_source(file_path: Path) -> SourceRecord:
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_TEXT_EXTENSIONS:
        raise ValueError(f"Unsupported text file extension: {suffix}")
    raw_text = file_path.read_text(encoding="utf-8")
    return build_source_record(raw_text=raw_text, source_type=suffix.lstrip("."), file_path=file_path)


def build_source_record(raw_text: str, source_type: str, file_path: Path | None) -> SourceRecord:
    now = utc_now_iso()
    return SourceRecord(
        source_id=str(uuid4()),
        source_type=source_type,
        file_path=str(file_path.resolve()) if file_path else None,
        raw_text=raw_text,
        clean_text="",
        text_hash=sha256_text(raw_text),
        created_at=now,
        imported_at=now,
        status="new",
    )

