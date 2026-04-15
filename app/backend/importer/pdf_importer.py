from __future__ import annotations

from pathlib import Path

from app.backend.importer.text_importer import build_source_record
from app.backend.models import SourceRecord


def build_pdf_source(file_path: Path) -> SourceRecord:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PDF parsing requires optional dependency: pypdf") from exc

    reader = PdfReader(str(file_path))
    page_text: list[str] = []
    for index, page in enumerate(reader.pages):
        try:
            page_text.append(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"PDF text extraction failed at page {index + 1}: {exc}") from exc

    raw_text = "\n\n".join(part.strip() for part in page_text if part.strip())
    if not raw_text.strip():
        raise RuntimeError("PDF text extraction returned no text; paste manual excerpts instead.")
    return build_source_record(raw_text=raw_text, source_type="pdf", file_path=file_path)

