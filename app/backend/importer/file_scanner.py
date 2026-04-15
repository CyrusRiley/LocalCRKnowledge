from __future__ import annotations

from pathlib import Path

from app.backend.importer.text_importer import SUPPORTED_TEXT_EXTENSIONS


def scan_text_files(directory: Path, *, recursive: bool = True) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(str(directory))

    pattern = "**/*" if recursive else "*"
    files = [
        path
        for path in directory.glob(pattern)
        if path.is_file() and path.suffix.lower() in SUPPORTED_TEXT_EXTENSIONS
    ]
    return sorted(files, key=lambda item: str(item).lower())

