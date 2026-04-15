from __future__ import annotations

import re
from pathlib import Path


def export_markdown(content: str, export_dir: Path, filename: str, *, overwrite: bool = False) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    safe_name = safe_filename(filename)
    if not safe_name.endswith(".md"):
        safe_name += ".md"
    target = export_dir / safe_name
    if target.exists() and not overwrite:
        target = _next_available(target)
    target.write_text(content, encoding="utf-8")
    return target


def safe_filename(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", name).strip(" ._")
    return cleaned[:120] or "export"


def _next_available(path: Path) -> Path:
    stem = path.stem
    suffix = path.suffix
    for index in range(1, 1000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Cannot find available filename for {path}")

