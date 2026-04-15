from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    db_path: Path = PROJECT_ROOT / "data" / "db" / "knowledge.sqlite3"
    export_dir: Path = PROJECT_ROOT / "data" / "exports"
    log_dir: Path = PROJECT_ROOT / "data" / "logs"
    import_dir: Path = PROJECT_ROOT / "data" / "imports"
    llm_base_url: str = "http://127.0.0.1:8080/v1"
    llm_model: str = "qwen2.5"
    llm_timeout_seconds: int = 120
    chunk_size: int = 1800
    chunk_overlap: int = 180
    retrieval_limit: int = 15


def load_settings() -> Settings:
    return Settings(
        db_path=Path(os.getenv("LK_DB_PATH", Settings.db_path.as_posix())),
        export_dir=Path(os.getenv("LK_EXPORT_DIR", Settings.export_dir.as_posix())),
        log_dir=Path(os.getenv("LK_LOG_DIR", Settings.log_dir.as_posix())),
        import_dir=Path(os.getenv("LK_IMPORT_DIR", Settings.import_dir.as_posix())),
        llm_base_url=os.getenv("QWEN_BASE_URL", Settings.llm_base_url).rstrip("/"),
        llm_model=os.getenv("QWEN_MODEL", Settings.llm_model),
        llm_timeout_seconds=int(os.getenv("QWEN_TIMEOUT_SECONDS", Settings.llm_timeout_seconds)),
        chunk_size=int(os.getenv("LK_CHUNK_SIZE", Settings.chunk_size)),
        chunk_overlap=int(os.getenv("LK_CHUNK_OVERLAP", Settings.chunk_overlap)),
        retrieval_limit=int(os.getenv("LK_RETRIEVAL_LIMIT", Settings.retrieval_limit)),
    )
