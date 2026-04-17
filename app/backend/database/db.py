from __future__ import annotations

import sqlite3
import os
from pathlib import Path


def connect(db_path: Path | str) -> sqlite3.Connection:
    if str(db_path) != ":memory:":
        db_path = Path(db_path)
        db_path = _usable_db_path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(f"PRAGMA journal_mode = {_journal_mode()}")
    conn.execute("PRAGMA temp_store = MEMORY")
    return conn


def init_db(conn: sqlite3.Connection, schema_path: Path | None = None) -> None:
    if schema_path is None:
        schema_path = Path(__file__).with_name("schema.sql")
    conn.executescript(schema_path.read_text(encoding="utf-8"))
    _run_migrations(conn)
    conn.commit()


def _run_migrations(conn: sqlite3.Connection) -> None:
    _ensure_column(conn, "notes_structured", "faithful_content", "TEXT")
    _ensure_column(conn, "knowledge_relations", "relation_layer", "TEXT NOT NULL DEFAULT 'semantic'")
    _ensure_column(conn, "knowledge_relations", "relation_strength", "TEXT NOT NULL DEFAULT 'medium'")
    _ensure_column(conn, "knowledge_relations", "display_default", "INTEGER NOT NULL DEFAULT 1")


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    if any(row["name"] == column for row in rows):
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _usable_db_path(db_path: Path) -> Path:
    journal_path = db_path.with_name(f"{db_path.name}-journal")
    if db_path.exists() and db_path.stat().st_size == 0 and journal_path.exists():
        try:
            db_path.unlink()
            journal_path.unlink()
        except PermissionError:
            return db_path.with_name(f"{db_path.stem}_active{db_path.suffix}")
    return db_path


def _journal_mode() -> str:
    mode = os.getenv("LK_SQLITE_JOURNAL_MODE", "MEMORY").upper()
    return mode if mode in {"DELETE", "TRUNCATE", "PERSIST", "MEMORY", "WAL", "OFF"} else "MEMORY"
