from __future__ import annotations

import json
import hashlib
import re
import sqlite3
from collections.abc import Iterable
from typing import Any

from app.backend.models import ChunkRecord, KnowledgeGroup, KnowledgeUnit, SearchResult, SourceRecord, StructuredNote


class KnowledgeRepository:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def upsert_source(self, source: SourceRecord) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO sources (
                  source_id, source_type, file_path, raw_text, clean_text, text_hash,
                  created_at, imported_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                  source_type=excluded.source_type,
                  file_path=excluded.file_path,
                  raw_text=excluded.raw_text,
                  clean_text=excluded.clean_text,
                  text_hash=excluded.text_hash,
                  imported_at=excluded.imported_at,
                  status=excluded.status
                """,
                (
                    source.source_id,
                    source.source_type,
                    source.file_path,
                    source.raw_text,
                    source.clean_text,
                    source.text_hash,
                    source.created_at,
                    source.imported_at,
                    source.status,
                ),
            )

    def get_source_by_path(self, file_path: str) -> SourceRecord | None:
        row = self.conn.execute("SELECT * FROM sources WHERE file_path = ? LIMIT 1", (file_path,)).fetchone()
        return _source_from_row(row) if row else None

    def get_source_by_hash(self, text_hash: str) -> SourceRecord | None:
        row = self.conn.execute("SELECT * FROM sources WHERE text_hash = ? LIMIT 1", (text_hash,)).fetchone()
        return _source_from_row(row) if row else None

    def get_source(self, source_id: str) -> SourceRecord | None:
        row = self.conn.execute("SELECT * FROM sources WHERE source_id = ?", (source_id,)).fetchone()
        return _source_from_row(row) if row else None

    def set_source_status(self, source_id: str, status: str) -> None:
        with self.conn:
            self.conn.execute("UPDATE sources SET status = ? WHERE source_id = ?", (status, source_id))

    def delete_notes_for_source(self, source_id: str) -> None:
        note_rows = self.conn.execute(
            "SELECT note_id FROM notes_structured WHERE source_id = ?",
            (source_id,),
        ).fetchall()
        with self.conn:
            self.conn.execute("DELETE FROM knowledge_units WHERE source_id = ?", (source_id,))
            self.conn.execute("DELETE FROM knowledge_groups WHERE source_id = ?", (source_id,))
            for row in note_rows:
                self.conn.execute("DELETE FROM fts_notes WHERE note_id = ?", (row["note_id"],))
            self.conn.execute("DELETE FROM notes_structured WHERE source_id = ?", (source_id,))

    def insert_note(self, note: StructuredNote, chunks: Iterable[ChunkRecord]) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO notes_structured (
                  note_id, source_id, title, note_type, themes_json, summary, faithful_content,
                  key_points_json, usage_scenarios_json, user_insights, keywords_json,
                  source_excerpt, markdown_content, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    note.note_id,
                    note.source_id,
                    note.title,
                    note.note_type,
                    _json(note.themes),
                    note.summary,
                    note.faithful_content,
                    _json(note.key_points),
                    _json(note.usage_scenarios),
                    note.user_insights,
                    _json(note.keywords),
                    note.source_excerpt,
                    note.markdown_content,
                    note.created_at,
                    note.updated_at,
                ),
            )
            for chunk in chunks:
                self.conn.execute(
                    """
                    INSERT INTO chunks (
                      chunk_id, note_id, chunk_text, chunk_order, chunk_type, keywords_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.note_id,
                        chunk.chunk_text,
                        chunk.chunk_order,
                        chunk.chunk_type,
                        _json(chunk.keywords),
                    ),
                )
            self._insert_fts(note)

    def replace_note_for_source(self, source_id: str, note: StructuredNote, chunks: Iterable[ChunkRecord]) -> None:
        with self.conn:
            self.delete_notes_for_source(source_id)
            self.insert_note(note, chunks)

    def list_notes(
        self,
        *,
        limit: int = 100,
        theme: str | None = None,
        note_type: str | None = None,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        if theme:
            where.append("themes_json LIKE ?")
            params.append(f"%{theme}%")
        if note_type:
            where.append("note_type = ?")
            params.append(note_type)
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        rows = self.conn.execute(
            f"""
            SELECT n.*, s.file_path, s.imported_at
            FROM notes_structured n
            LEFT JOIN sources s ON s.source_id = n.source_id
            {where_sql}
            ORDER BY n.updated_at DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
        return [_note_dict(row) for row in rows]

    def list_import_history(self, *, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT
              s.source_id,
              s.source_type,
              s.file_path,
              s.created_at,
              s.imported_at,
              s.status,
              COUNT(n.note_id) AS note_count,
              MAX(n.updated_at) AS latest_note_at,
              GROUP_CONCAT(n.title, '||') AS note_titles
            FROM sources s
            LEFT JOIN notes_structured n ON n.source_id = s.source_id
            GROUP BY s.source_id
            ORDER BY s.imported_at DESC, latest_note_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        history: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["note_count"] = int(item.get("note_count") or 0)
            item["note_titles"] = [part for part in str(item.get("note_titles") or "").split("||") if part][:8]
            history.append(item)
        return history

    def get_note(self, note_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT n.*, s.file_path, s.imported_at
            FROM notes_structured n
            LEFT JOIN sources s ON s.source_id = n.source_id
            WHERE n.note_id = ?
            """,
            (note_id,),
        ).fetchone()
        return _note_dict(row) if row else None

    def list_notes_without_units(self, *, limit: int = 1000) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT n.*, s.file_path, s.imported_at
            FROM notes_structured n
            LEFT JOIN knowledge_units u ON u.note_id = n.note_id
            LEFT JOIN sources s ON s.source_id = n.source_id
            WHERE u.unit_id IS NULL
            ORDER BY n.updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [_note_dict(row) for row in rows]

    def list_notes_for_organize(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT *
            FROM notes_structured
            ORDER BY updated_at DESC
            """
        ).fetchall()
        return [_note_dict(row) for row in rows]

    def list_notes_updated_after(self, cutoff: str, *, limit: int = 10000) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT *
            FROM notes_structured
            WHERE updated_at > ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (cutoff, limit),
        ).fetchall()
        return [_note_dict(row) for row in rows]

    def delete_note(self, note_id: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM fts_notes WHERE note_id = ?", (note_id,))
            self.conn.execute("DELETE FROM notes_structured WHERE note_id = ?", (note_id,))

    def update_note_structured(
        self,
        *,
        note_id: str,
        title: str,
        note_type: str,
        themes: list[str],
        summary: str,
        faithful_content: str,
        key_points: list[str],
        usage_scenarios: list[str],
        user_insights: str,
        keywords: list[str],
        source_excerpt: str,
        markdown_content: str,
        updated_at: str,
    ) -> bool:
        with self.conn:
            cur = self.conn.execute(
                """
                UPDATE notes_structured
                SET
                  title = ?,
                  note_type = ?,
                  themes_json = ?,
                  summary = ?,
                  faithful_content = ?,
                  key_points_json = ?,
                  usage_scenarios_json = ?,
                  user_insights = ?,
                  keywords_json = ?,
                  source_excerpt = ?,
                  markdown_content = ?,
                  updated_at = ?
                WHERE note_id = ?
                """,
                (
                    title,
                    note_type,
                    _json(themes),
                    summary,
                    faithful_content,
                    _json(key_points),
                    _json(usage_scenarios),
                    user_insights,
                    _json(keywords),
                    source_excerpt,
                    markdown_content,
                    updated_at,
                    note_id,
                ),
            )
            changed = cur.rowcount > 0
            if changed:
                self._refresh_fts_by_note_id(note_id)
            return changed

    def upsert_knowledge_group(self, group: KnowledgeGroup) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO knowledge_groups (
                  group_id, source_id, group_title, group_type, summary, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(group_id) DO UPDATE SET
                  group_title=excluded.group_title,
                  group_type=excluded.group_type,
                  summary=excluded.summary,
                  updated_at=excluded.updated_at
                """,
                (
                    group.group_id,
                    group.source_id,
                    group.group_title,
                    group.group_type,
                    group.summary,
                    group.created_at,
                    group.updated_at,
                ),
            )

    def get_first_group_for_source(self, source_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT *
            FROM knowledge_groups
            WHERE source_id = ?
            ORDER BY created_at
            LIMIT 1
            """,
            (source_id,),
        ).fetchone()
        return dict(row) if row else None

    def insert_knowledge_unit(self, unit: KnowledgeUnit) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO knowledge_units (
                  unit_id, group_id, source_id, note_id, parent_unit_id, title, content,
                  evidence, note_type, order_index, confidence, attributes_json,
                  created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    unit.unit_id,
                    unit.group_id,
                    unit.source_id,
                    unit.note_id,
                    unit.parent_unit_id,
                    unit.title,
                    unit.content,
                    unit.evidence,
                    unit.note_type,
                    unit.order_index,
                    float(unit.confidence),
                    _json(unit.attributes),
                    unit.created_at,
                    unit.updated_at,
                ),
            )

    def upsert_unit_embedding(
        self,
        *,
        unit_id: str,
        note_id: str | None,
        embedding_model: str,
        text_hash: str,
        vector: list[float],
        now: str,
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO unit_embeddings (
                  embedding_id, unit_id, note_id, embedding_model, text_hash, vector_json,
                  dimensions, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(unit_id, embedding_model) DO UPDATE SET
                  note_id=excluded.note_id,
                  text_hash=excluded.text_hash,
                  vector_json=excluded.vector_json,
                  dimensions=excluded.dimensions,
                  updated_at=excluded.updated_at
                """,
                (
                    _embedding_id(unit_id, embedding_model),
                    unit_id,
                    note_id,
                    embedding_model,
                    text_hash,
                    _json(vector),
                    len(vector),
                    now,
                    now,
                ),
            )

    def update_unit_for_note(
        self,
        *,
        note_id: str,
        title: str,
        content: str,
        evidence: str,
        note_type: str,
        attributes: dict[str, Any],
        updated_at: str,
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                UPDATE knowledge_units
                SET title = ?, content = ?, evidence = ?, note_type = ?,
                    attributes_json = ?, updated_at = ?
                WHERE note_id = ?
                """,
                (title, content, evidence, note_type, _json(attributes), updated_at, note_id),
            )

    def get_unit_by_note_id(self, note_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM knowledge_units WHERE note_id = ? LIMIT 1",
            (note_id,),
        ).fetchone()
        if not row:
            return None
        data = dict(row)
        data["attributes"] = _loads(data.pop("attributes_json", None))
        return data

    def list_units_for_relation_build(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT
              u.*, g.group_title, n.updated_at AS note_updated_at,
              n.summary AS note_summary, n.faithful_content AS note_faithful_content,
              n.themes_json, n.keywords_json
            FROM knowledge_units u
            LEFT JOIN knowledge_groups g ON g.group_id = u.group_id
            LEFT JOIN notes_structured n ON n.note_id = u.note_id
            WHERE u.note_id IS NOT NULL
            ORDER BY u.group_id, u.order_index, u.updated_at
            """
        ).fetchall()
        data: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["attributes"] = _loads(item.pop("attributes_json", None))
            item["themes"] = _loads(item.pop("themes_json", None))
            item["keywords"] = _loads(item.pop("keywords_json", None))
            data.append(item)
        return data

    def list_units_for_embedding(self, *, limit: int = 10000, model_name: str | None = None) -> list[dict[str, Any]]:
        model_join = "AND e.embedding_model = ?" if model_name else ""
        params: list[Any] = []
        if model_name:
            params.append(model_name)
        params.append(limit)
        rows = self.conn.execute(
            f"""
            SELECT
              u.*, g.group_title, n.title AS note_title, n.summary AS note_summary,
              n.faithful_content AS note_faithful_content, n.themes_json, n.keywords_json,
              e.embedding_model, e.text_hash AS embedding_text_hash
            FROM knowledge_units u
            LEFT JOIN knowledge_groups g ON g.group_id = u.group_id
            LEFT JOIN notes_structured n ON n.note_id = u.note_id
            LEFT JOIN unit_embeddings e ON e.unit_id = u.unit_id {model_join}
            WHERE u.note_id IS NOT NULL
            ORDER BY u.updated_at DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        data: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["attributes"] = _loads(item.pop("attributes_json", None))
            item["themes"] = _loads(item.pop("themes_json", None))
            item["keywords"] = _loads(item.pop("keywords_json", None))
            data.append(item)
        return data

    def list_unit_embeddings(self, *, model_name: str | None = None, limit: int = 20000) -> list[dict[str, Any]]:
        params: list[Any] = []
        where = ""
        if model_name:
            where = "WHERE e.embedding_model = ?"
            params.append(model_name)
        params.append(limit)
        rows = self.conn.execute(
            f"""
            SELECT
              e.unit_id, e.note_id, e.embedding_model, e.text_hash, e.vector_json,
              u.title AS unit_title, u.content AS unit_content, u.evidence AS unit_evidence,
              n.*, s.file_path, s.imported_at
            FROM unit_embeddings e
            JOIN knowledge_units u ON u.unit_id = e.unit_id
            JOIN notes_structured n ON n.note_id = e.note_id
            LEFT JOIN sources s ON s.source_id = n.source_id
            {where}
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        data: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["vector"] = _loads(item.pop("vector_json", None))
            item["themes"] = _loads(item.pop("themes_json", None))
            item["keywords"] = _loads(item.pop("keywords_json", None))
            item["key_points"] = _loads(item.pop("key_points_json", None))
            item["usage_scenarios"] = _loads(item.pop("usage_scenarios_json", None))
            data.append(item)
        return data

    def embedding_counts_by_model(self) -> dict[str, int]:
        rows = self.conn.execute(
            """
            SELECT embedding_model, COUNT(*) AS cnt
            FROM unit_embeddings
            GROUP BY embedding_model
            ORDER BY embedding_model
            """
        ).fetchall()
        return {str(row["embedding_model"]): int(row["cnt"]) for row in rows}

    def count_knowledge_units(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS cnt FROM knowledge_units WHERE note_id IS NOT NULL").fetchone()
        return int(row["cnt"] if row else 0)

    def list_units_for_graph(self, *, limit: int = 120) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT
              u.*, g.group_title, g.group_type, n.title AS note_title, n.note_type AS note_note_type,
              n.summary AS note_summary, n.faithful_content AS note_faithful_content,
              n.themes_json, n.keywords_json, n.updated_at AS note_updated_at,
              s.file_path, s.imported_at
            FROM knowledge_units u
            LEFT JOIN knowledge_groups g ON g.group_id = u.group_id
            LEFT JOIN notes_structured n ON n.note_id = u.note_id
            LEFT JOIN sources s ON s.source_id = u.source_id
            WHERE u.note_id IS NOT NULL
            ORDER BY n.updated_at DESC, u.updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        data: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["attributes"] = _loads(item.pop("attributes_json", None))
            item["themes"] = _loads(item.pop("themes_json", None))
            item["keywords"] = _loads(item.pop("keywords_json", None))
            data.append(item)
        return data

    def list_keyword_links_for_units(self, unit_ids: list[str]) -> list[dict[str, Any]]:
        ids = [unit_id for unit_id in unit_ids if unit_id]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self.conn.execute(
            f"""
            SELECT
              uk.unit_id, uk.term_id, uk.confidence, uk.matched_by,
              t.canonical_name, t.normalized_name
            FROM unit_keywords uk
            JOIN keyword_terms t ON t.term_id = uk.term_id
            WHERE uk.unit_id IN ({placeholders})
            ORDER BY t.canonical_name, uk.confidence DESC
            """,
            tuple(ids),
        ).fetchall()
        return [dict(row) for row in rows]

    def list_unit_keyword_links(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT
              u.unit_id, u.note_id, uk.term_id, uk.confidence, uk.matched_by,
              t.canonical_name
            FROM unit_keywords uk
            JOIN knowledge_units u ON u.unit_id = uk.unit_id
            JOIN keyword_terms t ON t.term_id = uk.term_id
            WHERE u.note_id IS NOT NULL
            ORDER BY t.canonical_name, u.note_id
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def find_keyword_term(self, normalized_name: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM keyword_terms WHERE normalized_name = ? LIMIT 1",
            (normalized_name,),
        ).fetchone()
        return dict(row) if row else None

    def find_keyword_alias(self, normalized_alias: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT t.*, a.alias, a.normalized_alias
            FROM keyword_aliases a
            JOIN keyword_terms t ON t.term_id = a.term_id
            WHERE a.normalized_alias = ?
            LIMIT 1
            """,
            (normalized_alias,),
        ).fetchone()
        return dict(row) if row else None

    def list_keyword_terms(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT t.*,
                   GROUP_CONCAT(a.alias, '||') AS aliases,
                   GROUP_CONCAT(a.normalized_alias, '||') AS normalized_aliases
            FROM keyword_terms t
            LEFT JOIN keyword_aliases a ON a.term_id = t.term_id
            GROUP BY t.term_id
            ORDER BY t.canonical_name
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def insert_keyword_term(
        self,
        *,
        term_id: str,
        canonical_name: str,
        normalized_name: str,
        description: str,
        created_at: str,
        updated_at: str,
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO keyword_terms (
                  term_id, canonical_name, normalized_name, description, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'active', ?, ?)
                """,
                (term_id, canonical_name, normalized_name, description, created_at, updated_at),
            )

    def insert_keyword_alias(
        self,
        *,
        alias_id: str,
        term_id: str,
        alias: str,
        normalized_alias: str,
        source: str,
        confidence: float,
        created_at: str,
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO keyword_aliases (
                  alias_id, term_id, alias, normalized_alias, source, confidence, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (alias_id, term_id, alias, normalized_alias, source, float(confidence), created_at),
            )

    def replace_unit_keywords(self, unit_id: str, keyword_links: list[dict[str, Any]]) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM unit_keywords WHERE unit_id = ?", (unit_id,))
            for link in keyword_links:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO unit_keywords (
                      unit_id, term_id, confidence, matched_by
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        unit_id,
                        link["term_id"],
                        float(link.get("confidence", 0.8)),
                        link.get("matched_by") or "",
                    ),
                )

    def search_notes_by_keyword_names(self, names: list[str], *, limit: int = 10) -> list[SearchResult]:
        normalized = _keyword_query_keys(names)
        if not normalized:
            return []
        placeholders = ",".join("?" for _ in normalized)
        rows = self.conn.execute(
            f"""
            SELECT DISTINCT
              n.*, s.file_path, s.imported_at,
              5.0 AS score,
              'keyword:' || t.canonical_name AS snippet
            FROM keyword_terms t
            LEFT JOIN keyword_aliases a ON a.term_id = t.term_id
            JOIN unit_keywords uk ON uk.term_id = t.term_id
            JOIN knowledge_units u ON u.unit_id = uk.unit_id
            JOIN notes_structured n ON n.note_id = u.note_id
            LEFT JOIN sources s ON s.source_id = n.source_id
            WHERE t.normalized_name IN ({placeholders})
               OR a.normalized_alias IN ({placeholders})
            ORDER BY n.updated_at DESC
            LIMIT ?
            """,
            (*normalized, *normalized, limit),
        ).fetchall()
        return [_search_result(row) for row in rows]

    def search_notes_by_title(self, query: str, *, limit: int = 10) -> list[SearchResult]:
        terms = _query_terms(query)
        if not terms:
            return []
        filters = " OR ".join("n.title LIKE ?" for _ in terms)
        params = [f"%{term}%" for term in terms]
        rows = self.conn.execute(
            f"""
            SELECT DISTINCT
              n.*, s.file_path, s.imported_at,
              2.0 AS score,
              'title:' || n.title AS snippet
            FROM notes_structured n
            LEFT JOIN sources s ON s.source_id = n.source_id
            WHERE {filters}
            ORDER BY n.updated_at DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
        return [_search_result(row) for row in rows]

    def search_notes_by_unit_text(self, query: str, *, limit: int = 10) -> list[SearchResult]:
        terms = _query_terms(query)
        if not terms:
            return []
        filters = " OR ".join("(u.title LIKE ? OR u.content LIKE ? OR u.evidence LIKE ?)" for _ in terms)
        params: list[Any] = []
        for term in terms:
            like = f"%{term}%"
            params.extend([like, like, like])
        rows = self.conn.execute(
            f"""
            SELECT DISTINCT
              n.*, s.file_path, s.imported_at,
              8.0 AS score,
              'unit:' || COALESCE(u.title, n.title) AS snippet
            FROM knowledge_units u
            JOIN notes_structured n ON n.note_id = u.note_id
            LEFT JOIN sources s ON s.source_id = n.source_id
            WHERE {filters}
            ORDER BY n.updated_at DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
        return [_search_result(row) for row in rows]

    def get_group_peer_notes(self, note_ids: list[str], *, limit: int = 8) -> list[SearchResult]:
        normalized_ids = [note_id for note_id in note_ids if note_id]
        if not normalized_ids:
            return []
        placeholders = ",".join("?" for _ in normalized_ids)
        rows = self.conn.execute(
            f"""
            SELECT DISTINCT
              n.*, s.file_path, s.imported_at,
              40.0 AS score,
              'near_group:' || COALESCE(g.group_title, '') AS snippet
            FROM knowledge_units seed
            JOIN knowledge_units peer ON peer.group_id = seed.group_id
              AND peer.note_id IS NOT NULL
              AND peer.note_id != seed.note_id
              AND ABS(peer.order_index - seed.order_index) <= 2
            JOIN notes_structured n ON n.note_id = peer.note_id
            LEFT JOIN knowledge_groups g ON g.group_id = peer.group_id
            LEFT JOIN sources s ON s.source_id = n.source_id
            WHERE seed.note_id IN ({placeholders})
              AND peer.note_id NOT IN ({placeholders})
            ORDER BY peer.order_index ASC, n.updated_at DESC
            LIMIT ?
            """,
            (*normalized_ids, *normalized_ids, limit),
        ).fetchall()
        return [_search_result(row) for row in rows]

    def get_relation_peer_notes(self, note_ids: list[str], *, limit: int = 8) -> list[SearchResult]:
        normalized_ids = [note_id for note_id in note_ids if note_id]
        if not normalized_ids:
            return []
        placeholders = ",".join("?" for _ in normalized_ids)
        rows = self.conn.execute(
            f"""
            WITH latest_run AS (
              SELECT run_id
              FROM organize_runs
              WHERE status = 'completed'
              ORDER BY finished_at DESC, started_at DESC
              LIMIT 1
            ),
            peers AS (
              SELECT to_note_id AS note_id, relation_type, relation_strength, reason, score AS rel_score
              FROM knowledge_relations
              WHERE from_note_id IN ({placeholders})
                AND run_id = (SELECT run_id FROM latest_run)
                AND display_default = 1
                AND relation_layer IN ('semantic', 'structure')
                AND relation_strength IN ('strong', 'medium')
              UNION ALL
              SELECT from_note_id AS note_id, relation_type, relation_strength, reason, score AS rel_score
              FROM knowledge_relations
              WHERE to_note_id IN ({placeholders})
                AND run_id = (SELECT run_id FROM latest_run)
                AND display_default = 1
                AND relation_layer IN ('semantic', 'structure')
                AND relation_strength IN ('strong', 'medium')
            )
            SELECT DISTINCT
              n.*, s.file_path, s.imported_at,
              55.0 AS score,
              'relation:' || peers.relation_type || ':' || peers.relation_strength || ':' || COALESCE(peers.reason, '') AS snippet
            FROM peers
            JOIN notes_structured n ON n.note_id = peers.note_id
            LEFT JOIN sources s ON s.source_id = n.source_id
            WHERE peers.note_id NOT IN ({placeholders})
            ORDER BY peers.rel_score DESC, n.updated_at DESC
            LIMIT ?
            """,
            (*normalized_ids, *normalized_ids, *normalized_ids, limit),
        ).fetchall()
        return [_search_result(row) for row in rows]

    def search_notes(
        self,
        query: str,
        *,
        limit: int = 10,
        theme: str | None = None,
        note_type: str | None = None,
    ) -> list[SearchResult]:
        match_query = _match_query(query)
        if not match_query:
            return []

        filters: list[str] = []
        params: list[Any] = [match_query]
        if theme:
            filters.append("n.themes_json LIKE ?")
            params.append(f"%{theme}%")
        if note_type:
            filters.append("n.note_type = ?")
            params.append(note_type)
        filter_sql = f"AND {' AND '.join(filters)}" if filters else ""
        params.append(limit)

        try:
            rows = self.conn.execute(
                f"""
                SELECT
                  n.*, s.file_path, s.imported_at,
                  bm25(fts_notes) AS score,
                  snippet(fts_notes, 4, '[', ']', '...', 32) AS snippet
                FROM fts_notes
                JOIN notes_structured n ON n.note_id = fts_notes.note_id
                LEFT JOIN sources s ON s.source_id = n.source_id
                WHERE fts_notes MATCH ?
                {filter_sql}
                ORDER BY score, n.updated_at DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
            if rows:
                return [_search_result(row) for row in rows]
        except sqlite3.OperationalError:
            pass

        return self._like_search(query, limit=limit, theme=theme, note_type=note_type)

    def search_notes_by_vector(self, query_vector: list[float], *, model_name: str, limit: int = 10) -> list[SearchResult]:
        from app.backend.embeddings.local_embedder import cosine_similarity

        rows = self.list_unit_embeddings(model_name=model_name, limit=20000)
        scored: list[tuple[dict[str, Any], float]] = []
        for row in rows:
            score = cosine_similarity(query_vector, row.get("vector") or [])
            if score > 0:
                scored.append((row, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        results: list[SearchResult] = []
        for row, score in scored[:limit]:
            results.append(
                SearchResult(
                    note_id=row["note_id"],
                    source_id=row["source_id"],
                    title=row["title"],
                    note_type=row.get("note_type") or "",
                    summary=row.get("summary") or "",
                    markdown_content=row.get("markdown_content") or "",
                    source_excerpt=row.get("source_excerpt") or "",
                    themes=row.get("themes") or [],
                    keywords=row.get("keywords") or [],
                    file_path=row.get("file_path"),
                    imported_at=row.get("imported_at"),
                    score=float(score),
                    snippet=f"vector:{row.get('unit_title') or row.get('title') or ''}",
                )
            )
        return results

    def get_adjacent_chunk_notes(
        self,
        note_ids: list[str],
        *,
        distance: int = 1,
        limit: int = 6,
    ) -> list[SearchResult]:
        normalized_ids = [note_id for note_id in note_ids if note_id]
        if not normalized_ids or limit <= 0:
            return []

        placeholders = ",".join("?" for _ in normalized_ids)
        params: list[Any] = [distance, *normalized_ids, *normalized_ids, limit]
        rows = self.conn.execute(
            f"""
            SELECT DISTINCT
              n.*, s.file_path, s.imported_at, 99.0 AS score, n.summary AS snippet
            FROM chunks base
            JOIN chunks near ON near.chunk_type = 'source_chunk'
              AND base.chunk_type = 'source_chunk'
              AND near.note_id != base.note_id
              AND ABS(near.chunk_order - base.chunk_order) <= ?
            JOIN notes_structured b ON b.note_id = base.note_id
            JOIN notes_structured n ON n.note_id = near.note_id AND n.source_id = b.source_id
            LEFT JOIN sources s ON s.source_id = n.source_id
            WHERE base.note_id IN ({placeholders})
              AND near.note_id NOT IN ({placeholders})
            ORDER BY n.updated_at DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        return [_search_result(row) for row in rows]

    def create_organize_run(self, run_id: str, *, status: str, started_at: str) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO organize_runs (
                  run_id, status, started_at, finished_at, stats_json, report_markdown, error_message
                ) VALUES (?, ?, ?, NULL, NULL, NULL, NULL)
                """,
                (run_id, status, started_at),
            )

    def update_organize_run(
        self,
        run_id: str,
        *,
        status: str,
        finished_at: str | None = None,
        stats: dict[str, Any] | None = None,
        report_markdown: str | None = None,
        error_message: str | None = None,
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                UPDATE organize_runs
                SET
                  status = ?,
                  finished_at = ?,
                  stats_json = ?,
                  report_markdown = ?,
                  error_message = ?
                WHERE run_id = ?
                """,
                (
                    status,
                    finished_at,
                    _json(stats) if stats is not None else None,
                    report_markdown,
                    error_message,
                    run_id,
                ),
            )

    def get_organize_run(self, run_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM organize_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if not row:
            return None
        return _organize_run_dict(row)

    def get_latest_organize_run(self) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT *
            FROM organize_runs
            ORDER BY started_at DESC
            LIMIT 1
            """
        ).fetchone()
        if not row:
            return None
        return _organize_run_dict(row)

    def get_latest_completed_organize_run(self) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT *
            FROM organize_runs
            WHERE status = 'completed'
            ORDER BY finished_at DESC, started_at DESC
            LIMIT 1
            """
        ).fetchone()
        if not row:
            return None
        return _organize_run_dict(row)

    def replace_relations_for_run(self, run_id: str, relations: list[dict[str, Any]]) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM knowledge_relations WHERE run_id = ?", (run_id,))
            for rel in relations:
                self.conn.execute(
                    """
                    INSERT INTO knowledge_relations (
                      relation_id, run_id, from_note_id, to_note_id, relation_type,
                      relation_layer, relation_strength, display_default, score, reason, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rel["relation_id"],
                        run_id,
                        rel["from_note_id"],
                        rel["to_note_id"],
                        rel["relation_type"],
                        rel.get("relation_layer") or "semantic",
                        rel.get("relation_strength") or "medium",
                        1 if rel.get("display_default", True) else 0,
                        float(rel.get("score", 0.0)),
                        rel.get("reason"),
                        rel["created_at"],
                    ),
                )

    def delete_relations_except_run(self, run_id: str) -> int:
        with self.conn:
            cur = self.conn.execute("DELETE FROM knowledge_relations WHERE run_id != ?", (run_id,))
            return int(cur.rowcount or 0)

    def list_relations_for_run(self, run_id: str, *, limit: int = 300) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT *
            FROM knowledge_relations
            WHERE run_id = ?
            ORDER BY display_default DESC, score DESC, created_at DESC
            LIMIT ?
            """,
            (run_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def list_relations_for_latest_completed_run(self, *, limit: int = 100000) -> list[dict[str, Any]]:
        run = self.get_latest_completed_organize_run()
        if not run:
            return []
        return self.list_relations_for_run(str(run["run_id"]), limit=limit)

    def list_display_relations_for_latest_run(self, *, limit: int = 300) -> list[dict[str, Any]]:
        row = self.conn.execute(
            """
            SELECT run_id
            FROM organize_runs
            WHERE status = 'completed'
            ORDER BY started_at DESC
            LIMIT 1
            """
        ).fetchone()
        if not row:
            return []
        rows = self.conn.execute(
            """
            SELECT *
            FROM knowledge_relations
            WHERE run_id = ?
              AND display_default = 1
            ORDER BY
              CASE relation_strength
                WHEN 'strong' THEN 0
                WHEN 'medium' THEN 1
                ELSE 2
              END,
              score DESC,
              created_at DESC
            LIMIT ?
            """,
            (row["run_id"], limit),
        ).fetchall()
        return [dict(item) for item in rows]

    def _like_search(
        self,
        query: str,
        *,
        limit: int,
        theme: str | None = None,
        note_type: str | None = None,
    ) -> list[SearchResult]:
        terms = _query_terms(query)
        if not terms:
            return []
        text_filters: list[str] = []
        params: list[Any] = []
        for term in terms[:48]:
            text_filters.append(
                "(n.title LIKE ? OR n.summary LIKE ? OR n.faithful_content LIKE ? OR n.markdown_content LIKE ? "
                "OR n.source_excerpt LIKE ? OR n.themes_json LIKE ? OR n.keywords_json LIKE ?)"
            )
            like = f"%{term}%"
            params.extend([like, like, like, like, like, like, like])
        filters = [f"({' OR '.join(text_filters)})"]
        if theme:
            filters.append("n.themes_json LIKE ?")
            params.append(f"%{theme}%")
        if note_type:
            filters.append("n.note_type = ?")
            params.append(note_type)
        params.append(limit)
        rows = self.conn.execute(
            f"""
            SELECT n.*, s.file_path, s.imported_at, 100.0 AS score, n.summary AS snippet
            FROM notes_structured n
            LEFT JOIN sources s ON s.source_id = n.source_id
            WHERE {' AND '.join(filters)}
            ORDER BY n.updated_at DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        return [_search_result(row) for row in rows]

    def _insert_fts(self, note: StructuredNote) -> None:
        self.conn.execute("DELETE FROM fts_notes WHERE note_id = ?", (note.note_id,))
        self.conn.execute(
            """
            INSERT INTO fts_notes (
              note_id, title, summary, key_points, markdown_content, source_excerpt, themes, keywords
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                note.note_id,
                note.title,
                _join_text(note.summary, note.faithful_content),
                "\n".join(note.key_points),
                _join_text(note.markdown_content, note.faithful_content),
                note.source_excerpt,
                " ".join(note.themes),
                " ".join(note.keywords),
            ),
        )

    def _refresh_fts_by_note_id(self, note_id: str) -> None:
        row = self.conn.execute(
            """
            SELECT note_id, title, summary, faithful_content, key_points_json, markdown_content,
                   source_excerpt, themes_json, keywords_json
            FROM notes_structured
            WHERE note_id = ?
            """,
            (note_id,),
        ).fetchone()
        if not row:
            return
        self.conn.execute("DELETE FROM fts_notes WHERE note_id = ?", (note_id,))
        self.conn.execute(
            """
            INSERT INTO fts_notes (
              note_id, title, summary, key_points, markdown_content, source_excerpt, themes, keywords
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["note_id"],
                row["title"] or "",
                _join_text(row["summary"], row["faithful_content"]),
                "\n".join(_loads(row["key_points_json"])),
                _join_text(row["markdown_content"], row["faithful_content"]),
                row["source_excerpt"] or "",
                " ".join(_loads(row["themes_json"])),
                " ".join(_loads(row["keywords_json"])),
            ),
        )


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _embedding_id(unit_id: str, embedding_model: str) -> str:
    digest = hashlib.sha1(f"{unit_id}:{embedding_model}".encode("utf-8")).hexdigest()
    return f"emb-{digest}"


def _join_text(*values: Any) -> str:
    parts: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in parts:
            parts.append(text)
    return "\n".join(parts)


def _loads(value: str | None) -> Any:
    if not value:
        return []
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return []


def _source_from_row(row: sqlite3.Row) -> SourceRecord:
    return SourceRecord(
        source_id=row["source_id"],
        source_type=row["source_type"],
        file_path=row["file_path"],
        raw_text=row["raw_text"],
        clean_text=row["clean_text"] or "",
        text_hash=row["text_hash"],
        created_at=row["created_at"],
        imported_at=row["imported_at"],
        status=row["status"],
    )


def _note_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["themes"] = _loads(data.pop("themes_json", None))
    data["key_points"] = _loads(data.pop("key_points_json", None))
    data["usage_scenarios"] = _loads(data.pop("usage_scenarios_json", None))
    data["keywords"] = _loads(data.pop("keywords_json", None))
    return data


def _organize_run_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["stats"] = _loads(data.pop("stats_json", None))
    return data


def _search_result(row: sqlite3.Row) -> SearchResult:
    return SearchResult(
        note_id=row["note_id"],
        source_id=row["source_id"],
        title=row["title"],
        note_type=row["note_type"] or "",
        summary=row["summary"] or "",
        markdown_content=row["markdown_content"] or "",
        source_excerpt=row["source_excerpt"] or "",
        themes=_loads(row["themes_json"]),
        keywords=_loads(row["keywords_json"]),
        file_path=row["file_path"],
        imported_at=row["imported_at"],
        score=float(row["score"]),
        snippet=row["snippet"] or row["summary"] or "",
    )


def _match_query(query: str) -> str:
    tokens = _query_terms(query)
    if not tokens:
        return ""
    return " OR ".join(f'"{token.replace(chr(34), chr(34) + chr(34))}"' for token in tokens[:8])


def _query_terms(query: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}|[0-9]+|[\u4e00-\u9fff]+", query, flags=re.UNICODE)
    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        candidates = [token] if not _contains_cjk(token) else _cjk_query_terms(token)
        for candidate in candidates:
            if len(candidate) < 2 and not candidate.isascii():
                continue
            key = candidate.lower()
            if key not in seen:
                result.append(candidate)
                seen.add(key)
    return result[:64]


def _contains_cjk(value: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", value))


def _cjk_query_terms(token: str) -> list[str]:
    token = re.sub(
        r"(我之前|我以前|以前|关于|有没有|有哪些|哪一些|哪些|什么|怎么|如何|是否|可以|用于|用来|内容|想法|意见|请|帮我|整理|一下)",
        " ",
        token,
    )
    runs = [part for part in re.split(r"\s+", token) if len(part) >= 2]
    terms: list[str] = []
    for run in runs:
        if 2 <= len(run) <= 18:
            terms.append(run)
        max_n = min(6, len(run))
        for size in range(max_n, 1, -1):
            for start in range(0, len(run) - size + 1):
                piece = run[start : start + size]
                if _useful_cjk_piece(piece):
                    terms.append(piece)
    return _dedupe_terms(terms)[:48]


def _useful_cjk_piece(piece: str) -> bool:
    if len(piece) < 2:
        return False
    stop_pieces = {
        "的",
        "了",
        "是",
        "和",
        "与",
        "有",
        "几次",
        "几个",
        "主要",
        "阶段",
    }
    return piece not in stop_pieces and not all(ch in "的是了和与有在中对及或" for ch in piece)


def _dedupe_terms(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.lower()
        if value and key not in seen:
            result.append(value)
            seen.add(key)
    return result


def _normalize_keyword_name(value: str) -> str:
    return re.sub(r"[\s_\-·•/／]+", "", str(value).strip().lower())


def _keyword_query_keys(values: list[str]) -> list[str]:
    suffixes = ("技术", "方法", "方式", "机制", "模型", "系统", "体系", "理论", "研究", "分析", "工具", "策略")
    keys: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalize_keyword_name(value)
        candidates = [normalized]
        for suffix in suffixes:
            if normalized.endswith(suffix) and len(normalized) > len(suffix) + 2:
                candidates.append(normalized[: -len(suffix)])
                break
        for candidate in candidates:
            if candidate and candidate not in seen:
                keys.append(candidate)
                seen.add(candidate)
    return keys
