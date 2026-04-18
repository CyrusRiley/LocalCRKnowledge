from __future__ import annotations

import argparse
import json
import mimetypes
import threading
from dataclasses import dataclass, field
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from app.backend.config import Settings, load_settings
from app.backend.database.db import connect, init_db
from app.backend.database.repository import KnowledgeRepository
from app.backend.export.markdown_exporter import export_markdown, safe_filename
from app.backend.formatter.markdown_builder import build_note_markdown
from app.backend.embeddings.indexer import index_unit_embeddings, rebuild_unit_embeddings
from app.backend.embeddings.providers import active_embedding_models, configured_embedding_models
from app.backend.importer.text_importer import build_file_source, build_manual_source
from app.backend.keywords.normalizer import canonicalize_keywords
from app.backend.llm.client import QwenClient
from app.backend.maintenance.v2_backfill import backfill_v2_structures
from app.backend.organizer.knowledge_organizer import organize_knowledge_base
from app.backend.retrieval.answer_builder import build_answer_result
from app.backend.retrieval.graph_builder import build_library_graph
from app.backend.retrieval.search_service import SearchService
from app.backend.updater.incremental_updater import IncrementalUpdater
from app.backend.utils.logger import setup_logger
from app.backend.utils.time_utils import utc_now_iso
from app.backend.workflow import process_source


@dataclass
class Runtime:
    settings: Settings
    frontend_dir: Path
    logger_name: str
    conn: object | None = None
    repo: KnowledgeRepository | None = None
    llm_client: QwenClient | None = None
    logger: object | None = None
    organize_jobs: dict[str, dict] = field(default_factory=dict)
    organize_lock: threading.Lock = field(default_factory=threading.Lock)
    import_jobs: dict[str, dict] = field(default_factory=dict)
    import_lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class ServiceContext:
    conn: object
    repo: KnowledgeRepository
    llm_client: QwenClient
    logger: object
    settings: Settings

    @classmethod
    def from_runtime(cls, runtime: Runtime) -> "ServiceContext":
        if runtime.conn is None or runtime.repo is None or runtime.llm_client is None or runtime.logger is None:
            raise RuntimeError("Runtime service context is not initialized")
        return cls(
            conn=runtime.conn,
            repo=runtime.repo,
            llm_client=runtime.llm_client,
            logger=runtime.logger,
            settings=runtime.settings,
        )

    def close(self) -> None:
        return


class RequestHandler(BaseHTTPRequestHandler):
    runtime: Runtime
    server_version = "LocalKnowledgeHTTP/1.0"

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._handle_api_get(parsed)
            return
        self._serve_static(parsed.path)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._handle_api_post(parsed)
            return
        self._json_response(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def _handle_api_get(self, parsed) -> None:
        try:
            if parsed.path == "/api/health":
                self._json_response(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "llm_base_url": self.runtime.settings.llm_base_url,
                        "llm_model": self.runtime.settings.llm_model,
                    },
                )
                return

            if parsed.path == "/api/notes":
                params = parse_qs(parsed.query)
                limit = _to_int(params.get("limit", ["50"])[0], default=50, min_value=1, max_value=200)
                theme = _first_or_none(params.get("theme"))
                note_type = _first_or_none(params.get("note_type"))
                ctx = ServiceContext.from_runtime(self.runtime)
                try:
                    notes = ctx.repo.list_notes(limit=limit, theme=theme, note_type=note_type)
                    self._json_response(HTTPStatus.OK, {"notes": notes})
                finally:
                    ctx.close()
                return

            if parsed.path.startswith("/api/notes/"):
                note_id = parsed.path.rsplit("/", 1)[-1]
                ctx = ServiceContext.from_runtime(self.runtime)
                try:
                    note = ctx.repo.get_note(note_id)
                    if not note:
                        self._json_response(HTTPStatus.NOT_FOUND, {"error": "Note not found"})
                        return
                    self._json_response(HTTPStatus.OK, {"note": note})
                finally:
                    ctx.close()
                return

            if parsed.path == "/api/search":
                params = parse_qs(parsed.query)
                query = (params.get("query", [""])[0] or "").strip()
                if not query:
                    self._json_response(HTTPStatus.BAD_REQUEST, {"error": "query is required"})
                    return
                limit = _to_int(params.get("limit", ["10"])[0], default=10, min_value=1, max_value=50)
                ctx = ServiceContext.from_runtime(self.runtime)
                try:
                    service = SearchService(ctx.repo, ctx.llm_client, ctx.logger)
                    results = service.keyword_search(query, limit=limit)
                    self._json_response(HTTPStatus.OK, {"results": [_result_dict(item) for item in results]})
                finally:
                    ctx.close()
                return

            if parsed.path == "/api/keywords":
                ctx = ServiceContext.from_runtime(self.runtime)
                try:
                    keywords = []
                    for item in ctx.repo.list_keyword_terms():
                        aliases = [part for part in str(item.get("aliases") or "").split("||") if part]
                        keywords.append(
                            {
                                "term_id": item.get("term_id"),
                                "canonical_name": item.get("canonical_name"),
                                "description": item.get("description") or "",
                                "status": item.get("status") or "active",
                                "aliases": aliases,
                                "updated_at": item.get("updated_at"),
                            }
                        )
                    self._json_response(HTTPStatus.OK, {"keywords": keywords})
                finally:
                    ctx.close()
                return

            if parsed.path == "/api/import/status":
                params = parse_qs(parsed.query)
                job_id = (params.get("job_id", [""])[0] or "").strip()
                if not job_id:
                    self._json_response(HTTPStatus.BAD_REQUEST, {"error": "job_id is required"})
                    return
                with self.runtime.import_lock:
                    job = dict(self.runtime.import_jobs.get(job_id, {}))
                if not job:
                    self._json_response(HTTPStatus.NOT_FOUND, {"error": "import job not found"})
                    return
                self._json_response(HTTPStatus.OK, {"job": job})
                return

            if parsed.path == "/api/embeddings/status":
                ctx = ServiceContext.from_runtime(self.runtime)
                try:
                    configured = configured_embedding_models()
                    active = active_embedding_models()
                    counts = ctx.repo.embedding_counts_by_model()
                    total_units = ctx.repo.count_knowledge_units()
                    self._json_response(
                        HTTPStatus.OK,
                        {
                            "total_units": total_units,
                            "configured_models": [_embedding_config_dict(item, counts) for item in configured],
                            "active_models": [item.model_name for item in active],
                        },
                    )
                finally:
                    ctx.close()
                return

            if parsed.path == "/api/library/organize/status":
                params = parse_qs(parsed.query)
                run_id = (params.get("run_id", [""])[0] or "").strip()
                if not run_id:
                    self._json_response(HTTPStatus.BAD_REQUEST, {"error": "run_id is required"})
                    return
                ctx = ServiceContext.from_runtime(self.runtime)
                try:
                    run = ctx.repo.get_organize_run(run_id)
                    if not run:
                        self._json_response(HTTPStatus.NOT_FOUND, {"error": "run not found"})
                        return
                    with self.runtime.organize_lock:
                        live = dict(self.runtime.organize_jobs.get(run_id, {}))
                    run["live"] = live
                    self._json_response(HTTPStatus.OK, {"run": run})
                finally:
                    ctx.close()
                return

            if parsed.path == "/api/library/organize/latest":
                ctx = ServiceContext.from_runtime(self.runtime)
                try:
                    run = ctx.repo.get_latest_organize_run()
                    if not run:
                        self._json_response(HTTPStatus.OK, {"run": None, "relations": []})
                        return
                    relations = ctx.repo.list_relations_for_run(run["run_id"], limit=120)
                    self._json_response(HTTPStatus.OK, {"run": run, "relations": relations})
                finally:
                    ctx.close()
                return

            if parsed.path == "/api/library/graph":
                params = parse_qs(parsed.query)
                limit = _to_int(params.get("limit", ["120"])[0], default=120, min_value=20, max_value=300)
                ctx = ServiceContext.from_runtime(self.runtime)
                try:
                    graph = build_library_graph(ctx.repo, limit=limit)
                    latest = ctx.repo.get_latest_organize_run()
                    self._json_response(HTTPStatus.OK, {"graph": graph, "run": latest})
                finally:
                    ctx.close()
                return

            if parsed.path == "/api/library/history":
                params = parse_qs(parsed.query)
                limit = _to_int(params.get("limit", ["50"])[0], default=50, min_value=1, max_value=200)
                ctx = ServiceContext.from_runtime(self.runtime)
                try:
                    history = ctx.repo.list_import_history(limit=limit)
                    total_sources = len(history)
                    total_notes = sum(int(item.get("note_count") or 0) for item in history)
                    self._json_response(
                        HTTPStatus.OK,
                        {
                            "history": history,
                            "summary": {
                                "sources": total_sources,
                                "notes": total_notes,
                            },
                        },
                    )
                finally:
                    ctx.close()
                return

            self._json_response(HTTPStatus.NOT_FOUND, {"error": "Not found"})
        except Exception as exc:  # noqa: BLE001
            self._json_response(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _handle_api_post(self, parsed) -> None:
        try:
            payload = self._read_json()

            if parsed.path == "/api/import/start":
                kind = str(payload.get("kind") or "").strip()
                if kind not in {"text", "files", "file_path", "dir"}:
                    self._json_response(HTTPStatus.BAD_REQUEST, {"error": "kind must be text, files, file_path, or dir"})
                    return
                if kind == "text" and not str(payload.get("text") or "").strip():
                    self._json_response(HTTPStatus.BAD_REQUEST, {"error": "text is required"})
                    return
                if kind == "files":
                    files = payload.get("files") or []
                    if not isinstance(files, list) or not files:
                        self._json_response(HTTPStatus.BAD_REQUEST, {"error": "files is required"})
                        return
                if kind == "file_path" and not str(payload.get("path") or "").strip():
                    self._json_response(HTTPStatus.BAD_REQUEST, {"error": "path is required"})
                    return
                if kind == "dir" and not str(payload.get("path") or "").strip():
                    self._json_response(HTTPStatus.BAD_REQUEST, {"error": "path is required"})
                    return

                with self.runtime.import_lock:
                    running = [
                        job_id
                        for job_id, item in self.runtime.import_jobs.items()
                        if item.get("status") == "running"
                    ]
                    if running:
                        self._json_response(
                            HTTPStatus.OK,
                            {"status": "running", "job_id": running[0], "message": "already running"},
                        )
                        return

                job_id = str(uuid4())
                started_at = utc_now_iso()
                with self.runtime.import_lock:
                    self.runtime.import_jobs[job_id] = {
                        "job_id": job_id,
                        "kind": kind,
                        "status": "running",
                        "message": "导入任务已启动",
                        "percent": 1,
                        "started_at": started_at,
                        "updated_at": started_at,
                    }
                worker = threading.Thread(
                    target=_run_import_job,
                    args=(self.runtime, job_id, payload),
                    daemon=True,
                )
                worker.start()
                self._json_response(HTTPStatus.OK, {"status": "running", "job_id": job_id})
                return

            if parsed.path == "/api/import/text":
                text = str(payload.get("text") or "").strip()
                direction = str(payload.get("direction") or "提炼为可用于写作的知识摘要")
                should_export = bool(payload.get("export"))
                if not text:
                    self._json_response(HTTPStatus.BAD_REQUEST, {"error": "text is required"})
                    return
                ctx = ServiceContext.from_runtime(self.runtime)
                try:
                    result = process_source(
                        build_manual_source(text),
                        direction=direction,
                        repo=ctx.repo,
                        llm_client=ctx.llm_client,
                        settings=ctx.settings,
                        logger=ctx.logger,
                    )
                    if should_export and result.get("markdown"):
                        result["exported_path"] = str(
                            export_markdown(
                                result["markdown"],
                                ctx.settings.export_dir,
                                result.get("note_id") or result.get("source_id") or "note",
                            )
                        )
                    self._json_response(HTTPStatus.OK, result)
                finally:
                    ctx.close()
                return

            if parsed.path == "/api/import/files":
                files = payload.get("files") or []
                direction = str(payload.get("direction") or "提炼为可用于写作的知识摘要")
                should_export = bool(payload.get("export"))
                if not isinstance(files, list) or not files:
                    self._json_response(HTTPStatus.BAD_REQUEST, {"error": "files is required"})
                    return
                ctx = ServiceContext.from_runtime(self.runtime)
                try:
                    upload_dir = ctx.settings.import_dir / "uploaded"
                    upload_dir.mkdir(parents=True, exist_ok=True)
                    results: list[dict] = []
                    for item in files:
                        name = safe_filename(str(item.get("name") or "uploaded.txt"))
                        if not name.lower().endswith((".txt", ".md")):
                            results.append({"status": "failed", "file_name": name, "reason": "unsupported_extension"})
                            continue
                        content = str(item.get("content") or "")
                        target = upload_dir / name
                        target.write_text(content, encoding="utf-8")
                        source = build_file_source(target)
                        result = process_source(
                            source,
                            direction=direction,
                            repo=ctx.repo,
                            llm_client=ctx.llm_client,
                            settings=ctx.settings,
                            logger=ctx.logger,
                        )
                        if should_export and result.get("markdown"):
                            result["exported_path"] = str(
                                export_markdown(
                                    result["markdown"],
                                    ctx.settings.export_dir,
                                    result.get("note_id") or result.get("source_id") or "note",
                                )
                            )
                        results.append(result)
                    self._json_response(HTTPStatus.OK, {"results": results})
                finally:
                    ctx.close()
                return

            if parsed.path == "/api/import/file-path":
                file_path = str(payload.get("path") or "").strip()
                direction = str(payload.get("direction") or "提炼为可用于写作的知识摘要")
                should_export = bool(payload.get("export"))
                if not file_path:
                    self._json_response(HTTPStatus.BAD_REQUEST, {"error": "path is required"})
                    return
                path_obj = Path(file_path)
                if not path_obj.exists() or not path_obj.is_file():
                    self._json_response(HTTPStatus.BAD_REQUEST, {"error": f"file not found: {file_path}"})
                    return
                if path_obj.suffix.lower() not in {".txt", ".md"}:
                    self._json_response(HTTPStatus.BAD_REQUEST, {"error": "only .txt/.md are supported in this quick UI"})
                    return
                ctx = ServiceContext.from_runtime(self.runtime)
                try:
                    result = process_source(
                        build_file_source(path_obj),
                        direction=direction,
                        repo=ctx.repo,
                        llm_client=ctx.llm_client,
                        settings=ctx.settings,
                        logger=ctx.logger,
                    )
                    if should_export and result.get("markdown"):
                        result["exported_path"] = str(
                            export_markdown(
                                result["markdown"],
                                ctx.settings.export_dir,
                                result.get("note_id") or result.get("source_id") or "note",
                            )
                        )
                    self._json_response(HTTPStatus.OK, result)
                finally:
                    ctx.close()
                return

            if parsed.path == "/api/import/dir":
                dir_path = str(payload.get("path") or "").strip()
                direction = str(payload.get("direction") or "提炼为可用于写作的知识摘要")
                recursive = bool(payload.get("recursive", True))
                if not dir_path:
                    self._json_response(HTTPStatus.BAD_REQUEST, {"error": "path is required"})
                    return
                path_obj = Path(dir_path)
                if not path_obj.exists() or not path_obj.is_dir():
                    self._json_response(HTTPStatus.BAD_REQUEST, {"error": f"directory not found: {dir_path}"})
                    return
                ctx = ServiceContext.from_runtime(self.runtime)
                try:
                    updater = IncrementalUpdater(
                        repo=ctx.repo,
                        llm_client=ctx.llm_client,
                        settings=ctx.settings,
                        logger=ctx.logger,
                    )
                    results = updater.update_directory(path_obj, direction=direction, recursive=recursive)
                    self._json_response(HTTPStatus.OK, {"status": "ok", "results": results})
                finally:
                    ctx.close()
                return

            if parsed.path == "/api/ask":
                question = str(payload.get("question") or "").strip()
                if not question:
                    self._json_response(HTTPStatus.BAD_REQUEST, {"error": "question is required"})
                    return
                limit = _to_int(payload.get("limit"), default=15, min_value=1, max_value=30)
                should_export = bool(payload.get("export"))
                filename = str(payload.get("filename") or question)
                overwrite = bool(payload.get("overwrite"))
                ctx = ServiceContext.from_runtime(self.runtime)
                try:
                    service = SearchService(ctx.repo, ctx.llm_client, ctx.logger)
                    parsed_question, results = service.question_search(question, limit=limit)
                    answer_result = build_answer_result(question, results, llm_client=ctx.llm_client, logger=ctx.logger)
                    answer = answer_result.content
                    exported_path = None
                    if should_export:
                        exported_path = str(
                            export_markdown(
                                answer,
                                ctx.settings.export_dir,
                                filename,
                                overwrite=overwrite,
                            )
                        )
                    self._json_response(
                        HTTPStatus.OK,
                        {
                            "parsed_question": parsed_question,
                            "results": [_result_dict(item) for item in results],
                            "answer": answer,
                            "answer_mode": answer_result.mode,
                            "exported_path": exported_path,
                        },
                    )
                finally:
                    ctx.close()
                return

            if parsed.path == "/api/library/organize/start":
                mode = str(payload.get("mode") or "quick").strip().lower()
                if mode not in {"quick", "full"}:
                    self._json_response(HTTPStatus.BAD_REQUEST, {"error": "mode must be quick or full"})
                    return
                ctx = ServiceContext.from_runtime(self.runtime)
                try:
                    with self.runtime.organize_lock:
                        running = [
                            run_id
                            for run_id, item in self.runtime.organize_jobs.items()
                            if item.get("status") == "running"
                        ]
                        if running:
                            run_id = running[0]
                            self._json_response(
                                HTTPStatus.OK,
                                {"status": "running", "run_id": run_id, "message": "already running"},
                            )
                            return

                    run_id = str(uuid4())
                    started_at = utc_now_iso()
                    ctx.repo.create_organize_run(run_id, status="running", started_at=started_at)
                    with self.runtime.organize_lock:
                        self.runtime.organize_jobs[run_id] = {
                            "status": "running",
                            "message": f"{mode} job started",
                            "percent": 1,
                            "mode": mode,
                            "started_at": started_at,
                        }
                    worker = threading.Thread(
                        target=_run_organize_job,
                        args=(self.runtime, run_id, mode),
                        daemon=True,
                    )
                    worker.start()
                    self._json_response(HTTPStatus.OK, {"status": "running", "run_id": run_id, "mode": mode})
                finally:
                    ctx.close()
                return

            if parsed.path == "/api/embeddings/rebuild":
                mode = str(payload.get("mode") or "active").strip().lower()
                configs = configured_embedding_models() if mode == "all" else active_embedding_models()
                limit = _to_int(payload.get("limit"), default=10000, min_value=1, max_value=100000)
                ctx = ServiceContext.from_runtime(self.runtime)
                try:
                    stats = rebuild_unit_embeddings(ctx.repo, limit=limit, configs=configs)
                    self._json_response(HTTPStatus.OK, {"status": "ok", "stats": stats})
                finally:
                    ctx.close()
                return

            if parsed.path == "/api/notes/update":
                note_id = str(payload.get("note_id") or "").strip()
                if not note_id:
                    self._json_response(HTTPStatus.BAD_REQUEST, {"error": "note_id is required"})
                    return
                ctx = ServiceContext.from_runtime(self.runtime)
                try:
                    existing = ctx.repo.get_note(note_id)
                    if not existing:
                        self._json_response(HTTPStatus.NOT_FOUND, {"error": "Note not found"})
                        return
                    source = ctx.repo.get_source(str(existing.get("source_id")))
                    if not source:
                        self._json_response(HTTPStatus.NOT_FOUND, {"error": "Source not found"})
                        return
                    normalized = _normalize_note_payload(payload, existing)
                    canonical_keywords, keyword_links = canonicalize_keywords(
                        ctx.repo,
                        [*normalized["keywords"], *normalized["themes"]],
                        source="manual_edit",
                    )
                    if canonical_keywords:
                        normalized["keywords"] = canonical_keywords[:12]
                    markdown_content = build_note_markdown(normalized, source)
                    updated = ctx.repo.update_note_structured(
                        note_id=note_id,
                        title=normalized["title"],
                        note_type=normalized["note_type"],
                        themes=normalized["themes"],
                        summary=normalized["summary"],
                        faithful_content=normalized["faithful_content"],
                        key_points=normalized["key_points"],
                        usage_scenarios=normalized["usage_scenarios"],
                        user_insights=normalized["user_insights"],
                        keywords=normalized["keywords"],
                        source_excerpt=normalized["source_excerpt"],
                        markdown_content=markdown_content,
                        updated_at=utc_now_iso(),
                    )
                    if not updated:
                        self._json_response(HTTPStatus.NOT_FOUND, {"error": "Note not found"})
                        return
                    ctx.repo.update_unit_for_note(
                        note_id=note_id,
                        title=normalized["title"],
                        content=_unit_content_from_payload(normalized),
                        evidence=normalized["source_excerpt"],
                        note_type=normalized["note_type"],
                        attributes={
                            "themes": normalized["themes"],
                            "faithful_content": normalized["faithful_content"][:1200],
                            "key_points": normalized["key_points"],
                            "usage_scenarios": normalized["usage_scenarios"],
                            "user_insights": normalized["user_insights"],
                        },
                        updated_at=utc_now_iso(),
                    )
                    unit = ctx.repo.get_unit_by_note_id(note_id)
                    if unit:
                        ctx.repo.replace_unit_keywords(unit["unit_id"], keyword_links)
                        index_unit_embeddings(
                            ctx.repo,
                            {
                                **unit,
                                "title": normalized["title"],
                                "content": _unit_content_from_payload(normalized),
                                "evidence": normalized["source_excerpt"],
                                "note_summary": normalized["summary"],
                                "note_faithful_content": normalized["faithful_content"],
                                "themes": normalized["themes"],
                                "keywords": normalized["keywords"],
                            },
                        )
                    self._json_response(HTTPStatus.OK, {"status": "ok", "note": ctx.repo.get_note(note_id)})
                finally:
                    ctx.close()
                return

            if parsed.path == "/api/export/markdown":
                content = str(payload.get("content") or "")
                filename = str(payload.get("filename") or f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                overwrite = bool(payload.get("overwrite"))
                if not content.strip():
                    self._json_response(HTTPStatus.BAD_REQUEST, {"error": "content is required"})
                    return
                settings = self.runtime.settings
                path = export_markdown(content, settings.export_dir, filename, overwrite=overwrite)
                self._json_response(HTTPStatus.OK, {"status": "ok", "exported_path": str(path)})
                return

            if parsed.path == "/api/export/note":
                note_id = str(payload.get("note_id") or "").strip()
                filename = str(payload.get("filename") or "")
                overwrite = bool(payload.get("overwrite"))
                if not note_id:
                    self._json_response(HTTPStatus.BAD_REQUEST, {"error": "note_id is required"})
                    return
                ctx = ServiceContext.from_runtime(self.runtime)
                try:
                    note = ctx.repo.get_note(note_id)
                    if not note:
                        self._json_response(HTTPStatus.NOT_FOUND, {"error": "Note not found"})
                        return
                    target_name = filename or str(note.get("title") or note_id)
                    path = export_markdown(
                        str(note.get("markdown_content") or ""),
                        ctx.settings.export_dir,
                        target_name,
                        overwrite=overwrite,
                    )
                    self._json_response(HTTPStatus.OK, {"status": "ok", "exported_path": str(path)})
                finally:
                    ctx.close()
                return

            self._json_response(HTTPStatus.NOT_FOUND, {"error": "Not found"})
        except Exception as exc:  # noqa: BLE001
            self._json_response(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _serve_static(self, path: str) -> None:
        normalized = path.strip() or "/"
        if normalized == "/":
            normalized = "/index.html"
        full = (self.runtime.frontend_dir / normalized.lstrip("/")).resolve()
        frontend_root = self.runtime.frontend_dir.resolve()
        if not str(full).startswith(str(frontend_root)):
            self._json_response(HTTPStatus.FORBIDDEN, {"error": "Forbidden"})
            return
        if not full.exists() or not full.is_file():
            self._json_response(HTTPStatus.NOT_FOUND, {"error": "File not found"})
            return
        data = full.read_bytes()
        content_type = mimetypes.guess_type(str(full))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length) if length > 0 else b"{}"
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _json_response(self, status: HTTPStatus, payload: dict) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(raw)

    def _set_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")

    def log_message(self, format: str, *args) -> None:
        if self.runtime.logger:
            self.runtime.logger.info("api %s", format % args)


def _to_int(value, *, default: int, min_value: int, max_value: int) -> int:
    try:
        num = int(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(num, max_value))


def _first_or_none(values: list[str] | None) -> str | None:
    if not values:
        return None
    text = values[0].strip()
    return text or None


def _result_dict(item) -> dict:
    return {
        "note_id": item.note_id,
        "source_id": item.source_id,
        "title": item.title,
        "note_type": item.note_type,
        "summary": item.summary,
        "themes": item.themes,
        "keywords": item.keywords,
        "file_path": item.file_path,
        "imported_at": item.imported_at,
        "score": item.score,
        "snippet": item.snippet,
        "relevance_score": item.relevance_score,
        "relevance_level": item.relevance_level,
        "relevance_reason": item.relevance_reason,
        "relation_type": item.relation_type,
        "relation_strength": item.relation_strength,
        "match_source": item.match_source,
    }


def _embedding_config_dict(config, counts: dict[str, int]) -> dict:
    return {
        "key": config.key,
        "provider": config.provider,
        "model_path": config.model_path,
        "model_name": config.model_name,
        "weight": config.weight,
        "indexed_units": counts.get(config.model_name, 0),
    }


def _split_lines(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [line.strip() for line in str(value).splitlines() if line.strip()]


def _split_csv(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value).replace("，", ",").split(",") if part.strip()]


def _normalize_note_payload(payload: dict, existing: dict) -> dict:
    return {
        "title": str(payload.get("title") or existing.get("title") or "未命名条目").strip(),
        "note_type": str(payload.get("note_type") or existing.get("note_type") or "未分类").strip(),
        "themes": _split_csv(payload.get("themes") if "themes" in payload else existing.get("themes")),
        "summary": str(payload.get("summary") if "summary" in payload else existing.get("summary") or "").strip(),
        "faithful_content": str(
            payload.get("faithful_content") if "faithful_content" in payload else existing.get("faithful_content") or ""
        ).strip(),
        "key_points": _split_lines(payload.get("key_points") if "key_points" in payload else existing.get("key_points")),
        "usage_scenarios": _split_lines(
            payload.get("usage_scenarios") if "usage_scenarios" in payload else existing.get("usage_scenarios")
        ),
        "user_insights": str(
            payload.get("user_insights") if "user_insights" in payload else existing.get("user_insights") or ""
        ).strip(),
        "keywords": _split_csv(payload.get("keywords") if "keywords" in payload else existing.get("keywords")),
        "source_excerpt": str(
            payload.get("source_excerpt") if "source_excerpt" in payload else existing.get("source_excerpt") or ""
        ).strip(),
    }


def _unit_content_from_payload(payload: dict) -> str:
    parts: list[str] = []
    if payload.get("faithful_content"):
        parts.append(str(payload["faithful_content"]))
    if payload.get("summary"):
        parts.append(str(payload["summary"]))
    for item in payload.get("key_points") or []:
        parts.append(f"- {item}")
    if payload.get("user_insights"):
        parts.append(str(payload["user_insights"]))
    return "\n".join(parts).strip() or str(payload.get("source_excerpt") or "")


def _run_import_job(runtime: Runtime, job_id: str, payload: dict) -> None:
    kind = str(payload.get("kind") or "").strip()
    direction = str(payload.get("direction") or "提炼为可用于写作的知识摘要")
    should_export = bool(payload.get("export"))

    def set_live(status: str, message: str, percent: int, **extra) -> None:
        with runtime.import_lock:
            current = dict(runtime.import_jobs.get(job_id, {}))
            current.update(
                {
                    "job_id": job_id,
                    "kind": kind,
                    "status": status,
                    "message": message,
                    "percent": max(0, min(100, int(percent))),
                    "updated_at": utc_now_iso(),
                    **extra,
                }
            )
            runtime.import_jobs[job_id] = current

    def map_source_progress(item: dict, *, start: int = 5, end: int = 92, **extra) -> None:
        source_percent = int(item.get("percent") or 0)
        overall = start + int((source_percent / 100) * max(1, end - start))
        set_live(
            "running",
            str(item.get("message") or "正在处理来源"),
            min(end, overall),
            stage=item.get("stage"),
            current_chunk=item.get("current_chunk"),
            total_chunks=item.get("total_chunks"),
            **extra,
        )

    conn = connect(runtime.settings.db_path)
    init_db(conn)
    repo = KnowledgeRepository(conn)
    llm_client = QwenClient(
        base_url=runtime.settings.llm_base_url,
        model=runtime.settings.llm_model,
        timeout_seconds=runtime.settings.llm_timeout_seconds,
    )
    logger = setup_logger(runtime.settings.log_dir, f"{runtime.logger_name}_import")
    try:
        set_live("running", "正在准备导入任务", 2)
        if kind == "text":
            result = process_source(
                build_manual_source(str(payload.get("text") or "")),
                direction=direction,
                repo=repo,
                llm_client=llm_client,
                settings=runtime.settings,
                logger=logger,
                progress_cb=lambda item: map_source_progress(item),
            )
            if should_export and result.get("markdown"):
                set_live("running", "正在导出 Markdown", 94)
                result["exported_path"] = str(
                    export_markdown(
                        result["markdown"],
                        runtime.settings.export_dir,
                        result.get("note_id") or result.get("source_id") or "note",
                    )
                )
            set_live("completed", "导入完成", 100, result=result)
            return

        if kind == "file_path":
            file_path = str(payload.get("path") or "").strip()
            path_obj = Path(file_path)
            if not path_obj.exists() or not path_obj.is_file():
                raise ValueError(f"file not found: {file_path}")
            if path_obj.suffix.lower() not in {".txt", ".md"}:
                raise ValueError("only .txt/.md are supported in this quick UI")
            result = process_source(
                build_file_source(path_obj),
                direction=direction,
                repo=repo,
                llm_client=llm_client,
                settings=runtime.settings,
                logger=logger,
                progress_cb=lambda item: map_source_progress(item, file_path=str(path_obj)),
            )
            if should_export and result.get("markdown"):
                set_live("running", "正在导出 Markdown", 94)
                result["exported_path"] = str(
                    export_markdown(
                        result["markdown"],
                        runtime.settings.export_dir,
                        result.get("note_id") or result.get("source_id") or "note",
                    )
                )
            set_live("completed", "导入完成", 100, result=result)
            return

        if kind == "files":
            files = payload.get("files") or []
            if not isinstance(files, list) or not files:
                raise ValueError("files is required")
            upload_dir = runtime.settings.import_dir / "uploaded"
            upload_dir.mkdir(parents=True, exist_ok=True)
            results: list[dict] = []
            total_files = len(files)
            for index, item in enumerate(files):
                name = safe_filename(str(item.get("name") or "uploaded.txt"))
                file_start = 5 + int((index / total_files) * 87)
                file_end = 5 + int(((index + 1) / total_files) * 87)
                set_live(
                    "running",
                    f"正在准备第 {index + 1}/{total_files} 个文件：{name}",
                    file_start,
                    current_file=index + 1,
                    total_files=total_files,
                )
                if not name.lower().endswith((".txt", ".md")):
                    results.append({"status": "failed", "file_name": name, "reason": "unsupported_extension"})
                    continue
                content = str(item.get("content") or "")
                target = upload_dir / name
                target.write_text(content, encoding="utf-8")
                try:
                    result = process_source(
                        build_file_source(target),
                        direction=direction,
                        repo=repo,
                        llm_client=llm_client,
                        settings=runtime.settings,
                        logger=logger,
                        progress_cb=lambda event, start=file_start, end=file_end, idx=index, fname=name: map_source_progress(
                            event,
                            start=start,
                            end=end,
                            current_file=idx + 1,
                            total_files=total_files,
                            file_name=fname,
                        ),
                    )
                    if should_export and result.get("markdown"):
                        result["exported_path"] = str(
                            export_markdown(
                                result["markdown"],
                                runtime.settings.export_dir,
                                result.get("note_id") or result.get("source_id") or "note",
                            )
                        )
                    results.append(result)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Failed to import uploaded file %s: %s", name, exc)
                    results.append({"status": "failed", "file_name": name, "reason": str(exc)})
            set_live("completed", "批量导入完成", 100, result={"results": results})
            return

        if kind == "dir":
            dir_path = str(payload.get("path") or "").strip()
            path_obj = Path(dir_path)
            if not path_obj.exists() or not path_obj.is_dir():
                raise ValueError(f"directory not found: {dir_path}")
            updater = IncrementalUpdater(
                repo=repo,
                llm_client=llm_client,
                settings=runtime.settings,
                logger=logger,
            )
            results = updater.update_directory(
                path_obj,
                direction=direction,
                recursive=bool(payload.get("recursive", True)),
                progress_cb=lambda item: set_live(
                    "failed" if str(item.get("status") or "") == "failed" else "running",
                    str(item.get("message") or ""),
                    min(99, int(item.get("percent") or 0)),
                    stage=item.get("stage"),
                    current_file=item.get("current_file"),
                    total_files=item.get("total_files"),
                    file_path=item.get("file_path"),
                    current_chunk=item.get("current_chunk"),
                    total_chunks=item.get("total_chunks"),
                ),
            )
            set_live("completed", "目录增量导入完成", 100, result={"status": "ok", "results": results})
            return

        raise ValueError(f"Unsupported import kind: {kind}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Import job failed: %s", exc)
        set_live("failed", str(exc), 100, error=str(exc))
    finally:
        conn.close()


def _run_organize_job(runtime: Runtime, run_id: str, mode: str = "quick") -> None:
    def set_live(status: str, message: str, percent: int) -> None:
        with runtime.organize_lock:
            runtime.organize_jobs[run_id] = {
                "status": status,
                "message": message,
                "percent": percent,
                "mode": mode,
                "updated_at": utc_now_iso(),
            }

    conn = connect(runtime.settings.db_path)
    init_db(conn)
    repo = KnowledgeRepository(conn)
    logger = setup_logger(runtime.settings.log_dir, f"{runtime.logger_name}_organizer")
    try:
        result = organize_knowledge_base(
            repo,
            run_id=run_id,
            logger=logger,
            mode=mode,
            progress_cb=lambda item: set_live(
                str(item.get("status") or "running"),
                str(item.get("message") or ""),
                int(item.get("percent") or 0),
            ),
        )
        repo.update_organize_run(
            run_id,
            status="completed",
            finished_at=utc_now_iso(),
            stats=result.get("stats") or {},
            report_markdown=result.get("report_markdown") or "",
            error_message=None,
        )
        repo.delete_relations_except_run(run_id)
        set_live("completed", "Knowledge organization finished", 100)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Organize run failed: %s", exc)
        repo.update_organize_run(
            run_id,
            status="failed",
            finished_at=utc_now_iso(),
            stats={},
            report_markdown=None,
            error_message=str(exc),
        )
        set_live("failed", str(exc), 100)
    finally:
        conn.close()


def run_server(*, host: str, port: int) -> None:
    settings = load_settings()
    conn = connect(settings.db_path)
    init_db(conn)
    runtime = Runtime(
        settings=settings,
        frontend_dir=Path(__file__).resolve().parents[1] / "frontend",
        logger_name="local_knowledge_api",
        conn=conn,
        repo=KnowledgeRepository(conn),
        llm_client=QwenClient(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            timeout_seconds=settings.llm_timeout_seconds,
        ),
        logger=setup_logger(settings.log_dir, "local_knowledge_api"),
    )
    if not runtime.frontend_dir.exists():
        raise RuntimeError(f"Frontend directory not found: {runtime.frontend_dir}")
    backfill_v2_structures(runtime.repo, runtime.logger)

    handler_cls = type("BoundHandler", (RequestHandler,), {"runtime": runtime})
    server = HTTPServer((host, port), handler_cls)
    logger = runtime.logger
    logger.info("API server running at http://%s:%s", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("API server stopped by keyboard interrupt")
    finally:
        server.server_close()
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LocalKnowledge frontend API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
