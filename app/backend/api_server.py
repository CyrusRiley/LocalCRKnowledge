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
from app.backend.importer.text_importer import build_file_source, build_manual_source
from app.backend.llm.client import QwenClient
from app.backend.organizer.knowledge_organizer import organize_knowledge_base
from app.backend.retrieval.answer_builder import build_answer
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

            self._json_response(HTTPStatus.NOT_FOUND, {"error": "Not found"})
        except Exception as exc:  # noqa: BLE001
            self._json_response(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _handle_api_post(self, parsed) -> None:
        try:
            payload = self._read_json()

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
                    answer = build_answer(question, results, llm_client=ctx.llm_client, logger=ctx.logger)
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
                            "exported_path": exported_path,
                        },
                    )
                finally:
                    ctx.close()
                return

            if parsed.path == "/api/library/organize/start":
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
                            "message": "Job started",
                            "percent": 1,
                            "started_at": started_at,
                        }
                    worker = threading.Thread(
                        target=_run_organize_job,
                        args=(self.runtime, run_id),
                        daemon=True,
                    )
                    worker.start()
                    self._json_response(HTTPStatus.OK, {"status": "running", "run_id": run_id})
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
                    markdown_content = build_note_markdown(normalized, source)
                    updated = ctx.repo.update_note_structured(
                        note_id=note_id,
                        title=normalized["title"],
                        note_type=normalized["note_type"],
                        themes=normalized["themes"],
                        summary=normalized["summary"],
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


def _run_organize_job(runtime: Runtime, run_id: str) -> None:
    def set_live(status: str, message: str, percent: int) -> None:
        with runtime.organize_lock:
            runtime.organize_jobs[run_id] = {
                "status": status,
                "message": message,
                "percent": percent,
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
