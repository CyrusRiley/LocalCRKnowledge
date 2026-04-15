from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from app.backend.config import Settings, load_settings
from app.backend.database.db import connect, init_db
from app.backend.database.repository import KnowledgeRepository
from app.backend.export.markdown_exporter import export_markdown
from app.backend.importer.pdf_importer import build_pdf_source
from app.backend.importer.text_importer import build_file_source, build_manual_source
from app.backend.llm.client import QwenClient
from app.backend.retrieval.answer_builder import build_answer
from app.backend.retrieval.search_service import SearchService
from app.backend.updater.incremental_updater import IncrementalUpdater
from app.backend.utils.logger import setup_logger
from app.backend.workflow import process_source


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "disable_llm", False):
        os.environ["LK_DISABLE_LLM"] = "1"

    settings = load_settings()
    logger = setup_logger(settings.log_dir)
    conn = connect(settings.db_path)
    init_db(conn)
    repo = KnowledgeRepository(conn)
    llm_client = QwenClient(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        timeout_seconds=settings.llm_timeout_seconds,
    )

    if args.command == "init-db":
        print_json({"status": "ok", "db_path": actual_database_path(conn) or str(settings.db_path)})
    elif args.command == "import-text":
        result = process_source(
            build_manual_source(args.text),
            direction=args.direction,
            repo=repo,
            llm_client=llm_client,
            settings=settings,
            logger=logger,
        )
        maybe_export_result(result, settings, args.export)
        print_json(_public_result(result))
    elif args.command == "import-file":
        source = build_source_from_path(Path(args.path))
        result = process_source(
            source,
            direction=args.direction,
            repo=repo,
            llm_client=llm_client,
            settings=settings,
            logger=logger,
        )
        maybe_export_result(result, settings, args.export)
        print_json(_public_result(result))
    elif args.command == "import-dir":
        updater = IncrementalUpdater(repo=repo, llm_client=llm_client, settings=settings, logger=logger)
        results = updater.update_directory(Path(args.path), direction=args.direction, recursive=not args.no_recursive)
        print_json({"status": "ok", "results": results})
    elif args.command == "list":
        print_json({"notes": repo.list_notes(limit=args.limit, theme=args.theme, note_type=args.note_type)})
    elif args.command == "search":
        service = SearchService(repo, llm_client, logger)
        results = service.keyword_search(args.query, limit=args.limit)
        print_json({"results": [_result_dict(item) for item in results]})
    elif args.command == "theme":
        service = SearchService(repo, llm_client, logger)
        print_json({"results": service.theme_search(args.theme, limit=args.limit)})
    elif args.command == "ask":
        service = SearchService(repo, llm_client, logger)
        parsed, results = service.question_search(args.question, limit=args.limit)
        answer = build_answer(args.question, results, llm_client=llm_client, logger=logger)
        exported = export_markdown(answer, settings.export_dir, args.filename or args.question, overwrite=args.overwrite) if args.export else None
        print_json(
            {
                "parsed_question": parsed,
                "results": [_result_dict(item) for item in results],
                "answer": answer,
                "exported_path": str(exported) if exported else None,
            }
        )
    elif args.command == "export-note":
        note = repo.get_note(args.note_id)
        if not note:
            raise SystemExit(f"Note not found: {args.note_id}")
        export_dir = Path(args.out_dir) if args.out_dir else settings.export_dir
        path = export_markdown(note["markdown_content"], export_dir, args.filename or note["title"], overwrite=args.overwrite)
        print_json({"status": "ok", "exported_path": str(path)})
    else:
        parser.print_help()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local knowledge organizer backend")
    parser.add_argument("--disable-llm", action="store_true", help="Use deterministic fallback instead of calling Qwen.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db")

    import_text = sub.add_parser("import-text")
    import_text.add_argument("--text", required=True)
    import_text.add_argument("--direction", default="提炼为可用于写作的知识摘要")
    import_text.add_argument("--export", action="store_true")

    import_file = sub.add_parser("import-file")
    import_file.add_argument("path")
    import_file.add_argument("--direction", default="提炼为可用于写作的知识摘要")
    import_file.add_argument("--export", action="store_true")

    import_dir = sub.add_parser("import-dir")
    import_dir.add_argument("path")
    import_dir.add_argument("--direction", default="提炼为可用于写作的知识摘要")
    import_dir.add_argument("--no-recursive", action="store_true")

    list_cmd = sub.add_parser("list")
    list_cmd.add_argument("--limit", type=int, default=50)
    list_cmd.add_argument("--theme")
    list_cmd.add_argument("--note-type")

    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=10)

    theme = sub.add_parser("theme")
    theme.add_argument("theme")
    theme.add_argument("--limit", type=int, default=10)

    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--limit", type=int, default=15)
    ask.add_argument("--export", action="store_true")
    ask.add_argument("--filename")
    ask.add_argument("--overwrite", action="store_true")

    export_note = sub.add_parser("export-note")
    export_note.add_argument("note_id")
    export_note.add_argument("--out-dir")
    export_note.add_argument("--filename")
    export_note.add_argument("--overwrite", action="store_true")
    return parser


def build_source_from_path(path: Path):
    if path.suffix.lower() == ".pdf":
        return build_pdf_source(path)
    return build_file_source(path)


def maybe_export_result(result: dict, settings: Settings, should_export: bool) -> None:
    if should_export and result.get("markdown"):
        title = result.get("note_id") or result.get("source_id") or "note"
        result["exported_path"] = str(export_markdown(result["markdown"], settings.export_dir, title))


def print_json(data: dict[str, Any]) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def actual_database_path(conn) -> str | None:
    row = conn.execute("PRAGMA database_list").fetchone()
    return row["file"] if row and row["file"] else None


def _public_result(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "markdown"}


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


if __name__ == "__main__":
    main()
