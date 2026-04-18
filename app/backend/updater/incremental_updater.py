from __future__ import annotations

from logging import Logger
from pathlib import Path
from typing import Callable

from app.backend.config import Settings
from app.backend.database.repository import KnowledgeRepository
from app.backend.importer.file_scanner import scan_text_files
from app.backend.importer.text_importer import build_file_source
from app.backend.llm.client import QwenClient
from app.backend.workflow import process_source

ProgressCallback = Callable[[dict], None]


class IncrementalUpdater:
    def __init__(
        self,
        *,
        repo: KnowledgeRepository,
        llm_client: QwenClient,
        settings: Settings,
        logger: Logger,
    ):
        self.repo = repo
        self.llm_client = llm_client
        self.settings = settings
        self.logger = logger

    def update_directory(
        self,
        directory: Path,
        *,
        direction: str,
        recursive: bool = True,
        progress_cb: ProgressCallback | None = None,
    ) -> list[dict]:
        def emit(stage: str, message: str, percent: int, **extra) -> None:
            if progress_cb:
                progress_cb(
                    {
                        "status": "running",
                        "stage": stage,
                        "message": message,
                        "percent": max(0, min(100, int(percent))),
                        **extra,
                    }
                )

        emit("scan", f"正在扫描目录：{directory}", 1)
        files = scan_text_files(directory, recursive=recursive)
        results: list[dict] = []
        self.logger.info("Incremental scan %s found %s files", directory, len(files))
        if not files:
            emit("completed", "未发现可导入的 txt/md 文件", 100, total_files=0)
            return results

        total_files = len(files)
        for index, file_path in enumerate(files):
            file_start = int((index / total_files) * 100)
            file_span = max(1, int(100 / total_files))

            def source_progress(item: dict, *, file_index: int = index, file_path: Path = file_path) -> None:
                source_percent = int(item.get("percent") or 0)
                overall = min(99, int((file_index / total_files) * 100 + (source_percent / 100) * file_span))
                emit(
                    str(item.get("stage") or "source"),
                    str(item.get("message") or f"正在处理 {file_path.name}"),
                    overall,
                    current_file=file_index + 1,
                    total_files=total_files,
                    file_path=str(file_path),
                    current_chunk=item.get("current_chunk"),
                    total_chunks=item.get("total_chunks"),
                )

            try:
                source = build_file_source(file_path)
                results.append(
                    process_source(
                        source,
                        direction=direction,
                        repo=self.repo,
                        llm_client=self.llm_client,
                        settings=self.settings,
                        logger=self.logger,
                        progress_cb=source_progress,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.exception("Failed to update file %s: %s", file_path, exc)
                results.append({"status": "failed", "file_path": str(file_path), "reason": str(exc)})
                emit(
                    "file_failed",
                    f"文件处理失败：{file_path.name}",
                    min(99, file_start + file_span),
                    current_file=index + 1,
                    total_files=total_files,
                    file_path=str(file_path),
                )
        emit("completed", f"目录处理完成，共处理 {len(results)} 个文件", 100, total_files=total_files)
        return results
