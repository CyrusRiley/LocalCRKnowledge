from __future__ import annotations

from logging import Logger
from pathlib import Path

from app.backend.config import Settings
from app.backend.database.repository import KnowledgeRepository
from app.backend.importer.file_scanner import scan_text_files
from app.backend.importer.text_importer import build_file_source
from app.backend.llm.client import QwenClient
from app.backend.workflow import process_source


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

    def update_directory(self, directory: Path, *, direction: str, recursive: bool = True) -> list[dict]:
        files = scan_text_files(directory, recursive=recursive)
        results: list[dict] = []
        self.logger.info("Incremental scan %s found %s files", directory, len(files))
        for file_path in files:
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
                    )
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.exception("Failed to update file %s: %s", file_path, exc)
                results.append({"status": "failed", "file_path": str(file_path), "reason": str(exc)})
        return results
