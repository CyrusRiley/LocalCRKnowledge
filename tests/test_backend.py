from __future__ import annotations

import os
import sqlite3
import unittest
from pathlib import Path

from app.backend.config import Settings
from app.backend.database.db import connect, init_db
from app.backend.database.repository import KnowledgeRepository
from app.backend.formatter.markdown_builder import build_note_markdown
from app.backend.llm.client import QwenClient
from app.backend.llm.parser import parse_note_json, parse_section_markers_json
from app.backend.keywords.normalizer import canonicalize_keywords
from app.backend.maintenance.v2_backfill import backfill_v2_structures
from app.backend.models import ChunkRecord, KnowledgeGroup, KnowledgeUnit, SourceRecord, StructuredNote
from app.backend.organizer.knowledge_organizer import organize_knowledge_base
from app.backend.preprocess.cleaner import build_chunk_contexts, chunk_text, normalize_text, preprocess_text
from app.backend.relations.relation_builder import build_structural_relations
from app.backend.retrieval.search_service import SearchService
from app.backend.utils.logger import setup_logger
from app.backend.utils.time_utils import utc_now_iso
from app.backend.workflow import process_source


class BackendTests(unittest.TestCase):
    def test_preprocess_normalizes_and_chunks(self) -> None:
        text = "第一段\r\n\r\n\r\n第二段\t  ABM 路径偏好"
        result = preprocess_text(text, chunk_size=8, chunk_overlap=2)
        self.assertEqual(normalize_text(text), "第一段\n\n第二段 ABM 路径偏好")
        self.assertGreaterEqual(len(result.chunks), 1)
        self.assertIn("abm", result.pre_keywords)
        self.assertTrue(chunk_text("abc", chunk_size=10))
        contexts = build_chunk_contexts(result.chunks, context_chars=6)
        self.assertEqual(len(contexts), len(result.chunks))
        self.assertEqual(contexts[0].chunk_index, 0)

    def test_preprocess_splits_by_heading_units(self) -> None:
        text = """# xxx发展的七个要点

一、技术路径演进
这里讨论第一个要点，包含完整说明。

二、数据来源变化
这里讨论第二个要点，包含完整说明。

三、评价体系更新
这里讨论第三个要点。
"""
        chunks = chunk_text(text, chunk_size=1000, chunk_overlap=0)
        self.assertEqual(len(chunks), 3)
        self.assertIn("技术路径演进", chunks[0])
        self.assertIn("数据来源变化", chunks[1])
        self.assertIn("评价体系更新", chunks[2])

    def test_parser_recovers_json_payload(self) -> None:
        content = '```json\n{"title":"标题","themes":["公共空间"],"key_points":["要点"]}\n```'
        payload = parse_note_json(content, fallback_title="候选", source_text="原文内容")
        self.assertEqual(payload["title"], "标题")
        self.assertEqual(payload["note_type"], "未分类")
        self.assertEqual(payload["themes"], ["公共空间"])

    def test_parser_recovers_structure_marker_array(self) -> None:
        content = '```json\n[{"title":"技术路径","start_quote":"一、技术路径","end_quote":"完整说明。"}]\n```'
        markers = parse_section_markers_json(content)
        self.assertEqual(markers[0]["title"], "技术路径")
        self.assertEqual(markers[0]["start_quote"], "一、技术路径")

    def test_markdown_template_contains_source_info(self) -> None:
        source = _source("科研灵感：慢跑路径偏好")
        markdown = build_note_markdown(
            {
                "title": "慢跑路径偏好",
                "note_type": "科研灵感",
                "themes": ["行为模拟"],
                "summary": "摘要",
                "key_points": ["要点"],
                "usage_scenarios": ["论文第二章"],
                "user_insights": "",
                "keywords": ["ABM"],
                "source_excerpt": "摘录",
            },
            source,
        )
        self.assertIn("# 慢跑路径偏好", markdown)
        self.assertIn(f"- source_id: {source.source_id}", markdown)

    def test_repository_fts_search(self) -> None:
        conn = memory_conn()
        repo = KnowledgeRepository(conn)
        source = _source("ABM 路径偏好")
        source.clean_text = source.raw_text
        repo.upsert_source(source)
        note = StructuredNote(
            note_id="note-1",
            source_id=source.source_id,
            title="慢跑路径偏好",
            note_type="科研灵感",
            themes=["行为模拟"],
            summary="ABM 可用于解释路径偏好。",
            key_points=["路径偏好与公共空间有关"],
            usage_scenarios=["论文第二章"],
            keywords=["ABM", "路径偏好"],
            source_excerpt="ABM 路径偏好",
            markdown_content="ABM 路径偏好 公共空间",
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
        )
        repo.insert_note(note, [ChunkRecord("chunk-1", "note-1", "ABM 路径偏好", 0, "source", ["ABM"])])
        results = repo.search_notes("ABM", limit=5)
        self.assertEqual(results[0].title, "慢跑路径偏好")

    def test_repository_update_note_refreshes_fts(self) -> None:
        conn = memory_conn()
        repo = KnowledgeRepository(conn)
        source = _source("原始内容")
        source.clean_text = source.raw_text
        repo.upsert_source(source)
        note = StructuredNote(
            note_id="note-update",
            source_id=source.source_id,
            title="原始标题",
            note_type="科研灵感",
            themes=["行为模拟"],
            summary="原始摘要",
            key_points=["原始要点"],
            usage_scenarios=["原始场景"],
            keywords=["原始关键词"],
            source_excerpt="原始摘录",
            markdown_content="原始 markdown",
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
        )
        repo.insert_note(note, [])
        changed = repo.update_note_structured(
            note_id="note-update",
            title="更新标题",
            note_type="论文修改意见",
            themes=["更新主题"],
            summary="更新摘要",
            key_points=["更新要点"],
            usage_scenarios=["更新场景"],
            user_insights="更新想法",
            keywords=["更新关键词"],
            source_excerpt="更新摘录",
            markdown_content="更新 markdown 包含 新词汇",
            updated_at=utc_now_iso(),
        )
        self.assertTrue(changed)
        results = repo.search_notes("新词汇", limit=5)
        self.assertEqual(results[0].note_id, "note-update")

    def test_workflow_fallback_without_llm(self) -> None:
        os.environ["LK_DISABLE_LLM"] = "1"
        conn = memory_conn()
        repo = KnowledgeRepository(conn)
        settings = Settings(db_path=Path(":memory:"), log_dir=Path("data/logs"), export_dir=Path("data/exports"))
        logger = setup_logger(Path("data/logs"), name=f"test_logger_{id(conn)}")
        result = process_source(
            _source("科研灵感：ABM 可用于分析慢跑路径偏好。"),
            direction="提炼为科研灵感条目",
            repo=repo,
            llm_client=QwenClient("http://127.0.0.1:8080/v1", "qwen2.5", timeout_seconds=1),
            settings=settings,
            logger=logger,
        )
        self.assertEqual(result["status"], "failed")
        self.assertGreaterEqual(result["note_count"], 1)
        self.assertTrue(repo.search_notes("ABM", limit=5))
        os.environ.pop("LK_DISABLE_LLM", None)

    def test_workflow_long_text_split_into_multiple_notes(self) -> None:
        os.environ["LK_DISABLE_LLM"] = "1"
        conn = memory_conn()
        repo = KnowledgeRepository(conn)
        settings = Settings(
            db_path=Path(":memory:"),
            log_dir=Path("data/logs"),
            export_dir=Path("data/exports"),
            chunk_size=120,
            chunk_overlap=0,
        )
        logger = setup_logger(Path("data/logs"), name=f"test_logger_{id(conn)}")
        long_text = (
            ("第一段：关于慢跑路径偏好的观察，强调路径连通性与遮荫，记录晨间与晚间差异。 " * 6)
            + "\n\n"
            + ("第二段：关于公共空间设施密度与停留行为之间关系，比较不同街区样本。 " * 6)
            + "\n\n"
            + ("第三段：关于ABM参数标定时需要考虑年龄结构差异，并标注数据来源。 " * 6)
        )
        result = process_source(
            _source(long_text),
            direction="提炼为科研灵感条目",
            repo=repo,
            llm_client=QwenClient("http://127.0.0.1:8080/v1", "qwen2.5", timeout_seconds=1),
            settings=settings,
            logger=logger,
        )
        self.assertGreater(result["note_count"], 1)
        count = conn.execute("SELECT COUNT(*) AS cnt FROM notes_structured WHERE source_id = ?", (result["source_id"],)).fetchone()["cnt"]
        self.assertEqual(count, result["note_count"])
        group_count = conn.execute("SELECT COUNT(*) AS cnt FROM knowledge_groups WHERE source_id = ?", (result["source_id"],)).fetchone()["cnt"]
        unit_count = conn.execute("SELECT COUNT(*) AS cnt FROM knowledge_units WHERE source_id = ?", (result["source_id"],)).fetchone()["cnt"]
        self.assertEqual(group_count, 1)
        self.assertEqual(unit_count, result["note_count"])
        os.environ.pop("LK_DISABLE_LLM", None)

    def test_keyword_canonicalization_merges_suffix_aliases(self) -> None:
        conn = memory_conn()
        repo = KnowledgeRepository(conn)
        canonical, links = canonicalize_keywords(repo, ["行为模拟技术", "行为模拟方法"], source="test")
        self.assertEqual(canonical, ["行为模拟"])
        self.assertEqual(len({link["term_id"] for link in links}), 1)
        aliases = conn.execute("SELECT alias FROM keyword_aliases ORDER BY alias").fetchall()
        self.assertGreaterEqual(len(aliases), 1)

    def test_keyword_search_uses_alias_mapping(self) -> None:
        conn = memory_conn()
        repo = KnowledgeRepository(conn)
        source = _source("行为模拟技术 可以用于公共空间研究")
        source.clean_text = source.raw_text
        repo.upsert_source(source)
        now = utc_now_iso()
        group_id = "group-kw"
        repo.upsert_knowledge_group(
            KnowledgeGroup(
                group_id=group_id,
                source_id=source.source_id,
                group_title="行为模拟",
                created_at=now,
                updated_at=now,
            )
        )
        note = StructuredNote(
            note_id="note-kw",
            source_id=source.source_id,
            title="行为模拟在公共空间研究中的用途",
            note_type="知识摘要",
            themes=["行为模拟"],
            summary="行为模拟技术可以用于公共空间研究。",
            key_points=["模拟个体行为"],
            usage_scenarios=[],
            keywords=["行为模拟"],
            source_excerpt="行为模拟技术 可以用于公共空间研究",
            markdown_content="行为模拟技术 可以用于公共空间研究",
            created_at=now,
            updated_at=now,
        )
        repo.insert_note(note, [ChunkRecord("chunk-kw", "note-kw", note.summary, 0, "source_chunk", note.keywords)])
        canonical, links = canonicalize_keywords(repo, ["行为模拟技术"], source="test")
        unit_id = "unit-kw"
        repo.insert_knowledge_unit(
            KnowledgeUnit(
                unit_id=unit_id,
                group_id=group_id,
                source_id=source.source_id,
                note_id=note.note_id,
                title=note.title,
                content=note.summary,
                evidence=note.source_excerpt,
                note_type=note.note_type,
                created_at=now,
                updated_at=now,
            )
        )
        repo.replace_unit_keywords(unit_id, links)
        results = repo.search_notes_by_keyword_names(["行为模拟方法"], limit=5)
        self.assertEqual(canonical, ["行为模拟"])
        self.assertEqual(results[0].note_id, "note-kw")

    def test_backfill_creates_units_for_legacy_notes(self) -> None:
        conn = memory_conn()
        repo = KnowledgeRepository(conn)
        source = _source("ABM 行为模拟")
        source.clean_text = source.raw_text
        repo.upsert_source(source)
        now = utc_now_iso()
        note = StructuredNote(
            note_id="legacy-note",
            source_id=source.source_id,
            title="ABM 行为模拟",
            note_type="知识摘要",
            themes=["行为模拟技术"],
            summary="ABM 可用于行为模拟。",
            key_points=["行为模拟技术用于路径选择研究"],
            usage_scenarios=[],
            keywords=["行为模拟技术"],
            source_excerpt="ABM 行为模拟",
            markdown_content="ABM 行为模拟",
            created_at=now,
            updated_at=now,
        )
        repo.insert_note(note, [])
        stats = backfill_v2_structures(repo)
        self.assertEqual(stats["units"], 1)
        self.assertTrue(repo.get_unit_by_note_id("legacy-note"))
        self.assertTrue(repo.search_notes_by_keyword_names(["行为模拟方法"], limit=5))

    def test_search_expands_to_same_group_units(self) -> None:
        os.environ["LK_DISABLE_LLM"] = "1"
        conn = memory_conn()
        repo = KnowledgeRepository(conn)
        source = _source("ABM group")
        source.clean_text = source.raw_text
        repo.upsert_source(source)
        now = utc_now_iso()
        group_id = "group-search"
        repo.upsert_knowledge_group(
            KnowledgeGroup(
                group_id=group_id,
                source_id=source.source_id,
                group_title="ABM research bundle",
                created_at=now,
                updated_at=now,
            )
        )
        note_a = _note("note-abm", source.source_id, "ABM simulation", "ABM can model route choice.", ["ABM"])
        note_b = _note("note-path", source.source_id, "Path preference", "Shade and continuity affect jogging routes.", ["route"])
        repo.insert_note(note_a, [ChunkRecord("chunk-abm", note_a.note_id, note_a.summary, 0, "source_chunk", note_a.keywords)])
        repo.insert_note(note_b, [ChunkRecord("chunk-path", note_b.note_id, note_b.summary, 1, "source_chunk", note_b.keywords)])
        repo.insert_knowledge_unit(
            KnowledgeUnit("unit-abm", group_id, source.source_id, note_a.note_id, note_a.title, note_a.summary, created_at=now, updated_at=now)
        )
        repo.insert_knowledge_unit(
            KnowledgeUnit("unit-path", group_id, source.source_id, note_b.note_id, note_b.title, note_b.summary, created_at=now, updated_at=now, order_index=1)
        )
        service = SearchService(repo, QwenClient("http://127.0.0.1:8080/v1", "qwen2.5", timeout_seconds=1), setup_logger(Path("data/logs"), name="search_group_test"))
        results = service.keyword_search("ABM", limit=5)
        self.assertIn("note-abm", {item.note_id for item in results})
        self.assertIn("note-path", {item.note_id for item in results})
        os.environ.pop("LK_DISABLE_LLM", None)

    def test_search_expands_to_relation_peers(self) -> None:
        os.environ["LK_DISABLE_LLM"] = "1"
        conn = memory_conn()
        repo = KnowledgeRepository(conn)
        source = _source("relation source")
        source.clean_text = source.raw_text
        repo.upsert_source(source)
        now = utc_now_iso()
        note_a = _note("note-method", source.source_id, "ABM method", "ABM supports behavior simulation.", ["ABM"])
        note_b = _note("note-evidence", source.source_id, "Observation evidence", "Route observation can calibrate parameters.", ["observation"])
        repo.insert_note(note_a, [])
        repo.insert_note(note_b, [])
        run_id = "run-relation-search"
        repo.create_organize_run(run_id, status="completed", started_at=now)
        repo.replace_relations_for_run(
            run_id,
            [
                {
                    "relation_id": "rel-1",
                    "from_note_id": note_a.note_id,
                    "to_note_id": note_b.note_id,
                    "relation_type": "supports",
                    "score": 0.9,
                    "reason": "test relation",
                    "created_at": now,
                }
            ],
        )
        service = SearchService(repo, QwenClient("http://127.0.0.1:8080/v1", "qwen2.5", timeout_seconds=1), setup_logger(Path("data/logs"), name="search_relation_test"))
        results = service.keyword_search("ABM", limit=5)
        self.assertIn("note-method", {item.note_id for item in results})
        self.assertIn("note-evidence", {item.note_id for item in results})
        os.environ.pop("LK_DISABLE_LLM", None)

    def test_relation_builder_creates_group_sequence_and_keyword_relations(self) -> None:
        conn = memory_conn()
        repo = KnowledgeRepository(conn)
        source = _source("relation builder")
        source.clean_text = source.raw_text
        repo.upsert_source(source)
        now = utc_now_iso()
        group_id = "group-rel-builder"
        repo.upsert_knowledge_group(
            KnowledgeGroup(
                group_id=group_id,
                source_id=source.source_id,
                group_title="Relation group",
                created_at=now,
                updated_at=now,
            )
        )
        note_a = _note("rel-a", source.source_id, "Behavior simulation", "ABM models behavior.", ["behavior"])
        note_b = _note("rel-b", source.source_id, "Path preference", "Behavior affects route choices.", ["behavior"])
        repo.insert_note(note_a, [])
        repo.insert_note(note_b, [])
        repo.insert_knowledge_unit(
            KnowledgeUnit("unit-rel-a", group_id, source.source_id, note_a.note_id, note_a.title, note_a.summary, order_index=0, created_at=now, updated_at=now)
        )
        repo.insert_knowledge_unit(
            KnowledgeUnit("unit-rel-b", group_id, source.source_id, note_b.note_id, note_b.title, note_b.summary, order_index=1, created_at=now, updated_at=now)
        )
        _, links_a = canonicalize_keywords(repo, ["behavior simulation"], source="test")
        _, links_b = canonicalize_keywords(repo, ["behavior simulation"], source="test")
        repo.replace_unit_keywords("unit-rel-a", links_a)
        repo.replace_unit_keywords("unit-rel-b", links_b)
        relations = build_structural_relations(repo)
        relation_types = {item["relation_type"] for item in relations}
        self.assertIn("same_group", relation_types)
        self.assertIn("sequence_next", relation_types)
        self.assertIn("shared_keyword", relation_types)

    def test_organize_knowledge_base_marks_duplicates_and_relates(self) -> None:
        conn = memory_conn()
        repo = KnowledgeRepository(conn)
        source = _source("整理测试文本")
        source.clean_text = source.raw_text
        repo.upsert_source(source)
        now = utc_now_iso()
        notes = [
            StructuredNote(
                note_id="n1",
                source_id=source.source_id,
                title="慢跑路径偏好",
                note_type="科研灵感",
                themes=["路径偏好"],
                summary="公共空间中的慢跑路径偏好",
                key_points=["遮荫影响路径选择"],
                usage_scenarios=[],
                keywords=["慢跑", "路径偏好"],
                source_excerpt="公共空间中的慢跑路径偏好",
                markdown_content="公共空间中的慢跑路径偏好",
                created_at=now,
                updated_at=now,
            ),
            StructuredNote(
                note_id="n2",
                source_id=source.source_id,
                title="慢跑路径偏好-重复",
                note_type="科研灵感",
                themes=["路径偏好"],
                summary="公共空间中的慢跑路径偏好",
                key_points=["遮荫影响路径选择"],
                usage_scenarios=[],
                keywords=["慢跑", "路径偏好"],
                source_excerpt="公共空间中的慢跑路径偏好",
                markdown_content="公共空间中的慢跑路径偏好",
                created_at=now,
                updated_at=now,
            ),
            StructuredNote(
                note_id="n3",
                source_id=source.source_id,
                title="设施密度与停留",
                note_type="知识摘要",
                themes=["公共空间"],
                summary="设施密度会影响停留行为",
                key_points=["座椅与停留相关"],
                usage_scenarios=[],
                keywords=["设施密度", "停留行为"],
                source_excerpt="设施密度会影响停留行为",
                markdown_content="设施密度会影响停留行为",
                created_at=now,
                updated_at=now,
            ),
        ]
        for note in notes:
            repo.insert_note(note, [ChunkRecord(f"c_{note.note_id}", note.note_id, note.summary, 0, "source_chunk", note.keywords)])

        run_id = "run-test-1"
        repo.create_organize_run(run_id, status="running", started_at=now)
        result = organize_knowledge_base(repo, run_id=run_id, logger=setup_logger(Path("data/logs"), name="organize_test"))
        repo.update_organize_run(run_id, status="completed", finished_at=utc_now_iso(), stats=result["stats"], report_markdown=result["report_markdown"])

        remaining = repo.list_notes_for_organize()
        self.assertEqual(result["stats"]["removed_duplicates"], 0)
        self.assertEqual(result["stats"]["duplicate_candidates"], 1)
        self.assertEqual(len(remaining), 3)
        self.assertTrue(repo.list_relations_for_run(run_id, limit=20))


def memory_conn() -> sqlite3.Connection:
    conn = connect(":memory:")
    init_db(conn)
    return conn


def _source(raw_text: str) -> SourceRecord:
    now = utc_now_iso()
    return SourceRecord(
        source_id=f"source-{abs(hash(raw_text))}",
        source_type="manual",
        file_path=None,
        raw_text=raw_text,
        clean_text="",
        text_hash=str(abs(hash(raw_text))),
        created_at=now,
        imported_at=now,
        status="new",
    )


def _note(note_id: str, source_id: str, title: str, summary: str, keywords: list[str]) -> StructuredNote:
    now = utc_now_iso()
    return StructuredNote(
        note_id=note_id,
        source_id=source_id,
        title=title,
        note_type="test",
        themes=keywords,
        summary=summary,
        key_points=[summary],
        usage_scenarios=[],
        keywords=keywords,
        source_excerpt=summary,
        markdown_content=f"# {title}\n\n{summary}",
        created_at=now,
        updated_at=now,
    )


if __name__ == "__main__":
    unittest.main()
