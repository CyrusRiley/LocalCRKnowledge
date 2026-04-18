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
from app.backend.models import ChunkRecord, KnowledgeGroup, KnowledgeUnit, SearchResult, SourceRecord, StructuredNote
from app.backend.organizer.knowledge_organizer import organize_knowledge_base
from app.backend.preprocess.cleaner import build_chunk_contexts, chunk_text, normalize_text, preprocess_text
from app.backend.relations.relation_builder import build_structural_relations
from app.backend.retrieval.answer_builder import build_answer_result
from app.backend.retrieval.graph_builder import build_library_graph
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
        self.assertEqual(payload["faithful_content"], "原文内容")

    def test_parser_keeps_faithful_content_from_json(self) -> None:
        content = '{"title":"ABM method","faithful_content":"This unit preserves the full method description."}'
        payload = parse_note_json(content, fallback_title="Fallback", source_text="raw source")
        self.assertEqual(payload["faithful_content"], "This unit preserves the full method description.")

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

    def test_repository_chinese_sentence_search_uses_like_terms(self) -> None:
        conn = memory_conn()
        repo = KnowledgeRepository(conn)
        source = _source("人行为模拟技术的发展经历了七次核心转向。")
        source.clean_text = source.raw_text
        repo.upsert_source(source)
        note = StructuredNote(
            note_id="note-chinese-search",
            source_id=source.source_id,
            title="行为模拟技术发展转向",
            note_type="知识摘要",
            themes=["行为模拟"],
            summary="人行为模拟技术的发展经历了七次核心转向。",
            key_points=["从结果拟合转向过程生成"],
            usage_scenarios=[],
            keywords=["行为模拟", "发展转向"],
            source_excerpt=source.raw_text,
            markdown_content="# 行为模拟技术发展转向\n\n人行为模拟技术的发展经历了七次核心转向。",
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
        )
        repo.insert_note(note, [ChunkRecord("chunk-cn-search", note.note_id, note.summary, 0, "source_chunk", note.keywords)])
        results = repo.search_notes("人行为模拟技术经历几次发展转向", limit=5)
        self.assertEqual(results[0].note_id, "note-chinese-search")

    def test_repository_searches_faithful_content(self) -> None:
        conn = memory_conn()
        repo = KnowledgeRepository(conn)
        source = _source("raw text")
        source.clean_text = source.raw_text
        repo.upsert_source(source)
        note = _note("note-faithful", source.source_id, "Faithful field", "Short summary", ["method"])
        note.faithful_content = "The detailed preserved content mentions sidewalk permeability and route choice."
        note.markdown_content = "# Faithful field\n\nShort summary"
        repo.insert_note(note, [])
        results = repo.search_notes("sidewalk permeability", limit=5)
        self.assertEqual(results[0].note_id, "note-faithful")

    def test_repository_import_history_counts_notes_by_source(self) -> None:
        conn = memory_conn()
        repo = KnowledgeRepository(conn)
        source = _source("demo import history")
        source.clean_text = source.raw_text
        repo.upsert_source(source)
        note = _note("history-note", source.source_id, "History Note", "A note for import history.", ["history"])
        repo.insert_note(note, [ChunkRecord("history-chunk", note.note_id, note.summary, 0, "source_chunk", note.keywords)])
        history = repo.list_import_history(limit=5)
        self.assertEqual(history[0]["source_id"], source.source_id)
        self.assertEqual(history[0]["note_count"], 1)
        self.assertIn("History Note", history[0]["note_titles"])

    def test_answer_builder_reports_extract_mode_without_results(self) -> None:
        result = build_answer_result(
            "没有命中的问题",
            [],
            llm_client=QwenClient("http://127.0.0.1:8080/v1", "qwen2.5", timeout_seconds=1),
            logger=setup_logger(Path("data/logs"), name="answer_mode_test"),
        )
        self.assertEqual(result.mode, "extract")
        self.assertIn("没有检索到", result.content)

    def test_answer_builder_always_uses_extract_mode_with_results(self) -> None:
        item = SearchResult(
            note_id="note-answer-mode",
            source_id="source-answer-mode",
            title="ABM route preference",
            note_type="summary",
            summary="ABM can explain route preference.",
            markdown_content="# ABM route preference\n\nABM can explain route preference.",
            source_excerpt="ABM can explain route preference.",
            themes=["ABM"],
            keywords=["ABM"],
            file_path=None,
            imported_at=None,
            score=1.0,
            snippet="ABM can explain route preference.",
        )
        result = build_answer_result(
            "What did I note about ABM?",
            [item],
            llm_client=QwenClient("http://127.0.0.1:1/v1", "unused", timeout_seconds=1),
            logger=setup_logger(Path("data/logs"), name="answer_extract_mode_test"),
        )
        self.assertEqual(result.mode, "extract")
        self.assertIn("ABM can explain route preference.", result.content)
        self.assertIn("主要相关材料", result.content)
        self.assertIn("可直接用于写作的段落", result.content)

    def test_answer_builder_uses_constrained_model_when_available(self) -> None:
        item = SearchResult(
            note_id="note-answer-model",
            source_id="source-answer-model",
            title="慢跑路径偏好",
            note_type="summary",
            summary="慢跑路径选择受到遮荫、连续性和安全性的共同影响。",
            markdown_content="# 慢跑路径偏好\n\n## 保真整理\n慢跑路径选择受到遮荫、连续性和安全性的共同影响，适合用于解释公共空间中的路径偏好。",
            source_excerpt="慢跑路径选择受到遮荫、连续性和安全性的共同影响。",
            themes=["路径偏好"],
            keywords=["慢跑", "路径偏好"],
            file_path=None,
            imported_at=None,
            score=1.0,
            snippet="慢跑路径偏好",
            relevance_score=2.0,
            relevance_level="高相关",
            relevance_reason="标题命中“路径偏好”",
        )
        result = build_answer_result(
            "慢跑路径偏好是什么？",
            [item],
            llm_client=FakeAnswerClient(),
            logger=setup_logger(Path("data/logs"), name="answer_model_mode_test"),
        )
        self.assertEqual(result.mode, "model")
        self.assertIn("综合回答", result.content)
        self.assertIn("慢跑路径偏好可以理解为", result.content)

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
            faithful_content="更新保真整理内容 包含 新词汇",
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

    def test_search_expands_to_near_group_units(self) -> None:
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
                    "relation_layer": "semantic",
                    "relation_strength": "strong",
                    "display_default": True,
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
        evidence = next(item for item in results if item.note_id == "note-evidence")
        self.assertEqual(evidence.relation_type, "supports")
        self.assertIn("知识关系网扩展", evidence.relevance_reason)
        os.environ.pop("LK_DISABLE_LLM", None)

    def test_vector_search_adds_semantic_candidate(self) -> None:
        os.environ["LK_DISABLE_LLM"] = "1"
        conn = memory_conn()
        repo = KnowledgeRepository(conn)
        source = _source("vector source")
        source.clean_text = source.raw_text
        repo.upsert_source(source)
        now = utc_now_iso()
        group_id = "group-vector"
        repo.upsert_knowledge_group(
            KnowledgeGroup(
                group_id=group_id,
                source_id=source.source_id,
                group_title="Vector group",
                created_at=now,
                updated_at=now,
            )
        )
        note = _note("note-vector-route", source.source_id, "跑步路线选择", "跑步者会根据遮荫和连续性选择路线。", ["路线选择"])
        repo.insert_note(note, [])
        repo.insert_knowledge_unit(
            KnowledgeUnit("unit-vector-route", group_id, source.source_id, note.note_id, note.title, note.summary, evidence=note.source_excerpt, order_index=0, created_at=now, updated_at=now)
        )
        service = SearchService(repo, QwenClient("http://127.0.0.1:8080/v1", "qwen2.5", timeout_seconds=1), setup_logger(Path("data/logs"), name="search_vector_test"))
        results = service.keyword_search("慢跑路径偏好", limit=5)
        self.assertIn("note-vector-route", {item.note_id for item in results})
        item = next(item for item in results if item.note_id == "note-vector-route")
        self.assertTrue(item.relevance_score > 0)
        self.assertIn("向量", item.relevance_reason)
        os.environ.pop("LK_DISABLE_LLM", None)

    def test_repository_stores_multiple_embedding_models_per_unit(self) -> None:
        conn = memory_conn()
        repo = KnowledgeRepository(conn)
        source = _source("multi embedding")
        source.clean_text = source.raw_text
        repo.upsert_source(source)
        now = utc_now_iso()
        note = _note("note-multi-emb", source.source_id, "Multi embedding", "same unit, multiple vectors", ["embedding"])
        repo.insert_note(note, [])
        repo.insert_knowledge_unit(
            KnowledgeUnit("unit-multi-emb", None, source.source_id, note.note_id, note.title, note.summary, created_at=now, updated_at=now)
        )
        repo.upsert_unit_embedding(
            unit_id="unit-multi-emb",
            note_id=note.note_id,
            embedding_model="model-a",
            text_hash="hash-a",
            vector=[1.0, 0.0],
            now=now,
        )
        repo.upsert_unit_embedding(
            unit_id="unit-multi-emb",
            note_id=note.note_id,
            embedding_model="model-b",
            text_hash="hash-b",
            vector=[0.0, 1.0],
            now=now,
        )
        counts = repo.embedding_counts_by_model()
        self.assertEqual(counts["model-a"], 1)
        self.assertEqual(counts["model-b"], 1)

    def test_relation_builder_creates_layered_sequence_without_group_mesh(self) -> None:
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
        _, links_a = canonicalize_keywords(repo, ["behavior simulation", "public space"], source="test")
        _, links_b = canonicalize_keywords(repo, ["behavior simulation", "public space"], source="test")
        repo.replace_unit_keywords("unit-rel-a", links_a)
        repo.replace_unit_keywords("unit-rel-b", links_b)
        relations = build_structural_relations(repo)
        relation_types = {item["relation_type"] for item in relations}
        self.assertIn("sequence_next", relation_types)
        self.assertNotIn("same_group", relation_types)
        self.assertIn("shared_keyword", relation_types)
        sequence = next(item for item in relations if item["relation_type"] == "sequence_next")
        shared = next(item for item in relations if item["relation_type"] == "shared_keyword")
        self.assertEqual(sequence["relation_layer"], "structure")
        self.assertTrue(sequence["display_default"])
        self.assertEqual(shared["relation_layer"], "weak")
        self.assertFalse(shared["display_default"])

    def test_relation_builder_creates_semantic_relations(self) -> None:
        conn = memory_conn()
        repo = KnowledgeRepository(conn)
        source = _source("semantic relation builder")
        source.clean_text = source.raw_text
        repo.upsert_source(source)
        now = utc_now_iso()
        group_id = "group-semantic-rel"
        repo.upsert_knowledge_group(
            KnowledgeGroup(
                group_id=group_id,
                source_id=source.source_id,
                group_title="Semantic group",
                created_at=now,
                updated_at=now,
            )
        )
        note_method = _note(
            "semantic-method",
            source.source_id,
            "ABM method",
            "This method uses an agent based model and parameter calibration.",
            ["ABM"],
        )
        note_case = _note(
            "semantic-case",
            source.source_id,
            "Route choice case",
            "This application case uses observations for route choice scenarios.",
            ["route choice"],
        )
        repo.insert_note(note_method, [])
        repo.insert_note(note_case, [])
        repo.insert_knowledge_unit(
            KnowledgeUnit(
                "unit-semantic-method",
                group_id,
                source.source_id,
                note_method.note_id,
                note_method.title,
                note_method.summary,
                order_index=0,
                created_at=now,
                updated_at=now,
            )
        )
        repo.insert_knowledge_unit(
            KnowledgeUnit(
                "unit-semantic-case",
                group_id,
                source.source_id,
                note_case.note_id,
                note_case.title,
                note_case.summary,
                order_index=1,
                created_at=now,
                updated_at=now,
            )
        )
        relations = build_structural_relations(repo)
        relation_types = {item["relation_type"] for item in relations}
        self.assertIn("method_for", relation_types)
        method_rel = next(item for item in relations if item["relation_type"] == "method_for")
        self.assertEqual(method_rel["relation_layer"], "semantic")
        self.assertEqual(method_rel["relation_strength"], "strong")

    def test_library_graph_builds_structure_nodes_and_semantic_edges(self) -> None:
        conn = memory_conn()
        repo = KnowledgeRepository(conn)
        source = _source("graph source")
        source.clean_text = source.raw_text
        repo.upsert_source(source)
        now = utc_now_iso()
        group_id = "group-graph"
        repo.upsert_knowledge_group(
            KnowledgeGroup(
                group_id=group_id,
                source_id=source.source_id,
                group_title="Graph group",
                created_at=now,
                updated_at=now,
            )
        )
        note_a = _note("graph-a", source.source_id, "ABM method", "ABM method for route choice.", ["ABM"])
        note_b = _note("graph-b", source.source_id, "Route case", "Observation case for route choice.", ["route choice"])
        repo.insert_note(note_a, [])
        repo.insert_note(note_b, [])
        repo.insert_knowledge_unit(
            KnowledgeUnit("unit-graph-a", group_id, source.source_id, note_a.note_id, note_a.title, note_a.summary, order_index=0, created_at=now, updated_at=now)
        )
        repo.insert_knowledge_unit(
            KnowledgeUnit("unit-graph-b", group_id, source.source_id, note_b.note_id, note_b.title, note_b.summary, order_index=1, created_at=now, updated_at=now)
        )
        _, links = canonicalize_keywords(repo, ["ABM"], source="test")
        repo.replace_unit_keywords("unit-graph-a", links)
        run_id = "run-graph"
        repo.create_organize_run(run_id, status="completed", started_at=now)
        repo.replace_relations_for_run(
            run_id,
            [
                {
                    "relation_id": "rel-graph",
                    "from_note_id": note_a.note_id,
                    "to_note_id": note_b.note_id,
                    "relation_type": "method_for",
                    "relation_layer": "semantic",
                    "relation_strength": "strong",
                    "display_default": True,
                    "score": 0.9,
                    "reason": "test semantic relation",
                    "created_at": now,
                }
            ],
        )
        graph = build_library_graph(repo, limit=20)
        node_types = {item["node_type"] for item in graph["nodes"]}
        edge_types = {item["relation_type"] for item in graph["edges"]}
        self.assertIn("group", node_types)
        self.assertIn("concept", node_types)
        self.assertIn("unit", node_types)
        self.assertIn("contains", edge_types)
        self.assertIn("concept_contains", edge_types)
        self.assertIn("method_for", edge_types)

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

    def test_quick_organize_keeps_old_relations_and_updates_changed_notes(self) -> None:
        conn = memory_conn()
        repo = KnowledgeRepository(conn)
        source = _source("quick organize")
        source.clean_text = source.raw_text
        repo.upsert_source(source)
        now = utc_now_iso()
        group_id = "group-quick-organize"
        repo.upsert_knowledge_group(
            KnowledgeGroup(
                group_id=group_id,
                source_id=source.source_id,
                group_title="Quick group",
                created_at=now,
                updated_at=now,
            )
        )
        note_a = _note("quick-a", source.source_id, "Existing method", "ABM method for behavior simulation.", ["ABM"])
        note_b = _note("quick-b", source.source_id, "Existing case", "Application case for behavior simulation.", ["case"])
        note_c = _note("quick-c", source.source_id, "New evidence", "Observation evidence supports the case.", ["evidence"])
        note_a.updated_at = "2000-01-01T00:00:00+00:00"
        note_b.updated_at = "2000-01-01T00:00:00+00:00"
        repo.insert_note(note_a, [])
        repo.insert_note(note_b, [])
        repo.insert_knowledge_unit(
            KnowledgeUnit("unit-quick-a", group_id, source.source_id, note_a.note_id, note_a.title, note_a.summary, order_index=0, created_at=now, updated_at=now)
        )
        repo.insert_knowledge_unit(
            KnowledgeUnit("unit-quick-b", group_id, source.source_id, note_b.note_id, note_b.title, note_b.summary, order_index=1, created_at=now, updated_at=now)
        )
        previous_run = "run-quick-old"
        repo.create_organize_run(previous_run, status="completed", started_at="2000-01-01T00:00:00+00:00")
        repo.update_organize_run(previous_run, status="completed", finished_at="2000-01-01T00:00:01+00:00", stats={}, report_markdown="")
        repo.replace_relations_for_run(
            previous_run,
            [
                {
                    "relation_id": "old-rel-quick",
                    "from_note_id": note_a.note_id,
                    "to_note_id": note_b.note_id,
                    "relation_type": "method_for",
                    "relation_layer": "semantic",
                    "relation_strength": "strong",
                    "display_default": True,
                    "score": 0.9,
                    "reason": "old relation",
                    "created_at": now,
                }
            ],
        )
        repo.insert_note(note_c, [])
        repo.insert_knowledge_unit(
            KnowledgeUnit("unit-quick-c", group_id, source.source_id, note_c.note_id, note_c.title, note_c.summary, order_index=2, created_at=utc_now_iso(), updated_at=utc_now_iso())
        )
        quick_run = "run-quick-new"
        repo.create_organize_run(quick_run, status="running", started_at=utc_now_iso())
        result = organize_knowledge_base(repo, run_id=quick_run, logger=setup_logger(Path("data/logs"), name="organize_quick_test"), mode="quick")
        relation_types = {rel["relation_type"] for rel in result["relations"]}
        self.assertEqual(result["stats"]["mode"], "quick")
        self.assertEqual(result["stats"]["affected_notes"], 1)
        self.assertIn("method_for", relation_types)
        self.assertIn("sequence_next", relation_types)


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


class FakeAnswerClient:
    def build_constrained_answer(self, question: str, evidence_markdown: str) -> str:
        return (
            "慢跑路径偏好可以理解为个体在公共空间中选择慢跑路线时形成的综合判断，它并不只取决于距离，"
            "还受到遮荫、连续性和安全性的共同影响。已有材料表明，这类偏好适合被放入公共空间行为机制的讨论中，"
            "用于解释为什么不同空间条件会引导不同的路径选择。\n\n"
            "从写作角度看，这些材料可以支撑论文中关于路径选择机制、环境因素和行为模拟参数设定的论述。"
        )


if __name__ == "__main__":
    unittest.main()
