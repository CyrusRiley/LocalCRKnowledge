from __future__ import annotations

import hashlib
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import combinations
from logging import Logger
from typing import Callable
from uuid import uuid4

from app.backend.database.repository import KnowledgeRepository
from app.backend.preprocess.cleaner import extract_keywords
from app.backend.relations.relation_builder import build_relations_for_notes, build_structural_relations, dedupe_relations
from app.backend.utils.time_utils import utc_now_iso


ProgressCallback = Callable[[dict], None]


@dataclass
class NoteVector:
    note_id: str
    title: str
    note_type: str
    updated_at: str
    normalized_text: str
    tokens: set[str]


class UnionFind:
    def __init__(self, items: list[str]):
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            nxt = self.parent[item]
            self.parent[item] = root
            item = nxt
        return root

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def organize_knowledge_base(
    repo: KnowledgeRepository,
    *,
    run_id: str,
    logger: Logger,
    mode: str = "full",
    progress_cb: ProgressCallback | None = None,
) -> dict:
    def push(status: str, message: str, percent: int) -> None:
        if progress_cb:
            progress_cb({"status": status, "message": message, "percent": percent})

    normalized_mode = "quick" if mode == "quick" else "full"
    if normalized_mode == "quick":
        return _organize_quick(repo, run_id=run_id, logger=logger, progress_cb=progress_cb)

    push("running", "Loading notes...", 5)
    notes = repo.list_notes_for_organize()
    if not notes:
        stats = {
            "total_notes_before": 0,
            "total_notes_after": 0,
            "removed_duplicates": 0,
            "duplicate_candidates": 0,
            "relations_saved": 0,
        }
        report = _build_report(stats, [], [])
        return {"stats": stats, "report_markdown": report, "relations": []}

    vectors = [_vectorize(note) for note in notes]
    note_by_id = {item.note_id: item for item in vectors}

    push("running", "Detecting duplicates...", 20)
    duplicate_pairs, related_pairs = _find_relations(vectors)
    uf = UnionFind([item.note_id for item in vectors])
    for a, b, _ in duplicate_pairs:
        uf.union(a, b)

    clusters: dict[str, list[str]] = {}
    for note_id in note_by_id:
        root = uf.find(note_id)
        clusters.setdefault(root, []).append(note_id)

    duplicate_map: dict[str, str] = {}
    for member_ids in clusters.values():
        keeper = _pick_keeper([note_by_id[note_id] for note_id in member_ids])
        for note_id in member_ids:
            if note_id != keeper:
                duplicate_map[note_id] = keeper

    push("running", "Building knowledge relations...", 55)
    relations: list[dict] = []
    now = utc_now_iso()
    for dup_id, keeper_id in duplicate_map.items():
        relations.append(
            {
                "relation_id": str(uuid4()),
                "from_note_id": dup_id,
                "to_note_id": keeper_id,
                "relation_type": "duplicate_of",
                "relation_layer": "semantic",
                "relation_strength": "strong",
                "display_default": True,
                "score": 1.0,
                "reason": "high lexical similarity; kept as candidate, not deleted automatically",
                "created_at": now,
            }
        )

    seen_rel: set[tuple[str, str]] = set()
    for a, b, score in related_pairs:
        a_root = duplicate_map.get(a, a)
        b_root = duplicate_map.get(b, b)
        if a_root == b_root:
            continue
        pair = tuple(sorted((a_root, b_root)))
        if pair in seen_rel:
            continue
        seen_rel.add(pair)
        relations.append(
            {
                "relation_id": str(uuid4()),
                "from_note_id": pair[0],
                "to_note_id": pair[1],
                "relation_type": "related_to",
                "relation_layer": "weak",
                "relation_strength": "weak",
                "display_default": False,
                "score": round(score, 4),
                "reason": "shared themes/keywords",
                "created_at": now,
            }
        )

    push("running", "Adding group, sequence, and keyword relations...", 75)
    relations = dedupe_relations([*relations, *build_structural_relations(repo)])
    repo.replace_relations_for_run(run_id, relations)

    push("running", "Generating report...", 90)
    stats = {
        "mode": "full",
        "total_notes_before": len(vectors),
        "total_notes_after": len(vectors),
        "removed_duplicates": 0,
        "duplicate_candidates": len(duplicate_map),
        "duplicate_clusters": sum(1 for values in clusters.values() if len(values) > 1),
        "relations_saved": len(relations),
        "display_relations": sum(1 for rel in relations if rel.get("display_default")),
        "structure_relations": sum(1 for rel in relations if rel.get("relation_layer") == "structure"),
        "semantic_relations": sum(1 for rel in relations if rel.get("relation_layer") == "semantic"),
        "weak_relations": sum(1 for rel in relations if rel.get("relation_layer") == "weak"),
    }
    report = _build_report(stats, clusters, relations)
    push("completed", "Knowledge organization finished.", 100)
    logger.info("Organize run %s finished: %s", run_id, stats)
    return {"stats": stats, "report_markdown": report, "relations": relations}


def _organize_quick(
    repo: KnowledgeRepository,
    *,
    run_id: str,
    logger: Logger,
    progress_cb: ProgressCallback | None = None,
) -> dict:
    def push(status: str, message: str, percent: int) -> None:
        if progress_cb:
            progress_cb({"status": status, "message": message, "percent": percent})

    push("running", "Loading previous relation network...", 5)
    previous_run = repo.get_latest_completed_organize_run()
    if not previous_run:
        logger.info("Quick organize falls back to full rebuild: no completed run")
        result = organize_knowledge_base(repo, run_id=run_id, logger=logger, mode="full", progress_cb=progress_cb)
        result["stats"]["mode"] = "full"
        result["stats"]["quick_fallback"] = "no_completed_run"
        return result

    cutoff = str(previous_run.get("finished_at") or previous_run.get("started_at") or "")
    affected_notes = repo.list_notes_updated_after(cutoff) if cutoff else repo.list_notes_for_organize()
    affected_ids = {str(note.get("note_id") or "") for note in affected_notes if note.get("note_id")}
    previous_relations = repo.list_relations_for_run(str(previous_run["run_id"]), limit=100000)

    push("running", f"Found {len(affected_ids)} changed notes.", 25)
    if not affected_ids:
        relations = [_copy_relation(rel) for rel in previous_relations]
        repo.replace_relations_for_run(run_id, relations)
        stats = _relation_stats(
            relations,
            mode="quick",
            total_notes=len(repo.list_notes_for_organize()),
            affected_notes=0,
            copied_relations=len(relations),
            rebuilt_relations=0,
        )
        report = _build_report(stats, {}, relations)
        push("completed", "No changed notes; copied current relation network.", 100)
        return {"stats": stats, "report_markdown": report, "relations": relations}

    push("running", "Keeping unaffected relations...", 40)
    kept_relations = [
        _copy_relation(rel)
        for rel in previous_relations
        if str(rel.get("from_note_id") or "") not in affected_ids and str(rel.get("to_note_id") or "") not in affected_ids
    ]

    push("running", "Rebuilding changed note relations...", 60)
    vectors = [_vectorize(note) for note in repo.list_notes_for_organize()]
    duplicate_pairs, related_pairs = _find_relations_for_affected(vectors, affected_ids)
    now = utc_now_iso()
    rebuilt_relations: list[dict] = []
    for a, b, score in duplicate_pairs:
        rebuilt_relations.append(
            {
                "relation_id": str(uuid4()),
                "from_note_id": a,
                "to_note_id": b,
                "relation_type": "duplicate_of",
                "relation_layer": "semantic",
                "relation_strength": "strong",
                "display_default": True,
                "score": round(score, 4),
                "reason": "high lexical similarity in quick organize",
                "created_at": now,
            }
        )
    for a, b, score in related_pairs:
        rebuilt_relations.append(
            {
                "relation_id": str(uuid4()),
                "from_note_id": a,
                "to_note_id": b,
                "relation_type": "related_to",
                "relation_layer": "weak",
                "relation_strength": "weak",
                "display_default": False,
                "score": round(score, 4),
                "reason": "shared themes/keywords in quick organize",
                "created_at": now,
            }
        )
    rebuilt_relations.extend(build_relations_for_notes(repo, affected_ids))

    push("running", "Saving merged relation network...", 85)
    relations = dedupe_relations([*kept_relations, *rebuilt_relations])
    repo.replace_relations_for_run(run_id, relations)
    stats = _relation_stats(
        relations,
        mode="quick",
        total_notes=len(vectors),
        affected_notes=len(affected_ids),
        copied_relations=len(kept_relations),
        rebuilt_relations=len(rebuilt_relations),
    )
    report = _build_report(stats, {}, relations)
    push("completed", "Quick relation update finished.", 100)
    logger.info("Quick organize run %s finished: %s", run_id, stats)
    return {"stats": stats, "report_markdown": report, "relations": relations}


def _vectorize(note: dict) -> NoteVector:
    pieces = [
        str(note.get("title") or ""),
        str(note.get("summary") or ""),
        str(note.get("faithful_content") or ""),
        str(note.get("source_excerpt") or ""),
        "\n".join(note.get("key_points") or []),
        "\n".join(note.get("usage_scenarios") or []),
    ]
    normalized_text = _normalize(" ".join(pieces))
    tokens = _build_tokens(note, normalized_text)
    return NoteVector(
        note_id=str(note.get("note_id") or ""),
        title=str(note.get("title") or ""),
        note_type=str(note.get("note_type") or ""),
        updated_at=str(note.get("updated_at") or ""),
        normalized_text=normalized_text,
        tokens=tokens,
    )


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _build_tokens(note: dict, normalized_text: str) -> set[str]:
    tokens: set[str] = set()
    for value in note.get("themes") or []:
        token = str(value).strip().lower()
        if len(token) >= 2:
            tokens.add(token)
    for value in note.get("keywords") or []:
        token = str(value).strip().lower()
        if len(token) >= 2:
            tokens.add(token)
    for token in extract_keywords(normalized_text, limit=20):
        if len(token) >= 2:
            tokens.add(token)
    return tokens


def _find_relations(vectors: list[NoteVector]) -> tuple[list[tuple[str, str, float]], list[tuple[str, str, float]]]:
    exact_buckets: dict[str, list[NoteVector]] = {}
    for vec in vectors:
        digest = hashlib.sha256(vec.normalized_text.encode("utf-8")).hexdigest()
        exact_buckets.setdefault(digest, []).append(vec)

    duplicate_pairs: list[tuple[str, str, float]] = []
    for bucket in exact_buckets.values():
        if len(bucket) > 1:
            for a, b in combinations(bucket, 2):
                duplicate_pairs.append((a.note_id, b.note_id, 1.0))

    inverted: dict[str, list[int]] = {}
    for idx, vec in enumerate(vectors):
        for token in vec.tokens:
            inverted.setdefault(token, []).append(idx)

    related_pairs: list[tuple[str, str, float]] = []
    seen: set[tuple[int, int]] = set()
    for idx, vec in enumerate(vectors):
        candidate_score: dict[int, int] = {}
        for token in vec.tokens:
            for other in inverted.get(token, []):
                if other <= idx:
                    continue
                candidate_score[other] = candidate_score.get(other, 0) + 1

        ranked_candidates = sorted(candidate_score.items(), key=lambda item: (-item[1], item[0]))[:120]
        for other_idx, shared in ranked_candidates:
            if shared < 2:
                continue
            pair = (idx, other_idx)
            if pair in seen:
                continue
            seen.add(pair)
            other = vectors[other_idx]
            jac = _jaccard(vec.tokens, other.tokens)
            if jac < 0.30:
                continue
            ratio = SequenceMatcher(None, vec.normalized_text, other.normalized_text).ratio()
            if jac >= 0.86 or (jac >= 0.74 and ratio >= 0.90):
                duplicate_pairs.append((vec.note_id, other.note_id, max(jac, ratio)))
            elif jac >= 0.38:
                related_pairs.append((vec.note_id, other.note_id, jac))

    return duplicate_pairs, related_pairs


def _find_relations_for_affected(
    vectors: list[NoteVector],
    affected_note_ids: set[str],
) -> tuple[list[tuple[str, str, float]], list[tuple[str, str, float]]]:
    affected = [vec for vec in vectors if vec.note_id in affected_note_ids]
    if not affected:
        return [], []

    duplicate_pairs: list[tuple[str, str, float]] = []
    related_pairs: list[tuple[str, str, float]] = []
    seen: set[tuple[str, str]] = set()
    for vec in affected:
        for other in vectors:
            if vec.note_id == other.note_id:
                continue
            pair = tuple(sorted((vec.note_id, other.note_id)))
            if pair in seen:
                continue
            seen.add(pair)
            shared = len(vec.tokens & other.tokens)
            if shared < 2 and vec.normalized_text != other.normalized_text:
                continue
            jac = _jaccard(vec.tokens, other.tokens)
            ratio = SequenceMatcher(None, vec.normalized_text, other.normalized_text).ratio()
            if vec.normalized_text == other.normalized_text or jac >= 0.86 or (jac >= 0.74 and ratio >= 0.90):
                duplicate_pairs.append((vec.note_id, other.note_id, max(jac, ratio)))
            elif jac >= 0.38:
                related_pairs.append((vec.note_id, other.note_id, jac))
    return duplicate_pairs, related_pairs


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return intersection / union


def _pick_keeper(items: list[NoteVector]) -> str:
    ranked = sorted(
        items,
        key=lambda item: (len(item.normalized_text), item.updated_at, item.note_id),
        reverse=True,
    )
    return ranked[0].note_id


def _copy_relation(relation: dict) -> dict:
    return {
        "relation_id": str(uuid4()),
        "from_note_id": relation.get("from_note_id"),
        "to_note_id": relation.get("to_note_id"),
        "relation_type": relation.get("relation_type"),
        "relation_layer": relation.get("relation_layer") or "semantic",
        "relation_strength": relation.get("relation_strength") or "medium",
        "display_default": bool(relation.get("display_default", True)),
        "score": float(relation.get("score") or 0.0),
        "reason": relation.get("reason") or "",
        "created_at": utc_now_iso(),
    }


def _relation_stats(
    relations: list[dict],
    *,
    mode: str,
    total_notes: int,
    affected_notes: int = 0,
    copied_relations: int = 0,
    rebuilt_relations: int = 0,
) -> dict:
    return {
        "mode": mode,
        "total_notes_before": total_notes,
        "total_notes_after": total_notes,
        "removed_duplicates": 0,
        "duplicate_candidates": sum(1 for rel in relations if rel.get("relation_type") == "duplicate_of"),
        "duplicate_clusters": 0,
        "affected_notes": affected_notes,
        "copied_relations": copied_relations,
        "rebuilt_relations": rebuilt_relations,
        "relations_saved": len(relations),
        "display_relations": sum(1 for rel in relations if rel.get("display_default")),
        "structure_relations": sum(1 for rel in relations if rel.get("relation_layer") == "structure"),
        "semantic_relations": sum(1 for rel in relations if rel.get("relation_layer") == "semantic"),
        "weak_relations": sum(1 for rel in relations if rel.get("relation_layer") == "weak"),
    }


def _build_report(stats: dict, clusters: dict[str, list[str]] | list, relations: list[dict]) -> str:
    duplicate_examples: list[str] = []
    if isinstance(clusters, dict):
        for ids in clusters.values():
            if len(ids) > 1:
                duplicate_examples.append("- " + ", ".join(ids[:6]))
            if len(duplicate_examples) >= 8:
                break

    relation_examples = [
        f"- {rel['relation_type']}: {rel['from_note_id']} -> {rel['to_note_id']} (score={rel['score']})"
        for rel in relations[:12]
    ]

    parts = [
        "# 知识库整理报告",
        "",
        "## 统计信息",
        f"- 整理模式: {stats.get('mode', 'full')}",
        f"- 整理前条目数: {stats.get('total_notes_before', 0)}",
        f"- 整理后条目数: {stats.get('total_notes_after', 0)}",
        f"- 快速整理受影响条目数: {stats.get('affected_notes', 0)}",
        f"- 沿用旧关系数量: {stats.get('copied_relations', 0)}",
        f"- 重建关系候选数量: {stats.get('rebuilt_relations', 0)}",
        f"- 自动删除条目数: {stats.get('removed_duplicates', 0)}",
        f"- 重复候选条目数: {stats.get('duplicate_candidates', 0)}",
        f"- 发现重复簇数量: {stats.get('duplicate_clusters', 0)}",
        f"- 保存关系数量: {stats.get('relations_saved', 0)}",
        f"- 默认展示关系数量: {stats.get('display_relations', 0)}",
        f"- 结构边数量: {stats.get('structure_relations', 0)}",
        f"- 语义边数量: {stats.get('semantic_relations', 0)}",
        f"- 弱关系数量: {stats.get('weak_relations', 0)}",
        "",
        "## 重复簇示例",
    ]
    if duplicate_examples:
        parts.extend(duplicate_examples)
    else:
        parts.append("- （无明显重复簇）")

    parts.extend(["", "## 关系示例"])
    if relation_examples:
        parts.extend(relation_examples)
    else:
        parts.append("- （无明显关系）")
    return "\n".join(parts) + "\n"
