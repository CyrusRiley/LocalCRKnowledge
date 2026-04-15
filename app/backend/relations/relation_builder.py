from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from uuid import uuid4

from app.backend.database.repository import KnowledgeRepository
from app.backend.utils.time_utils import utc_now_iso


def build_structural_relations(repo: KnowledgeRepository, *, max_keyword_pairs_per_term: int = 220) -> list[dict]:
    now = utc_now_iso()
    relations: list[dict] = []
    units = repo.list_units_for_relation_build()

    by_group: dict[str, list[dict]] = defaultdict(list)
    for unit in units:
        group_id = str(unit.get("group_id") or "")
        note_id = str(unit.get("note_id") or "")
        if group_id and note_id:
            by_group[group_id].append(unit)

    for group_units in by_group.values():
        ordered = sorted(group_units, key=lambda item: (int(item.get("order_index") or 0), str(item.get("note_id") or "")))
        for left, right in zip(ordered, ordered[1:]):
            relations.append(
                _relation(
                    left["note_id"],
                    right["note_id"],
                    "sequence_next",
                    0.92,
                    "adjacent units in the same source group",
                    now,
                )
            )
        for left, right in combinations(ordered[:80], 2):
            relations.append(
                _relation(
                    left["note_id"],
                    right["note_id"],
                    "same_group",
                    0.72,
                    "units belong to the same imported source group",
                    now,
                )
            )

    keyword_links = repo.list_unit_keyword_links()
    by_term: dict[str, list[dict]] = defaultdict(list)
    for link in keyword_links:
        if link.get("term_id") and link.get("note_id"):
            by_term[str(link["term_id"])].append(link)

    shared_counts: dict[tuple[str, str], tuple[int, list[str]]] = {}
    for links in by_term.values():
        unique_by_note = {str(link["note_id"]): link for link in links}
        unique_links = list(unique_by_note.values())
        if len(unique_links) < 2:
            continue
        for left, right in combinations(unique_links[:max_keyword_pairs_per_term], 2):
            pair = tuple(sorted((str(left["note_id"]), str(right["note_id"]))))
            count, names = shared_counts.get(pair, (0, []))
            canonical = str(left.get("canonical_name") or "")
            shared_counts[pair] = (count + 1, [*names, canonical] if canonical and canonical not in names else names)

    for (left_id, right_id), (count, names) in shared_counts.items():
        if left_id == right_id:
            continue
        score = min(0.88, 0.48 + count * 0.08)
        reason = "shared canonical keywords"
        if names:
            reason = f"shared canonical keywords: {', '.join(names[:5])}"
        relations.append(_relation(left_id, right_id, "shared_keyword", score, reason, now))

    return dedupe_relations(relations)


def dedupe_relations(relations: list[dict]) -> list[dict]:
    best: dict[tuple[str, str, str], dict] = {}
    for relation in relations:
        left = str(relation.get("from_note_id") or "")
        right = str(relation.get("to_note_id") or "")
        rel_type = str(relation.get("relation_type") or "")
        if not left or not right or left == right or not rel_type:
            continue
        if rel_type in {"same_group", "shared_keyword", "related_to"}:
            left, right = sorted((left, right))
            relation = {**relation, "from_note_id": left, "to_note_id": right}
        key = (left, right, rel_type)
        current = best.get(key)
        if current is None or float(relation.get("score", 0.0)) > float(current.get("score", 0.0)):
            best[key] = relation
    return sorted(best.values(), key=lambda item: (-float(item.get("score", 0.0)), item["relation_type"]))[:5000]


def _relation(
    from_note_id: str,
    to_note_id: str,
    relation_type: str,
    score: float,
    reason: str,
    now: str,
) -> dict:
    return {
        "relation_id": str(uuid4()),
        "from_note_id": from_note_id,
        "to_note_id": to_note_id,
        "relation_type": relation_type,
        "score": round(score, 4),
        "reason": reason,
        "created_at": now,
    }

