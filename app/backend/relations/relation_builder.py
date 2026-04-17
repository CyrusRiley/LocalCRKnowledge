from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from uuid import uuid4

from app.backend.database.repository import KnowledgeRepository
from app.backend.utils.time_utils import utc_now_iso


def build_structural_relations(repo: KnowledgeRepository, *, max_keyword_pairs_per_term: int = 60) -> list[dict]:
    now = utc_now_iso()
    units = repo.list_units_for_relation_build()
    return _build_relations_from_units(units, repo.list_unit_keyword_links(), now, max_keyword_pairs_per_term=max_keyword_pairs_per_term)


def build_relations_for_notes(
    repo: KnowledgeRepository,
    affected_note_ids: set[str],
    *,
    max_keyword_pairs_per_term: int = 60,
) -> list[dict]:
    now = utc_now_iso()
    normalized_ids = {str(note_id) for note_id in affected_note_ids if note_id}
    if not normalized_ids:
        return []
    units = repo.list_units_for_relation_build()
    relations = _build_relations_from_units(
        units,
        repo.list_unit_keyword_links(),
        now,
        affected_note_ids=normalized_ids,
        max_keyword_pairs_per_term=max_keyword_pairs_per_term,
    )
    return dedupe_relations(relations)


def _build_relations_from_units(
    units: list[dict],
    keyword_links: list[dict],
    now: str,
    *,
    affected_note_ids: set[str] | None = None,
    max_keyword_pairs_per_term: int,
) -> list[dict]:
    affected = affected_note_ids or set()
    relations: list[dict] = []
    by_group: dict[str, list[dict]] = defaultdict(list)
    for unit in units:
        group_id = str(unit.get("group_id") or "")
        note_id = str(unit.get("note_id") or "")
        if group_id and note_id:
            by_group[group_id].append(unit)

    for group_units in by_group.values():
        ordered = sorted(group_units, key=lambda item: (int(item.get("order_index") or 0), str(item.get("note_id") or "")))
        for left, right in zip(ordered, ordered[1:]):
            if affected and left["note_id"] not in affected and right["note_id"] not in affected:
                continue
            relations.append(
                _relation(
                    left["note_id"],
                    right["note_id"],
                    "sequence_next",
                    0.92,
                    "adjacent units in the same source group",
                    now,
                    relation_layer="structure",
                    relation_strength="strong",
                    display_default=True,
                )
            )

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
        if len(unique_links) > max_keyword_pairs_per_term:
            continue
        for left, right in combinations(unique_links, 2):
            if affected and str(left["note_id"]) not in affected and str(right["note_id"]) not in affected:
                continue
            pair = tuple(sorted((str(left["note_id"]), str(right["note_id"]))))
            count, names = shared_counts.get(pair, (0, []))
            canonical = str(left.get("canonical_name") or "")
            shared_counts[pair] = (count + 1, [*names, canonical] if canonical and canonical not in names else names)

    for (left_id, right_id), (count, names) in shared_counts.items():
        if left_id == right_id:
            continue
        score = min(0.78, 0.46 + count * 0.08)
        if score < 0.62:
            continue
        reason = "shared canonical keywords"
        if names:
            reason = f"shared canonical keywords: {', '.join(names[:5])}"
        relations.append(
            _relation(
                left_id,
                right_id,
                "shared_keyword",
                score,
                reason,
                now,
                relation_layer="weak",
                relation_strength="weak",
                display_default=False,
            )
        )

    relations.extend(_semantic_relations(by_group, now, affected_note_ids=affected))
    return dedupe_relations(relations)


def dedupe_relations(relations: list[dict]) -> list[dict]:
    best: dict[tuple[str, str, str], dict] = {}
    note_ids: set[str] = set()
    for relation in relations:
        left = str(relation.get("from_note_id") or "")
        right = str(relation.get("to_note_id") or "")
        rel_type = str(relation.get("relation_type") or "")
        if not left or not right or left == right or not rel_type:
            continue
        note_ids.update([left, right])
        relation = _normalize_relation_metadata(relation)
        if rel_type in {"shared_keyword", "related_to", "contrasts"}:
            left, right = sorted((left, right))
            relation = {**relation, "from_note_id": left, "to_note_id": right}
        key = (left, right, rel_type)
        current = best.get(key)
        if current is None or float(relation.get("score", 0.0)) > float(current.get("score", 0.0)):
            best[key] = relation
    return _cap_relations(list(best.values()), note_count=len(note_ids))


def _relation(
    from_note_id: str,
    to_note_id: str,
    relation_type: str,
    score: float,
    reason: str,
    now: str,
    *,
    relation_layer: str = "semantic",
    relation_strength: str = "medium",
    display_default: bool = True,
) -> dict:
    return {
        "relation_id": str(uuid4()),
        "from_note_id": from_note_id,
        "to_note_id": to_note_id,
        "relation_type": relation_type,
        "relation_layer": relation_layer,
        "relation_strength": relation_strength,
        "display_default": display_default,
        "score": round(score, 4),
        "reason": reason,
        "created_at": now,
    }


def _semantic_relations(by_group: dict[str, list[dict]], now: str, *, affected_note_ids: set[str] | None = None) -> list[dict]:
    relations: list[dict] = []
    affected = affected_note_ids or set()
    for group_units in by_group.values():
        ordered = sorted(group_units, key=lambda item: (int(item.get("order_index") or 0), str(item.get("note_id") or "")))
        for left, right in combinations(ordered[:90], 2):
            if affected and str(left.get("note_id") or "") not in affected and str(right.get("note_id") or "") not in affected:
                continue
            relations.extend(_classify_semantic_pair(left, right, now))
    return relations


def _classify_semantic_pair(left: dict, right: dict, now: str) -> list[dict]:
    left_id = str(left.get("note_id") or "")
    right_id = str(right.get("note_id") or "")
    if not left_id or not right_id or left_id == right_id:
        return []

    left_text = _unit_text(left)
    right_text = _unit_text(right)
    relations: list[dict] = []

    method_left = _has_any(left_text, _METHOD_MARKERS)
    method_right = _has_any(right_text, _METHOD_MARKERS)
    application_left = _has_any(left_text, _APPLICATION_MARKERS)
    application_right = _has_any(right_text, _APPLICATION_MARKERS)
    example_left = _has_any(left_text, _EXAMPLE_MARKERS)
    example_right = _has_any(right_text, _EXAMPLE_MARKERS)
    concept_left = _has_any(left_text, _CONCEPT_MARKERS)
    concept_right = _has_any(right_text, _CONCEPT_MARKERS)
    background_left = _has_any(left_text, _BACKGROUND_MARKERS)
    background_right = _has_any(right_text, _BACKGROUND_MARKERS)
    evidence_left = _has_any(left_text, _EVIDENCE_MARKERS)
    evidence_right = _has_any(right_text, _EVIDENCE_MARKERS)
    extension_left = _has_any(left_text, _EXTENSION_MARKERS)
    extension_right = _has_any(right_text, _EXTENSION_MARKERS)

    if method_left and application_right:
        relations.append(_relation(left_id, right_id, "method_for", 0.72, "method-like unit points to application-like unit", now, relation_strength="strong"))
    if method_right and application_left:
        relations.append(_relation(right_id, left_id, "method_for", 0.72, "method-like unit points to application-like unit", now, relation_strength="strong"))

    if example_left and concept_right:
        relations.append(_relation(left_id, right_id, "example_of", 0.7, "example/case unit illustrates concept/framework unit", now, relation_strength="strong"))
    if example_right and concept_left:
        relations.append(_relation(right_id, left_id, "example_of", 0.7, "example/case unit illustrates concept/framework unit", now, relation_strength="strong"))

    if background_left and (method_right or application_right or concept_right):
        relations.append(_relation(left_id, right_id, "background_for", 0.68, "background/review unit precedes a more operational unit", now, relation_strength="medium"))
    if background_right and (method_left or application_left or concept_left):
        relations.append(_relation(right_id, left_id, "background_for", 0.68, "background/review unit precedes a more operational unit", now, relation_strength="medium"))

    if evidence_left and (concept_right or method_right or application_right):
        relations.append(_relation(left_id, right_id, "supports", 0.66, "evidence/data/result unit may support the paired unit", now, relation_strength="medium"))
    if evidence_right and (concept_left or method_left or application_left):
        relations.append(_relation(right_id, left_id, "supports", 0.66, "evidence/data/result unit may support the paired unit", now, relation_strength="medium"))

    if _has_any(left_text + "\n" + right_text, _CONTRAST_MARKERS):
        relations.append(_relation(left_id, right_id, "contrasts", 0.64, "contrast/limitation markers found in paired units", now, relation_strength="medium"))

    if extension_left and (concept_right or method_right):
        relations.append(_relation(left_id, right_id, "extends", 0.63, "extension/improvement unit may extend the paired concept or method", now, relation_strength="medium"))
    if extension_right and (concept_left or method_left):
        relations.append(_relation(right_id, left_id, "extends", 0.63, "extension/improvement unit may extend the paired concept or method", now, relation_strength="medium"))

    return relations[:4]


def _unit_text(unit: dict) -> str:
    attrs = unit.get("attributes") or {}
    pieces = [
        str(unit.get("title") or ""),
        str(unit.get("content") or ""),
        str(unit.get("evidence") or ""),
        str(unit.get("note_summary") or ""),
        str(unit.get("note_faithful_content") or ""),
        str(unit.get("group_title") or ""),
        " ".join(str(item) for item in unit.get("themes") or []),
        " ".join(str(item) for item in unit.get("keywords") or []),
        " ".join(str(item) for item in attrs.get("key_points") or []),
        str(attrs.get("faithful_content") or ""),
    ]
    return "\n".join(piece for piece in pieces if piece).lower()


def _has_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker.lower() in text for marker in markers)


_METHOD_MARKERS = ("方法", "技术", "模型", "流程", "步骤", "机制", "算法", "参数", "method", "model", "protocol", "procedure")
_APPLICATION_MARKERS = ("应用", "用于", "场景", "任务", "案例", "实践", "写作", "use", "application", "case", "scenario")
_EXAMPLE_MARKERS = ("案例", "例子", "例如", "样本", "实证", "观察", "example", "case", "sample", "observation")
_CONCEPT_MARKERS = ("概念", "理论", "框架", "主题", "问题", "定义", "观点", "concept", "theory", "framework", "argument")
_BACKGROUND_MARKERS = ("背景", "综述", "历史", "发展", "脉络", "文献", "background", "review", "history", "literature")
_EVIDENCE_MARKERS = ("证据", "数据", "结果", "发现", "表明", "支持", "验证", "evidence", "data", "result", "finding", "support")
_CONTRAST_MARKERS = ("但是", "然而", "相反", "不足", "局限", "限制", "冲突", "反驳", "but", "however", "contrast", "limitation")
_EXTENSION_MARKERS = ("扩展", "延伸", "进一步", "补充", "改进", "细化", "优化", "extend", "extension", "improve", "refine")


def _normalize_relation_metadata(relation: dict) -> dict:
    rel_type = str(relation.get("relation_type") or "")
    layer = str(relation.get("relation_layer") or "")
    strength = str(relation.get("relation_strength") or "")
    display_default = relation.get("display_default")
    if not layer:
        layer = _default_layer(rel_type)
    if not strength:
        strength = _default_strength(rel_type)
    if display_default is None:
        display_default = layer != "weak" and rel_type not in {"related_to", "shared_keyword"}
    return {
        **relation,
        "relation_layer": layer,
        "relation_strength": strength,
        "display_default": bool(display_default),
    }


def _default_layer(relation_type: str) -> str:
    if relation_type in {"sequence_next"}:
        return "structure"
    if relation_type in {"shared_keyword", "related_to"}:
        return "weak"
    return "semantic"


def _default_strength(relation_type: str) -> str:
    if relation_type in {"duplicate_of", "sequence_next", "method_for", "example_of"}:
        return "strong"
    if relation_type in {"shared_keyword", "related_to"}:
        return "weak"
    return "medium"


def _cap_relations(relations: list[dict], *, note_count: int) -> list[dict]:
    priority = {"strong": 0, "medium": 1, "weak": 2}
    ordered = sorted(
        relations,
        key=lambda item: (
            priority.get(str(item.get("relation_strength") or "medium"), 1),
            0 if item.get("display_default") else 1,
            -float(item.get("score", 0.0)),
            str(item.get("relation_type") or ""),
        ),
    )
    per_node_caps = {
        ("structure", "strong"): 8,
        ("semantic", "strong"): 8,
        ("semantic", "medium"): 5,
        ("weak", "weak"): 2,
    }
    always_keep = {"duplicate_of", "sequence_next"}
    incident: dict[tuple[str, str, str], int] = defaultdict(int)
    kept: list[dict] = []
    dynamic_total = max(300, note_count * 14)
    for relation in ordered:
        rel_type = str(relation.get("relation_type") or "")
        layer = str(relation.get("relation_layer") or "semantic")
        strength = str(relation.get("relation_strength") or "medium")
        if layer == "weak" and float(relation.get("score", 0.0)) < 0.62:
            continue
        if layer == "semantic" and float(relation.get("score", 0.0)) < 0.62:
            continue
        left = str(relation.get("from_note_id") or "")
        right = str(relation.get("to_note_id") or "")
        cap = per_node_caps.get((layer, strength), 4)
        left_key = (left, layer, strength)
        right_key = (right, layer, strength)
        if rel_type not in always_keep and (incident[left_key] >= cap or incident[right_key] >= cap):
            continue
        kept.append(relation)
        incident[left_key] += 1
        incident[right_key] += 1
        if len(kept) >= dynamic_total:
            break
    return kept
