from __future__ import annotations

import re
from typing import Any

from app.backend.database.repository import KnowledgeRepository


def build_library_graph(repo: KnowledgeRepository, *, limit: int = 120) -> dict[str, Any]:
    units = repo.list_units_for_graph(limit=limit)
    unit_ids = [str(unit.get("unit_id") or "") for unit in units]
    keyword_links = repo.list_keyword_links_for_units(unit_ids)
    links_by_unit: dict[str, list[dict[str, Any]]] = {}
    for link in keyword_links:
        links_by_unit.setdefault(str(link.get("unit_id") or ""), []).append(link)

    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}

    def add_node(node_id: str, label: str, node_type: str, **extra: Any) -> None:
        if not node_id or not label:
            return
        current = nodes.get(node_id)
        data = {"id": node_id, "label": label, "node_type": node_type, **extra}
        if current:
            current.update({key: value for key, value in data.items() if value not in (None, "", [])})
        else:
            nodes[node_id] = data

    def add_edge(from_id: str, to_id: str, relation_type: str, *, layer: str, strength: str, score: float, reason: str = "") -> None:
        if not from_id or not to_id or from_id == to_id:
            return
        key = (from_id, to_id, relation_type)
        edge = {
            "from_id": from_id,
            "to_id": to_id,
            "relation_type": relation_type,
            "relation_layer": layer,
            "relation_strength": strength,
            "score": round(float(score), 4),
            "reason": reason,
        }
        current = edges.get(key)
        if current is None or edge["score"] > float(current.get("score", 0.0)):
            edges[key] = edge

    visible_note_ids: set[str] = set()
    concept_theme_links: set[tuple[str, str]] = set()

    for unit in units:
        note_id = str(unit.get("note_id") or "")
        if not note_id:
            continue
        visible_note_ids.add(note_id)
        unit_label = str(unit.get("title") or unit.get("note_title") or "未命名知识")
        unit_kind = _classify_unit(unit)
        add_node(
            note_id,
            unit_label,
            "unit",
            unit_kind=unit_kind,
            note_id=note_id,
            note_type=unit.get("note_type") or unit.get("note_note_type") or "",
            summary=unit.get("note_summary") or unit.get("content") or "",
            updated_at=unit.get("note_updated_at") or unit.get("updated_at") or "",
        )

        group_id = str(unit.get("group_id") or "")
        if group_id:
            group_node_id = f"group:{group_id}"
            add_node(group_node_id, str(unit.get("group_title") or "知识组"), "group", group_id=group_id)
            add_edge(group_node_id, note_id, "contains", layer="structure", strength="strong", score=0.96, reason="same imported source group")

        themes = [str(item).strip() for item in unit.get("themes") or [] if str(item).strip()]
        for theme in themes[:4]:
            topic_id = f"topic:{_slug(theme)}"
            add_node(topic_id, theme, "topic")
            add_edge(topic_id, note_id, "topic_contains", layer="structure", strength="medium", score=0.78, reason="theme tag")

        for link in links_by_unit.get(str(unit.get("unit_id") or ""), [])[:8]:
            term_id = str(link.get("term_id") or "")
            concept_id = f"concept:{term_id}"
            concept_label = str(link.get("canonical_name") or "")
            if not term_id or not concept_label:
                continue
            add_node(concept_id, concept_label, "concept", term_id=term_id)
            add_edge(
                concept_id,
                note_id,
                "concept_contains",
                layer="structure",
                strength="strong",
                score=float(link.get("confidence") or 0.82),
                reason="canonical keyword",
            )
            for theme in themes[:3]:
                topic_id = f"topic:{_slug(theme)}"
                pair = (topic_id, concept_id)
                if pair not in concept_theme_links:
                    concept_theme_links.add(pair)
                    add_edge(topic_id, concept_id, "topic_contains", layer="structure", strength="medium", score=0.72, reason="theme to concept")

    for relation in repo.list_display_relations_for_latest_run(limit=max(80, limit * 3)):
        from_id = str(relation.get("from_note_id") or "")
        to_id = str(relation.get("to_note_id") or "")
        if from_id not in visible_note_ids or to_id not in visible_note_ids:
            continue
        add_edge(
            from_id,
            to_id,
            str(relation.get("relation_type") or "related"),
            layer=str(relation.get("relation_layer") or "semantic"),
            strength=str(relation.get("relation_strength") or "medium"),
            score=float(relation.get("score") or 0.0),
            reason=str(relation.get("reason") or ""),
        )

    edge_values = sorted(
        edges.values(),
        key=lambda item: (
            _layer_order(str(item.get("relation_layer") or "")),
            _strength_order(str(item.get("relation_strength") or "")),
            -float(item.get("score") or 0.0),
        ),
    )
    return {
        "nodes": list(nodes.values()),
        "edges": edge_values[: max(160, limit * 5)],
        "summary": {
            "nodes": len(nodes),
            "edges": len(edge_values),
            "unit_nodes": sum(1 for node in nodes.values() if node.get("node_type") == "unit"),
            "concept_nodes": sum(1 for node in nodes.values() if node.get("node_type") == "concept"),
            "topic_nodes": sum(1 for node in nodes.values() if node.get("node_type") == "topic"),
            "group_nodes": sum(1 for node in nodes.values() if node.get("node_type") == "group"),
            "structure_edges": sum(1 for edge in edge_values if edge.get("relation_layer") == "structure"),
            "semantic_edges": sum(1 for edge in edge_values if edge.get("relation_layer") == "semantic"),
        },
    }


def _classify_unit(unit: dict[str, Any]) -> str:
    text = " ".join(
        [
            str(unit.get("title") or ""),
            str(unit.get("content") or ""),
            str(unit.get("note_summary") or ""),
            str(unit.get("note_faithful_content") or ""),
            str(unit.get("note_type") or ""),
        ]
    ).lower()
    if _has_any(text, ("方法", "技术", "模型", "流程", "method", "model")):
        return "method"
    if _has_any(text, ("案例", "实证", "观察", "数据", "证据", "case", "evidence", "data", "observation")):
        return "evidence"
    if _has_any(text, ("问题", "疑问", "假设", "question", "hypothesis")):
        return "question"
    if _has_any(text, ("结论", "观点", "发现", "conclusion", "finding", "argument")):
        return "claim"
    return "general"


def _has_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _slug(value: str) -> str:
    return re.sub(r"\s+", "-", value.strip().lower())


def _layer_order(layer: str) -> int:
    return {"structure": 0, "semantic": 1, "weak": 2}.get(layer, 3)


def _strength_order(strength: str) -> int:
    return {"strong": 0, "medium": 1, "weak": 2}.get(strength, 3)
