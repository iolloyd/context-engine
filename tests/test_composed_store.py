"""Tests for ComposedStore."""

from __future__ import annotations

from context_engine.composed_store import ComposedStore
from context_engine.store import GraphStore


def _make_store() -> GraphStore:
    return GraphStore(":memory:")


def test_project_node_wins_on_id_collision() -> None:
    project = _make_store()
    global_ = _make_store()
    project.upsert_node("project version", node_id="x")
    global_.upsert_node("global version", node_id="x")

    composed = ComposedStore(project, global_)
    node = composed.get_node("x")
    assert node is not None
    assert node.content == "project version"


def test_global_node_visible_when_absent_from_project() -> None:
    project = _make_store()
    global_ = _make_store()
    global_.upsert_node("global only", node_id="y")

    composed = ComposedStore(project, global_)
    node = composed.get_node("y")
    assert node is not None
    assert node.content == "global only"


def test_out_edges_union() -> None:
    project = _make_store()
    global_ = _make_store()

    project.upsert_node("a", node_id="a")
    project.upsert_node("b", node_id="b")
    project.upsert_edge("a", "b", edge_type="type1")

    global_.upsert_node("a", node_id="a")
    global_.upsert_node("c", node_id="c")
    global_.upsert_edge("a", "c", edge_type="type2")

    composed = ComposedStore(project, global_)
    edges = composed.out_edges("a")
    targets = {e.target for e in edges}
    assert "b" in targets
    assert "c" in targets


def test_all_node_ids_merged() -> None:
    project = _make_store()
    global_ = _make_store()

    project.upsert_node("a", node_id="a")
    project.upsert_node("b", node_id="b")
    global_.upsert_node("b", node_id="b")
    global_.upsert_node("c", node_id="c")

    composed = ComposedStore(project, global_)
    ids = set(composed.all_node_ids())
    assert ids == {"a", "b", "c"}
