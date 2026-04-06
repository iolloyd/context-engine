from __future__ import annotations

from context_engine.store import GraphStore


def test_round_trip_node() -> None:
    s = GraphStore(":memory:")
    n = s.upsert_node("hello", metadata={"type": "fact"})
    fetched = s.get_node(n.id)
    assert fetched is not None
    assert fetched.content == "hello"
    assert fetched.type == "fact"


def test_edge_weight_update() -> None:
    s = GraphStore(":memory:")
    a = s.upsert_node("a")
    b = s.upsert_node("b")
    e = s.upsert_edge(a.id, b.id, edge_type="requires", weight=0.5)
    s.update_edge_weight(e.id, 0.8)
    refreshed = s.get_edge(e.id)
    assert refreshed is not None
    assert refreshed.weight == 0.8


def test_filter_nodes_by_metadata(store: GraphStore) -> None:
    avoided = store.filter_nodes({"status": "avoided"})
    assert [n.id for n in avoided] == ["squat"]


def test_out_edges_filter(store: GraphStore) -> None:
    edges = store.out_edges("ohp", edge_types=["has_observation"])
    assert len(edges) == 5
    assert all(e.type == "has_observation" for e in edges)
