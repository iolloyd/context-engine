from __future__ import annotations

from datetime import date, datetime

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


def test_upsert_node_coerces_date_metadata() -> None:
    """YAML frontmatter parses dates as datetime.date; they must round-trip
    as ISO strings rather than explode the JSON encoder."""
    s = GraphStore(":memory:")
    n = s.upsert_node(
        "hello",
        metadata={
            "type": "decision",
            "decided_on": date(2026, 2, 14),
            "noticed_at": datetime(2026, 2, 14, 10, 30),
        },
    )
    fetched = s.get_node(n.id)
    assert fetched is not None
    assert fetched.metadata["decided_on"] == "2026-02-14"
    assert fetched.metadata["noticed_at"].startswith("2026-02-14T10:30")
