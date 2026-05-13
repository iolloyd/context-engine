"""Parity tests for storage backends.

Every backend that claims to implement ``StoreProtocol`` runs through the
same scenarios. The SQLite ``GraphStore`` is always exercised. Additional
backends (currently ``PgGraphStore``, when it lands per ADR 0002) attach
to the ``BACKENDS`` registry below and are skipped automatically when
their preconditions are not met.

Add a backend by appending one entry to ``BACKENDS``. The factory must
return a fresh, empty store that satisfies ``StoreProtocol`` and accepts
a configurable ``embed_dim``.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from datetime import date, datetime

import pytest

from context_engine.store import GraphStore
from context_engine.types import StoreProtocol

# A tiny embedding dimension keeps tests fast and deterministic — backends
# must accept ``embed_dim`` at construction time.
EMBED_DIM = 4

StoreFactory = Callable[[], StoreProtocol]


def _sqlite_factory() -> StoreProtocol:
    return GraphStore(":memory:", embed_dim=EMBED_DIM)


def _pg_factory() -> StoreProtocol:
    dsn = os.environ.get("CTX_PG_DSN")
    if not dsn:
        pytest.skip("CTX_PG_DSN not set — Postgres backend not configured")
    try:
        from context_engine.pg_store import PgGraphStore  # type: ignore[import-not-found]
    except ImportError:
        pytest.skip("PgGraphStore not implemented yet (see ADR 0002)")
    return PgGraphStore(dsn=dsn, embed_dim=EMBED_DIM, schema_isolation="test")  # type: ignore[call-arg]


BACKENDS: list[tuple[str, StoreFactory]] = [
    ("sqlite", _sqlite_factory),
    ("postgres", _pg_factory),
]


@pytest.fixture(params=BACKENDS, ids=lambda b: b[0])
def store(request: pytest.FixtureRequest) -> Iterator[StoreProtocol]:
    _, factory = request.param
    s = factory()
    try:
        yield s
    finally:
        s.close()


# ── protocol conformance ───────────────────────────────────────────────────


def test_backend_implements_protocol(store: StoreProtocol) -> None:
    assert isinstance(store, StoreProtocol)


# ── node CRUD ──────────────────────────────────────────────────────────────


def test_node_round_trip(store: StoreProtocol) -> None:
    n = store.upsert_node("hello", metadata={"type": "fact"})
    fetched = store.get_node(n.id)
    assert fetched is not None
    assert fetched.content == "hello"
    assert fetched.type == "fact"


def test_node_upsert_updates_existing(store: StoreProtocol) -> None:
    n = store.upsert_node("first", node_id="fixed-id")
    again = store.upsert_node("second", node_id="fixed-id")
    assert n.id == again.id
    fetched = store.get_node("fixed-id")
    assert fetched is not None
    assert fetched.content == "second"


def test_get_nodes_batch(store: StoreProtocol) -> None:
    a = store.upsert_node("a", node_id="a")
    b = store.upsert_node("b", node_id="b")
    store.upsert_node("c", node_id="c")
    fetched = store.get_nodes([a.id, b.id])
    assert {n.id for n in fetched} == {"a", "b"}


def test_get_nodes_empty_iterable(store: StoreProtocol) -> None:
    assert store.get_nodes([]) == []


def test_delete_node_cascades_edges(store: StoreProtocol) -> None:
    store.upsert_node("a", node_id="a")
    store.upsert_node("b", node_id="b")
    store.upsert_edge("a", "b", edge_type="links")
    store.delete_node("a")
    assert store.get_node("a") is None
    # edge whose source was deleted must be gone too
    assert store.out_edges("a") == []
    assert store.in_edges("b") == []


def test_filter_nodes_by_metadata(store: StoreProtocol) -> None:
    store.upsert_node("ohp", node_id="ohp", metadata={"type": "exercise", "status": "stalled"})
    store.upsert_node("squat", node_id="squat", metadata={"type": "exercise", "status": "avoided"})
    store.upsert_node("bench", node_id="bench", metadata={"type": "exercise"})
    avoided = store.filter_nodes({"status": "avoided"})
    assert [n.id for n in avoided] == ["squat"]


def test_filter_nodes_empty_where_returns_all(store: StoreProtocol) -> None:
    store.upsert_node("a")
    store.upsert_node("b")
    assert len(store.filter_nodes({})) == 2


def test_node_metadata_date_coercion(store: StoreProtocol) -> None:
    """YAML frontmatter parses dates as ``date`` / ``datetime``; both must
    round-trip as ISO strings rather than break the JSON encoder."""
    store.upsert_node(
        "hello",
        node_id="dated",
        metadata={
            "type": "decision",
            "decided_on": date(2026, 2, 14),
            "noticed_at": datetime(2026, 2, 14, 10, 30),
        },
    )
    fetched = store.get_node("dated")
    assert fetched is not None
    assert fetched.metadata["decided_on"] == "2026-02-14"
    assert str(fetched.metadata["noticed_at"]).startswith("2026-02-14T10:30")


# ── edge CRUD ──────────────────────────────────────────────────────────────


def test_edge_round_trip(store: StoreProtocol) -> None:
    store.upsert_node("a", node_id="a")
    store.upsert_node("b", node_id="b")
    e = store.upsert_edge("a", "b", edge_type="requires", weight=0.7, content="why")
    fetched = store.get_edge(e.id)
    assert fetched is not None
    assert fetched.source == "a"
    assert fetched.target == "b"
    assert fetched.type == "requires"
    assert fetched.weight == 0.7
    assert fetched.content == "why"


def test_edge_weight_update_is_clamped(store: StoreProtocol) -> None:
    store.upsert_node("a", node_id="a")
    store.upsert_node("b", node_id="b")
    e = store.upsert_edge("a", "b", edge_type="requires", weight=0.5)
    store.update_edge_weight(e.id, 1.5)
    refreshed = store.get_edge(e.id)
    assert refreshed is not None
    assert refreshed.weight == 1.0
    store.update_edge_weight(e.id, -0.2)
    refreshed = store.get_edge(e.id)
    assert refreshed is not None
    assert refreshed.weight == 0.0


def test_out_edges_filter_by_type(store: StoreProtocol) -> None:
    store.upsert_node("a", node_id="a")
    store.upsert_node("b", node_id="b")
    store.upsert_node("c", node_id="c")
    store.upsert_edge("a", "b", edge_type="requires", weight=0.6)
    store.upsert_edge("a", "c", edge_type="because_of", weight=0.6)
    edges = store.out_edges("a", edge_types=["requires"])
    assert len(edges) == 1
    assert edges[0].type == "requires"


def test_out_edges_filter_by_weight_floor(store: StoreProtocol) -> None:
    store.upsert_node("a", node_id="a")
    store.upsert_node("b", node_id="b")
    store.upsert_node("c", node_id="c")
    store.upsert_edge("a", "b", edge_type="requires", weight=0.1)
    store.upsert_edge("a", "c", edge_type="requires", weight=0.8)
    edges = store.out_edges("a", weight_floor=0.5)
    assert {e.target for e in edges} == {"c"}


def test_in_edges_symmetric_to_out(store: StoreProtocol) -> None:
    store.upsert_node("a", node_id="a")
    store.upsert_node("b", node_id="b")
    store.upsert_edge("a", "b", edge_type="links", weight=0.5)
    assert [e.source for e in store.in_edges("b")] == ["a"]
    assert [e.target for e in store.out_edges("a")] == ["b"]


# ── embeddings & knn ───────────────────────────────────────────────────────


def test_embedding_round_trip(store: StoreProtocol) -> None:
    vec = [0.1, 0.2, 0.3, 0.4]
    store.upsert_node("a", node_id="a", embedding=vec)
    assert store.has_embedding("a") is True
    retrieved = store.get_embedding("a")
    assert retrieved is not None
    assert len(retrieved) == EMBED_DIM
    for got, want in zip(retrieved, vec, strict=True):
        assert abs(got - want) < 1e-6


def test_has_embedding_false_when_unset(store: StoreProtocol) -> None:
    store.upsert_node("a", node_id="a")
    assert store.has_embedding("a") is False
    assert store.get_embedding("a") is None


def test_knn_returns_closest_first(store: StoreProtocol) -> None:
    store.upsert_node("near", node_id="near", embedding=[1.0, 0.0, 0.0, 0.0])
    store.upsert_node("mid", node_id="mid", embedding=[0.5, 0.5, 0.0, 0.0])
    store.upsert_node("far", node_id="far", embedding=[0.0, 1.0, 0.0, 0.0])
    results = store.knn([1.0, 0.0, 0.0, 0.0], k=3)
    assert [node_id for node_id, _ in results] == ["near", "mid", "far"]


def test_knn_k_truncates(store: StoreProtocol) -> None:
    for i in range(5):
        v = [0.0] * EMBED_DIM
        v[i % EMBED_DIM] = 1.0 - i * 0.1
        store.upsert_node(f"n{i}", node_id=f"n{i}", embedding=v)
    assert len(store.knn([1.0, 0.0, 0.0, 0.0], k=2)) == 2


# ── inventory & counts ─────────────────────────────────────────────────────


def test_counts_track_writes(store: StoreProtocol) -> None:
    assert store.node_count() == 0
    assert store.edge_count() == 0
    store.upsert_node("a", node_id="a")
    store.upsert_node("b", node_id="b")
    store.upsert_edge("a", "b", edge_type="links")
    assert store.node_count() == 2
    assert store.edge_count() == 1


def test_all_node_ids_and_all_edges(store: StoreProtocol) -> None:
    store.upsert_node("a", node_id="a")
    store.upsert_node("b", node_id="b")
    store.upsert_edge("a", "b", edge_type="links")
    assert set(store.all_node_ids()) == {"a", "b"}
    edges = store.all_edges()
    assert len(edges) == 1
    assert edges[0].source == "a" and edges[0].target == "b"


# ── transactions ───────────────────────────────────────────────────────────


def test_transaction_commits_on_clean_exit(store: StoreProtocol) -> None:
    with store.transaction():
        store.upsert_node("inside", node_id="committed")
    assert store.get_node("committed") is not None


def test_transaction_rolls_back_on_exception(store: StoreProtocol) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with store.transaction():
            store.upsert_node("inside", node_id="rolled-back")
            raise RuntimeError("boom")
    assert store.get_node("rolled-back") is None
