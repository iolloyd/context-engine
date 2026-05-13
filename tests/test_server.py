"""Smoke tests for the FastAPI service.

Exercises the recall endpoint against an in-memory SQLite engine — no
Postgres required. Verifies the wire format that brain's edge functions
will consume.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="serve extra not installed")
pytest.importorskip("httpx", reason="fastapi TestClient depends on httpx")

from fastapi.testclient import TestClient  # noqa: E402

from context_engine.engine import ContextEngine  # noqa: E402
from context_engine.server import build_app  # noqa: E402
from context_engine.store import GraphStore  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    store = GraphStore(":memory:", embed_dim=4)
    store.upsert_node(
        "the squat is a compound lower-body lift",
        node_id="exercises/squat",
        metadata={"type": "exercise", "status": "avoided"},
        embedding=[1.0, 0.0, 0.0, 0.0],
    )
    store.upsert_node(
        "knee pain from a chronic meniscus issue",
        node_id="conditions/knee-pain",
        metadata={"type": "condition"},
        embedding=[0.9, 0.1, 0.0, 0.0],
    )
    store.upsert_edge(
        "exercises/squat",
        "conditions/knee-pain",
        edge_type="because_of",
        weight=0.9,
        content="squats aggravate the meniscus",
    )
    engine = ContextEngine(store=store)
    app = build_app(engine)
    return TestClient(app)


def test_health_returns_counts(client: TestClient) -> None:
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["node_count"] == 2
    assert body["edge_count"] == 1


def test_recall_returns_slice_for_explicit_seed(client: TestClient) -> None:
    res = client.post(
        "/recall",
        json={
            "query": "why do I avoid squats?",
            "explicit_refs": ["exercises/squat"],
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert "exercises/squat" in body["slice"]["seeds"]
    node_ids = {n["id"] for n in body["slice"]["nodes"]}
    assert "exercises/squat" in node_ids
    assert body["intent"] and body["focus"]  # classified
    assert isinstance(body["confidence"], float)


def test_recall_rejects_empty_query(client: TestClient) -> None:
    res = client.post("/recall", json={"query": "   "})
    assert res.status_code == 400


def test_recall_propagates_metadata_filter(client: TestClient) -> None:
    res = client.post(
        "/recall",
        json={
            "query": "what are we avoiding?",
            "metadata_filter": {"status": "avoided"},
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert "exercises/squat" in body["slice"]["seeds"]
