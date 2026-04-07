"""Tests for the garden health diagnostic."""

from __future__ import annotations

import math

from context_engine.garden import GardenConfig, Gardener
from context_engine.store import GraphStore


def _make_store() -> GraphStore:
    return GraphStore(":memory:")


# ── helpers ──────────────────────────────────────────────────────────────────


def _unit_vec(dim: int, index: int = 0) -> list[float]:
    """Return a unit vector with 1.0 at `index`."""
    v = [0.0] * dim
    v[index] = 1.0
    return v


def _near_unit_vec(dim: int, index: int = 0, eps: float = 1e-4) -> list[float]:
    """Return a vector very close to _unit_vec(dim, index)."""
    v = [0.0] * dim
    v[index] = 1.0
    v[(index + 1) % dim] = eps
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


# ── orphan tests ──────────────────────────────────────────────────────────────


def test_orphan_detected() -> None:
    store = _make_store()
    store.upsert_node("alpha", node_id="n1")
    store.upsert_node("beta", node_id="n2")

    gardener = Gardener(store)
    findings = gardener.inspect()
    orphan_subjects = {f.subject for f in findings if f.category == "orphan"}
    assert "n1" in orphan_subjects
    assert "n2" in orphan_subjects

    # Add an incoming edge to n1 — only n2 should remain.
    store.upsert_node("root", node_id="root")
    store.upsert_edge("root", "n1", edge_type="links_to")
    findings2 = gardener.inspect()
    orphan_subjects2 = {f.subject for f in findings2 if f.category == "orphan"}
    assert "n1" not in orphan_subjects2
    assert "n2" in orphan_subjects2


def test_orphan_skips_category_and_strategy() -> None:
    store = _make_store()
    store.upsert_node("cat node", metadata={"type": "category"}, node_id="cat1")
    store.upsert_node(
        "strategy node",
        metadata={"type": "strategy", "success_count": 0, "failure_count": 0, "downgrade_count": 0},
        node_id="strat1",
    )

    gardener = Gardener(store)
    findings = gardener.inspect()
    orphan_subjects = {f.subject for f in findings if f.category == "orphan"}
    assert "cat1" not in orphan_subjects
    assert "strat1" not in orphan_subjects


# ── duplicate tests ───────────────────────────────────────────────────────────


def test_duplicate_detected() -> None:
    store = _make_store()
    dim = store.embed_dim
    store.upsert_node("nearly identical content here", node_id="d1",
                      embedding=_unit_vec(dim, 0))
    store.upsert_node("nearly identical content here", node_id="d2",
                      embedding=_near_unit_vec(dim, 0))

    gardener = Gardener(store)
    findings = [f for f in gardener.inspect() if f.category == "duplicate"]
    assert len(findings) == 1
    assert "d1" in findings[0].subject
    assert "d2" in findings[0].subject


def test_duplicate_skipped_when_no_embeddings() -> None:
    store = _make_store()
    store.upsert_node("identical content", node_id="e1")
    store.upsert_node("identical content", node_id="e2")

    gardener = Gardener(store)
    findings = [f for f in gardener.inspect() if f.category == "duplicate"]
    assert findings == []


# ── dead weight tests ─────────────────────────────────────────────────────────


def test_dead_weight_detected() -> None:
    store = _make_store()
    store.upsert_node("src", node_id="s")
    store.upsert_node("tgt", node_id="t")
    store.upsert_edge("s", "t", edge_type="links_to", weight=0.05)

    gardener = Gardener(store)
    findings = [f for f in gardener.inspect() if f.category == "dead_weight"]
    assert len(findings) == 1
    assert "s -> t:links_to" in findings[0].subject


def test_dead_weight_skips_derived() -> None:
    store = _make_store()
    store.upsert_node("src", node_id="s")
    store.upsert_node("tgt", node_id="t")
    store.upsert_edge("s", "t", edge_type="part_of", weight=0.05, metadata={"derived": True})

    gardener = Gardener(store)
    findings = [f for f in gardener.inspect() if f.category == "dead_weight"]
    assert findings == []


# ── strategy drift tests ──────────────────────────────────────────────────────


def test_strategy_drift_detected() -> None:
    store = _make_store()
    store.upsert_node(
        "strat",
        metadata={"type": "strategy", "success_count": 2, "failure_count": 8},
        node_id="drift1",
    )

    gardener = Gardener(store)
    findings = [f for f in gardener.inspect() if f.category == "drift"]
    assert len(findings) == 1
    assert findings[0].subject == "drift1"


def test_strategy_drift_skipped_when_low_samples() -> None:
    store = _make_store()
    store.upsert_node(
        "strat",
        metadata={"type": "strategy", "success_count": 0, "failure_count": 2},
        node_id="low1",
    )

    gardener = Gardener(store)
    findings = [f for f in gardener.inspect() if f.category == "drift"]
    assert findings == []


# ── uncovered tuple tests ─────────────────────────────────────────────────────


def test_uncovered_detected() -> None:
    store = _make_store()
    store.upsert_node(
        "strat",
        metadata={"type": "strategy", "success_count": 0, "failure_count": 0, "downgrade_count": 0},
        node_id="unc1",
    )

    gardener = Gardener(store)
    findings = [f for f in gardener.inspect() if f.category == "uncovered"]
    assert len(findings) == 1
    assert findings[0].subject == "unc1"


# ── recent activity tests ─────────────────────────────────────────────────────


def test_recent_activity_reported() -> None:
    import time

    store = _make_store()
    # Insert nodes with slight delay so updated_at differs.
    store.upsert_node("first", node_id="r1")
    time.sleep(0.01)
    store.upsert_node("second", node_id="r2")
    time.sleep(0.01)
    store.upsert_node("third", node_id="r3")

    config = GardenConfig(recent_activity_count=2)
    gardener = Gardener(store, config)
    findings = [f for f in gardener.inspect() if f.category == "recent"]
    # Only top 2 by updated_at descending.
    assert len(findings) == 2
    assert findings[0].subject == "r3"
    assert findings[1].subject == "r2"


# ── empty graph ───────────────────────────────────────────────────────────────


def test_empty_graph_no_crash() -> None:
    store = _make_store()
    gardener = Gardener(store)
    findings = gardener.inspect()
    assert findings == []
