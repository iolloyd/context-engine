"""Tests for strategy_for (legacy) and StrategyResolver (learned)."""

from __future__ import annotations

import json

from context_engine.store import GraphStore
from context_engine.strategy import StrategyResolver, strategy_for
from context_engine.types import Focus, Intent, QueryTuple

# ── helpers ──────────────────────────────────────────────────────────────────


def _tuple(intent: Intent, focus: Focus) -> QueryTuple:
    return QueryTuple(intent=intent, focus=focus)


# ── legacy function ───────────────────────────────────────────────────────────


def test_legacy_strategy_for_evaluate_conditional() -> None:
    """strategy_for returns depth=4 and if_then edge type for evaluate/conditional."""
    t = _tuple(Intent.EVALUATE, Focus.CONDITIONAL)
    s = strategy_for(t, seeds=[])
    assert s.depth == 4
    assert "if_then" in s.edge_types


# ── bootstrap ─────────────────────────────────────────────────────────────────


def test_bootstrap_creates_all_strategy_nodes() -> None:
    """Bootstrap seeds exactly Intent×Focus nodes."""
    store = GraphStore(":memory:")
    resolver = StrategyResolver(store)
    count = resolver.bootstrap()
    expected = len(Intent) * len(Focus)
    assert count == expected

    # Spot-check node ids.
    assert store.get_node("strategy/retrieve/causal") is not None
    assert store.get_node("strategy/evaluate/conditional") is not None
    assert store.get_node("strategy/scan/temporal") is not None


def test_bootstrap_is_idempotent() -> None:
    """A second bootstrap call creates no additional nodes."""
    store = GraphStore(":memory:")
    resolver = StrategyResolver(store)
    resolver.bootstrap()
    second = resolver.bootstrap()
    assert second == 0


# ── resolve ───────────────────────────────────────────────────────────────────


def test_resolve_returns_learned_config() -> None:
    """After bootstrap, a manually mutated strategy node is reflected by resolve."""
    store = GraphStore(":memory:")
    resolver = StrategyResolver(store)
    resolver.bootstrap()

    t = _tuple(Intent.EVALUATE, Focus.CONDITIONAL)
    node_id = resolver.get_strategy_node_id(t)
    node = store.get_node(node_id)
    assert node is not None
    content = json.loads(node.content)
    content["edge_types"].append("custom_edge")
    content["edge_type_scores"]["custom_edge"] = 0.5
    store.upsert_node(content=json.dumps(content), metadata=node.metadata, node_id=node_id)

    strategy = resolver.resolve(t, seeds=[])
    assert "custom_edge" in strategy.edge_types


def test_resolve_falls_back_when_node_missing() -> None:
    """resolve() falls back to hardcoded strategy when the node is absent."""
    store = GraphStore(":memory:")
    resolver = StrategyResolver(store)
    resolver.bootstrap()

    t = _tuple(Intent.RETRIEVE, Focus.CAUSAL)
    store.delete_node(resolver.get_strategy_node_id(t))
    # Mark bootstrapped so we don't recreate it.
    resolver._bootstrapped = True

    strategy = resolver.resolve(t, seeds=[])
    hardcoded = strategy_for(t, seeds=[])
    assert strategy.depth == hardcoded.depth
    assert set(strategy.edge_types) == set(hardcoded.edge_types)


# ── record_success / record_failure ──────────────────────────────────────────


def test_record_success_increments_count_and_scores() -> None:
    """record_success bumps success_count and raises if_then score."""
    store = GraphStore(":memory:")
    resolver = StrategyResolver(store)
    resolver.bootstrap()

    t = _tuple(Intent.EVALUATE, Focus.CONDITIONAL)
    resolver.record_success(t, helpful_edge_types=["if_then"], noisy_edge_types=[])

    node = resolver.get_strategy_node(t)
    assert node is not None
    assert node.metadata["success_count"] == 1
    content = json.loads(node.content)
    assert content["edge_type_scores"]["if_then"] > 0.5


def test_record_failure_increments_count_and_penalises() -> None:
    """record_failure bumps failure_count and lowers the noisy edge score."""
    store = GraphStore(":memory:")
    resolver = StrategyResolver(store)
    resolver.bootstrap()

    t = _tuple(Intent.EVALUATE, Focus.CONDITIONAL)
    resolver.record_failure(t, helpful_edge_types=[], noisy_edge_types=["if_then"])

    node = resolver.get_strategy_node(t)
    assert node is not None
    assert node.metadata["failure_count"] == 1
    content = json.loads(node.content)
    assert content["edge_type_scores"]["if_then"] < 0.5


# ── score ordering ────────────────────────────────────────────────────────────


def test_resolve_orders_edge_types_by_score() -> None:
    """resolve() returns edge_types sorted descending by score."""
    store = GraphStore(":memory:")
    resolver = StrategyResolver(store)
    resolver.bootstrap()

    t = _tuple(Intent.EVALUATE, Focus.CONDITIONAL)
    node_id = resolver.get_strategy_node_id(t)
    node = store.get_node(node_id)
    assert node is not None
    content = json.loads(node.content)

    # Ensure both edge types are present.
    for et in ("if_then", "threshold"):
        if et not in content["edge_types"]:
            content["edge_types"].append(et)
    content["edge_type_scores"]["if_then"] = 0.9
    content["edge_type_scores"]["threshold"] = 0.3
    store.upsert_node(content=json.dumps(content), metadata=node.metadata, node_id=node_id)

    strategy = resolver.resolve(t, seeds=[])
    types = strategy.edge_types
    assert types.index("if_then") < types.index("threshold")
