"""Tests for strategy_for (legacy) and StrategyResolver (learned)."""

from __future__ import annotations

import json

import pytest

from context_engine.store import GraphStore
from context_engine.strategy import StrategyResolver, strategy_for, widen
from context_engine.types import Focus, Intent, QueryTuple, TraversalStrategy

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


# ── widen levels ──────────────────────────────────────────────────────────────


def _base_strategy() -> TraversalStrategy:
    return TraversalStrategy(
        budget=20,
        depth=3,
        weight_floor=0.3,
        edge_types=["if_then", "threshold"],
    )


def test_widen_level_0_is_noop() -> None:
    """widen(s, level=0) returns an identical strategy."""
    s = _base_strategy()
    result = widen(s, level=0)
    assert result.budget == s.budget
    assert result.depth == s.depth
    assert result.weight_floor == s.weight_floor
    assert result.edge_types == s.edge_types


def test_widen_level_1_doubles_budget_relaxes_floor() -> None:
    """widen(s, level=1) doubles budget, increments depth, decreases weight_floor."""
    s = _base_strategy()
    result = widen(s, level=1)
    assert result.budget == 40
    assert result.depth == 4
    assert result.weight_floor == pytest.approx(0.2)
    assert result.edge_types == s.edge_types  # preserved


def test_widen_level_2_clears_filter_full_widen() -> None:
    """widen(s, level=2) applies 4× budget, +2 depth, weight_floor=0.0, no filter."""
    s = _base_strategy()
    result = widen(s, level=2)
    assert result.budget == 80
    assert result.depth == 5
    assert result.weight_floor == 0.0
    assert result.edge_types == []


def test_widen_level_clamps_at_2() -> None:
    """widen(s, level=5) behaves identically to widen(s, level=2)."""
    s = _base_strategy()
    assert widen(s, level=5).budget == widen(s, level=2).budget
    assert widen(s, level=5).depth == widen(s, level=2).depth
    assert widen(s, level=5).edge_types == []


# ── StrategyResolver.downgrade ────────────────────────────────────────────────


def test_downgrade_bumps_stored_params() -> None:
    """downgrade() increases budget/depth and decreases weight_floor; sets downgrade_count."""
    store = GraphStore(":memory:")
    resolver = StrategyResolver(store)
    resolver.bootstrap()

    t = _tuple(Intent.EVALUATE, Focus.CONDITIONAL)
    node_before = resolver.get_strategy_node(t)
    assert node_before is not None
    content_before = json.loads(node_before.content)
    budget_before = content_before["budget"]
    depth_before = content_before["depth"]
    floor_before = content_before["weight_floor"]

    resolver.downgrade(t)

    node_after = resolver.get_strategy_node(t)
    assert node_after is not None
    content_after = json.loads(node_after.content)

    assert content_after["budget"] > budget_before
    assert content_after["depth"] > depth_before
    assert content_after["weight_floor"] < floor_before
    assert node_after.metadata["downgrade_count"] == 1


def test_downgrade_caps_budget_and_depth() -> None:
    """Repeated downgrade() calls never exceed budget=200 or depth=10."""
    store = GraphStore(":memory:")
    resolver = StrategyResolver(store)
    resolver.bootstrap()

    t = _tuple(Intent.EVALUATE, Focus.CONDITIONAL)
    for _ in range(10):
        resolver.downgrade(t)

    node = resolver.get_strategy_node(t)
    assert node is not None
    content = json.loads(node.content)
    assert content["budget"] <= 200
    assert content["depth"] <= 10
