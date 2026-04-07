"""Tests for the Prolog-backed logic engine.

Tests 1–3 mock subprocess so they pass even without SWI-Prolog installed.
Tests 4–6 require a real ``swipl`` binary and are skipped when unavailable.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from context_engine.prolog_logic import (
    PrologLogicEngine,
    _serialise_facts,
)
from context_engine.store import GraphStore
from context_engine.strategy import strategy_for
from context_engine.traversal import Traverser
from context_engine.types import (
    ContextSlice,
    Edge,
    Focus,
    Intent,
    Node,
    Query,
    QueryTuple,
    TraversalStrategy,
)

_SWIPL_AVAILABLE = PrologLogicEngine().is_available()


# ── helpers ──────────────────────────────────────────────────────────────────


def _minimal_slice(*extra_nodes: Node, edges: list[Edge] | None = None) -> ContextSlice:
    """Build a ContextSlice from *extra_nodes* with an empty strategy."""
    strategy = TraversalStrategy(seeds=[], edge_types=[], depth=3, budget=50)
    return ContextSlice(
        nodes=list(extra_nodes),
        edges=edges or [],
        seeds=[],
        strategy=strategy,
    )


def _bench_obs_node(node_id: str, value: float, ordinal: int) -> Node:
    return Node(
        id=node_id,
        content=f"bench session: {value}kg",
        metadata={
            "type": "observation",
            "metric": "bench_weight",
            "value": value,
            "ordinal": ordinal,
        },
    )


def _prolog_rule_node(node_id: str, body: str) -> Node:
    return Node(
        id=node_id,
        content=body,
        metadata={"type": "rule", "language": "prolog"},
    )


# ── test 1: fact serialisation ────────────────────────────────────────────────


def test_serialise_facts_contains_expected_terms(store: GraphStore) -> None:
    """_serialise_facts must emit node/3 and edge/4 terms from a real slice."""
    t = Traverser(store)
    tuple_ = QueryTuple(intent=Intent.EVALUATE, focus=Focus.CONDITIONAL)
    strategy = strategy_for(tuple_, seeds=["squat"])
    slice_ = t.traverse(strategy, tuple_)

    facts = _serialise_facts(slice_)

    assert "node('squat', 'exercise', 'avoided')." in facts
    assert any(line.startswith("edge(") for line in facts.splitlines()), \
        "expected at least one edge/4 fact"


# ── test 2: mocked pass ───────────────────────────────────────────────────────


def test_evaluate_mocked_pass() -> None:
    """evaluate() returns LogicResult(passed=True) when subprocess emits 'pass'."""
    rule = _prolog_rule_node("r1", "true")
    slice_ = _minimal_slice(rule)

    mock_result = MagicMock()
    mock_result.stdout = "pass\n"

    engine = PrologLogicEngine()
    with patch("subprocess.run", return_value=mock_result):
        results = engine.evaluate(slice_)

    assert len(results) == 1
    assert results[0].rule_id == "r1"
    assert results[0].passed is True
    assert results[0].explanation == "passed"
    assert results[0].missing == []


# ── test 3: mocked missing ────────────────────────────────────────────────────


def test_evaluate_mocked_missing() -> None:
    """evaluate() surfaces missing predicate when subprocess emits missing(…)."""
    rule = _prolog_rule_node("r2", "latest(X), X > 0")
    slice_ = _minimal_slice(rule)

    mock_result = MagicMock()
    mock_result.stdout = "missing(latest/2)\n"

    engine = PrologLogicEngine()
    with patch("subprocess.run", return_value=mock_result):
        results = engine.evaluate(slice_)

    assert len(results) == 1
    r = results[0]
    assert r.passed is False
    assert r.missing == ["latest/2"]
    assert "latest/2" in r.explanation


# ── test 4: real swipl — simple threshold rule ────────────────────────────────


@pytest.mark.skipif(not _SWIPL_AVAILABLE, reason="swipl not available")
def test_real_swipl_threshold_rule() -> None:
    """A rule checking bench_weight > 65 passes when the observation value is 70."""
    bench = Node(id="bench", content="bench press", metadata={"type": "exercise"})
    obs = _bench_obs_node("obs1", value=70.0, ordinal=1)
    rule = _prolog_rule_node(
        "bench_rule",
        "metadata(N, metric, bench_weight),\n"
        "    metadata(N, value, V),\n"
        "    V > 65",
    )
    bench_to_obs = Edge(
        id="e1",
        source="bench",
        target="obs1",
        metadata={"type": "has_observation", "weight": 0.6},
    )
    slice_ = _minimal_slice(bench, obs, rule, edges=[bench_to_obs])

    engine = PrologLogicEngine()
    results = engine.evaluate(slice_)

    assert len(results) == 1
    assert results[0].passed is True, f"expected pass; got: {results[0].explanation}"


# ── test 5: real swipl — helper predicate via %--- separator ─────────────────


@pytest.mark.skipif(not _SWIPL_AVAILABLE, reason="swipl not available")
def test_real_swipl_helper_predicate() -> None:
    """A rule using a helper predicate defined before %--- passes correctly."""
    bench = Node(id="bench", content="bench press", metadata={"type": "exercise"})
    obs = _bench_obs_node("obs1", value=70.0, ordinal=1)
    body = (
        "heavy(W) :- W > 65.\n"
        "%---\n"
        "metadata(N, metric, bench_weight), metadata(N, value, V), heavy(V)"
    )
    rule = _prolog_rule_node("bench_rule_helper", body)
    bench_to_obs = Edge(
        id="e1",
        source="bench",
        target="obs1",
        metadata={"type": "has_observation", "weight": 0.6},
    )
    slice_ = _minimal_slice(bench, obs, rule, edges=[bench_to_obs])

    engine = PrologLogicEngine()
    results = engine.evaluate(slice_)

    assert len(results) == 1
    assert results[0].passed is True, f"expected pass; got: {results[0].explanation}"


# ── test 6: ContextEngine swap — PrologLogicEngine as drop-in ────────────────


@pytest.mark.skipif(not _SWIPL_AVAILABLE, reason="swipl not available")
def test_context_engine_with_prolog_logic_engine(store: GraphStore) -> None:
    """ContextEngine accepts PrologLogicEngine without orchestration changes."""
    from context_engine.engine import ContextEngine

    # Add a Prolog rule and an observation node to the store fixture.
    # The bench observations already exist in the fixture (bench_obs_0 … bench_obs_4).
    body = (
        "metadata(N, metric, bench_weight),\n"
        "    metadata(N, value, V),\n"
        "    V > 65"
    )
    store.upsert_node(
        body,
        node_id="prolog_bench_rule",
        metadata={"type": "rule", "language": "prolog", "name": "prolog_progression"},
    )
    store.upsert_edge("bench", "prolog_bench_rule", edge_type="if_then", weight=0.6)

    engine = ContextEngine(store)
    engine.logic = PrologLogicEngine()

    query = Query(text="should I increase bench weight?", explicit_refs=["bench"])
    response = engine.answer(query)

    assert response.logic_results, "expected at least one logic result"
    rule_ids = {r.rule_id for r in response.logic_results}
    assert "prolog_bench_rule" in rule_ids, (
        f"prolog_bench_rule not in logic results: {rule_ids}"
    )
    prolog_result = next(r for r in response.logic_results if r.rule_id == "prolog_bench_rule")
    assert prolog_result.passed is True
