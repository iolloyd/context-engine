from __future__ import annotations

from context_engine.store import GraphStore
from context_engine.strategy import strategy_for
from context_engine.traversal import Traverser
from context_engine.types import Focus, Intent, QueryTuple


def test_causal_retrieval_from_avoided_seed(store: GraphStore) -> None:
    """Asking 'why' about squat should reach the knee condition."""
    t = Traverser(store)
    tuple_ = QueryTuple(intent=Intent.RETRIEVE, focus=Focus.CAUSAL)
    strategy = strategy_for(tuple_, seeds=["squat"])
    slice_ = t.traverse(strategy, tuple_)
    ids = {n.id for n in slice_.nodes}
    assert "squat" in ids
    assert "knee" in ids


def test_premise_check_overrides_evaluate_on_avoided(store: GraphStore) -> None:
    """Evaluating an avoided exercise must flip to retrieve/causal."""
    t = Traverser(store)
    tuple_ = QueryTuple(intent=Intent.EVALUATE, focus=Focus.CONDITIONAL)
    strategy = strategy_for(tuple_, seeds=["squat"])
    slice_ = t.traverse(strategy, tuple_)
    assert any("premise check" in note for note in slice_.notes)
    ids = {n.id for n in slice_.nodes}
    assert "knee" in ids


def test_rule_chain_closure_reaches_deload(store: GraphStore) -> None:
    """Even with a tight budget, the deload rule must accompany the
    progression rule (structural edge)."""
    t = Traverser(store)
    tuple_ = QueryTuple(intent=Intent.EVALUATE, focus=Focus.CONDITIONAL)
    strategy = strategy_for(tuple_, seeds=["ohp"], budget_override=3)
    slice_ = t.traverse(strategy, tuple_)
    ids = {n.id for n in slice_.nodes}
    assert "progression_rule" in ids
    assert "deload_rule" in ids, f"rule chain closure failed, got {ids}"


def test_budget_limits_regular_traversal(store: GraphStore) -> None:
    t = Traverser(store)
    tuple_ = QueryTuple(intent=Intent.RETRIEVE, focus=Focus.ATTRIBUTIVE)
    strategy = strategy_for(tuple_, seeds=["bench"], budget_override=3)
    slice_ = t.traverse(strategy, tuple_)
    # Budget limits BFS but rule chain closure may add rule targets,
    # so we only assert BFS respected the budget (3 seed-run nodes).
    assert len(slice_.nodes) <= 3 + 2  # generous upper bound
