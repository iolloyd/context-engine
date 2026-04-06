from __future__ import annotations

from context_engine.logic import LogicEngine
from context_engine.store import GraphStore
from context_engine.strategy import strategy_for
from context_engine.traversal import Traverser
from context_engine.types import Focus, Intent, QueryTuple


def test_progression_rule_evaluates_against_bench_observations(store: GraphStore) -> None:
    t = Traverser(store)
    tuple_ = QueryTuple(intent=Intent.EVALUATE, focus=Focus.CONDITIONAL)
    strategy = strategy_for(tuple_, seeds=["bench"])
    slice_ = t.traverse(strategy, tuple_)
    engine = LogicEngine()
    results = engine.evaluate(slice_)
    assert results, "expected at least one rule to be evaluated"
    progression = next(r for r in results if r.rule_id == "progression_rule")
    assert progression.passed, "bench weight (70kg) > 65 → should pass progression"


def test_deload_rule_reached_via_rule_chain(store: GraphStore) -> None:
    t = Traverser(store)
    tuple_ = QueryTuple(intent=Intent.EVALUATE, focus=Focus.CONDITIONAL)
    strategy = strategy_for(tuple_, seeds=["ohp"])
    slice_ = t.traverse(strategy, tuple_)
    engine = LogicEngine()
    results = engine.evaluate(slice_)
    rule_ids = {r.rule_id for r in results}
    assert "deload_rule" in rule_ids
