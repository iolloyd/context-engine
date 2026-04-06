from __future__ import annotations

from context_engine.engine import ContextEngine
from context_engine.store import GraphStore
from context_engine.types import Query


def test_engine_answers_causal_query(store: GraphStore) -> None:
    engine = ContextEngine(store)
    resp = engine.answer(
        Query(text="Why do I avoid squats?", explicit_refs=["squat"])
    )
    assert "knee" in resp.text.lower() or any(
        n.id == "knee" for n in resp.slice_.nodes
    )


def test_engine_evaluates_progression(store: GraphStore) -> None:
    engine = ContextEngine(store)
    resp = engine.answer(
        Query(text="Should I increase bench weight?", explicit_refs=["bench"])
    )
    assert resp.logic_results, "evaluate intent must produce logic results"
    passed = any(r.passed for r in resp.logic_results)
    assert passed
