from __future__ import annotations

from context_engine.engine import ContextEngine, EngineResponse
from context_engine.store import GraphStore
from context_engine.types import ContextSlice, Focus, Intent, Query, QueryTuple, TraversalStrategy


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


def test_resolver_bootstraps_on_first_answer(store: GraphStore) -> None:
    """ContextEngine.answer() triggers lazy bootstrap of strategy nodes."""
    engine = ContextEngine(store)
    # Before answer(), the resolver has not bootstrapped yet.
    assert not engine.strategies._bootstrapped
    engine.answer(Query(text="Why do I avoid squats?", explicit_refs=["squat"]))
    # After answer(), strategy nodes must exist in the store.
    assert store.get_node("strategy/retrieve/causal") is not None


def test_engine_response_warnings_defaults_to_empty_list() -> None:
    slice_ = ContextSlice(nodes=[], edges=[], seeds=[], strategy=TraversalStrategy())
    resp = EngineResponse(
        text="",
        tuple_=QueryTuple(intent=Intent.RETRIEVE, focus=Focus.CAUSAL),
        slice_=slice_,
        logic_results=[],
    )
    assert resp.warnings == []
