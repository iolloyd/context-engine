from __future__ import annotations

from context_engine.engine import ContextEngine, EngineResponse
from context_engine.logic import LogicResult
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


# ── progressive widen ─────────────────────────────────────────────────────────


class _AlwaysMissingSynthesiser:
    """Synthesiser that always reports missing knowledge."""

    def synthesise(
        self,
        query: Query,
        tuple_: QueryTuple,
        slice_: ContextSlice,
        logic_results: list[LogicResult],
    ) -> EngineResponse:
        return EngineResponse(
            text="I need more context.",
            tuple_=tuple_,
            slice_=slice_,
            logic_results=logic_results,
            used_node_ids=[],
            missing=["always"],
            confidence=0.1,
        )


class _AlwaysOkSynthesiser:
    """Synthesiser that never reports missing knowledge."""

    def synthesise(
        self,
        query: Query,
        tuple_: QueryTuple,
        slice_: ContextSlice,
        logic_results: list[LogicResult],
    ) -> EngineResponse:
        return EngineResponse(
            text="All good.",
            tuple_=tuple_,
            slice_=slice_,
            logic_results=logic_results,
            used_node_ids=[],
            missing=[],
            confidence=1.0,
        )


def test_three_tier_widen_progression(store: GraphStore) -> None:
    """Engine escalates through all widen levels and sets stats + flags correctly."""
    engine = ContextEngine(store, synthesiser=_AlwaysMissingSynthesiser())  # type: ignore[arg-type]

    resp = engine.answer(Query(text="Why avoid squats?", explicit_refs=["squat"]))

    assert resp.needed_widen is True
    assert resp.gap_flag is True
    assert engine.stats["queries_total"] == 1
    assert engine.stats["widens_level_1"] == 1
    assert engine.stats["widens_level_2"] == 1


def test_no_widen_on_successful_answer(store: GraphStore) -> None:
    """When synthesiser succeeds immediately, no widen counters are incremented."""
    engine = ContextEngine(store, synthesiser=_AlwaysOkSynthesiser())  # type: ignore[arg-type]

    resp = engine.answer(Query(text="Why avoid squats?", explicit_refs=["squat"]))

    assert resp.needed_widen is False
    assert resp.gap_flag is False
    assert engine.stats["widens_level_1"] == 0
    assert engine.stats["widens_level_2"] == 0


def test_strategy_downgrade_fires_after_threshold(store: GraphStore) -> None:
    """After threshold repeated max-widen failures, downgrade fires and mutates node."""
    engine = ContextEngine(
        store,
        synthesiser=_AlwaysMissingSynthesiser(),  # type: ignore[arg-type]
        downgrade_threshold=2,
    )
    q = Query(text="Why avoid squats?", explicit_refs=["squat"])

    engine.answer(q)
    assert engine.stats["strategy_downgrades"] == 0

    engine.answer(q)
    assert engine.stats["strategy_downgrades"] == 1

    # The strategy node should have been mutated.
    tuple_ = engine.classifier.classify(q)
    node = engine.strategies.get_strategy_node(tuple_)
    assert node is not None
    assert node.metadata.get("downgrade_count", 0) == 1


def test_engine_uses_default_embedder_when_none_passed() -> None:
    from context_engine.embedding import HashEmbedder, SentenceTransformerEmbedder

    store = GraphStore(":memory:")
    engine = ContextEngine(store)
    assert engine._embedder is not None
    assert isinstance(engine._embedder, (SentenceTransformerEmbedder, HashEmbedder))


def test_engine_composes_global_graph_for_seeds() -> None:
    """Global-store nodes are visible in query results when the project store has none."""
    from context_engine.embedding import HashEmbedder

    project = GraphStore(":memory:")
    global_ = GraphStore(":memory:")

    embedder = HashEmbedder(dim=project.embed_dim)

    # Global store has a node about pathlib; project store is empty.
    global_.upsert_node("normalize paths with pathlib", node_id="conventions/pathlib")
    global_._set_embedding("conventions/pathlib", embedder.embed("normalize paths with pathlib"))

    engine = ContextEngine(project, embedder=embedder, global_store=global_)
    resp = engine.answer(Query(text="how to normalize paths"))

    node_ids = {n.id for n in resp.slice_.nodes}
    assert "conventions/pathlib" in node_ids, (
        f"expected global node 'conventions/pathlib' in slice, got {node_ids}"
    )
