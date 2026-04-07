from __future__ import annotations

from context_engine.feedback import FeedbackApplier
from context_engine.store import GraphStore
from context_engine.strategy import StrategyResolver
from context_engine.types import FeedbackSignal, Focus, Intent, QueryTuple


def test_helpful_edge_weight_increases(store: GraphStore) -> None:
    applier = FeedbackApplier(store, alpha=0.2)
    edge = store.out_edges("progression_rule", edge_types=["falls_back_to"])[0]
    before = edge.weight
    signal = FeedbackSignal(
        query_text="ohp deload",
        used_node_ids=["ohp"],
        helpful_edge_ids=[edge.id],
        source="logic_engine",
        delta=1.0,
    )
    applier.apply(signal)
    after = store.get_edge(edge.id)
    assert after is not None
    assert after.weight > before


def test_strategy_mutation_via_feedback_signal(store: GraphStore) -> None:
    """FeedbackApplier with a StrategyResolver updates success_count on the strategy node."""
    resolver = StrategyResolver(store)
    resolver.bootstrap()
    applier = FeedbackApplier(store, strategies=resolver)

    edge = store.out_edges("squat", edge_types=["because_of"])[0]
    t = QueryTuple(intent=Intent.RETRIEVE, focus=Focus.CAUSAL)
    signal = FeedbackSignal(
        query_text="why avoid squat",
        used_node_ids=["squat", "knee"],
        helpful_edge_ids=[edge.id],
        source="user",
        delta=1.0,
        query_tuple=t,
    )
    applier.apply(signal)

    node = resolver.get_strategy_node(t)
    assert node is not None
    assert node.metadata["success_count"] >= 1


def test_missing_nodes_create_learned_edges(store: GraphStore) -> None:
    applier = FeedbackApplier(store)
    signal = FeedbackSignal(
        query_text="ohp progress",
        used_node_ids=["ohp"],
        missing_node_ids=["deload_rule"],
        source="llm",
    )
    updates = applier.apply(signal)
    assert any(
        (e := store.get_edge(eid)) is not None and e.type == "learned"
        for eid in updates
    )
