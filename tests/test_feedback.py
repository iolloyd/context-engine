from __future__ import annotations

from context_engine.feedback import FeedbackApplier
from context_engine.store import GraphStore
from context_engine.types import FeedbackSignal


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
