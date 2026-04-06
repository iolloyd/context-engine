"""Shared fixtures: a small synthetic graph modelled after the design doc's
workout domain (no domain-specific logic in the engine itself)."""

from __future__ import annotations

import json

import pytest

from context_engine.store import GraphStore


@pytest.fixture
def store() -> GraphStore:
    s = GraphStore(":memory:")
    _seed_small_graph(s)
    return s


def _seed_small_graph(s: GraphStore) -> None:
    # Entities
    s.upsert_node("bench press", node_id="bench", metadata={"type": "exercise"})
    s.upsert_node("squat", node_id="squat", metadata={"type": "exercise", "status": "avoided"})
    s.upsert_node(
        "overhead press", node_id="ohp", metadata={"type": "exercise", "status": "stalled"}
    )
    s.upsert_node("row", node_id="row", metadata={"type": "exercise"})
    s.upsert_node(
        "close-grip bench", node_id="cgb", metadata={"type": "exercise"}
    )

    # Observations (time-series)
    for i, (weight, ordinal) in enumerate(
        [(35, 1), (37.5, 2), (37.5, 3), (37.5, 4), (37.5, 5)]
    ):
        s.upsert_node(
            f"OHP session: {weight}kg",
            node_id=f"ohp_obs_{i}",
            metadata={
                "type": "observation",
                "metric": "ohp_weight",
                "value": weight,
                "ordinal": ordinal,
                "recency": i / 5,
            },
        )
        s.upsert_edge("ohp", f"ohp_obs_{i}", edge_type="has_observation", weight=0.6)

    for i, (weight, ordinal) in enumerate(
        [(60, 1), (62.5, 2), (65, 3), (67.5, 4), (70, 5)]
    ):
        s.upsert_node(
            f"bench session: {weight}kg",
            node_id=f"bench_obs_{i}",
            metadata={
                "type": "observation",
                "metric": "bench_weight",
                "value": weight,
                "ordinal": ordinal,
                "recency": i / 5,
            },
        )
        s.upsert_edge("bench", f"bench_obs_{i}", edge_type="has_observation", weight=0.6)

    # Rules
    progression_rule = {
        "predicate": "threshold",
        "args": {"metric": "bench_weight", "op": ">", "value": 65},
        "on_pass": "increase weight by 2.5kg",
        "on_fail": "maintain current weight",
    }
    s.upsert_node(
        json.dumps(progression_rule),
        node_id="progression_rule",
        metadata={"type": "rule", "name": "progression"},
    )
    deload_rule = {
        "predicate": "threshold",
        "args": {"metric": "ohp_weight", "op": "<", "value": 40},
        "on_pass": "deload to 80% for one session",
        "on_fail": "continue",
    }
    s.upsert_node(
        json.dumps(deload_rule),
        node_id="deload_rule",
        metadata={"type": "rule", "name": "deload"},
    )

    # Rule chain — progression falls back to deload
    s.upsert_edge(
        "progression_rule",
        "deload_rule",
        edge_type="falls_back_to",
        weight=0.5,
        content="when progression fails, try deload",
    )

    # Exercises → rules (conditional)
    for ex in ("bench", "ohp", "row", "cgb"):
        s.upsert_edge(ex, "progression_rule", edge_type="if_then", weight=0.6)

    # Causal — squat is avoided because of knee
    s.upsert_node("knee pain", node_id="knee", metadata={"type": "condition"})
    s.upsert_edge("squat", "knee", edge_type="because_of", weight=0.9)

    # Attributive
    s.upsert_edge("bench", "cgb", edge_type="is_a", weight=0.4)
    s.upsert_edge("row", "bench", edge_type="contradicts", weight=0.3)
