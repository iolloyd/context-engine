from __future__ import annotations

from context_engine.scan import Scanner, stalled_analyser
from context_engine.store import GraphStore


def test_scan_finds_stalled_exercise(store: GraphStore) -> None:
    scanner = Scanner(store)
    slice_ = scanner.scan(
        entity_filter={"type": "exercise"},
        observation_edge_type="has_observation",
        analyser=stalled_analyser(threshold=0.0, min_observations=3),
    )
    ids = {n.id for n in slice_.nodes}
    assert "ohp" in ids, "OHP should be flagged as stalled"
    assert "bench" not in ids, "bench is progressing, should not be flagged"
