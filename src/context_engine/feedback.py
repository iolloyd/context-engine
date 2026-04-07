"""Feedback loop: update edge weights based on downstream signals.

Signals arrive from three sources:

  - User: explicit correction or confirmation
  - LLM: gap-detection flag ("I needed more context")
  - Logic engine: MissingFact exception during rule evaluation

Update rule::

    w ← clip(w + α · sign · s, 0, 1)

where ``sign`` is +1 for helpful edges and −1 for noisy ones. Source
reliability is encoded in ``s``: user > logic > llm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from context_engine.store import GraphStore
from context_engine.types import FeedbackSignal

if TYPE_CHECKING:
    from context_engine.strategy import StrategyResolver

_SOURCE_SCALE = {"user": 1.0, "logic_engine": 0.8, "llm": 0.4}


class FeedbackApplier:
    def __init__(
        self,
        store: GraphStore,
        alpha: float = 0.15,
        strategies: StrategyResolver | None = None,
    ) -> None:
        self.store = store
        self.alpha = alpha
        self.strategies = strategies

    def apply(self, signal: FeedbackSignal) -> dict[str, float]:
        """Apply signal and return the updated weights keyed by edge id."""
        scale = _SOURCE_SCALE.get(signal.source, 0.3)
        step = self.alpha * scale * signal.delta
        updates: dict[str, float] = {}

        for edge_id in signal.helpful_edge_ids:
            edge = self.store.get_edge(edge_id)
            if edge is None:
                continue
            new_w = _clip(edge.weight + step)
            self.store.update_edge_weight(edge_id, new_w)
            updates[edge_id] = new_w

        for edge_id in signal.noisy_edge_ids:
            edge = self.store.get_edge(edge_id)
            if edge is None:
                continue
            new_w = _clip(edge.weight - step)
            self.store.update_edge_weight(edge_id, new_w)
            updates[edge_id] = new_w

        # Bootstrap missing edges: when the LLM or logic engine flags nodes
        # that *should* have been in the slice, create weak edges from any
        # used node to them. These edges strengthen over time if they prove
        # useful.
        for used in signal.used_node_ids:
            for missing in signal.missing_node_ids:
                if used == missing:
                    continue
                edge = self.store.upsert_edge(
                    source=used,
                    target=missing,
                    edge_type="learned",
                    weight=0.35,
                    content="inferred from feedback signal",
                )
                updates[edge.id] = edge.weight

        # Propagate the signal to the strategy resolver when available.
        if self.strategies is not None and signal.query_tuple is not None:
            helpful_types = [
                e.type
                for eid in signal.helpful_edge_ids
                if (e := self.store.get_edge(eid)) is not None
            ]
            noisy_types = [
                e.type
                for eid in signal.noisy_edge_ids
                if (e := self.store.get_edge(eid)) is not None
            ]
            if len(helpful_types) >= len(noisy_types):
                self.strategies.record_success(signal.query_tuple, helpful_types, noisy_types)
            else:
                self.strategies.record_failure(signal.query_tuple, helpful_types, noisy_types)

        return updates


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))
