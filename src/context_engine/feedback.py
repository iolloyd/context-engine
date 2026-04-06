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

from context_engine.store import GraphStore
from context_engine.types import FeedbackSignal

_SOURCE_SCALE = {"user": 1.0, "logic_engine": 0.8, "llm": 0.4}


class FeedbackApplier:
    def __init__(self, store: GraphStore, alpha: float = 0.15) -> None:
        self.store = store
        self.alpha = alpha

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

        return updates


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))
