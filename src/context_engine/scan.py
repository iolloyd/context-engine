"""Scan intent: aggregate analysis across many entities.

Scan queries (e.g. "which exercises have stalled?") don't fit BFS traversal.
The scanner:

  1. Filters nodes by metadata (entity class)
  2. For each entity, runs a lightweight analyser
  3. Returns only entities where the analyser flags something interesting
     (plus the observation nodes the analyser cited)

Analysers are plug-in callables so the engine stays domain-agnostic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from context_engine.store import GraphStore
from context_engine.types import ContextSlice, Node, TraversalStrategy


@dataclass
class ScanResult:
    entity: Node
    cited_nodes: list[Node]
    note: str


Analyser = Callable[[Node, list[Node]], ScanResult | None]


class Scanner:
    def __init__(self, store: GraphStore) -> None:
        self.store = store

    def scan(
        self,
        entity_filter: dict[str, object],
        observation_edge_type: str,
        analyser: Analyser,
        budget: int = 100,
    ) -> ContextSlice:
        entities = self.store.filter_nodes(entity_filter)
        hits: list[ScanResult] = []

        for entity in entities:
            observations = [
                self.store.get_node(e.target)
                for e in self.store.out_edges(entity.id, edge_types=[observation_edge_type])
            ]
            obs_nodes = [n for n in observations if n is not None]
            result = analyser(entity, obs_nodes)
            if result is not None:
                hits.append(result)

        hits = hits[:budget]
        nodes: list[Node] = []
        seen: set[str] = set()
        for h in hits:
            if h.entity.id not in seen:
                nodes.append(h.entity)
                seen.add(h.entity.id)
            for n in h.cited_nodes:
                if n.id not in seen:
                    nodes.append(n)
                    seen.add(n.id)

        return ContextSlice(
            nodes=nodes,
            edges=[],
            seeds=[],
            strategy=TraversalStrategy(budget=budget, depth=0),
            notes=[f"scan: {len(hits)} of {len(entities)} entities flagged"],
        )


# ── example analyser ───────────────────────────────────────────────────────


def stalled_analyser(
    threshold: float = 0.0,
    min_observations: int = 3,
) -> Analyser:
    """Flag entities whose latest `value` has not increased over `min_observations`.

    Observation nodes must have `metadata.value` (numeric) and
    `metadata.ordinal` (sortable). Domain-agnostic.
    """

    def _an(entity: Node, observations: list[Node]) -> ScanResult | None:
        ordered = sorted(
            observations,
            key=lambda n: n.metadata.get("ordinal", 0),
        )
        if len(ordered) < min_observations:
            return None
        recent = ordered[-min_observations:]
        values: list[float] = []
        for n in recent:
            try:
                values.append(float(n.metadata.get("value", 0.0)))
            except (TypeError, ValueError):
                return None
        delta = values[-1] - values[0]
        if delta <= threshold:
            return ScanResult(
                entity=entity,
                cited_nodes=recent,
                note=(
                    f"{entity.content}: {values[0]} → {values[-1]} "
                    f"over {min_observations} observations (Δ={delta:+.2f})"
                ),
            )
        return None

    return _an
