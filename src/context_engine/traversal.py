"""Graph traversal: seeds + strategy → context slice.

Priority-queue BFS. Each step applies:

  - edge-type filter (from focus)
  - weight floor (feedback loop tunes this)
  - anti-hub: when stepping away from a non-seed node, deprioritise edges that
    return to the same node-type as the seed (prevents rule → many exercises
    flooding the budget before rule → deload_rule is reached)
  - recency weighting: observation-type targets get a recency boost

After traversal completes, rule chain closure walks structural edges out of
every ``type=rule`` node already in the slice, adding their targets
unconditionally. Budget cannot sever rule chains.

Premise check runs before traversal: if the query intent contradicts a seed's
status (e.g. evaluate an ``avoided`` exercise), the tuple is overridden to
``(retrieve, causal)``.
"""

from __future__ import annotations

import contextlib
import heapq
from dataclasses import dataclass, field

from context_engine.store import GraphStore
from context_engine.strategy import STRUCTURAL_EDGE_TYPES, strategy_for
from context_engine.types import (
    ContextSlice,
    Edge,
    Focus,
    Intent,
    Node,
    QueryTuple,
    TraversalStrategy,
)


@dataclass(order=True)
class _PQItem:
    priority: float
    depth: int = field(compare=False)
    node_id: str = field(compare=False)
    via_edge: Edge | None = field(compare=False, default=None)


class Traverser:
    def __init__(self, store: GraphStore) -> None:
        self.store = store

    # ── public entry ────────────────────────────────────────────────────────

    def traverse(
        self,
        strategy: TraversalStrategy,
        tuple_: QueryTuple,
    ) -> ContextSlice:
        notes: list[str] = []
        new_tuple, notes_override = self._premise_check(strategy.seeds, tuple_)
        notes.extend(notes_override)
        if new_tuple != tuple_:
            # Rebuild the strategy with the overridden tuple so edge filters
            # match the new intent/focus. Preserve the original budget.
            strategy = strategy_for(
                new_tuple, seeds=strategy.seeds, budget_override=strategy.budget
            )
            tuple_ = new_tuple

        if tuple_.intent == Intent.SCAN:
            return ContextSlice(
                nodes=[],
                edges=[],
                seeds=strategy.seeds,
                strategy=strategy,
                notes=["scan intent — use Scanner, not Traverser"],
            )

        collected_node_ids: set[str] = set()
        collected_edges: dict[str, Edge] = {}

        # Multi-seed budget split for compare intent.
        per_seed_budget = (
            max(1, strategy.budget // max(1, len(strategy.seeds)))
            if tuple_.intent == Intent.COMPARE
            else strategy.budget
        )

        for seed in strategy.seeds:
            self._bfs_from(
                seed=seed,
                strategy=strategy,
                tuple_=tuple_,
                collected_node_ids=collected_node_ids,
                collected_edges=collected_edges,
                budget=per_seed_budget,
            )

        if strategy.preserve_rule_chains:
            added = self._close_rule_chains(collected_node_ids, collected_edges)
            if added:
                notes.append(f"rule chain closure added {added} node(s)")

        nodes = self.store.get_nodes(collected_node_ids)
        edges = list(collected_edges.values())
        return ContextSlice(
            nodes=nodes,
            edges=edges,
            seeds=strategy.seeds,
            strategy=strategy,
            notes=notes,
        )

    # ── premise check ───────────────────────────────────────────────────────

    def _premise_check(
        self, seeds: list[str], tuple_: QueryTuple
    ) -> tuple[QueryTuple, list[str]]:
        notes: list[str] = []
        if tuple_.intent not in (Intent.EVALUATE, Intent.ACT):
            return tuple_, notes
        for sid in seeds:
            node = self.store.get_node(sid)
            if node is None:
                continue
            if node.status in {"avoided", "blocked", "deprecated"}:
                notes.append(
                    f"premise check: seed {sid} status={node.status} → "
                    f"overriding tuple to (retrieve, causal)"
                )
                return QueryTuple(intent=Intent.RETRIEVE, focus=Focus.CAUSAL), notes
        return tuple_, notes

    # ── bfs ────────────────────────────────────────────────────────────────

    def _bfs_from(
        self,
        seed: str,
        strategy: TraversalStrategy,
        tuple_: QueryTuple,
        collected_node_ids: set[str],
        collected_edges: dict[str, Edge],
        budget: int,
    ) -> None:
        seed_node = self.store.get_node(seed)
        if seed_node is None:
            return
        seed_type = seed_node.type

        pq: list[_PQItem] = []
        heapq.heappush(pq, _PQItem(priority=0.0, depth=0, node_id=seed))
        visited: set[str] = set()
        collected_this_run = 0

        while pq and collected_this_run < budget:
            item = heapq.heappop(pq)
            if item.node_id in visited:
                continue
            visited.add(item.node_id)

            # Add node to the slice
            if item.node_id not in collected_node_ids:
                collected_node_ids.add(item.node_id)
                collected_this_run += 1
                if item.via_edge is not None:
                    collected_edges[item.via_edge.id] = item.via_edge

            if item.depth >= strategy.depth:
                continue

            # Expand
            edges = self.store.out_edges(
                item.node_id,
                edge_types=strategy.edge_types or None,
                weight_floor=strategy.weight_floor,
            )
            for edge in edges:
                if edge.target in visited:
                    continue

                target = self.store.get_node(edge.target)
                if target is None:
                    continue

                prio = self._priority(
                    edge=edge,
                    target=target,
                    depth=item.depth + 1,
                    seed_type=seed_type,
                    strategy=strategy,
                    tuple_=tuple_,
                    from_seed=item.node_id == seed,
                )
                heapq.heappush(
                    pq,
                    _PQItem(
                        priority=prio,
                        depth=item.depth + 1,
                        node_id=edge.target,
                        via_edge=edge,
                    ),
                )

    # ── priority function ──────────────────────────────────────────────────

    def _priority(
        self,
        edge: Edge,
        target: Node,
        depth: int,
        seed_type: str | None,
        strategy: TraversalStrategy,
        tuple_: QueryTuple,
        from_seed: bool,
    ) -> float:
        """Lower is better (min-heap)."""
        # Base: depth penalty + inverse weight.
        p = float(depth) + (1.0 - edge.weight)

        # Anti-hub: when *not* stepping from the seed, penalise edges that
        # return to the seed's own node type. Prevents rule → many exercises
        # flooding before rule → deload_rule.
        if strategy.anti_hub and not from_seed and seed_type and target.type == seed_type:
            p += 0.75

        # Recency boost for observations on temporal/act/evaluate intents.
        if target.type == "observation" and tuple_.intent in (
            Intent.EVALUATE,
            Intent.ACT,
            Intent.RETRIEVE,
        ):
            ts = target.metadata.get("recency", 0.0)
            with contextlib.suppress(TypeError, ValueError):
                p -= float(ts) * 0.5

        # Structural edges are cheap — they must stay in the slice.
        if edge.type in STRUCTURAL_EDGE_TYPES:
            p -= 0.5

        # Rule-seeking boost: for evaluate/act, rule targets are the point of
        # the query. Without this, high-recency observations starve rules out
        # of the budget (the Q2 OHP deload failure mode from the design doc).
        if target.type == "rule" and tuple_.intent in (Intent.EVALUATE, Intent.ACT):
            p -= 1.0

        return p

    # ── rule chain closure ─────────────────────────────────────────────────

    def _close_rule_chains(
        self,
        collected_node_ids: set[str],
        collected_edges: dict[str, Edge],
    ) -> int:
        """Walk structural edges out of every rule node already in the slice.

        Returns the number of nodes added. Budget does not constrain this.
        """
        frontier = [
            nid
            for nid in collected_node_ids
            if (n := self.store.get_node(nid)) is not None and n.type == "rule"
        ]
        added = 0
        seen: set[str] = set(frontier)
        while frontier:
            nid = frontier.pop()
            for edge in self.store.out_edges(nid):
                if edge.type not in STRUCTURAL_EDGE_TYPES:
                    continue
                collected_edges[edge.id] = edge
                if edge.target in collected_node_ids:
                    continue
                target = self.store.get_node(edge.target)
                if target is None:
                    continue
                collected_node_ids.add(edge.target)
                added += 1
                if target.type == "rule" and edge.target not in seen:
                    frontier.append(edge.target)
                    seen.add(edge.target)
        return added
