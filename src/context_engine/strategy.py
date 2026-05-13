"""Query tuple → TraversalStrategy mapping.

Starts as a static table. Keeping it as data (not code) means the feedback
loop can learn edge-type weights per tuple dimension later without touching
the traversal code.

``StrategyResolver`` extends the static table with graph-backed learned
strategies: a strategy node per (intent, focus) pair stores edge-type
weights that the feedback loop updates over time.
"""

from __future__ import annotations

import json
from context_engine.types import Focus, Intent, Node, QueryTuple, StoreProtocol, TraversalStrategy

# Edge types prioritised by focus.
_FOCUS_EDGE_TYPES: dict[Focus, list[str]] = {
    Focus.CAUSAL: ["because_of", "supports", "contradicts", "causes", "explains"],
    Focus.PROCEDURAL: ["requires", "depends_on", "next_step", "part_of", "progresses_to"],
    Focus.TEMPORAL: ["triggered_by", "after", "before", "if_then", "scheduled_for"],
    Focus.ATTRIBUTIVE: ["is_a", "has_property", "belongs_to", "instance_of"],
    Focus.CONDITIONAL: ["if_then", "threshold", "requires", "blocked_by", "falls_back_to"],
}

# Structural edges whose targets must always accompany a rule node (rule chain
# closure). Budget can't sever these.
STRUCTURAL_EDGE_TYPES: set[str] = {"falls_back_to", "requires", "next_step", "part_of"}


def strategy_for(
    tuple_: QueryTuple,
    seeds: list[str],
    budget_override: int | None = None,
) -> TraversalStrategy:
    edge_types = list(_FOCUS_EDGE_TYPES.get(tuple_.focus, []))

    # Intent-driven tweaks.
    depth = 3
    budget = 20
    weight_floor = 0.2

    if tuple_.intent == Intent.COMPARE:
        # Need breadth across seeds, not depth.
        depth = 2
        budget = 30
    elif tuple_.intent == Intent.EVALUATE:
        # Must reach rules and thresholds plus the observations they evaluate.
        edge_types = list(
            dict.fromkeys(
                [
                    *edge_types,
                    "if_then",
                    "threshold",
                    "falls_back_to",
                    "has_observation",
                    "has_state",
                ]
            )
        )
        depth = 4
    elif tuple_.intent == Intent.ACT:
        # Current state + rules + schedule.
        edge_types = list(
            dict.fromkeys(
                [
                    *edge_types,
                    "has_state",
                    "has_observation",
                    "scheduled_for",
                    "next_step",
                ]
            )
        )
        depth = 4
    elif tuple_.intent == Intent.SCAN:
        # Scan doesn't traverse. Strategy is used only to carry budget.
        depth = 0
        budget = 100

    if budget_override is not None:
        budget = budget_override

    return TraversalStrategy(
        seeds=seeds,
        edge_types=edge_types,
        depth=depth,
        weight_floor=weight_floor,
        budget=budget,
        anti_hub=True,
        preserve_rule_chains=True,
    )


class StrategyResolver:
    """Resolve query tuples to ``TraversalStrategy`` objects via the graph store.

    On first use, seeds a strategy node for every (intent, focus) combination
    that does not yet exist.  Subsequent calls consult the graph so that the
    feedback loop can tune edge-type weights without touching this file.
    """

    def __init__(self, store: StoreProtocol) -> None:
        self.store = store
        self._bootstrapped = False

    # ── node id / fetch ─────────────────────────────────────────────────────

    def get_strategy_node_id(self, tuple_: QueryTuple) -> str:
        """Return the stable node id for *tuple_*."""
        return f"strategy/{tuple_.intent.value}/{tuple_.focus.value}"

    def get_strategy_node(self, tuple_: QueryTuple) -> Node | None:
        """Return the strategy node for *tuple_*, or ``None`` if absent."""
        return self.store.get_node(self.get_strategy_node_id(tuple_))

    # ── bootstrap ───────────────────────────────────────────────────────────

    def bootstrap(self) -> int:
        """Seed strategy nodes for every (intent, focus) pair that lacks one.

        Returns the number of nodes created.  Idempotent.
        """
        created = 0
        for intent in Intent:
            for focus in Focus:
                tuple_ = QueryTuple(intent=intent, focus=focus)
                node_id = self.get_strategy_node_id(tuple_)
                if self.store.get_node(node_id) is not None:
                    continue
                hardcoded = strategy_for(tuple_, seeds=[])
                edge_type_scores = {et: 0.5 for et in hardcoded.edge_types}
                content = {
                    "edge_types": hardcoded.edge_types,
                    "depth": hardcoded.depth,
                    "weight_floor": hardcoded.weight_floor,
                    "budget": hardcoded.budget,
                    "anti_hub": hardcoded.anti_hub,
                    "preserve_rule_chains": hardcoded.preserve_rule_chains,
                    "edge_type_scores": edge_type_scores,
                }
                metadata = {
                    "type": "strategy",
                    "intent": intent.value,
                    "focus": focus.value,
                    "success_count": 0,
                    "failure_count": 0,
                }
                self.store.upsert_node(
                    content=json.dumps(content),
                    metadata=metadata,
                    node_id=node_id,
                )
                created += 1
        self._bootstrapped = True
        return created

    # ── resolve ─────────────────────────────────────────────────────────────

    def resolve(
        self,
        tuple_: QueryTuple,
        seeds: list[str],
        budget_override: int | None = None,
    ) -> TraversalStrategy:
        """Look up a strategy node for *tuple_*, falling back to the hardcoded
        table when no matching node exists.

        Seeds are always taken from the *seeds* argument; the stored node does
        not carry them.  Edge types are ordered by descending score so
        higher-scoring types are tried first by the traverser.
        """
        if not self._bootstrapped:
            self.bootstrap()

        node = self.get_strategy_node(tuple_)
        if node is None:
            # Foreign graph with no strategy nodes — fall back gracefully.
            return strategy_for(tuple_, seeds=seeds, budget_override=budget_override)

        content = json.loads(node.content)
        scores: dict[str, float] = content.get("edge_type_scores", {})
        raw_edge_types: list[str] = content.get("edge_types", [])
        # Sort descending by score; unknown types get 0.5.
        edge_types = sorted(raw_edge_types, key=lambda et: scores.get(et, 0.5), reverse=True)

        budget = content.get("budget", 20)
        if budget_override is not None:
            budget = budget_override

        return TraversalStrategy(
            seeds=seeds,
            edge_types=edge_types,
            depth=content.get("depth", 3),
            weight_floor=content.get("weight_floor", 0.2),
            budget=budget,
            anti_hub=content.get("anti_hub", True),
            preserve_rule_chains=content.get("preserve_rule_chains", True),
        )

    # ── downgrade ────────────────────────────────────────────────────────────

    def downgrade(self, tuple_: QueryTuple) -> None:
        """Mark the strategy for *tuple_* as underpowered.

        Permanently bumps the stored strategy's budget, depth, and weight_floor
        so future queries of the same shape start wider.  No-op if the strategy
        node does not exist.

        Caps: budget ≤ 200, depth ≤ 10.
        """
        node = self.get_strategy_node(tuple_)
        if node is None:
            return

        meta = dict(node.metadata)
        content = json.loads(node.content)

        budget = min(int(content.get("budget", 20) * 1.5), 200)
        depth = min(content.get("depth", 3) + 1, 10)
        weight_floor = max(0.0, content.get("weight_floor", 0.2) - 0.05)

        content["budget"] = budget
        content["depth"] = depth
        content["weight_floor"] = weight_floor

        meta["downgrade_count"] = int(meta.get("downgrade_count", 0)) + 1

        self.store.upsert_node(
            content=json.dumps(content),
            metadata=meta,
            node_id=self.get_strategy_node_id(tuple_),
        )

    # ── feedback ─────────────────────────────────────────────────────────────

    def record_success(
        self,
        tuple_: QueryTuple,
        helpful_edge_types: list[str],
        noisy_edge_types: list[str],
    ) -> None:
        """Increment success_count and adjust edge-type scores upward for
        helpful types, downward for noisy types."""
        self._record(tuple_, success=True, helpful=helpful_edge_types, noisy=noisy_edge_types)

    def record_failure(
        self,
        tuple_: QueryTuple,
        helpful_edge_types: list[str],
        noisy_edge_types: list[str],
    ) -> None:
        """Increment failure_count and adjust edge-type scores downward for
        both noisy types and (less aggressively) helpful ones."""
        self._record(tuple_, success=False, helpful=helpful_edge_types, noisy=noisy_edge_types)

    def _record(
        self,
        tuple_: QueryTuple,
        *,
        success: bool,
        helpful: list[str],
        noisy: list[str],
    ) -> None:
        node = self.get_strategy_node(tuple_)
        if node is None:
            return

        meta = dict(node.metadata)
        if success:
            meta["success_count"] = int(meta.get("success_count", 0)) + 1
        else:
            meta["failure_count"] = int(meta.get("failure_count", 0)) + 1

        content = json.loads(node.content)
        scores: dict[str, float] = content.get("edge_type_scores", {})
        edge_types: list[str] = content.get("edge_types", [])

        if success:
            helpful_bump, helpful_noisy_bump = +0.05, -0.02
            noisy_bump = -0.02
        else:
            helpful_bump, helpful_noisy_bump = -0.02, -0.05
            noisy_bump = -0.05

        for et in helpful:
            scores[et] = max(0.0, min(1.0, scores.get(et, 0.5) + helpful_bump))
            if et not in edge_types:
                edge_types.append(et)
                scores[et] = 0.35  # conservative initial weight for newly learned type

        for et in noisy:
            scores[et] = max(0.0, min(1.0, scores.get(et, 0.5) + noisy_bump))
            _ = helpful_noisy_bump  # referenced above; suppress unused warning

        content["edge_type_scores"] = scores
        content["edge_types"] = edge_types

        self.store.upsert_node(
            content=json.dumps(content),
            metadata=meta,
            node_id=self.get_strategy_node_id(tuple_),
        )


def widen(strategy: TraversalStrategy, level: int = 1) -> TraversalStrategy:
    """Escape hatch for repeated gap-detection failures.

    Widen is always applied to the *original* strategy — callers must pass the
    desired level rather than chaining calls.

    - ``level=0`` — no-op; returns *strategy* unchanged.
    - ``level=1`` — 2× budget, +1 depth, weight_floor decreased by 0.1 (min 0.0),
      edge_types preserved.
    - ``level=2`` — 4× budget, +2 depth, weight_floor = 0.0, edge_types cleared
      (no filter).
    - ``level >= 3`` — clamped to level 2.
    """
    if level <= 0:
        return strategy
    level = min(level, 2)
    if level == 1:
        return strategy.model_copy(
            update={
                "budget": strategy.budget * 2,
                "depth": strategy.depth + 1,
                "weight_floor": max(0.0, strategy.weight_floor - 0.1),
            }
        )
    # level == 2: full widen, no filter
    return strategy.model_copy(
        update={
            "budget": strategy.budget * 4,
            "depth": strategy.depth + 2,
            "weight_floor": 0.0,
            "edge_types": [],
        }
    )
