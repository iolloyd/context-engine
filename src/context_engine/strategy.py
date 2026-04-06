"""Query tuple → TraversalStrategy mapping.

Starts as a static table. Keeping it as data (not code) means the feedback
loop can learn edge-type weights per tuple dimension later without touching
the traversal code.
"""

from __future__ import annotations

from context_engine.types import Focus, Intent, QueryTuple, TraversalStrategy

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


def widen(strategy: TraversalStrategy) -> TraversalStrategy:
    """Escape hatch for repeated gap-detection failures."""
    return strategy.model_copy(
        update={
            "budget": strategy.budget * 2,
            "depth": strategy.depth + 1,
            "weight_floor": max(0.0, strategy.weight_floor - 0.1),
            "edge_types": [],  # remove the filter entirely
        }
    )
