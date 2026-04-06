"""Logic engine: deterministic rule evaluation over a context slice.

Phase 6 (Python predicates). A rule is a node with ``type=rule`` whose
``content`` encodes a predicate the engine knows how to evaluate. Predicates
are registered by name — domain-agnostic, user-extensible.

Example rule node content (YAML-ish, but just JSON-compatible dict here)::

    {
        "predicate": "threshold",
        "args": {"metric": "rpe", "op": "<", "value": 8},
        "on_pass": "proceed",
        "on_fail": "deload"
    }

A Prolog bridge can replace this module later without the traversal or
feedback loop needing to change.
"""

from __future__ import annotations

import json
import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from context_engine.types import ContextSlice, Node

Predicate = Callable[[dict[str, Any], ContextSlice], bool]

_OPS: dict[str, Callable[[Any, Any], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    ">": operator.gt,
}


@dataclass
class LogicResult:
    rule_id: str
    passed: bool
    explanation: str
    missing: list[str]  # ids of nodes the engine needed but didn't find


class LogicEngine:
    def __init__(self) -> None:
        self._predicates: dict[str, Predicate] = {}
        self.register("threshold", self._threshold)
        self.register("status_is", self._status_is)

    def register(self, name: str, predicate: Predicate) -> None:
        self._predicates[name] = predicate

    # ── evaluation ──────────────────────────────────────────────────────────

    def evaluate(self, slice_: ContextSlice) -> list[LogicResult]:
        results: list[LogicResult] = []
        for node in slice_.nodes:
            if node.type != "rule":
                continue
            spec = self._parse_rule(node)
            if spec is None:
                continue
            pred_name = str(spec.get("predicate", ""))
            pred = self._predicates.get(pred_name)
            if pred is None:
                results.append(
                    LogicResult(
                        rule_id=node.id,
                        passed=False,
                        explanation=f"unknown predicate {pred_name!r}",
                        missing=[],
                    )
                )
                continue
            try:
                passed = pred(spec.get("args", {}), slice_)
                label = spec.get("on_pass" if passed else "on_fail", "")
                results.append(
                    LogicResult(
                        rule_id=node.id,
                        passed=passed,
                        explanation=f"{pred_name} {'passed' if passed else 'failed'} → {label}",
                        missing=[],
                    )
                )
            except MissingFact as exc:
                results.append(
                    LogicResult(
                        rule_id=node.id,
                        passed=False,
                        explanation=str(exc),
                        missing=exc.missing_ids,
                    )
                )
        return results

    # ── built-in predicates ─────────────────────────────────────────────────

    @staticmethod
    def _threshold(args: dict[str, Any], slice_: ContextSlice) -> bool:
        metric = args.get("metric")
        op = _OPS.get(str(args.get("op", "")))
        value = args.get("value")
        if op is None or metric is None:
            raise MissingFact(f"threshold rule missing metric/op: {args}", [])

        candidates = [
            n for n in slice_.nodes if n.metadata.get("metric") == metric
        ]
        if not candidates:
            raise MissingFact(f"no observations for metric {metric!r}", [])

        latest = max(candidates, key=lambda n: n.metadata.get("ordinal", 0))
        observed = latest.metadata.get("value")
        return bool(op(observed, value))

    @staticmethod
    def _status_is(args: dict[str, Any], slice_: ContextSlice) -> bool:
        target_id = args.get("node_id")
        expected = args.get("status")
        for n in slice_.nodes:
            if n.id == target_id:
                return n.status == expected
        raise MissingFact(f"node {target_id} not in slice", [str(target_id)])

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_rule(node: Node) -> dict[str, Any] | None:
        content = node.content.strip()
        if not content:
            return None
        try:
            return dict(json.loads(content))
        except json.JSONDecodeError:
            return None


class MissingFact(Exception):
    def __init__(self, message: str, missing_ids: list[str]) -> None:
        super().__init__(message)
        self.missing_ids = missing_ids
