"""Query classification: text → (intent, focus) tuple.

Phase 2 uses a deterministic keyword classifier so the traversal engine can be
tested without an LLM in the loop. An LLM-backed classifier slots in behind
the same `Classifier` protocol later.
"""

from __future__ import annotations

import re
from typing import Protocol

from context_engine.types import Focus, Intent, Query, QueryTuple


class Classifier(Protocol):
    def classify(self, query: Query) -> QueryTuple: ...


# ── intent rules ────────────────────────────────────────────────────────────

_INTENT_PATTERNS: list[tuple[Intent, list[str]]] = [
    (Intent.SCAN, [r"\bwhich\b.*\b(all|any|stalled|stuck|failed|avoided)\b", r"\blist all\b"]),
    (Intent.COMPARE, [r"\bvs\b", r"\bversus\b", r"\bcompare\b", r"\bdifference between\b"]),
    # ACT must be checked before EVALUATE because "what should i do" contains "should i".
    (
        Intent.ACT,
        [r"\bwhat should i do\b", r"\bwhat do i do\b", r"\bplan for\b", r"\bnext step\b"],
    ),
    (
        Intent.EVALUATE,
        [
            r"\bshould i\b",
            r"\bcan i\b",
            r"\bis it safe\b",
            r"\bam i ready\b",
            r"\bready to\b",
        ],
    ),
]

# ── focus rules ────────────────────────────────────────────────────────────

_FOCUS_PATTERNS: list[tuple[Focus, list[str]]] = [
    (Focus.CAUSAL, [r"\bwhy\b", r"\bbecause\b", r"\breason\b", r"\bcause\b"]),
    (
        Focus.PROCEDURAL,
        [r"\bhow do i\b", r"\bhow to\b", r"\bsteps\b", r"\bprogress\b", r"\bprogram\b"],
    ),
    (
        Focus.TEMPORAL,
        [r"\bwhen\b", r"\bhow often\b", r"\bschedule\b", r"\bdeload\b", r"\btoday\b"],
    ),
    (Focus.CONDITIONAL, [r"\bif\b", r"\bwhether\b", r"\bready\b", r"\bsafe\b", r"\bincrease\b"]),
    (Focus.ATTRIBUTIVE, [r"\bwhat is\b", r"\btell me about\b", r"\bdescribe\b"]),
]


class KeywordClassifier:
    """Regex-based classifier. Deterministic, fast, good enough for tests."""

    def classify(self, query: Query) -> QueryTuple:
        text = query.text.lower()
        intent = self._match(text, _INTENT_PATTERNS, default=Intent.RETRIEVE)
        focus = self._match(text, _FOCUS_PATTERNS, default=Focus.ATTRIBUTIVE)
        return QueryTuple(intent=intent, focus=focus)

    @staticmethod
    def _match(text: str, rules: list[tuple[object, list[str]]], default: object) -> object:
        for label, patterns in rules:
            for p in patterns:
                if re.search(p, text):
                    return label
        return default
