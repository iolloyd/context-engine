from __future__ import annotations

import pytest

from context_engine.classifier import KeywordClassifier
from context_engine.types import Focus, Intent, Query


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Why do I avoid squats?", (Intent.RETRIEVE, Focus.CAUSAL)),
        ("Should I increase bench weight?", (Intent.EVALUATE, Focus.CONDITIONAL)),
        ("Sled drags vs reverse lunges?", (Intent.COMPARE, Focus.ATTRIBUTIVE)),
        ("What should I do today?", (Intent.ACT, Focus.TEMPORAL)),
        ("When should I deload?", (Intent.EVALUATE, Focus.TEMPORAL)),
        ("How do I progress overhead press?", (Intent.RETRIEVE, Focus.PROCEDURAL)),
        ("Which exercises have stalled?", (Intent.SCAN, Focus.ATTRIBUTIVE)),
    ],
)
def test_classify(text: str, expected: tuple[Intent, Focus]) -> None:
    c = KeywordClassifier()
    t = c.classify(Query(text=text))
    assert (t.intent, t.focus) == expected
