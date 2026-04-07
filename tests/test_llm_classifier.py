"""Tests for LLMClassifier.

Covers happy-path parsing, malformed response fallback, exception fallback,
and an integration smoke test (skipped without ANTHROPIC_API_KEY).
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from context_engine.classifier import KeywordClassifier
from context_engine.llm_classifier import LLMClassifier
from context_engine.types import Focus, Intent, Query, QueryTuple

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_tool_use_block(input_dict: dict) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.input = input_dict
    return block


def _make_response(input_dict: dict, input_tokens: int = 10, output_tokens: int = 5) -> MagicMock:
    block = _make_tool_use_block(input_dict)
    response = MagicMock()
    response.content = [block]
    response.usage.input_tokens = input_tokens
    response.usage.output_tokens = output_tokens
    return response


def _make_client(response_or_exc) -> MagicMock:
    """Return a fake Anthropic client whose messages.create() returns or raises."""
    client = MagicMock()
    if isinstance(response_or_exc, Exception):
        client.messages.create.side_effect = response_or_exc
    else:
        client.messages.create.return_value = response_or_exc
    return client


# ── test 1: happy path ────────────────────────────────────────────────────────


def test_happy_path_returns_correct_tuple() -> None:
    """Mocked client returns valid tool_use; classifier parses and returns QueryTuple."""
    response = _make_response(
        {"intent": "retrieve", "focus": "causal", "explicit_refs": ["squats"]},
        input_tokens=12,
        output_tokens=8,
    )
    client = _make_client(response)
    clf = LLMClassifier(client=client)

    result = clf.classify(Query(text="Why do I avoid squats?"))

    assert result == QueryTuple(intent=Intent.RETRIEVE, focus=Focus.CAUSAL)
    assert clf.last_explicit_refs == ["squats"]
    assert clf.last_warning is None
    assert clf.last_input_tokens == 12
    assert clf.last_output_tokens == 8
    assert clf.last_latency_ms >= 0.0


# ── test 2: malformed response falls back ─────────────────────────────────────


def test_malformed_intent_falls_back_to_keyword_classifier() -> None:
    """tool_use block with invalid intent value triggers fallback."""
    response = _make_response({"intent": "bogus", "focus": "causal", "explicit_refs": []})
    client = _make_client(response)
    clf = LLMClassifier(client=client)

    query = Query(text="Why do I avoid squats?")
    result = clf.classify(query)

    # Fallback should return what KeywordClassifier would return.
    expected = KeywordClassifier().classify(query)
    assert result == expected
    assert clf.last_warning is not None
    assert clf.last_warning.startswith("LLMClassifier fallback:")


# ── test 3: network exception falls back ──────────────────────────────────────


def test_exception_falls_back_and_records_warning() -> None:
    """RuntimeError from messages.create triggers fallback with warning containing the message."""
    client = _make_client(RuntimeError("network"))
    clf = LLMClassifier(client=client)

    query = Query(text="How do I progress overhead press?")
    result = clf.classify(query)

    expected = KeywordClassifier().classify(query)
    assert result == expected
    assert clf.last_warning is not None
    assert "network" in clf.last_warning
    assert clf.last_warning.startswith("LLMClassifier fallback:")


# ── test 4: integration smoke ─────────────────────────────────────────────────


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
def test_integration_smoke() -> None:
    """Real LLMClassifier with no explicit client classifies a causal query correctly."""
    clf = LLMClassifier()
    result = clf.classify(Query(text="Why do I avoid squats?"))
    assert result.intent == Intent.RETRIEVE
    assert result.focus == Focus.CAUSAL
