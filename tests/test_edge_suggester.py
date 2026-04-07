"""Tests for EdgeSuggester.

Covers the happy path, invalid-target filtering, network failure, missing SDK,
and an integration smoke test (skipped without ANTHROPIC_API_KEY).
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from context_engine.edge_suggester import EdgeSuggester, EdgeSuggestion
from context_engine.types import Node

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_node(node_id: str, content: str = "Some content.") -> Node:
    return Node(
        id=node_id,
        content=content,
        metadata={"type": "exercise"},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def _make_tool_use_block(input_dict: dict) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.input = input_dict
    return block


def _make_response(suggestions: list[dict]) -> MagicMock:
    block = _make_tool_use_block({"suggestions": suggestions})
    response = MagicMock()
    response.content = [block]
    return response


def _make_client(response_or_exc) -> MagicMock:
    client = MagicMock()
    if isinstance(response_or_exc, Exception):
        client.messages.create.side_effect = response_or_exc
    else:
        client.messages.create.return_value = response_or_exc
    return client


# ── test 1: happy path ────────────────────────────────────────────────────────


def test_happy_path_returns_two_suggestions() -> None:
    """Mocked client returns two valid suggestions; both are returned."""
    target = _make_node("exercises/squat", "Squat is a compound lower-body movement.")
    candidates = [
        _make_node("conditions/knee-pain", "Knee pain limits range of motion."),
        _make_node("rules/progression", "Progressive overload drives adaptation."),
    ]
    raw_suggestions = [
        {
            "target_id": "conditions/knee-pain",
            "edge_type": "because_of",
            "weight": 0.9,
            "rationale": "Squats aggravate knee pain.",
        },
        {
            "target_id": "rules/progression",
            "edge_type": "if_then",
            "weight": 0.7,
            "rationale": "Progression rule governs squat loading.",
        },
    ]
    client = _make_client(_make_response(raw_suggestions))
    suggester = EdgeSuggester(client=client)

    result = suggester.suggest(target, candidates)

    assert len(result) == 2
    assert result[0].target == "conditions/knee-pain"
    assert result[0].type == "because_of"
    assert abs(result[0].weight - 0.9) < 1e-6
    assert result[0].rationale == "Squats aggravate knee pain."
    assert result[1].target == "rules/progression"
    assert result[1].type == "if_then"


# ── test 2: invalid target filtered ──────────────────────────────────────────


def test_invalid_target_id_filtered_out() -> None:
    """Suggestions referencing unknown target_ids are dropped; valid ones survive."""
    target = _make_node("exercises/squat")
    candidates = [
        _make_node("conditions/knee-pain"),
        _make_node("rules/progression"),
    ]
    raw_suggestions = [
        {
            "target_id": "conditions/knee-pain",
            "edge_type": "because_of",
            "weight": 0.9,
            "rationale": "Valid.",
        },
        {
            "target_id": "exercises/NONEXISTENT",
            "edge_type": "if_then",
            "weight": 0.5,
            "rationale": "Invalid target.",
        },
        {
            "target_id": "rules/progression",
            "edge_type": "supports",
            "weight": 0.6,
            "rationale": "Also valid.",
        },
    ]
    client = _make_client(_make_response(raw_suggestions))
    suggester = EdgeSuggester(client=client)

    result = suggester.suggest(target, candidates)

    assert len(result) == 2
    result_targets = {s.target for s in result}
    assert result_targets == {"conditions/knee-pain", "rules/progression"}


# ── test 3: network failure returns empty and warns ───────────────────────────


def test_network_failure_returns_empty_and_warns(capsys: pytest.CaptureFixture) -> None:
    """RuntimeError from messages.create causes suggest() to return [] and print a warning."""
    target = _make_node("exercises/squat")
    candidates = [_make_node("conditions/knee-pain")]
    client = _make_client(RuntimeError("network timeout"))
    suggester = EdgeSuggester(client=client)

    result = suggester.suggest(target, candidates)

    assert result == []
    captured = capsys.readouterr()
    assert "network timeout" in captured.err


# ── test 4: no SDK available ──────────────────────────────────────────────────


def test_no_sdk_is_available_returns_false_and_suggest_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With anthropic removed from sys.modules, is_available() is False and suggest() is []."""
    monkeypatch.setitem(sys.modules, "anthropic", None)  # type: ignore[arg-type]

    suggester = EdgeSuggester()  # must NOT raise

    assert not suggester.is_available()

    target = _make_node("exercises/squat")
    candidates = [_make_node("conditions/knee-pain")]
    result = suggester.suggest(target, candidates)
    assert result == []


# ── test 5: integration smoke test ───────────────────────────────────────────


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
def test_integration_smoke() -> None:
    """Real EdgeSuggester with a small fixture; asserts the live API path does not raise."""
    suggester = EdgeSuggester()
    assert suggester.is_available()

    target = _make_node(
        "exercises/squat",
        "Squat is a compound lower-body movement that loads the knee joint.",
    )
    candidates = [
        _make_node("conditions/knee-pain", "Knee pain limits squat depth and range of motion."),
        _make_node("rules/progression", "Progressive overload increases load each session."),
        _make_node("exercises/deadlift", "Deadlift is a hip-hinge posterior chain movement."),
    ]
    result = suggester.suggest(target, candidates)

    # Non-negative count; no assertion on semantic content — just that the live path works.
    assert isinstance(result, list)
    assert all(isinstance(s, EdgeSuggestion) for s in result)
