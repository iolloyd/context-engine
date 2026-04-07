"""Tests for LLMSynthesiser and the gap-detection / auto-feedback paths in ContextEngine."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock

import pytest

from context_engine.engine import ContextEngine, EngineResponse
from context_engine.llm_synthesiser import LLMSynthesiser
from context_engine.logic import LogicResult
from context_engine.store import GraphStore
from context_engine.types import ContextSlice, Query, QueryTuple, TraversalStrategy

# ── helpers ────────────────────────────────────────────────────────────────────


def _empty_slice() -> ContextSlice:
    return ContextSlice(
        nodes=[],
        edges=[],
        seeds=[],
        strategy=TraversalStrategy(),
    )


def _make_tool_use_block(data: dict[str, Any]) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "answer"
    block.input = data
    return block


def _make_client_response(tool_input: dict[str, Any]) -> MagicMock:
    msg = MagicMock()
    msg.content = [_make_tool_use_block(tool_input)]
    return msg


# ── test 1: happy path with mocked client ──────────────────────────────────────


def test_llm_synthesiser_happy_path() -> None:
    """Mocked client returns a valid tool_use block; EngineResponse fields are set."""
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _make_client_response(
        {
            "answer": "You should increase bench weight.",
            "used_node_ids": ["bench", "bench_obs_4"],
            "missing": [],
            "confidence": 0.9,
        }
    )

    synth = LLMSynthesiser(client=fake_client, model="claude-test")
    query = Query(text="Should I increase bench weight?", explicit_refs=["bench"])
    qt = QueryTuple(intent="evaluate", focus="conditional")  # type: ignore[arg-type]
    lr = LogicResult(
        rule_id="progression_rule", passed=True, explanation="threshold met", missing=[]
    )

    resp = synth.synthesise(query, qt, _empty_slice(), [lr])

    assert resp.text == "You should increase bench weight."
    assert resp.used_node_ids == ["bench", "bench_obs_4"]
    assert resp.missing == []
    assert abs(resp.confidence - 0.9) < 1e-9
    assert fake_client.messages.create.called


# ── test 2: missing field triggers widen in ContextEngine ─────────────────────


class _TwoPassSynthesiser:
    """Returns `missing` on the first call, empty on the second."""

    def __init__(self) -> None:
        self.call_count = 0

    def synthesise(
        self,
        query: Query,
        tuple_: QueryTuple,
        slice_: ContextSlice,
        logic_results: list[LogicResult],
    ) -> EngineResponse:
        self.call_count += 1
        if self.call_count == 1:
            return EngineResponse(
                text="I need more context.",
                tuple_=tuple_,
                slice_=slice_,
                logic_results=logic_results,
                used_node_ids=[],
                missing=["progression rule was not in context"],
                confidence=0.3,
            )
        return EngineResponse(
            text="You should increase bench weight.",
            tuple_=tuple_,
            slice_=slice_,
            logic_results=logic_results,
            used_node_ids=["bench"],
            missing=[],
            confidence=0.9,
        )


def test_missing_field_triggers_widen(store: GraphStore) -> None:
    """Engine widens and re-synthesises when synthesiser reports missing knowledge."""
    synth = _TwoPassSynthesiser()
    engine = ContextEngine(store, synthesiser=synth)  # type: ignore[arg-type]

    resp = engine.answer(
        Query(text="Should I increase bench weight?", explicit_refs=["bench"])
    )

    assert synth.call_count >= 2, "engine must synthesise at least twice on gap"
    assert resp.needed_widen is True
    assert resp.text == "You should increase bench weight."


# ── test 3: auto_feedback updates edge weights ─────────────────────────────────


class _FixedSynthesiser:
    """Always reports progression_rule as the only used node."""

    def synthesise(
        self,
        query: Query,
        tuple_: QueryTuple,
        slice_: ContextSlice,
        logic_results: list[LogicResult],
    ) -> EngineResponse:
        return EngineResponse(
            text="Bench result.",
            tuple_=tuple_,
            slice_=slice_,
            logic_results=logic_results,
            used_node_ids=["progression_rule"],
            missing=[],
            confidence=1.0,
        )


def test_auto_feedback_updates_weights(store: GraphStore) -> None:
    """With auto_feedback=True, edges whose target is used get their weight updated.

    The bench → progression_rule (if_then) edge should be treated as helpful
    because progression_rule is in used_node_ids. Its weight must change.
    """
    engine = ContextEngine(
        store,
        synthesiser=_FixedSynthesiser(),  # type: ignore[arg-type]
        auto_feedback=True,
    )

    # The if_then edge from bench → progression_rule will appear in the slice
    # when querying with explicit_refs=["bench"].
    candidate_edges = [
        e for e in store.out_edges("bench") if e.target == "progression_rule"
    ]
    assert candidate_edges, "test fixture must have bench → progression_rule edge"
    edge = candidate_edges[0]
    before = edge.weight

    engine.answer(Query(text="Should I increase bench weight?", explicit_refs=["bench"]))

    refreshed = store.get_edge(edge.id)
    assert refreshed is not None
    # The edge targets progression_rule which is in used_node_ids → helpful → weight goes up.
    assert refreshed.weight != before, "weight must change after auto_feedback"


# ── test 4: integration smoke — skipped without API key ───────────────────────


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping live LLM test",
)
def test_llm_synthesiser_integration_smoke() -> None:
    """Build a real LLMSynthesiser and run a tiny slice through it."""
    synth = LLMSynthesiser()
    query = Query(text="What is bench press?", explicit_refs=["bench"])
    qt = QueryTuple(intent="retrieve", focus="attributive")  # type: ignore[arg-type]
    resp = synth.synthesise(query, qt, _empty_slice(), [])
    assert isinstance(resp.text, str)
    assert len(resp.text) > 0
    assert isinstance(resp.confidence, float)
    assert 0.0 <= resp.confidence <= 1.0
