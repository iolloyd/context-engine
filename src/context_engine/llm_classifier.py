"""LLM-backed classifier.

Drop-in replacement for KeywordClassifier that sends the query to Claude
and asks for a structured (intent, focus) tuple plus any entity references
the query names directly. Uses tool-call JSON mode for deterministic
parsing. Falls through to KeywordClassifier on any failure — network
error, invalid JSON, out-of-range enum values, or a missing Anthropic SDK.

The LLM call latency and token usage are recorded on the classifier
instance (last_latency_ms, last_input_tokens, last_output_tokens) so
callers can read them after classify() returns. These fields are reset
on every call.
"""

from __future__ import annotations

import time
from typing import Any

from context_engine.classifier import Classifier, KeywordClassifier
from context_engine.types import Focus, Intent, Query, QueryTuple

_TOOL_SCHEMA: dict[str, Any] = {
    "name": "classify_query",
    "description": (
        "Classify a natural-language query into an (intent, focus) tuple and extract "
        "any entity names the query references directly."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": [e.value for e in Intent],
                "description": "What the user wants: controls which pipeline components run.",
            },
            "focus": {
                "type": "string",
                "enum": [e.value for e in Focus],
                "description": (
                    "What kind of knowledge is needed: controls traversal edge priorities."
                ),
            },
            "explicit_refs": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Entity names the query references directly, e.g. 'squats', 'bench press'. "
                    "Empty list if the query contains no explicit entity references."
                ),
            },
        },
        "required": ["intent", "focus", "explicit_refs"],
    },
}

_SYSTEM_PROMPT = """\
You are a query classifier for a graph-based context retrieval engine.

Given a natural-language query, determine:
  - intent: what the user wants (retrieve, evaluate, compare, act, scan)
  - focus: what kind of knowledge is needed (causal, procedural, temporal, attributive, conditional)
  - explicit_refs: entity names the query names directly (e.g. "squats", "bench press")

Intent definitions:
  retrieve  — get information
  evaluate  — assess state against rules
  compare   — contrast two or more things
  act       — determine what to do
  scan      — aggregate or list across many entities

Focus definitions:
  causal      — why something is the case
  procedural  — how something works or is done
  temporal    — when something applies or triggers
  attributive — what something is or has
  conditional — whether criteria are met

Call the classify_query tool with your answer. Do not add explanation outside the tool call.\
"""


class LLMClassifier:
    """Classifier backed by Claude via structured tool-call output.

    Falls through to ``KeywordClassifier`` on any failure. Records latency
    and token usage on instance attributes after each call.
    """

    def __init__(
        self,
        client: Any | None = None,
        model: str = "claude-sonnet-4-5",
        fallback: Classifier | None = None,
    ) -> None:
        self.model = model
        self.fallback: Classifier = fallback if fallback is not None else KeywordClassifier()

        self._import_error: Exception | None = None
        if client is None:
            try:
                import anthropic  # noqa: PLC0415

                self._client: Any = anthropic.Anthropic()
            except Exception as exc:  # noqa: BLE001
                self._import_error = exc
                self._client = None
        else:
            self._client = client

        # Per-call metrics — reset at the start of every classify() call.
        self.last_latency_ms: float = 0.0
        self.last_input_tokens: int = 0
        self.last_output_tokens: int = 0
        self.last_warning: str | None = None
        self.last_explicit_refs: list[str] = []

    def classify(self, query: Query) -> QueryTuple:
        """Classify *query* via LLM; fall back to KeywordClassifier on any error."""
        # Reset per-call state.
        self.last_latency_ms = 0.0
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_warning = None
        self.last_explicit_refs = []

        if self._import_error is not None:
            self.last_warning = f"LLMClassifier fallback: {self._import_error}"
            return self.fallback.classify(query)

        try:
            return self._call_llm(query)
        except Exception as exc:  # noqa: BLE001
            self.last_warning = f"LLMClassifier fallback: {exc}"
            return self.fallback.classify(query)

    def _call_llm(self, query: Query) -> QueryTuple:
        t0 = time.perf_counter()
        response = self._client.messages.create(
            model=self.model,
            max_tokens=256,
            system=_SYSTEM_PROMPT,
            tools=[_TOOL_SCHEMA],
            tool_choice={"type": "tool", "name": "classify_query"},
            messages=[{"role": "user", "content": query.text}],
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.last_latency_ms = elapsed_ms
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.last_input_tokens = getattr(usage, "input_tokens", 0)
            self.last_output_tokens = getattr(usage, "output_tokens", 0)

        # Extract the tool_use block.
        tool_block = next(
            (block for block in response.content if getattr(block, "type", None) == "tool_use"),
            None,
        )
        if tool_block is None:
            raise ValueError("LLM response contained no tool_use block")

        raw: dict[str, Any] = tool_block.input

        # Validate enum values — raises ValueError on bad input, triggering fallback.
        intent = Intent(raw["intent"])
        focus = Focus(raw["focus"])

        self.last_explicit_refs = list(raw.get("explicit_refs") or [])

        return QueryTuple(intent=intent, focus=focus)
