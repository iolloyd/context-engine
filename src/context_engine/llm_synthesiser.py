"""LLM-backed synthesiser with gap detection.

Asks Claude to answer the query using only the provided slice. The model
is required to return a structured JSON payload via tool-call mode listing:

  - answer         — the natural-language response
  - used_node_ids  — ids it actually referenced (drives the feedback loop)
  - missing        — free-text gaps that widen() should try to fill
  - confidence     — 0.0-1.0 self-reported confidence

On any parse failure or client error, returns an EngineResponse that
wraps OfflineSynthesiser.synthesise() and records the failure reason.
"""

from __future__ import annotations

from typing import Any

from context_engine.engine import EngineResponse, OfflineSynthesiser, Synthesiser
from context_engine.logic import LogicResult
from context_engine.types import ContextSlice, Query, QueryTuple

_SYSTEM_PROMPT = (
    "You are a precise assistant. Answer the user's question using ONLY the provided "
    "knowledge fragments. If critical information is missing, list it under 'missing'. "
    "Always call the answer tool."
)

_ANSWER_TOOL: dict[str, Any] = {
    "name": "answer",
    "description": (
        "Return a structured answer. Use only the knowledge fragments provided. "
        "List node ids you referenced in used_node_ids. List knowledge gaps in missing."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The natural-language answer to the user's question.",
            },
            "used_node_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "IDs of knowledge fragments that informed the answer.",
            },
            "missing": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Free-text descriptions of knowledge that was needed but absent "
                    "from the provided fragments."
                ),
            },
            "confidence": {
                "type": "number",
                "description": "Self-reported confidence between 0.0 and 1.0.",
            },
        },
        "required": ["answer", "used_node_ids", "missing", "confidence"],
    },
}


def _build_user_message(
    query: Query,
    tuple_: QueryTuple,
    slice_: ContextSlice,
    logic_results: list[LogicResult],
) -> str:
    parts: list[str] = []
    parts.append(f"Question: {query.text}")
    parts.append(f"Query tuple: {tuple_}")
    parts.append("")

    if slice_.nodes:
        parts.append("Knowledge fragments:")
        for i, node in enumerate(slice_.nodes, start=1):
            node_type = node.type or "node"
            parts.append(f"  {i}. [NODE id={node.id} type={node_type}] {node.content}")
    else:
        parts.append("Knowledge fragments: (none)")

    if logic_results:
        parts.append("")
        parts.append("Logic evaluation results:")
        for r in logic_results:
            parts.append(
                f"  [LOGIC rule_id={r.rule_id} passed={r.passed} explanation={r.explanation}]"
            )

    return "\n".join(parts)


class LLMSynthesiser:
    """Production synthesiser backed by Claude via the Anthropic SDK.

    Parameters
    ----------
    client:
        An ``anthropic.Anthropic`` client instance. If ``None``, one is
        constructed lazily on first use. If the import or construction fails,
        all calls fall through to ``fallback``.
    model:
        The Claude model identifier to use.
    fallback:
        Synthesiser to use when the LLM call fails. Defaults to
        ``OfflineSynthesiser()``.
    """

    def __init__(
        self,
        client: Any | None = None,
        model: str = "claude-sonnet-4-5",
        fallback: Synthesiser | None = None,
    ) -> None:
        self._client = client
        self.model = model
        self.fallback = fallback if fallback is not None else OfflineSynthesiser()
        self._client_error: str | None = None

        if client is None:
            try:
                import anthropic  # noqa: PLC0415

                self._client = anthropic.Anthropic()
            except Exception as exc:  # pragma: no cover
                self._client_error = str(exc)

    def synthesise(
        self,
        query: Query,
        tuple_: QueryTuple,
        slice_: ContextSlice,
        logic_results: list[LogicResult],
    ) -> EngineResponse:
        """Synthesise a response, falling back to offline mode on any error."""
        if self._client is None:
            fallback_resp = self.fallback.synthesise(query, tuple_, slice_, logic_results)
            if self._client_error:
                fallback_resp.missing = [
                    f"LLM client unavailable: {self._client_error}"
                ]
            return fallback_resp

        user_message = _build_user_message(query, tuple_, slice_, logic_results)

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=_SYSTEM_PROMPT,
                tools=[_ANSWER_TOOL],
                tool_choice={"type": "tool", "name": "answer"},
                messages=[{"role": "user", "content": user_message}],
            )
        except Exception as exc:
            fallback_resp = self.fallback.synthesise(query, tuple_, slice_, logic_results)
            fallback_resp.missing = [f"LLM call failed: {exc}"]
            return fallback_resp

        # Extract the tool_use block.
        tool_input: dict[str, Any] | None = None
        for block in response.content:
            if hasattr(block, "type") and block.type == "tool_use" and block.name == "answer":
                tool_input = block.input
                break

        if tool_input is None:
            fallback_resp = self.fallback.synthesise(query, tuple_, slice_, logic_results)
            fallback_resp.missing = ["LLM did not call the answer tool"]
            return fallback_resp

        try:
            answer_text = str(tool_input["answer"])
            used_node_ids = [str(x) for x in tool_input.get("used_node_ids", [])]
            missing = [str(x) for x in tool_input.get("missing", [])]
            confidence = float(tool_input.get("confidence", 1.0))
        except (KeyError, TypeError, ValueError) as exc:
            fallback_resp = self.fallback.synthesise(query, tuple_, slice_, logic_results)
            fallback_resp.missing = [f"LLM response parse error: {exc}"]
            return fallback_resp

        return EngineResponse(
            text=answer_text,
            tuple_=tuple_,
            slice_=slice_,
            logic_results=logic_results,
            used_node_ids=used_node_ids,
            missing=missing,
            confidence=confidence,
        )
