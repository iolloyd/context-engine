"""LLM-assisted edge suggestion.

Given a target node, asks Claude to propose edges from that node to
any of the other nodes in the graph that it judges semantically
related. Uses forced tool-call mode for structured output. All
proposals go through an interactive y/n/skip loop by default so the
human stays in the loop; a non-interactive mode returns the list for
programmatic use.

Falls back silently when the Anthropic SDK is not installed or no
API key is set in the environment. Callers should expect an empty
list in those cases, not an exception.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from context_engine.types import Node

_TOOL_SCHEMA: dict[str, Any] = {
    "name": "suggest_edges",
    "description": "Suggest edges from the target node to semantically related candidate nodes.",
    "input_schema": {
        "type": "object",
        "properties": {
            "suggestions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "target_id": {"type": "string"},
                        "edge_type": {"type": "string"},
                        "weight": {"type": "number"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["target_id", "edge_type", "weight", "rationale"],
                },
            }
        },
        "required": ["suggestions"],
    },
}

_SYSTEM_PROMPT = """\
You are helping a user build a typed knowledge graph. Given a target node and a list of
candidate nodes, suggest edges **from** the target node **to** any candidate that is
semantically related. For each suggestion, choose:

- `target_id`: exact id from the candidates list
- `edge_type`: a relationship name — prefer reusing types the user already uses (provided
  in the prompt); common ones include `because_of`, `supports`, `contradicts`, `is_a`,
  `has_property`, `requires`, `depends_on`, `if_then`, `explains`, `part_of`
- `weight`: 0.0–1.0 confidence
- `rationale`: one sentence explaining why the edge is justified

Only suggest edges you're genuinely confident about. It's better to return zero suggestions
than a wrong one. Always call the suggest_edges tool, even if the list is empty.\
"""


def _first_sentence(text: str) -> str:
    """Return the first sentence of *text*, truncated to 200 characters."""
    for sep in (". ", ".\n", "! ", "? "):
        idx = text.find(sep)
        if idx != -1:
            return text[: idx + 1].strip()
    return text[:200].strip()


@dataclass
class EdgeSuggestion:
    """A proposed edge from the target node to a candidate node."""

    target: str
    type: str
    weight: float
    rationale: str


class EdgeSuggester:
    """LLM-backed edge suggester using Claude's structured tool-call output.

    Proposes typed edges from a target node to semantically related candidates.
    Falls back silently (returns empty list) when the Anthropic SDK is not
    installed, no API key is set, or any error occurs during the LLM call.
    """

    def __init__(
        self,
        client: Any | None = None,
        model: str = "claude-sonnet-4-5",
        max_candidates: int = 50,
    ) -> None:
        self.model = model
        self.max_candidates = max_candidates
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

    def is_available(self) -> bool:
        """Return True if the Anthropic SDK is available and a client is configured."""
        return self._client is not None and self._import_error is None

    def suggest(
        self,
        target_node: Node,
        candidate_nodes: list[Node],
        target_embedding: list[float] | None = None,
        existing_edge_types: list[str] | None = None,
    ) -> list[EdgeSuggestion]:
        """Suggest edges from *target_node* to semantically related candidates.

        Parameters
        ----------
        target_node:
            The node for which to find outgoing edge suggestions.
        candidate_nodes:
            Nodes that may be linked to the target. Should exclude the target
            itself and any nodes already connected to it.
        target_embedding:
            If provided, candidates are assumed to have been pre-sorted by
            vector similarity to the target (e.g. via ``store.knn``). Otherwise
            candidates are ranked by ``updated_at`` descending.
        existing_edge_types:
            Edge types already present in the graph; the model is encouraged
            to reuse these rather than inventing new labels.

        Returns
        -------
        list[EdgeSuggestion]
            Validated suggestions. Returns ``[]`` on any failure.
        """
        if not self.is_available():
            return []
        if not candidate_nodes:
            return []

        ranked = self._rank_candidates(candidate_nodes, target_embedding)

        try:
            return self._call_llm(target_node, ranked, existing_edge_types or [])
        except Exception as exc:  # noqa: BLE001
            print(f"EdgeSuggester: LLM call failed: {exc}", file=sys.stderr)
            return []

    def _rank_candidates(
        self,
        candidates: list[Node],
        target_embedding: list[float] | None,
    ) -> list[Node]:
        """Rank and truncate candidates.

        When *target_embedding* is provided the caller is expected to have
        pre-sorted *candidates* by vector similarity (via ``store.knn``);
        this method only truncates.  Without an embedding, candidates are
        sorted by ``updated_at`` descending before truncating.
        """
        if target_embedding is None:
            candidates = sorted(candidates, key=lambda n: n.updated_at, reverse=True)
        return candidates[: self.max_candidates]

    def _build_user_message(
        self,
        target_node: Node,
        candidates: list[Node],
        existing_edge_types: list[str],
    ) -> str:
        parts: list[str] = []
        parts.append(f"Target node [id={target_node.id}]:")
        parts.append(target_node.content)
        parts.append("")

        if existing_edge_types:
            parts.append(f"Edge types already used in this graph: {', '.join(existing_edge_types)}")
            parts.append("")

        parts.append("Candidate nodes (suggest edges to any that are semantically related):")
        for i, node in enumerate(candidates, start=1):
            snippet = _first_sentence(node.content)
            parts.append(f"  {i}. [id={node.id}] {snippet}")

        return "\n".join(parts)

    def _call_llm(
        self,
        target_node: Node,
        candidates: list[Node],
        existing_edge_types: list[str],
    ) -> list[EdgeSuggestion]:
        candidate_ids = {n.id for n in candidates}
        user_message = self._build_user_message(target_node, candidates, existing_edge_types)

        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            tools=[_TOOL_SCHEMA],
            tool_choice={"type": "tool", "name": "suggest_edges"},
            messages=[{"role": "user", "content": user_message}],
        )

        # Extract the tool_use block.
        tool_block = next(
            (block for block in response.content if getattr(block, "type", None) == "tool_use"),
            None,
        )
        if tool_block is None:
            raise ValueError("LLM response contained no tool_use block")

        raw: dict[str, Any] = tool_block.input
        raw_suggestions: list[Any] = raw.get("suggestions") or []

        results: list[EdgeSuggestion] = []
        for item in raw_suggestions:
            if not isinstance(item, dict):
                continue
            target_id = str(item.get("target_id", ""))
            if target_id not in candidate_ids:
                # Drop suggestions that reference unknown nodes.
                continue
            try:
                results.append(
                    EdgeSuggestion(
                        target=target_id,
                        type=str(item["edge_type"]),
                        weight=float(item["weight"]),
                        rationale=str(item["rationale"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue

        return results
