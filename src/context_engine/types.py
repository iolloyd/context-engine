"""Core data types for the context engine.

Two primitives — Node and Edge — carry everything. Metadata is deliberately
open so the schema can extend without code changes. The query tuple
(intent × focus) drives traversal strategy.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _utc_now() -> datetime:
    return datetime.now(UTC)


class Intent(StrEnum):
    """What the user wants — controls which pipeline components run."""

    RETRIEVE = "retrieve"
    EVALUATE = "evaluate"
    COMPARE = "compare"
    ACT = "act"
    SCAN = "scan"


class Focus(StrEnum):
    """What kind of knowledge is needed — controls traversal edge priorities."""

    CAUSAL = "causal"
    PROCEDURAL = "procedural"
    TEMPORAL = "temporal"
    ATTRIBUTIVE = "attributive"
    CONDITIONAL = "conditional"


class Node(BaseModel):
    """A knowledge atom. Metadata distinguishes fact / rule / observation / etc."""

    model_config = ConfigDict(extra="forbid")

    id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    @property
    def type(self) -> str | None:
        value = self.metadata.get("type")
        return str(value) if value is not None else None

    @property
    def status(self) -> str | None:
        value = self.metadata.get("status")
        return str(value) if value is not None else None

    @property
    def domain(self) -> str | None:
        value = self.metadata.get("domain")
        return str(value) if value is not None else None


class Edge(BaseModel):
    """A relationship between two nodes. Rich — carries its own content and weight."""

    model_config = ConfigDict(extra="forbid")

    id: str
    source: str
    target: str
    content: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    @property
    def type(self) -> str:
        value = self.metadata.get("type")
        if value is None:
            raise ValueError(f"edge {self.id} is missing metadata.type")
        return str(value)

    @property
    def weight(self) -> float:
        return float(self.metadata.get("weight", 0.5))

    @weight.setter
    def weight(self, value: float) -> None:
        self.metadata["weight"] = max(0.0, min(1.0, value))


class QueryTuple(BaseModel):
    """(intent, focus) — the two orthogonal axes that drive the pipeline."""

    model_config = ConfigDict(extra="forbid")

    intent: Intent
    focus: Focus

    def __str__(self) -> str:
        return f"({self.intent.value}, {self.focus.value})"


class Query(BaseModel):
    """A user query plus optional explicit references and conversation context."""

    model_config = ConfigDict(extra="forbid")

    text: str
    explicit_refs: list[str] = Field(default_factory=list)
    conversation_node_ids: list[str] = Field(default_factory=list)


class TraversalStrategy(BaseModel):
    """Parameters that configure a single traversal run."""

    model_config = ConfigDict(extra="forbid")

    seeds: list[str] = Field(default_factory=list)
    edge_types: list[str] = Field(default_factory=list)
    depth: int = 3
    weight_floor: float = 0.2
    budget: int = 20
    anti_hub: bool = True
    preserve_rule_chains: bool = True


class ContextSlice(BaseModel):
    """The output of traversal — nodes + edges that form the query's context."""

    model_config = ConfigDict(extra="forbid")

    nodes: list[Node]
    edges: list[Edge]
    seeds: list[str]
    strategy: TraversalStrategy
    notes: list[str] = Field(default_factory=list)

    def char_count(self) -> int:
        return sum(len(n.content) for n in self.nodes)


class FeedbackSignal(BaseModel):
    """A signal produced downstream that the traversal should learn from."""

    model_config = ConfigDict(extra="forbid")

    query_text: str
    used_node_ids: list[str]
    missing_node_ids: list[str] = Field(default_factory=list)
    helpful_edge_ids: list[str] = Field(default_factory=list)
    noisy_edge_ids: list[str] = Field(default_factory=list)
    source: str = "llm"  # "llm", "user", "logic_engine"
    delta: float = 0.1
    query_tuple: QueryTuple | None = None
