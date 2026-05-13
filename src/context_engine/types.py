"""Core data types for the context engine.

Two primitives — Node and Edge — carry everything. Metadata is deliberately
open so the schema can extend without code changes. The query tuple
(intent × focus) drives traversal strategy.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

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


@runtime_checkable
class StoreProtocol(Protocol):
    """Storage backend contract for nodes, edges, and embeddings.

    Two implementations satisfy this protocol: ``GraphStore`` (SQLite +
    ``sqlite-vec``) and ``PgGraphStore`` (Postgres + ``pgvector``). The
    engine layer above the store talks to this interface and nothing
    below it. See ADR 0002 for the rationale.
    """

    def upsert_node(
        self,
        content: str,
        metadata: dict[str, Any] | None = ...,
        node_id: str | None = ...,
        embedding: list[float] | None = ...,
    ) -> Node: ...

    def get_node(self, node_id: str) -> Node | None: ...

    def get_nodes(self, node_ids: Iterable[str]) -> list[Node]: ...

    def delete_node(self, node_id: str) -> None: ...

    def filter_nodes(self, where: dict[str, Any]) -> list[Node]: ...

    def upsert_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        weight: float = ...,
        content: str | None = ...,
        metadata: dict[str, Any] | None = ...,
        edge_id: str | None = ...,
    ) -> Edge: ...

    def get_edge(self, edge_id: str) -> Edge | None: ...

    def out_edges(
        self,
        node_id: str,
        edge_types: list[str] | None = ...,
        weight_floor: float = ...,
    ) -> list[Edge]: ...

    def in_edges(
        self,
        node_id: str,
        edge_types: list[str] | None = ...,
        weight_floor: float = ...,
    ) -> list[Edge]: ...

    def update_edge_weight(self, edge_id: str, new_weight: float) -> None: ...

    def has_embedding(self, node_id: str) -> bool: ...

    def get_embedding(self, node_id: str) -> list[float] | None: ...

    def knn(self, vec: list[float], k: int = ...) -> list[tuple[str, float]]: ...

    def all_node_ids(self) -> list[str]: ...

    def all_edges(self) -> list[Edge]: ...

    def node_count(self) -> int: ...

    def edge_count(self) -> int: ...

    def transaction(self) -> AbstractContextManager[Any]: ...

    def close(self) -> None: ...
