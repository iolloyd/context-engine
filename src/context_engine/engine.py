"""End-to-end orchestration.

The engine wires classifier → seed selector → strategy → traversal → logic →
(optional) widen → synthesis. LLM calls are pluggable via the ``Synthesiser``
protocol; the default is offline and returns a structured dump of the slice
so tests can run without network access.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from context_engine.classifier import Classifier, KeywordClassifier
from context_engine.feedback import FeedbackApplier
from context_engine.logic import LogicEngine, LogicResult
from context_engine.seeds import SeedSelector
from context_engine.store import GraphStore
from context_engine.strategy import strategy_for, widen
from context_engine.traversal import Traverser
from context_engine.types import (
    ContextSlice,
    FeedbackSignal,
    Intent,
    Query,
    QueryTuple,
)

if TYPE_CHECKING:
    from context_engine.embedding import Embedder


class Synthesiser(Protocol):
    def synthesise(
        self,
        query: Query,
        tuple_: QueryTuple,
        slice_: ContextSlice,
        logic_results: list[LogicResult],
    ) -> EngineResponse: ...


@dataclass
class EngineResponse:
    text: str
    tuple_: QueryTuple
    slice_: ContextSlice
    logic_results: list[LogicResult]
    needed_widen: bool = False
    gap_flag: bool = False
    warnings: list[str] = field(default_factory=list)


class OfflineSynthesiser:
    """Default synthesiser: dumps the slice as a readable report.

    Useful for tests and for running the engine without an LLM backend.
    """

    def synthesise(
        self,
        query: Query,
        tuple_: QueryTuple,
        slice_: ContextSlice,
        logic_results: list[LogicResult],
    ) -> EngineResponse:
        lines = [
            f"Query: {query.text}",
            f"Tuple: {tuple_}",
            f"Seeds: {', '.join(slice_.seeds) or '(none)'}",
            f"Nodes: {len(slice_.nodes)}  Chars: {slice_.char_count()}",
        ]
        if slice_.notes:
            lines.append("Notes: " + "; ".join(slice_.notes))
        lines.append("")
        for n in slice_.nodes:
            lines.append(f"- [{n.type or 'node'}] {n.content}")
        if logic_results:
            lines.append("")
            lines.append("Logic results:")
            for r in logic_results:
                lines.append(f"- {r.rule_id}: {r.explanation}")
        return EngineResponse(
            text="\n".join(lines),
            tuple_=tuple_,
            slice_=slice_,
            logic_results=logic_results,
        )


class ContextEngine:
    def __init__(
        self,
        store: GraphStore,
        classifier: Classifier | None = None,
        synthesiser: Synthesiser | None = None,
        embed_fn: Callable[[str], list[float]] | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.store = store
        self.classifier = classifier or KeywordClassifier()
        self.synthesiser = synthesiser or OfflineSynthesiser()
        self._embedder = embedder
        # embedder.embed takes precedence over the raw embed_fn callable.
        resolved_embed_fn = embedder.embed if embedder is not None else embed_fn
        self.seeds = SeedSelector(store, embed_fn=resolved_embed_fn)
        self.traverser = Traverser(store)
        self.logic = LogicEngine()
        self.feedback = FeedbackApplier(store)

    def index_all(self) -> tuple[int, int]:
        """Backfill embeddings for every node that does not yet have one.

        Returns ``(indexed, already_had)`` counts.
        Raises ``RuntimeError`` if no embedder was configured.
        """
        if self._embedder is None:
            raise RuntimeError("ContextEngine.index_all() requires an embedder")
        indexed = 0
        already_had = 0
        for nid in self.store.all_node_ids():
            if self.store.has_embedding(nid):
                already_had += 1
                continue
            node = self.store.get_node(nid)
            if node is None:
                continue
            self.store._set_embedding(nid, self._embedder.embed(node.content))
            indexed += 1
        return indexed, already_had

    def answer(
        self,
        query: Query,
        metadata_filter: dict[str, object] | None = None,
    ) -> EngineResponse:
        tuple_ = self.classifier.classify(query)
        seeds = self.seeds.select(query, metadata_filter=metadata_filter)
        strategy = strategy_for(tuple_, seeds=seeds)
        slice_ = self.traverser.traverse(strategy, tuple_)

        logic_results: list[LogicResult] = []
        if tuple_.intent in (Intent.EVALUATE, Intent.ACT):
            logic_results = self.logic.evaluate(slice_)

        # Gap detection: any logic result that flagged missing facts widens
        # the traversal and retries once.
        needed_widen = False
        if any(r.missing for r in logic_results):
            needed_widen = True
            wider = widen(strategy)
            slice_ = self.traverser.traverse(wider, tuple_)
            logic_results = self.logic.evaluate(slice_)

        response = self.synthesiser.synthesise(query, tuple_, slice_, logic_results)
        response.needed_widen = needed_widen
        response.gap_flag = needed_widen
        return response

    def record_feedback(self, signal: FeedbackSignal) -> dict[str, float]:
        return self.feedback.apply(signal)
