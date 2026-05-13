"""Seed selection: query → entry node ids.

Three strategies, tried in order:

  1. Explicit reference — the query names an id or exact slug
  2. Metadata filter — e.g. ``(status=avoided, type=exercise)``
  3. Vector similarity — fallback
  4. Conversation context — appended when other sources underflow

Returning zero seeds is valid: scan intents run without seeds.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from context_engine.types import Query, StoreProtocol

if TYPE_CHECKING:
    from context_engine.composed_store import ComposedStore


class SeedSelector:
    def __init__(
        self,
        store: StoreProtocol,
        embed_fn: Callable[[str], list[float]] | None = None,
        global_store: StoreProtocol | None = None,
    ) -> None:
        self.store = store
        self.embed_fn = embed_fn
        self.global_store = global_store

    def _composed(self) -> StoreProtocol | ComposedStore:
        """Return a ComposedStore when a global store is present, else the project store."""
        if self.global_store is not None:
            from context_engine.composed_store import ComposedStore  # noqa: PLC0415

            return ComposedStore(self.store, self.global_store)
        return self.store

    def select(
        self,
        query: Query,
        metadata_filter: dict[str, object] | None = None,
        k: int = 5,
    ) -> list[str]:
        composed = self._composed()
        seeds: list[str] = []

        # 1. explicit references (ids the caller provided)
        for ref in query.explicit_refs:
            if composed.get_node(ref) is not None:
                seeds.append(ref)

        # 2. metadata filter — only project store supports filter_nodes
        if metadata_filter:
            for node in self.store.filter_nodes(metadata_filter):
                if node.id not in seeds:
                    seeds.append(node.id)
            if self.global_store is not None:
                for node in self.global_store.filter_nodes(metadata_filter):
                    if node.id not in seeds:
                        seeds.append(node.id)

        # 3. vector similarity
        if not seeds and self.embed_fn is not None:
            vec = self.embed_fn(query.text)
            for node_id, _dist in composed.knn(vec, k=k):
                if node_id not in seeds:
                    seeds.append(node_id)

        # 4. conversation context (append, not replace)
        for cid in query.conversation_node_ids:
            if cid not in seeds and composed.get_node(cid) is not None:
                seeds.append(cid)

        return seeds[:k] if seeds else seeds
