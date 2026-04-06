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

from context_engine.store import GraphStore
from context_engine.types import Query


class SeedSelector:
    def __init__(
        self,
        store: GraphStore,
        embed_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self.store = store
        self.embed_fn = embed_fn

    def select(
        self,
        query: Query,
        metadata_filter: dict[str, object] | None = None,
        k: int = 5,
    ) -> list[str]:
        seeds: list[str] = []

        # 1. explicit references (ids the caller provided)
        for ref in query.explicit_refs:
            if self.store.get_node(ref) is not None:
                seeds.append(ref)

        # 2. metadata filter
        if metadata_filter:
            for node in self.store.filter_nodes(metadata_filter):
                if node.id not in seeds:
                    seeds.append(node.id)

        # 3. vector similarity
        if not seeds and self.embed_fn is not None:
            vec = self.embed_fn(query.text)
            for node_id, _dist in self.store.knn(vec, k=k):
                if node_id not in seeds:
                    seeds.append(node_id)

        # 4. conversation context (append, not replace)
        for cid in query.conversation_node_ids:
            if cid not in seeds and self.store.get_node(cid) is not None:
                seeds.append(cid)

        return seeds[:k] if seeds else seeds
