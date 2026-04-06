"""Tests for SeedSelector with HashEmbedder."""

from __future__ import annotations

import pytest

from context_engine.embedding import HashEmbedder
from context_engine.seeds import SeedSelector
from context_engine.store import GraphStore
from context_engine.types import Query


@pytest.fixture
def indexed_store(store: GraphStore) -> GraphStore:
    """Return the conftest store with embeddings backfilled via HashEmbedder."""
    emb = HashEmbedder(dim=store.embed_dim)
    for nid in store.all_node_ids():
        node = store.get_node(nid)
        assert node is not None
        store._set_embedding(nid, emb.embed(node.content))
    return store


def test_seed_selector_returns_squat_for_squat_query(indexed_store: GraphStore) -> None:
    selector = SeedSelector(indexed_store, embed_fn=HashEmbedder(dim=indexed_store.embed_dim).embed)
    seeds = selector.select(Query(text="Why do I avoid squats?"), k=5)
    assert "squat" in seeds, f"expected 'squat' in top-k seeds, got {seeds}"


def test_seed_selector_no_seeds_without_embed_fn(store: GraphStore) -> None:
    """Without embed_fn or explicit refs, selector returns empty."""
    selector = SeedSelector(store)
    seeds = selector.select(Query(text="Why do I avoid squats?"))
    assert seeds == []
