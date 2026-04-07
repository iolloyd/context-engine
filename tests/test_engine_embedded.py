"""End-to-end tests for ContextEngine with embedding-based seed selection."""

from __future__ import annotations

from pathlib import Path

from context_engine.embedding import HashEmbedder
from context_engine.engine import ContextEngine
from context_engine.source import FolderTreeSource
from context_engine.store import GraphStore
from context_engine.types import Query

FIXTURE_TREE = Path(__file__).resolve().parents[1] / "fixtures" / "tree"


def test_engine_free_form_query_finds_squat(store: GraphStore) -> None:
    """ContextEngine with HashEmbedder answers a free-form query without explicit_refs."""
    engine = ContextEngine(store, embedder=HashEmbedder(dim=store.embed_dim))
    engine.index_all()

    resp = engine.answer(Query(text="Why do I avoid squats?"))
    node_ids = {n.id for n in resp.slice_.nodes}
    assert "squat" in node_ids or "knee" in node_ids, (
        f"expected squat or knee in slice, got {node_ids}"
    )


def test_index_all_counts(store: GraphStore) -> None:
    engine = ContextEngine(store, embedder=HashEmbedder(dim=store.embed_dim))
    indexed, already_had = engine.index_all()
    total = store.node_count()
    assert indexed > 0
    assert indexed + already_had == total

    # Second call: everything already has embeddings.
    indexed2, already_had2 = engine.index_all()
    assert indexed2 == 0
    assert already_had2 == total


def test_index_all_works_with_default_embedder() -> None:
    """ContextEngine constructed without an explicit embedder uses the default."""
    store = GraphStore(":memory:")
    engine = ContextEngine(store)
    # default_embedder() always succeeds (SentenceTransformerEmbedder or HashEmbedder)
    assert engine._embedder is not None
    # index_all() on an empty store should return (0, 0) without raising
    indexed, already_had = engine.index_all()
    assert indexed == 0
    assert already_had == 0


def test_index_all_skips_freshly_indexed_nodes() -> None:
    """Second call to index_all() skips all nodes whose indexed_hash matches source_hash."""
    store = GraphStore(":memory:")
    FolderTreeSource(FIXTURE_TREE).import_into(store)
    engine = ContextEngine(store, embedder=HashEmbedder(dim=store.embed_dim))

    indexed_1, already_fresh_1 = engine.index_all()
    assert indexed_1 > 0

    indexed_2, already_fresh_2 = engine.index_all()
    assert indexed_2 == 0
    assert already_fresh_2 > 0
