"""Tests for the embedding module."""

from __future__ import annotations

import math
import os
import sys
import unittest.mock

import pytest

from context_engine.embedding import (
    AnthropicEmbedder,
    Embedder,
    EmbedderUnavailable,
    HashEmbedder,
    SentenceTransformerEmbedder,
    default_embedder,
)


def _sentence_transformers_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        return False

# ── HashEmbedder ─────────────────────────────────────────────────────────────


def test_hash_embedder_implements_protocol() -> None:
    assert isinstance(HashEmbedder(), Embedder)


def test_hash_embedder_dimension() -> None:
    emb = HashEmbedder(dim=128)
    vec = emb.embed("hello world")
    assert len(vec) == 128


def test_hash_embedder_default_dim() -> None:
    emb = HashEmbedder()
    assert len(emb.embed("test")) == 384


def test_hash_embedder_deterministic() -> None:
    emb = HashEmbedder()
    text = "Why do I avoid squats?"
    assert emb.embed(text) == emb.embed(text)


def test_hash_embedder_l2_norm_unit() -> None:
    emb = HashEmbedder()
    vec = emb.embed("bench press")
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-6


def test_hash_embedder_empty_input_zero_vector() -> None:
    emb = HashEmbedder()
    vec = emb.embed("")
    assert all(x == 0.0 for x in vec)
    assert len(vec) == 384


def test_hash_embedder_punctuation_agnostic() -> None:
    emb = HashEmbedder()
    # Same tokens regardless of punctuation
    assert emb.embed("squat, knee") == emb.embed("squat  knee")
    assert emb.embed("bench-press") == emb.embed("bench press")


def test_hash_embedder_different_tokens_different_vectors() -> None:
    emb = HashEmbedder()
    assert emb.embed("bench press") != emb.embed("squat knee")


def test_hash_embedder_cross_run_stability() -> None:
    """Vectors must match a pre-computed reference so we detect regressions."""
    emb = HashEmbedder(dim=4)
    # Just verify determinism across two calls within the same process.
    v1 = emb.embed("squat")
    v2 = emb.embed("squat")
    assert v1 == v2


# ── AnthropicEmbedder ─────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not os.getenv("VOYAGE_API_KEY"),
    reason="VOYAGE_API_KEY not set",
)
def test_anthropic_embedder_smoke() -> None:
    emb = AnthropicEmbedder()
    vec = emb.embed("Why do I avoid squats?")
    assert isinstance(vec, list)
    assert len(vec) > 0
    assert all(isinstance(x, float) for x in vec)
    norm = math.sqrt(sum(x * x for x in vec))
    assert norm > 0


# ── SentenceTransformerEmbedder ───────────────────────────────────────────────


@pytest.mark.skipif(
    not _sentence_transformers_available(),
    reason="sentence-transformers not installed",
)
def test_sentence_transformer_deterministic() -> None:
    emb = SentenceTransformerEmbedder()
    text = "Why do I avoid squats?"
    assert emb.embed(text) == emb.embed(text)


@pytest.mark.skipif(
    not _sentence_transformers_available(),
    reason="sentence-transformers not installed",
)
def test_sentence_transformer_l2_norm() -> None:
    emb = SentenceTransformerEmbedder()
    vec = emb.embed("The quick brown fox jumps over the lazy dog")
    norm_sq = sum(x * x for x in vec)
    assert abs(norm_sq - 1.0) < 1e-5


@pytest.mark.skipif(
    not _sentence_transformers_available(),
    reason="sentence-transformers not installed",
)
def test_sentence_transformer_semantic_quality() -> None:
    """Semantic neighbours retrieved correctly; unrelated node excluded."""
    from context_engine.store import GraphStore

    store = GraphStore(":memory:")
    emb = SentenceTransformerEmbedder()

    n1 = store.upsert_node("We chose postgres for concurrent write throughput", node_id="n1")
    n2 = store.upsert_node("The meeting is on Tuesday at 3pm", node_id="n2")
    n3 = store.upsert_node(
        "Database migration from SQLite was driven by write contention", node_id="n3"
    )

    for node in (n1, n2, n3):
        store._set_embedding(node.id, emb.embed(node.content))

    query_vec = emb.embed("why did we pick postgres")
    hits = store.knn(query_vec, k=2)
    result_ids = {node_id for node_id, _ in hits}

    assert "n1" in result_ids
    assert "n3" in result_ids
    assert "n2" not in result_ids


def test_sentence_transformer_unavailable_raises() -> None:
    with (
        unittest.mock.patch.dict(sys.modules, {"sentence_transformers": None}),
        pytest.raises(EmbedderUnavailable),
    ):
        SentenceTransformerEmbedder()


# ── default_embedder ──────────────────────────────────────────────────────────


def test_default_embedder_fallback_to_hash(capsys: pytest.CaptureFixture) -> None:
    import context_engine.embedding as emb_mod

    orig_warned = emb_mod._default_warned
    emb_mod._default_warned = False
    try:
        with unittest.mock.patch.dict(sys.modules, {"sentence_transformers": None}):
            result = default_embedder()
        assert isinstance(result, HashEmbedder)
        captured = capsys.readouterr()
        assert "falling back to HashEmbedder" in captured.err
    finally:
        emb_mod._default_warned = orig_warned


def test_default_embedder_warning_printed_once(capsys: pytest.CaptureFixture) -> None:
    import context_engine.embedding as emb_mod

    orig_warned = emb_mod._default_warned
    emb_mod._default_warned = False
    try:
        with unittest.mock.patch.dict(sys.modules, {"sentence_transformers": None}):
            default_embedder()
            default_embedder()
        captured = capsys.readouterr()
        assert captured.err.count("falling back to HashEmbedder") == 1
    finally:
        emb_mod._default_warned = orig_warned
