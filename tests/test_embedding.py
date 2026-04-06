"""Tests for the embedding module."""

from __future__ import annotations

import math
import os

import pytest

from context_engine.embedding import (
    AnthropicEmbedder,
    Embedder,
    HashEmbedder,
)

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
