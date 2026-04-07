"""Embedder protocol and implementations.

Three implementations ship out of the box:

- ``HashEmbedder`` — deterministic, offline, for tests.  No network, no deps
  beyond stdlib.  Default dim=384 matches ``GraphStore``'s default.
- ``AnthropicEmbedder`` — thin wrapper around the Voyage AI HTTP API.
  Lazy urllib.request import; zero extra deps.  Env vars: ``VOYAGE_API_KEY``
  (or ``ANTHROPIC_API_KEY`` as fallback), ``CONTEXT_ENGINE_EMBED_MODEL``
  (default ``voyage-3-lite``, 512-dim).  The caller must construct
  ``GraphStore(embed_dim=512)`` when using this embedder.
- ``SentenceTransformerEmbedder`` — local semantic embedder using the
  ``sentence-transformers`` optional package.  384-dim, CPU-fast, no API key.
  Install with: ``pip install 'context-engine[embeddings]'``.

``default_embedder()`` returns ``SentenceTransformerEmbedder`` when the package
is available, falling back to ``HashEmbedder`` with a one-time stderr warning.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Single-method protocol: map text to a float vector."""

    def embed(self, text: str) -> list[float]: ...


class EmbedderUnavailable(RuntimeError):
    """Raised when an embedder cannot be constructed (missing package, missing API key, etc.)."""


class HashEmbedder:
    """Deterministic, offline embedder.

    Algorithm:
      1. Tokenise: split on non-alphanumeric characters, lowercase, drop empties.
      2. For each token hash with blake2b (digest_size=8) → 8-byte digest →
         interpret as a little-endian uint64 → bucket index ``% dim``.
      3. Increment that bucket (accumulate counts).
      4. L2-normalise.  Empty input → zero vector.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        tokens = [t for t in re.split(r"[^a-zA-Z0-9]+", text.lower()) if t]
        if not tokens:
            return [0.0] * self.dim

        vec = [0.0] * self.dim
        for token in tokens:
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            seed = int.from_bytes(digest, "little")
            vec[seed % self.dim] += 1.0

        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec]


class AnthropicEmbedder:
    """Voyage AI embeddings via direct HTTP (stdlib urllib.request; no extra deps).

    Env vars:
      - ``VOYAGE_API_KEY`` (preferred) or ``ANTHROPIC_API_KEY`` — authentication.
      - ``CONTEXT_ENGINE_EMBED_MODEL`` — model name (default ``voyage-3-lite``).

    voyage-3-lite produces 512-dimensional vectors.  Construct
    ``GraphStore(embed_dim=512)`` when wiring this embedder.
    """

    ENDPOINT = "https://api.voyageai.com/v1/embeddings"
    DEFAULT_MODEL = "voyage-3-lite"

    def __init__(self) -> None:
        import os

        self._api_key = os.environ.get("VOYAGE_API_KEY") or os.environ.get(
            "ANTHROPIC_API_KEY"
        )
        if not self._api_key:
            raise ValueError("VOYAGE_API_KEY or ANTHROPIC_API_KEY must be set")
        self._model = os.environ.get("CONTEXT_ENGINE_EMBED_MODEL", self.DEFAULT_MODEL)

    def embed(self, text: str) -> list[float]:
        import json
        import urllib.request

        payload = json.dumps({"input": [text], "model": self._model}).encode()
        req = urllib.request.Request(
            self.ENDPOINT,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )
        with urllib.request.urlopen(req) as resp:  # noqa: S310
            data = json.loads(resp.read())
        return data["data"][0]["embedding"]


class SentenceTransformerEmbedder:
    """Local semantic embedder using sentence-transformers.

    Default model: ``sentence-transformers/all-MiniLM-L6-v2`` — 384 dimensions,
    ~80MB, runs on CPU in a few ms per encoding.  No API key, no network after the
    first model download.

    The ``sentence_transformers`` package is imported lazily in ``__init__`` so this
    class can be defined even when the optional dependency is absent.  Attempting to
    construct it without the package installed raises ``EmbedderUnavailable`` with a
    clear install hint.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    DIM = 384

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbedderUnavailable(
                "sentence-transformers not installed; "
                "install with: pipx inject context-engine sentence-transformers "
                "or pip install 'context-engine[embeddings]'"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed(self, text: str) -> list[float]:
        vec = self._model.encode(text, normalize_embeddings=True)
        return [float(x) for x in vec]


_default_warned = False


def default_embedder() -> Embedder:
    """Return the best available embedder.

    Tries ``SentenceTransformerEmbedder`` first; falls back to ``HashEmbedder``
    with a one-time warning printed to stderr so the user knows why semantic quality
    may be poor.
    """
    global _default_warned
    try:
        return SentenceTransformerEmbedder()
    except EmbedderUnavailable as exc:
        if not _default_warned:
            import sys

            print(
                f"context-engine: {exc}\n"
                "               falling back to HashEmbedder "
                "(semantic search will be keyword-based).",
                file=sys.stderr,
            )
            _default_warned = True
        return HashEmbedder()
