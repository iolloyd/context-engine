"""Embedder protocol and implementations.

Two implementations ship out of the box:

- ``HashEmbedder`` — deterministic, offline, for tests.  No network, no deps
  beyond stdlib.  Default dim=384 matches ``GraphStore``'s default.
- ``AnthropicEmbedder`` — thin wrapper around the Voyage AI HTTP API.
  Lazy urllib.request import; zero extra deps.  Env vars: ``VOYAGE_API_KEY``
  (or ``ANTHROPIC_API_KEY`` as fallback), ``CONTEXT_ENGINE_EMBED_MODEL``
  (default ``voyage-3-lite``, 512-dim).  The caller must construct
  ``GraphStore(embed_dim=512)`` when using this embedder.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Single-method protocol: map text to a float vector."""

    def embed(self, text: str) -> list[float]: ...


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
