"""HTTP service exposing the engine's ``/recall`` surface.

Framebrain calls this from brain's edge functions (or directly from the
MCP server) when a query needs graph-aware context, not just flat vector
similarity. The service composes seed selection → traversal → optional
logic → synthesis and returns the slice plus the synthesised text.

Single endpoint plus a health check. No auth — runs inside the
framebrain trust boundary; expose via brain's edge functions or a
sidecar reverse proxy if external traffic is involved.

Run with::

    CTX_PG_DSN=postgresql://... uvicorn context_engine.server:app

or via the module entry point::

    python -m context_engine.server --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from context_engine.engine import ContextEngine, EngineResponse
from context_engine.pg_store import PgGraphStore
from context_engine.types import ContextSlice, Query

log = logging.getLogger(__name__)


class RecallRequest(BaseModel):
    query: str = Field(..., description="Natural-language query text")
    explicit_refs: list[str] = Field(
        default_factory=list,
        description="Node ids the caller already knows are relevant",
    )
    conversation_node_ids: list[str] = Field(
        default_factory=list,
        description="Node ids surfaced earlier in the same conversation",
    )
    metadata_filter: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata equality filter applied to seed selection",
    )


class NodePayload(BaseModel):
    id: str
    content: str
    metadata: dict[str, Any]


class EdgePayload(BaseModel):
    id: str
    source: str
    target: str
    type: str
    weight: float
    content: str | None


class SlicePayload(BaseModel):
    nodes: list[NodePayload]
    edges: list[EdgePayload]
    seeds: list[str]
    notes: list[str]


class RecallResponse(BaseModel):
    text: str
    intent: str
    focus: str
    slice: SlicePayload
    needed_widen: bool
    gap_flag: bool
    confidence: float


def _slice_to_payload(slice_: ContextSlice) -> SlicePayload:
    return SlicePayload(
        nodes=[
            NodePayload(id=n.id, content=n.content, metadata=n.metadata)
            for n in slice_.nodes
        ],
        edges=[
            EdgePayload(
                id=e.id,
                source=e.source,
                target=e.target,
                type=e.type,
                weight=e.weight,
                content=e.content,
            )
            for e in slice_.edges
        ],
        seeds=slice_.seeds,
        notes=slice_.notes,
    )


def _response_to_payload(response: EngineResponse) -> RecallResponse:
    return RecallResponse(
        text=response.text,
        intent=response.tuple_.intent.value,
        focus=response.tuple_.focus.value,
        slice=_slice_to_payload(response.slice_),
        needed_widen=response.needed_widen,
        gap_flag=response.gap_flag,
        confidence=response.confidence,
    )


def build_app(engine: ContextEngine) -> FastAPI:
    """Construct a FastAPI app bound to *engine*.

    Factored out so tests can mount the same routes against an
    in-memory SQLite engine without reading environment variables.
    """

    @asynccontextmanager
    async def lifespan(_app: FastAPI):  # type: ignore[no-untyped-def]
        yield
        if hasattr(engine.store, "close"):
            engine.store.close()

    app = FastAPI(title="context-engine", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "node_count": engine.store.node_count(),
            "edge_count": engine.store.edge_count(),
        }

    @app.post("/recall", response_model=RecallResponse)
    def recall(request: RecallRequest) -> RecallResponse:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="query must be non-empty")
        query = Query(
            text=request.query,
            explicit_refs=request.explicit_refs,
            conversation_node_ids=request.conversation_node_ids,
        )
        response = engine.answer(query, metadata_filter=request.metadata_filter)
        return _response_to_payload(response)

    return app


def _build_default_app() -> FastAPI:
    dsn = os.environ.get("CTX_PG_DSN")
    if not dsn:
        raise RuntimeError("CTX_PG_DSN must be set to run the server")
    embed_dim = int(os.environ.get("CTX_EMBED_DIM", "512"))
    store = PgGraphStore(dsn=dsn, embed_dim=embed_dim)
    engine = ContextEngine(store=store)
    # Fail fast on embedder/store dim mismatch — without this guard, a
    # misconfigured deploy (e.g. VOYAGE_API_KEY missing → 384-dim fallback
    # against a 512-dim Postgres column) would only surface on the first
    # user /recall, after CDN propagation, after the user has already
    # blamed the integration. Better to refuse to come up.
    try:
        engine.boot_check_embed_dim(store.embed_dim)
    except RuntimeError as exc:
        log.error("context-engine: %s", exc)
        raise SystemExit(2) from exc
    return build_app(engine)


# Module-level ``app`` for ``uvicorn context_engine.server:app``. Lazy so the
# DSN check only fires when something actually starts the server.
def __getattr__(name: str) -> Any:
    if name == "app":
        return _build_default_app()
    raise AttributeError(name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="context-engine-server")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "info"))
    args = parser.parse_args(argv)

    import uvicorn

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    uvicorn.run(_build_default_app(), host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
