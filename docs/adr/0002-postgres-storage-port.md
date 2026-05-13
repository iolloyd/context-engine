# ADR 0002 — Postgres storage backend

**Status**: accepted
**Date**: 2026-05-13

## Context

The engine currently stores nodes, edges, and embeddings in SQLite + `sqlite-vec`,
single file, single writer. That choice was right for Phase 0–3: zero ops, local
authoring, fast tests. It is wrong for the product target.

Context-engine is being fused with [`brain`](https://github.com/iolloyd/brain) —
a Supabase-hosted vector memory store — into a single product, **framebrain**.
Brain already runs Postgres + `pgvector` (HNSW indexes, `match_memories` RPC,
domain isolation, Slack and MCP write paths) at production scale. Brain's own
README names its gap: *"Flat memory model. No graph relationships between
memories. Retrieval is purely similarity-based."* Context-engine fills exactly
that gap, but only if the two systems share a database. Two stores synced over
HTTP is the wrong production answer — two writers, drift risk, doubled ops.

The storage layer in this repo is narrow by design. `GraphStore`
(`src/context_engine/store.py`) is the only class that touches SQL. Every other
module — `engine.py`, `traversal.py`, `seeds.py`, `classifier.py`,
`feedback.py`, `synthesiser.py`, `prolog_logic.py` — talks to that interface
and nothing below it. The port is therefore a single-class substitution, not a
rewrite of the engine.

## Decision

Adopt **Postgres + `pgvector` as the production storage backend**. Introduce a
`PgGraphStore` class with the same public interface as `GraphStore`. Both
classes stay supported:

| Store | Use |
|---|---|
| `GraphStore` (SQLite + `sqlite-vec`) | Local authoring, offline tests, single-tenant trees |
| `PgGraphStore` (Postgres + `pgvector`) | Framebrain production, multi-tenant deployments, shared writes from brain |

Selection is by constructor: `engine.load(store=PgGraphStore(dsn=...))` or the
existing `GraphStore(path=...)`. The CLI gains a `--pg <dsn>` flag that, when
present, instantiates the Postgres store instead of opening the SQLite file at
`--db`.

The engine layer is unchanged. The folder-tree source-of-truth model from
[ADR 0001](0001-knowledge-source.md) is unchanged. SQLite remains the default
for `ctx import`/`ctx export` and for the test suite. Postgres is opt-in via
DSN.

## Rationale

**`GraphStore` is the only porting boundary.** The interface is small and
already abstract: `upsert_node`, `get_node`, `filter_nodes`, `upsert_edge`,
`out_edges`/`in_edges`, `update_edge_weight`, `knn`. Every traversal,
classifier, and feedback path calls into these methods and nothing more.
A second implementation against the same interface does not perturb the
engine.

**Brain already runs the stack.** Postgres, `pgvector`, HNSW indexes,
domain-partitioned similarity search, and migration tooling all exist in brain
today. Re-using brain's Supabase project means we inherit backups, monitoring,
RLS, secret management, and the operational muscle that the team already has.
We do not stand up a new database tier.

**Single source of truth.** Brain writes loose memories; context-engine
traverses typed edges; framebrain serves the combined slice. Sharing one
Postgres lets brain's `docs` table feed nodes directly into the graph layer
without a synchronisation job. A SQLite sidecar would force two writers and a
constant export/import loop.

**Concurrent writers, online migrations, RLS.** SQLite's single-writer model
breaks the moment two services (brain's edge functions and the engine sidecar)
both write. Multi-tenant framebrain also needs row-level access controls.
These are Postgres-shaped problems, not SQLite-shaped problems.

**Embedding dimension already negotiated.** Brain's embedding provider is
configurable (Voyage 512, OpenAI 1536, Gemini 768); switching providers
re-embeds via `scripts/switch-provider.sh`. Aligning context-engine's seed
embeddings to the same provider is a config change, not new infrastructure.

## Consequences

- New module `src/context_engine/pg_store.py` implementing the `GraphStore`
  public interface against `psycopg` and `pgvector`. The class is import-time
  optional — `psycopg` is an `[postgres]` extra, mirroring how
  `sentence-transformers` is gated behind `[embeddings]`.
- The existing `GraphStore` class stays. Tests run both backends through the
  same parity suite (`tests/test_store_parity.py`) so divergence is caught at
  CI time.
- Embedding alignment: the default embedder for `PgGraphStore` is the same
  provider brain is configured for (Voyage by default). `HashEmbedder` and the
  sentence-transformers path remain for offline-only test runs.
- Deployment shape: the engine runs as a Python service connecting directly to
  brain's Supabase Postgres. Brain's Deno edge functions are untouched. A new
  framebrain endpoint (`/recall+graph`) proxies to the engine service or
  routes via MCP — that wiring is a separate ADR.
- Schema lives in a new migration set in `brain/supabase/migrations/` (because
  brain owns the database). The migration adds `cx_nodes`, `cx_edges`, and a
  `cx_node_embeddings` table parallel to brain's existing `memories` / `docs`
  / `doc_links` tables. A one-shot importer reads `docs` and `doc_links` into
  `cx_nodes` and `cx_edges`. Brain's existing CLI and Slack flows are
  unmodified.
- Round-trip guarantee from ADR 0001 (`import(export(tree)) == tree`) is
  enforced for both stores. The Postgres backend must satisfy the same
  invariant.

## Schema sketch

```sql
-- pgvector dimension matches the brain embedding provider in use.
-- For Voyage voyage-3-lite this is 512; rebind via migration when switching.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE cx_nodes (
    id          text PRIMARY KEY,
    content     text NOT NULL,
    metadata    jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX cx_nodes_type_idx   ON cx_nodes ((metadata->>'type'));
CREATE INDEX cx_nodes_status_idx ON cx_nodes ((metadata->>'status'));

CREATE TABLE cx_edges (
    id          text PRIMARY KEY,
    source      text NOT NULL REFERENCES cx_nodes(id) ON DELETE CASCADE,
    target      text NOT NULL REFERENCES cx_nodes(id) ON DELETE CASCADE,
    content     text,
    metadata    jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX cx_edges_source_idx ON cx_edges (source);
CREATE INDEX cx_edges_target_idx ON cx_edges (target);
CREATE INDEX cx_edges_type_idx   ON cx_edges ((metadata->>'type'));

CREATE TABLE cx_node_embeddings (
    node_id   text PRIMARY KEY REFERENCES cx_nodes(id) ON DELETE CASCADE,
    embedding vector(512) NOT NULL
);

CREATE INDEX cx_node_embeddings_hnsw
    ON cx_node_embeddings
    USING hnsw (embedding vector_cosine_ops);
```

`knn` becomes an `ORDER BY embedding <=> $1 LIMIT k` against the HNSW index.
`out_edges`/`in_edges` are straightforward indexed lookups with optional
type/weight filters in the `WHERE` clause. `update_edge_weight` is a `jsonb`
field update.

## Interface contract

`PgGraphStore` must satisfy:

```python
class StoreProtocol(Protocol):
    def upsert_node(self, content: str, metadata: dict, node_id: str | None,
                    embedding: list[float] | None) -> Node: ...
    def get_node(self, node_id: str) -> Node | None: ...
    def get_nodes(self, ids: Iterable[str]) -> list[Node]: ...
    def delete_node(self, node_id: str) -> None: ...
    def filter_nodes(self, where: dict) -> list[Node]: ...
    def upsert_edge(self, source: str, target: str, edge_type: str,
                    weight: float, content: str | None, metadata: dict,
                    edge_id: str | None) -> Edge: ...
    def get_edge(self, edge_id: str) -> Edge | None: ...
    def out_edges(self, node_id: str, edge_types: list[str] | None,
                  weight_floor: float) -> list[Edge]: ...
    def in_edges(self, node_id: str, edge_types: list[str] | None,
                 weight_floor: float) -> list[Edge]: ...
    def update_edge_weight(self, edge_id: str, new_weight: float) -> None: ...
    def has_embedding(self, node_id: str) -> bool: ...
    def get_embedding(self, node_id: str) -> list[float] | None: ...
    def knn(self, vec: list[float], k: int) -> list[tuple[str, float]]: ...
    def all_node_ids(self) -> list[str]: ...
    def all_edges(self) -> list[Edge]: ...
    def node_count(self) -> int: ...
    def edge_count(self) -> int: ...
    @contextmanager
    def transaction(self) -> Iterator[Any]: ...
```

A formal `Protocol` is added to `types.py` and both stores declared to satisfy
it. Parity tests instantiate the same scenario against each store and assert
identical results.

## Alternatives rejected

**Sidecar with SQLite over HTTP.** The engine stays on SQLite, brain exports
docs/links periodically, sync runs in a loop. Two writers, two truths, an
export/import cron, and every operation now touches a network. Brain's docs
get edited from MCP, Slack, and CLI; reflecting those into SQLite without race
conditions is its own project.

**Port the engine layer to Deno alongside brain's edge functions.** Single
deployment, but rewriting the Prolog logic engine, the LLM classifier, the
feedback loop, and the traversal in TypeScript is months of work for no
production gain. Python is also the right ecosystem for the LLM and logic
pieces.

**Supabase Python edge functions.** Now generally available but cold-start
heavy, and the ecosystem for the dependencies we need (Prolog bridge,
sentence-transformers fallback, psycopg) is less mature than running our own
container.

**Unify schema by replacing brain's `docs` / `doc_links` directly.** The right
long-term shape, but it forces brain's CLI, Slack handlers, and MCP server to
change in lockstep. Parallel `cx_nodes` / `cx_edges` tables let both worlds
coexist; behaviour migrates incrementally. A future ADR can collapse the
schemas once the graph layer is load-bearing.

**Cloud-hosted graph databases (Neo4j, TigerGraph, Neptune).** Add a third
data tier and an ops surface brain does not already carry. The graph is small
(thousands of nodes, not millions), `pgvector` already serves the seed
selection well, and Postgres handles the traversal queries fine at this scale.
