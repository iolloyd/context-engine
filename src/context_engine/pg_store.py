"""Postgres + pgvector graph store.

Sibling of :class:`context_engine.store.GraphStore` with the same public
interface — see ADR 0002 for the rationale. Connects directly to brain's
Supabase Postgres so the engine and brain share one source of truth.

The class is import-time optional: the ``psycopg`` and ``pgvector``
dependencies live in the ``postgres`` extra. Importing this module
without that extra raises ``ImportError`` at the import site, which
``test_store_parity`` catches and turns into a skip.

Two operating modes:

* ``schema_isolation=None`` (production) — connects to the configured
  schema (default ``public``) and assumes the migration in
  ``brain/supabase/migrations/20260513120000_cx_graph_tables.sql`` has
  already created ``cx_nodes`` and ``cx_edges``.

* ``schema_isolation="<prefix>"`` (tests) — creates a uniquely named
  schema ``cx_<prefix>_<uuid8>``, writes the DDL into it, and drops the
  schema on ``close()``. Lets parity tests run against a real Postgres
  without polluting any shared schema.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from context_engine.types import Edge, Node


def _json_default(value: Any) -> Any:
    """Mirror ``store._json_default`` so metadata coercion is identical
    across backends — YAML date/datetime values round-trip as ISO strings."""
    if isinstance(value, datetime | date):
        return value.isoformat()
    return str(value)


def _dumps(value: Any) -> str:
    return json.dumps(value, default=_json_default)


def _jsonb(value: dict[str, Any]) -> Jsonb:
    return Jsonb(value, dumps=_dumps)


class PgGraphStore:
    """Postgres-backed graph store. Single connection, autocommit by default;
    :meth:`transaction` opens an explicit BEGIN/COMMIT block."""

    def __init__(
        self,
        dsn: str,
        embed_dim: int = 512,
        schema_isolation: str | None = None,
        schema: str = "public",
    ) -> None:
        self.embed_dim = embed_dim
        self._conn = psycopg.connect(dsn, autocommit=True)
        register_vector(self._conn)
        if schema_isolation is not None:
            self._schema = f"cx_{schema_isolation}_{uuid.uuid4().hex[:8]}"
            self._owns_schema = True
            self._create_isolated_schema()
        else:
            self._schema = schema
            self._owns_schema = False
        self._nodes_tbl = f'"{self._schema}".cx_nodes'
        self._edges_tbl = f'"{self._schema}".cx_edges'

    # ── lifecycle ───────────────────────────────────────────────────────────

    def _create_isolated_schema(self) -> None:
        """Create the test schema and the two tables. Mirrors the production
        migration but without RLS — isolation is provided by the unique
        schema name, and tests connect with a role that owns it."""
        ddl = f"""
            CREATE SCHEMA "{self._schema}";

            CREATE TABLE "{self._schema}".cx_nodes (
                id         text PRIMARY KEY,
                content    text NOT NULL,
                metadata   jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                embedding  vector({self.embed_dim}),
                created_at timestamptz NOT NULL DEFAULT now(),
                updated_at timestamptz NOT NULL DEFAULT now()
            );

            CREATE INDEX ON "{self._schema}".cx_nodes
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            CREATE INDEX ON "{self._schema}".cx_nodes USING gin (metadata);
            CREATE INDEX ON "{self._schema}".cx_nodes ((metadata->>'type'));
            CREATE INDEX ON "{self._schema}".cx_nodes ((metadata->>'status'));

            CREATE TABLE "{self._schema}".cx_edges (
                id         text PRIMARY KEY,
                source     text NOT NULL REFERENCES "{self._schema}".cx_nodes(id) ON DELETE CASCADE,
                target     text NOT NULL REFERENCES "{self._schema}".cx_nodes(id) ON DELETE CASCADE,
                content    text,
                metadata   jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                created_at timestamptz NOT NULL DEFAULT now(),
                updated_at timestamptz NOT NULL DEFAULT now()
            );

            CREATE INDEX ON "{self._schema}".cx_edges (source);
            CREATE INDEX ON "{self._schema}".cx_edges (target);
            CREATE INDEX ON "{self._schema}".cx_edges ((metadata->>'type'));
        """
        with self._conn.cursor() as cur:
            cur.execute(ddl)

    def close(self) -> None:
        if self._owns_schema:
            with self._conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA IF EXISTS "{self._schema}" CASCADE')
        self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[Any]:
        with self._conn.transaction():
            yield self._conn

    # ── nodes ───────────────────────────────────────────────────────────────

    def upsert_node(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        node_id: str | None = None,
        embedding: list[float] | None = None,
    ) -> Node:
        node_id = node_id or str(uuid.uuid4())
        meta = metadata or {}
        if embedding is not None and len(embedding) != self.embed_dim:
            raise ValueError(f"expected dim {self.embed_dim}, got {len(embedding)}")
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {self._nodes_tbl} (id, content, metadata, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    content   = EXCLUDED.content,
                    metadata  = EXCLUDED.metadata,
                    embedding = COALESCE(EXCLUDED.embedding, {self._nodes_tbl}.embedding)
                """,
                (node_id, content, _jsonb(meta), embedding),
            )
        return self.get_node(node_id)  # type: ignore[return-value]

    def get_node(self, node_id: str) -> Node | None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT id, content, metadata, created_at, updated_at "
                f"FROM {self._nodes_tbl} WHERE id = %s",
                (node_id,),
            )
            row = cur.fetchone()
        return _row_to_node(row) if row else None

    def get_nodes(self, node_ids: Iterable[str]) -> list[Node]:
        ids = list(node_ids)
        if not ids:
            return []
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT id, content, metadata, created_at, updated_at "
                f"FROM {self._nodes_tbl} WHERE id = ANY(%s)",
                (ids,),
            )
            rows = cur.fetchall()
        return [_row_to_node(r) for r in rows]

    def delete_node(self, node_id: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self._nodes_tbl} WHERE id = %s", (node_id,))

    def filter_nodes(self, where: dict[str, Any]) -> list[Node]:
        with self._conn.cursor() as cur:
            if not where:
                cur.execute(
                    f"SELECT id, content, metadata, created_at, updated_at "
                    f"FROM {self._nodes_tbl}"
                )
            else:
                cur.execute(
                    f"SELECT id, content, metadata, created_at, updated_at "
                    f"FROM {self._nodes_tbl} WHERE metadata @> %s",
                    (_jsonb(where),),
                )
            rows = cur.fetchall()
        return [_row_to_node(r) for r in rows]

    # ── edges ───────────────────────────────────────────────────────────────

    def upsert_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        weight: float = 0.5,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
        edge_id: str | None = None,
    ) -> Edge:
        edge_id = edge_id or str(uuid.uuid4())
        meta = {"type": edge_type, "weight": weight, **(metadata or {})}
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {self._edges_tbl} (id, source, target, content, metadata)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    source   = EXCLUDED.source,
                    target   = EXCLUDED.target,
                    content  = EXCLUDED.content,
                    metadata = EXCLUDED.metadata
                """,
                (edge_id, source, target, content, _jsonb(meta)),
            )
        return self.get_edge(edge_id)  # type: ignore[return-value]

    def get_edge(self, edge_id: str) -> Edge | None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT id, source, target, content, metadata, created_at, updated_at "
                f"FROM {self._edges_tbl} WHERE id = %s",
                (edge_id,),
            )
            row = cur.fetchone()
        return _row_to_edge(row) if row else None

    def out_edges(
        self,
        node_id: str,
        edge_types: list[str] | None = None,
        weight_floor: float = 0.0,
    ) -> list[Edge]:
        return self._adjacency("source", node_id, edge_types, weight_floor)

    def in_edges(
        self,
        node_id: str,
        edge_types: list[str] | None = None,
        weight_floor: float = 0.0,
    ) -> list[Edge]:
        return self._adjacency("target", node_id, edge_types, weight_floor)

    def _adjacency(
        self,
        column: str,
        node_id: str,
        edge_types: list[str] | None,
        weight_floor: float,
    ) -> list[Edge]:
        sql = (
            f"SELECT id, source, target, content, metadata, created_at, updated_at "
            f"FROM {self._edges_tbl} "
            f"WHERE {column} = %s "
            f"  AND COALESCE((metadata->>'weight')::float, 0.5) >= %s "
            f"  AND (%s::text[] IS NULL OR metadata->>'type' = ANY(%s))"
        )
        with self._conn.cursor() as cur:
            cur.execute(sql, (node_id, weight_floor, edge_types, edge_types))
            rows = cur.fetchall()
        return [_row_to_edge(r) for r in rows]

    def update_edge_weight(self, edge_id: str, new_weight: float) -> None:
        clamped = max(0.0, min(1.0, new_weight))
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {self._edges_tbl}
                   SET metadata = jsonb_set(metadata, '{{weight}}', to_jsonb(%s::float))
                 WHERE id = %s
                """,
                (clamped, edge_id),
            )

    # ── embeddings / seed search ────────────────────────────────────────────

    def has_embedding(self, node_id: str) -> bool:
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT embedding IS NOT NULL FROM {self._nodes_tbl} WHERE id = %s",
                (node_id,),
            )
            row = cur.fetchone()
        return bool(row and row[0])

    def get_embedding(self, node_id: str) -> list[float] | None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT embedding FROM {self._nodes_tbl} WHERE id = %s",
                (node_id,),
            )
            row = cur.fetchone()
        if row is None or row[0] is None:
            return None
        return list(row[0])

    def knn(self, vec: list[float], k: int = 5) -> list[tuple[str, float]]:
        if len(vec) != self.embed_dim:
            raise ValueError(f"expected dim {self.embed_dim}, got {len(vec)}")
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, embedding <=> %s::vector AS distance
                  FROM {self._nodes_tbl}
                 WHERE embedding IS NOT NULL
              ORDER BY embedding <=> %s::vector
                 LIMIT %s
                """,
                (vec, vec, k),
            )
            rows = cur.fetchall()
        return [(r[0], float(r[1])) for r in rows]

    # ── inventory ──────────────────────────────────────────────────────────

    def all_node_ids(self) -> list[str]:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT id FROM {self._nodes_tbl}")
            return [r[0] for r in cur.fetchall()]

    def all_edges(self) -> list[Edge]:
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT id, source, target, content, metadata, created_at, updated_at "
                f"FROM {self._edges_tbl}"
            )
            rows = cur.fetchall()
        return [_row_to_edge(r) for r in rows]

    def node_count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self._nodes_tbl}")
            row = cur.fetchone()
        return int(row[0]) if row else 0

    def edge_count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self._edges_tbl}")
            row = cur.fetchone()
        return int(row[0]) if row else 0


# ── row adapters ────────────────────────────────────────────────────────────


def _row_to_node(row: tuple[Any, ...]) -> Node:
    return Node(
        id=row[0],
        content=row[1],
        metadata=row[2] if isinstance(row[2], dict) else json.loads(row[2]),
        created_at=_as_utc(row[3]),
        updated_at=_as_utc(row[4]),
    )


def _row_to_edge(row: tuple[Any, ...]) -> Edge:
    return Edge(
        id=row[0],
        source=row[1],
        target=row[2],
        content=row[3],
        metadata=row[4] if isinstance(row[4], dict) else json.loads(row[4]),
        created_at=_as_utc(row[5]),
        updated_at=_as_utc(row[6]),
    )


def _as_utc(value: datetime) -> datetime:
    """Postgres returns timezone-aware ``datetime``; normalise to UTC to
    match the SQLite store's ``datetime.now(UTC).isoformat()`` round-trip."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
