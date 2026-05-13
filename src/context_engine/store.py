"""SQLite-backed graph store.

Two tables — `nodes` and `edges` — plus a `sqlite-vec` virtual table for seed
selection by vector similarity. Metadata is stored as JSON. Indexes cover the
hot paths: lookup by type/status and adjacency by source/target.
"""

from __future__ import annotations

import json
import sqlite3
import struct
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from context_engine.types import Edge, Node

EMBED_DIM = 384  # default; override via GraphStore(embed_dim=...)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json_default(value: Any) -> Any:
    """Coerce metadata values that are not natively JSON-serialisable.

    YAML frontmatter parses ``2026-02-14`` as :class:`datetime.date` and
    ``2026-02-14T10:00:00`` as :class:`datetime.datetime`; both land here.
    Any other non-scalar value is stringified as a last resort so a single
    bad key cannot break an entire import.
    """
    if isinstance(value, datetime | date):
        return value.isoformat()
    return str(value)


def _dumps(value: Any) -> str:
    return json.dumps(value, default=_json_default)


def _pack_embedding(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack_embedding(blob: bytes, dim: int) -> list[float]:
    return list(struct.unpack(f"{dim}f", blob))


class GraphStore:
    """SQLite graph store. Thread-safe for single-writer use."""

    def __init__(self, path: str | Path = ":memory:", embed_dim: int = EMBED_DIM) -> None:
        self.path = str(path)
        self.embed_dim = embed_dim
        self._conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._load_vec_extension()
        self._create_schema()

    # ── lifecycle ───────────────────────────────────────────────────────────

    def _load_vec_extension(self) -> None:
        try:
            import sqlite_vec

            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            self._vec_enabled = True
        except (ImportError, sqlite3.OperationalError):
            # sqlite-vec unavailable → fall back to brute-force cosine in Python
            self._vec_enabled = False

    def _create_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                id          TEXT PRIMARY KEY,
                content     TEXT NOT NULL,
                metadata    TEXT NOT NULL DEFAULT '{}',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS edges (
                id          TEXT PRIMARY KEY,
                source      TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                target      TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                content     TEXT,
                metadata    TEXT NOT NULL DEFAULT '{}',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);

            CREATE TABLE IF NOT EXISTS node_embeddings (
                node_id     TEXT PRIMARY KEY REFERENCES nodes(id) ON DELETE CASCADE,
                embedding   BLOB NOT NULL
            );
            """
        )
        if self._vec_enabled:
            cur.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS node_vec USING vec0("
                f"node_id TEXT PRIMARY KEY, embedding FLOAT[{self.embed_dim}])"
            )

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Cursor]:
        cur = self._conn.cursor()
        cur.execute("BEGIN")
        try:
            yield cur
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise

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
        now = _now()
        self._conn.execute(
            """
            INSERT INTO nodes (id, content, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                content=excluded.content,
                metadata=excluded.metadata,
                updated_at=excluded.updated_at
            """,
            (node_id, content, _dumps(meta), now, now),
        )
        if embedding is not None:
            self._set_embedding(node_id, embedding)
        return self.get_node(node_id)  # type: ignore[return-value]

    def get_node(self, node_id: str) -> Node | None:
        row = self._conn.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()
        return _row_to_node(row) if row else None

    def get_nodes(self, node_ids: Iterable[str]) -> list[Node]:
        ids = list(node_ids)
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"SELECT * FROM nodes WHERE id IN ({placeholders})", ids
        ).fetchall()
        return [_row_to_node(r) for r in rows]

    def delete_node(self, node_id: str) -> None:
        self._conn.execute("DELETE FROM nodes WHERE id=?", (node_id,))

    def filter_nodes(self, where: dict[str, Any]) -> list[Node]:
        """Return all nodes whose metadata matches every key/value in `where`.

        Uses JSON1 functions for simple equality on scalar metadata keys.
        """
        if not where:
            rows = self._conn.execute("SELECT * FROM nodes").fetchall()
            return [_row_to_node(r) for r in rows]

        clauses = []
        params: list[Any] = []
        for key, value in where.items():
            clauses.append(f"json_extract(metadata, '$.{key}') = ?")
            params.append(value)
        sql = f"SELECT * FROM nodes WHERE {' AND '.join(clauses)}"
        rows = self._conn.execute(sql, params).fetchall()
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
        now = _now()
        self._conn.execute(
            """
            INSERT INTO edges (id, source, target, content, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                source=excluded.source,
                target=excluded.target,
                content=excluded.content,
                metadata=excluded.metadata,
                updated_at=excluded.updated_at
            """,
            (edge_id, source, target, content, _dumps(meta), now, now),
        )
        return self.get_edge(edge_id)  # type: ignore[return-value]

    def get_edge(self, edge_id: str) -> Edge | None:
        row = self._conn.execute("SELECT * FROM edges WHERE id=?", (edge_id,)).fetchone()
        return _row_to_edge(row) if row else None

    def out_edges(
        self, node_id: str, edge_types: list[str] | None = None, weight_floor: float = 0.0
    ) -> list[Edge]:
        rows = self._conn.execute("SELECT * FROM edges WHERE source=?", (node_id,)).fetchall()
        edges = [_row_to_edge(r) for r in rows]
        return [
            e
            for e in edges
            if e.weight >= weight_floor and (edge_types is None or e.type in edge_types)
        ]

    def in_edges(
        self, node_id: str, edge_types: list[str] | None = None, weight_floor: float = 0.0
    ) -> list[Edge]:
        rows = self._conn.execute("SELECT * FROM edges WHERE target=?", (node_id,)).fetchall()
        edges = [_row_to_edge(r) for r in rows]
        return [
            e
            for e in edges
            if e.weight >= weight_floor and (edge_types is None or e.type in edge_types)
        ]

    def update_edge_weight(self, edge_id: str, new_weight: float) -> None:
        row = self._conn.execute(
            "SELECT metadata FROM edges WHERE id=?", (edge_id,)
        ).fetchone()
        if row is None:
            return
        meta = json.loads(row["metadata"])
        meta["weight"] = max(0.0, min(1.0, new_weight))
        self._conn.execute(
            "UPDATE edges SET metadata=?, updated_at=? WHERE id=?",
            (_dumps(meta), _now(), edge_id),
        )

    # ── embeddings / seed search ────────────────────────────────────────────

    def has_embedding(self, node_id: str) -> bool:
        """Return True if a stored embedding exists for *node_id*."""
        row = self._conn.execute(
            "SELECT 1 FROM node_embeddings WHERE node_id=?", (node_id,)
        ).fetchone()
        return row is not None

    def _set_embedding(self, node_id: str, vec: list[float]) -> None:
        if len(vec) != self.embed_dim:
            raise ValueError(f"expected dim {self.embed_dim}, got {len(vec)}")
        blob = _pack_embedding(vec)
        self._conn.execute(
            "INSERT OR REPLACE INTO node_embeddings (node_id, embedding) VALUES (?, ?)",
            (node_id, blob),
        )
        if self._vec_enabled:
            self._conn.execute(
                "INSERT OR REPLACE INTO node_vec (node_id, embedding) VALUES (?, ?)",
                (node_id, blob),
            )

    def knn(self, vec: list[float], k: int = 5) -> list[tuple[str, float]]:
        if self._vec_enabled:
            rows = self._conn.execute(
                "SELECT node_id, distance FROM node_vec "
                "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (_pack_embedding(vec), k),
            ).fetchall()
            return [(r["node_id"], float(r["distance"])) for r in rows]
        # Fallback: brute-force cosine similarity in Python.
        rows = self._conn.execute("SELECT node_id, embedding FROM node_embeddings").fetchall()
        scored: list[tuple[str, float]] = []
        for r in rows:
            other = _unpack_embedding(r["embedding"], self.embed_dim)
            scored.append((r["node_id"], _cosine_distance(vec, other)))
        scored.sort(key=lambda t: t[1])
        return scored[:k]

    # ── utility ────────────────────────────────────────────────────────────

    def get_embedding(self, node_id: str) -> list[float] | None:
        row = self._conn.execute(
            "SELECT embedding FROM node_embeddings WHERE node_id=?",
            (node_id,),
        ).fetchone()
        if row is None:
            return None
        return _unpack_embedding(row["embedding"], self.embed_dim)

    def all_edges(self) -> list[Edge]:
        rows = self._conn.execute("SELECT * FROM edges").fetchall()
        return [_row_to_edge(r) for r in rows]

    def all_node_ids(self) -> list[str]:
        return [r["id"] for r in self._conn.execute("SELECT id FROM nodes").fetchall()]

    def node_count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) AS c FROM nodes").fetchone()["c"])

    def edge_count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) AS c FROM edges").fetchone()["c"])


# ── row adapters ────────────────────────────────────────────────────────────


def _row_to_node(row: sqlite3.Row) -> Node:
    return Node(
        id=row["id"],
        content=row["content"],
        metadata=json.loads(row["metadata"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _row_to_edge(row: sqlite3.Row) -> Edge:
    return Edge(
        id=row["id"],
        source=row["source"],
        target=row["target"],
        content=row["content"],
        metadata=json.loads(row["metadata"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - dot / (na * nb)
