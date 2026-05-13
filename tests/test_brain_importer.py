"""Integration test for the brain → cx importer.

Sets up a brain-like schema (``nodes`` + ``node_edges``) in an isolated
Postgres schema, seeds a handful of rows, imports into a fresh
``PgGraphStore`` (also isolated), and asserts the resulting cx graph
matches expectations. Skipped without ``CTX_PG_DSN`` because the test
needs a live Postgres with pgvector available.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest

EMBED_DIM = 4

pytest.importorskip("psycopg", reason="postgres extra not installed")
pytest.importorskip("pgvector", reason="postgres extra not installed")

import psycopg  # noqa: E402
from pgvector.psycopg import register_vector  # noqa: E402

from context_engine.brain_importer import import_from_brain  # noqa: E402
from context_engine.pg_store import PgGraphStore  # noqa: E402

DSN = os.environ.get("CTX_PG_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="CTX_PG_DSN not set")


@pytest.fixture
def brain_schema() -> Iterator[str]:
    """Create a unique schema containing brain-like nodes / node_edges
    tables, yield its name, and drop it on teardown."""
    assert DSN is not None
    schema = f"brain_test_{uuid.uuid4().hex[:8]}"
    conn = psycopg.connect(DSN, autocommit=True)
    register_vector(conn)
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS ltree")
            cur.execute(f'CREATE SCHEMA "{schema}"')
            cur.execute(
                f"""
                CREATE TABLE "{schema}".nodes (
                    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                    path       ltree NOT NULL,
                    title      text NOT NULL,
                    content    text NOT NULL DEFAULT '',
                    embedding  vector({EMBED_DIM}),
                    metadata   jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                    domain     text NOT NULL DEFAULT 'general',
                    tier       smallint NOT NULL DEFAULT 3,
                    created_at timestamptz NOT NULL DEFAULT now(),
                    updated_at timestamptz NOT NULL DEFAULT now(),
                    UNIQUE (domain, path)
                )
                """
            )
            cur.execute(
                f"""
                CREATE TABLE "{schema}".node_edges (
                    from_id    uuid NOT NULL REFERENCES "{schema}".nodes(id) ON DELETE CASCADE,
                    to_id      uuid NOT NULL REFERENCES "{schema}".nodes(id) ON DELETE CASCADE,
                    relation   text NOT NULL DEFAULT 'related',
                    metadata   jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at timestamptz NOT NULL DEFAULT now(),
                    PRIMARY KEY (from_id, to_id, relation)
                )
                """
            )
            cur.execute(
                f"""
                CREATE TABLE "{schema}".memories (
                    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                    content    text NOT NULL,
                    embedding  vector({EMBED_DIM}),
                    source     text,
                    tags       text[] DEFAULT '{{}}'::text[],
                    metadata   jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                    domain     text NOT NULL DEFAULT 'general',
                    created_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
        yield schema
    finally:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        conn.close()


@pytest.fixture
def cx_store() -> Iterator[PgGraphStore]:
    assert DSN is not None
    store = PgGraphStore(
        dsn=DSN, embed_dim=EMBED_DIM, schema_isolation="importer"
    )
    try:
        yield store
    finally:
        store.close()


def _seed_brain(schema: str) -> dict[str, str]:
    """Insert a small graph; return {path: uuid_str} so the test can
    reference rows by path."""
    assert DSN is not None
    conn = psycopg.connect(DSN, autocommit=True)
    register_vector(conn)
    ids: dict[str, str] = {}
    try:
        with conn.cursor() as cur:
            rows = [
                ("training", "Training", "Overview", None, "general", 1),
                ("training.legs", "Legs", "Quads and hams", [0.1, 0.0, 0.0, 0.0], "general", 2),
                ("training.legs.pendulum", "Pendulum squat", "Machine alternative",
                 [0.0, 1.0, 0.0, 0.0], "general", 3),
                ("finance.q3", "Q3 review", "", None, "finance", 3),
            ]
            for path, title, content, embedding, domain, tier in rows:
                cur.execute(
                    f"""
                    INSERT INTO "{schema}".nodes (path, title, content, embedding, domain, tier)
                    VALUES (%s::ltree, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (path, title, content, embedding, domain, tier),
                )
                row = cur.fetchone()
                assert row is not None
                ids[path] = str(row[0])

            cur.execute(
                f"""
                INSERT INTO "{schema}".node_edges (from_id, to_id, relation, metadata)
                VALUES
                    (%s, %s, 'references', '{{}}'::jsonb),
                    (%s, %s, 'contains',   '{{}}'::jsonb)
                """,
                (
                    ids["training.legs"], ids["training.legs.pendulum"],
                    ids["training"],      ids["training.legs"],
                ),
            )
    finally:
        conn.close()
    return ids


# ── tests ──────────────────────────────────────────────────────────────────


def test_import_nodes_and_edges(brain_schema: str, cx_store: PgGraphStore) -> None:
    assert DSN is not None
    ids = _seed_brain(brain_schema)
    report = import_from_brain(DSN, cx_store, brain_schema=brain_schema)

    assert report.nodes_imported == 4
    assert report.edges_imported == 2
    assert cx_store.node_count() == 4
    assert cx_store.edge_count() == 2

    legs = cx_store.get_node("training.legs")
    assert legs is not None
    assert legs.metadata["brain_id"] == ids["training.legs"]
    assert legs.metadata["domain"] == "general"
    assert legs.metadata["tier"] == 2
    assert legs.metadata["title"] == "Legs"
    assert "# Legs" in legs.content

    # Content empty in brain → title carries the meaning alone (no H1 prefix).
    q3 = cx_store.get_node("finance.q3")
    assert q3 is not None
    assert q3.content == "Q3 review"


def test_import_edges_translate_uuid_to_path(brain_schema: str, cx_store: PgGraphStore) -> None:
    _seed_brain(brain_schema)
    import_from_brain(DSN, cx_store, brain_schema=brain_schema)

    out = cx_store.out_edges("training.legs")
    assert {(e.target, e.type) for e in out} == {("training.legs.pendulum", "references")}


def test_import_preserves_embeddings(brain_schema: str, cx_store: PgGraphStore) -> None:
    _seed_brain(brain_schema)
    import_from_brain(DSN, cx_store, brain_schema=brain_schema)

    vec = cx_store.get_embedding("training.legs.pendulum")
    assert vec is not None
    assert len(vec) == EMBED_DIM
    for got, want in zip(vec, [0.0, 1.0, 0.0, 0.0], strict=True):
        assert abs(got - want) < 1e-6

    # Node without an embedding in brain stays embedding-less in cx.
    assert cx_store.has_embedding("training") is False


def test_domain_filter_restricts_both_endpoints(
    brain_schema: str, cx_store: PgGraphStore
) -> None:
    _seed_brain(brain_schema)
    report = import_from_brain(
        DSN, cx_store, brain_schema=brain_schema, domain="finance"
    )

    assert report.nodes_imported == 1
    assert report.edges_imported == 0  # both training edges have general-domain endpoints
    assert cx_store.node_count() == 1
    assert cx_store.get_node("finance.q3") is not None
    assert cx_store.get_node("training") is None


def test_reimport_is_idempotent(brain_schema: str, cx_store: PgGraphStore) -> None:
    _seed_brain(brain_schema)
    import_from_brain(DSN, cx_store, brain_schema=brain_schema)
    import_from_brain(DSN, cx_store, brain_schema=brain_schema)

    # Counts unchanged after a second run — ON CONFLICT DO UPDATE on both
    # tables means re-running upserts rather than duplicating.
    assert cx_store.node_count() == 4
    assert cx_store.edge_count() == 2


# ── memories source ────────────────────────────────────────────────────────


def _seed_memories(schema: str) -> list[str]:
    """Insert a few memory rows; return their UUIDs as strings."""
    assert DSN is not None
    conn = psycopg.connect(DSN, autocommit=True)
    register_vector(conn)
    ids: list[str] = []
    try:
        with conn.cursor() as cur:
            rows = [
                ("notion ingest: Q3 review took longer than expected",
                 [1.0, 0.0, 0.0, 0.0], "notion-sync", ["finance", "retros"], "general"),
                ("slack capture: bug in the deploy pipeline",
                 [0.0, 1.0, 0.0, 0.0], "slack:lloyd", ["bug"], "general"),
                ("notion ingest: customer call notes",
                 None, "notion-sync", ["customer"], "general"),
            ]
            for content, embedding, source, tags, domain in rows:
                cur.execute(
                    f"""
                    INSERT INTO "{schema}".memories
                        (content, embedding, source, tags, domain)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (content, embedding, source, tags, domain),
                )
                row = cur.fetchone()
                assert row is not None
                ids.append(str(row[0]))
    finally:
        conn.close()
    return ids


def test_import_memories(brain_schema: str, cx_store: PgGraphStore) -> None:
    mem_ids = _seed_memories(brain_schema)
    report = import_from_brain(
        DSN, cx_store, source="memories", brain_schema=brain_schema
    )

    assert report.memories_imported == 3
    assert report.nodes_imported == 0
    assert report.edges_imported == 0
    assert cx_store.node_count() == 3

    first = cx_store.get_node(f"memory/{mem_ids[0]}")
    assert first is not None
    assert "Q3 review" in first.content
    assert first.metadata["type"] == "memory"
    assert first.metadata["source"] == "notion-sync"
    assert first.metadata["tags"] == ["finance", "retros"]
    assert first.metadata["brain_memory_id"] == mem_ids[0]

    # Embedding round-trips for memories that have one.
    vec = cx_store.get_embedding(f"memory/{mem_ids[0]}")
    assert vec is not None
    assert len(vec) == EMBED_DIM
    assert cx_store.has_embedding(f"memory/{mem_ids[2]}") is False


def test_import_both_combines_nodes_and_memories(
    brain_schema: str, cx_store: PgGraphStore
) -> None:
    _seed_brain(brain_schema)
    _seed_memories(brain_schema)

    report = import_from_brain(
        DSN, cx_store, source="both", brain_schema=brain_schema
    )

    # 4 docs + 3 memories + 2 edges
    assert report.nodes_imported == 4
    assert report.memories_imported == 3
    assert report.edges_imported == 2
    assert cx_store.node_count() == 7
    assert cx_store.edge_count() == 2


def test_invalid_source_rejected(brain_schema: str, cx_store: PgGraphStore) -> None:
    with pytest.raises(ValueError, match="source must be one of"):
        import_from_brain(
            DSN, cx_store, source="bogus", brain_schema=brain_schema
        )
