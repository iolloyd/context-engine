"""Import brain's content into the context-engine graph layer.

Brain has two ingestion targets that both ship 512-dim pgvector
embeddings:

* ``public.nodes`` / ``public.node_edges`` — the hierarchical docs
  layer (ltree paths, typed edges). Written by ``doc-write``.
* ``public.memories`` — the flat memory pool. Written by every other
  ingestion path: Notion sync, Slack capture, MCP saves.

Framebrain treats both as graph primitives. This importer copies
either or both into ``cx_nodes`` / ``cx_edges`` so the engine's seed
selection and traversal can reach them.

ID mapping
----------

Brain uses UUID primary keys; context-engine uses text ids.

* Docs (``public.nodes``): the ltree path becomes the cx node id
  (matches the path-based identity model from ADR 0001). The original
  UUID is preserved in ``metadata.brain_id`` for traceability.
* Memories (``public.memories``): the cx node id is
  ``memory/<uuid>``. Namespaced so memory ids cannot collide with
  doc paths even if a doc path is ever named "memory/..."

Edge ids (docs only) are deterministic:
``{src_path}->{tgt_path}:{relation}``. Re-imports upsert rather than
duplicate. Memories don't carry edges — they become standalone graph
nodes that the feedback loop or ``suggest-edges`` can connect over
time.

Brain doesn't track edge weights, so imported edges land at ``weight =
0.5`` — the same default the SQLite store uses.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any

import psycopg
from pgvector.psycopg import register_vector

from context_engine.types import StoreProtocol

log = logging.getLogger(__name__)


@dataclass(slots=True)
class ImportReport:
    nodes_imported: int
    edges_imported: int
    memories_imported: int = 0
    entities_imported: int = 0
    mention_edges_imported: int = 0


VALID_SOURCES = ("nodes", "memories", "both")

# Phase 4a: when ``cx_entities`` is present in brain, project the
# canonical entities as graph nodes and lay down ``mentions`` edges
# from memories to entities they reference (via
# ``memory.metadata.entities``). Both passes are idempotent.


def import_from_brain(
    brain_dsn: str,
    store: StoreProtocol,
    *,
    source: str = "nodes",
    domain: str | None = None,
    brain_schema: str = "public",
) -> ImportReport:
    """Copy brain content into the destination graph store.

    ``source`` selects which of brain's ingestion tables to read:

    * ``"nodes"``    — the docs layer (``public.nodes`` +
      ``public.node_edges``). Default for backwards compatibility.
    * ``"memories"`` — the flat memory pool (``public.memories``).
      Each row becomes a standalone cx node id-prefixed with
      ``memory/``.
    * ``"both"``     — runs both passes in one call.

    ``domain`` filters every read to a single brain domain (``"general"``,
    ``"finance"``, etc.). When ``None``, every domain is imported and the
    domain lands in node metadata. Edges where either endpoint is
    filtered out are silently dropped — they would dangle anyway.
    """

    if source not in VALID_SOURCES:
        raise ValueError(f"source must be one of {VALID_SOURCES}; got {source!r}")

    conn = psycopg.connect(brain_dsn)
    register_vector(conn)
    nodes_imported = 0
    edges_imported = 0
    memories_imported = 0
    entities_imported = 0
    mention_edges_imported = 0
    try:
        with conn.cursor() as cur:
            if source in ("nodes", "both"):
                nodes_imported = _import_nodes(cur, store, domain, brain_schema)
                edges_imported = _import_edges(cur, store, domain, brain_schema)
            if source in ("memories", "both"):
                memories_imported = _import_memories(cur, store, domain, brain_schema)
            # Always run the entity + mentions pass when cx_entities
            # exists. It's a no-op when the table is missing
            # (pre-Phase 4a brain).
            if _has_cx_entities(cur, brain_schema):
                entities_imported = _import_entities(cur, store, brain_schema)
                mention_edges_imported = _import_mentions(
                    cur, store, domain, brain_schema
                )
    finally:
        conn.close()
    return ImportReport(
        nodes_imported=nodes_imported,
        edges_imported=edges_imported,
        memories_imported=memories_imported,
        entities_imported=entities_imported,
        mention_edges_imported=mention_edges_imported,
    )


def _has_cx_entities(cur: psycopg.Cursor[Any], brain_schema: str) -> bool:
    cur.execute(
        """
        SELECT EXISTS (
            SELECT 1
              FROM information_schema.tables
             WHERE table_schema = %s
               AND table_name = 'cx_entities'
        )
        """,
        (brain_schema,),
    )
    row = cur.fetchone()
    return bool(row and row[0])


def _import_entities(
    cur: psycopg.Cursor[Any],
    store: StoreProtocol,
    brain_schema: str,
) -> int:
    cur.execute(
        f'SELECT id, canonical_name, kind, aliases, metadata '
        f'FROM "{brain_schema}".cx_entities'
    )
    rows = cur.fetchall()
    count = 0
    for entity_id, canonical_name, kind, aliases, metadata in rows:
        meta = {
            **(metadata or {}),
            "type": "entity",
            "kind": kind,
            "canonical_name": canonical_name,
            "aliases": list(aliases) if aliases else [],
        }
        # Entities are small canonical nodes; no embedding (we don't
        # want them showing up in similarity search as ad-hoc memories).
        store.upsert_node(
            content=canonical_name,
            metadata=meta,
            node_id=entity_id,
            embedding=None,
        )
        count += 1
    return count


def _import_mentions(
    cur: psycopg.Cursor[Any],
    store: StoreProtocol,
    domain: str | None,
    brain_schema: str,
) -> int:
    sql = f"""
        SELECT id, metadata
          FROM "{brain_schema}".memories
         WHERE metadata ? 'entities'
           AND jsonb_array_length(metadata->'entities') > 0
         {"AND domain = %s" if domain else ""}
    """
    cur.execute(sql, (domain,) if domain else ())
    rows = cur.fetchall()
    count = 0
    for brain_id, metadata in rows:
        entities = (metadata or {}).get("entities", [])
        src = f"memory/{brain_id}"
        for entity_id in entities:
            if not isinstance(entity_id, str) or not entity_id:
                continue
            edge_id = f"{src}->{entity_id}:mentions"
            store.upsert_edge(
                source=src,
                target=entity_id,
                edge_type="mentions",
                weight=1.0,
                metadata={"brain_memory_id": str(brain_id)},
                edge_id=edge_id,
            )
            count += 1
    return count


def _import_nodes(
    cur: psycopg.Cursor[Any],
    store: StoreProtocol,
    domain: str | None,
    brain_schema: str,
) -> int:
    sql = f"""
        SELECT id, path::text, title, content, embedding, metadata, domain, tier
          FROM "{brain_schema}".nodes
         {"WHERE domain = %s" if domain else ""}
    """
    cur.execute(sql, (domain,) if domain else ())
    rows = cur.fetchall()

    count = 0
    for brain_id, path, title, content, embedding, metadata, row_domain, tier in rows:
        cx_metadata = {
            **(metadata or {}),
            "brain_id": str(brain_id),
            "domain": row_domain,
            "tier": tier,
            "title": title,
        }
        store.upsert_node(
            content=_merge_title_and_content(title, content),
            metadata=cx_metadata,
            node_id=path,
            embedding=list(embedding) if embedding is not None else None,
        )
        count += 1
    return count


def _import_edges(
    cur: psycopg.Cursor[Any],
    store: StoreProtocol,
    domain: str | None,
    brain_schema: str,
) -> int:
    # Join through nodes to translate UUID endpoints to path ids. When a
    # domain filter is set, restrict *both* endpoints to that domain.
    domain_clause = "AND f.domain = %s AND t.domain = %s" if domain else ""
    sql = f"""
        SELECT f.path::text AS src_path,
               t.path::text AS tgt_path,
               e.relation,
               e.metadata,
               e.from_id,
               e.to_id
          FROM "{brain_schema}".node_edges e
          JOIN "{brain_schema}".nodes f ON f.id = e.from_id
          JOIN "{brain_schema}".nodes t ON t.id = e.to_id
         WHERE 1 = 1
           {domain_clause}
    """
    params = (domain, domain) if domain else ()
    cur.execute(sql, params)
    rows = cur.fetchall()

    count = 0
    for src_path, tgt_path, relation, metadata, from_id, to_id in rows:
        edge_id = f"{src_path}->{tgt_path}:{relation}"
        store.upsert_edge(
            source=src_path,
            target=tgt_path,
            edge_type=relation,
            weight=0.5,
            metadata={
                **(metadata or {}),
                "brain_from_id": str(from_id),
                "brain_to_id": str(to_id),
            },
            edge_id=edge_id,
        )
        count += 1
    return count


def _import_memories(
    cur: psycopg.Cursor[Any],
    store: StoreProtocol,
    domain: str | None,
    brain_schema: str,
) -> int:
    sql = f"""
        SELECT id, content, embedding, source, tags, metadata, domain
          FROM "{brain_schema}".memories
         WHERE content IS NOT NULL AND content <> ''
         {"AND domain = %s" if domain else ""}
    """
    cur.execute(sql, (domain,) if domain else ())
    rows = cur.fetchall()

    count = 0
    for brain_id, content, embedding, mem_source, tags, metadata, row_domain in rows:
        cx_metadata = {
            **(metadata or {}),
            "brain_memory_id": str(brain_id),
            "type": "memory",
            "domain": row_domain,
            "source": mem_source,
            "tags": list(tags) if tags else [],
        }
        store.upsert_node(
            content=content,
            metadata=cx_metadata,
            node_id=f"memory/{brain_id}",
            embedding=list(embedding) if embedding is not None else None,
        )
        count += 1
    return count


def _merge_title_and_content(title: str, content: str) -> str:
    """Brain stores title and content separately; cx_nodes has one
    content field. Use the title as an H1 prefix unless the content is
    empty (in which case the title carries the meaning on its own)."""
    if not content:
        return title
    return f"# {title}\n\n{content}"


# ── CLI entry point ─────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """Standalone runner.

    ``python -m context_engine.brain_importer --brain-dsn ... --pg-dsn ...``

    Two DSNs because brain's docs and the cx graph may live in different
    Postgres instances even though framebrain co-locates them today.

    When the DSN flags are omitted both fall back to the ``CTX_PG_DSN``
    env var — same secret the engine uses. Lets the importer run inside
    distroless containers (Fly scheduled Machines) where no shell is
    available to do ``--pg-dsn "$CTX_PG_DSN"`` expansion.
    """

    env_dsn = os.environ.get("CTX_PG_DSN")

    parser = argparse.ArgumentParser(
        prog="brain-importer",
        description="Import brain docs into the context-engine graph layer.",
    )
    parser.add_argument(
        "--brain-dsn",
        default=env_dsn,
        help="postgres DSN for brain's database (default: $CTX_PG_DSN)",
    )
    parser.add_argument(
        "--pg-dsn",
        default=env_dsn,
        help="postgres DSN for the destination cx_nodes / cx_edges schema "
        "(default: $CTX_PG_DSN)",
    )
    parser.add_argument(
        "--source",
        choices=VALID_SOURCES,
        default="nodes",
        help="which brain table(s) to read (default: nodes)",
    )
    parser.add_argument("--domain", help="restrict to a single brain domain")
    parser.add_argument(
        "--brain-schema",
        default="public",
        help="schema where brain's tables live (default: public)",
    )
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    if not args.brain_dsn or not args.pg_dsn:
        parser.error(
            "missing DSN: pass --brain-dsn and --pg-dsn, "
            "or set CTX_PG_DSN in the environment"
        )

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Imported here so the module stays importable without the postgres extra.
    from context_engine.pg_store import PgGraphStore

    store = PgGraphStore(dsn=args.pg_dsn, embed_dim=args.embed_dim)
    try:
        report = import_from_brain(
            args.brain_dsn,
            store,
            source=args.source,
            domain=args.domain,
            brain_schema=args.brain_schema,
        )
    finally:
        store.close()

    print(f"nodes imported:    {report.nodes_imported}")
    print(f"edges imported:    {report.edges_imported}")
    print(f"memories imported: {report.memories_imported}")
    print(f"entities imported: {report.entities_imported}")
    print(f"mentions imported: {report.mention_edges_imported}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
