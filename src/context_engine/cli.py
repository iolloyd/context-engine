"""Tiny CLI for smoke-testing the engine.

Usage::

    ctx init                     # create an empty db
    ctx add-node "text" --type exercise
    ctx add-edge SRC TGT --type requires --weight 0.7
    ctx query "why do I avoid squats?" --db graph.db
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from context_engine.embedding import HashEmbedder
from context_engine.engine import ContextEngine
from context_engine.source import FolderTreeSource
from context_engine.store import GraphStore
from context_engine.types import Query


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ctx")
    parser.add_argument("--db", default="graph.db", help="sqlite path")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="create empty db")

    p_node = sub.add_parser("add-node")
    p_node.add_argument("content")
    p_node.add_argument("--id")
    p_node.add_argument("--type")
    p_node.add_argument("--status")
    p_node.add_argument("--meta", help="extra metadata as JSON", default="{}")

    p_edge = sub.add_parser("add-edge")
    p_edge.add_argument("source")
    p_edge.add_argument("target")
    p_edge.add_argument("--type", required=True)
    p_edge.add_argument("--weight", type=float, default=0.5)
    p_edge.add_argument("--content")

    p_query = sub.add_parser("query")
    p_query.add_argument("text")
    p_query.add_argument("--seed", action="append", default=[], help="explicit seed node id")
    p_query.add_argument("--filter", help="metadata filter as JSON")

    sub.add_parser("repl", help="interactive graph editor")

    sub.add_parser("dump")

    p_import = sub.add_parser("import", help="build the graph from a folder tree")
    p_import.add_argument("tree", help="path to the knowledge/ root")

    p_export = sub.add_parser("export", help="write the graph back to a folder tree")
    p_export.add_argument("tree", help="path to the knowledge/ root")

    p_index = sub.add_parser("index", help="backfill embeddings for all nodes")
    p_index.add_argument(
        "--embedder",
        choices=["hash", "anthropic"],
        default="hash",
        help="embedding backend (default: hash)",
    )

    args = parser.parse_args(argv)
    store = GraphStore(Path(args.db))

    if args.cmd == "init":
        print(f"initialised {args.db}")
        return 0

    if args.cmd == "add-node":
        meta = json.loads(args.meta)
        if args.type:
            meta["type"] = args.type
        if args.status:
            meta["status"] = args.status
        node = store.upsert_node(content=args.content, metadata=meta, node_id=args.id)
        print(node.id)
        return 0

    if args.cmd == "add-edge":
        edge = store.upsert_edge(
            source=args.source,
            target=args.target,
            edge_type=args.type,
            weight=args.weight,
            content=args.content,
        )
        print(edge.id)
        return 0

    if args.cmd == "query":
        embedder = HashEmbedder(dim=store.embed_dim)
        engine = ContextEngine(store, embedder=embedder)
        metadata_filter = json.loads(args.filter) if args.filter else None
        query = Query(text=args.text, explicit_refs=args.seed)
        resp = engine.answer(query, metadata_filter=metadata_filter)
        print(resp.text)
        return 0

    if args.cmd == "repl":
        from context_engine.repl import Repl  # noqa: PLC0415

        engine = ContextEngine(store)
        Repl(store=store, engine=engine).run()
        return 0

    if args.cmd == "dump":
        print(f"nodes: {store.node_count()}  edges: {store.edge_count()}")
        return 0

    if args.cmd == "import":
        source = FolderTreeSource(args.tree)
        report = source.import_into(store)
        print(
            f"imported {report.nodes} nodes, "
            f"{report.explicit_edges} explicit edges, "
            f"{report.derived_edges} derived edges"
        )
        return 0

    if args.cmd == "export":
        source = FolderTreeSource(args.tree)
        source.export_from(store)
        print(f"exported {store.node_count()} nodes to {args.tree}")
        return 0

    if args.cmd == "index":
        if args.embedder == "anthropic":
            from context_engine.embedding import AnthropicEmbedder

            embedder = AnthropicEmbedder()
        else:
            embedder = HashEmbedder(dim=store.embed_dim)
        engine = ContextEngine(store, embedder=embedder)
        indexed, already_had = engine.index_all()
        print(f"indexed {indexed} nodes ({already_had} already had embeddings)")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
