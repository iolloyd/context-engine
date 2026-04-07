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
from context_engine.source import (
    EDGES_FILE,
    README,
    FolderTreeSource,
    ParsedNode,
    _merge_edges,
    parse_edges,
    parse_readme,
    serialise_readme,
)
from context_engine.store import GraphStore
from context_engine.types import Query


def _cmd_migrate(args: argparse.Namespace) -> int:
    """Inline edges.yaml content into readme frontmatter (migration helper)."""
    if not args.inline_edges:
        print("nothing to do — pass --inline-edges to migrate edge files")
        return 0

    tree = Path(args.tree).resolve()
    edges_files = sorted(tree.rglob(EDGES_FILE))
    if not edges_files:
        print("no edges.yaml files found — nothing to migrate")
        return 0

    changes: list[tuple[Path, Path]] = []  # (readme_path, edges_path)
    for ef in edges_files:
        readme_path = ef.parent / README
        if not readme_path.exists():
            print(f"  skip {ef} (no readme.md sibling)")
            continue
        node_id = ef.parent.resolve().relative_to(tree).as_posix()
        ext_edges = parse_edges(ef, node_id)
        if not ext_edges:
            print(f"  skip {ef} (empty)")
            continue

        node = parse_readme(readme_path, node_id)
        merged = _merge_edges(node.frontmatter_edges, ext_edges)

        # Build the edge dicts for the frontmatter.
        edge_dicts = [
            {
                k: v
                for k, v in {
                    "target": e.target,
                    "type": e.edge_type,
                    "weight": round(e.weight, 4),
                    "content": e.content,
                }.items()
                if v is not None
            }
            for e in merged
        ]
        meta_with_edges = {**node.metadata, "edges": edge_dicts}
        updated = ParsedNode(
            node_id=node.node_id,
            content=node.content,
            metadata=meta_with_edges,
        )
        new_text = serialise_readme(updated)
        old_text = readme_path.read_text(encoding="utf-8")
        print(f"  {readme_path.relative_to(tree)}: +{len(merged)} edge(s) inlined")
        if old_text != new_text:
            changes.append((readme_path, ef))
            if args.yes:
                readme_path.write_text(new_text, encoding="utf-8")

        if args.yes:
            ef.unlink()

    if not args.yes:
        print(f"\ndry-run: {len(changes)} readme(s) would be updated; pass --yes to apply")
    else:
        print(f"\nmigrated {len(changes)} file(s); edges.yaml files deleted")
    return 0


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

    p_migrate = sub.add_parser("migrate", help="migrate source format in-place")
    p_migrate.add_argument("tree", help="path to knowledge/ root")
    p_migrate.add_argument(
        "--inline-edges",
        action="store_true",
        help="move edges.yaml content into readme frontmatter",
    )
    p_migrate.add_argument(
        "--yes",
        action="store_true",
        help="commit changes (without this flag it's a dry-run)",
    )

    p_index = sub.add_parser("index", help="backfill embeddings for all nodes")
    p_index.add_argument(
        "--embedder",
        choices=["hash", "anthropic", "sentence-transformers"],
        default="sentence-transformers",
        help="embedding backend (default: sentence-transformers, falls back to hash)",
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
        # Derive tree_root from the db path: look for a knowledge/ sibling directory.
        db_path = Path(args.db).resolve()
        candidate = db_path.parent / "knowledge"
        tree_root = candidate if candidate.is_dir() else None
        engine = ContextEngine(store, embedder=embedder, tree_root=tree_root)
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
            f"imported: {report.nodes_created} created, "
            f"{report.nodes_updated} updated, "
            f"{report.nodes_skipped_unchanged} unchanged, "
            f"{report.explicit_edges} explicit edges, "
            f"{report.derived_edges} derived edges"
        )
        return 0

    if args.cmd == "export":
        source = FolderTreeSource(args.tree)
        source.export_from(store)
        print(f"exported {store.node_count()} nodes to {args.tree}")
        return 0

    if args.cmd == "migrate":
        return _cmd_migrate(args)

    if args.cmd == "index":
        if args.embedder == "anthropic":
            from context_engine.embedding import AnthropicEmbedder

            embedder = AnthropicEmbedder()
        elif args.embedder == "sentence-transformers":
            from context_engine.embedding import default_embedder

            embedder = default_embedder()
        else:
            embedder = HashEmbedder(dim=store.embed_dim)
        engine = ContextEngine(store, embedder=embedder)
        indexed, already_fresh = engine.index_all()
        print(
            f"indexed {indexed} nodes ({already_fresh} already fresh)"
            f" using {type(embedder).__name__}"
        )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
