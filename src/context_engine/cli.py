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
import os
import sys
from pathlib import Path

from context_engine.embedding import HashEmbedder
from context_engine.engine import ContextEngine
from context_engine.source import (
    EDGES_FILE,
    README,
    FolderTreeSource,
    ParsedEdge,
    ParsedNode,
    _merge_edges,
    append_frontmatter_edges,
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


def _cmd_suggest_edges(args: argparse.Namespace, store: GraphStore) -> int:
    """Handler for ``ctx suggest-edges``."""
    from context_engine.edge_suggester import EdgeSuggester, EdgeSuggestion  # noqa: PLC0415

    target_node = store.get_node(args.node_id)
    if target_node is None:
        print(f"error: node '{args.node_id}' not found in the store", file=sys.stderr)
        return 1

    # Load all candidate nodes — skip strategy nodes and the target itself.
    all_nodes = [
        n
        for n in store.get_nodes(store.all_node_ids())
        if n.id != args.node_id and n.type != "strategy"
    ]
    if not all_nodes:
        print("no suggestions — no other nodes in the store")
        return 0

    # Load target embedding if available; use knn to pre-rank candidates by similarity.
    target_embedding: list[float] | None = None
    if store.has_embedding(args.node_id):
        import struct  # noqa: PLC0415

        row = store._conn.execute(  # noqa: SLF001
            "SELECT embedding FROM node_embeddings WHERE node_id=?", (args.node_id,)
        ).fetchone()
        if row is not None:
            dim = store.embed_dim
            target_embedding = list(struct.unpack(f"{dim}f", row[0]))
            # Re-rank candidates by knn similarity and keep top max_candidates.
            knn_results = store.knn(target_embedding, k=args.max_candidates)
            knn_ids = {nid for nid, _ in knn_results}
            ranked_candidates = [n for n in all_nodes if n.id in knn_ids]
            # Preserve knn order.
            id_rank = {nid: i for i, (nid, _) in enumerate(knn_results)}
            ranked_candidates.sort(key=lambda n: id_rank.get(n.id, 999))
            all_nodes = ranked_candidates if ranked_candidates else all_nodes

    # Collect edge types already used in the graph for the system prompt hint.
    existing_edge_types: list[str] = []
    seen_types: set[str] = set()
    for nid in store.all_node_ids():
        for edge in store.out_edges(nid):
            t = edge.metadata.get("type")
            if t and t not in seen_types and t != "part_of":
                seen_types.add(str(t))
                existing_edge_types.append(str(t))

    suggester = EdgeSuggester(model="claude-sonnet-4-5", max_candidates=args.max_candidates)

    if not suggester.is_available():
        print(
            "EdgeSuggester unavailable — set ANTHROPIC_API_KEY and install the anthropic SDK",
            file=sys.stderr,
        )
        return 1

    suggestions: list[EdgeSuggestion] = suggester.suggest(
        target_node=target_node,
        candidate_nodes=all_nodes,
        target_embedding=target_embedding,
        existing_edge_types=existing_edge_types,
    )

    if not suggestions:
        print("no suggestions")
        return 0

    if not args.interactive:
        # Print a rich table.
        try:
            from rich.console import Console  # noqa: PLC0415
            from rich.table import Table  # noqa: PLC0415

            console = Console()
            table = Table(title=f"Edge suggestions for '{args.node_id}'")
            table.add_column("target", style="cyan")
            table.add_column("type", style="magenta")
            table.add_column("weight", justify="right")
            table.add_column("rationale")
            for s in suggestions:
                table.add_row(s.target, s.type, f"{s.weight:.2f}", s.rationale)
            console.print(table)
        except ImportError:
            # Fallback plain output if rich is unavailable.
            for s in suggestions:
                print(f"{s.target}  {s.type}  {s.weight:.2f}  {s.rationale}")
        return 0

    # Interactive mode.
    if not args.tree:
        print(
            "error: --tree is required in interactive mode to write edges back to disk",
            file=sys.stderr,
        )
        return 1

    accepted: list[ParsedEdge] = []
    for s in suggestions:
        print(f"\n  target:    {s.target}")
        print(f"  type:      {s.type}")
        print(f"  weight:    {s.weight:.2f}")
        print(f"  rationale: {s.rationale}")
        try:
            answer = input("[y/n/skip] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if answer == "y":
            accepted.append(
                ParsedEdge(
                    source=args.node_id,
                    target=s.target,
                    edge_type=s.type,
                    weight=s.weight,
                    content=None,
                )
            )

    if accepted:
        append_frontmatter_edges(Path(args.tree), args.node_id, accepted)
        print(f"\naccepted {len(accepted)} suggestion(s) and wrote to {args.node_id}/readme.md")
    else:
        print("\nno suggestions accepted")
    return 0


def _global_db_path() -> Path:
    return Path(os.environ.get("CTX_GLOBAL_DB", Path.home() / ".ctx" / "global.db"))


def _build_engine_with_global_fallback(
    store: GraphStore, embedder: HashEmbedder | None = None
) -> ContextEngine:
    global_path = _global_db_path()
    global_store = GraphStore(global_path) if global_path.exists() else None
    if embedder is None:
        embedder = HashEmbedder(dim=store.embed_dim)
    return ContextEngine(store, embedder=embedder, global_store=global_store)


def _cmd_global_init() -> int:
    global_path = _global_db_path()
    global_root = global_path.parent
    global_root.mkdir(parents=True, exist_ok=True)
    knowledge_dir = global_root / "knowledge"
    knowledge_dir.mkdir(exist_ok=True)
    # Create (or touch) the database by opening a GraphStore.
    GraphStore(global_path).close()
    readme_path = global_root / "README.md"
    if not readme_path.exists():
        readme_path.write_text(
            "# Global context-engine graph\n\n"
            "This directory holds knowledge that applies across all projects.\n\n"
            "## Layout\n\n"
            "- `global.db` — the SQLite graph database\n"
            "- `knowledge/` — source nodes as markdown files (one per folder)\n\n"
            "## Usage\n\n"
            "Add nodes with:\n\n"
            "    ctx --global add-node \"your convention\""
            " --id conventions/my-rule --type convention\n\n"
            "Import from a folder tree:\n\n"
            "    ctx --global import knowledge/\n",
            encoding="utf-8",
        )
    print(f"initialised global graph at {global_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ctx")
    parser.add_argument("--db", default="graph.db", help="sqlite path")
    parser.add_argument(
        "--global",
        dest="use_global",
        action="store_true",
        help="target the global db (~/.ctx/global.db) instead of the project db",
    )
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

    p_garden = sub.add_parser("garden", help="graph health diagnostic")
    p_garden.add_argument("--dead-weight", type=float, default=0.1,
                          help="edges below this weight are flagged")
    p_garden.add_argument("--drift-rate", type=float, default=0.3,
                          help="strategy nodes below this success rate are flagged")

    p_suggest = sub.add_parser("suggest-edges", help="propose edges from an LLM")
    p_suggest.add_argument("node_id", help="id of the node to suggest edges for")
    p_suggest.add_argument(
        "--interactive",
        action="store_true",
        help="prompt y/n/skip for each suggestion",
    )
    p_suggest.add_argument(
        "--tree",
        help="path to knowledge/ root (required with --interactive to write back)",
    )
    p_suggest.add_argument("--max-candidates", type=int, default=50)

    sub.add_parser("global-init", help="create the global graph at ~/.ctx/global.db")

    args = parser.parse_args(argv)

    # Resolve effective db path: --global overrides --db.
    db_path = _global_db_path() if args.use_global else Path(args.db)
    store = GraphStore(db_path)

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
        resolved_db = db_path.resolve()
        candidate = resolved_db.parent / "knowledge"
        tree_root = candidate if candidate.is_dir() else None
        engine = _build_engine_with_global_fallback(store, embedder=embedder)
        # Wire tree_root into the feedback applier after construction.
        if tree_root is not None:
            from context_engine.feedback import FeedbackApplier  # noqa: PLC0415

            engine.feedback = FeedbackApplier(
                store, strategies=engine.strategies, tree_root=tree_root
            )
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

    if args.cmd == "garden":
        from rich.console import Console  # noqa: PLC0415
        from rich.table import Table  # noqa: PLC0415

        from context_engine.garden import GardenConfig, Gardener  # noqa: PLC0415

        config = GardenConfig(
            dead_weight_threshold=args.dead_weight,
            drift_success_rate=args.drift_rate,
        )
        gardener = Gardener(store, config)
        findings = gardener.inspect()
        if not findings:
            print("garden: nothing to report — graph looks healthy")
            return 0
        table = Table(title="garden report")
        table.add_column("category")
        table.add_column("node / edge")
        table.add_column("suggestion")
        for f in findings:
            table.add_row(f.category, f.subject, f.suggestion)
        Console().print(table)
        return 0

    if args.cmd == "suggest-edges":
        return _cmd_suggest_edges(args, store)

    if args.cmd == "global-init":
        return _cmd_global_init()

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
