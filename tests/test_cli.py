from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from context_engine.cli import main
from context_engine.edge_suggester import EdgeSuggestion


def _make_two_node_tree(root: Path) -> None:
    """Minimal two-node tree: node 'a' has an edges.yaml pointing to 'b'."""
    (root / "a").mkdir(parents=True)
    (root / "a" / "readme.md").write_text(
        "---\ntype: x\n---\nNode A.\n", encoding="utf-8"
    )
    (root / "a" / "edges.yaml").write_text(
        "edges:\n  - target: b\n    type: if_then\n    weight: 0.6\n",
        encoding="utf-8",
    )
    (root / "b").mkdir()
    (root / "b" / "readme.md").write_text(
        "---\ntype: y\n---\nNode B.\n", encoding="utf-8"
    )


def test_migrate_inline_edges_dry_run(tmp_path: Path) -> None:
    tree = tmp_path / "knowledge"
    tree.mkdir()
    _make_two_node_tree(tree)

    rc = main(["--db", str(tmp_path / "graph.db"), "migrate", str(tree), "--inline-edges"])
    assert rc == 0

    # Dry run: edges.yaml must still exist.
    assert (tree / "a" / "edges.yaml").exists()
    # Readme must be unchanged.
    content = (tree / "a" / "readme.md").read_text(encoding="utf-8")
    assert "edges" not in content


def test_migrate_inline_edges_with_yes(tmp_path: Path) -> None:
    tree = tmp_path / "knowledge"
    tree.mkdir()
    _make_two_node_tree(tree)

    rc = main(
        ["--db", str(tmp_path / "graph.db"), "migrate", str(tree), "--inline-edges", "--yes"]
    )
    assert rc == 0

    # edges.yaml must be deleted.
    assert not (tree / "a" / "edges.yaml").exists()

    # Edge must now live in the readme frontmatter.
    text = (tree / "a" / "readme.md").read_text(encoding="utf-8")
    assert "edges:" in text
    assert "if_then" in text

    # Re-import must produce the same graph as a pre-migration import.
    from context_engine.source import FolderTreeSource
    from context_engine.store import GraphStore

    store = GraphStore(":memory:")
    FolderTreeSource(tree).import_into(store)

    edges = store.out_edges("a", edge_types=["if_then"])
    assert len(edges) == 1
    assert edges[0].target == "b"


def test_suggest_edges_non_interactive_mocked(tmp_path: Path, capsys) -> None:
    """suggest-edges with patched EdgeSuggester.suggest prints a table with the suggestion."""
    from context_engine.store import GraphStore

    db_path = tmp_path / "graph.db"
    store = GraphStore(str(db_path))
    store.upsert_node(content="Node A content.", metadata={"type": "exercise"}, node_id="a")
    store.upsert_node(content="Node B content.", metadata={"type": "exercise"}, node_id="b")

    canned = [
        EdgeSuggestion(
            target="b",
            type="supports",
            weight=0.8,
            rationale="A supports B because of shared context.",
        )
    ]

    with patch("context_engine.edge_suggester.EdgeSuggester.suggest", return_value=canned), \
         patch("context_engine.edge_suggester.EdgeSuggester.is_available", return_value=True):
        rc = main(["--db", str(db_path), "suggest-edges", "a"])

    assert rc == 0
    captured = capsys.readouterr()
    assert "b" in captured.out
    assert "supports" in captured.out
