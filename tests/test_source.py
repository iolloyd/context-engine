from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from context_engine.source import (
    _SYSTEM_METADATA_KEYS,
    FolderTreeSource,
    ParsedEdge,
    ParsedNode,
    SourceError,
    parse_edges,
    parse_readme,
    serialise_edges,
    serialise_readme,
)
from context_engine.store import GraphStore

FIXTURE_TREE = Path(__file__).resolve().parents[1] / "fixtures" / "tree"


# ── parse / serialise round-trip ───────────────────────────────────────────


def test_parse_readme_extracts_frontmatter_and_body(tmp_path: Path) -> None:
    path = tmp_path / "readme.md"
    path.write_text(
        "---\n"
        "type: exercise\n"
        "status: avoided\n"
        "---\n"
        "Body text here.\n",
        encoding="utf-8",
    )
    node = parse_readme(path, "exercises/squat")
    assert node.node_id == "exercises/squat"
    assert node.metadata == {"type": "exercise", "status": "avoided"}
    assert node.content == "Body text here."


def test_parse_readme_without_frontmatter(tmp_path: Path) -> None:
    path = tmp_path / "readme.md"
    path.write_text("Just a body.\n", encoding="utf-8")
    node = parse_readme(path, "x")
    assert node.metadata == {}
    assert node.content == "Just a body."


def test_parse_edges(tmp_path: Path) -> None:
    path = tmp_path / "edges.yaml"
    path.write_text(
        "edges:\n"
        "  - target: rules/progression\n"
        "    type: if_then\n"
        "    weight: 0.6\n"
        "  - target: conditions/knee\n"
        "    type: because_of\n"
        "    weight: 0.9\n"
        "    content: aggravates meniscus\n",
        encoding="utf-8",
    )
    edges = parse_edges(path, "exercises/squat")
    assert len(edges) == 2
    assert edges[0].target == "rules/progression"
    assert edges[0].weight == 0.6
    assert edges[1].content == "aggravates meniscus"


def test_parse_edges_missing_target(tmp_path: Path) -> None:
    path = tmp_path / "edges.yaml"
    path.write_text("edges:\n  - type: if_then\n", encoding="utf-8")
    with pytest.raises(SourceError):
        parse_edges(path, "x")


def test_serialise_round_trip(tmp_path: Path) -> None:
    node = ParsedNode(
        node_id="exercises/bench",
        content="Body paragraph.",
        metadata={"type": "exercise", "tags": ["push", "compound"]},
    )
    text = serialise_readme(node)
    (tmp_path / "readme.md").write_text(text, encoding="utf-8")
    parsed = parse_readme(tmp_path / "readme.md", "exercises/bench")
    assert parsed.content == node.content
    assert parsed.metadata == node.metadata


def test_edges_round_trip(tmp_path: Path) -> None:
    edges = [
        ParsedEdge(
            source="exercises/squat",
            target="conditions/knee",
            edge_type="because_of",
            weight=0.9,
            content="meniscus",
        ),
    ]
    (tmp_path / "edges.yaml").write_text(serialise_edges(edges), encoding="utf-8")
    parsed = parse_edges(tmp_path / "edges.yaml", "exercises/squat")
    assert len(parsed) == 1
    assert parsed[0].target == "conditions/knee"
    assert parsed[0].weight == 0.9
    assert parsed[0].content == "meniscus"


# ── importer on the fixture tree ───────────────────────────────────────────


def test_import_fixture_tree() -> None:
    store = GraphStore(":memory:")
    source = FolderTreeSource(FIXTURE_TREE)
    report = source.import_into(store)

    assert report.nodes >= 7  # exercises/, bench-press, squat, rules/, prog, deload, cond, knee
    assert store.get_node("exercises/squat") is not None
    assert store.get_node("rules/progression") is not None

    # Status metadata preserved from frontmatter
    squat = store.get_node("exercises/squat")
    assert squat is not None
    assert squat.status == "avoided"

    # Explicit edges
    out = store.out_edges("exercises/squat", edge_types=["because_of"])
    assert len(out) == 1
    assert out[0].target == "conditions/knee-pain"


def test_derived_part_of_edges() -> None:
    store = GraphStore(":memory:")
    source = FolderTreeSource(FIXTURE_TREE)
    source.import_into(store)

    # exercises/squat should have a derived part_of edge → exercises
    part_of = store.out_edges("exercises/squat", edge_types=["part_of"])
    assert len(part_of) == 1
    assert part_of[0].target == "exercises"
    assert part_of[0].metadata.get("derived") is True


def test_import_rejects_edge_to_missing_target(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "readme.md").write_text("---\ntype: x\n---\nA", encoding="utf-8")
    (tmp_path / "a" / "edges.yaml").write_text(
        "edges:\n  - target: b\n    type: if_then\n", encoding="utf-8"
    )
    store = GraphStore(":memory:")
    with pytest.raises(SourceError, match="target does not exist"):
        FolderTreeSource(tmp_path).import_into(store)


# ── exporter round-trip ────────────────────────────────────────────────────


def test_export_then_reimport_is_stable(tmp_path: Path) -> None:
    store1 = GraphStore(":memory:")
    FolderTreeSource(FIXTURE_TREE).import_into(store1)

    target = tmp_path / "out"
    FolderTreeSource(target).export_from(store1)

    store2 = GraphStore(":memory:")
    FolderTreeSource(target).import_into(store2)

    # Node counts match
    assert store1.node_count() == store2.node_count()

    # Every node in store1 appears in store2 with same user metadata and content.
    # System-generated keys (source_hash, indexed_hash) are excluded from comparison
    # because they are recomputed on each import from the raw file bytes.
    for nid in store1.all_node_ids():
        n1 = store1.get_node(nid)
        n2 = store2.get_node(nid)
        assert n2 is not None, f"missing {nid}"
        assert n1 is not None
        assert n1.content == n2.content
        user_meta1 = {k: v for k, v in n1.metadata.items() if k not in _SYSTEM_METADATA_KEYS}
        user_meta2 = {k: v for k, v in n2.metadata.items() if k not in _SYSTEM_METADATA_KEYS}
        assert user_meta1 == user_meta2

    # Every explicit edge survives the round-trip
    for nid in store1.all_node_ids():
        explicit_before = sorted(
            (e.target, e.type, e.weight)
            for e in store1.out_edges(nid)
            if not e.metadata.get("derived", False)
        )
        explicit_after = sorted(
            (e.target, e.type, e.weight)
            for e in store2.out_edges(nid)
            if not e.metadata.get("derived", False)
        )
        assert explicit_before == explicit_after


# ── incremental import / content-hash caching ─────────────────────────────


def test_reimport_skips_unchanged_nodes() -> None:
    store = GraphStore(":memory:")
    source = FolderTreeSource(FIXTURE_TREE)
    report_1 = source.import_into(store)
    report_2 = source.import_into(store)

    assert report_2.nodes_skipped_unchanged == report_1.nodes_created
    assert report_2.nodes_created == 0
    assert report_2.nodes_updated == 0


def test_reimport_detects_edited_node(tmp_path: Path) -> None:
    tree_copy = tmp_path / "tree"
    shutil.copytree(FIXTURE_TREE, tree_copy)

    store = GraphStore(":memory:")
    source = FolderTreeSource(tree_copy)
    report_1 = source.import_into(store)

    # Modify exactly one readme to change its content
    target_readme = tree_copy / "exercises" / "squat" / "readme.md"
    original = target_readme.read_text(encoding="utf-8")
    target_readme.write_text(original + "\nExtra line added.\n", encoding="utf-8")

    report_2 = source.import_into(store)
    n = report_1.nodes_created

    assert report_2.nodes_updated == 1
    assert report_2.nodes_skipped_unchanged == n - 1
    assert report_2.nodes_created == 0


def test_source_hash_persisted_in_metadata() -> None:
    store = GraphStore(":memory:")
    FolderTreeSource(FIXTURE_TREE).import_into(store)

    for nid in store.all_node_ids():
        node = store.get_node(nid)
        assert node is not None
        source_hash = node.metadata.get("source_hash")
        assert source_hash is not None, f"node {nid} missing source_hash"
        assert len(source_hash) == 64, f"node {nid} source_hash is not a 64-char hex string"
        assert all(c in "0123456789abcdef" for c in source_hash)
