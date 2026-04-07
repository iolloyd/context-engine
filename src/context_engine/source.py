"""Folder-tree knowledge source.

Source of truth is a directory tree under ``knowledge/``. Each node is a
folder containing:

  - ``readme.md``   — YAML frontmatter (metadata) + body (content)
  - ``edges.yaml``  — optional outgoing edges

Node id is the path relative to the tree root, POSIX-normalised
(``exercises/bench-press``). The importer rebuilds the SQLite store from the
tree. The exporter writes the current store state back to the tree — used
by the feedback loop so weight updates are reviewable in git.

Structural edges derived from parent folders (``part_of``) are never written
back to ``edges.yaml``.

See ``docs/adr/0001-knowledge-source.md`` for the full rationale.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from context_engine.store import GraphStore

README = "readme.md"
EDGES_FILE = "edges.yaml"
FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


# ── parse / serialise single node ──────────────────────────────────────────


@dataclass
class ParsedNode:
    node_id: str
    content: str
    metadata: dict[str, Any]


@dataclass
class ParsedEdge:
    source: str
    target: str
    edge_type: str
    weight: float
    content: str | None


def parse_readme(path: Path, node_id: str) -> ParsedNode:
    text = path.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(text)
    if match:
        frontmatter = yaml.safe_load(match.group(1)) or {}
        body = match.group(2).strip()
    else:
        frontmatter = {}
        body = text.strip()
    if not isinstance(frontmatter, dict):
        raise SourceError(f"{path}: frontmatter must be a mapping")
    return ParsedNode(
        node_id=node_id,
        content=body,
        metadata=frontmatter,
    )


def parse_edges(path: Path, source_id: str) -> list[ParsedEdge]:
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise SourceError(f"{path}: top level must be a mapping")
    items = raw.get("edges", [])
    if not isinstance(items, list):
        raise SourceError(f"{path}: 'edges' must be a list")

    out: list[ParsedEdge] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise SourceError(f"{path}: edge[{i}] must be a mapping")
        try:
            out.append(
                ParsedEdge(
                    source=source_id,
                    target=str(item["target"]),
                    edge_type=str(item["type"]),
                    weight=float(item.get("weight", 0.5)),
                    content=item.get("content"),
                )
            )
        except KeyError as exc:
            raise SourceError(f"{path}: edge[{i}] missing {exc}") from exc
    return out


def serialise_readme(node: ParsedNode) -> str:
    frontmatter = yaml.safe_dump(node.metadata, sort_keys=True).strip()
    return f"---\n{frontmatter}\n---\n{node.content.rstrip()}\n"


def serialise_edges(edges: list[ParsedEdge]) -> str:
    payload = {
        "edges": [
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
            for e in edges
        ]
    }
    return yaml.safe_dump(payload, sort_keys=False)


# ── importer ───────────────────────────────────────────────────────────────

_SYSTEM_METADATA_KEYS = frozenset({"source_hash", "indexed_hash"})


def _compute_source_hash(readme_path: Path, edges_path: Path | None) -> str:
    """SHA-256 of the readme bytes, optionally combined with the edges file bytes."""
    h = hashlib.sha256()
    h.update(readme_path.read_bytes())
    if edges_path is not None and edges_path.exists():
        h.update(b"\n---edges---\n")
        h.update(edges_path.read_bytes())
    return h.hexdigest()


class SourceError(Exception):
    pass


class FolderTreeSource:
    """Round-trip a folder tree with a ``GraphStore``."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()

    # ── import ─────────────────────────────────────────────────────────────

    def walk_nodes(self) -> Iterable[ParsedNode]:
        """Yield every node in the tree."""
        if not self.root.exists():
            raise SourceError(f"source root does not exist: {self.root}")
        for readme in sorted(self.root.rglob(README)):
            node_id = self._id_for(readme.parent)
            yield parse_readme(readme, node_id)

    def walk_edges(self) -> Iterable[ParsedEdge]:
        """Yield every explicit edge declared under the tree."""
        for edges_file in sorted(self.root.rglob(EDGES_FILE)):
            source_id = self._id_for(edges_file.parent)
            yield from parse_edges(edges_file, source_id)

    def derive_structural_edges(
        self, node_ids: set[str]
    ) -> Iterable[ParsedEdge]:
        """Synthesise ``part_of`` edges from child to parent folder nodes."""
        for nid in node_ids:
            parent = str(Path(nid).parent).replace("\\", "/")
            if parent in (".", "/"):
                continue
            if parent in node_ids:
                yield ParsedEdge(
                    source=nid,
                    target=parent,
                    edge_type="part_of",
                    weight=0.5,
                    content=None,
                )

    def import_into(self, store: GraphStore) -> ImportReport:
        """Rebuild the graph store from the tree, skipping unchanged nodes."""
        nodes = list(self.walk_nodes())
        node_ids = {n.node_id for n in nodes}

        nodes_created = 0
        nodes_updated = 0
        nodes_skipped = 0
        changed_node_ids: set[str] = set()

        for node in nodes:
            readme_path = self.root / node.node_id / README
            edges_path = self.root / node.node_id / EDGES_FILE
            new_hash = _compute_source_hash(readme_path, edges_path)

            existing = store.get_node(node.node_id)
            if existing is not None and existing.metadata.get("source_hash") == new_hash:
                nodes_skipped += 1
                continue

            node.metadata["source_hash"] = new_hash
            store.upsert_node(
                content=node.content,
                metadata=node.metadata,
                node_id=node.node_id,
            )
            changed_node_ids.add(node.node_id)
            if existing is None:
                nodes_created += 1
            else:
                nodes_updated += 1

        edges = list(self.walk_edges())
        for edge in edges:
            if edge.target not in node_ids:
                raise SourceError(
                    f"edge {edge.source} → {edge.target}: target does not exist"
                )
            if edge.source in changed_node_ids:
                self._upsert_edge(store, edge, derived=False)


        derived_count = 0
        for edge in self.derive_structural_edges(node_ids):
            self._upsert_edge(store, edge, derived=True)
            derived_count += 1

        return ImportReport(
            nodes=len(nodes),
            explicit_edges=len(edges),
            derived_edges=derived_count,
            nodes_created=nodes_created,
            nodes_updated=nodes_updated,
            nodes_skipped_unchanged=nodes_skipped,
        )

    @staticmethod
    def _upsert_edge(
        store: GraphStore, edge: ParsedEdge, derived: bool
    ) -> None:
        edge_id = f"{edge.source}::{edge.edge_type}::{edge.target}"
        store.upsert_edge(
            source=edge.source,
            target=edge.target,
            edge_type=edge.edge_type,
            weight=edge.weight,
            content=edge.content,
            metadata={"derived": derived},
            edge_id=edge_id,
        )

    # ── export ─────────────────────────────────────────────────────────────

    def export_from(self, store: GraphStore) -> None:
        """Write every node and non-derived edge from the store back to disk."""
        for nid in store.all_node_ids():
            node = store.get_node(nid)
            if node is None:
                continue
            folder = self.root / nid
            folder.mkdir(parents=True, exist_ok=True)
            clean_meta = {k: v for k, v in node.metadata.items() if k not in _SYSTEM_METADATA_KEYS}
            parsed = ParsedNode(
                node_id=nid, content=node.content, metadata=clean_meta
            )
            (folder / README).write_text(serialise_readme(parsed), encoding="utf-8")

            explicit = [
                ParsedEdge(
                    source=e.source,
                    target=e.target,
                    edge_type=e.type,
                    weight=e.weight,
                    content=e.content,
                )
                for e in store.out_edges(nid)
                if not e.metadata.get("derived", False)
            ]
            edges_path = folder / EDGES_FILE
            if explicit:
                edges_path.write_text(serialise_edges(explicit), encoding="utf-8")
            elif edges_path.exists():
                edges_path.unlink()

    # ── helpers ────────────────────────────────────────────────────────────

    def _id_for(self, folder: Path) -> str:
        rel = folder.resolve().relative_to(self.root)
        return rel.as_posix()


@dataclass
class ImportReport:
    nodes: int
    explicit_edges: int
    derived_edges: int
    nodes_created: int = field(default=0)
    nodes_updated: int = field(default=0)
    nodes_skipped_unchanged: int = field(default=0)
