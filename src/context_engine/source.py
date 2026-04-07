"""Folder-tree knowledge source.

Source of truth is a directory tree under ``knowledge/``. Each node is a
folder containing:

  - ``readme.md``   — YAML frontmatter (metadata + optional ``edges:`` list) + body
  - ``edges.yaml``  — optional outgoing edges (legacy; both can coexist)

Node id is the path relative to the tree root, POSIX-normalised
(``exercises/bench-press``). The importer rebuilds the SQLite store from the
tree. The exporter writes the current store state back to the tree — used
by the feedback loop so weight updates are reviewable in git.

Structural edges derived from parent folders (``part_of``) are never written
back to ``edges.yaml``.

Feedback weights are written to ``<root>/.ctx/weights.yaml`` (the sidecar).
This file is auto-generated; do not edit by hand. Add ``knowledge/.ctx/`` to
``.gitignore`` alongside ``.ctx.db``.

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
SIDECAR_DIR = ".ctx"
WEIGHTS_FILE = "weights.yaml"
FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


# ── parse / serialise single node ──────────────────────────────────────────


@dataclass
class ParsedNode:
    node_id: str
    content: str
    metadata: dict[str, Any]
    frontmatter_edges: list[ParsedEdge] = field(default_factory=list)


@dataclass
class ParsedEdge:
    source: str
    target: str
    edge_type: str
    weight: float
    content: str | None


def _parse_edge_items(
    items: list[Any],
    source_id: str,
    path: Path,
    context: str = "",
) -> list[ParsedEdge]:
    """Parse a list of edge dicts into ParsedEdge objects."""
    prefix = f"{path}: {context} " if context else f"{path}: "
    out: list[ParsedEdge] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise SourceError(f"{prefix}edge[{i}] must be a mapping")
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
            raise SourceError(f"{prefix}edge[{i}] missing {exc}") from exc
    return out


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

    raw_edges = frontmatter.pop("edges", None) or []
    if not isinstance(raw_edges, list):
        raise SourceError(f"{path}: frontmatter 'edges' must be a list")
    frontmatter_edges = _parse_edge_items(raw_edges, node_id, path, context="frontmatter")

    return ParsedNode(
        node_id=node_id,
        content=body,
        metadata=frontmatter,
        frontmatter_edges=frontmatter_edges,
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
    return _parse_edge_items(items, source_id, path)


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
    """SHA-256 of the readme bytes, optionally combined with the edges file bytes.

    Note: the sidecar (``<root>/.ctx/weights.yaml``) is intentionally excluded
    from the source hash.  Sidecar overrides are applied on every import pass
    regardless of whether the hash has changed, so re-weighting is always
    idempotent and cheap without forcing a full node re-upsert.
    """
    h = hashlib.sha256()
    h.update(readme_path.read_bytes())
    if edges_path is not None and edges_path.exists():
        h.update(b"\n---edges---\n")
        h.update(edges_path.read_bytes())
    return h.hexdigest()


def _load_weights_sidecar(root: Path) -> dict[tuple[str, str, str], float]:
    """Load ``<root>/.ctx/weights.yaml`` if it exists.

    Returns a map from ``(source, edge_type, target) -> weight``.
    Missing file or empty file returns ``{}``.

    Key format on disk: ``"<source> -> <target>:<type>"``
    """
    sidecar_path = root / SIDECAR_DIR / WEIGHTS_FILE
    if not sidecar_path.exists():
        return {}
    raw = yaml.safe_load(sidecar_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return {}
    edges = raw.get("edges") or {}
    if not isinstance(edges, dict):
        return {}
    result: dict[tuple[str, str, str], float] = {}
    for key, val in edges.items():
        if not isinstance(val, dict):
            continue
        try:
            source_part, rest = str(key).split(" -> ", 1)
            target_part, type_part = rest.rsplit(":", 1)
            result[(source_part.strip(), type_part.strip(), target_part.strip())] = float(
                val["weight"]
            )
        except (ValueError, KeyError):
            continue
    return result


def _merge_edges(
    frontmatter_edges: list[ParsedEdge],
    external_edges: list[ParsedEdge],
) -> list[ParsedEdge]:
    """Merge frontmatter and external edges; frontmatter wins on duplicates.

    Deduplication key is ``(source, target, edge_type)``.
    """
    seen: dict[tuple[str, str, str], ParsedEdge] = {}
    for edge in external_edges:
        seen[(edge.source, edge.target, edge.edge_type)] = edge
    for edge in frontmatter_edges:
        seen[(edge.source, edge.target, edge.edge_type)] = edge
    return list(seen.values())


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
        """Rebuild the graph store from the tree, skipping unchanged nodes.

        Node upserts are skipped when the source hash is unchanged.  Edge
        upserts (including sidecar weight overrides) always run so that
        ``<root>/.ctx/weights.yaml`` changes take effect without touching
        any authored files.
        """
        nodes = list(self.walk_nodes())
        node_ids = {n.node_id for n in nodes}
        sidecar = _load_weights_sidecar(self.root)

        nodes_created = 0
        nodes_updated = 0
        nodes_skipped = 0
        total_explicit_edges = 0

        # Pass 1 — upsert nodes so all targets exist before edges are inserted.
        for node in nodes:
            readme_path = self.root / node.node_id / README
            edges_path = self.root / node.node_id / EDGES_FILE
            new_hash = _compute_source_hash(readme_path, edges_path)

            existing = store.get_node(node.node_id)
            if existing is None or existing.metadata.get("source_hash") != new_hash:
                node.metadata["source_hash"] = new_hash
                store.upsert_node(
                    content=node.content,
                    metadata=node.metadata,
                    node_id=node.node_id,
                )
                if existing is None:
                    nodes_created += 1
                else:
                    nodes_updated += 1
            else:
                nodes_skipped += 1

        # Pass 2 — merge frontmatter + external edges, apply sidecar, upsert edges.
        # Always runs so sidecar overrides take effect even for hash-skipped nodes.
        for node in nodes:
            edges_path = self.root / node.node_id / EDGES_FILE
            ext_edges = parse_edges(edges_path, node.node_id)
            merged = _merge_edges(node.frontmatter_edges, ext_edges)

            for edge in merged:
                if edge.target not in node_ids:
                    raise SourceError(
                        f"edge {edge.source} → {edge.target}: target does not exist"
                    )

            for edge in merged:
                key = (edge.source, edge.edge_type, edge.target)
                if key in sidecar:
                    edge.weight = sidecar[key]
                self._upsert_edge(store, edge, derived=False)

            total_explicit_edges += len(merged)

        derived_count = 0
        for edge in self.derive_structural_edges(node_ids):
            self._upsert_edge(store, edge, derived=True)
            derived_count += 1

        return ImportReport(
            nodes=len(nodes),
            explicit_edges=total_explicit_edges,
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
