"""Graph health diagnostic.

Runs a fixed set of checks against a graph store and returns a list
of findings. The ``ctx garden`` CLI command renders findings as a
rich table.

Checks:

  - orphans          : nodes with no incoming edges (excluding
                       category roots and internal strategy nodes)
  - duplicates       : near-identical pairs by embedding + content
                       similarity
  - dead_weights     : edges the feedback loop has pushed near zero
  - strategy_drift   : strategy nodes whose traversal config is
                       consistently producing bad slices
  - uncovered_tuples : strategy nodes that have never been used
  - recent_activity  : the 10 most recently updated non-strategy
                       nodes — not a problem, just situational
                       awareness

Thresholds are configurable via GardenConfig; the defaults are
conservative.
"""

from __future__ import annotations

import difflib
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

from context_engine.store import GraphStore

log = logging.getLogger(__name__)

_SKIP_TYPES = {"strategy", "category"}


@dataclass
class GardenConfig:
    dead_weight_threshold: float = 0.1
    duplicate_embed_threshold: float = 0.95
    duplicate_content_threshold: float = 0.8
    drift_min_samples: int = 5
    drift_success_rate: float = 0.3
    recent_activity_count: int = 10


@dataclass
class Finding:
    category: str     # one of: orphan, duplicate, dead_weight, drift, uncovered, recent
    subject: str      # node id or "src -> tgt:type"
    suggestion: str   # short actionable string


@dataclass
class Gardener:
    store: GraphStore
    config: GardenConfig = field(default_factory=GardenConfig)

    def __init__(self, store: GraphStore, config: GardenConfig | None = None) -> None:
        self.store = store
        self.config = config or GardenConfig()

    def inspect(self) -> list[Finding]:
        """Run all checks and return all findings in a stable category order."""
        results: list[Finding] = []
        results.extend(self._find_orphans())
        results.extend(self._find_duplicates())
        results.extend(self._find_dead_weights())
        results.extend(self._find_strategy_drift())
        results.extend(self._find_uncovered_tuples())
        results.extend(self._find_recent_activity())
        return results

    def _find_orphans(self) -> Iterator[Finding]:
        for node_id in self.store.all_node_ids():
            node = self.store.get_node(node_id)
            if node is None:
                continue
            if node.metadata.get("type") in _SKIP_TYPES:
                continue
            if not self.store.in_edges(node_id):
                yield Finding(
                    category="orphan",
                    subject=node_id,
                    suggestion="add incoming edge or delete",
                )

    def _find_duplicates(self) -> Iterator[Finding]:
        node_ids = self.store.all_node_ids()
        # Only run if at least one embedding exists.
        has_any = any(self.store.get_embedding(nid) is not None for nid in node_ids)
        if not has_any:
            return

        if len(node_ids) > 1000:
            log.warning(
                "duplicate detection skipped: %d nodes exceeds the 1000-node limit",
                len(node_ids),
            )
            return

        nodes = [n for nid in node_ids if (n := self.store.get_node(nid)) is not None]
        embeddings = {n.id: self.store.get_embedding(n.id) for n in nodes}

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a, b = nodes[i], nodes[j]
                ea, eb = embeddings.get(a.id), embeddings.get(b.id)
                if ea is None or eb is None:
                    continue
                sim = _cosine_similarity(ea, eb)
                if sim < self.config.duplicate_embed_threshold:
                    continue
                ratio = difflib.SequenceMatcher(None, a.content, b.content).ratio()
                if ratio < self.config.duplicate_content_threshold:
                    continue
                yield Finding(
                    category="duplicate",
                    subject=f"{a.id}, {b.id}",
                    suggestion=f"merge (emb={sim:.2f}, content={ratio:.2f})",
                )

    def _find_dead_weights(self) -> Iterator[Finding]:
        for e in self.store.all_edges():
            if e.weight >= self.config.dead_weight_threshold:
                continue
            if e.metadata.get("derived") is True:
                continue
            yield Finding(
                category="dead_weight",
                subject=f"{e.source} -> {e.target}:{e.type}",
                suggestion=f"prune (w={e.weight:.2f})",
            )

    def _find_strategy_drift(self) -> Iterator[Finding]:
        for node in self.store.filter_nodes({"type": "strategy"}):
            success = int(node.metadata.get("success_count", 0))
            failure = int(node.metadata.get("failure_count", 0))
            total = success + failure
            if total < self.config.drift_min_samples:
                continue
            success_rate = success / (total + 1)
            if success_rate < self.config.drift_success_rate:
                yield Finding(
                    category="drift",
                    subject=node.id,
                    suggestion=f"reset (success={success}, failure={failure})",
                )

    def _find_uncovered_tuples(self) -> Iterator[Finding]:
        for node in self.store.filter_nodes({"type": "strategy"}):
            success = int(node.metadata.get("success_count", 0))
            failure = int(node.metadata.get("failure_count", 0))
            downgrade = int(node.metadata.get("downgrade_count", 0))
            if success == 0 and failure == 0 and downgrade == 0:
                yield Finding(
                    category="uncovered",
                    subject=node.id,
                    suggestion="no queries of this shape yet",
                )

    def _find_recent_activity(self) -> Iterator[Finding]:
        nodes = [
            n
            for nid in self.store.all_node_ids()
            if (n := self.store.get_node(nid)) is not None
            and n.metadata.get("type") != "strategy"
        ]
        nodes.sort(key=lambda n: n.updated_at, reverse=True)
        for node in nodes[: self.config.recent_activity_count]:
            yield Finding(
                category="recent",
                subject=node.id,
                suggestion=node.updated_at.isoformat(),
            )


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
