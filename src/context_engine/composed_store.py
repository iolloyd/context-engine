"""Read-only composite over two GraphStore instances.

Provides the subset of GraphStore methods used by Traverser and SeedSelector.
The project store takes priority: on node id collisions the project version wins.
Edge results are unioned across both stores.
"""

from __future__ import annotations

from context_engine.types import Edge, Node, StoreProtocol


class ComposedStore:
    """Duck-typed read-only view over a project store plus a global fallback store.

    Only implements the methods actually called by Traverser and SeedSelector:
    get_node, get_nodes, out_edges, in_edges, get_edge, knn, all_node_ids.
    """

    def __init__(self, project: StoreProtocol, global_: StoreProtocol) -> None:
        self._project = project
        self._global = global_

    # ── node access ────────────────────────────────────────────────────────

    def get_node(self, node_id: str) -> Node | None:
        node = self._project.get_node(node_id)
        if node is not None:
            return node
        return self._global.get_node(node_id)

    def get_nodes(self, node_ids: list[str]) -> list[Node]:
        # Project wins on id collision; fill missing from global.
        project_map = {n.id: n for n in self._project.get_nodes(node_ids)}
        missing = [nid for nid in node_ids if nid not in project_map]
        global_nodes = self._global.get_nodes(missing) if missing else []
        global_map = {n.id: n for n in global_nodes}
        # Preserve order.
        result: list[Node] = []
        for nid in node_ids:
            if nid in project_map:
                result.append(project_map[nid])
            elif nid in global_map:
                result.append(global_map[nid])
        return result

    def all_node_ids(self) -> list[str]:
        project_ids = set(self._project.all_node_ids())
        global_ids = set(self._global.all_node_ids())
        return list(project_ids | global_ids)

    # ── edge access ────────────────────────────────────────────────────────

    def out_edges(
        self,
        node_id: str,
        edge_types: list[str] | None = None,
        weight_floor: float = 0.0,
    ) -> list[Edge]:
        project_edges = self._project.out_edges(
            node_id, edge_types=edge_types, weight_floor=weight_floor
        )
        global_edges = self._global.out_edges(
            node_id, edge_types=edge_types, weight_floor=weight_floor
        )
        # Union: project edge ids take priority.
        seen_ids = {e.id for e in project_edges}
        deduped_global = [e for e in global_edges if e.id not in seen_ids]
        return project_edges + deduped_global

    def in_edges(
        self,
        node_id: str,
        edge_types: list[str] | None = None,
        weight_floor: float = 0.0,
    ) -> list[Edge]:
        project_edges = self._project.in_edges(
            node_id, edge_types=edge_types, weight_floor=weight_floor
        )
        global_edges = self._global.in_edges(
            node_id, edge_types=edge_types, weight_floor=weight_floor
        )
        seen_ids = {e.id for e in project_edges}
        deduped_global = [e for e in global_edges if e.id not in seen_ids]
        return project_edges + deduped_global

    def get_edge(self, edge_id: str) -> Edge | None:
        edge = self._project.get_edge(edge_id)
        if edge is not None:
            return edge
        return self._global.get_edge(edge_id)

    # ── vector similarity ──────────────────────────────────────────────────

    def knn(self, vec: list[float], k: int = 5) -> list[tuple[str, float]]:
        """Merge knn results from both stores, project wins on id collision."""
        project_results = self._project.knn(vec, k=k)
        global_results = self._global.knn(vec, k=k)

        # Merge and deduplicate; project ids win on collision.
        seen_ids = {nid for nid, _ in project_results}
        merged = list(project_results)
        for nid, dist in global_results:
            if nid not in seen_ids:
                merged.append((nid, dist))
                seen_ids.add(nid)

        # Re-sort by distance, return top-k.
        merged.sort(key=lambda t: t[1])
        return merged[:k]
