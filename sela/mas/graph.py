"""DAG-based execution engine."""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Type

from .artifacts import Artifact

logger = logging.getLogger(__name__)


class Mailbox:
    """Per-node inbox.  Artifacts are stored both by sender ID and by type,"""

    def __init__(self) -> None:
        self._by_source: Dict[str, Artifact] = {}
        self._by_type: Dict[Type[Artifact], Dict[str, Artifact]] = defaultdict(dict)

    def receive(self, source_id: str, artifact: Artifact) -> None:
        self._by_source[source_id] = artifact
        self._by_type[type(artifact)][source_id] = artifact

    def get_by_type(self, artifact_type: Type[Artifact]) -> List[Artifact]:
        """Return all received artifacts of the given type, sorted by sender ID."""
        group = self._by_type.get(artifact_type, {})
        return [group[k] for k in sorted(group)]

    def get_first(self, artifact_type: Type[Artifact]) -> Optional[Artifact]:
        items = self.get_by_type(artifact_type)
        return items[0] if items else None

    def sender_count(self) -> int:
        return len(self._by_source)

    def reset(self) -> None:
        self._by_source.clear()
        self._by_type.clear()


class BaseNode:
    """Abstract graph node."""

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self.mailbox = Mailbox()

    async def execute(self) -> Dict[Type[Artifact], Artifact]:
        raise NotImplementedError(f"{type(self).__name__}.execute() not implemented")

    def reset_mailbox(self) -> None:
        self.mailbox.reset()


class GraphEngine:
    """Executes a DAG of BaseNodes."""

    def __init__(self) -> None:
        self._nodes: Dict[str, BaseNode] = {}
        self._out_edges: Dict[str, List[Tuple[str, Type[Artifact]]]] = defaultdict(list)
        self._indegree: Dict[str, int] = defaultdict(int)


    def add_node(self, node: BaseNode) -> None:
        if node.node_id in self._nodes:
            raise ValueError(f"Node '{node.node_id}' already exists in the graph.")
        self._nodes[node.node_id] = node
        if node.node_id not in self._indegree:
            self._indegree[node.node_id] = 0

    def add_edge(
        self, src: str, dst: str, artifact_type: Type[Artifact]
    ) -> None:
        """Add a directed edge src → dst carrying artifacts of artifact_type."""
        if src not in self._nodes:
            raise ValueError(f"Source node '{src}' not in graph.")
        if dst not in self._nodes:
            raise ValueError(f"Destination node '{dst}' not in graph.")
        self._out_edges[src].append((dst, artifact_type))
        self._indegree[dst] += 1


    async def run(self) -> Dict[str, Dict[Type[Artifact], Artifact]]:
        """Execute all nodes in topological order."""
        levels = self._topological_levels()
        all_results: Dict[str, Dict[Type[Artifact], Artifact]] = {}

        for level in levels:
            level_outputs = await asyncio.gather(
                *[self._execute_node(nid) for nid in level]
            )
            for node_id, outputs in zip(level, level_outputs):
                all_results[node_id] = outputs
                for dst_id, art_type in self._out_edges.get(node_id, []):
                    if art_type in outputs:
                        self._nodes[dst_id].mailbox.receive(node_id, outputs[art_type])
                    else:
                        logger.warning(
                            "Node '%s' did not produce expected artifact type %s",
                            node_id, art_type.__name__,
                        )

        return all_results

    async def _execute_node(
        self, node_id: str
    ) -> Dict[Type[Artifact], Artifact]:
        node = self._nodes[node_id]
        try:
            return await node.execute()
        except Exception:
            logger.exception("Node '%s' raised an exception.", node_id)
            raise


    def _topological_levels(self) -> List[List[str]]:
        indegree = dict(self._indegree)
        queue: List[str] = [n for n, d in indegree.items() if d == 0]
        levels: List[List[str]] = []

        while queue:
            levels.append(list(queue))
            next_q: List[str] = []
            for nid in queue:
                for dst_id, _ in self._out_edges.get(nid, []):
                    indegree[dst_id] -= 1
                    if indegree[dst_id] == 0:
                        next_q.append(dst_id)
            queue = next_q

        scheduled = sum(len(lvl) for lvl in levels)
        if scheduled != len(self._nodes):
            raise RuntimeError(
                "GraphEngine detected a cycle — topological sort failed."
            )
        return levels
