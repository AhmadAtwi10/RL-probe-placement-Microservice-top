"""
src/simulator/graph
-------------------
Public API for the graph layer (Phase 1).

Typical usage in the environment:

    from src.simulator.graph import EpisodeGraphBuilder

    builder = EpisodeGraphBuilder(p_fail=0.15, p_rec=0.60, seed=42)
    ep_graph = builder.next_episode()   # call at env.reset()
"""

from src.simulator.graph.topology import (
    build_full_graph,
    get_neighbors,
    get_call_subgraph,
    get_upstream_nodes,
    get_downstream_nodes,
    adjacency_matrix,
    edge_index,
    CALL_EDGES,
    INFRA_EDGES,
    ALL_EDGES,
)

from src.simulator.graph.episode_graph import (
    EpisodeGraph,
    EpisodeGraphBuilder,
    DEFAULT_P_FAIL,
    DEFAULT_P_REC,
)

__all__ = [
    "build_full_graph",
    "get_neighbors",
    "get_call_subgraph",
    "get_upstream_nodes",
    "get_downstream_nodes",
    "adjacency_matrix",
    "edge_index",
    "CALL_EDGES",
    "INFRA_EDGES",
    "ALL_EDGES",
    "EpisodeGraph",
    "EpisodeGraphBuilder",
    "DEFAULT_P_FAIL",
    "DEFAULT_P_REC",
]