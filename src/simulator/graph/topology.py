"""
topology.py
-----------
Static edge definitions for the Hotel Reservation topology (DeathStarBench).

This module contains ONLY the full-health topology — i.e., all 24 nodes
present, all edges active. It is the canonical ground truth from which
episode graphs are derived via perturbation (see episode_graph.py).

Two edge categories (from §2.1 of probe_placement_formulation.md):

  CALL_EDGES   — gRPC/HTTP inter-service call dependencies
                 (derived from source code and docker-compose service links)

  INFRA_EDGES  — infrastructure dependency edges
                 (derived from docker-compose depends_on declarations)

Both sets are directed: (caller, callee) or (dependent, dependency).
Together they form the full directed graph G_full = (V_full, E_full).
"""

from typing import List, Tuple, Dict, Set
import networkx as nx

from src.simulator.config.node_config import NODE_CATALOG, ALL_NODES


# ---------------------------------------------------------------------------
# Edge definitions
# ---------------------------------------------------------------------------

# gRPC / HTTP call graph edges  (caller → callee)
# Source: frontend source code + docker-compose service links
CALL_EDGES: List[Tuple[str, str]] = [
    # frontend fans out to all business-logic services
    ("frontend", "search"),
    ("frontend", "profile"),
    ("frontend", "recommendation"),
    ("frontend", "reservation"),
    ("frontend", "user"),
    ("frontend", "review"),
    ("frontend", "attractions"),
    # search depends on geo and rate for its composite result
    ("search", "geo"),
    ("search", "rate"),
]

# Infrastructure dependency edges  (service → its data-store / infra dep)
# Source: docker-compose depends_on declarations
INFRA_EDGES: List[Tuple[str, str]] = [
    # MongoDB backends
    ("geo",            "mongodb-geo"),
    ("rate",           "mongodb-rate"),
    ("profile",        "mongodb-profile"),
    ("recommendation", "mongodb-recommendation"),
    ("reservation",    "mongodb-reservation"),
    ("user",           "mongodb-user"),
    ("review",         "mongodb-review"),
    ("attractions",    "mongodb-attractions"),
    # Memcached backends
    ("rate",        "memcached-rate"),
    ("profile",     "memcached-profile"),
    ("reservation", "memcached-reserve"),
    ("review",      "memcached-review"),
    # All business-logic services register/resolve via consul
    ("frontend",       "consul"),
    ("search",         "consul"),
    ("geo",            "consul"),
    ("rate",           "consul"),
    ("profile",        "consul"),
    ("recommendation", "consul"),
    ("reservation",    "consul"),
    ("user",           "consul"),
    ("review",         "consul"),
    ("attractions",    "consul"),
    # All services emit traces to jaeger
    ("frontend",       "jaeger"),
    ("search",         "jaeger"),
    ("geo",            "jaeger"),
    ("rate",           "jaeger"),
    ("profile",        "jaeger"),
    ("recommendation", "jaeger"),
    ("reservation",    "jaeger"),
    ("user",           "jaeger"),
    ("review",         "jaeger"),
    ("attractions",    "jaeger"),
]

# Combined full edge list
ALL_EDGES: List[Tuple[str, str]] = CALL_EDGES + INFRA_EDGES


# ---------------------------------------------------------------------------
# Full-health graph (all 24 nodes, all edges)
# ---------------------------------------------------------------------------

def build_full_graph() -> nx.DiGraph:
    """
    Returns the full-health networkx DiGraph G_full with all 24 nodes
    and all edges. Node attributes are populated from NODE_CATALOG.

    This graph is the starting point for episode-level perturbation.
    It is NOT used directly by the environment — The graphs that comes from
    episode_graph.py is used for per-episode graph.
    """
    G = nx.DiGraph()

    # Add all nodes with their metadata as attributes
    for name, meta in NODE_CATALOG.items():
        G.add_node(
            name,
            node_type=meta.node_type,
            core=meta.core,
            probeable=meta.probeable,
            slis=list(meta.slis),
            type_one_hot=meta.type_one_hot(),
        )

    # Add all edges with a label indicating edge category
    for src, dst in CALL_EDGES:
        G.add_edge(src, dst, edge_type="call")

    for src, dst in INFRA_EDGES:
        G.add_edge(src, dst, edge_type="infra")

    return G


# ---------------------------------------------------------------------------
# Adjacency helpers
# ---------------------------------------------------------------------------

def get_neighbors(node: str, G: nx.DiGraph) -> Dict[str, List[str]]:
    """
    Returns the direct predecessors and successors of `node` in graph G.

    Example:
        get_neighbors("search", G)
        → {"predecessors": ["frontend"], "successors": ["geo", "rate", "consul", "jaeger"]}
    """
    return {
        "predecessors": list(G.predecessors(node)),
        "successors":   list(G.successors(node)),
    }


def get_call_subgraph(G: nx.DiGraph) -> nx.DiGraph:
    """Returns a subgraph containing only gRPC/HTTP call edges."""
    call_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "call"]
    return G.edge_subgraph(call_edges).copy()


def get_upstream_nodes(node: str, G: nx.DiGraph, call_only: bool = True) -> Set[str]:
    """
    Returns all nodes that can transitively affect `node` via upstream calls.
    If call_only=True, restricts traversal to call edges only (ignores infra).

    Useful for propagating stress signals: if `node` is degraded, all nodes
    in its upstream set may contribute to that degradation.
    """
    graph = get_call_subgraph(G) if call_only else G
    return nx.ancestors(graph, node)


def get_downstream_nodes(node: str, G: nx.DiGraph, call_only: bool = True) -> Set[str]:
    """
    Returns all nodes that are transitively affected by `node` via downstream calls.
    If call_only=True, restricts traversal to call edges only.

    Useful for impact analysis: if `node` degrades, which services will feel it?
    """
    graph = get_call_subgraph(G) if call_only else G
    return nx.descendants(graph, node)


def adjacency_matrix(G: nx.DiGraph) -> Tuple[List[str], List[List[int]]]:
    """
    Returns (node_order, adj_matrix) where:
      - node_order: list of node names in a fixed canonical order
      - adj_matrix: NxN binary matrix, adj[i][j]=1 if edge i→j exists in G

    The canonical order follows ALL_NODES from node_config (preserving
    the layer structure: gateway → business_logic → data_stores → infra).
    Nodes not present in G (removed by perturbation) are excluded.
    """
    present_nodes = [n for n in ALL_NODES if n in G.nodes]
    idx = {n: i for i, n in enumerate(present_nodes)}
    N = len(present_nodes)
    matrix = [[0] * N for _ in range(N)]
    for u, v in G.edges():
        if u in idx and v in idx:
            matrix[idx[u]][idx[v]] = 1
    return present_nodes, matrix


def edge_index(G: nx.DiGraph) -> Tuple[List[str], Tuple[List[int], List[int]]]:
    """
    Returns (node_order, (src_list, dst_list)) where:
      - node_order  : list of node names in canonical ALL_NODES order,
                      restricted to nodes present in G. Maps integer index -> name.
      - src_list    : list of integer source indices (COO format)
      - dst_list    : list of integer destination indices (COO format)
 
    Node indices are positions within node_order, NOT positions in ALL_NODES.
    Nodes absent from G are excluded and indices are re-compacted.
 
    Example usage:
        node_order, (src, dst) = edge_index(G)
        edge_index_tensor = torch.tensor([src, dst], dtype=torch.long)
        # node_order[src[i]] -> node_order[dst[i]] for each edge i
    """
    present_nodes = [n for n in ALL_NODES if n in G.nodes]
    idx = {n: i for i, n in enumerate(present_nodes)}
    src_list, dst_list = [], []
    for u, v in G.edges():
        if u in idx and v in idx:
            src_list.append(idx[u])
            dst_list.append(idx[v])
    return present_nodes, (src_list, dst_list)
 