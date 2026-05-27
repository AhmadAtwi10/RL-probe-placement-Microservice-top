"""
episode_graph.py
----------------
Generates the service graph G^(e) for a given episode e.

The full-health topology (topology.py) is the starting point. At each
episode boundary, an independent Bernoulli perturbation is applied over
the 14 non-core nodes (§2.1, §7.5 of probe_placement_formulation.md):

  Non-core business-logic (2):  review, attractions
  Non-core data stores    (12):  all mongodb-*, all memcached-*

  For each non-core node v:
    if v was present in G^(e):
      with prob p_fail  → v is absent from G^(e+1)   (pod failure / eviction)
      with prob 1-p_fail → v remains present
    if v was absent in G^(e):
      with prob p_rec   → v returns in G^(e+1)        (pod restart / recovery)
      with prob 1-p_rec → v stays absent

Core nodes (10) are NEVER removed:
  frontend, search, geo, rate, profile, recommendation,
  reservation, user, consul, jaeger

When a non-core node is absent:
  - It and all its edges are removed from G^(e)
  - Any SLO that required that node is uncoverable for the episode
  - The action space shrinks: add_probe(v) is invalid for absent nodes

The EpisodeGraph object is the single source of truth the environment
uses. It is rebuilt at every env.reset() call.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional, Set, Tuple, List

import networkx as nx

from src.simulator.config.node_config import (
    NODE_CATALOG,
    CORE_NODES,
    NON_CORE_NODES,
    ALL_NODES,
)
from src.simulator.config.slo_config import SLO_CATALOG
from src.simulator.graph.topology import build_full_graph, edge_index


# ---------------------------------------------------------------------------
# Default perturbation probabilities  (tunable via configs/default.yaml later)
# ---------------------------------------------------------------------------
DEFAULT_P_FAIL = 0.15   # prob a present non-core node fails at episode boundary
DEFAULT_P_REC  = 0.60   # prob an absent non-core node recovers at episode boundary


# ---------------------------------------------------------------------------
# EpisodeGraph dataclass
# ---------------------------------------------------------------------------

@dataclass
class EpisodeGraph:
    """
    The service graph for a single episode.

    Attributes
    ----------
    G : nx.DiGraph
        The episode graph with only the present nodes and their edges.
    present_nodes : List[str]
        Nodes active in this episode, in canonical ALL_NODES order.
    absent_nodes : Set[str]
        Non-core nodes removed by perturbation.
    probeable_nodes : List[str]
        Subset of present_nodes where is_probeable=True.
        This defines the valid action space for the episode.
    coverable_slos : List[int]
        SLO ids whose candidate node(s) are present in this episode.
        SLOs whose only candidate node is absent are uncoverable.
    node_index : Dict[str, int]
        Maps node name → integer index within present_nodes.
        Used for building GNN input tensors.
    """

    G: nx.DiGraph
    present_nodes: List[str]
    absent_nodes: Set[str]
    probeable_nodes: List[str]
    coverable_slos: List[int]
    node_index: Dict[str, int]

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        return len(self.present_nodes)

    @property
    def num_probeable(self) -> int:
        return len(self.probeable_nodes)

    @property
    def action_space_size(self) -> int:
        """2 * |probeable nodes| + 1  (add, remove, no-op)."""
        return 2 * self.num_probeable + 1

    def is_present(self, node: str) -> bool:
        return node in self.G.nodes

    def is_probeable(self, node: str) -> bool:
        return node in self.probeable_nodes

    def get_edge_index(self) -> Tuple[List[str], Tuple[List[int], List[int]]]:
        """COO edge index for PyTorch Geometric, restricted to present nodes."""
        return edge_index(self.G)

    def slo_coverable(self, slo_id: int) -> bool:
        return slo_id in self.coverable_slos

    def summary(self) -> str:
        lines = [
            f"Nodes present : {self.num_nodes} / {len(ALL_NODES)}",
            f"Absent nodes  : {sorted(self.absent_nodes) if self.absent_nodes else 'none'}",
            f"Probeable     : {self.num_probeable}",
            f"Action space  : {self.action_space_size}",
            f"Coverable SLOs: {self.coverable_slos}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal state tracker for Bernoulli perturbation across episodes
# ---------------------------------------------------------------------------

class _NodePresenceState:
    """
    Tracks which non-core nodes are currently present/absent.
    Updated at each episode boundary via Bernoulli transitions.
    """

    def __init__(self, p_fail: float, p_rec: float, rng: random.Random):
        self.p_fail = p_fail
        self.p_rec  = p_rec
        self.rng    = rng
        # Start with all non-core nodes present
        self._present: Set[str] = set(NON_CORE_NODES)

    def transition(self) -> None:
        """Apply one Bernoulli transition step (called at each episode boundary)."""
        new_present: Set[str] = set()
        for node in NON_CORE_NODES:
            if node in self._present:
                # Currently present: may fail
                if self.rng.random() >= self.p_fail:
                    new_present.add(node)
                # else: node fails → not added to new_present
            else:
                # Currently absent: may recover
                if self.rng.random() < self.p_rec:
                    new_present.add(node)
                # else: stays absent
        self._present = new_present

    @property
    def present(self) -> FrozenSet[str]:
        return frozenset(self._present)

    @property
    def absent(self) -> FrozenSet[str]:
        return frozenset(set(NON_CORE_NODES) - self._present)


# ---------------------------------------------------------------------------
# EpisodeGraphBuilder — the main public API for Phase 1
# ---------------------------------------------------------------------------

class EpisodeGraphBuilder:
    """
    Generates EpisodeGraph instances for successive episodes.

    Usage
    -----
    builder = EpisodeGraphBuilder(p_fail=0.15, p_rec=0.60, seed=42)

    # At env.reset():
    episode_graph = builder.next_episode()

    Parameters
    ----------
    p_fail : float
        Probability that a currently-present non-core node fails at the
        next episode boundary. Default: 0.15.
    p_rec : float
        Probability that a currently-absent non-core node recovers at the
        next episode boundary. Default: 0.60.
    seed : int or None
        Random seed for reproducibility. Pass None for non-deterministic runs.
    force_all_present : bool
        If True, always return the full-health graph (all 24 nodes).
        Useful for debugging and curriculum start.
    """

    def __init__(
        self,
        p_fail: float = DEFAULT_P_FAIL,
        p_rec:  float = DEFAULT_P_REC,
        seed:   Optional[int] = None,
        force_all_present: bool = False,
    ):
        self.p_fail = p_fail
        self.p_rec  = p_rec
        self.force_all_present = force_all_present

        self._rng = random.Random(seed)
        self._state = _NodePresenceState(p_fail, p_rec, self._rng)
        self._full_graph = build_full_graph()   # cached; never mutated
        self._episode_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def next_episode(self) -> EpisodeGraph:
        """
        Returns the EpisodeGraph for the next episode.

        On the very first call (episode 0) the state is already initialised
        with all non-core nodes present, so no transition is applied.
        From episode 1 onward, a Bernoulli transition is applied first.
        """
        if self._episode_count > 0:
            self._state.transition()
        self._episode_count += 1
        return self._build_episode_graph()

    def reset_to_full(self) -> EpisodeGraph:
        """
        Forces all non-core nodes back to present and returns the full
        episode graph. Useful at the start of curriculum training.
        """
        self._state._present = set(NON_CORE_NODES)
        self._episode_count += 1
        return self._build_episode_graph()

    @property
    def episode_count(self) -> int:
        return self._episode_count

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _build_episode_graph(self) -> EpisodeGraph:
        if self.force_all_present:
            absent: Set[str] = set()
        else:
            absent = self._state.absent

        # Build the subgraph: remove absent nodes and their incident edges
        G_ep = self._full_graph.copy()
        G_ep.remove_nodes_from(absent)

        # Canonical ordering: preserve ALL_NODES order, filter to present
        present_nodes = [n for n in ALL_NODES if n in G_ep.nodes]
        node_index    = {n: i for i, n in enumerate(present_nodes)}

        # Probeable nodes: present AND is_probeable=True
        probeable_nodes = [
            n for n in present_nodes
            if NODE_CATALOG[n].probeable
        ]

        # Coverable SLOs: those whose candidate node(s) are present
        coverable_slos = [
            slo.id for slo in SLO_CATALOG
            if any(n in G_ep.nodes for n in slo.nodes)
        ]

        return EpisodeGraph(
            G=G_ep,
            present_nodes=present_nodes,
            absent_nodes=absent,
            probeable_nodes=probeable_nodes,
            coverable_slos=coverable_slos,
            node_index=node_index,
        )