"""
probe_env.py
------------
Gymnasium environment for the probe placement POMDP.

Implements the full episode loop from §4 and §5 of
probe_placement_formulation.md.

─────────────────────────────────────────────────────────────────
Action encoding
─────────────────────────────────────────────────────────────────
Actions are integers in [0, 2*|V_p^(e)| + 1).
The mapping is rebuilt at each reset() because |V_p^(e)| may
change across episodes (non-core nodes come and go).

  index 0                        → no_op
  index 1 .. |V_p|               → add_probe(probeable_nodes[i-1])
  index |V_p|+1 .. 2*|V_p|       → remove_probe(probeable_nodes[i-|V_p|-1])

The mapping is exposed via:
  env.action_to_str(a)  — human-readable label
  env.probeable_nodes   — ordered list of probeable nodes in this episode

─────────────────────────────────────────────────────────────────
Observation dict
─────────────────────────────────────────────────────────────────
reset() and step() return a flat dict:

  {
    "node_features" : float32  (|V^(e)|, NODE_FEATURE_DIM)   — GNN input
    "edge_index"    : int64    (2, |E^(e)|)                   — COO edge index
    "slo_health"    : float32  (NUM_SLOS,)                    — H_t
    "action_mask"   : bool     (action_space_size,)           — valid actions
  }

action_mask is True for valid actions, False for invalid ones
(e.g. add_probe(v) when v is already probed).  The policy should
multiply logits by this mask before softmax.

─────────────────────────────────────────────────────────────────
Episode structure (§4.7)
─────────────────────────────────────────────────────────────────
  - Max T timesteps per episode
  - Failure termination if cumulative blind violations > K
  - Terminal bonus on survival, penalty on failure

─────────────────────────────────────────────────────────────────
Public API
─────────────────────────────────────────────────────────────────
  ProbeEnvConfig      — episode hyperparameters (T, K, seeds, ...)
  ProbeEnv            — Gymnasium Env subclass
    .reset()          → obs, info
    .step(action)     → obs, reward, terminated, truncated, info
    .action_to_str(a) → str
    .probeable_nodes  → List[str]
    .current_graph    → EpisodeGraph
"""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.simulator.config.node_config import PROBEABLE_NODES
from src.simulator.config.slo_config import NUM_SLOS
from src.simulator.graph.episode_graph import EpisodeGraph, EpisodeGraphBuilder
from src.simulator.workload.sli_generator import SLIGenerator
from src.simulator.workload.failure_injector import FailureInjector
from src.simulator.env.observation_builder import (
    ObservationBuilder, NODE_FEATURE_DIM, flatten_sli,
)
from src.simulator.env.reward import (
    RewardConfig, RewardInput, RewardOutput,
    compute_reward, terminal_reward,
    ADD_PROBE, REMOVE_PROBE, NO_OP,
)


# ---------------------------------------------------------------------------
# ProbeEnvConfig
# ---------------------------------------------------------------------------

@dataclass
class ProbeEnvConfig:
    """
    Episode-level hyperparameters for ProbeEnv.

    Attributes
    ----------
    episode_length : int
        T — maximum number of timesteps per episode.
    blind_violation_budget : int
        K — cumulative blind violations allowed before failure termination.
    min_failures : int
        Minimum number of failure events injected per episode (inclusive).
        Set to 0 to allow failure-free episodes.
    max_failures : int
        Maximum number of failure events injected per episode (inclusive).
        Actual count sampled uniformly in [min_failures, max_failures]
        at each reset(), seeded by workload_seed for reproducibility.
    reward_config : RewardConfig
        λ, μ, ρ and terminal bonus/penalty values.
    window_size : int
        N — rolling SLI history window for the ObservationBuilder.
    graph_seed : Optional[int]
        Seed for EpisodeGraphBuilder (None = non-deterministic).
    workload_seed : Optional[int]
        Seed for SLIGenerator and FailureInjector (None = non-deterministic).
    diurnal_amplitude : float
        Amplitude of the diurnal sine pattern in SLIGenerator.
    """
    episode_length:         int         = 100
    blind_violation_budget: int         = 50
    min_failures:           int         = 0
    max_failures:           int         = 3
    reward_config:          RewardConfig = field(default_factory=RewardConfig)
    window_size:            int         = 4
    graph_seed:             Optional[int] = None
    workload_seed:          Optional[int] = None
    diurnal_amplitude:      float       = 0.3

    def __post_init__(self):
        assert self.episode_length         > 0
        assert self.blind_violation_budget >= 0
        assert self.min_failures >= 0
        assert self.max_failures >= self.min_failures
        assert self.window_size            >= 1


# ---------------------------------------------------------------------------
# ProbeEnv
# ---------------------------------------------------------------------------

class ProbeEnv(gym.Env):
    """
    Gymnasium environment for the probe placement POMDP.

    The environment implements the full episode loop:
      reset() → [step()] * T → terminated/truncated

    At each step:
      1. Decode action → (action_type, action_node)
      2. Update probe set
      3. Advance workload to next timestep
      4. Flatten SLI snapshot
      5. Build observation (node_features, slo_health, action_mask)
      6. Compute reward
      7. Check termination

    Parameters
    ----------
    config : ProbeEnvConfig
        Episode hyperparameters.
    """

    metadata = {"render_modes": []} # tells gymnasium that this env doesn't support rendering

    def __init__(self, config: Optional[ProbeEnvConfig] = None):
        super().__init__()
        self.cfg = config or ProbeEnvConfig()

        # Sub-components (initialised at reset)
        self._graph_builder = EpisodeGraphBuilder(
            seed=self.cfg.graph_seed,
        )
        self._obs_builder = ObservationBuilder(window_size=self.cfg.window_size)

        # Episode state — set by reset()
        self._ep:           Optional[EpisodeGraph]              = None
        self._probe_set:    set                                  = set()
        self._traces:       Optional[Dict[str, Dict[str, np.ndarray]]] = None
        self._t:            int                                  = 0
        self._cum_blind:    int                                  = 0   # Σ blind violations
        self._probeable:    List[str]                            = []  # ordered for action encoding
        self._action_size:  int                                  = 1   # 2*|V_p|+1

        # Spaces — overwritten at each reset() because |V_p| may change.
        # We define a max-size space here for gym registry compatibility;
        # the actual valid actions are communicated via action_mask.
        max_probeable = len(PROBEABLE_NODES)
        self.action_space = spaces.Discrete(2 * max_probeable + 1)
        self.observation_space = spaces.Dict({
            "node_features": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(24, NODE_FEATURE_DIM),   # 24 = max nodes
                dtype=np.float32,
            ),
            "edge_index": spaces.Box(
                low=0, high=23,
                shape=(2, 200),   # upper bound on edges
                dtype=np.int64,
            ),
            "slo_health": spaces.Box(
                low=0.0, high=1.0,
                shape=(NUM_SLOS,),
                dtype=np.float32,
            ),
            "action_mask": spaces.Box(
                low=0, high=1,
                shape=(2 * max_probeable + 1,),
                dtype=bool,
            ),
        })

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def probeable_nodes(self) -> List[str]:
        """Ordered list of probeable nodes in the current episode."""
        return list(self._probeable)

    @property
    def current_graph(self) -> Optional[EpisodeGraph]:
        """The active EpisodeGraph (None before first reset)."""
        return self._ep

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Start a new episode.

        1. Sample a new episode graph (new non-core node set).
        2. Generate full SLI traces for the episode.
        3. Inject failure events into the traces.
        4. Reset probe set, timestep counter, blind violation counter.
        5. Build and return the initial observation.

        Returns
        -------
        obs : dict
            Initial observation (at t=0, before any action).
        info : dict
            Diagnostic info: episode_id, present_nodes, probeable_nodes,
            coverable_slos, min_failures, max_failures.
        """
        super().reset(seed=seed)

        # ── 1. New episode graph ──────────────────────────────────────────
        self._ep = self._graph_builder.next_episode()

        # ── 2. Build ordered probeable node list and action encoding ──────
        # Use ep.probeable_nodes (set) sorted for deterministic ordering
        self._probeable   = sorted(self._ep.probeable_nodes)
        self._action_size = 2 * len(self._probeable) + 1

        # ── 3. Generate SLI traces ────────────────────────────────────────
        workload_seed = (
            self.cfg.workload_seed
            if self.cfg.workload_seed is not None
            else self.np_random.integers(0, 2**31).item()
        )
        gen    = SLIGenerator(
            self._ep,
            episode_length=self.cfg.episode_length,
            diurnal_amplitude=self.cfg.diurnal_amplitude,
            seed=workload_seed,
        )
        self._traces = gen.full_traces()

        # ── 4. Inject failures ────────────────────────────────────────────
        _rng = random.Random(workload_seed)
        n = _rng.randint(self.cfg.min_failures, self.cfg.max_failures)
        if n > 0:
            fi     = FailureInjector(self._ep, episode_length=self.cfg.episode_length,
                                     seed=workload_seed)
            events = fi.sample_failures(n_failures=n)
            self._traces = fi.apply(self._traces, events)
        else:
            events = []

        # ── 5. Reset episode state ────────────────────────────────────────
        self._probe_set = set()
        self._t         = 0
        self._cum_blind = 0

        # ── 6. Reset observation builder ──────────────────────────────────
        self._obs_builder.reset(self._ep)

        # ── 7. Build initial observation ──────────────────────────────────
        flat_sli = self._flat_sli_at(self._t)
        obs      = self._build_obs(flat_sli)

        info = {
            "episode_graph":   self._ep,
            "present_nodes":   self._ep.present_nodes,
            "probeable_nodes": self._probeable,
            "coverable_slos":  list(self._ep.coverable_slos),
            "n_failures":      len(events),
            "t":               self._t,
        }
        return obs, info

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(
        self, action: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Execute one timestep.

        Parameters
        ----------
        action : int
            Integer action in [0, action_space.n).
            Use env.action_to_str(action) for a human-readable label.

        Returns
        -------
        obs         : dict
        reward      : float
        terminated  : bool   — failure termination (blind violations > K)
        truncated   : bool   — normal termination (t == T)
        info        : dict   — diagnostic breakdown
        """
        assert self._ep is not None, "Call reset() before step()"

        # ── 1. Decode action ──────────────────────────────────────────────
        action_type, action_node = self._decode_action(action)

        # ── 2. Update probe set ───────────────────────────────────────────
        if action_type == ADD_PROBE and action_node is not None:
            self._probe_set.add(action_node)
        elif action_type == REMOVE_PROBE and action_node is not None:
            self._probe_set.discard(action_node)
        # NO_OP: no change

        # ── 3. Get flat SLI snapshot at current t ─────────────────────────
        flat_sli = self._flat_sli_at(self._t)

        # ── 4. Compute reward ─────────────────────────────────────────────
        reward_inp = RewardInput(
            probe_set=set(self._probe_set),
            sli_values=flat_sli,
            action_type=action_type,
            action_node=action_node,
            episode_graph=self._ep,
        )
        reward_out: RewardOutput = compute_reward(reward_inp, self.cfg.reward_config)

        # ── 5. Update cumulative blind violation count ────────────────────
        self._cum_blind += len(reward_out.blind_slos)

        # ── 6. Build observation ──────────────────────────────────────────
        obs = self._build_obs(flat_sli)

        # ── 7. Termination checks ─────────────────────────────────────────
        terminated = self._cum_blind > self.cfg.blind_violation_budget
        truncated  = (self._t + 1) >= self.cfg.episode_length

        # Terminal reward
        scalar_reward = reward_out.total
        if terminated or truncated:
            scalar_reward += terminal_reward(
                survived=truncated and not terminated,
                cfg=self.cfg.reward_config,
            )

        # ── 8. Advance timestep ───────────────────────────────────────────
        self._t += 1

        info = {
            "t":                self._t,
            "action_str":       self.action_to_str(action),
            "probe_set":        set(self._probe_set),
            "reward_breakdown": reward_out,
            "cum_blind":        self._cum_blind,
            "blind_slos":       reward_out.blind_slos,
            "covered_slos":     reward_out.covered_slos,
            "terminated":       terminated,
            "truncated":        truncated,
        }
        return obs, scalar_reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Action encoding helpers
    # ------------------------------------------------------------------

    def action_to_str(self, action: int) -> str:
        """Return a human-readable label for an action integer."""
        action_type, action_node = self._decode_action(action)
        if action_type == NO_OP:
            return "no_op"
        prefix = "add_probe" if action_type == ADD_PROBE else "remove_probe"
        return f"{prefix}({action_node})"

    def build_action_mask(self) -> np.ndarray:
        """
        Returns a boolean mask of shape (action_space.n,).

        True  = action is currently valid.
        False = action is invalid (e.g. add_probe on already-probed node,
                remove_probe on non-probed node, or action index out of
                range for this episode's smaller probeable set).
        """
        n   = self.action_space.n
        mask = np.zeros(n, dtype=bool)

        # no_op always valid
        mask[0] = True

        n_p = len(self._probeable)
        for i, node in enumerate(self._probeable):
            # add_probe(node): valid iff node not currently probed
            mask[1 + i] = node not in self._probe_set
            # remove_probe(node): valid iff node currently probed
            mask[1 + n_p + i] = node in self._probe_set

        # Indices beyond 2*n_p+1 (larger episodes) are always False (default)
        return mask

    def _decode_action(self, action: int) -> Tuple[int, Optional[str]]:
        """
        Decode integer action to (action_type, action_node).

        Returns
        -------
        action_type : int   ADD_PROBE | REMOVE_PROBE | NO_OP
        action_node : str | None
        """
        n_p = len(self._probeable)

        if action == 0:
            return NO_OP, None
        elif 1 <= action <= n_p:
            return ADD_PROBE, self._probeable[action - 1]
        elif n_p + 1 <= action <= 2 * n_p:
            return REMOVE_PROBE, self._probeable[action - n_p - 1]
        else:
            # Out-of-range action (e.g. from fixed-size action space
            # when this episode has fewer probeable nodes) — treat as no_op
            return NO_OP, None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flat_sli_at(self, t: int) -> Dict[Tuple[str, str], float]:
        """
        Extract a flat SLI snapshot at timestep t from the pre-computed traces.

        Converts from SLIGenerator's nested format:
            {node: {metric: array}} → {(node, metric): float}
        """
        nested_t = {
            node: {metric: float(arr[t]) for metric, arr in metrics.items()}
            for node, metrics in self._traces.items()
        }
        return flatten_sli(nested_t)

    def _build_obs(
        self, flat_sli: Dict[Tuple[str, str], float]
    ) -> Dict[str, Any]:
        """
        Build the full observation dict at the current timestep.

        Components:
          node_features : (|V|, NODE_FEATURE_DIM) — GNN node input
          edge_index    : (2, |E|) COO format     — GNN graph structure
          slo_health    : (NUM_SLOS,)              — H_t coverage+health vector
          action_mask   : (action_space.n,)        — valid action mask
        """
        node_features, slo_health = self._obs_builder.build(
            probe_set=self._probe_set,
            sli_snapshot=flat_sli,
            t=self._t,
        )

        # Pad node_features to max size (24 nodes) for fixed observation space
        max_nodes = self.observation_space["node_features"].shape[0]
        if node_features.shape[0] < max_nodes:
            pad = np.zeros(
                (max_nodes - node_features.shape[0], NODE_FEATURE_DIM),
                dtype=np.float32,
            )
            node_features = np.vstack([node_features, pad])

        # edge_index: COO from episode graph, padded to max edge count
        # get_edge_index() returns (node_names, (src_indices, dst_indices))
        _, (src_idx, dst_idx) = self._ep.get_edge_index()
        edge_index_ep = np.array([src_idx, dst_idx], dtype=np.int64)  # (2, |E|)
        max_edges     = self.observation_space["edge_index"].shape[1]
        if edge_index_ep.shape[1] < max_edges:
            pad_cols   = np.zeros(
                (2, max_edges - edge_index_ep.shape[1]),
                dtype=np.int64,
            )
            edge_index = np.hstack([edge_index_ep, pad_cols])
        else:
            edge_index = edge_index_ep[:, :max_edges]

        return {
            "node_features": node_features,
            "edge_index":    edge_index,
            "slo_health":    slo_health,
            "action_mask":   self.build_action_mask(),
        }