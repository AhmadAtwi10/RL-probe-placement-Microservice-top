"""
observation_builder.py
----------------------
Builds the structured observation o_t fed to the policy network each timestep.

From §4.2 of probe_placement_formulation.md:

    o_t = (G^(e), P_t, ô_t, H_t)

Concretely this module produces two tensors:

  node_features  : float32 array  shape (|V^(e)|, STATIC_DIM + DYNAMIC_DIM)
                   One row per present node.  Row order matches
                   episode_graph.present_nodes (canonical ALL_NODES order).

  slo_health     : float32 array  shape (NUM_SLOS,)
                   H_t[k] = 1 if SLO k is covered AND not violated, else 0.

The caller (probe_env.py) also has direct access to:
  - episode_graph  (contains G^(e) and the COO edge_index for the GNN)
  - probe_set      (P_t — which nodes are probed)

─────────────────────────────────────────────────────────────────
Node feature layout  x_v(t) = [ x_v^static | x_v^dynamic(t) ]
─────────────────────────────────────────────────────────────────

STATIC (17 values, fixed for the whole episode):
  [0  :  7]   type_one_hot(v)       7-dim one-hot over NODE_TYPES
  [7  :  8]   is_probeable(v)       1 if agent can probe v, else 0
  [8  :  9]   degree_in(v)          in-degree in G^(e)
  [9  : 10]   degree_out(v)         out-degree in G^(e)
  [10 : 16]   slo_coverage_mask(v)  binary, 1 if node covers SLO k
  [16 : 17]   slo_importance(v)     Σ_{k covered by v} w_k

DYNAMIC (45 values with default N=4, updated every timestep):
  [17 : 18]   P_t[v]                1 if probed, else 0
  [18 : 54]   ô_t(v)                6 SLOs × (N + 2) values each
                per SLO k slot (N+2 values):
                  [0 : N]   rolling window of SLI_k(t-N+1..t)  (0 if masked)
                  [N]       margin(k, t)                        (0 if masked)
                  [N+1]     Δmargin(k, t)                       (0 if masked)
  [54 : 60]   staleness(v, k)       timesteps since last observation of SLO k
                                    on node v; 0 when currently probed, counts
                                    up every timestep the node is unprobed,
                                    reset to 0 on re-probe.
  [60 : 61]   incoming_call_rate(v, t)   always observable (infra traces)
  [61 : 62]   outgoing_call_rate(v, t)   always observable (infra traces)

Total per-node feature size = 17 + 45 = 62  (with N=4)

─────────────────────────────────────────────────────────────────
SLI snapshot format expected by build()
─────────────────────────────────────────────────────────────────
  sli_snapshot : Dict[Tuple[str, str], float]
      Flat dict keyed by (node_name, metric_name).
      Example: {("search", "search_latency_p99"): 80.0, ...}

  The SLIGenerator produces a NESTED dict:
      {"search": {"search_latency_p99": 80.0, ...}, ...}

  Flattening is the responsibility of probe_env.py, which owns the
  data pipeline.  A helper is provided:
      flatten_sli(nested: dict) -> Dict[Tuple[str, str], float]

─────────────────────────────────────────────────────────────────
Public API
─────────────────────────────────────────────────────────────────
  WINDOW_SIZE              : int   default rolling window N
  STATIC_DIM               : int   17
  DYNAMIC_DIM              : int   1 + NUM_SLOS*(N+2) + NUM_SLOS + 2
  NODE_FEATURE_DIM         : int   STATIC_DIM + DYNAMIC_DIM

  flatten_sli(nested)      — convert SLIGenerator output to flat snapshot

  ObservationBuilder(window_size)
      .reset(episode_graph) — call at env.reset(); recomputes static features,
                              clears history and staleness buffers
      .build(probe_set, sli_snapshot, t)
                            — call every timestep; returns (node_features, slo_health)
                              t is accepted for future time-encoding features
                              (e.g. diurnal position) but not yet consumed

  NodeFeatureIndex         — named offsets into the feature vector (for tests/debug)
"""

from typing import Dict, Optional, Set, Tuple

import numpy as np

from src.simulator.config.node_config import NODE_CATALOG, NODE_TYPES
from src.simulator.config.slo_config import (
    SLO_CATALOG, SLO_BY_ID, NODE_TO_SLOS, NUM_SLOS,
)
from src.simulator.graph.episode_graph import EpisodeGraph


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------

WINDOW_SIZE: int = 4   # N — rolling history length (tunable)

# Static block
_TYPE_ONE_HOT_DIM = len(NODE_TYPES)                        # 7
_STATIC_FIXED     = 1 + 1 + 1 + NUM_SLOS + 1              # probeable + deg_in + deg_out + mask + importance
STATIC_DIM: int   = _TYPE_ONE_HOT_DIM + _STATIC_FIXED     # 17


def _dynamic_dim(window_size: int) -> int:
    # P_t[v](1) + ô_t slots(K*(N+2)) + staleness(K) + call_rates(2)
    return 1 + NUM_SLOS * (window_size + 2) + NUM_SLOS + 2


DYNAMIC_DIM: int      = _dynamic_dim(WINDOW_SIZE)          # 45 at N=4
NODE_FEATURE_DIM: int = STATIC_DIM + DYNAMIC_DIM           # 62 at N=4


# ---------------------------------------------------------------------------
# Named offsets into the feature vector (useful for tests and debugging)
# ---------------------------------------------------------------------------

class NodeFeatureIndex:
    """Slice boundaries for each block in x_v(t).

    All indices are relative to the full feature vector of length
    NODE_FEATURE_DIM.  Computed from the dimension constants so they
    stay correct if WINDOW_SIZE or NUM_SLOS ever change.
    """
    type_one_hot_start: int = 0
    type_one_hot_end:   int = _TYPE_ONE_HOT_DIM            # [0:7]
    is_probeable:       int = _TYPE_ONE_HOT_DIM            # [7]
    degree_in:          int = _TYPE_ONE_HOT_DIM + 1        # [8]
    degree_out:         int = _TYPE_ONE_HOT_DIM + 2        # [9]
    slo_mask_start:     int = _TYPE_ONE_HOT_DIM + 3        # [10:16]
    slo_mask_end:       int = _TYPE_ONE_HOT_DIM + 3 + NUM_SLOS
    slo_importance:     int = _TYPE_ONE_HOT_DIM + 3 + NUM_SLOS   # [16]

    # dynamic block
    probe_flag:         int = STATIC_DIM                   # [17]
    obs_window_start:   int = STATIC_DIM + 1               # [18]

    @staticmethod
    def staleness_start(window_size: int = WINDOW_SIZE) -> int:
        """Index of first staleness entry (depends on window_size)."""
        return STATIC_DIM + 1 + NUM_SLOS * (window_size + 2)

    @staticmethod
    def staleness_end(window_size: int = WINDOW_SIZE) -> int:
        return NodeFeatureIndex.staleness_start(window_size) + NUM_SLOS

    # call rates are always the last two entries
    incoming_call_rate: int = -2
    outgoing_call_rate: int = -1


IDX = NodeFeatureIndex()


# ---------------------------------------------------------------------------
# Public helper: flatten SLIGenerator output
# ---------------------------------------------------------------------------

def flatten_sli(
    nested: Dict[str, Dict[str, float]],
) -> Dict[Tuple[str, str], float]:
    """
    Convert SLIGenerator's nested output to the flat format expected by build().

    Parameters
    ----------
    nested : dict
        SLIGenerator output at a single timestep t:
            {"node_name": {"metric_name": value, ...}, ...}

    Returns
    -------
    flat : dict
        {("node_name", "metric_name"): value, ...}

    Example
    -------
    >>> flat = flatten_sli({"search": {"search_latency_p99": 80.0}})
    >>> flat[("search", "search_latency_p99")]
    80.0
    """
    flat: Dict[Tuple[str, str], float] = {}
    for node, metrics in nested.items():
        for metric, value in metrics.items():
            flat[(node, metric)] = value
    return flat


# ---------------------------------------------------------------------------
# ObservationBuilder
# ---------------------------------------------------------------------------

class ObservationBuilder:
    """
    Builds structured per-timestep observations for the probe placement POMDP.

    Lifecycle
    ---------
    1. Construct once per environment instance.
    2. Call reset(episode_graph) at every env.reset() — recomputes static
       features and clears all history and staleness buffers.
    3. Call build(probe_set, sli_snapshot, t) at every env.step() to get
       the (node_features, slo_health) observation tuple.

    Parameters
    ----------
    window_size : int
        N — number of timesteps in the rolling SLI history window.

    Notes
    -----
    sli_snapshot must be a FLAT dict keyed by (node, metric).
    Use flatten_sli() to convert SLIGenerator's nested output before
    passing it here.  probe_env.py is responsible for this conversion.
    """

    def __init__(self, window_size: int = WINDOW_SIZE):
        self.window_size = window_size
        self._dyn_dim    = _dynamic_dim(window_size)
        self._feat_dim   = STATIC_DIM + self._dyn_dim

        # Staleness block offset (computed once)
        # Relative offset inside the dynamic vector (IDX gives absolute, subtract STATIC_DIM)
        self._staleness_offset = 1 + NUM_SLOS * (window_size + 2)

        # Set by reset() — Optional until first reset() call
        self._ep:              Optional[EpisodeGraph] = None
        self._static_features: Optional[np.ndarray]  = None

        # Per-node, per-SLO buffers — keyed by (node, slo_id)
        self._sli_history:    Dict[Tuple[str, int], np.ndarray] = {}
        self._margin_history: Dict[Tuple[str, int], np.ndarray] = {}
        # staleness[k] = timesteps since last probe on that (node, slo_id)
        self._staleness:      Dict[Tuple[str, int], int]        = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def node_feature_dim(self) -> int:
        return self._feat_dim

    @property
    def num_slos(self) -> int:
        return NUM_SLOS

    def reset(self, episode_graph: EpisodeGraph) -> None:
        """
        Initialise for a new episode.

        Must be called at env.reset() before the first build() call.
        Recomputes all static features from the new episode graph and
        clears the SLI history and staleness buffers.
        """
        self._ep              = episode_graph
        self._static_features = self._compute_static_features()
        self._sli_history     = {}
        self._margin_history  = {}
        self._staleness       = {}

        # Pre-allocate buffers for every (node, slo_id) pair in this episode
        for node in episode_graph.present_nodes:
            for slo_id in NODE_TO_SLOS.get(node, []):
                self._sli_history[(node, slo_id)]    = np.zeros(self.window_size)
                self._margin_history[(node, slo_id)] = np.zeros(self.window_size)
                self._staleness[(node, slo_id)]      = 0   # unknown at episode start

    def build(
        self,
        probe_set:    Set[str],
        sli_snapshot: Dict[Tuple[str, str], float],
        t:            int,  # noqa: ARG002  reserved for future time-encoding features
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the observation for the current timestep t.

        Parameters
        ----------
        probe_set : Set[str]
            Currently probed node names (P_t).
        sli_snapshot : Dict[Tuple[str, str], float]
            Flat dict of ground-truth SLI values for ALL present nodes at
            time t, keyed by (node_name, metric_name).
            Use flatten_sli() to convert SLIGenerator's nested output.
            Unprobed node values are present here (ground truth used internally)
            but are masked to zero in the observation output.
        t : int
            Current timestep index.  Accepted for API stability and future
            use (e.g. diurnal position encoding) — not consumed in this version.

        Returns
        -------
        node_features : np.ndarray  shape (|V^(e)|, node_feature_dim)
            One row per present node in canonical present_nodes order.
        slo_health : np.ndarray  shape (NUM_SLOS,)
            H_t[k] = 1 if SLO k is covered AND not violated, else 0.
        """
        assert self._ep is not None, "Call reset(episode_graph) before build()"

        # Step 1 — update SLI history and staleness counters
        self._update_history(probe_set, sli_snapshot)

        # Step 2 — build node feature matrix
        n_nodes       = self._ep.num_nodes
        node_features = np.zeros((n_nodes, self._feat_dim), dtype=np.float32)

        for i, node in enumerate(self._ep.present_nodes):
            node_features[i, :STATIC_DIM] = self._static_features[i]
            node_features[i, STATIC_DIM:] = self._build_dynamic(
                node, probe_set, sli_snapshot
            )

        # Step 3 — build SLO health vector H_t
        slo_health = self._build_slo_health(probe_set, sli_snapshot)

        return node_features, slo_health

    # ------------------------------------------------------------------
    # Static features  (computed once per episode at reset)
    # ------------------------------------------------------------------

    def _compute_static_features(self) -> np.ndarray:
        """
        Builds x_v^static for every present node.
        Returns shape (|V^(e)|, STATIC_DIM).
        """
        ep    = self._ep
        feats = np.zeros((ep.num_nodes, STATIC_DIM), dtype=np.float32)

        for i, node in enumerate(ep.present_nodes):
            meta = NODE_CATALOG[node]
            row  = feats[i]
            ptr  = 0

            # type_one_hot  [0:7]
            row[ptr:ptr + _TYPE_ONE_HOT_DIM] = meta.type_one_hot()
            ptr += _TYPE_ONE_HOT_DIM

            # is_probeable  [7]
            row[ptr] = 1.0 if meta.probeable else 0.0
            ptr += 1

            # degree_in  [8]
            row[ptr] = float(ep.G.in_degree(node))
            ptr += 1

            # degree_out  [9]
            row[ptr] = float(ep.G.out_degree(node))
            ptr += 1

            # slo_coverage_mask  [10:16]
            for k, slo in enumerate(SLO_CATALOG):
                row[ptr + k] = 1.0 if node in slo.nodes else 0.0
            ptr += NUM_SLOS

            # slo_importance  [16]
            row[ptr] = sum(slo.weight for slo in SLO_CATALOG if node in slo.nodes)

        return feats

    # ------------------------------------------------------------------
    # History + staleness update
    # ------------------------------------------------------------------

    def _update_history(
        self,
        probe_set:    Set[str],
        sli_snapshot: Dict[Tuple[str, str], float],
    ) -> None:
        """
        For every (node, slo_id) pair in the episode:
          - If node is currently PROBED:
              shift the SLI + margin window forward with the new value,
              reset staleness counter to 0.
          - If node is currently UNPROBED:
              do NOT touch the SLI/margin buffers (the history is frozen
              internally, but masked to zero in the observation while the
              node is unprobed; the staleness counter tells the agent how
              long it has been since the last observation),
              increment staleness counter by 1.

        Invariant: buffers are pre-allocated by reset(), so every
        (node, slo_id) key is guaranteed to exist.  We skip with a
        guard rather than silently creating a buffer.
        """
        for node in self._ep.present_nodes:
            is_probed = node in probe_set
            for slo_id in NODE_TO_SLOS.get(node, []):
                buf_key = (node, slo_id)

                # Guard: buffer must exist (pre-allocated at reset)
                if buf_key not in self._sli_history:
                    # Should never happen after a proper reset() — skip safely
                    continue

                if is_probed:
                    slo     = SLO_BY_ID[slo_id]
                    sli_key = (node, slo.metric)

                    # Skip if metric absent from snapshot — leave buffer frozen;
                    # staleness will keep incrementing until a valid read arrives
                    if sli_key not in sli_snapshot:
                        continue

                    value  = sli_snapshot[sli_key]
                    margin = slo.margin(value)

                    buf_sli = np.roll(self._sli_history[buf_key],    -1)
                    buf_mar = np.roll(self._margin_history[buf_key], -1)
                    buf_sli[-1] = value
                    buf_mar[-1] = margin

                    self._sli_history[buf_key]    = buf_sli
                    self._margin_history[buf_key] = buf_mar
                    self._staleness[buf_key]      = 0         # fresh observation

                else:
                    # Unprobed: freeze history, increment staleness
                    self._staleness[buf_key] += 1

    # ------------------------------------------------------------------
    # Dynamic features  (built per node every timestep)
    # ------------------------------------------------------------------

    def _build_dynamic(
        self,
        node:         str,
        probe_set:    Set[str],
        sli_snapshot: Dict[Tuple[str, str], float],
    ) -> np.ndarray:
        """
        Builds x_v^dynamic(t) for a single node.

        Layout (with N=4, K=6):
          [0]          P_t[v]
          [1 : 37]     ô_t(v)       — K × (N+2) SLO slots
          [37 : 43]    staleness(v) — K staleness counters
          [43]         incoming_call_rate
          [44]         outgoing_call_rate
        """
        N    = self.window_size
        slot = N + 2    # N history + margin + Δmargin
        vec  = np.zeros(self._dyn_dim, dtype=np.float32)

        is_probed = node in probe_set

        # P_t[v]  [0]
        vec[0] = 1.0 if is_probed else 0.0

        # ô_t(v) SLO slots  [1 : 1 + K*(N+2)]
        obs_start = 1
        for k, slo in enumerate(SLO_CATALOG):
            slot_start = obs_start + k * slot

            if not is_probed or node not in slo.nodes:
                # Masked — leave zeros
                continue

            buf_sli = self._sli_history.get((node, slo.id), np.zeros(N))
            buf_mar = self._margin_history.get((node, slo.id), np.zeros(N))

            # Rolling SLI window  [slot_start : slot_start+N]
            vec[slot_start : slot_start + N] = buf_sli

            # margin(k, t)  [slot_start + N]
            current_margin = float(buf_mar[-1])
            vec[slot_start + N] = current_margin

            # Δmargin(k, t) = margin(t) − margin(t−1)  [slot_start + N + 1]
            prev_margin = float(buf_mar[-2]) if N >= 2 else 0.0
            vec[slot_start + N + 1] = current_margin - prev_margin

        # Staleness counters  [staleness_start : staleness_start + K]
        # Populated for ALL nodes (not just probed) — the agent always knows
        # how stale each SLO observation is.
        st_start = self._staleness_offset
        for k, slo in enumerate(SLO_CATALOG):
            if node in slo.nodes:
                vec[st_start + k] = float(self._staleness.get((node, slo.id), 0))
            # else: node doesn't cover this SLO → staleness stays 0 (meaningless)

        # Call rates — always observable regardless of probe placement
        vec[-2] = float(sli_snapshot.get((node, "incoming_call_rate"), 0.0))
        vec[-1] = float(sli_snapshot.get((node, "outgoing_call_rate"), 0.0))

        return vec

    # ------------------------------------------------------------------
    # SLO health vector  H_t
    # ------------------------------------------------------------------

    def _build_slo_health(
        self,
        probe_set:    Set[str],
        sli_snapshot: Dict[Tuple[str, str], float],
    ) -> np.ndarray:
        """
        H_t[k] = 1  iff  SLO k is covered AND not violated.
                = 0  otherwise (unprobed, violated, or candidate absent).
        """
        health = np.zeros(NUM_SLOS, dtype=np.float32)
        for slo in SLO_CATALOG:
            if slo.id not in self._ep.coverable_slos:
                continue
            covered = any(node in probe_set for node in slo.nodes)
            if not covered:
                continue
            for node in slo.nodes:
                key = (node, slo.metric)
                if key in sli_snapshot:
                    if not slo.is_violated(sli_snapshot[key]):
                        health[slo.id] = 1.0
        return health