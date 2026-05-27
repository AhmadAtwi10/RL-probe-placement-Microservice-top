"""
sli_generator.py
----------------
Generates per-node SLI time-series for a single episode.

Each timestep t ∈ [0, T) produces a complete SLI snapshot:
    sli_snapshot[node][metric] = float

The signal model per metric is:

    SLI(v, metric, t) = baseline(v, metric)
                        * diurnal_factor(t)
                        * (1 + gaussian_noise(sigma=NOISE_STD))

Where:
  - baseline   : resting value under normal load (defined per node/metric below)
  - diurnal    : sine-wave traffic pattern over the episode length
                 (ramps up in the first half, cools in the second half)
  - noise      : small Gaussian jitter (sigma = 5% of baseline) to avoid
                 perfectly flat traces

The generator produces the GROUND TRUTH signal — i.e., SLI values for
ALL present nodes, including unprobed ones. The environment uses this
ground truth internally to compute blind violation rewards. The agent
only observes the probed subset.

The FailureInjector (failure_injector.py) mutates these values on top
of the clean signal to inject anomalies.

Design constraints:
  - Deterministic given a seed (reproducible episodes)
  - Vectorised over timesteps (pre-allocates full episode arrays)
  - SLI metric names match node_config.py slis lists exactly
  - Values are always physically plausible (non-negative, bounded where needed)
"""

import math
import random
from typing import Dict, List, Optional

import numpy as np

from src.simulator.config.node_config import NODE_CATALOG
from src.simulator.graph.episode_graph import EpisodeGraph


# ---------------------------------------------------------------------------
# Noise standard deviation (fraction of baseline value)
# ---------------------------------------------------------------------------
NOISE_STD = 0.05   # 5% Gaussian jitter


# ---------------------------------------------------------------------------
# Explicit set of all percentage metrics (must stay in [0, 100])
#
# Using an exact-match set rather than substring keywords avoids false
# positives on metrics like incoming_call_rate / outgoing_call_rate which
# contain "rate" but are measured in requests/sec, not percent.
# ---------------------------------------------------------------------------
PERCENT_METRICS: frozenset = frozenset({
    "error_rate",
    "failure_rate",
    "hit_rate",
    "miss_rate",
    "eviction_rate",
    "cpu_utilization",
    "memory_utilization",
    "disk_io_utilization",
    "connection_pool_utilization",
    "trace_drop_rate",       # jaeger: % of traces dropped
})


# ---------------------------------------------------------------------------
# Baseline SLI values under normal load
# Each entry: node_name → {metric_name → baseline_value}
#
# Units:
#   latency metrics   : milliseconds (ms)
#   rate metrics      : requests/sec or calls/sec
#   utilization/rate  : percentage (0–100)
#   queue depth       : integer count
# ---------------------------------------------------------------------------
BASELINES: Dict[str, Dict[str, float]] = {

    # ── Gateway ──────────────────────────────────────────────────────────────
    "frontend": {
        "request_rate":        150.0,   # req/s
        "p99_latency":          80.0,   # ms  (well below SLO; frontend aggregates all)
        "p50_latency":          25.0,   # ms
        "error_rate":            0.1,   # %
        "active_connections":   40.0,   # count
        "incoming_call_rate":  150.0,   # calls/s
        "outgoing_call_rate":  150.0,   # calls/s (fans out to 7 services)
    },

    # ── Business logic — core ────────────────────────────────────────────────
    "search": {
        "search_latency_p99":   80.0,   # ms  (SLO threshold = 200ms)
        "search_latency_p50":   20.0,   # ms
        "result_count":         10.0,   # avg results per query
        "incoming_call_rate":  150.0,
        "outgoing_call_rate":  300.0,   # fans out to geo + rate
    },
    "geo": {
        "processing_time_p99":  20.0,   # ms  (SLO threshold = 50ms)
        "failure_rate":          0.1,   # %
        "cpu_utilization":      30.0,   # %
        "incoming_call_rate":  150.0,
        "outgoing_call_rate":  150.0,
    },
    "rate": {
        "processing_time_p99":  15.0,   # ms
        "failure_rate":          0.1,   # %
        "request_queue_depth":   2.0,   # count
        "incoming_call_rate":  150.0,
        "outgoing_call_rate":  150.0,
    },
    "profile": {
        "processing_time_p99":  25.0,   # ms
        "failure_rate":          0.1,   # %
        "cpu_utilization":      25.0,   # %
        "incoming_call_rate":  150.0,
        "outgoing_call_rate":  150.0,
    },
    "recommendation": {
        "recommendation_latency_p99": 30.0,  # ms
        "failure_rate":          0.1,   # %
        "incoming_call_rate":  150.0,
        "outgoing_call_rate":  150.0,
    },
    "reservation": {
        "processing_time_p99":  80.0,   # ms  (SLO threshold = 300ms)
        "failure_rate":          0.2,   # %   (SLO threshold = 0.5%)
        "request_queue_depth":   3.0,
        "incoming_call_rate":  150.0,
        "outgoing_call_rate":  150.0,
    },
    "user": {
        "processing_time_p99":  20.0,   # ms
        "failure_rate":          0.1,   # %
        "cpu_utilization":      20.0,   # %
        "incoming_call_rate":  150.0,
        "outgoing_call_rate":  150.0,
    },

    # ── Business logic — non-core ────────────────────────────────────────────
    "review": {
        "processing_time_p99":  35.0,
        "failure_rate":          0.1,
        "cpu_utilization":      20.0,
        "incoming_call_rate":  100.0,
        "outgoing_call_rate":  100.0,
    },
    "attractions": {
        "processing_time_p99":  30.0,
        "failure_rate":          0.1,
        "cpu_utilization":      15.0,
        "incoming_call_rate":  100.0,
        "outgoing_call_rate":  100.0,
    },

    # ── MongoDB data stores ──────────────────────────────────────────────────
    # All mongo instances share the same baseline shape;
    # mongodb-rate is tied to SLO_4 (threshold = 10ms)
    "mongodb-geo": {
        "query_latency_p99":         3.0,   # ms
        "write_latency_p99":         5.0,   # ms
        "connection_pool_utilization": 20.0,  # %
        "disk_io_utilization":       15.0,  # %
    },
    "mongodb-profile": {
        "query_latency_p99":         3.0,
        "write_latency_p99":         5.0,
        "connection_pool_utilization": 20.0,
        "disk_io_utilization":       15.0,
    },
    "mongodb-rate": {
        "query_latency_p99":         3.0,   # ms  (SLO_4 threshold = 10ms)
        "write_latency_p99":         5.0,
        "connection_pool_utilization": 20.0,
        "disk_io_utilization":       15.0,
    },
    "mongodb-recommendation": {
        "query_latency_p99":         3.0,
        "write_latency_p99":         5.0,
        "connection_pool_utilization": 20.0,
        "disk_io_utilization":       15.0,
    },
    "mongodb-reservation": {
        "query_latency_p99":         4.0,
        "write_latency_p99":         6.0,
        "connection_pool_utilization": 25.0,
        "disk_io_utilization":       20.0,
    },
    "mongodb-user": {
        "query_latency_p99":         3.0,
        "write_latency_p99":         5.0,
        "connection_pool_utilization": 20.0,
        "disk_io_utilization":       15.0,
    },
    "mongodb-review": {
        "query_latency_p99":         3.0,
        "write_latency_p99":         5.0,
        "connection_pool_utilization": 20.0,
        "disk_io_utilization":       15.0,
    },
    "mongodb-attractions": {
        "query_latency_p99":         3.0,
        "write_latency_p99":         5.0,
        "connection_pool_utilization": 20.0,
        "disk_io_utilization":       15.0,
    },

    # ── Memcached caches ─────────────────────────────────────────────────────
    # memcached-rate tied to SLO_5 (hit_rate >= 90%)
    "memcached-rate": {
        "hit_rate":           97.0,   # %  (SLO_5 threshold = 90%; headroom for noise+diurnal)
        "miss_rate":           5.0,   # %
        "eviction_rate":       1.0,   # %
        "memory_utilization": 40.0,   # %
    },
    "memcached-profile": {
        "hit_rate":           97.0,
        "miss_rate":           5.0,
        "eviction_rate":       1.0,
        "memory_utilization": 40.0,
    },
    "memcached-reserve": {
        "hit_rate":           97.0,
        "miss_rate":           5.0,
        "eviction_rate":       1.0,
        "memory_utilization": 40.0,
    },
    "memcached-review": {
        "hit_rate":           97.0,
        "miss_rate":           5.0,
        "eviction_rate":       1.0,
        "memory_utilization": 40.0,
    },

    # ── Infrastructure (non-probeable but present in graph) ──────────────────
    "consul": {
        "request_rate":   50.0,
        "p99_latency":     2.0,   # ms
        "error_rate":      0.0,   # %
    },
    "jaeger": {
        "trace_ingestion_rate": 500.0,  # traces/s
        "trace_drop_rate":        0.0,  # %
    },
}


# ---------------------------------------------------------------------------
# Diurnal pattern  (traffic ramp over the episode)
# ---------------------------------------------------------------------------

def _diurnal_factor(t: int, T: int, amplitude: float = 0.3) -> float:
    """
    Returns a multiplicative traffic factor in [1-amplitude, 1+amplitude].

    Models a single traffic peak at the midpoint of the episode:
      t=0       → factor ≈ 1 - amplitude  (low traffic at start)
      t=T//2    → factor = 1 + amplitude  (peak)
      t=T-1     → factor ≈ 1 - amplitude  (cool-down)

    amplitude=0.3 means ±30% variation around the baseline.
    """
    phase = math.pi * t / max(T - 1, 1)   # 0 → π over episode
    return 1.0 + amplitude * math.sin(phase)


# ---------------------------------------------------------------------------
# SLIGenerator
# ---------------------------------------------------------------------------

class SLIGenerator:
    """
    Generates ground-truth SLI traces for all present nodes in an episode.

    Pre-allocates full episode arrays at construction time (call once per
    env.reset()). The FailureInjector then mutates a copy of these arrays.

    Parameters
    ----------
    episode_graph : EpisodeGraph
        The episode graph for this episode (which nodes are present).
    episode_length : int
        Number of timesteps T in the episode.
    diurnal_amplitude : float
        Amplitude of the sinusoidal traffic pattern (default 0.3 = ±30%).
    seed : int or None
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        episode_graph: EpisodeGraph,
        episode_length: int,
        diurnal_amplitude: float = 0.3,
        seed: Optional[int] = None,
    ):
        self.episode_graph = episode_graph
        self.T = episode_length
        self.diurnal_amplitude = diurnal_amplitude
        self._rng = np.random.default_rng(seed)

        # Pre-compute clean traces for all present nodes
        # Structure: {node: {metric: np.ndarray shape (T,)}}
        self._clean_traces: Dict[str, Dict[str, np.ndarray]] = {}
        self._generate_all()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def snapshot(self, t: int) -> Dict[str, Dict[str, float]]:
        """
        Returns the clean SLI snapshot at timestep t.
        Structure: {node_name: {metric_name: value}}

        This is the ground-truth signal BEFORE failure injection.
        The env should use FailureInjector.apply() on top of this.
        """
        assert 0 <= t < self.T, f"Timestep {t} out of range [0, {self.T})"
        return {
            node: {metric: float(arr[t]) for metric, arr in metrics.items()}
            for node, metrics in self._clean_traces.items()
        }

    def full_traces(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Returns the full pre-computed trace arrays for all nodes.
        Structure: {node_name: {metric_name: np.ndarray shape (T,)}}

        Used by FailureInjector to build a mutable copy.
        """
        return {
            node: {metric: arr.copy() for metric, arr in metrics.items()}
            for node, metrics in self._clean_traces.items()
        }

    def nodes(self) -> List[str]:
        """Returns the list of nodes for which traces were generated."""
        return list(self._clean_traces.keys())

    # ------------------------------------------------------------------
    # Internal generation
    # ------------------------------------------------------------------

    # Metrics that should NOT scale up with traffic (higher traffic = lower hit rate, etc.)
    # These get an inverse diurnal: dip slightly at peak traffic, stable otherwise
    _INVERSE_DIURNAL_METRICS = frozenset(["hit_rate", "miss_rate", "eviction_rate"])

    def _generate_all(self) -> None:
        """Pre-computes clean SLI traces for all present nodes."""
        diurnal = np.array([
            _diurnal_factor(t, self.T, self.diurnal_amplitude)
            for t in range(self.T)
        ])
        # Inverse diurnal: hit_rate dips slightly at peak traffic (mild, 5% amplitude)
        inverse_diurnal = np.array([
            _diurnal_factor(t, self.T, amplitude=-0.05)
            for t in range(self.T)
        ])

        for node in self.episode_graph.present_nodes:
            if node not in BASELINES:
                # Safety: skip nodes with no defined baseline (shouldn't happen)
                continue

            node_traces: Dict[str, np.ndarray] = {}
            for metric, baseline in BASELINES[node].items():
                # Gaussian noise: σ = NOISE_STD × baseline
                noise = self._rng.normal(
                    loc=0.0,
                    scale=NOISE_STD * baseline,
                    size=self.T,
                )
                # Choose diurnal direction based on metric semantics
                # Inverse metrics (hit_rate etc.) are flat — no traffic scaling
                # Clamp noise to ±2% of baseline for tightly-bounded metrics
                if metric in self._INVERSE_DIURNAL_METRICS:
                    clamped_noise = np.clip(noise, -0.02 * baseline, 0.02 * baseline)
                    raw = baseline + clamped_noise
                else:
                    raw = baseline * diurnal + noise

                # Apply physical bounds per metric type
                raw = self._apply_bounds(metric, baseline, raw)
                node_traces[metric] = raw

            self._clean_traces[node] = node_traces

    @staticmethod
    def _apply_bounds(metric: str, baseline: float, arr: np.ndarray) -> np.ndarray:
        """
        Clamps values to physically plausible ranges based on metric type.
        Rules:
          - All metrics          : non-negative  (lower bound = 0)
          - PERCENT_METRICS      : upper bound = 100
          - count / depth / conn : round to nearest integer
        """
        arr = np.maximum(arr, 0.0)

        if metric in PERCENT_METRICS or metric.endswith("_utilization"):
            arr = np.minimum(arr, 100.0)

        if "count" in metric or "depth" in metric or "connections" in metric:
            arr = np.round(arr)

        return arr