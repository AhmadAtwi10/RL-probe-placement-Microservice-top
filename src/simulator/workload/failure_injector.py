"""
failure_injector.py
-------------------
Injects failure scenarios into SLI traces produced by SLIGenerator.

Three failure modes:

  1. LATENCY_SPIKE   — a node's latency metric jumps sharply above normal,
                       potentially breaching its SLO threshold.

  2. ERROR_BURST     — a node's failure_rate suddenly rises, potentially
                       breaching its SLO threshold.

  3. CACHE_DEGRADE   — a memcached node's hit_rate drops sharply, potentially
                       breaching SLO_5.

All failures are modelled as a trapezoid pulse:
  - Linear ramp-up   over `ramp_steps` timesteps
  - Flat hold        at peak magnitude for `duration` timesteps
  - Linear ramp-down over `ramp_steps` timesteps

Causal propagation (critical for realism):
  When a downstream node (e.g. `geo`) degrades, its upstream callers
  (e.g. `search`, `frontend`) absorb a fraction of that degradation via
  PROPAGATION_FACTOR. This is applied transitively one hop at a time
  through the call graph.

  Example:
    geo.processing_time_p99 spikes by +40ms
    -> search.search_latency_p99 += 40ms * 0.6  (= +24ms)
    -> frontend.p99_latency      += 24ms * 0.6  (= +14.4ms)

Usage
-----
    generator = SLIGenerator(ep_graph, T=100, seed=42)
    injector  = FailureInjector(ep_graph, seed=42)

    # Get mutable copy of clean traces
    traces = generator.full_traces()

    # Inject failures
    events = injector.sample_failures(n_failures=2)
    traces = injector.apply(traces, events)

    # traces now contains degraded SLI values for the full episode
    snapshot_t = {node: {m: traces[node][m][t] for m in traces[node]}
                  for node in traces}
"""

import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.simulator.config.node_config import NODE_CATALOG
from src.simulator.config.slo_config import SLO_CATALOG, SLO_BY_ID
from src.simulator.graph.episode_graph import EpisodeGraph
from src.simulator.graph.topology import get_upstream_nodes
from src.simulator.workload.sli_generator import BASELINES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fraction of a downstream node's latency spike absorbed by each upstream caller
PROPAGATION_FACTOR = 0.6

# Ramp steps for the trapezoid pulse (timesteps)
DEFAULT_RAMP_STEPS = 3


# ---------------------------------------------------------------------------
# Failure mode enum and event dataclass
# ---------------------------------------------------------------------------

class FailureMode(Enum):
    LATENCY_SPIKE  = auto()
    ERROR_BURST    = auto()
    CACHE_DEGRADE  = auto()


@dataclass
class FailureEvent:
    """
    A single failure event injected into the episode.

    Attributes
    ----------
    node        : target node name
    mode        : which failure mode
    start_t     : first timestep of ramp-up
    duration    : timesteps at peak (flat top of trapezoid)
    magnitude   : peak delta added to (or subtracted from) the metric value
    metric      : the specific metric key being perturbed
    ramp_steps  : timesteps for linear ramp-up and ramp-down
    """
    node:       str
    mode:       FailureMode
    start_t:    int
    duration:   int
    magnitude:  float
    metric:     str
    ramp_steps: int = DEFAULT_RAMP_STEPS


# ---------------------------------------------------------------------------
# Trapezoid pulse builder
# ---------------------------------------------------------------------------

def _trapezoid_pulse(
    T: int,
    start_t: int,
    duration: int,
    magnitude: float,
    ramp_steps: int,
) -> np.ndarray:
    """
    Returns a 1-D array of length T representing a trapezoid pulse:

      0 ... 0 | /ramp/ flat /ramp/ | 0 ... 0
              start_t            end_t

    Values are in [0, magnitude].
    """
    pulse = np.zeros(T)
    ramp  = min(ramp_steps, duration // 2)

    ramp_up_start  = start_t
    flat_start     = start_t + ramp
    flat_end       = flat_start + duration
    ramp_down_end  = flat_end + ramp

    for t in range(T):
        if ramp_up_start <= t < flat_start:
            frac = (t - ramp_up_start) / max(ramp, 1)
            pulse[t] = magnitude * frac
        elif flat_start <= t < flat_end:
            pulse[t] = magnitude
        elif flat_end <= t < ramp_down_end:
            frac = 1.0 - (t - flat_end) / max(ramp, 1)
            pulse[t] = magnitude * max(frac, 0.0)

    return pulse


# ---------------------------------------------------------------------------
# FailureInjector
# ---------------------------------------------------------------------------

class FailureInjector:
    """
    Samples and applies failure events to SLI trace dictionaries.

    Parameters
    ----------
    episode_graph   : EpisodeGraph  — active episode (determines valid targets)
    episode_length  : int           — T (number of timesteps)
    seed            : int or None   — for reproducibility
    propagate       : bool          — enable causal propagation (default True)
    """

    def __init__(
        self,
        episode_graph: EpisodeGraph,
        episode_length: int,
        seed: Optional[int] = None,
        propagate: bool = True,
    ):
        self.ep      = episode_graph
        self.T       = episode_length
        self.propagate = propagate
        self._rng    = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        # Build candidate target lists per failure mode
        self._latency_targets  = self._find_latency_targets()
        self._error_targets    = self._find_error_targets()
        self._cache_targets    = self._find_cache_targets()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_failures(
        self,
        n_failures: int = 1,
        min_start: int = 5,
        duration_range: Tuple[int, int] = (8, 20),
    ) -> List[FailureEvent]:
        """
        Randomly samples `n_failures` FailureEvent objects.

        Parameters
        ----------
        n_failures     : number of failure events to inject
        min_start      : earliest possible start timestep (give agent time to react)
        duration_range : (min, max) flat-top duration in timesteps
        """
        events: List[FailureEvent] = []

        for _ in range(n_failures):
            mode = self._rng.choice([
                FailureMode.LATENCY_SPIKE,
                FailureMode.ERROR_BURST,
                FailureMode.CACHE_DEGRADE,
            ])
            event = self._sample_event(mode, min_start, duration_range)
            if event is not None:
                events.append(event)

        return events

    def apply(
        self,
        traces: Dict[str, Dict[str, np.ndarray]],
        events: List[FailureEvent],
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Applies all FailureEvents to the trace dict (mutates in place).

        Parameters
        ----------
        traces : mutable dict from SLIGenerator.full_traces()
        events : list of FailureEvent from sample_failures()

        Returns the mutated traces dict.
        """
        for event in events:
            self._apply_event(traces, event)
        return traces

    # ------------------------------------------------------------------
    # Event sampling helpers
    # ------------------------------------------------------------------

    def _sample_event(
        self,
        mode: FailureMode,
        min_start: int,
        duration_range: Tuple[int, int],
    ) -> Optional[FailureEvent]:
        """Samples a single FailureEvent for the given mode."""

        duration   = self._rng.randint(*duration_range)
        latest_start = self.T - duration - DEFAULT_RAMP_STEPS * 2 - 1
        if latest_start < min_start:
            return None
        start_t = self._rng.randint(min_start, latest_start)

        if mode == FailureMode.LATENCY_SPIKE:
            return self._sample_latency_spike(start_t, duration)
        elif mode == FailureMode.ERROR_BURST:
            return self._sample_error_burst(start_t, duration)
        else:
            return self._sample_cache_degrade(start_t, duration)

    def _sample_latency_spike(self, start_t: int, duration: int) -> Optional[FailureEvent]:
        """
        Samples a latency spike on a random latency-capable present node.
        Magnitude is chosen to push the metric above its SLO threshold.
        """
        if not self._latency_targets:
            return None

        node, metric, slo_id = self._rng.choice(self._latency_targets)
        slo       = SLO_BY_ID[slo_id]
        # Spike to 1.2–2.0× the SLO threshold (guarantees a violation)
        target    = slo.threshold * self._rng.uniform(1.2, 2.0)

        # Get current baseline at node
        baseline  = BASELINES.get(node, {}).get(metric, slo.threshold)
        magnitude = max(target - baseline, 1.0)

        return FailureEvent(
            node=node, mode=FailureMode.LATENCY_SPIKE,
            start_t=start_t, duration=duration,
            magnitude=magnitude, metric=metric,
        )

    def _sample_error_burst(self, start_t: int, duration: int) -> Optional[FailureEvent]:
        """
        Samples an error burst on a random failure-rate capable present node.
        """
        if not self._error_targets:
            return None

        node, metric, slo_id = self._rng.choice(self._error_targets)
        slo      = SLO_BY_ID[slo_id]
        # Burst to 1.5–3× the SLO threshold
        target   = slo.threshold * self._rng.uniform(1.5, 3.0)

        from src.simulator.workload.sli_generator import BASELINES
        baseline = BASELINES.get(node, {}).get(metric, 0.1)
        magnitude = max(target - baseline, 0.1)

        return FailureEvent(
            node=node, mode=FailureMode.ERROR_BURST,
            start_t=start_t, duration=duration,
            magnitude=magnitude, metric=metric,
        )

    def _sample_cache_degrade(self, start_t: int, duration: int) -> Optional[FailureEvent]:
        """
        Samples a cache degradation on memcached-rate (the only cache SLO node).
        Hit rate drops below the SLO threshold.
        """
        if not self._cache_targets:
            return None

        node, metric, slo_id = self._rng.choice(self._cache_targets)
        slo      = SLO_BY_ID[slo_id]
        # Drop to 0.5–0.9× the SLO threshold (guarantees violation)
        target   = slo.threshold * self._rng.uniform(0.5, 0.9)

        from src.simulator.workload.sli_generator import BASELINES
        baseline  = BASELINES.get(node, {}).get(metric, 95.0)
        # magnitude is how much to SUBTRACT (hit_rate drops)
        magnitude = max(baseline - target, 1.0)

        return FailureEvent(
            node=node, mode=FailureMode.CACHE_DEGRADE,
            start_t=start_t, duration=duration,
            magnitude=magnitude, metric=metric,
        )

    # ------------------------------------------------------------------
    # Event application
    # ------------------------------------------------------------------

    def _apply_event(
        self,
        traces: Dict[str, Dict[str, np.ndarray]],
        event: FailureEvent,
    ) -> None:
        """Applies a single FailureEvent to traces, with causal propagation."""

        pulse = _trapezoid_pulse(
            self.T, event.start_t, event.duration,
            event.magnitude, event.ramp_steps,
        )

        if event.node not in traces or event.metric not in traces[event.node]:
            return

        # Apply primary failure
        if event.mode == FailureMode.CACHE_DEGRADE:
            # Cache: subtract from hit_rate, clamp to [0, 100]
            traces[event.node][event.metric] -= pulse
            traces[event.node][event.metric] = np.maximum(
                traces[event.node][event.metric], 0.0
            )
        else:
            # Latency / error: add to metric
            traces[event.node][event.metric] += pulse

        # Causal propagation through call graph
        if self.propagate and event.mode == FailureMode.LATENCY_SPIKE:
            self._propagate_latency(traces, event.node, pulse)

    def _propagate_latency(
        self,
        traces: Dict[str, Dict[str, np.ndarray]],
        failed_node: str,
        original_pulse: np.ndarray,
    ) -> None:
        """
        Propagates a latency spike upstream through the call graph.

        For each upstream caller of `failed_node`:
          propagated_delta = original_pulse * PROPAGATION_FACTOR^hop_distance

        Only affects latency-type metrics on upstream nodes.
        Propagation stops when PROPAGATION_FACTOR^hop < 0.05 (< 5% signal).
        """
        G = self.ep.G
        visited = {failed_node}
        # BFS over predecessors in the call graph
        frontier = [(failed_node, original_pulse)]

        while frontier:
            next_frontier = []
            for (source, current_pulse) in frontier:
                for predecessor in G.predecessors(source):
                    edge_type = G[predecessor][source].get("edge_type", "")
                    if edge_type != "call":
                        continue
                    if predecessor in visited:
                        continue
                    visited.add(predecessor)

                    propagated = current_pulse * PROPAGATION_FACTOR
                    if propagated.max() < 0.05 * original_pulse.max():
                        continue  # signal too small, stop propagation

                    # Apply to the primary latency metric of the predecessor
                    lat_metric = self._primary_latency_metric(predecessor)
                    if lat_metric and predecessor in traces and lat_metric in traces[predecessor]:
                        traces[predecessor][lat_metric] += propagated

                    next_frontier.append((predecessor, propagated))

            frontier = next_frontier

    # ------------------------------------------------------------------
    # Target list builders
    # ------------------------------------------------------------------

    def _find_latency_targets(self) -> List[Tuple[str, str, int]]:
        """
        Returns (node, metric, slo_id) for all latency SLOs whose candidate
        node is present and probeable in this episode.
        Latency SLOs: ids 0 (search), 1 (geo), 2 (reservation), 4 (mongodb-rate).
        """
        targets = []
        latency_slo_ids = [0, 1, 2, 4]
        metric_map = {
            0: ("search",       "search_latency_p99"),
            1: ("geo",          "processing_time_p99"),
            2: ("reservation",  "processing_time_p99"),
            4: ("mongodb-rate", "query_latency_p99"),
        }
        for slo_id in latency_slo_ids:
            node, metric = metric_map[slo_id]
            if self.ep.is_present(node):
                targets.append((node, metric, slo_id))
        return targets

    def _find_error_targets(self) -> List[Tuple[str, str, int]]:
        """
        Returns (node, metric, slo_id) for the error-rate SLO (id=3).
        """
        targets = []
        if self.ep.is_present("reservation"):
            targets.append(("reservation", "failure_rate", 3))
        return targets

    def _find_cache_targets(self) -> List[Tuple[str, str, int]]:
        """
        Returns (node, metric, slo_id) for the cache SLO (id=5).
        """
        targets = []
        if self.ep.is_present("memcached-rate"):
            targets.append(("memcached-rate", "hit_rate", 5))
        return targets

    @staticmethod
    def _primary_latency_metric(node: str) -> Optional[str]:
        """
        Returns the primary latency metric name for upstream propagation,
        or None if the node has no latency metric.
        """
        latency_metrics = {
            "frontend":       "p99_latency",
            "search":         "search_latency_p99",
            "geo":            "processing_time_p99",
            "rate":           "processing_time_p99",
            "profile":        "processing_time_p99",
            "recommendation": "recommendation_latency_p99",
            "reservation":    "processing_time_p99",
            "user":           "processing_time_p99",
            "review":         "processing_time_p99",
            "attractions":    "processing_time_p99",
        }
        return latency_metrics.get(node)