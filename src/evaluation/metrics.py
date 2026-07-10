"""
metrics.py
----------
Metric definitions for evaluating the probe placement agent.

─────────────────────────────────────────────────────────────────
Per-episode metrics (EpisodeResult)
─────────────────────────────────────────────────────────────────
Collected during a single deterministic or stochastic episode:

  total_reward     : Σ r_t over the episode
  episode_length   : number of timesteps T_ep
  survived         : True if truncated (reached T), False if terminated
                     (blind violations exceeded K)
  cum_blind        : total blind violation events Σ_t |blind_slos_t|
  mean_probe_count : mean |P_t| per step — average instrumentation cost
  coverage_rate    : fraction of steps with at least one SLO covered
  blind_rate       : fraction of steps with at least one blind violation
  probe_efficiency : coverage_rate / max(mean_probe_count, 1)
                     — coverage per unit of probe cost
  weighted_coverage: mean over steps of Σ_{k covered} w_k / Σ_{k coverable} w_k
                     — coverage weighted by SLO importance, with the
                     denominator taken over SLOs coverable this episode
                     (dynamic; excludes SLOs whose candidate node was perturbed out)
  slo_coverage     : List[float] length NUM_SLOS
                     per-SLO fraction of steps where SLO k is covered
  slo_blind        : List[float] length NUM_SLOS
                     per-SLO fraction of steps with blind violation on k

─────────────────────────────────────────────────────────────────
Aggregate metrics (AggregateMetrics)
─────────────────────────────────────────────────────────────────
Mean and std of per-episode metrics over N evaluation episodes.
Also includes:
  survival_rate    : fraction of episodes that survived (not terminated)
  n_episodes       : number of episodes evaluated

─────────────────────────────────────────────────────────────────
Public API
─────────────────────────────────────────────────────────────────
  EpisodeResult        — dataclass, one per episode
  AggregateMetrics     — dataclass, summary over N episodes
  aggregate(results)   → AggregateMetrics
  format_summary(agg)  → human-readable string
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from src.simulator.config.slo_config import NUM_SLOS, SLO_CATALOG

# Total SLO weight (denominator for weighted coverage)
_TOTAL_WEIGHT = sum(slo.weight for slo in SLO_CATALOG)


# ---------------------------------------------------------------------------
# EpisodeResult
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    """
    All metrics from a single evaluation episode.

    Attributes
    ----------
    total_reward       : float   Σ r_t
    episode_length     : int     number of timesteps
    survived           : bool    True = truncated, False = terminated
    cum_blind          : int     Σ_t |blind_slos_t|
    mean_probe_count   : float   mean |P_t| per step
    coverage_rate      : float   fraction of steps with ≥1 SLO covered
    blind_rate         : float   fraction of steps with ≥1 blind violation
    probe_efficiency   : float   coverage_rate / max(mean_probe_count, 1)
    weighted_coverage  : float   importance-weighted coverage rate (dynamic denom)
    slo_coverage       : List[float]  per-SLO coverage fractions (NUM_SLOS,)
    slo_blind          : List[float]  per-SLO blind violation fractions
    """
    total_reward:      float
    episode_length:    int
    survived:          bool
    cum_blind:         int
    mean_probe_count:  float
    coverage_rate:     float
    blind_rate:        float
    probe_efficiency:  float
    weighted_coverage: float
    slo_coverage:      List[float] = field(default_factory=lambda: [0.0]*NUM_SLOS)
    slo_blind:         List[float] = field(default_factory=lambda: [0.0]*NUM_SLOS)
    # --- new fractional / violation-based metrics ---
    coverage_frac:        float = 0.0            # mean_t |covered| / |coverable|  (dynamic denom)
    detection_rate:       float = float("nan")   # Σ_t covered_viol / Σ_t total_viol  (nan if no violations)
    blind_violation_rate: float = float("nan")   # Σ_t blind_viol   / Σ_t total_viol  (= 1 − detection_rate)
    n_violation_events:   int   = 0              # Σ_t (number of coverable-SLO violations)


# ---------------------------------------------------------------------------
# AggregateMetrics
# ---------------------------------------------------------------------------

@dataclass
class AggregateMetrics:
    """
    Summary statistics over N evaluation episodes.

    All scalar fields have a corresponding _std field.
    """
    n_episodes:          int

    mean_reward:         float
    std_reward:          float

    survival_rate:       float       # fraction of episodes survived

    mean_ep_length:      float
    std_ep_length:       float

    mean_cum_blind:      float
    std_cum_blind:       float

    mean_probe_count:    float
    std_probe_count:     float

    mean_coverage_rate:  float
    std_coverage_rate:   float

    mean_blind_rate:     float
    std_blind_rate:      float

    mean_probe_eff:      float
    std_probe_eff:       float

    mean_weighted_cov:   float
    std_weighted_cov:    float

    # --- new fractional / violation-based metrics ---
    mean_coverage_frac:   float = 0.0
    std_coverage_frac:    float = 0.0
    mean_detection_rate:  float = float("nan")   # averaged over episodes that had violations
    std_detection_rate:   float = float("nan")
    mean_blind_viol_rate: float = float("nan")
    std_blind_viol_rate:  float = float("nan")
    n_episodes_with_violations: int = 0

    # Per-SLO means (length NUM_SLOS)
    mean_slo_coverage:   List[float] = field(default_factory=lambda: [0.0]*NUM_SLOS)
    mean_slo_blind:      List[float] = field(default_factory=lambda: [0.0]*NUM_SLOS)


# ---------------------------------------------------------------------------
# aggregate()
# ---------------------------------------------------------------------------

def aggregate(results: List[EpisodeResult]) -> AggregateMetrics:
    """
    Compute aggregate statistics over a list of EpisodeResult objects.

    Parameters
    ----------
    results : List[EpisodeResult]
        Per-episode results from evaluate().

    Returns
    -------
    AggregateMetrics
    """
    assert len(results) > 0, "Need at least one episode result"

    def _mean(attr): return float(np.mean([getattr(r, attr) for r in results]))
    def _std(attr):  return float(np.std( [getattr(r, attr) for r in results]))

    # Per-SLO means
    slo_cov  = [float(np.mean([r.slo_coverage[k] for r in results]))
                for k in range(NUM_SLOS)]
    slo_blind = [float(np.mean([r.slo_blind[k]   for r in results]))
                 for k in range(NUM_SLOS)]

    # Violation-based rates: average only over episodes that actually had
    # violations (episodes with none carry nan and are ignored by nanmean).
    _det   = np.array([r.detection_rate       for r in results], dtype=float)
    _bviol = np.array([r.blind_violation_rate for r in results], dtype=float)
    _n_with_viol = int(sum(1 for r in results if r.n_violation_events > 0))

    def _nanmean(a): return float(np.nanmean(a)) if np.any(~np.isnan(a)) else float("nan")
    def _nanstd(a):  return float(np.nanstd(a))  if np.any(~np.isnan(a)) else float("nan")

    return AggregateMetrics(
        n_episodes         = len(results),
        mean_reward        = _mean("total_reward"),
        std_reward         = _std("total_reward"),
        survival_rate      = float(np.mean([r.survived for r in results])),
        mean_ep_length     = _mean("episode_length"),
        std_ep_length      = _std("episode_length"),
        mean_cum_blind     = _mean("cum_blind"),
        std_cum_blind      = _std("cum_blind"),
        mean_probe_count   = _mean("mean_probe_count"),
        std_probe_count    = _std("mean_probe_count"),
        mean_coverage_rate = _mean("coverage_rate"),
        std_coverage_rate  = _std("coverage_rate"),
        mean_blind_rate    = _mean("blind_rate"),
        std_blind_rate     = _std("blind_rate"),
        mean_probe_eff     = _mean("probe_efficiency"),
        std_probe_eff      = _std("probe_efficiency"),
        mean_weighted_cov  = _mean("weighted_coverage"),
        std_weighted_cov   = _std("weighted_coverage"),
        mean_coverage_frac = _mean("coverage_frac"),
        std_coverage_frac  = _std("coverage_frac"),
        mean_detection_rate  = _nanmean(_det),
        std_detection_rate   = _nanstd(_det),
        mean_blind_viol_rate = _nanmean(_bviol),
        std_blind_viol_rate  = _nanstd(_bviol),
        n_episodes_with_violations = _n_with_viol,
        mean_slo_coverage  = slo_cov,
        mean_slo_blind     = slo_blind,
    )


# ---------------------------------------------------------------------------
# format_summary()
# ---------------------------------------------------------------------------

def format_summary(agg: AggregateMetrics) -> str:
    """
    Return a human-readable summary string for an AggregateMetrics object.
    """
    lines = [
        f"{'─'*55}",
        f"  Evaluation Summary  ({agg.n_episodes} episodes)",
        f"{'─'*55}",
        f"  Reward         : {agg.mean_reward:+.3f} ± {agg.std_reward:.3f}",
        f"  Survival rate  : {agg.survival_rate*100:.1f}%",
        f"  Episode length : {agg.mean_ep_length:.1f} ± {agg.std_ep_length:.1f}",
        f"  Blind events   : {agg.mean_cum_blind:.2f} ± {agg.std_cum_blind:.2f}",
        f"  Probe count    : {agg.mean_probe_count:.2f} ± {agg.std_probe_count:.2f}",
        f"  Coverage rate  : {agg.mean_coverage_rate:.3f} ± {agg.std_coverage_rate:.3f}",
        f"  Blind rate     : {agg.mean_blind_rate:.3f} ± {agg.std_blind_rate:.3f}",
        f"  Probe eff.     : {agg.mean_probe_eff:.3f} ± {agg.std_probe_eff:.3f}",
        f"  Weighted cov.  : {agg.mean_weighted_cov:.3f} ± {agg.std_weighted_cov:.3f}",
        f"  Coverage frac  : {agg.mean_coverage_frac:.3f} ± {agg.std_coverage_frac:.3f}"
        f"   (covered/coverable SLOs per step)",
        f"  Detection rate : {agg.mean_detection_rate:.3f} ± {agg.std_detection_rate:.3f}"
        f"   (covered/total violations; {agg.n_episodes_with_violations}/{agg.n_episodes} eps had violations)",
        f"  Blind v. rate  : {agg.mean_blind_viol_rate:.3f} ± {agg.std_blind_viol_rate:.3f}"
        f"   (blind/total violations)",
        f"{'─'*55}",
        f"  Per-SLO coverage / blind rate:",
    ]
    for k, slo in enumerate(SLO_CATALOG):
        lines.append(
            f"    SLO_{k} (w={slo.weight}): "
            f"cov={agg.mean_slo_coverage[k]:.3f}  "
            f"blind={agg.mean_slo_blind[k]:.3f}"
        )
    lines.append(f"{'─'*55}")
    return "\n".join(lines)