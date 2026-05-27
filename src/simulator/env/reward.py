"""
reward.py
---------
Reward function for the probe placement POMDP (§4.6 of probe_placement_formulation.md).

The reward at each timestep is:

    R_t =   Σ_k  w_k · covered(k, t)           ← (1) observability reward
          − λ  · |P_t|                           ← (2) probe overhead penalty
          − μ  · Σ_k  blind_violation(k, t)      ← (3) blind violation penalty
          − ρ  · removal_risk(a_t, t)            ← (4) risk-weighted removal penalty

where:

    covered(k, t)        = 1  iff  ∃ v probed s.t. metric_k ∈ SLI(v)
    violation(k, t)      = 1  iff  SLI_k(t) breaches (threshold_k, operator_k)
    blind_violation(k,t) = 1  iff  violation(k,t)=1  AND  covered(k,t)=0

    removal_risk(v, t)   = Σ_{k: metric_k ∈ SLI(v)}  w_k · max(0, 1 − margin(k, t))
                           (non-zero ONLY when action is remove_probe(v))

Term (4) fires immediately at removal time to give an early shaping signal;
term (3) fires later when a blind violation actually occurs.  Together they
close the temporal credit-assignment gap (§4.6, "Temporal interplay").

Public API
----------
    RewardConfig          — dataclass holding λ, μ, ρ hyperparameters
    RewardInput           — dataclass bundling all per-timestep inputs
    RewardOutput          — dataclass holding R_t and its four components
    compute_reward(...)   — pure function, no side effects
    terminal_reward(...)  — one-off bonus/penalty at episode end
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from src.simulator.config.slo_config import SLO_CATALOG, SLO_BY_ID, NODE_TO_SLOS
from src.simulator.graph.episode_graph import EpisodeGraph


# ---------------------------------------------------------------------------
# Action sentinel values
# (kept minimal — probe_env.py will use the same integers)
# ---------------------------------------------------------------------------

NO_OP        = -1   # sentinel: no change to probe set
ADD_PROBE    =  0   # action type tag
REMOVE_PROBE =  1   # action type tag


# ---------------------------------------------------------------------------
# RewardConfig  — hyperparameters (§4.6 Table)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RewardConfig:
    """
    Hyperparameters that control the relative weight of each reward term.

    Attributes
    ----------
    lam : float
        λ — probe overhead penalty per active probe.
        Higher → agent uses fewer probes overall.
    mu : float
        μ — blind violation penalty per undetected SLO breach.
        Higher → agent strongly avoids being unprobed during violations.
    rho : float
        ρ — removal risk sensitivity.
        Higher → agent hesitates to remove probes from near-threshold nodes.
    terminal_bonus : float
        Reward added at episode end when the agent survives T timesteps
        without exceeding the blind violation budget K.
    terminal_penalty : float
        Penalty applied when cumulative blind violations exceed K
        (failure termination).  Should be a large negative value.
    """
    lam:              float = 0.05
    mu:               float = 2.0
    rho:              float = 0.5
    terminal_bonus:   float = 5.0
    terminal_penalty: float = -10.0

    def __post_init__(self):
        assert self.lam >= 0, "λ must be non-negative"
        assert self.mu  >= 0, "μ must be non-negative"
        assert self.rho >= 0, "ρ must be non-negative"


# ---------------------------------------------------------------------------
# RewardInput  — all per-timestep data needed to compute R_t
# ---------------------------------------------------------------------------

@dataclass
class RewardInput:
    """
    All inputs required to compute the reward at timestep t.

    Attributes
    ----------
    probe_set : set[str]
        Names of nodes that are currently probed (P_t).
    sli_values : dict[(str, str), float]
        True SLI values for ALL present nodes at time t, keyed by
        (node, metric).  Includes unprobed nodes (ground truth used
        internally for blind violation detection; the agent never sees
        unprobed values directly).
    action_type : int
        ADD_PROBE, REMOVE_PROBE, or NO_OP.
    action_node : str or None
        The node targeted by add/remove_probe.  None for NO_OP.
    episode_graph : EpisodeGraph
        The active episode graph (needed for coverable_slos and SLO nodes).
    """
    probe_set:     Set[str]
    sli_values:    Dict[Tuple[str, str], float]   # (node, metric) → value
    action_type:   int
    action_node:   Optional[str]
    episode_graph: EpisodeGraph


# ---------------------------------------------------------------------------
# RewardOutput  — R_t decomposed into its four terms
# ---------------------------------------------------------------------------

@dataclass
class RewardOutput:
    """
    The scalar reward R_t and its four additive components.

    Attributes
    ----------
    total : float
        R_t — the scalar reward fed to the RL algorithm.
    observability : float
        Term (1): Σ_k  w_k · covered(k, t)        ≥ 0
    overhead : float
        Term (2): −λ · |P_t|                       ≤ 0
    blind_violation : float
        Term (3): −μ · Σ_k w_k · blind_violation(k, t)   ≤ 0
    removal_risk : float
        Term (4): −ρ · removal_risk(a_t, t)        ≤ 0
    covered_slos : list[int]
        SLO ids covered at this timestep (for logging).
    blind_slos : list[int]
        SLO ids with a blind violation at this timestep (for logging).
    """
    total:           float
    observability:   float
    overhead:        float
    blind_violation: float
    removal_risk:    float
    covered_slos:    List[int] = field(default_factory=list)
    blind_slos:      List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core helper functions
# ---------------------------------------------------------------------------

def _covered(slo_id: int, probe_set: set) -> bool:
    """
    covered(k, t) = 1  iff  ∃ v ∈ probe_set  s.t.  metric_k ∈ SLI(v)

    Coverage is purely about probe placement — not about the metric value.
    """
    slo = SLO_BY_ID[slo_id]
    return any(node in probe_set for node in slo.nodes)


def _violation(slo_id: int, sli_values: Dict[Tuple[str, str], float]) -> bool:
    """
    violation(k, t) = 1  iff  SLI_k(t) breaches (threshold_k, operator_k)

    Uses ground-truth SLI values (including unprobed nodes).
    Returns False if the candidate node is absent from the episode
    (no signal → no violation possible).
    """
    slo = SLO_BY_ID[slo_id]
    for node in slo.nodes:
        key = (node, slo.metric)
        if key in sli_values:
            return slo.is_violated(sli_values[key])
    return False


def _compute_removal_risk(
    node: str,
    sli_values: Dict[Tuple[str, str], float],
) -> float:
    """
    removal_risk(v, t) = Σ_{k: metric_k ∈ SLI(v)}  w_k · max(0, 1 − margin(k, t))

    Called only when action = remove_probe(v).

    max(0, 1 − margin) interpretation:
      margin ≈ 1  (safe, far from threshold) → risk contribution ≈ 0
      margin ≈ 0  (danger zone)              → risk contribution ≈ w_k
      margin < 0  (already violated)         → risk contribution > w_k
    """
    risk = 0.0
    for slo_id in NODE_TO_SLOS.get(node, []):
        slo = SLO_BY_ID[slo_id]
        key = (node, slo.metric)
        if key in sli_values:
            margin = slo.margin(sli_values[key])
            risk += slo.weight * max(0.0, 1.0 - margin)
    return risk


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_reward(inp: RewardInput, cfg: RewardConfig) -> RewardOutput:
    """
    Computes R_t from the four-term reward formula (§4.6).

    Parameters
    ----------
    inp : RewardInput
        All per-timestep inputs (probe set, SLI values, action taken).
    cfg : RewardConfig
        Hyperparameters λ, μ, ρ.

    Returns
    -------
    RewardOutput
        Scalar reward R_t and its four components, plus diagnostic fields
        (covered_slos, blind_slos) useful for logging and debugging.
    """

    # ── Term (1): Observability reward  Σ_k w_k · covered(k, t) ─────────────
    covered_slos: List[int] = []
    obs_reward = 0.0
    for slo in SLO_CATALOG:
        if slo.id not in inp.episode_graph.coverable_slos:
            continue   # candidate node absent this episode — skip entirely
        if _covered(slo.id, inp.probe_set):
            obs_reward += slo.weight
            covered_slos.append(slo.id)

    # ── Term (2): Probe overhead penalty  −λ · |P_t| ─────────────────────────
    overhead = -cfg.lam * len(inp.probe_set)

    # ── Term (3): Blind violation penalty  −μ · Σ_k w_k · blind_violation(k, t) ──
    blind_slos: List[int] = []
    blind_weight_sum = 0.0
    for slo in SLO_CATALOG:
        if slo.id not in inp.episode_graph.coverable_slos:
            continue
        if _violation(slo.id, inp.sli_values) and not _covered(slo.id, inp.probe_set):
            blind_weight_sum += slo.weight
            blind_slos.append(slo.id)
    blind_penalty = -cfg.mu * blind_weight_sum

    # ── Term (4): Risk-weighted removal penalty  −ρ · removal_risk(v, t) ──────
    # Non-zero ONLY when action = remove_probe(v)
    risk_penalty = 0.0
    if inp.action_type == REMOVE_PROBE and inp.action_node is not None:
        risk = _compute_removal_risk(inp.action_node, inp.sli_values)
        risk_penalty = -cfg.rho * risk

    # ── Total ─────────────────────────────────────────────────────────────────
    total = obs_reward + overhead + blind_penalty + risk_penalty

    return RewardOutput(
        total=total,
        observability=obs_reward,
        overhead=overhead,
        blind_violation=blind_penalty,
        removal_risk=risk_penalty,
        covered_slos=covered_slos,
        blind_slos=blind_slos,
    )


# ---------------------------------------------------------------------------
# Terminal reward  (called once at episode end by probe_env.py)
# ---------------------------------------------------------------------------

def terminal_reward(survived: bool, cfg: RewardConfig) -> float:
    """
    One-off terminal reward at episode end.

    Parameters
    ----------
    survived : bool
        True  → normal termination (reached T without exceeding budget K).
        False → failure termination (blind violations exceeded K).
    """
    return cfg.terminal_bonus if survived else cfg.terminal_penalty