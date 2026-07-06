"""
evaluate.py
-----------
Deterministic evaluation of the probe placement agent.

Runs N complete episodes with the policy in eval mode and collects
per-step and per-episode metrics defined in metrics.py.

─────────────────────────────────────────────────────────────────
Public API
─────────────────────────────────────────────────────────────────
  EvalConfig                 — evaluation hyperparameters
  evaluate(policy, config)   → List[EpisodeResult]
  evaluate_and_aggregate(policy, config) → AggregateMetrics
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from src.simulator.env.probe_env import ProbeEnv, ProbeEnvConfig
from src.simulator.config.slo_config import NUM_SLOS, SLO_CATALOG
from src.models.policy_network import PolicyNetwork
from src.evaluation.metrics import (
    EpisodeResult, AggregateMetrics,
    aggregate, format_summary, _TOTAL_WEIGHT,
)


# ---------------------------------------------------------------------------
# EvalConfig
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """
    Hyperparameters for evaluation.

    Attributes
    ----------
    n_episodes : int
        Number of episodes to evaluate over.
    deterministic : bool
        If True, use greedy (argmax) action selection.
        If False, sample from the distribution.
    env_config : ProbeEnvConfig
        Environment configuration for evaluation episodes.
        Should match the target difficulty (final curriculum stage K).
    device : str
        "cpu" or "cuda".
    seed : int
        Base seed for evaluation environments.
        Each episode uses seed + episode_index for reproducibility.
    verbose : bool
        If True, print per-episode summaries.
    """
    n_episodes:    int           = 20
    deterministic: bool          = True
    env_config:    ProbeEnvConfig = None   # set in __post_init__
    device:        str           = "cpu"
    seed:          int           = 0
    verbose:       bool          = False

    def __post_init__(self):
        if self.env_config is None:
            self.env_config = ProbeEnvConfig()
        assert self.n_episodes >= 1


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------

def evaluate(
    policy: PolicyNetwork,
    config: EvalConfig,
) -> List[EpisodeResult]:
    """
    Run N complete evaluation episodes and return per-episode results.

    Parameters
    ----------
    policy : PolicyNetwork
        The trained (or partially trained) policy network.
    config : EvalConfig
        Evaluation hyperparameters.

    Returns
    -------
    List[EpisodeResult]
        One EpisodeResult per episode.
    """
    device  = torch.device(config.device)
    policy  = policy.to(device)
    policy.eval()

    results: List[EpisodeResult] = []

    for ep_idx in range(config.n_episodes):

        # Each episode gets a unique seed for reproducibility
        ep_seed = config.seed + ep_idx
        env_cfg = ProbeEnvConfig(
            episode_length         = config.env_config.episode_length,
            blind_violation_budget = config.env_config.blind_violation_budget,
            min_failures           = config.env_config.min_failures,
            max_failures           = config.env_config.max_failures,
            reward_config          = config.env_config.reward_config,
            window_size            = config.env_config.window_size,
            graph_seed             = ep_seed,
            workload_seed          = ep_seed,
            diurnal_amplitude      = config.env_config.diurnal_amplitude,
        )
        env = ProbeEnv(env_cfg)
        obs, _ = env.reset()
        ep      = env.current_graph

        # ── Per-step accumulators ─────────────────────────────────────
        total_reward  = 0.0
        step          = 0
        probe_counts: List[int]   = []
        covered_steps: List[bool] = []
        blind_steps:   List[bool] = []
        weighted_covs: List[float] = []

        # Per-SLO step counters
        slo_covered_steps = [0] * NUM_SLOS
        slo_blind_steps   = [0] * NUM_SLOS

        terminated = False
        truncated  = False

        while True:
            # ── Act ───────────────────────────────────────────────────
            action, _, _, _ = policy.act(obs, ep, deterministic=config.deterministic)

            # ── Step ──────────────────────────────────────────────────
            obs_next, reward, terminated, truncated, info = env.step(action)

            # ── Accumulate ────────────────────────────────────────────
            total_reward += reward
            step         += 1

            rb           = info["reward_breakdown"]
            probe_set    = info["probe_set"]
            covered_slos = rb.covered_slos
            blind_slos   = rb.blind_slos

            probe_counts.append(len(probe_set))
            covered_steps.append(len(covered_slos) > 0)
            blind_steps.append(len(blind_slos) > 0)

            # Weighted coverage at this step
            w_cov = sum(SLO_CATALOG[k].weight for k in covered_slos)
            weighted_covs.append(w_cov / _TOTAL_WEIGHT)

            # Per-SLO
            for k in covered_slos:
                slo_covered_steps[k] += 1
            for k in blind_slos:
                slo_blind_steps[k]   += 1

            if terminated or truncated:
                break
            obs = obs_next
            ep  = env.current_graph

        # ── Compute episode metrics ───────────────────────────────────
        T = max(step, 1)
        mean_probes    = float(np.mean(probe_counts))
        coverage_rate  = float(np.mean(covered_steps))
        blind_rate     = float(np.mean(blind_steps))
        probe_eff      = coverage_rate / max(mean_probes, 1.0)
        weighted_cov   = float(np.mean(weighted_covs))

        slo_cov_rates  = [slo_covered_steps[k] / T for k in range(NUM_SLOS)]
        slo_blind_rates = [slo_blind_steps[k]   / T for k in range(NUM_SLOS)]

        result = EpisodeResult(
            total_reward      = total_reward,
            episode_length    = step,
            survived          = bool(truncated and not terminated),
            cum_blind         = int(info["cum_blind"]),
            mean_probe_count  = mean_probes,
            coverage_rate     = coverage_rate,
            blind_rate        = blind_rate,
            probe_efficiency  = probe_eff,
            weighted_coverage = weighted_cov,
            slo_coverage      = slo_cov_rates,
            slo_blind         = slo_blind_rates,
        )
        results.append(result)

        if config.verbose:
            status = "SURVIVED" if result.survived else "TERMINATED"
            print(
                f"  ep={ep_idx:3d} [{status}] "
                f"R={result.total_reward:+7.3f} "
                f"T={result.episode_length:3d} "
                f"cov={result.coverage_rate:.3f} "
                f"blind={result.blind_rate:.3f} "
                f"probes={result.mean_probe_count:.1f} "
                f"eff={result.probe_efficiency:.3f}"
            )

    return results


# ---------------------------------------------------------------------------
# evaluate_and_aggregate()
# ---------------------------------------------------------------------------

def evaluate_and_aggregate(
    policy:  PolicyNetwork,
    config:  EvalConfig,
    print_summary: bool = True,
) -> AggregateMetrics:
    """
    Evaluate N episodes and return aggregate metrics.

    Parameters
    ----------
    policy : PolicyNetwork
    config : EvalConfig
    print_summary : bool
        If True, print a formatted summary to stdout.

    Returns
    -------
    AggregateMetrics
    """
    results = evaluate(policy, config)
    agg     = aggregate(results)

    if print_summary:
        print(format_summary(agg))

    return agg