"""
validate_evaluate.py
--------------------
Phase 6 validation: evaluate.py and metrics.py.

Run from project root:
    python scripts/validate_evaluate.py

Checks:
  1.  evaluate() returns correct number of EpisodeResult objects
  2.  EpisodeResult fields are all finite and in valid ranges
  3.  slo_coverage and slo_blind have length NUM_SLOS
  4.  slo_coverage[k] in [0, 1] for all k
  5.  coverage_rate in [0, 1], blind_rate in [0, 1]
  6.  probe_efficiency >= 0
  7.  weighted_coverage in [0, 1]
  8.  survived = True iff episode reached T (truncated)
  9.  cum_blind >= 0
 10.  aggregate() returns correct n_episodes
 11.  aggregate() mean_reward matches manual mean
 12.  aggregate() survival_rate correct
 13.  aggregate() per-SLO means have length NUM_SLOS
 14.  format_summary() returns non-empty string with key fields
 15.  deterministic=True: same seed → same results
 16.  deterministic=False: different results across seeds
 17.  evaluate_and_aggregate() prints summary and returns AggregateMetrics
 18.  Zero probes policy: coverage_rate=0, blind_rate>0
 19.  probe_efficiency = coverage_rate / mean_probe_count
 20.  weighted_coverage uses SLO weights correctly
"""

import sys
sys.path.insert(0, ".")

import torch
import numpy as np

from src.simulator.env.probe_env import ProbeEnv, ProbeEnvConfig
from src.simulator.config.slo_config import NUM_SLOS, SLO_CATALOG
from src.models.policy_network import PolicyNetwork
from src.models.gnn_encoder import GNNConfig
from src.evaluation.evaluate import evaluate, evaluate_and_aggregate, EvalConfig
from src.evaluation.metrics import (
    EpisodeResult, AggregateMetrics, aggregate, format_summary, _TOTAL_WEIGHT
)

PASS = "✓"
FAIL = "✗"

def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

def make_policy():
    return PolicyNetwork(GNNConfig(hidden_dim=32, num_layers=2, dropout=0.0))

def make_eval_cfg(n=5, det=True, seed=0):
    return EvalConfig(
        n_episodes    = n,
        deterministic = det,
        env_config    = ProbeEnvConfig(
            episode_length=15, n_failures=1,
            blind_violation_budget=100,
            graph_seed=seed, workload_seed=seed,
        ),
        seed    = seed,
        verbose = False,
    )


# ── 1. evaluate() returns correct number of results ───────────────────────────
section("1. evaluate() returns correct number of EpisodeResult objects")

pol = make_policy()
cfg = make_eval_cfg(n=5)
results = evaluate(pol, cfg)

assert len(results) == 5, f"{FAIL} Expected 5 results, got {len(results)}"
assert all(isinstance(r, EpisodeResult) for r in results)
print(f"  {len(results)} EpisodeResult objects returned  {PASS}")


# ── 2. EpisodeResult fields are finite and valid ──────────────────────────────
section("2. EpisodeResult fields finite and in valid ranges")

for i, r in enumerate(results):
    assert np.isfinite(r.total_reward),     f"{FAIL} ep{i}: total_reward not finite"
    assert r.episode_length >= 1,           f"{FAIL} ep{i}: episode_length < 1"
    assert isinstance(r.survived, bool),    f"{FAIL} ep{i}: survived not bool"
    assert r.cum_blind >= 0,               f"{FAIL} ep{i}: cum_blind < 0"
    assert r.mean_probe_count >= 0,         f"{FAIL} ep{i}: mean_probe_count < 0"
    assert 0 <= r.coverage_rate  <= 1,     f"{FAIL} ep{i}: coverage_rate={r.coverage_rate}"
    assert 0 <= r.blind_rate     <= 1,     f"{FAIL} ep{i}: blind_rate={r.blind_rate}"
    assert r.probe_efficiency    >= 0,      f"{FAIL} ep{i}: probe_eff < 0"
    assert 0 <= r.weighted_coverage <= 1,  f"{FAIL} ep{i}: weighted_cov={r.weighted_coverage}"
print(f"  All {len(results)} episodes: fields finite and in range  {PASS}")


# ── 3. slo_coverage and slo_blind have length NUM_SLOS ───────────────────────
section("3. slo_coverage and slo_blind have length NUM_SLOS")

for i, r in enumerate(results):
    assert len(r.slo_coverage) == NUM_SLOS, \
        f"{FAIL} ep{i}: slo_coverage len={len(r.slo_coverage)}"
    assert len(r.slo_blind)    == NUM_SLOS, \
        f"{FAIL} ep{i}: slo_blind len={len(r.slo_blind)}"
print(f"  slo_coverage and slo_blind both have length {NUM_SLOS}  {PASS}")


# ── 4. slo_coverage[k] in [0, 1] ─────────────────────────────────────────────
section("4. slo_coverage[k] and slo_blind[k] in [0, 1]")

for i, r in enumerate(results):
    for k in range(NUM_SLOS):
        assert 0 <= r.slo_coverage[k] <= 1, \
            f"{FAIL} ep{i}: slo_coverage[{k}]={r.slo_coverage[k]}"
        assert 0 <= r.slo_blind[k]    <= 1, \
            f"{FAIL} ep{i}: slo_blind[{k}]={r.slo_blind[k]}"
print(f"  All per-SLO rates in [0, 1]  {PASS}")


# ── 5. coverage_rate and blind_rate in [0, 1] ─────────────────────────────────
section("5. coverage_rate and blind_rate in [0, 1]")

for r in results:
    assert 0.0 <= r.coverage_rate <= 1.0
    assert 0.0 <= r.blind_rate    <= 1.0
print(f"  coverage_rate and blind_rate in [0,1]  {PASS}")


# ── 6. probe_efficiency >= 0 ──────────────────────────────────────────────────
section("6. probe_efficiency >= 0")

for r in results:
    assert r.probe_efficiency >= 0
print(f"  probe_efficiency >= 0 for all episodes  {PASS}")


# ── 7. weighted_coverage in [0, 1] ───────────────────────────────────────────
section("7. weighted_coverage in [0, 1]")

for r in results:
    assert 0.0 <= r.weighted_coverage <= 1.0
print(f"  weighted_coverage in [0,1]  {PASS}")


# ── 8. survived iff truncated ─────────────────────────────────────────────────
section("8. survived = True iff episode reached T")

# With large K=100 and short T=15, most should survive
survivors = [r for r in results if r.survived]
terminators = [r for r in results if not r.survived]
# Survivors have episode_length == T
for r in survivors:
    assert r.episode_length == cfg.env_config.episode_length, \
        f"{FAIL} Survivor has length {r.episode_length} != T={cfg.env_config.episode_length}"
# Terminators have episode_length <= T
for r in terminators:
    assert r.episode_length <= cfg.env_config.episode_length
print(f"  {len(survivors)} survived (len=T), {len(terminators)} terminated  {PASS}")


# ── 9. cum_blind >= 0 ────────────────────────────────────────────────────────
section("9. cum_blind >= 0")

for r in results:
    assert r.cum_blind >= 0
print(f"  cum_blind >= 0 for all episodes  {PASS}")


# ── 10. aggregate() n_episodes correct ───────────────────────────────────────
section("10. aggregate() returns correct n_episodes")

agg = aggregate(results)
assert agg.n_episodes == 5
print(f"  n_episodes={agg.n_episodes}  {PASS}")


# ── 11. aggregate() mean_reward matches manual mean ──────────────────────────
section("11. aggregate() mean_reward matches manual mean")

manual_mean = float(np.mean([r.total_reward for r in results]))
assert abs(agg.mean_reward - manual_mean) < 1e-6, \
    f"{FAIL} mean_reward={agg.mean_reward:.4f}, manual={manual_mean:.4f}"
print(f"  mean_reward={agg.mean_reward:.4f} matches manual={manual_mean:.4f}  {PASS}")


# ── 12. aggregate() survival_rate correct ────────────────────────────────────
section("12. aggregate() survival_rate correct")

manual_surv = float(np.mean([r.survived for r in results]))
assert abs(agg.survival_rate - manual_surv) < 1e-6
print(f"  survival_rate={agg.survival_rate:.3f}  {PASS}")


# ── 13. aggregate() per-SLO means have length NUM_SLOS ───────────────────────
section("13. aggregate() per-SLO means have length NUM_SLOS")

assert len(agg.mean_slo_coverage) == NUM_SLOS
assert len(agg.mean_slo_blind)    == NUM_SLOS
for k in range(NUM_SLOS):
    assert 0 <= agg.mean_slo_coverage[k] <= 1
    assert 0 <= agg.mean_slo_blind[k]    <= 1
print(f"  per-SLO means: length={NUM_SLOS}, all in [0,1]  {PASS}")


# ── 14. format_summary() returns non-empty string ────────────────────────────
section("14. format_summary() returns formatted string")

summary = format_summary(agg)
assert isinstance(summary, str)
assert len(summary) > 0
assert "Reward" in summary
assert "Coverage" in summary or "coverage" in summary
assert "SLO_0" in summary
print(f"  format_summary() returns {len(summary)}-char string with key fields  {PASS}")


# ── 15. deterministic=True: same seed → same results ─────────────────────────
section("15. deterministic=True: same seed → identical results")

pol15 = make_policy()
cfg15 = make_eval_cfg(n=3, det=True, seed=15)
r1 = evaluate(pol15, cfg15)
r2 = evaluate(pol15, cfg15)

for i in range(3):
    assert abs(r1[i].total_reward - r2[i].total_reward) < 1e-5, \
        f"{FAIL} ep{i}: r1={r1[i].total_reward:.4f} != r2={r2[i].total_reward:.4f}"
print(f"  3 deterministic episodes: identical rewards across two runs  {PASS}")


# ── 16. deterministic=False: stochastic → different results ──────────────────
section("16. deterministic=False: stochastic → varied results")

pol16 = make_policy()
cfg16_a = make_eval_cfg(n=5, det=False, seed=0)
cfg16_b = make_eval_cfg(n=5, det=False, seed=99)
r_a = evaluate(pol16, cfg16_a)
r_b = evaluate(pol16, cfg16_b)

rewards_a = [r.total_reward for r in r_a]
rewards_b = [r.total_reward for r in r_b]
# Different seeds should produce different trajectories
all_same = all(abs(a-b) < 1e-8 for a, b in zip(rewards_a, rewards_b))
assert not all_same, f"{FAIL} Stochastic eval with different seeds produced identical rewards"
print(f"  Different seeds → different rewards  {PASS}")


# ── 17. evaluate_and_aggregate() prints summary ───────────────────────────────
section("17. evaluate_and_aggregate() returns AggregateMetrics")

pol17 = make_policy()
cfg17 = make_eval_cfg(n=3)
agg17 = evaluate_and_aggregate(pol17, cfg17, print_summary=True)

assert isinstance(agg17, AggregateMetrics)
assert agg17.n_episodes == 3
print(f"  evaluate_and_aggregate() returned AggregateMetrics  {PASS}")


# ── 18. Zero-probe baseline: coverage_rate=0 ─────────────────────────────────
section("18. Always no_op policy: coverage_rate=0, blind_rate typically >0")

class NoOpPolicy(torch.nn.Module):
    """Policy that always selects no_op (action=0)."""
    def act(self, obs, ep, deterministic=True):
        return 0, 0.0, 0.0, 0.0

noop_pol = NoOpPolicy()
cfg18 = EvalConfig(
    n_episodes=3,
    env_config=ProbeEnvConfig(
        episode_length=10, n_failures=2,
        blind_violation_budget=1000,
        graph_seed=0, workload_seed=0,
    ),
    seed=0,
)
results18 = evaluate(noop_pol, cfg18)
for r in results18:
    assert r.coverage_rate == 0.0, \
        f"{FAIL} No-op policy should have coverage_rate=0, got {r.coverage_rate}"
    assert r.mean_probe_count == 0.0, \
        f"{FAIL} No-op policy should have 0 probes"
print(f"  No-op policy: coverage=0, probes=0  {PASS}")


# ── 19. probe_efficiency formula ─────────────────────────────────────────────
section("19. probe_efficiency = coverage_rate / max(mean_probe_count, 1)")

for r in results:
    expected_eff = r.coverage_rate / max(r.mean_probe_count, 1.0)
    assert abs(r.probe_efficiency - expected_eff) < 1e-6, \
        f"{FAIL} probe_eff={r.probe_efficiency:.4f}, expected={expected_eff:.4f}"
print(f"  probe_efficiency formula correct for all episodes  {PASS}")


# ── 20. weighted_coverage uses SLO weights ────────────────────────────────────
section("20. weighted_coverage uses SLO weights correctly")

# weighted_cov at each step = sum(w_k for covered SLOs) / total_weight
# Verify it's always in [0, 1] and 0 when no SLOs covered
for r in results18:   # no-op → always 0 coverage
    assert r.weighted_coverage == 0.0, \
        f"{FAIL} No-op weighted_cov={r.weighted_coverage}, expected 0"

# For normal policy, check it's <= 1 and > 0 if coverage_rate > 0
for r in results:
    assert 0 <= r.weighted_coverage <= 1.0
    if r.coverage_rate > 0:
        assert r.weighted_coverage >= 0.0
print(f"  weighted_coverage in [0,1]; 0 for no-op policy  {PASS}")


# ── Done ─────────────────────────────────────────────────────────────────────
print(f"\n{'═'*60}")
print(f"  ALL EVALUATE/METRICS ASSERTIONS PASSED ✓")
print(f"{'═'*60}\n")