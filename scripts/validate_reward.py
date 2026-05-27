"""
validate_reward.py
------------------
Phase 3a validation: reward function.

Run from the project root:
    python scripts/validate_reward.py

Checks:
  1.  No probes → zero observability, no overhead, blind violations if SLOs fire
  2.  All probes → full observability, overhead scales with probe count
  3.  Covered + no violation → positive reward (worked example from §6)
  4.  Blind violation fires μ penalty
  5.  Covered violation does NOT trigger blind penalty
  6.  remove_probe on safe node → near-zero removal risk
  7.  remove_probe on near-threshold node → high removal risk
  8.  remove_probe on violated node → removal risk > w_k
  9.  NO_OP and ADD_PROBE → zero removal risk regardless of node state
 10.  Terminal reward: survived → bonus, failed → penalty
 11.  Absent SLO candidate node → skipped in all terms
 12.  Worked example from §6 of formulation (numeric check)
"""

import sys
sys.path.insert(0, ".")

from src.simulator.graph import EpisodeGraphBuilder
from src.simulator.config.slo_config import SLO_BY_ID, TOTAL_WEIGHT
from src.simulator.env.reward import (
    RewardConfig, RewardInput, RewardOutput,
    compute_reward, terminal_reward,
    ADD_PROBE, REMOVE_PROBE, NO_OP,
)

PASS = "✓"
FAIL = "✗"

cfg = RewardConfig(lam=0.05, mu=2.0, rho=0.5)

builder  = EpisodeGraphBuilder(seed=42)
ep_full  = builder.next_episode()   # all 24 nodes present

def section(title):
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")

# Helper: build sli_values dict from (node, metric) pairs
def sli(**kwargs):
    """kwargs: node__metric=value  (double-underscore separator)"""
    return {tuple(k.split("__", 1)): v for k, v in kwargs.items()}

# Safe SLI snapshot — all metrics well within thresholds
safe_sli = {
    ("search",        "search_latency_p99"):   80.0,
    ("geo",           "processing_time_p99"):  20.0,
    ("reservation",   "processing_time_p99"):  80.0,
    ("reservation",   "failure_rate"):          0.2,
    ("mongodb-rate",  "query_latency_p99"):      3.0,
    ("memcached-rate","hit_rate"):              97.0,
}


# ── 1. No probes → zero observability, blind violations if SLO fires ─────────
section("1. No probes")

out = compute_reward(RewardInput(
    probe_set={},
    sli_values=safe_sli,
    action_type=NO_OP,
    action_node=None,
    episode_graph=ep_full,
), cfg)

assert out.observability == 0.0,   f"{FAIL} Expected 0 observability, got {out.observability}"
assert out.overhead      == 0.0,   f"{FAIL} Expected 0 overhead, got {out.overhead}"
assert out.removal_risk  == 0.0,   f"{FAIL} Expected 0 removal risk"
assert out.blind_violation == 0.0, f"{FAIL} No violations in safe_sli, blind penalty should be 0"
assert out.covered_slos == [],     f"{FAIL} No SLOs covered with empty probe set"
print(f"  observability=0, overhead=0, blind_penalty=0  {PASS}")


# ── 2. All probes → full observability, overhead scales ──────────────────────
section("2. All probes")

all_probes = set(ep_full.probeable_nodes)   # 22 nodes
out2 = compute_reward(RewardInput(
    probe_set=all_probes,
    sli_values=safe_sli,
    action_type=NO_OP,
    action_node=None,
    episode_graph=ep_full,
), cfg)

assert abs(out2.observability - TOTAL_WEIGHT) < 1e-9, \
    f"{FAIL} Full coverage should yield TOTAL_WEIGHT={TOTAL_WEIGHT}, got {out2.observability}"
assert abs(out2.overhead - (-cfg.lam * len(all_probes))) < 1e-9, \
    f"{FAIL} Overhead mismatch"
assert out2.blind_violation == 0.0, f"{FAIL} No blind violations with all probes"
assert set(out2.covered_slos) == {0,1,2,3,4,5}, f"{FAIL} All 6 SLOs should be covered"
print(f"  observability={out2.observability:.2f} (={TOTAL_WEIGHT}), overhead={out2.overhead:.2f}  {PASS}")


# ── 3. Partial probes, safe signal → positive reward ─────────────────────────
section("3. Optimal probe set {search, geo} — safe signal")

out3 = compute_reward(RewardInput(
    probe_set={"search", "geo"},
    sli_values=safe_sli,
    action_type=NO_OP,
    action_node=None,
    episode_graph=ep_full,
), cfg)

expected_obs = SLO_BY_ID[0].weight + SLO_BY_ID[1].weight   # 0.5 + 0.3 = 0.8
expected_oh  = -cfg.lam * 2
assert abs(out3.observability - expected_obs) < 1e-9, \
    f"{FAIL} observability={out3.observability}, expected {expected_obs}"
assert abs(out3.overhead - expected_oh) < 1e-9, \
    f"{FAIL} overhead={out3.overhead}, expected {expected_oh}"
assert out3.total > 0, f"{FAIL} Reward should be positive with good coverage and safe signal"
print(f"  obs={out3.observability:.2f}, overhead={out3.overhead:.3f}, total={out3.total:.3f}  {PASS}")


# ── 4. Blind violation fires μ penalty ───────────────────────────────────────
section("4. Blind violation: geo spiked, no probe on geo")

violated_sli = dict(safe_sli)
violated_sli[("geo", "processing_time_p99")] = 65.0   # > 50ms threshold → violation

out4 = compute_reward(RewardInput(
    probe_set={"search"},      # search probed, geo NOT probed
    sli_values=violated_sli,
    action_type=NO_OP,
    action_node=None,
    episode_graph=ep_full,
), cfg)

assert 1 in out4.blind_slos, f"{FAIL} SLO_1 (geo) should be a blind violation"
assert abs(out4.blind_violation - (-cfg.mu * SLO_BY_ID[1].weight)) < 1e-9, \
    f"{FAIL} blind_penalty={out4.blind_violation}, expected {-cfg.mu * SLO_BY_ID[1].weight}"
print(f"  blind_slos={out4.blind_slos}, blind_penalty={out4.blind_violation:.2f}  {PASS}")


# ── 5. Covered violation → no blind penalty ──────────────────────────────────
section("5. Covered violation: geo spiked, probe IS on geo")

out5 = compute_reward(RewardInput(
    probe_set={"search", "geo"},
    sli_values=violated_sli,
    action_type=NO_OP,
    action_node=None,
    episode_graph=ep_full,
), cfg)

assert 1 not in out5.blind_slos,    f"{FAIL} SLO_1 covered → should NOT be blind"
assert out5.blind_violation == 0.0, f"{FAIL} No blind penalty when violation is covered"
assert 1 in out5.covered_slos,      f"{FAIL} SLO_1 should appear in covered_slos"
print(f"  covered_slos={out5.covered_slos}, blind_penalty=0  {PASS}")


# ── 6. remove_probe on safe node → near-zero removal risk ────────────────────
section("6. remove_probe on geo — safe signal (margin ≈ 0.6)")

# geo.processing_time_p99=20ms, threshold=50ms → margin=(50-20)/50=0.6
# risk = w_1 * max(0, 1-0.6) = 0.3 * 0.4 = 0.12
out6 = compute_reward(RewardInput(
    probe_set={"search", "geo"},
    sli_values=safe_sli,
    action_type=REMOVE_PROBE,
    action_node="geo",
    episode_graph=ep_full,
), cfg)

expected_risk = -cfg.rho * (SLO_BY_ID[1].weight * max(0.0, 1.0 - 0.6))
assert abs(out6.removal_risk - expected_risk) < 1e-6, \
    f"{FAIL} removal_risk={out6.removal_risk:.4f}, expected {expected_risk:.4f}"
print(f"  removal_risk={out6.removal_risk:.4f} (expected {expected_risk:.4f})  {PASS}")


# ── 7. remove_probe on near-threshold node → higher removal risk ──────────────
section("7. remove_probe on geo — near-threshold signal (margin ≈ 0.04)")

# geo.processing_time_p99=48ms, threshold=50ms → margin=(50-48)/50=0.04
near_sli = dict(safe_sli)
near_sli[("geo", "processing_time_p99")] = 48.0

out7 = compute_reward(RewardInput(
    probe_set={"search", "geo"},
    sli_values=near_sli,
    action_type=REMOVE_PROBE,
    action_node="geo",
    episode_graph=ep_full,
), cfg)

assert out7.removal_risk < out6.removal_risk, \
    f"{FAIL} Near-threshold removal risk should be larger (more negative) than safe"
expected_risk7 = -cfg.rho * (SLO_BY_ID[1].weight * max(0.0, 1.0 - 0.04))
assert abs(out7.removal_risk - expected_risk7) < 1e-6, \
    f"{FAIL} removal_risk={out7.removal_risk:.4f}, expected {expected_risk7:.4f}"
print(f"  safe removal_risk={out6.removal_risk:.4f}, near-threshold={out7.removal_risk:.4f}  {PASS}")


# ── 8. remove_probe on violated node → removal risk > w_k ────────────────────
section("8. remove_probe on geo — already violated (margin < 0)")

# geo.processing_time_p99=65ms → margin=(50-65)/50=-0.3  → max(0,1-(-0.3))=1.3
out8 = compute_reward(RewardInput(
    probe_set={"search", "geo"},
    sli_values=violated_sli,
    action_type=REMOVE_PROBE,
    action_node="geo",
    episode_graph=ep_full,
), cfg)

expected_risk8 = -cfg.rho * (SLO_BY_ID[1].weight * max(0.0, 1.0 - (-0.3)))
assert abs(out8.removal_risk - expected_risk8) < 1e-6, \
    f"{FAIL} removal_risk={out8.removal_risk:.4f}, expected {expected_risk8:.4f}"
assert out8.removal_risk < out7.removal_risk, \
    f"{FAIL} Violated node should have even larger (more negative) removal risk"
print(f"  violated removal_risk={out8.removal_risk:.4f} > near-threshold={out7.removal_risk:.4f}  {PASS}")


# ── 9. NO_OP and ADD_PROBE → zero removal risk ────────────────────────────────
section("9. NO_OP and ADD_PROBE always yield zero removal risk")

out_noop = compute_reward(RewardInput(
    probe_set={"search"},
    sli_values=violated_sli,
    action_type=NO_OP,
    action_node=None,
    episode_graph=ep_full,
), cfg)
assert out_noop.removal_risk == 0.0, f"{FAIL} NO_OP should have zero removal risk"

out_add = compute_reward(RewardInput(
    probe_set={"search"},
    sli_values=violated_sli,
    action_type=ADD_PROBE,
    action_node="geo",
    episode_graph=ep_full,
), cfg)
assert out_add.removal_risk == 0.0, f"{FAIL} ADD_PROBE should have zero removal risk"
print(f"  NO_OP: removal_risk=0, ADD_PROBE: removal_risk=0  {PASS}")


# ── 10. Terminal reward ───────────────────────────────────────────────────────
section("10. Terminal reward")

assert terminal_reward(survived=True,  cfg=cfg) == cfg.terminal_bonus,   f"{FAIL} Bonus mismatch"
assert terminal_reward(survived=False, cfg=cfg) == cfg.terminal_penalty, f"{FAIL} Penalty mismatch"
print(f"  survived → bonus={cfg.terminal_bonus}, failed → penalty={cfg.terminal_penalty}  {PASS}")


# ── 11. Absent candidate node → skipped in all terms ─────────────────────────
section("11. Absent candidate node skipped in all terms")

# Build an episode where mongodb-rate and memcached-rate are absent
# → SLO_4 and SLO_5 should be uncoverable and not penalised
builder2 = EpisodeGraphBuilder(p_fail=0.99, p_rec=0.0, seed=5)
builder2.next_episode()       # ep0: all present
ep_absent = builder2.next_episode()   # ep1: many absent

assert 4 not in ep_absent.coverable_slos or 5 not in ep_absent.coverable_slos, \
    "Setup: at least one of SLO_4/SLO_5 should be uncoverable in this episode"

out11 = compute_reward(RewardInput(
    probe_set=set(),
    sli_values={},   # no values at all — absent nodes emit nothing
    action_type=NO_OP,
    action_node=None,
    episode_graph=ep_absent,
), cfg)

# Uncoverable SLOs must not appear in blind_slos (no signal → no blind violation)
for slo_id in [4, 5]:
    if slo_id not in ep_absent.coverable_slos:
        assert slo_id not in out11.blind_slos, \
            f"{FAIL} SLO_{slo_id} absent but appears in blind_slos"
print(f"  Absent SLOs not penalised  {PASS}")


# ── 12. Worked example from §6 ────────────────────────────────────────────────
section("12. Worked example from §6 (optimal placement {search, geo}, t=9)")

# SLO_0: search_latency_p99=80ms (<200ms) — safe
# SLO_1: geo_processing_time_p99=65ms (>50ms) — VIOLATED but covered
# probe_set = {search, geo}  → both SLOs covered → no blind violations
example_sli = dict(safe_sli)
example_sli[("geo", "processing_time_p99")] = 65.0

out12 = compute_reward(RewardInput(
    probe_set={"search", "geo"},
    sli_values=example_sli,
    action_type=NO_OP,
    action_node=None,
    episode_graph=ep_full,
), cfg)

expected_obs12 = SLO_BY_ID[0].weight + SLO_BY_ID[1].weight   # 0.5 + 0.3 = 0.8
expected_oh12  = -cfg.lam * 2

assert abs(out12.observability - expected_obs12) < 1e-9
assert abs(out12.overhead      - expected_oh12)  < 1e-9
assert out12.blind_violation == 0.0
assert out12.removal_risk    == 0.0

# From formulation §6: R = 0.8 − 2λ
expected_total = 0.8 - 2 * cfg.lam
assert abs(out12.total - expected_total) < 1e-9, \
    f"{FAIL} §6 example: expected R={expected_total:.3f}, got {out12.total:.3f}"
print(f"  §6 example: R = 0.8 − 2λ = {expected_total:.3f}  {PASS}")

# Counterexample: no probe on geo → blind violation on SLO_1
out12b = compute_reward(RewardInput(
    probe_set={"search"},
    sli_values=example_sli,
    action_type=NO_OP,
    action_node=None,
    episode_graph=ep_full,
), cfg)

expected_total_b = SLO_BY_ID[0].weight - cfg.lam * 1 - cfg.mu * SLO_BY_ID[1].weight
assert abs(out12b.total - expected_total_b) < 1e-9, \
    f"{FAIL} §6 counterexample: expected R={expected_total_b:.3f}, got {out12b.total:.3f}"
assert 1 in out12b.blind_slos
print(f"  §6 counterexample: R = w0 − λ − μ·w1 = {expected_total_b:.3f}  {PASS}")


# ── Done ──────────────────────────────────────────────────────────────────────
print(f"\n{'═' * 55}")
print(f"  ALL REWARD ASSERTIONS PASSED ✓")
print(f"{'═' * 55}\n")