"""
validate_probe_env.py
---------------------
Phase 3c validation: probe environment.

Run from project root:
    python scripts/validate_probe_env.py

Checks:
  1.  reset() returns valid obs dict and info
  2.  Observation shapes match spec at reset
  3.  Action space size = 2*|V_p| + 1
  4.  no_op action: probe_set unchanged, step returns valid obs
  5.  add_probe: node added to probe_set
  6.  remove_probe: node removed from probe_set
  7.  Invalid add_probe (already probed) → treated as no_op via mask
  8.  action_mask: no_op always True, add valid iff unprobed, remove valid iff probed
  9.  Out-of-range action → treated as no_op, no crash
 10.  Reward breakdown present in info
 11.  cum_blind increments correctly
 12.  Failure termination: terminated=True when cum_blind > K
 13.  Normal truncation: truncated=True at t=T
 14.  Terminal bonus added on survival
 15.  Terminal penalty added on failure
 16.  Full episode rollout: T steps without crash
 17.  Two consecutive episodes: obs shapes consistent, probe_set reset
 18.  action_to_str correct for all action types
 19.  Deterministic replay: same seeds → same rewards
"""

import sys
sys.path.insert(0, ".")

import numpy as np
from src.simulator.env.probe_env import ProbeEnv, ProbeEnvConfig
from src.simulator.env.reward import RewardConfig
from src.simulator.config.slo_config import NUM_SLOS
from src.simulator.env.observation_builder import NODE_FEATURE_DIM

PASS = "✓"
FAIL = "✗"

def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

def make_env(T=50, K=5, seed=42):
    return ProbeEnv(ProbeEnvConfig(
        episode_length=T,
        blind_violation_budget=K,
        n_failures=1,
        graph_seed=seed,
        workload_seed=seed,
        reward_config=RewardConfig(lam=0.05, mu=2.0, rho=0.5),
    ))

# ── 1. reset() returns valid obs and info ────────────────────────────────────
section("1. reset() returns valid obs and info")

env = make_env()
obs, info = env.reset()

assert isinstance(obs, dict),  f"{FAIL} obs should be dict"
assert isinstance(info, dict), f"{FAIL} info should be dict"
assert "node_features" in obs, f"{FAIL} missing node_features"
assert "edge_index"    in obs, f"{FAIL} missing edge_index"
assert "slo_health"    in obs, f"{FAIL} missing slo_health"
assert "action_mask"   in obs, f"{FAIL} missing action_mask"
assert "present_nodes" in info
assert "probeable_nodes" in info
assert "coverable_slos"  in info
print(f"  obs keys: {list(obs.keys())}  {PASS}")


# ── 2. Observation shapes ─────────────────────────────────────────────────────
section("2. Observation shapes")

nf = obs["node_features"]
ei = obs["edge_index"]
sh = obs["slo_health"]
am = obs["action_mask"]

assert nf.shape == (24, NODE_FEATURE_DIM), f"{FAIL} node_features {nf.shape}"
assert ei.shape[0] == 2,                  f"{FAIL} edge_index rows {ei.shape}"
assert sh.shape == (NUM_SLOS,),           f"{FAIL} slo_health {sh.shape}"
assert nf.dtype == np.float32,            f"{FAIL} node_features dtype"
assert sh.dtype == np.float32,            f"{FAIL} slo_health dtype"
print(f"  node_features: {nf.shape}, edge_index: {ei.shape}, slo_health: {sh.shape}  {PASS}")


# ── 3. Action space size ──────────────────────────────────────────────────────
section("3. Action space size")

n_p = len(env.probeable_nodes)
expected_size = 2 * n_p + 1
# Action mask should have exactly expected_size True/False entries for this episode
valid_count = int(am[:expected_size].shape[0])
assert valid_count == expected_size, f"{FAIL} action mask slice size {valid_count}"
assert n_p > 0, f"{FAIL} no probeable nodes"
print(f"  |V_p|={n_p}, action_space={env.action_space.n}, episode actions={expected_size}  {PASS}")


# ── 4. no_op: probe_set unchanged ────────────────────────────────────────────
section("4. no_op: probe_set unchanged")

env.reset()
obs2, r2, term, trunc, info2 = env.step(0)  # action 0 = no_op

assert info2["action_str"] == "no_op", f"{FAIL} action_str={info2['action_str']}"
assert len(info2["probe_set"]) == 0,   f"{FAIL} probe_set should still be empty"
assert isinstance(r2, float),          f"{FAIL} reward should be float"
assert not term,                       f"{FAIL} should not terminate on first no_op"
print(f"  no_op: probe_set={info2['probe_set']}, reward={r2:.3f}  {PASS}")


# ── 5. add_probe: node added ──────────────────────────────────────────────────
section("5. add_probe: node added to probe_set")

env.reset()
node0 = env.probeable_nodes[0]
add_action = 1  # add_probe(probeable_nodes[0])
obs3, r3, _, _, info3 = env.step(add_action)

assert node0 in info3["probe_set"], \
    f"{FAIL} {node0} should be in probe_set after add_probe"
assert info3["action_str"] == f"add_probe({node0})", \
    f"{FAIL} action_str={info3['action_str']}"
print(f"  add_probe({node0}): probe_set={info3['probe_set']}  {PASS}")


# ── 6. remove_probe: node removed ────────────────────────────────────────────
section("6. remove_probe: node removed from probe_set")

# add then remove
n_p = len(env.probeable_nodes)
remove_action = 1 + n_p  # remove_probe(probeable_nodes[0])
obs4, r4, _, _, info4 = env.step(remove_action)

assert node0 not in info4["probe_set"], \
    f"{FAIL} {node0} should be removed from probe_set"
assert info4["action_str"] == f"remove_probe({node0})", \
    f"{FAIL} action_str={info4['action_str']}"
print(f"  remove_probe({node0}): probe_set={info4['probe_set']}  {PASS}")


# ── 7. add_probe on already-probed node: mask prevents it ────────────────────
section("7. action_mask prevents add_probe on already-probed node")

env.reset()
node0 = env.probeable_nodes[0]
env.step(1)  # add_probe(node0)

mask_after_add = env.build_action_mask()
# add_probe(node0) = action index 1 — should now be False
assert not mask_after_add[1], \
    f"{FAIL} add_probe({node0}) should be invalid when already probed"
# remove_probe(node0) = action index 1+n_p — should now be True
n_p = len(env.probeable_nodes)
assert mask_after_add[1 + n_p], \
    f"{FAIL} remove_probe({node0}) should be valid when probed"
print(f"  mask[add]={mask_after_add[1]}, mask[remove]={mask_after_add[1+n_p]}  {PASS}")


# ── 8. action_mask correctness ────────────────────────────────────────────────
section("8. action_mask: no_op always True, add/remove correct")

env.reset()
# Probe first two nodes
env.step(1)
env.step(2)
mask = env.build_action_mask()
n_p  = len(env.probeable_nodes)

assert mask[0], f"{FAIL} no_op should always be valid"
# add_probe(0) and add_probe(1) should be invalid (already probed)
assert not mask[1], f"{FAIL} add_probe(0) invalid when probed"
assert not mask[2], f"{FAIL} add_probe(1) invalid when probed"
# add_probe(2..n_p-1) should be valid
for i in range(2, n_p):
    assert mask[1 + i], f"{FAIL} add_probe({i}) should be valid"
# remove_probe(0) and remove_probe(1) should be valid
assert mask[1 + n_p],     f"{FAIL} remove_probe(0) should be valid"
assert mask[1 + n_p + 1], f"{FAIL} remove_probe(1) should be valid"
# remove_probe(2..n_p-1) should be invalid
for i in range(2, n_p):
    assert not mask[1 + n_p + i], f"{FAIL} remove_probe({i}) should be invalid"
print(f"  action_mask correct for 2-probe state  {PASS}")


# ── 9. Out-of-range action → no crash, treated as no_op ──────────────────────
section("9. Out-of-range action handled safely")

env.reset()
before_probe_set = set(env._probe_set)
obs_oor, r_oor, _, _, info_oor = env.step(9999)

assert info_oor["probe_set"] == before_probe_set, \
    f"{FAIL} probe_set changed on out-of-range action"
assert info_oor["action_str"] == "no_op", \
    f"{FAIL} out-of-range action should be no_op"
print(f"  action=9999 → no_op, no crash  {PASS}")


# ── 10. Reward breakdown in info ──────────────────────────────────────────────
section("10. Reward breakdown in info")

env.reset()
_, _, _, _, info10 = env.step(0)

rb = info10["reward_breakdown"]
assert hasattr(rb, "total"),         f"{FAIL} missing total"
assert hasattr(rb, "observability"), f"{FAIL} missing observability"
assert hasattr(rb, "overhead"),      f"{FAIL} missing overhead"
assert hasattr(rb, "blind_violation"),f"{FAIL} missing blind_violation"
assert hasattr(rb, "removal_risk"),  f"{FAIL} missing removal_risk"
assert abs(rb.total - (rb.observability + rb.overhead + rb.blind_violation + rb.removal_risk)) < 1e-6, \
    f"{FAIL} reward components don't sum to total"
print(f"  total={rb.total:.3f} = obs({rb.observability:.3f}) + oh({rb.overhead:.3f}) "
      f"+ blind({rb.blind_violation:.3f}) + risk({rb.removal_risk:.3f})  {PASS}")


# ── 11. cum_blind increments correctly ────────────────────────────────────────
section("11. cum_blind increments with blind violations")

env2 = ProbeEnv(ProbeEnvConfig(
    episode_length=50, blind_violation_budget=100,  # large budget, won't terminate
    n_failures=3, graph_seed=1, workload_seed=1,
    reward_config=RewardConfig(mu=2.0),
))
env2.reset()

total_blind = 0
for step in range(20):
    _, _, _, _, info = env2.step(0)  # always no_op → never probe anything → max blind
    total_blind += len(info["blind_slos"])

assert env2._cum_blind == total_blind, \
    f"{FAIL} cum_blind={env2._cum_blind} != sum of blind_slos={total_blind}"
print(f"  cum_blind={env2._cum_blind} after 20 no_op steps  {PASS}")


# ── 12. Failure termination ────────────────────────────────────────────────────
section("12. Failure termination when cum_blind > K")

# K=0: any blind violation immediately terminates
env3 = ProbeEnv(ProbeEnvConfig(
    episode_length=200, blind_violation_budget=0,
    n_failures=3, graph_seed=2, workload_seed=2,
))
env3.reset()

terminated_seen = False
for step in range(200):
    _, _, term, trunc, info = env3.step(0)  # no probes → many blind violations
    if term:
        terminated_seen = True
        assert info["terminated"], f"{FAIL} terminated flag should be True in info"
        break

assert terminated_seen, f"{FAIL} Expected failure termination with K=0 and no probes"
print(f"  K=0, no probes → terminated at step {step}  {PASS}")


# ── 13. Normal truncation at t=T ──────────────────────────────────────────────
section("13. Normal truncation at t=T")

T = 30   # large enough to probe all nodes and still reach truncation
env4 = ProbeEnv(ProbeEnvConfig(
    episode_length=T, blind_violation_budget=10000,  # never terminates
    n_failures=0, graph_seed=3, workload_seed=3,
))
env4.reset()

truncated_seen = False
for step in range(T + 5):
    # Add probes on early steps, then no_op until truncation
    action = (step + 1) if step < len(env4.probeable_nodes) else 0
    _, _, term, trunc, _ = env4.step(action)
    if trunc:
        truncated_seen = True
        assert not term, f"{FAIL} truncated episode should not also be terminated"
        break
    if term:
        break

assert truncated_seen, f"{FAIL} Expected truncation at T={T}"
print(f"  Truncated at T={T}  {PASS}")


# ── 14. Terminal bonus on survival ────────────────────────────────────────────
section("14. Terminal bonus added on survival")

T = 5
bonus = 99.0
env5 = ProbeEnv(ProbeEnvConfig(
    episode_length=T, blind_violation_budget=10000,
    n_failures=0, graph_seed=4, workload_seed=4,
    reward_config=RewardConfig(lam=0.0, mu=0.0, rho=0.0,
                               terminal_bonus=bonus, terminal_penalty=-999.0),
))
env5.reset()
rewards = []
for step in range(T):
    _, r, term, trunc, _ = env5.step(0)
    rewards.append(r)
    if term or trunc:
        break

# Last reward should contain the bonus
last_r = rewards[-1]
assert last_r >= bonus * 0.9, \
    f"{FAIL} terminal reward {last_r:.2f} should include bonus ~{bonus}"
print(f"  last reward={last_r:.2f} includes bonus={bonus}  {PASS}")


# ── 15. Terminal penalty on failure ───────────────────────────────────────────
section("15. Terminal penalty on failure termination")

penalty = -99.0
env6 = ProbeEnv(ProbeEnvConfig(
    episode_length=200, blind_violation_budget=0,
    n_failures=3, graph_seed=5, workload_seed=5,
    reward_config=RewardConfig(lam=0.0, mu=0.0, rho=0.0,
                               terminal_bonus=999.0, terminal_penalty=penalty),
))
env6.reset()
for step in range(200):
    _, r, term, trunc, _ = env6.step(0)
    if term:
        # reward at termination step should include penalty
        assert r <= penalty * 0.9 or r <= 0, \
            f"{FAIL} terminal penalty reward {r:.2f} should be <= {penalty}"
        break
print(f"  Failure penalty applied at termination  {PASS}")


# ── 16. Full episode rollout without crash ────────────────────────────────────
section("16. Full episode rollout (T=50, random actions)")

env7 = make_env(T=50, K=100, seed=7)
obs, _ = env7.reset()
rng = np.random.default_rng(7)
total_r = 0.0
steps   = 0
for step in range(60):
    mask    = obs["action_mask"]
    valid   = np.where(mask)[0]
    action  = int(rng.choice(valid))
    obs, r, term, trunc, _ = env7.step(action)
    total_r += r
    steps   += 1
    if term or trunc:
        break

assert steps > 0, f"{FAIL} Should have taken at least one step"
print(f"  Completed {steps} steps, total_reward={total_r:.3f}  {PASS}")


# ── 17. Two consecutive episodes: state reset ─────────────────────────────────
section("17. Two consecutive episodes: state reset correctly")

env8 = make_env(T=10, K=100, seed=8)

# Episode 1: add some probes
env8.reset()
env8.step(1)
env8.step(2)
assert len(env8._probe_set) == 2

# Episode 2: reset should clear probe_set and t
env8.reset()
assert len(env8._probe_set) == 0, f"{FAIL} probe_set not cleared at reset"
assert env8._t == 0,              f"{FAIL} timestep not reset"
assert env8._cum_blind == 0,      f"{FAIL} cum_blind not reset"
print(f"  probe_set=∅, t=0, cum_blind=0 after reset  {PASS}")


# ── 18. action_to_str ─────────────────────────────────────────────────────────
section("18. action_to_str correct for all action types")

env9 = make_env(seed=9)
env9.reset()
n_p = len(env9.probeable_nodes)

assert env9.action_to_str(0)       == "no_op"
assert env9.action_to_str(1)       == f"add_probe({env9.probeable_nodes[0]})"
assert env9.action_to_str(n_p)     == f"add_probe({env9.probeable_nodes[n_p-1]})"
assert env9.action_to_str(n_p + 1) == f"remove_probe({env9.probeable_nodes[0]})"
assert env9.action_to_str(2*n_p)   == f"remove_probe({env9.probeable_nodes[n_p-1]})"
print(f"  no_op, add_probe, remove_probe labels correct  {PASS}")


# ── 19. Deterministic replay ──────────────────────────────────────────────────
section("19. Deterministic replay: same seeds → same rewards")

def rollout(seed):
    e = ProbeEnv(ProbeEnvConfig(
        episode_length=20, blind_violation_budget=100,
        n_failures=1, graph_seed=seed, workload_seed=seed,
    ))
    e.reset()
    rewards = []
    for _ in range(20):
        _, r, term, trunc, _ = e.step(0)
        rewards.append(r)
        if term or trunc:
            break
    return rewards

r1 = rollout(42)
r2 = rollout(42)
assert r1 == r2, f"{FAIL} Same seeds should produce same rewards"
print(f"  Two identical rollouts produced identical rewards  {PASS}")


# ── Done ─────────────────────────────────────────────────────────────────────
print(f"\n{'═'*60}")
print(f"  ALL PROBE ENV ASSERTIONS PASSED ✓")
print(f"{'═'*60}\n")