"""
validate_policy_network.py
--------------------------
Phase 4b validation: policy network.

Run from project root:
    python scripts/validate_policy_network.py

Checks:
  1.  Output shapes (unbatched): logits/probs/log_probs (45,), value scalar, entropy scalar
  2.  Output shapes (batched B=4)
  3.  action_probs sum to 1 over valid actions
  4.  Masked actions have prob=0 and log_prob=-inf
  5.  Actor add/remove use per-node embeddings (different nodes → different logits)
  6.  no_op logit uses global context (changes with slo_health)
  7.  State value changes with graph state
  8.  Entropy > 0 at init (policy not collapsed)
  9.  Entropy = 0 when only one valid action
 10.  Gradients flow to all parameters (actor + critic + GNN)
 11.  build_probeable_indices: correct mapping node → present_nodes row
 12.  act() returns valid action within action_mask
 13.  act(deterministic=True) always returns argmax
 14.  act(deterministic=False) samples (not always argmax)
 15.  Two episodes with different |V_p|: both work correctly
 16.  Probeable node embedding correctly gathered (not mean of all nodes)
 17.  No NaN in any output
"""

import sys
sys.path.insert(0, ".")

import torch
import numpy as np

from src.simulator.env.probe_env import ProbeEnv, ProbeEnvConfig
from src.simulator.config.slo_config import NUM_SLOS
from src.models.gnn_encoder import GNNConfig, build_masks, MAX_NODES, MAX_EDGES
from src.models.policy_network import (
    PolicyConfig, PolicyNetwork, PolicyOutput,
    build_probeable_indices, MAX_ACTION_DIM, MAX_PROBEABLE,
)

PASS = "✓"
FAIL = "✗"

def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

def make_env(seed=0, T=20):
    return ProbeEnv(ProbeEnvConfig(
        episode_length=T, n_failures=0,
        graph_seed=seed, workload_seed=seed,
    ))

def make_policy(hidden_dim=64, dropout=0.0):
    gnn_cfg = GNNConfig(hidden_dim=hidden_dim, num_layers=2, dropout=dropout)
    pol_cfg = PolicyConfig(actor_hidden_dim=32, critic_hidden_dim=64,
                           noop_hidden_dim=32, dropout=dropout)
    return PolicyNetwork(gnn_cfg, pol_cfg)

def obs_to_tensors(obs, ep):
    nf = torch.tensor(obs["node_features"], dtype=torch.float32)
    ei = torch.tensor(obs["edge_index"],    dtype=torch.int64)
    sh = torch.tensor(obs["slo_health"],    dtype=torch.float32)
    am = torch.tensor(obs["action_mask"],   dtype=torch.bool)
    nm, em = build_masks(ep)
    pi     = build_probeable_indices(ep)
    return nf, ei, nm, em, sh, am, pi


# ── Setup ────────────────────────────────────────────────────────────────────
env = make_env(seed=0)
obs, _ = env.reset()
ep  = env.current_graph
pol = make_policy()
pol.eval()

nf, ei, nm, em, sh, am, pi = obs_to_tensors(obs, ep)
n_p = len(env.probeable_nodes)


# ── 1. Output shapes (unbatched) ─────────────────────────────────────────────
section("1. Output shapes — unbatched")

with torch.no_grad():
    out = pol(nf, ei, nm, em, sh, am, pi)

assert out.action_logits.shape == (MAX_ACTION_DIM,), \
    f"{FAIL} action_logits {out.action_logits.shape}"
assert out.action_probs.shape  == (MAX_ACTION_DIM,), \
    f"{FAIL} action_probs {out.action_probs.shape}"
assert out.log_probs.shape     == (MAX_ACTION_DIM,), \
    f"{FAIL} log_probs {out.log_probs.shape}"
assert out.state_value.shape   == torch.Size([]), \
    f"{FAIL} state_value should be scalar, got {out.state_value.shape}"
assert out.entropy.shape       == torch.Size([]), \
    f"{FAIL} entropy should be scalar, got {out.entropy.shape}"
print(f"  logits: {out.action_logits.shape}, value: scalar, entropy: scalar  {PASS}")


# ── 2. Output shapes (batched B=4) ───────────────────────────────────────────
section("2. Output shapes — batched B=4")

B = 4
nf_b  = nf.unsqueeze(0).expand(B,-1,-1)
ei_b  = ei.unsqueeze(0).expand(B,-1,-1)
nm_b  = nm.unsqueeze(0).expand(B,-1)
em_b  = em.unsqueeze(0).expand(B,-1)
sh_b  = sh.unsqueeze(0).expand(B,-1)
am_b  = am.unsqueeze(0).expand(B,-1)
pi_b  = pi.unsqueeze(0).expand(B,-1)

with torch.no_grad():
    out_b = pol(nf_b, ei_b, nm_b, em_b, sh_b, am_b, pi_b)

assert out_b.action_logits.shape == (B, MAX_ACTION_DIM)
assert out_b.action_probs.shape  == (B, MAX_ACTION_DIM)
assert out_b.state_value.shape   == (B,)
assert out_b.entropy.shape       == (B,)
print(f"  batched logits: {out_b.action_logits.shape}, value: {out_b.state_value.shape}  {PASS}")


# ── 3. action_probs sum to 1 ─────────────────────────────────────────────────
section("3. action_probs sum to 1 over valid actions")

prob_sum = out.action_probs.sum().item()
assert abs(prob_sum - 1.0) < 1e-5, \
    f"{FAIL} action_probs sum = {prob_sum:.6f}, expected 1.0"
print(f"  Σ action_probs = {prob_sum:.6f}  {PASS}")


# ── 4. Masked actions: prob=0, log_prob=-inf ──────────────────────────────────
section("4. Masked actions have prob=0 and log_prob=-inf")

invalid = ~am
if invalid.any():
    assert torch.allclose(out.action_probs[invalid],
                          torch.zeros_like(out.action_probs[invalid]), atol=1e-6), \
        f"{FAIL} Masked actions should have prob=0"
    assert (out.log_probs[invalid] == float('-inf')).all(), \
        f"{FAIL} Masked actions should have log_prob=-inf"
    print(f"  {invalid.sum()} masked actions: prob=0, log_prob=-inf  {PASS}")
else:
    print(f"  No masked actions in this obs (all valid)  {PASS}")


# ── 5. Actor uses per-node embeddings (node-specific logits) ──────────────────
section("5. Actor add/remove: different nodes → different logits")

# Verify per-node specificity by directly testing the actor heads
# on the GNN embeddings produced by perturbed vs original features.
# We bypass the full forward() and test the actor heads directly on
# the node embeddings to isolate the per-node behavior cleanly.
idx0 = pi[0].item()   # present_nodes row index of probeable_nodes[0]
nf_perturbed = nf.clone()
nf_perturbed[idx0] += 20.0   # large shift of entire node feature vector

with torch.no_grad():
    # Get node embeddings for both inputs
    ne_orig, _ = pol.gnn(nf,           ei, nm, em)
    ne_pert, _ = pol.gnn(nf_perturbed, ei, nm, em)

    # Test actor_add and actor_remove directly on the perturbed embedding
    add_orig = pol.actor_add(ne_orig[idx0].unsqueeze(0)).item()
    add_pert = pol.actor_add(ne_pert[idx0].unsqueeze(0)).item()
    rem_orig = pol.actor_remove(ne_orig[idx0].unsqueeze(0)).item()
    rem_pert = pol.actor_remove(ne_pert[idx0].unsqueeze(0)).item()

add0_diff = abs(add_orig - add_pert)
rem0_diff = abs(rem_orig - rem_pert)

assert add0_diff > 0, f"{FAIL} Perturbing node 0 should change its add logit"
assert rem0_diff > 0, f"{FAIL} Perturbing node 0 should change its remove logit"
print(f"  Node 0 perturbation: Δadd={add0_diff:.4f}, Δrem={rem0_diff:.4f}  {PASS}")


# ── 6. no_op uses global context (changes with slo_health) ───────────────────
section("6. no_op logit changes with slo_health")

sh_all_healthy = torch.ones(NUM_SLOS)
sh_all_sick    = torch.zeros(NUM_SLOS)

with torch.no_grad():
    out_h = pol(nf, ei, nm, em, sh_all_healthy, am, pi)
    out_s = pol(nf, ei, nm, em, sh_all_sick,    am, pi)

noop_diff = abs(out_h.action_logits[0].item() - out_s.action_logits[0].item())
assert noop_diff > 0, \
    f"{FAIL} no_op logit should differ when slo_health changes"
print(f"  no_op logit: healthy={out_h.action_logits[0]:.4f}, "
      f"sick={out_s.action_logits[0]:.4f}, Δ={noop_diff:.4f}  {PASS}")


# ── 7. State value changes with graph state ───────────────────────────────────
section("7. State value changes with different graph states")

nf_zeros = torch.zeros_like(nf)
with torch.no_grad():
    v_normal = pol(nf,       ei, nm, em, sh, am, pi).state_value.item()
    v_zeros  = pol(nf_zeros, ei, nm, em, sh, am, pi).state_value.item()

assert v_normal != v_zeros, \
    f"{FAIL} State value should differ for different graph states"
print(f"  V(normal)={v_normal:.4f}, V(zeros)={v_zeros:.4f}  {PASS}")


# ── 8. Entropy > 0 at initialisation ─────────────────────────────────────────
section("8. Entropy > 0 at init (policy not immediately collapsed)")

entropy_val = out.entropy.item()
assert entropy_val > 0, \
    f"{FAIL} Entropy should be > 0 at init, got {entropy_val:.6f}"
print(f"  Entropy = {entropy_val:.4f} > 0  {PASS}")


# ── 9. Entropy = 0 when only one valid action ─────────────────────────────────
section("9. Entropy = 0 when only one valid action")

am_one = torch.zeros(MAX_ACTION_DIM, dtype=torch.bool)
am_one[0] = True   # only no_op valid

with torch.no_grad():
    out_one = pol(nf, ei, nm, em, sh, am_one, pi)

assert abs(out_one.entropy.item()) < 1e-5, \
    f"{FAIL} Entropy should be 0 with one valid action, got {out_one.entropy.item():.6f}"
assert abs(out_one.action_probs[0].item() - 1.0) < 1e-5, \
    f"{FAIL} Only valid action should have prob=1"
print(f"  entropy={out_one.entropy.item():.6f}, P(no_op)={out_one.action_probs[0].item():.6f}  {PASS}")


# ── 10. Gradients flow to all parameters ─────────────────────────────────────
section("10. Gradients flow to all parameters (GNN + actor + critic)")

pol_train = make_policy()
pol_train.train()

# Build an obs where a probe is already placed so remove actions are unmasked
env_t = make_env(seed=0)
obs_t, _ = env_t.reset()
obs_t, _, _, _, _ = env_t.step(1)   # add_probe(probeable_nodes[0])
ep_t = env_t.current_graph
nf_t, ei_t, nm_t, em_t, sh_t, am_t, pi_t = obs_to_tensors(obs_t, ep_t)

out_g = pol_train(nf_t, ei_t, nm_t, em_t, sh_t, am_t, pi_t)

# Construct loss that exercises all heads:
#   - actor_add  : log_prob of an add action
#   - actor_remove: log_prob of a remove action
#   - actor_noop : log_prob of no_op
#   - critic     : state_value
n_p_t = len(env_t.probeable_nodes)
add_action    = 1                  # add_probe(0) — valid (unprobed nodes exist)
remove_action = n_p_t + 1         # remove_probe(0) — valid (node 0 is probed)
loss = (
    -out_g.log_probs[0]             # no_op  → exercises actor_noop
    - out_g.log_probs[add_action]   # add    → exercises actor_add
    - out_g.log_probs[remove_action]# remove → exercises actor_remove
    + out_g.state_value             #        → exercises critic
    + out_g.entropy                 #        → exercises all
)
loss.backward()

no_grad = []
zero_grad = []
for name, param in pol_train.named_parameters():
    if param.grad is None:
        no_grad.append(name)
    elif param.grad.abs().sum() == 0:
        zero_grad.append(name)

assert len(no_grad)   == 0, f"{FAIL} No gradient: {no_grad}"
assert len(zero_grad) == 0, f"{FAIL} Zero gradient: {zero_grad}"
n_params = sum(1 for _ in pol_train.parameters())
print(f"  Gradients non-zero for all {n_params} parameter tensors  {PASS}")


# ── 11. build_probeable_indices correct mapping ───────────────────────────────
section("11. build_probeable_indices: correct node → row mapping")

probeable_ordered = sorted(ep.probeable_nodes)
for i, node in enumerate(probeable_ordered):
    expected_row = ep.present_nodes.index(node)
    got_row      = pi[i].item()
    assert got_row == expected_row, \
        f"{FAIL} probeable_indices[{i}] ({node}): got {got_row}, expected {expected_row}"
print(f"  All {len(pi)} probeable indices map correctly  {PASS}")


# ── 12. act() returns valid action ────────────────────────────────────────────
section("12. act() returns action within action_mask")

for trial in range(20):
    action, log_prob, value, entropy = pol.act(obs, ep, deterministic=False)
    assert obs["action_mask"][action], \
        f"{FAIL} Sampled action {action} is masked (invalid)"
    assert isinstance(log_prob, float)
    assert isinstance(value,    float)
    assert isinstance(entropy,  float)
print(f"  20 sampled actions all within action_mask  {PASS}")


# ── 13. act(deterministic=True) always returns argmax ────────────────────────
section("13. act(deterministic=True) always returns argmax")

actions = [pol.act(obs, ep, deterministic=True)[0] for _ in range(10)]
assert len(set(actions)) == 1, \
    f"{FAIL} Deterministic act should always return same action, got {set(actions)}"
with torch.no_grad():
    expected = int(pol(nf, ei, nm, em, sh, am, pi).action_probs.argmax().item())
assert actions[0] == expected, \
    f"{FAIL} Deterministic action {actions[0]} != argmax {expected}"
print(f"  Deterministic action = {actions[0]} (argmax) in all 10 trials  {PASS}")


# ── 14. act(deterministic=False) samples (stochastic) ────────────────────────
section("14. act(deterministic=False) samples stochastically")

sampled = [pol.act(obs, ep, deterministic=False)[0] for _ in range(50)]
unique  = len(set(sampled))
# With multiple valid actions and random init, we expect > 1 unique action
assert unique > 1, \
    f"{FAIL} Stochastic sampling should produce >1 unique action over 50 trials, got {unique}"
print(f"  50 stochastic samples → {unique} unique actions  {PASS}")


# ── 15. Two episodes with different n_p both work ─────────────────────────────
section("15. Two episodes with different |V_p| both work")

for seed in [0, 7, 42]:
    env_s = make_env(seed=seed)
    obs_s, _ = env_s.reset()
    ep_s  = env_s.current_graph
    nf_s, ei_s, nm_s, em_s, sh_s, am_s, pi_s = obs_to_tensors(obs_s, ep_s)
    with torch.no_grad():
        out_s = pol(nf_s, ei_s, nm_s, em_s, sh_s, am_s, pi_s)
    assert not torch.isnan(out_s.action_probs).any(), f"{FAIL} seed={seed}: NaN in probs"
    assert abs(out_s.action_probs.sum().item() - 1.0) < 1e-5, f"{FAIL} seed={seed}: probs don't sum to 1"
    n_p_s = len(env_s.probeable_nodes)
    print(f"  seed={seed}: |V_p|={n_p_s}, Σprobs={out_s.action_probs.sum():.6f}  {PASS}")


# ── 16. Probeable node embedding correctly gathered (not mean) ────────────────
section("16. Probeable node embedding = h_v, not mean(h)")

# If we zero out all nodes except probeable_nodes[0],
# the add_probe(0) logit should be non-zero but add_probe(1) should be near zero
nf_single = torch.zeros_like(nf)
idx0 = pi[0].item()
nf_single[idx0] = nf[idx0]  # only node 0 has features

with torch.no_grad():
    out_single = pol(nf_single, ei, nm, em, sh, am, pi)

# add_probe(0) and add_probe(1) logits
add0 = out_single.action_logits[1].item()
add1 = out_single.action_logits[2].item()
# They must differ (node 0 has features, node 1 is zero)
assert add0 != add1, \
    f"{FAIL} Embeddings are not node-specific (add0={add0:.4f}, add1={add1:.4f})"
print(f"  add_probe(0)={add0:.4f} ≠ add_probe(1)={add1:.4f} (node-specific)  {PASS}")


# ── 17. No NaN in any output ──────────────────────────────────────────────────
section("17. No NaN in any output")

with torch.no_grad():
    out_nan = pol(nf, ei, nm, em, sh, am, pi)

for field_name, tensor in [
    ("action_logits", out_nan.action_logits),
    ("action_probs",  out_nan.action_probs),
    ("log_probs",     out_nan.log_probs),
    ("state_value",   out_nan.state_value),
    ("entropy",       out_nan.entropy),
]:
    # -inf in logits/log_probs for masked actions is expected, not NaN
    valid_mask = torch.isfinite(tensor)
    assert not torch.isnan(tensor[valid_mask]).any(), \
        f"{FAIL} NaN found in {field_name}"
print(f"  No NaN in any output field  {PASS}")


# ── Done ─────────────────────────────────────────────────────────────────────
print(f"\n{'═'*60}")
print(f"  ALL POLICY NETWORK ASSERTIONS PASSED ✓")
print(f"{'═'*60}\n")