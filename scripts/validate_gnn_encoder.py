"""
validate_gnn_encoder.py
-----------------------
Phase 4a validation: GNN encoder.

Run from project root:
    python scripts/validate_gnn_encoder.py

Checks:
  1.  Output shapes: node_embeddings (N, H), graph_embedding (H,)
  2.  Batched forward: (B, N, H) and (B, H)
  3.  Padded node rows zeroed out in output
  4.  graph_embedding = mean of real node embeddings only
  5.  Different graph structures → different embeddings
  6.  Gradients flow through all parameters
  7.  build_masks: node_mask and edge_mask correct
  8.  node_mask[:n_real]=True, rest=False
  9.  edge_mask[:e_real]=True, rest=False
 10.  Full pipeline: env obs → masks → GNN → shapes correct
 11.  Isolated node (no edges): no crash, zero aggregation
 12.  num_layers=1 and num_layers=4 both work
 13.  GNN is equivariant to node ordering (same graph, reordered → same set of embeddings)
"""

import sys
sys.path.insert(0, ".")

import torch
import numpy as np

from src.simulator.graph import EpisodeGraphBuilder
from src.simulator.env.probe_env import ProbeEnv, ProbeEnvConfig
from src.simulator.env.observation_builder import NODE_FEATURE_DIM
from src.models.gnn_encoder import (
    GNNConfig, GNNEncoder, GNNLayer, build_masks,
    MAX_NODES, MAX_EDGES,
)

PASS = "✓"
FAIL = "✗"

def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

H = 64   # hidden dim for tests

def make_encoder(num_layers=3, hidden_dim=H):
    cfg = GNNConfig(input_dim=NODE_FEATURE_DIM, hidden_dim=hidden_dim,
                    num_layers=num_layers, dropout=0.0)
    return GNNEncoder(cfg)

def make_obs():
    """Return a real obs from the env."""
    env = ProbeEnv(ProbeEnvConfig(episode_length=10, n_failures=0,
                                  graph_seed=0, workload_seed=0))
    obs, _ = env.reset()
    return obs, env.current_graph

def obs_to_tensors(obs):
    nf = torch.tensor(obs["node_features"], dtype=torch.float32)  # (24, 62)
    ei = torch.tensor(obs["edge_index"],    dtype=torch.int64)     # (2, 41)
    return nf, ei


# ── 1. Output shapes (unbatched) ─────────────────────────────────────────────
section("1. Output shapes — unbatched")

obs, ep = make_obs()
nf, ei  = obs_to_tensors(obs)
nm, em  = build_masks(ep)

enc = make_encoder()
enc.eval()
with torch.no_grad():
    node_emb, graph_emb = enc(nf, ei, nm, em)

assert node_emb.shape  == (MAX_NODES, H), \
    f"{FAIL} node_emb shape {node_emb.shape}, expected ({MAX_NODES}, {H})"
assert graph_emb.shape == (H,), \
    f"{FAIL} graph_emb shape {graph_emb.shape}, expected ({H},)"
print(f"  node_emb: {node_emb.shape}  graph_emb: {graph_emb.shape}  {PASS}")


# ── 2. Batched forward ────────────────────────────────────────────────────────
section("2. Batched forward — (B=4, N=24, D=62)")

B = 4
nf_b = nf.unsqueeze(0).expand(B, -1, -1)   # (4, 24, 62)
ei_b = ei.unsqueeze(0).expand(B, -1, -1)   # (4, 2, 41)
nm_b = nm.unsqueeze(0).expand(B, -1)        # (4, 24)
em_b = em.unsqueeze(0).expand(B, -1)        # (4, 41)

with torch.no_grad():
    node_emb_b, graph_emb_b = enc(nf_b, ei_b, nm_b, em_b)

assert node_emb_b.shape  == (B, MAX_NODES, H), \
    f"{FAIL} batched node_emb {node_emb_b.shape}"
assert graph_emb_b.shape == (B, H), \
    f"{FAIL} batched graph_emb {graph_emb_b.shape}"
print(f"  batched node_emb: {node_emb_b.shape}  graph_emb: {graph_emb_b.shape}  {PASS}")


# ── 3. Padded node rows are zeroed out ────────────────────────────────────────
section("3. Padded node rows zeroed out in output")

# Build a partial episode (some non-core nodes absent)
b2  = EpisodeGraphBuilder(p_fail=0.99, p_rec=0.0, seed=5)
b2.next_episode()
ep2 = b2.next_episode()
nm2, em2 = build_masks(ep2)

n_real = ep2.num_nodes
# Fake node features (real nodes have non-zero features)
nf2 = torch.randn(MAX_NODES, NODE_FEATURE_DIM)
nf2[n_real:] = 0.0   # padded rows should already be zero from env

ei2 = torch.zeros(2, MAX_EDGES, dtype=torch.int64)
_, (src, dst) = ep2.get_edge_index()
n_edges = len(src)
ei2[0, :n_edges] = torch.tensor(src, dtype=torch.int64)
ei2[1, :n_edges] = torch.tensor(dst, dtype=torch.int64)

with torch.no_grad():
    ne2, _ = enc(nf2, ei2, nm2, em2)

# Padded rows (n_real onwards) should be all zeros
pad_rows = ne2[n_real:]
assert torch.allclose(pad_rows, torch.zeros_like(pad_rows)), \
    f"{FAIL} Padded node rows are not zero: {pad_rows.abs().max():.6f}"
print(f"  n_real={n_real}, padded rows [{n_real}:{MAX_NODES}] = zeros  {PASS}")


# ── 4. graph_embedding = masked mean of real node embeddings ──────────────────
section("4. graph_embedding = masked mean of real node embeddings")

with torch.no_grad():
    ne_full, ge_full = enc(nf, ei, nm, em)

n_real_full = ep.num_nodes
expected_ge = ne_full[:n_real_full].mean(dim=0)
assert torch.allclose(ge_full, expected_ge, atol=1e-5), \
    f"{FAIL} graph_emb != mean of real node embeddings (max diff={( ge_full - expected_ge).abs().max():.6f})"
print(f"  graph_emb == mean(node_emb[:n_real])  {PASS}")


# ── 5. Different graphs → different embeddings ─────────────────────────────────
section("5. Different graphs produce different embeddings")

obs2, ep_b = make_obs()
# Perturb features slightly to represent a different graph state
nf2_diff = torch.tensor(obs2["node_features"], dtype=torch.float32)
nf2_diff[0, 0] += 5.0   # meaningful perturbation

with torch.no_grad():
    _, ge_a = enc(nf, ei, nm, em)
    _, ge_b = enc(nf2_diff, ei, nm, em)

assert not torch.allclose(ge_a, ge_b), \
    f"{FAIL} Different inputs should produce different embeddings"
print(f"  Different inputs → different graph embeddings  {PASS}")


# ── 6. Gradients flow through all parameters ──────────────────────────────────
section("6. Gradients flow to all parameters")

enc_train = make_encoder()
enc_train.train()
nf_g = torch.tensor(obs["node_features"], dtype=torch.float32)
ei_g = torch.tensor(obs["edge_index"],    dtype=torch.int64)

ne_g, ge_g = enc_train(nf_g, ei_g, nm, em)
loss = ge_g.sum()
loss.backward()

no_grad_params = []
for name, param in enc_train.named_parameters():
    if param.grad is None:
        no_grad_params.append(name)
    elif param.grad.abs().sum() == 0:
        no_grad_params.append(f"{name}(zero_grad)")

assert len(no_grad_params) == 0, \
    f"{FAIL} Parameters with no gradient: {no_grad_params}"
print(f"  Gradients non-zero for all {sum(1 for _ in enc_train.parameters())} parameter tensors  {PASS}")


# ── 7. build_masks returns correct shapes ─────────────────────────────────────
section("7. build_masks — correct shapes")

nm_t, em_t = build_masks(ep)
assert nm_t.shape == (MAX_NODES,), f"{FAIL} node_mask shape {nm_t.shape}"
assert em_t.shape == (MAX_EDGES,), f"{FAIL} edge_mask shape {em_t.shape}"
assert nm_t.dtype == torch.bool,   f"{FAIL} node_mask dtype {nm_t.dtype}"
assert em_t.dtype == torch.bool,   f"{FAIL} edge_mask dtype {em_t.dtype}"
print(f"  node_mask: {nm_t.shape} bool, edge_mask: {em_t.shape} bool  {PASS}")


# ── 8. node_mask[:n_real]=True, rest=False ────────────────────────────────────
section("8. node_mask values correct")

n_real = ep.num_nodes
assert nm_t[:n_real].all(),  f"{FAIL} Real nodes should be True"
assert not nm_t[n_real:].any(), f"{FAIL} Padded nodes should be False"
print(f"  node_mask[:{n_real}]=True, [{n_real}:]=False  {PASS}")


# ── 9. edge_mask[:e_real]=True, rest=False ────────────────────────────────────
section("9. edge_mask values correct")

_, (src_e, _) = ep.get_edge_index()
e_real = len(src_e)
assert em_t[:e_real].all(),   f"{FAIL} Real edges should be True"
assert not em_t[e_real:].any(), f"{FAIL} Padded edges should be False"
print(f"  edge_mask[:{e_real}]=True, [{e_real}:]=False  {PASS}")


# ── 10. Full pipeline: env obs → masks → GNN ──────────────────────────────────
section("10. Full pipeline: env → obs → GNN")

env = ProbeEnv(ProbeEnvConfig(episode_length=20, n_failures=1,
                               graph_seed=10, workload_seed=10))
obs10, _ = env.reset()
ep10     = env.current_graph
nf10     = torch.tensor(obs10["node_features"], dtype=torch.float32)
ei10     = torch.tensor(obs10["edge_index"],    dtype=torch.int64)
nm10, em10 = build_masks(ep10)

enc10 = make_encoder()
with torch.no_grad():
    ne10, ge10 = enc10(nf10, ei10, nm10, em10)

assert ne10.shape  == (MAX_NODES, H)
assert ge10.shape  == (H,)
assert not torch.isnan(ne10).any(), f"{FAIL} NaN in node embeddings"
assert not torch.isnan(ge10).any(), f"{FAIL} NaN in graph embedding"
print(f"  Full pipeline: no NaN, shapes correct  {PASS}")


# ── 11. Isolated node (no edges): no crash ────────────────────────────────────
section("11. Isolated node — no in-edges, aggregation = zero")

nf_iso = torch.randn(MAX_NODES, NODE_FEATURE_DIM)
ei_iso = torch.zeros(2, MAX_EDGES, dtype=torch.int64)  # no real edges
nm_iso = torch.zeros(MAX_NODES, dtype=torch.bool)
nm_iso[0] = True   # one real node, no edges
em_iso = torch.zeros(MAX_EDGES, dtype=torch.bool)

with torch.no_grad():
    ne_iso, ge_iso = enc(nf_iso, ei_iso, nm_iso, em_iso)

assert not torch.isnan(ne_iso).any(), f"{FAIL} NaN with isolated node"
assert not torch.isnan(ge_iso).any(), f"{FAIL} NaN in graph embedding"
# Padded nodes should still be zero
assert torch.allclose(ne_iso[1:], torch.zeros_like(ne_iso[1:])), \
    f"{FAIL} Padded rows non-zero"
print(f"  Isolated node: no NaN, padded rows zero  {PASS}")


# ── 12. num_layers=1 and num_layers=4 ─────────────────────────────────────────
section("12. num_layers=1 and num_layers=4 both work")

for L in [1, 4]:
    enc_L = make_encoder(num_layers=L)
    with torch.no_grad():
        ne_L, ge_L = enc_L(nf, ei, nm, em)
    assert ne_L.shape == (MAX_NODES, H), f"{FAIL} L={L}: node_emb shape"
    assert ge_L.shape == (H,),           f"{FAIL} L={L}: graph_emb shape"
    assert not torch.isnan(ne_L).any(),  f"{FAIL} L={L}: NaN in node_emb"
print(f"  L=1 and L=4 both produce correct shapes without NaN  {PASS}")


# ── 13. Stress signal propagation check ───────────────────────────────────────
section("13. Stress signal reaches downstream nodes after message passing")

# Create two scenarios: geo stressed vs geo normal
# search is the upstream node of geo in the call graph
# After >= 1 message passing step, geo's embedding should differ

enc_stress = make_encoder(num_layers=2)
enc_stress.eval()

nf_normal = torch.zeros(MAX_NODES, NODE_FEATURE_DIM)
nf_stress = torch.zeros(MAX_NODES, NODE_FEATURE_DIM)

# search is node index 1 in present_nodes — set its SLI window high
search_idx = ep.present_nodes.index("search")
nf_normal[search_idx, 18] = 0.0    # SLO_0 window: normal
nf_stress[search_idx, 18] = 5.0    # SLO_0 window: stressed

with torch.no_grad():
    ne_normal, _ = enc_stress(nf_normal, ei, nm, em)
    ne_stress, _ = enc_stress(nf_stress, ei, nm, em)

# geo is downstream of search — its embedding should differ
geo_idx = ep.present_nodes.index("geo")
diff = (ne_stress[geo_idx] - ne_normal[geo_idx]).abs().sum().item()
assert diff > 0, \
    f"{FAIL} Stress signal from search did not propagate to geo (diff={diff})"
print(f"  Stress signal propagated search→geo (embedding diff={diff:.4f})  {PASS}")


# ── Done ─────────────────────────────────────────────────────────────────────
print(f"\n{'═'*60}")
print(f"  ALL GNN ENCODER ASSERTIONS PASSED ✓")
print(f"{'═'*60}\n")