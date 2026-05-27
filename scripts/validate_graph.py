"""
validate_graph.py
-----------------
Phase 1 validation: topology + episode graph perturbation.

Run from the project root:
    python scripts/validate_graph.py

Checks:
  1. Full graph structure (node/edge counts, attributes)
  2. Core nodes are never absent after perturbation
  3. Non-core nodes can appear/disappear correctly
  4. EpisodeGraph fields are consistent
  5. Coverable SLOs reflect node presence
  6. Edge index shape is correct
  7. Action space size formula holds
  8. Upstream/downstream helpers work
"""

import sys
sys.path.insert(0, ".")

from collections import Counter
from src.simulator.graph import (
    build_full_graph,
    get_neighbors,
    get_upstream_nodes,
    get_downstream_nodes,
    adjacency_matrix,
    edge_index,
    EpisodeGraphBuilder,
    CALL_EDGES,
    INFRA_EDGES,
)
from src.simulator.config.node_config import CORE_NODES, NON_CORE_NODES, PROBEABLE_NODES

PASS = "✓"
FAIL = "✗"


def section(title: str):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


# ── 1. Full graph ────────────────────────────────────────────────────────────
section("1. Full-health graph structure")

G_full = build_full_graph()

print(f"  Total nodes      : {G_full.number_of_nodes()}  (expected 24)")
print(f"  Call edges       : {len(CALL_EDGES)}            (expected 9)")
print(f"  Infra edges      : {len(INFRA_EDGES)}           (expected 32)")
print(f"  Total edges      : {G_full.number_of_edges()}  (expected 41)")

assert G_full.number_of_nodes() == 24,  f"{FAIL} Expected 24 nodes"
assert len(CALL_EDGES)  == 9,          f"{FAIL} Expected 9 call edges"
assert len(INFRA_EDGES) == 32,          f"{FAIL} Expected 32 infra edges"
assert G_full.number_of_edges() == 41,  f"{FAIL} Expected 41 edges total"

# All nodes have required attributes
for node in G_full.nodes:
    attrs = G_full.nodes[node]
    assert "node_type"    in attrs, f"{FAIL} {node} missing node_type"
    assert "core"         in attrs, f"{FAIL} {node} missing core"
    assert "probeable"    in attrs, f"{FAIL} {node} missing probeable"
    assert "type_one_hot" in attrs, f"{FAIL} {node} missing type_one_hot"
    assert sum(attrs["type_one_hot"]) == 1, f"{FAIL} {node} one-hot invalid"

print(f"  Node attributes  : {PASS}")
print(f"  {PASS} Full graph OK")


# ── 2. Topology helpers ──────────────────────────────────────────────────────
section("2. Topology helpers")

# get_neighbors
nbrs = get_neighbors("search", G_full)
assert "frontend" in nbrs["predecessors"],  f"{FAIL} frontend should precede search"
assert "geo"      in nbrs["successors"],    f"{FAIL} search should call geo"
assert "rate"     in nbrs["successors"],    f"{FAIL} search should call rate"
print(f"  get_neighbors(search): preds={nbrs['predecessors']}, succs={nbrs['successors'][:3]}... {PASS}")

# get_upstream_nodes (call edges only)
upstream_geo = get_upstream_nodes("geo", G_full, call_only=True)
assert "search"   in upstream_geo, f"{FAIL} search is upstream of geo"
assert "frontend" in upstream_geo, f"{FAIL} frontend is upstream of geo"
print(f"  upstream(geo, call_only): {sorted(upstream_geo)} {PASS}")

# get_downstream_nodes (call edges only)
downstream_search = get_downstream_nodes("search", G_full, call_only=True)
assert "geo"  in downstream_search, f"{FAIL} geo is downstream of search"
assert "rate" in downstream_search, f"{FAIL} rate is downstream of search"
print(f"  downstream(search, call_only): {sorted(downstream_search)} {PASS}")

# adjacency_matrix
node_order, matrix = adjacency_matrix(G_full)
assert len(node_order) == 24,               f"{FAIL} Expected 24 nodes in order"
assert len(matrix) == 24,                   f"{FAIL} Matrix rows mismatch"
assert all(len(r) == 24 for r in matrix),   f"{FAIL} Matrix cols mismatch"
total_ones = sum(sum(row) for row in matrix)
assert total_ones == 41, f"{FAIL} Expected 41 ones in adj matrix, got {total_ones}"
print(f"  adjacency_matrix: shape 24×24, {total_ones} edges {PASS}")

# edge_index (COO format)
node_order_ei, (src_list, dst_list) = edge_index(G_full)
assert len(src_list) == len(dst_list) == 41, f"{FAIL} COO length mismatch"
print(f"  edge_index: {len(src_list)} edges in COO format {PASS}")


# ── 3. Episode graph — first episode (all present) ───────────────────────────
section("3. Episode graph — episode 0 (all nodes present)")

builder = EpisodeGraphBuilder(p_fail=0.15, p_rec=0.60, seed=42)
ep0 = builder.next_episode()

print(ep0.summary())

assert ep0.num_nodes == 24,     f"{FAIL} Episode 0 should have 24 nodes"
assert len(ep0.absent_nodes) == 0, f"{FAIL} Episode 0 should have 0 absent nodes"
assert ep0.num_probeable == 22, f"{FAIL} Expected 22 probeable nodes"
assert ep0.action_space_size == 2 * 22 + 1 == 45, f"{FAIL} Expected action space 45"
assert ep0.node_index["frontend"] == 0, f"{FAIL} frontend should be index 0"
print(f"  {PASS} Episode 0 structure OK")


# ── 4. Core nodes never absent ───────────────────────────────────────────────
section("4. Core nodes are NEVER removed across 200 episodes")

builder2 = EpisodeGraphBuilder(p_fail=0.99, p_rec=0.01, seed=0)  # extreme failure rate
core_ever_absent = set()
for _ in range(200):
    ep = builder2.next_episode()
    for core_node in CORE_NODES:
        if not ep.is_present(core_node):
            core_ever_absent.add(core_node)

if core_ever_absent:
    print(f"  {FAIL} Core nodes were absent: {core_ever_absent}")
    sys.exit(1)
else:
    print(f"  Tested 200 episodes with p_fail=0.99  {PASS}")
    print(f"  No core node was ever absent           {PASS}")


# ── 5. Non-core nodes do appear/disappear ────────────────────────────────────
section("5. Non-core nodes transition correctly over 500 episodes")

builder3 = EpisodeGraphBuilder(p_fail=0.3, p_rec=0.5, seed=7)
ever_absent  = set()
ever_present = set()

for _ in range(500):
    ep = builder3.next_episode()
    ever_absent  |= ep.absent_nodes
    ever_present |= set(ep.present_nodes) & set(NON_CORE_NODES)

print(f"  Non-core nodes ever absent  : {sorted(ever_absent)}")
print(f"  Non-core nodes ever present : {sorted(ever_present)}")
assert len(ever_absent)  > 0, f"{FAIL} No non-core node was ever absent"
assert len(ever_present) > 0, f"{FAIL} No non-core node was ever present"
print(f"  {PASS} Bernoulli transitions working")


# ── 6. Coverable SLOs reflect node presence ──────────────────────────────────
section("6. Coverable SLOs reflect absent candidate nodes")

from src.simulator.config.slo_config import SLO_CATALOG

# Force mongodb-rate absent to make SLO_4 uncoverable
builder4 = EpisodeGraphBuilder(force_all_present=False, seed=99)

found_slo4_uncoverable = False
for _ in range(300):
    ep = builder4.next_episode()
    if "mongodb-rate" not in ep.present_nodes:
        assert 4 not in ep.coverable_slos, f"{FAIL} SLO_4 should be uncoverable"
        found_slo4_uncoverable = True
        break

# Always: if memcached-rate absent, SLO_5 uncoverable
for _ in range(300):
    ep = builder4.next_episode()
    if "memcached-rate" not in ep.present_nodes:
        assert 5 not in ep.coverable_slos, f"{FAIL} SLO_5 should be uncoverable"
        break

print(f"  SLO uncoverability when candidate node absent {PASS}")


# ── 7. EpisodeGraph node_index consistency ───────────────────────────────────
section("7. node_index is consistent with present_nodes")

builder5 = EpisodeGraphBuilder(p_fail=0.2, p_rec=0.5, seed=13)
for i in range(50):
    ep = builder5.next_episode()
    for name, idx in ep.node_index.items():
        assert ep.present_nodes[idx] == name, \
            f"{FAIL} node_index[{name}]={idx} but present_nodes[{idx}]={ep.present_nodes[idx]}"

print(f"  node_index consistent across 50 episodes {PASS}")


# ── 8. edge_index on episode graph ───────────────────────────────────────────
section("8. edge_index on episode graph (COO format)")

ep_test = builder.next_episode()
node_order_ep, (src_ep, dst_ep) = ep_test.get_edge_index()

assert len(src_ep) == len(dst_ep), f"{FAIL} COO src/dst length mismatch"
assert len(node_order_ep) == ep_test.num_nodes, f"{FAIL} node count mismatch"
# All indices must be within bounds
assert all(0 <= i < ep_test.num_nodes for i in src_ep), f"{FAIL} src index out of range"
assert all(0 <= j < ep_test.num_nodes for j in dst_ep), f"{FAIL} dst index out of range"
print(f"  edge_index: {len(src_ep)} edges, {ep_test.num_nodes} nodes {PASS}")


# ── Done ─────────────────────────────────────────────────────────────────────
print(f"\n{'═' * 50}")
print(f"  ALL PHASE 1 ASSERTIONS PASSED ✓")
print(f"{'═' * 50}\n")