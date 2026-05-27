"""
validate_observation_builder.py
--------------------------------
Phase 3b validation: observation builder.

Run from project root:
    python scripts/validate_observation_builder.py

Checks:
  1.  Dimensions: node_features=(|V|, 62), slo_health=(6,)
  2.  Static features correct for all 24 nodes
  3.  Static block identical across consecutive build() calls
  4.  Unprobed node: probe_flag + SLO window = zeros; call rates visible
  5.  Probed node: SLI window populated, margin correct
  6.  Rolling window shifts correctly over multiple timesteps
  7.  Δmargin = margin(t) − margin(t−1)
  8.  Masking: covers SLO but not probed → slot zeros
  9.  Probed node: slot for non-covered SLO stays zero
 10.  Call rates always visible regardless of probe status
 11.  slo_health: covered + not violated → 1
 12.  slo_health: covered + violated → 0
 13.  slo_health: not covered → 0
 14.  slo_health: absent candidate → 0
 15.  reset() clears history and staleness
 16.  NODE_FEATURE_DIM = STATIC_DIM + DYNAMIC_DIM = 62
 17.  flatten_sli: nested → flat conversion
 18.  staleness = 0 when probed
 19.  staleness increments each step when unprobed
 20.  staleness resets to 0 on re-probe
 21.  stale SLI window persists (frozen) while unprobed
"""

import sys
sys.path.insert(0, ".")

import numpy as np

from src.simulator.config.slo_config import SLO_BY_ID, NUM_SLOS, NODE_TO_SLOS
from src.simulator.config.node_config import NODE_CATALOG, NODE_TYPES
from src.simulator.graph import EpisodeGraphBuilder
from src.simulator.env.observation_builder import (
    ObservationBuilder, STATIC_DIM, DYNAMIC_DIM, NODE_FEATURE_DIM,
    WINDOW_SIZE, IDX, flatten_sli,
)

PASS = "✓"
FAIL = "✗"

def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

# ── Setup ────────────────────────────────────────────────────────────────────
builder = EpisodeGraphBuilder(seed=0)
ep      = builder.next_episode()
present = ep.present_nodes

ob = ObservationBuilder(window_size=WINDOW_SIZE)
ob.reset(ep)

N = WINDOW_SIZE
K = NUM_SLOS

STALE_START = IDX.staleness_start(N)
STALE_END   = IDX.staleness_end(N)
SLO0_SLOT   = IDX.obs_window_start + 0 * (N + 2)   # SLO_0 slot start
SLO1_SLOT   = IDX.obs_window_start + 1 * (N + 2)   # SLO_1 slot start

def make_sli(
    search_lat=80.0, geo_proc=20.0,
    res_proc=80.0, res_fail=0.2,
    mongo_lat=3.0, memcache_hit=97.0,
    inc=10.0, out=5.0,
):
    snap = {
        ("search",        "search_latency_p99"):  search_lat,
        ("geo",           "processing_time_p99"): geo_proc,
        ("reservation",   "processing_time_p99"): res_proc,
        ("reservation",   "failure_rate"):         res_fail,
        ("mongodb-rate",  "query_latency_p99"):    mongo_lat,
        ("memcached-rate","hit_rate"):              memcache_hit,
    }
    for node in present:
        snap[(node, "incoming_call_rate")] = inc
        snap[(node, "outgoing_call_rate")] = out
    return snap

safe_snap = make_sli()


# ── 1. Dimensions ────────────────────────────────────────────────────────────
section("1. Output shapes")

nf, sh = ob.build(probe_set=set(), sli_snapshot=safe_snap, t=0)

assert nf.shape == (len(present), NODE_FEATURE_DIM), \
    f"{FAIL} shape {nf.shape}, expected ({len(present)}, {NODE_FEATURE_DIM})"
assert sh.shape == (NUM_SLOS,), f"{FAIL} slo_health shape {sh.shape}"
assert nf.dtype == np.float32
assert sh.dtype == np.float32
print(f"  node_features: {nf.shape}  slo_health: {sh.shape}  {PASS}")


# ── 2. Static features ───────────────────────────────────────────────────────
section("2. Static features")

for i, node in enumerate(present):
    meta = NODE_CATALOG[node]
    row  = nf[i]
    assert np.allclose(row[IDX.type_one_hot_start:IDX.type_one_hot_end], meta.type_one_hot()), \
        f"{FAIL} {node}: type_one_hot"
    assert row[IDX.is_probeable] == (1.0 if meta.probeable else 0.0), \
        f"{FAIL} {node}: is_probeable"
    assert row[IDX.degree_in]  == float(ep.G.in_degree(node)),  f"{FAIL} {node}: degree_in"
    assert row[IDX.degree_out] == float(ep.G.out_degree(node)), f"{FAIL} {node}: degree_out"
    for k in range(NUM_SLOS):
        expected = 1.0 if node in SLO_BY_ID[k].nodes else 0.0
        assert row[IDX.slo_mask_start + k] == expected, f"{FAIL} {node}: slo_mask[{k}]"
    expected_imp = sum(SLO_BY_ID[k].weight for k in NODE_TO_SLOS.get(node, []))
    assert abs(row[IDX.slo_importance] - expected_imp) < 1e-6, \
        f"{FAIL} {node}: slo_importance={row[IDX.slo_importance]:.4f} != {expected_imp:.4f}"
print(f"  All {len(present)} nodes: static features correct  {PASS}")


# ── 3. Static block fixed across build() calls ───────────────────────────────
section("3. Static block fixed across timesteps")

nf2, _ = ob.build(probe_set={"search"}, sli_snapshot=safe_snap, t=1)
assert np.allclose(nf[:, :STATIC_DIM], nf2[:, :STATIC_DIM])
print(f"  Static block unchanged across calls  {PASS}")


# ── 4. Unprobed node: probe_flag + SLO window = 0; call rates visible ────────
section("4. Unprobed node: probe_flag + SLO window = zeros")

ob.reset(ep)
nf3, _ = ob.build(probe_set=set(), sli_snapshot=safe_snap, t=0)

for i, node in enumerate(present):
    # probe_flag + SLO window (exclude staleness and call rates)
    block = nf3[i, STATIC_DIM : STALE_START]
    assert np.allclose(block, 0.0), \
        f"{FAIL} {node}: probe_flag+window block should be zero, got {block}"
    # call rates always visible
    assert nf3[i, -2] == 10.0, f"{FAIL} {node}: incoming_call_rate"
    assert nf3[i, -1] ==  5.0, f"{FAIL} {node}: outgoing_call_rate"
print(f"  probe_flag + SLO window = zeros; call rates = [10, 5]  {PASS}")


# ── 5. Probed node: SLI window and margin ────────────────────────────────────
section("5. Probed node: SLI window and margin populated")

ob.reset(ep)
nf5, _ = ob.build(probe_set={"search"}, sli_snapshot=make_sli(search_lat=80.0), t=0)

i_s    = present.index("search")
slo0   = SLO_BY_ID[0]
win    = nf5[i_s, SLO0_SLOT : SLO0_SLOT + N]
assert win[-1] == 80.0,            f"{FAIL} window[-1]={win[-1]}, expected 80"
assert np.allclose(win[:-1], 0.0), f"{FAIL} earlier window entries should be 0 at t=0"

expected_margin = slo0.margin(80.0)
got_margin      = nf5[i_s, SLO0_SLOT + N]
assert abs(got_margin - expected_margin) < 1e-6, \
    f"{FAIL} margin={got_margin:.4f}, expected {expected_margin:.4f}"
print(f"  window[-1]=80, margin={got_margin:.3f}  {PASS}")


# ── 6. Rolling window shifts ─────────────────────────────────────────────────
section("6. Rolling window over 3 timesteps")

ob.reset(ep)
for step, val in enumerate([80.0, 120.0, 160.0]):
    nf_t, _ = ob.build(probe_set={"search"}, sli_snapshot=make_sli(search_lat=val), t=step)

win_final = nf_t[present.index("search"), SLO0_SLOT : SLO0_SLOT + N]
expected  = np.array([0.0, 80.0, 120.0, 160.0])
assert np.allclose(win_final, expected), f"{FAIL} window={win_final}, expected {expected}"
print(f"  window after 3 steps = {win_final}  {PASS}")


# ── 7. Δmargin ───────────────────────────────────────────────────────────────
section("7. Δmargin = margin(t) − margin(t−1)")

expected_delta = slo0.margin(160.0) - slo0.margin(120.0)
got_delta      = nf_t[present.index("search"), SLO0_SLOT + N + 1]
assert abs(got_delta - expected_delta) < 1e-6, \
    f"{FAIL} Δmargin={got_delta:.4f}, expected {expected_delta:.4f}"
print(f"  Δmargin={got_delta:.3f} (expected {expected_delta:.3f})  {PASS}")


# ── 8. Masking: covers SLO but not probed ────────────────────────────────────
section("8. Masking: geo covers SLO_1 but is not probed")

ob.reset(ep)
nf8, _ = ob.build(probe_set={"search"}, sli_snapshot=make_sli(geo_proc=45.0), t=0)

geo_slot = nf8[present.index("geo"), SLO1_SLOT : SLO1_SLOT + N + 2]
assert np.allclose(geo_slot, 0.0), \
    f"{FAIL} geo SLO_1 slot should be masked, got {geo_slot}"
print(f"  geo SLO_1 slot = zeros when unprobed  {PASS}")


# ── 9. Non-covered SLO slot stays zero even when probed ──────────────────────
section("9. Probed node: non-covered SLO slot stays zero")

ob.reset(ep)
nf9, _ = ob.build(probe_set={"search"}, sli_snapshot=safe_snap, t=0)

search_slo1 = nf9[present.index("search"), SLO1_SLOT : SLO1_SLOT + N + 2]
assert np.allclose(search_slo1, 0.0), \
    f"{FAIL} search doesn't cover SLO_1, slot should be zero"
print(f"  search SLO_1 slot = zeros (doesn't cover it)  {PASS}")


# ── 10. Call rates always observable ─────────────────────────────────────────
section("10. Call rates always observable")

ob.reset(ep)
nf10, _ = ob.build(probe_set=set(), sli_snapshot=make_sli(inc=42.0, out=17.0), t=0)

for i, node in enumerate(present):
    assert abs(nf10[i, -2] - 42.0) < 1e-6, f"{FAIL} {node}: incoming_call_rate"
    assert abs(nf10[i, -1] - 17.0) < 1e-6, f"{FAIL} {node}: outgoing_call_rate"
print(f"  All {len(present)} nodes: call rates visible with no probes  {PASS}")


# ── 11. slo_health: covered + safe → 1 ───────────────────────────────────────
section("11. slo_health: covered + safe → 1")

ob.reset(ep)
_, sh11 = ob.build(probe_set={"search", "geo"}, sli_snapshot=safe_snap, t=0)

for slo_id in [0, 1]:
    if slo_id in ep.coverable_slos:
        assert sh11[slo_id] == 1.0, f"{FAIL} SLO_{slo_id} should be healthy"
print(f"  slo_health={sh11}  {PASS}")


# ── 12. slo_health: covered + violated → 0 ───────────────────────────────────
section("12. slo_health: covered + violated → 0")

ob.reset(ep)
_, sh12 = ob.build(
    probe_set={"search", "geo"},
    sli_snapshot=make_sli(geo_proc=65.0),  # SLO_1 violated
    t=0,
)
if 1 in ep.coverable_slos:
    assert sh12[1] == 0.0, f"{FAIL} SLO_1 violated → health=0, got {sh12[1]}"
print(f"  slo_health[1] (covered+violated) = {sh12[1]}  {PASS}")


# ── 13. slo_health: not covered → 0 ──────────────────────────────────────────
section("13. slo_health: not covered → 0")

ob.reset(ep)
_, sh13 = ob.build(probe_set=set(), sli_snapshot=safe_snap, t=0)
assert np.allclose(sh13, 0.0), f"{FAIL} All health=0 with no probes, got {sh13}"
print(f"  slo_health = zeros with no probes  {PASS}")


# ── 14. slo_health: absent candidate → 0 ─────────────────────────────────────
section("14. slo_health: absent candidate → 0")

b2 = EpisodeGraphBuilder(p_fail=0.99, p_rec=0.0, seed=5)
b2.next_episode()
ep_absent = b2.next_episode()
ob2 = ObservationBuilder()
ob2.reset(ep_absent)
absent_snap = {
    (n, m): 1.0
    for n in ep_absent.present_nodes
    for m in ["search_latency_p99","processing_time_p99","failure_rate",
              "query_latency_p99","hit_rate","incoming_call_rate","outgoing_call_rate"]
}
_, sh14 = ob2.build(
    probe_set=set(ep_absent.probeable_nodes),
    sli_snapshot=absent_snap,
    t=0,
)
for slo_id in range(NUM_SLOS):
    if slo_id not in ep_absent.coverable_slos:
        assert sh14[slo_id] == 0.0, \
            f"{FAIL} SLO_{slo_id} absent → health=0, got {sh14[slo_id]}"
print(f"  Absent SLO candidate → health=0  {PASS}")


# ── 15. reset() clears history and staleness ──────────────────────────────────
section("15. reset() clears history and staleness")

ob.reset(ep)
for step in range(N):
    ob.build(probe_set={"search"}, sli_snapshot=make_sli(search_lat=180.0), t=step)

ob.reset(ep)
nf15, _ = ob.build(probe_set={"search"}, sli_snapshot=make_sli(search_lat=50.0), t=0)

win15 = nf15[present.index("search"), SLO0_SLOT : SLO0_SLOT + N]
assert np.allclose(win15[:-1], 0.0), f"{FAIL} After reset, window prefix should be 0"
assert abs(win15[-1] - 50.0) < 1e-6, f"{FAIL} After reset, window[-1] should be 50"

# staleness should also be reset to 0
stale_search = nf15[present.index("search"), STALE_START : STALE_END]
assert np.allclose(stale_search, 0.0), \
    f"{FAIL} After reset, staleness should be 0, got {stale_search}"
print(f"  window={win15}, staleness={stale_search}  {PASS}")


# ── 16. NODE_FEATURE_DIM = 62 ─────────────────────────────────────────────────
section("16. NODE_FEATURE_DIM constant consistency")

ob.reset(ep)
nf16, _ = ob.build(probe_set={"search"}, sli_snapshot=safe_snap, t=0)
assert nf16.shape[1] == NODE_FEATURE_DIM, \
    f"{FAIL} array width {nf16.shape[1]} != NODE_FEATURE_DIM={NODE_FEATURE_DIM}"
assert STATIC_DIM + DYNAMIC_DIM == NODE_FEATURE_DIM, \
    f"{FAIL} {STATIC_DIM}+{DYNAMIC_DIM} != {NODE_FEATURE_DIM}"
print(f"  NODE_FEATURE_DIM={NODE_FEATURE_DIM} = STATIC({STATIC_DIM}) + DYNAMIC({DYNAMIC_DIM})  {PASS}")


# ── 17. flatten_sli ──────────────────────────────────────────────────────────
section("17. flatten_sli: nested → flat")

nested = {
    "search":      {"search_latency_p99": 80.0, "incoming_call_rate": 10.0},
    "geo":         {"processing_time_p99": 20.0},
    "mongodb-rate":{"query_latency_p99": 3.0},
}
flat = flatten_sli(nested)
assert flat[("search",       "search_latency_p99")]  == 80.0
assert flat[("search",       "incoming_call_rate")]  == 10.0
assert flat[("geo",          "processing_time_p99")] == 20.0
assert flat[("mongodb-rate", "query_latency_p99")]   ==  3.0
assert len(flat) == 4
print(f"  flatten_sli: {len(flat)} entries, values correct  {PASS}")


# ── 18. Staleness = 0 when probed ────────────────────────────────────────────
section("18. Staleness = 0 when probed")

ob.reset(ep)
nf18, _ = ob.build(probe_set={"search", "geo"}, sli_snapshot=safe_snap, t=0)

i_s = present.index("search")
i_g = present.index("geo")
# SLO_0 → search, SLO_1 → geo
assert nf18[i_s, STALE_START + 0] == 0.0, f"{FAIL} search SLO_0 staleness should be 0 when probed"
assert nf18[i_g, STALE_START + 1] == 0.0, f"{FAIL} geo SLO_1 staleness should be 0 when probed"
print(f"  staleness=0 for probed nodes  {PASS}")


# ── 19. Staleness increments when unprobed ────────────────────────────────────
section("19. Staleness increments each step when unprobed")

ob.reset(ep)
# Probe geo at t=0, then remove probe for t=1,2,3
ob.build(probe_set={"geo"},  sli_snapshot=safe_snap, t=0)  # staleness=0
ob.build(probe_set=set(),    sli_snapshot=safe_snap, t=1)  # staleness=1
ob.build(probe_set=set(),    sli_snapshot=safe_snap, t=2)  # staleness=2
nf19, _ = ob.build(probe_set=set(), sli_snapshot=safe_snap, t=3)  # staleness=3

i_g = present.index("geo")
stale_geo_slo1 = nf19[i_g, STALE_START + 1]
assert stale_geo_slo1 == 3.0, \
    f"{FAIL} geo SLO_1 staleness after 3 unprobed steps = {stale_geo_slo1}, expected 3"
print(f"  geo SLO_1 staleness after 3 unprobed steps = {stale_geo_slo1}  {PASS}")


# ── 20. Staleness resets to 0 on re-probe ────────────────────────────────────
section("20. Staleness resets to 0 on re-probe")

# Continue from test 19 — now re-probe geo
nf20, _ = ob.build(probe_set={"geo"}, sli_snapshot=safe_snap, t=4)
stale_after_reprobe = nf20[present.index("geo"), STALE_START + 1]
assert stale_after_reprobe == 0.0, \
    f"{FAIL} staleness after re-probe = {stale_after_reprobe}, expected 0"
print(f"  staleness resets to 0 on re-probe  {PASS}")


# ── 21. Stale SLI window persists while unprobed ─────────────────────────────
section("21. Stale SLI window frozen while unprobed")

ob.reset(ep)
# Probe geo at 45ms, then remove probe for 3 steps
ob.build(probe_set={"geo"}, sli_snapshot=make_sli(geo_proc=45.0), t=0)

for step in range(1, 4):
    nf_unprobed, _ = ob.build(probe_set=set(), sli_snapshot=make_sli(geo_proc=99.0), t=step)

# Window should still show the last observed value (45.0), not the unseen 99.0
i_g  = present.index("geo")
win  = nf_unprobed[i_g, SLO1_SLOT : SLO1_SLOT + N + 2]
# Note: window is masked (zeros) because geo is unprobed — the frozen history
# is not shown, only the staleness counter reveals it was once observed
assert np.allclose(win, 0.0), \
    f"{FAIL} unprobed geo SLO_1 slot should be masked (zeros), got {win}"
# But internal history buffer should still hold the last seen value
internal_buf = ob._sli_history.get(("geo", 1), np.zeros(N))
assert internal_buf[-1] == 45.0, \
    f"{FAIL} internal history should preserve last seen value 45.0, got {internal_buf[-1]}"
print(f"  masked window = zeros; internal buffer frozen at last seen value  {PASS}")


# ── Done ─────────────────────────────────────────────────────────────────────
print(f"\n{'═'*60}")
print(f"  ALL OBSERVATION BUILDER ASSERTIONS PASSED ✓")
print(f"{'═'*60}\n")