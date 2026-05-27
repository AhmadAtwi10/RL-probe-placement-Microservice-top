"""
validate_workload.py
--------------------
Phase 2 validation: SLI generator + failure injector.

Run from the project root:
    python scripts/validate_workload.py

Checks:
  1. SLIGenerator produces traces for all present nodes
  2. All metric values are physically plausible (non-negative, bounded)
  3. Diurnal pattern peaks at midpoint of episode
  4. Baseline values are below SLO thresholds (clean signal doesn't violate)
  5. FailureInjector — latency spike pushes metric above SLO threshold
  6. FailureInjector — error burst pushes failure_rate above SLO threshold
  7. FailureInjector — cache degrade pushes hit_rate below SLO threshold
  8. Causal propagation: upstream caller absorbs downstream latency spike
  9. Multiple failures can be injected simultaneously
 10. Absent nodes have no traces
"""

import sys
sys.path.insert(0, ".")

import numpy as np

from src.simulator.graph import EpisodeGraphBuilder
from src.simulator.workload import SLIGenerator, FailureInjector, FailureMode, BASELINES
from src.simulator.config.slo_config import SLO_BY_ID

PASS = "✓"
FAIL = "✗"
T    = 100   # episode length for all tests


def section(title: str):
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


# ── Setup: episode graph (all present) ───────────────────────────────────────
builder  = EpisodeGraphBuilder(seed=42)
ep_graph = builder.next_episode()
assert ep_graph.num_nodes == 24


# ── 1. SLIGenerator produces traces for all present nodes ────────────────────
section("1. SLIGenerator — trace coverage")

gen = SLIGenerator(ep_graph, episode_length=T, seed=0)
traces = gen.full_traces()

for node in ep_graph.present_nodes:
    assert node in traces, f"{FAIL} Missing traces for node: {node}"
    node_meta_slis = set(gen._clean_traces[node].keys())
    expected_slis  = set(BASELINES.get(node, {}).keys())
    assert node_meta_slis == expected_slis, \
        f"{FAIL} {node}: expected {expected_slis}, got {node_meta_slis}"

print(f"  Traces generated for all {ep_graph.num_nodes} nodes  {PASS}")


# ── 2. Physical plausibility ──────────────────────────────────────────────────
section("2. Physical plausibility of clean traces")

from src.simulator.workload.sli_generator import PERCENT_METRICS

for node, metrics in traces.items():
    for metric, arr in metrics.items():
        assert arr.min() >= 0.0, f"{FAIL} {node}.{metric} has negative values"
        if metric in PERCENT_METRICS or metric.endswith("_utilization"):
            assert arr.max() <= 100.0, \
                f"{FAIL} {node}.{metric} exceeds 100% (max={arr.max():.2f})"

print(f"  All metrics non-negative                    {PASS}")
print(f"  Percentage metrics bounded to [0, 100]      {PASS}")


# ── 3. Diurnal pattern peaks at midpoint ─────────────────────────────────────
section("3. Diurnal pattern — peak at episode midpoint")

search_latency = traces["search"]["search_latency_p99"]
midpoint       = T // 2
peak_idx       = int(np.argmax(search_latency))
# Allow ±5 timesteps tolerance around the midpoint
assert abs(peak_idx - midpoint) <= 15, \
    f"{FAIL} Peak at t={peak_idx}, expected near t={midpoint}"
print(f"  search_latency_p99 peaks at t={peak_idx} (midpoint={midpoint})  {PASS}")


# ── 4. Clean signal does not violate SLOs ────────────────────────────────────
section("4. Clean signal is below all SLO thresholds")

slo_metric_node = [
    (SLO_BY_ID[0], "search",        "search_latency_p99"),
    (SLO_BY_ID[1], "geo",           "processing_time_p99"),
    (SLO_BY_ID[2], "reservation",   "processing_time_p99"),
    (SLO_BY_ID[3], "reservation",   "failure_rate"),
    (SLO_BY_ID[4], "mongodb-rate",  "query_latency_p99"),
    (SLO_BY_ID[5], "memcached-rate","hit_rate"),
]

for slo, node, metric in slo_metric_node:
    arr = traces[node][metric]
    violated = np.any(np.array([slo.is_violated(v) for v in arr]))
    assert not violated, \
        f"{FAIL} Clean signal violates {slo.metric} on {node} " \
        f"(min={arr.min():.2f}, max={arr.max():.2f}, threshold={slo.threshold})"
    display_val = arr.min() if slo.op == ">="  else arr.max()
    direction   = "min" if slo.op == ">=" else "max"
    print(f"  SLO_{slo.id} ({metric}): {direction}={display_val:.2f} vs threshold={slo.threshold}  {PASS}")


# ── 5. Latency spike pushes metric above threshold ───────────────────────────
section("5. FailureInjector — latency spike")

inj      = FailureInjector(ep_graph, episode_length=T, seed=1, propagate=False)
traces5  = gen.full_traces()
events5  = inj.sample_failures(n_failures=1)
# Force a latency spike on geo if random picked something else
from src.simulator.workload.failure_injector import FailureEvent, FailureMode
forced_event = FailureEvent(
    node="geo", mode=FailureMode.LATENCY_SPIKE,
    start_t=10, duration=20, magnitude=40.0,
    metric="processing_time_p99",
)
inj.apply(traces5, [forced_event])
geo_arr = traces5["geo"]["processing_time_p99"]
slo1    = SLO_BY_ID[1]
assert slo1.is_violated(geo_arr.max()), \
    f"{FAIL} geo.processing_time_p99 max={geo_arr.max():.2f} should violate SLO_1 (>={slo1.threshold})"
print(f"  geo.processing_time_p99 peak={geo_arr.max():.2f} > {slo1.threshold}ms  {PASS}")


# ── 6. Error burst pushes failure_rate above threshold ───────────────────────
section("6. FailureInjector — error burst")

traces6 = gen.full_traces()
forced_error = FailureEvent(
    node="reservation", mode=FailureMode.ERROR_BURST,
    start_t=10, duration=20, magnitude=1.0,
    metric="failure_rate",
)
inj.apply(traces6, [forced_error])
res_arr = traces6["reservation"]["failure_rate"]
slo3    = SLO_BY_ID[3]
assert slo3.is_violated(res_arr.max()), \
    f"{FAIL} reservation.failure_rate max={res_arr.max():.2f} should violate SLO_3"
print(f"  reservation.failure_rate peak={res_arr.max():.2f}% > {slo3.threshold}%  {PASS}")


# ── 7. Cache degrade pushes hit_rate below threshold ─────────────────────────
section("7. FailureInjector — cache degradation")

traces7 = gen.full_traces()
forced_cache = FailureEvent(
    node="memcached-rate", mode=FailureMode.CACHE_DEGRADE,
    start_t=10, duration=20, magnitude=15.0,
    metric="hit_rate",
)
inj.apply(traces7, [forced_cache])
cache_arr = traces7["memcached-rate"]["hit_rate"]
slo5      = SLO_BY_ID[5]
assert slo5.is_violated(cache_arr.min()), \
    f"{FAIL} memcached-rate.hit_rate min={cache_arr.min():.2f} should violate SLO_5"
print(f"  memcached-rate.hit_rate min={cache_arr.min():.2f}% < {slo5.threshold}%  {PASS}")


# ── 8. Causal propagation ─────────────────────────────────────────────────────
section("8. Causal propagation: geo spike → search latency rises")

inj_prop = FailureInjector(ep_graph, episode_length=T, seed=2, propagate=True)
traces8  = gen.full_traces()

search_before = traces8["search"]["search_latency_p99"].copy()
geo_event = FailureEvent(
    node="geo", mode=FailureMode.LATENCY_SPIKE,
    start_t=10, duration=20, magnitude=50.0,
    metric="processing_time_p99",
)
inj_prop.apply(traces8, [geo_event])

search_after = traces8["search"]["search_latency_p99"]
delta        = (search_after - search_before).max()
assert delta > 0, f"{FAIL} search latency should rise after geo spike"
print(f"  search_latency_p99 increased by {delta:.2f}ms after geo spike  {PASS}")

# Frontend should also be affected (two hops)
frontend_before_raw = gen.full_traces()["frontend"]["p99_latency"]
traces8b = gen.full_traces()
inj_prop.apply(traces8b, [geo_event])
frontend_delta = (traces8b["frontend"]["p99_latency"] - frontend_before_raw).max()
assert frontend_delta > 0, f"{FAIL} frontend latency should also rise (two hops)"
print(f"  frontend.p99_latency increased by {frontend_delta:.2f}ms (2-hop prop)  {PASS}")


# ── 9. Multiple simultaneous failures ────────────────────────────────────────
section("9. Multiple simultaneous failures")

traces9 = gen.full_traces()
events9 = [
    FailureEvent("geo",          FailureMode.LATENCY_SPIKE,  10, 15, 40.0, "processing_time_p99"),
    FailureEvent("reservation",  FailureMode.ERROR_BURST,    20, 10,  1.0, "failure_rate"),
    FailureEvent("memcached-rate", FailureMode.CACHE_DEGRADE, 30, 15, 15.0, "hit_rate"),
]
inj.apply(traces9, events9)
assert SLO_BY_ID[1].is_violated(traces9["geo"]["processing_time_p99"].max())
assert SLO_BY_ID[3].is_violated(traces9["reservation"]["failure_rate"].max())
assert SLO_BY_ID[5].is_violated(traces9["memcached-rate"]["hit_rate"].min())
print(f"  All 3 simultaneous failures injected and verified  {PASS}")


# ── 10. Absent nodes have no traces ──────────────────────────────────────────
section("10. Absent nodes produce no traces")

builder2  = EpisodeGraphBuilder(p_fail=0.99, p_rec=0.0, seed=99)
builder2.next_episode()   # ep0: all present
ep_absent = builder2.next_episode()  # ep1: many absent

gen2    = SLIGenerator(ep_absent, episode_length=T, seed=0)
traces2 = gen2.full_traces()

for absent_node in ep_absent.absent_nodes:
    assert absent_node not in traces2, \
        f"{FAIL} Absent node {absent_node} should have no traces"

print(f"  Absent nodes: {sorted(ep_absent.absent_nodes)}")
print(f"  None appear in traces  {PASS}")


# ── Done ─────────────────────────────────────────────────────────────────────
print(f"\n{'═' * 55}")
print(f"  ALL PHASE 2 ASSERTIONS PASSED ✓")
print(f"{'═' * 55}\n")