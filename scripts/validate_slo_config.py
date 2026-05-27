"""Quick validation script for slo_config.py"""
import sys
sys.path.insert(0, '.')

from src.simulator.config.slo_config import (
    SLO_CATALOG, SLO_BY_ID, NODE_TO_SLOS,
    NUM_SLOS, TOTAL_WEIGHT, APPLICATION_SLA_SLO_IDS,
    APPLICATION_SLA_NAME
)

print(f"SLA name       : {APPLICATION_SLA_NAME}")
print(f"Total SLOs     : {NUM_SLOS}")
print(f"SLA SLO ids    : {APPLICATION_SLA_SLO_IDS}")
print(f"Total weight   : {TOTAL_WEIGHT:.1f}")
print(f"Node → SLO map : {NODE_TO_SLOS}")

assert NUM_SLOS == 6, f"Expected 6 SLOs, got {NUM_SLOS}"
assert APPLICATION_SLA_SLO_IDS == (0, 1, 2, 3, 4, 5), "SLA must contain all 6 SLO ids"
assert abs(TOTAL_WEIGHT - 2.1) < 1e-9, f"Expected total weight 2.1, got {TOTAL_WEIGHT}"

# Confirm metric names now match the actual SLI keys emitted by candidate nodes
assert SLO_BY_ID[0].metric == "search_latency_p99",   "SLO_0 metric mismatch"
assert SLO_BY_ID[1].metric == "processing_time_p99",  "SLO_1 metric mismatch"
assert SLO_BY_ID[2].metric == "processing_time_p99",  "SLO_2 metric mismatch"
assert SLO_BY_ID[3].metric == "failure_rate",          "SLO_3 metric mismatch"
assert SLO_BY_ID[4].metric == "query_latency_p99",    "SLO_4 metric mismatch"
assert SLO_BY_ID[5].metric == "hit_rate",              "SLO_5 metric mismatch"

# Violation / margin checks for latency SLO
slo0 = SLO_BY_ID[0]   # search_latency_p99 <= 200ms
assert not slo0.is_violated(150.0)
assert not slo0.is_violated(200.0)
assert slo0.is_violated(250.0)
assert slo0.margin(150.0) > 0
assert slo0.margin(200.0) == 0
assert slo0.margin(250.0) < 0

# Violation / margin checks for hit-rate SLO
slo5 = SLO_BY_ID[5]   # memcached_rate_hit_rate >= 90%
assert not slo5.is_violated(95.0)
assert not slo5.is_violated(90.0)
assert slo5.is_violated(80.0)
assert slo5.margin(95.0) > 0
assert slo5.margin(90.0) == 0
assert slo5.margin(80.0) < 0

assert len(NODE_TO_SLOS["reservation"]) == 2, "reservation covers SLO_2 and SLO_3"

print("\nAll assertions passed ✓")