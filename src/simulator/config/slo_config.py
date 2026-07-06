"""
slo_config.py
-------------
Defines the SLO catalogue for the Hotel Reservation application (DeathStarBench).

The application is governed by a SINGLE SLA ("Hotel Reservation Service Agreement")
that bundles all 6 atomic SLOs covering latency, reliability, and data-store
performance. This follows the standard industry model (Google SRE Book, AWS
service agreements) where one application-level SLA contains all operating constraints.

Each SLO is an atomic triple (metric, threshold, operator) with an associated
importance weight and the single candidate node that provides it.

See §2.2 of probe_placement_formulation.md for the formal definition.
"""

from dataclasses import dataclass
from typing import Literal, List, Dict


# ---------------------------------------------------------------------------
# Operator type
# ---------------------------------------------------------------------------
Operator = Literal["<=", ">="]


# ---------------------------------------------------------------------------
# SLO dataclass  — one atomic constraint
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SLO:
    id: int             # unique index k ∈ 𝒦
    metric: str         # the SLI metric name this SLO measures
    threshold: float    # numeric bound
    op: Operator        # comparison direction: "<=" (metric must stay at or below) or ">=" (must stay at or above)
    weight: float       # importance weight w_k ∈ ℝ⁺  (used in reward computation)
    nodes: List[str]      # candidate_nodes(k) — all nodes that can cover this SLO
                          # (|nodes| = 1 for all current Hotel Reservation SLOs;
                          #  the formulation allows multiple candidates in general)

    def is_violated(self, value: float) -> bool:
        """Returns True if the metric value breaches the threshold."""
        if self.op == "<=":
            return value > self.threshold
        else:  # ">="
            return value < self.threshold

    def margin(self, value: float) -> float:
        """
        Returns normalised distance from threshold.
        Positive = safe, negative = violated.

        margin(k, t) = (threshold_k - value) / threshold_k   for op='<'
                     = (value - threshold_k) / threshold_k   for op='>'
        """
        if self.op == "<=":
            return (self.threshold - value) / self.threshold
        else:  # ">="
            return (value - self.threshold) / self.threshold


# ---------------------------------------------------------------------------
# SLO catalogue  (6 SLOs, all inside the single Hotel Reservation SLA)
# ---------------------------------------------------------------------------
SLO_CATALOG: List[SLO] = [
    SLO(
        id=0,
        metric="search_latency_p99",
        threshold=200.0,   # ms
        op="<=",
        weight= 1, #0.5 currently set to 1 to have all weights are equal,
        nodes=["search"],
    ),
    SLO(
        id=1,
        metric="processing_time_p99",
        threshold=50.0,    # ms
        op="<=",
        weight= 1,  #0.3,
        nodes=["geo"],
    ),
    SLO(
        id=2,
        metric="processing_time_p99",
        threshold=300.0,   # ms
        op="<=",
        weight= 1,  #0.4,
        nodes=["reservation"],
    ),
    SLO(
        id=3,
        metric="failure_rate",
        threshold=0.5,     # percent
        op="<=",
        weight= 1,  #0.5,
        nodes=["reservation"],
    ),
    SLO(
        id=4,
        metric="query_latency_p99",
        threshold=10.0,    # ms
        op="<=",
        weight= 1,  #0.2,
        nodes=["mongodb-rate"],
    ),
    SLO(
        id=5,
        metric="hit_rate",
        threshold=90.0,    # percent
        op=">=",
        weight= 1,  #0.2,
        nodes=["memcached-rate"],
    ),
]

# ---------------------------------------------------------------------------
# Derived lookups
# ---------------------------------------------------------------------------

# Single application-level SLA name
APPLICATION_SLA_NAME: str = "Hotel Reservation Service Agreement"

# Total number of SLOs  |𝒦|
NUM_SLOS: int = len(SLO_CATALOG)

# All SLO ids belonging to the single application SLA
APPLICATION_SLA_SLO_IDS: tuple = tuple(slo.id for slo in SLO_CATALOG)

# SLO by id
SLO_BY_ID: Dict[int, SLO] = {slo.id: slo for slo in SLO_CATALOG}

# Node → list of SLO ids it covers  (candidate_nodes mapping, inverted)
NODE_TO_SLOS: Dict[str, List[int]] = {}
for slo in SLO_CATALOG:
    for node in slo.nodes:
        NODE_TO_SLOS.setdefault(node, []).append(slo.id)

# Total weight budget (useful for normalising reward)
TOTAL_WEIGHT: float = sum(slo.weight for slo in SLO_CATALOG)