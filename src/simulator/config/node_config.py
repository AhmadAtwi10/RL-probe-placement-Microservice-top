"""
node_config.py
--------------
Static metadata for every node in the Hotel Reservation topology (DeathStarBench).

Each NodeMeta entry defines:
  - node_type   : service role (used for one-hot encoding in GNN features)
  - core        : True = never removed across episodes; False = subject to Bernoulli perturbation
  - probeable   : True = agent can place/remove a probe; False = infrastructure node (consul, jaeger)
  - slis        : list of raw SLI metric names the node can emit when probed
                  (these are the capability catalogue; the agent observes the SLO-indexed slice)

Node types used:
  gateway | business_logic | search_service | data_store | cache | registry | tracing

Core vs. non-core split (from §2.1 / §7.5 of probe_placement_formulation.md):
  Core  (10): frontend, search, geo, rate, profile, recommendation, reservation, user, consul, jaeger
  Non-core (14): review, attractions + all 12 mongodb-*/memcached-* instances
"""

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Node type constants (used for one-hot encoding)
# ---------------------------------------------------------------------------
NODE_TYPES = [
    "gateway",
    "business_logic",
    "search_service",
    "data_store",
    "cache",
    "registry",
    "tracing",
]

NODE_TYPE_INDEX = {t: i for i, t in enumerate(NODE_TYPES)}


# ---------------------------------------------------------------------------
# NodeMeta dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NodeMeta:
    node_type: str          # one of NODE_TYPES
    core: bool              # True  = never removed across episodes
    probeable: bool         # False = consul / jaeger (infrastructure)
    slis: List[str] = field(default_factory=list) # default factory creates for every object a new list,..

    def type_one_hot(self) -> List[int]:
        """Returns a one-hot vector over NODE_TYPES."""
        vec = [0] * len(NODE_TYPES)
        idx = NODE_TYPE_INDEX.get(self.node_type)
        if idx is not None:
            vec[idx] = 1
        return vec


# ---------------------------------------------------------------------------
# Node catalogue  (24 nodes total)
# ---------------------------------------------------------------------------
NODE_CATALOG: dict[str, NodeMeta] = {

    # ── Layer 0: Gateway (1 node, core) ─────────────────────────────────────
    "frontend": NodeMeta(
        node_type="gateway",
        core=True,
        probeable=True,
        slis=[
            "request_rate",
            "p99_latency",
            "p50_latency",
            "error_rate",
            "active_connections",
            "incoming_call_rate",
            "outgoing_call_rate",
        ],
    ),

    # ── Layer 1: Business logic — core (7 nodes) ────────────────────────────
    "search": NodeMeta(
        node_type="search_service",
        core=True,
        probeable=True,
        slis=[
            "search_latency_p99",
            "search_latency_p50",
            "result_count",
            "incoming_call_rate",
            "outgoing_call_rate",
        ],
    ),
    "geo": NodeMeta(
        node_type="business_logic",
        core=True,
        probeable=True,
        slis=[
            "processing_time_p99",
            "failure_rate",
            "cpu_utilization",
            "incoming_call_rate",
            "outgoing_call_rate",
        ],
    ),
    "rate": NodeMeta(
        node_type="business_logic",
        core=True,
        probeable=True,
        slis=[
            "processing_time_p99",
            "failure_rate",
            "request_queue_depth",
            "incoming_call_rate",
            "outgoing_call_rate",
        ],
    ),
    "profile": NodeMeta(
        node_type="business_logic",
        core=True,
        probeable=True,
        slis=[
            "processing_time_p99",
            "failure_rate",
            "cpu_utilization",
            "incoming_call_rate",
            "outgoing_call_rate",
        ],
    ),
    "recommendation": NodeMeta(
        node_type="business_logic",
        core=True,
        probeable=True,
        slis=[
            "recommendation_latency_p99",
            "failure_rate",
            "incoming_call_rate",
            "outgoing_call_rate",
        ],
    ),
    "reservation": NodeMeta(
        node_type="business_logic",
        core=True,
        probeable=True,
        slis=[
            "processing_time_p99",
            "failure_rate",
            "request_queue_depth",
            "incoming_call_rate",
            "outgoing_call_rate",
        ],
    ),
    "user": NodeMeta(
        node_type="business_logic",
        core=True,
        probeable=True,
        slis=[
            "processing_time_p99",
            "failure_rate",
            "cpu_utilization",
            "incoming_call_rate",
            "outgoing_call_rate",
        ],
    ),

    # ── Layer 1: Business logic — non-core (2 nodes, perturbable) ───────────
    "review": NodeMeta(
        node_type="business_logic",
        core=False,
        probeable=True,
        slis=[
            "processing_time_p99",
            "failure_rate",
            "cpu_utilization",
            "incoming_call_rate",
            "outgoing_call_rate",
        ],
    ),
    "attractions": NodeMeta(
        node_type="business_logic",
        core=False,
        probeable=True,
        slis=[
            "processing_time_p99",
            "failure_rate",
            "cpu_utilization",
            "incoming_call_rate",
            "outgoing_call_rate",
        ],
    ),

    # ── Layer 2a: MongoDB data stores — non-core (8 nodes, all perturbable) ─
    "mongodb-geo": NodeMeta(
        node_type="data_store", 
        core=False, 
        probeable=True,
        slis=[
            "query_latency_p99",
            "write_latency_p99",
            "connection_pool_utilization",
            "disk_io_utilization"
        ],
    ),
    "mongodb-profile": NodeMeta(
        node_type="data_store",
        core=False, 
        probeable=True,
        slis=[
            "query_latency_p99", 
            "write_latency_p99", 
            "connection_pool_utilization", 
            "disk_io_utilization"
        ],
    ),
    "mongodb-rate": NodeMeta(
        node_type="data_store", 
        core=False, 
        probeable=True,
        slis=[
            "query_latency_p99", 
            "write_latency_p99", 
            "connection_pool_utilization", 
            "disk_io_utilization"
        ],
    ),
    "mongodb-recommendation": NodeMeta(
        node_type="data_store", 
        core=False, 
        probeable=True,
        slis=[
            "query_latency_p99", 
            "write_latency_p99", 
            "connection_pool_utilization", 
            "disk_io_utilization"
        ],
    ),
    "mongodb-reservation": NodeMeta(
        node_type="data_store", 
        core=False, 
        probeable=True,
        slis=[
            "query_latency_p99", 
            "write_latency_p99", 
            "connection_pool_utilization", 
            "disk_io_utilization"
        ],
    ),
    "mongodb-user": NodeMeta(
        node_type="data_store", 
        core=False, 
        probeable=True,
        slis=[
            "query_latency_p99", 
            "write_latency_p99", 
            "connection_pool_utilization", 
            "disk_io_utilization"
        ],
    ),
    "mongodb-review": NodeMeta(
        node_type="data_store", 
        core=False, 
        probeable=True,
        slis=[
            "query_latency_p99", 
            "write_latency_p99", 
            "connection_pool_utilization", 
            "disk_io_utilization"
        ],
    ),
    "mongodb-attractions": NodeMeta(
        node_type="data_store", 
        core=False, 
        probeable=True,
        slis=[
            "query_latency_p99", 
            "write_latency_p99", 
            "connection_pool_utilization", 
            "disk_io_utilization"
        ],
    ),

    # ── Layer 2b: Memcached caches — non-core (4 nodes, all perturbable) ────
    "memcached-rate": NodeMeta(
        node_type="cache", 
        core=False, 
        probeable=True,
        slis=[
            "hit_rate", 
            "miss_rate", 
            "eviction_rate", 
            "memory_utilization"
        ],
    ),
    "memcached-profile": NodeMeta(
        node_type="cache", 
        core=False, 
        probeable=True,
        slis=[
            "hit_rate", 
            "miss_rate", 
            "eviction_rate", 
            "memory_utilization"
        ],
    ),
    "memcached-reserve": NodeMeta(
        node_type="cache", 
        core=False, 
        probeable=True,
        slis=[
            "hit_rate", 
            "miss_rate", 
            "eviction_rate", 
            "memory_utilization"
        ],
    ),
    "memcached-review": NodeMeta(
        node_type="cache", 
        core=False, 
        probeable=True,
        slis=[
            "hit_rate", 
            "miss_rate", 
            "eviction_rate", 
            "memory_utilization"
        ],
    ),

    # ── Layer 3: Infrastructure — core, non-probeable (2 nodes) ─────────────
    "consul": NodeMeta(
        node_type="registry",
        core=True,
        probeable=False,          # ← agent cannot probe infrastructure nodes
        slis=["request_rate", "p99_latency", "error_rate"],
    ),
    "jaeger": NodeMeta(
        node_type="tracing",
        core=True,
        probeable=False,          # ← circular dependency: jaeger IS the trace collector
        slis=["trace_ingestion_rate", "trace_drop_rate"],
    ),
}


# ---------------------------------------------------------------------------
# Derived collections (convenient lookups)
# ---------------------------------------------------------------------------

# All 24 node names in a fixed canonical order
ALL_NODES: List[str] = list(NODE_CATALOG.keys())

# Core nodes (10): never removed from G^(e)
CORE_NODES: List[str] = [n for n, m in NODE_CATALOG.items() if m.core]

# Non-core nodes (14): subject to Bernoulli perturbation
NON_CORE_NODES: List[str] = [n for n, m in NODE_CATALOG.items() if not m.core]

# Probeable nodes (22): agent can add/remove probes
PROBEABLE_NODES: List[str] = [n for n, m in NODE_CATALOG.items() if m.probeable]

# Infrastructure nodes (2): consul, jaeger
INFRA_NODES: List[str] = [n for n, m in NODE_CATALOG.items() if not m.probeable]
