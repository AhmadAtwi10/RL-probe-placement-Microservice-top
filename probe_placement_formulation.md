# RL-Based Probe Placement for SLA Verification in Microservice Architectures

## Formal Problem Formulation

---

## 1. Background and Motivation

In microservice architectures, ensuring service quality requires monitoring distributed components against defined contracts. This work formalizes the problem of **optimal probe placement** — determining which nodes to instrument in a service graph so that all Service Level Agreement (SLA) violations can be **detected before or as they occur**.

The three key concepts that underpin this formulation are:

- **SLI (Service Level Indicator):** A quantitative metric collected from a service node (e.g., `payment_success_rate`, `p99_latency`).
- **SLO (Service Level Objective):** A target threshold that an SLI must meet (e.g., `payment_success_rate > 99.95%`).
- **SLA (Service Level Agreement):** A formal contract binding the system to meet one or more SLOs.

The relationship between them is:

```
SLA (contract)    "E-commerce checkout must be reliable and fast"
  ├── SLO_1 (target)    "Payment success rate > 99.95% measured hourly"
  │     ↓
  │   SLI_1 (metric)    "payment_success_count / payment_total_count"
  │     ↑
  │   Probe             placed on Payment Service
  │
  └── SLO_2 (target)    "End-to-end latency < 500ms"
        ↓
      SLI_2 (metric)    "response_time_p99"
        ↑
      Probe             placed on API Gateway
```

In practice, a single SLA contract bundles **multiple SLOs**, each tracking a distinct SLI. An SLA is violated if **any** of its SLOs is breached. Our formulation therefore indexes at the **SLO level** — the atomic unit of a measurable constraint — and groups SLOs into logical SLAs.

A probe placed on a node is the mechanism by which SLIs are collected. Without a probe on a node, the SLIs it exposes cannot be observed, and any SLA that depends on those SLIs cannot be verified. A probe must be in place **before** a violation occurs — an agent that places a probe at time `t` while a violation first occurs also at time `t` is too late if it was not already monitoring that node in the preceding timesteps.

---

## 2. System Model

### 2.1 Service Graph

The microservice application is modeled as a directed graph. The graph is **static within each episode** but may change between episodes, reflecting infrequent real-world operational events such as circuit breaker activations, pod restarts, or sidecar proxy additions:

```
G^(e) = (V^(e), E^(e))   — graph for episode e, fixed throughout episode e
```

where:
- `V^(e)` is the set of service nodes for episode `e`
- `E^(e)` is the set of directed edges representing inter-service gRPC/HTTP call dependencies

**Concrete instantiation:** This formulation is grounded in the **Hotel Reservation** application from [DeathStarBench](https://github.com/delimitrou/DeathStarBench) — a realistic open-source microservice benchmark. The full topology is derived directly from the application's `docker-compose.yml` and consists of **24 nodes** across four layers:

```
Layer 0 — Gateway (1 node):
  frontend

Layer 1 — Business logic (9 nodes):
  search, geo, rate, profile, recommendation, reservation, user, review, attractions

Layer 2 — Data stores (13 nodes):
  mongodb-geo, mongodb-profile, mongodb-rate, mongodb-recommendation,
  mongodb-reservation, mongodb-user, mongodb-review, mongodb-attractions   (8 × MongoDB)
  memcached-rate, memcached-profile, memcached-reserve, memcached-review   (4 × Memcached)

Layer 3 — Infrastructure (2 nodes):
  consul (service registry), jaeger (distributed tracing)
```

All inter-service calls between business-logic services are gRPC-based. Each business-logic service registers with and resolves peers through `consul`. Distributed traces are emitted to `jaeger` by all services via the `JAEGER_SAMPLE_RATIO` environment variable.

The **core call graph** (edges between business-logic services, derived from source code) is:
```
frontend   → search, profile, recommendation, reservation, user, review, attractions
search     → geo, rate
```

The **infrastructure edges** (from `depends_on` in docker-compose) are:
```
geo          → mongodb-geo
rate         → mongodb-rate, memcached-rate
profile      → mongodb-profile, memcached-profile
recommendation → mongodb-recommendation
reservation  → mongodb-reservation, memcached-reserve
user         → mongodb-user
review       → mongodb-review, memcached-review
attractions  → mongodb-attractions
all services → consul
```

Note that `review` and `attractions` use custom service images extending the hotel experience with user reviews and nearby attractions discovery. Both are called by `frontend` via gRPC and are part of the main call graph. They are classified as **non-core** in the operational sense: they are subject to perturbation across episodes (pod failure, rolling restart), unlike the eight stable backbone services.

At the start of a new episode, a graph is derived from this base topology via a lightweight **operational perturbation**:

```
G^(e+1) ~ 𝒢   — independent Bernoulli perturbation over 14 non-core nodes:

  Non-core business-logic (2):  review, attractions
  Non-core data stores    (12):  all mongodb-*, all memcached-*

  For each non-core node v:
    with prob p_fail: v is absent from G^(e+1)  (pod failure, eviction)
    with prob p_rec:  a previously absent v returns  (pod restart, recovery)
```

The **8 backbone services** (`frontend`, `search`, `geo`, `rate`, `profile`, `recommendation`, `reservation`, `user`) and infrastructure nodes (`consul`, `jaeger`) are **never removed** — they form the stable core of the application. Data-store instances may fail independently of their parent business-logic service (a MongoDB replica crash while the application service keeps running), which is the dominant failure mode in production Kubernetes deployments. This gives a maximum of `2^14 = 16,384` distinct episode graph configurations, providing sufficient structural diversity for GNN generalization training. Within a single episode, the agent operates on a fully known, fixed graph. Across episodes, the GNN encoder recomputes node embeddings from the updated graph, enabling the agent to **generalize across operational configurations without retraining**.

Each node `v ∈ V^(e)` exposes a set of SLIs when probed:

```
SLI(v) = { m₁, m₂, ..., mₙ }   — the metrics node v can emit if instrumented
```

SLIs fall into two semantic categories, both attributed to nodes (consistent with how real APM tools such as Prometheus and Jaeger report metrics):

- **Node-intrinsic SLIs** — properties of the service itself, independent of who calls it: `cpu_utilization`, `memory_utilization`, `request_queue_depth`, `disk_io_utilization`, `connection_pool_utilization`.
- **Call-derived SLIs** — observed on the communication channel but reported by the receiving service: `p99_latency`, `error_rate`, `failure_rate`. These are attributed to the node that receives the call, which is how all standard APM tooling (Jaeger spans, Prometheus `http_request_duration_seconds`) records them.

Two additional SLIs are derived from **runtime call graph traces** (e.g., Jaeger) and incorporated as node-level features to capture which services are currently load-bearing:

```
incoming_call_rate(v, t)  — number of inbound gRPC/HTTP calls per second at node v
outgoing_call_rate(v, t)  — number of outbound calls per second initiated by node v
```

These call-rate features allow the GNN to distinguish structurally important nodes (high degree in topology) from currently hot nodes (high call rate at runtime), without requiring a separate call-graph architecture. The probe placement decision remains node-centric — a probe is placed on a node, not on an edge — so this representation is fully consistent with the action space.

**Concrete SLI mapping for the Hotel Reservation application (DeathStarBench):**

| Node             | Type           | Core | SLI(v)                                                                              |
|------------------|----------------|------|-------------------------------------------------------------------------------------|
| `frontend`       | gateway        | yes  | request_rate, p99_latency, p50_latency, error_rate, active_connections, incoming_call_rate, outgoing_call_rate |
| `search`         | search         | yes  | search_latency_p99, search_latency_p50, result_count, incoming_call_rate, outgoing_call_rate |
| `geo`            | business_logic | yes  | processing_time_p99, failure_rate, cpu_utilization, incoming_call_rate, outgoing_call_rate |
| `rate`           | business_logic | yes  | processing_time_p99, failure_rate, request_queue_depth, incoming_call_rate, outgoing_call_rate |
| `profile`        | business_logic | yes  | processing_time_p99, failure_rate, cpu_utilization, incoming_call_rate, outgoing_call_rate |
| `recommendation` | business_logic | yes  | recommendation_latency_p99, failure_rate, incoming_call_rate, outgoing_call_rate   |
| `reservation`    | business_logic | yes  | processing_time_p99, failure_rate, request_queue_depth, incoming_call_rate, outgoing_call_rate |
| `user`           | business_logic | yes  | processing_time_p99, failure_rate, cpu_utilization, incoming_call_rate, outgoing_call_rate |
| `review`         | business_logic | no   | processing_time_p99, failure_rate, cpu_utilization, incoming_call_rate, outgoing_call_rate |
| `attractions`    | business_logic | no   | processing_time_p99, failure_rate, cpu_utilization, incoming_call_rate, outgoing_call_rate |
| `mongodb-*`      | data_store     | no   | query_latency_p99, write_latency_p99, connection_pool_utilization, disk_io_utilization |
| `memcached-*`    | cache          | no   | hit_rate, miss_rate, eviction_rate, memory_utilization                              |
| `consul`         | registry       | yes  | request_rate, p99_latency, error_rate                                               |
| `jaeger`         | tracing        | yes  | trace_ingestion_rate, trace_drop_rate                                               |

> **Core vs. non-core:** The 8 backbone business-logic services (`frontend` through `user`) and infrastructure nodes (`consul`, `jaeger`) are core — never removed across episodes. All 12 data-store instances (`mongodb-*`, `memcached-*`) and the 2 extended services (`review`, `attractions`) are non-core and subject to independent Bernoulli perturbation at each episode boundary, giving up to `2^14 = 16,384` distinct episode graph configurations.

> **Infrastructure nodes:** `consul` and `jaeger` remain in `G^(e)` as graph nodes — their structural position (all services depend on consul; all services emit traces to jaeger) informs GNN message passing and captures real hub-and-spoke patterns in the topology. However, both are marked `is_probeable(v) = 0`: the agent cannot place or remove probes on them, and no SLOs are defined over their internal metrics. Monitoring infrastructure-level tools at the application level would create a circular dependency.

### 2.2 SLO and SLA Specification

**SLO (atomic constraint):** Each SLO is defined as a triple:

```
SLO_k = (metric_k, threshold_k, operator_k)
```

where:
- `metric_k` is the SLI the SLO is measured against
- `threshold_k` is the bound that must not be violated
- `operator_k ∈ { ≤, ≥ }` is the comparison direction

The importance weight of each SLO is defined **separately** as an optimization parameter — it belongs to the RL problem, not to the SLO contract:

```
W = { w_k ∈ ℝ⁺ | k ∈ 𝒦 }   — importance weights assigned by the system operator
```

**SLA:** The application is governed by a **single** SLA — an operator-defined contract that bundles all `|𝒦|` SLOs. The SLA is satisfied at time `t` if and only if **all** constituent SLOs are simultaneously satisfied:

```
satisfied(t) = 1   iff   ∀ k ∈ 𝒦,  covered(k, t) = 1  AND  ¬violation(k, t)
```

This follows the standard industry and research model (Google SRE Book; AWS service agreements; Gan et al. DeathStarBench; Ma et al. AutoMap) where a single application-level SLA defines acceptable operating conditions across latency, reliability, and resource dimensions.

**Example — Hotel Reservation SLA (DeathStarBench):**

```
SLA = "Hotel Reservation Service Agreement"
  ├── SLO_0 = (search_latency_p99,               200ms,  <)    w₀ = 0.5
  ├── SLO_1 = (geo_processing_time_p99,            50ms,  <)    w₁ = 0.3
  ├── SLO_2 = (reservation_processing_time_p99,   300ms,  <)    w₂ = 0.4
  ├── SLO_3 = (reservation_failure_rate,           0.5%,  <)    w₃ = 0.5
  ├── SLO_4 = (mongodb_rate_query_latency_p99,     10ms,  <)    w₄ = 0.2
  └── SLO_5 = (memcached_rate_hit_rate,             90%,   >)    w₅ = 0.2
```

Note that `SLO_0` and `SLO_1` are causally linked: a degraded `geo` service causes `search_latency_p99` to spike via the synchronous `search → geo` gRPC dependency. The agent must monitor **both** nodes to distinguish root cause from symptom — a key probe placement challenge this formulation addresses.

> **Note:** The probe placement problem operates at the **SLO level** — each SLO maps to one metric and requires observing a specific node. The SLA is the terminal criterion: if all SLOs are covered and violation-free at episode end, a terminal bonus is applied. Throughout the formulation, index `k` refers to an individual SLO.

### 2.3 SLI-to-Node Mapping

For each SLO `k`, the set of candidate nodes — those capable of providing the required SLI — is:

```
candidate_nodes(k) = { v ∈ V^(e) | metric_k ∈ SLI(v) }
```

This mapping is a **static input** to the system, defined by service metadata (e.g., OpenTelemetry instrumentation declarations). The agent does not learn or select which SLIs to collect — it only decides where to place probes.

### 2.4 Coverage and Violation (Separated Concepts)

Two distinct concepts must be kept separate:

**Coverage (observability):** Can the agent observe SLO `k`'s metric at time `t`?

```
covered(k, t) = 1   iff   ∃ v ∈ V^(e) : P_t[v] = 1  AND  metric_k ∈ SLI(v)
```

Coverage is purely about whether a probe is in place — it says nothing about the metric's current value. The probe placement vector `P_t` is implicit: `covered(k, t)` means *covered under the placement active at time t*. In the static optimization context of Section 3, we write `covered(k, S)` where `S ⊆ V^(e)` is the candidate probe set variable.

**Violation (threshold breach):** Is SLO `k`'s threshold breached at time `t`?

```
violation(k, t) = 1   iff   SLI_k(t)  violates  (threshold_k, operator_k)
```

A violation can occur whether or not a probe is in place — the difference is whether the agent can see it.

**Blind violation (the dangerous case):** A violation that the agent cannot detect:

```
blind_violation(k, t) = 1   iff   violation(k, t) = 1  AND  covered(k, t) = 0
```

The agent's primary goal is to eliminate blind violations by ensuring probes are placed proactively.

### 2.5 SLO Margin (Risk Signal)

For each SLO `k` with a currently probed node, the **normalized margin** from the threshold measures how close the system is to a violation:

```
margin(k, t) = (threshold_k − SLI_k(t)) / |threshold_k|    [for operator_k = <]
margin(k, t) = (SLI_k(t) − threshold_k) / |threshold_k|    [for operator_k = >]
```

| `margin(k, t)` | Interpretation |
|----------------|---------------|
| `≈ 1`          | Safe — far from threshold |
| `≈ 0`          | Danger zone — approaching threshold |
| `< 0`          | SLO violated |

The margin is **only observable for probed nodes**. For unprobed nodes, `margin(k, t)` is unknown — this is the core source of partial observability in the problem.

---

## 3. Formal Optimization Problem

### 3.1 Static Idealized Formulation (Theoretical Benchmark)

When the graph is static and the state is fully observable, the probe placement problem reduces to an instance of **weighted set cover**:

**Decision variable:** Let `S ⊆ V^(e)` be the set of probed nodes for the current episode `e`.

**Objective:** Find the minimum probe set `S*` that covers all SLOs:

```
S* = argmin |S|
     subject to:  ∀ k ∈ 𝒦,  covered(k, S) = 1
```

This is NP-hard in general and serves as a **lower bound** on probe count. It is solvable by exact methods (Integer Linear Programming, greedy set cover) under full observability.

### 3.2 Multi-SLO Extension (Weighted Coverage)

When SLOs have different importances, the problem becomes a weighted coverage problem. For each SLO `k` with importance weight `w_k`:

```
S* = argmin |S|
     subject to:  Σ_k  w_k · covered(k, S)  ≥  C_min
```

where `C_min` is a minimum total weighted coverage threshold (e.g., all critical SLOs must be covered).

> **Note:** This static formulation assumes a fixed workload and full observability. In practice, workloads are dynamic and nodes emit no signals unless probed. The POMDP in Section 4 addresses the realistic problem; the static formulation provides a useful baseline for evaluation.

---

## 4. POMDP Formulation

The probe placement problem under dynamic workloads and partial observability is modeled as a **Partially Observable Markov Decision Process (POMDP)**:

```
POMDP = (S, A, O, T, Ω, R, γ)
```

The distinction from a plain MDP is critical: the agent cannot observe the full system state. Any node without a probe emits no signals — its SLI values, trends, and current margin are completely hidden from the agent.

### 4.1 True State Space `S`

The true system state at timestep `t` within episode `e` is:

```
s_t = (G^(e), X_t, P_t)
```

where:
- `G^(e)` — service graph for episode `e`, fixed throughout the episode (known from service discovery)
- `X_t` — true SLI values for **all** nodes `v ∈ V^(e)` (including unprobed — completely hidden from agent)
- `P_t` — probe placement vector: `P_t[v] = 1` if node `v` is probed, `0` otherwise

The agent never directly observes `s_t`. In particular, `X_t(v)` for any unprobed node `v` (where `P_t[v] = 0`) is fully hidden.

### 4.2 Observation Space `O`

At each timestep `t`, the agent receives an observation `o_t ∈ O`:

```
o_t = (G^(e), P_t, ô_t, H_t)
```

where:
- `G^(e)` — episode graph, fully known from service discovery; fixed within the episode
- `P_t` — current probe placement vector (the agent knows where its probes are)
- `ô_t` — **masked observation vector**: for each node `v` and each SLO `k` with `metric_k ∈ SLI(v)`:

```
ô_t(v, k) = [ SLI_k(t−N+1), ..., SLI_k(t), margin(k, t), Δmargin(k, t) ]   if P_t[v] = 1
           = [ masked (0 or ∅) ]                                               if P_t[v] = 0
```

  The observation includes a **rolling window of N timesteps** and the current margin and its rate of change (`Δmargin = margin(k,t) − margin(k,t−1)`) for each monitored SLO metric. This enables the agent to detect deteriorating trends before a threshold is breached.

- `H_t` — SLO health vector: `H_t[k] = 1` if SLO `k` is currently covered **and** not violated, `0` otherwise. If a node is unprobed, its SLO entries are unknown and treated as `0`.

#### 4.2.1 Node Feature Vector

Before the GNN runs, each node `v ∈ V^(e)` is described by a raw feature vector `x_v` composed of two groups:

**Static features** — derived from `G^(e)` and service metadata; computed **once at the start of each episode** and fixed for its entire duration:

```
x_v^static = [
  node_type(v)          : one-hot encoding of service role (e.g. gateway, data_store, cache)
  is_probeable(v)       : 1 if the agent can place/remove a probe on v; 0 for infrastructure nodes (consul, jaeger)
  degree_in(v)          : number of incoming dependency edges in G^(e)
  degree_out(v)         : number of outgoing dependency edges in G^(e)
  slo_coverage_mask(v)  : binary vector over 𝒦 — which SLOs node v can cover
  slo_importance(v)     : Σ_{k: metric_k ∈ SLI(v)} w_k  — total SLO weight v carries
]
```

**Dynamic features** — updated at **every timestep** `t` as the agent acts and the workload evolves:

```
x_v^dynamic(t) = [
  P_t[v]                    : 1 if v is currently probed, 0 otherwise
  ô_t(v)                    : SLI window + margin(k,t) + Δmargin(k,t) per covered SLO k
                              (all zeros if P_t[v] = 0)
  incoming_call_rate(v, t)  : inbound gRPC/HTTP calls per second — derived from call graph traces
  outgoing_call_rate(v, t)  : outbound calls per second initiated by v — derived from call graph traces
]
```

> **On observation size:** Although `SLI(v)` may differ in size across node types (e.g., `frontend` exposes 7 raw metrics while `mongodb-*` exposes 4), `ô_t(v)` is **indexed over the fixed SLO set `𝒦`**, not over raw SLI names. Formally:
>
> ```
> ô_t(v) = [ ô_t(v, SLO_1) | ô_t(v, SLO_2) | ... | ô_t(v, SLO_|𝒦|) ]
>            ← always of size  |𝒦| × (N + 2),  regardless of |SLI(v)|
> ```
>
> If node `v` does not cover SLO `k` (i.e., `metric_k ∉ SLI(v)`), then `ô_t(v, k) = 0` always. The raw SLI diversity documented in §2.1 is a capability catalogue — only SLO-relevant metrics appear in the agent's observation. This keeps the GNN input **fixed-size per node** across all node types.

The two call-rate features are **always observable** regardless of probe placement — they are derived from infrastructure-level call graph traces (e.g., Jaeger service maps, Envoy access logs) rather than from application-level probes. This allows the agent to identify currently hot nodes even without a probe placed on them, providing indirect evidence about load distribution across the graph.

> **Note on call graph incorporation:** Rather than maintaining a separate call-graph architecture, call graph information is folded into node features via `incoming_call_rate` and `outgoing_call_rate`. This lets the GNN distinguish structurally central nodes (high `degree_in`) from currently load-bearing nodes (high `incoming_call_rate`) — a distinction the topology graph alone cannot express.


The full per-node input to the GNN at time `t` is:

```
x_v(t) = [ x_v^static  |  x_v^dynamic(t) ]
```

> **Cross-episode note:** Since `G^(e+1)` may differ from `G^(e)`, all static features are recomputed from scratch at each new episode. A new service node joining the graph in episode `e+1` gets its own fresh `x_v^static` from the updated topology. The GNN weights are shared across episodes — this is what enables **zero-shot generalization** to unseen topologies.

**Encoding pipeline:**

```
Step 1 — Build node feature vectors (each timestep t):
  x_v(t) = [ x_v^static | x_v^dynamic(t) ]   ∀ v ∈ V^(e)

Step 2 — GNN message passing (L layers):
  h_v^(0) = x_v(t)
  h_v^(l) = UPDATE( h_v^(l-1),  AGGREGATE({ h_u^(l-1) | u ∈ N(v) }) )
  h_v     = h_v^(L)     ← final node embedding enriched by L-hop neighborhood

Step 3 — Policy input:
  graph_embedding = READOUT({ h_v | v ∈ V^(e) })
  policy_input    = concat(graph_embedding, H_t)
                  → fed into π(a | o_t)
```

A **GNN** is used because: (i) it handles variable-size graphs across episodes without architecture changes; (ii) message passing propagates stress signals from probed nodes to their unprobed neighbors, giving the agent indirect evidence about unmonitored parts of the graph; (iii) structural features like `degree_in` and `slo_coverage_mask` are naturally encoded per-node without any global index assumptions.

### 4.3 Action Space `A`

At each timestep, the agent selects one action:

```
A = { add_probe(v)    | v ∈ V^(e),  is_probeable(v) = 1,  P_t[v] = 0 }
  ∪ { remove_probe(v) | v ∈ V^(e),  is_probeable(v) = 1,  P_t[v] = 1 }
  ∪ { no_op }
```

Let `V_p^(e) = { v ∈ V^(e) | is_probeable(v) = 1 }` denote the set of probeable nodes. This is a **discrete action space** of size `2|V_p^(e)| + 1`. Infrastructure nodes (`consul`, `jaeger`) are excluded: they appear in the graph for structural context but the agent has no probe actions over them.

- `add_probe(v)`: sets `P_{t+1}[v] = 1`; node `v`'s SLIs become observable from `t+1` onward.
- `remove_probe(v)`: sets `P_{t+1}[v] = 0`; node `v`'s SLIs become hidden from `t+1` onward.
- `no_op`: probe placement is unchanged (the agent is satisfied with the current configuration).

The `no_op` action is important for a continuously running agent — most timesteps, the optimal decision is to maintain the current placement.

### 4.4 Transition Function `T`

The environment transitions are stochastic due to workload dynamics:

```
s_{t+1} ~ T(s_t, a_t)
```

- The probe placement part of the state transitions deterministically (the action directly updates `P_t`).
- The SLI values `X_{t+1}` are drawn from a workload distribution that may include traffic spikes, failure injections, and diurnal patterns. In training, this is modeled by a **workload simulator**.

### 4.5 Observation Function `Ω`

```
o_t ~ Ω(s_t, P_t)
```

Formally: for each node `v`, if `P_t[v] = 1`, the true SLI value `X_t(v)` is revealed; otherwise it is masked. The observation function is deterministic given the true state.

### 4.6 Reward Function `R`

The reward at each timestep balances **observability**, **overhead**, **undetected violations**, and **risky probe removal**:

```
R_t =   Σ_k  w_k · covered(k, t)               ← (1) observability reward
      − λ  · |P_t|                               ← (2) probe overhead penalty
      − μ  · Σ_k  blind_violation(k, t)          ← (3) blind violation penalty
      − ρ  · removal_risk(a_t, t)                ← (4) risk-weighted removal penalty
```

**Term (1) — Observability reward:**  
The agent is rewarded for each SLO it can currently observe, weighted by that SLO's importance weight `w_k`. This incentivizes placing probes on nodes that cover high-priority SLOs.

**Term (2) — Probe overhead penalty:**  
Each active probe consumes monitoring resources. The penalty scales with the number of active probes, encouraging the agent to minimize instrumentation cost.

**Term (3) — Blind violation penalty:**  
The heaviest penalty. Triggered when an SLO threshold is breached on a node with no probe — the agent is unaware of the violation. This is the primary failure mode the agent must avoid.

**Term (4) — Risk-weighted removal penalty:**  
Applied at the moment the agent takes action `remove_probe(v)`. It penalizes removals from nodes whose metrics are close to breaching an SLO threshold:

```
removal_risk(v, t) = Σ_{k: metric_k ∈ SLI(v)}  w_k · max(0, 1 − margin(k, t))
```

This is non-zero only when `a_t = remove_probe(v)`, and is zero for `add_probe` and `no_op` actions. It teaches the agent to be cautious about removing probes from risky nodes, even before a violation occurs.

**Hyperparameter trade-offs:**

| Parameter | Role | Effect if increased |
|-----------|------|---------------------|
| `w_k`     | SLO importance weight | Agent prioritizes covering SLO `k` |
| `λ`       | Overhead sensitivity | Agent uses fewer probes overall |
| `μ`       | Blind violation cost | Agent strongly avoids being unprobed during SLO violations |
| `ρ`       | Removal risk sensitivity | Agent hesitates to remove probes from near-threshold nodes |

**Temporal interplay between `ρ` and `μ`:**
- `ρ` fires **immediately at the moment of removal** — provides early shaping signal when the last known margin is small.
- `μ` fires **when a violation occurs** without a probe in place — may be delayed by many timesteps after the bad removal decision.

Together, they close the temporal credit assignment gap: `ρ` makes the agent cautious at removal time; `μ` enforces the ultimate cost of having been blind.

### 4.7 Episode Structure and Termination

The agent operates continuously over a workload window of up to `T` timesteps. An episode terminates under two conditions:

```
t* = min( T,  min{ t : Σ_{τ ≤ t} Σ_k blind_violation(k, τ) > K } )
```

- **Normal termination** at `t = T`: the agent survived the full workload window without exceeding the blind violation budget. A terminal bonus may be awarded.
- **Failure termination** when cumulative blind violations exceed `K`: the SLA contract is considered definitively breached — too many SLO violations went undetected. A large terminal penalty is applied.

`K` is a **safety budget** representing the maximum tolerated number of undetected SLO violations per episode. It is a curriculum parameter: training begins with a generous `K` and reduces it as the agent improves.

### 4.8 Optimization Objective

The POMDP policy objective is:

```
π* = argmax E[ Σ_{t=0}^{t*} γ^t · R_t ]
     subject to:  E[ Σ_t Σ_k blind_violation(k, t) ] ≤ K
```

where `γ ∈ (0, 1]` is the discount factor. This objective corresponds to the **dynamic, online generalization** of the static set-cover formulation in Section 3:

- As workload stabilizes and the agent converges, its probe placement approaches the static optimal `S*` from Section 3.1.
- The `w_k` weights in the reward correspond directly to the importance weights in the weighted coverage objective of Section 3.2.
- The static formulation in Section 3 serves as a **lower bound** on probe count and a benchmark for evaluating the agent's efficiency.

### 4.9 Algorithm

**Proximal Policy Optimization (PPO)** is the recommended RL algorithm for this problem because:
- It handles discrete, combinatorial action spaces robustly
- It is more stable than DQN for large graphs
- It directly learns a stochastic policy `π(a | o_t)` that can express uncertainty over actions
- It supports the continuous online operation structure of this formulation

**DQN** is a simpler alternative suitable for small graphs where `|V|` is small and the action space is manageable.

---

## 5. End-to-End Workflow

The full pipeline from SLA definition to probe placement operates as follows:

### Step 1 — SLA Definition
**Input:** Human-readable contracts defined by the SRE team.  
**Output:** Formal triples `SLO_k = (metric_k, threshold_k, operator_k)` and importance weights `w_k`.  
**How:** Manual input. Not automated or learned by the agent.

### Step 2 — SLI Mapping
**Input:** Service graph `G`, service metadata, SLA definitions.  
**Output:** `SLI(v)` for each node; `candidate_nodes(k)` for each SLO.  
**How:** Static lookup table built from OpenTelemetry or Prometheus exporter declarations.

### Step 3 — Observation Construction
**Input:** Current episode graph `G^(e)`, probe vector `P_t`, raw SLI readings from probed nodes.  
**Output:** Masked observation `o_t` fed into the policy network.  
**How:** GNN encodes `G^(e)`; `P_t` and windowed SLI values (with margin and trend) are concatenated per node; `H_t` is appended at graph level.

### Step 4 — Agent Action (Probe Placement)
**Input:** Observation `o_t`.  
**Output:** Action `a_t ∈ A` (add probe, remove probe, or no-op).  
**How:** Policy network `π(a | o_t)` selects an action via PPO.

### Step 5 — Environment Transition
**Input:** Current state `s_t`, action `a_t`.  
**Output:** Next state `s_{t+1}`, new observation `o_{t+1}`.  
**How:** Probe placement updated deterministically; SLI values evolved by workload simulator.

### Step 6 — SLI Collection
**Input:** Updated probe placement `P_{t+1}`.  
**Output:** Observed SLI values for all probed nodes.  
**How:** Probes emit metrics; in training, a workload simulator generates realistic SLI values.

### Step 7 — Reward Computation
**Input:** Coverage vector, probe count, violation flags, margin values (from last known state of removed nodes), action taken.  
**Output:** Scalar reward `R_t`.  
**How:** Weighted reward formula combining observability, overhead, blind violation, and removal risk terms.

### Step 8 — Termination Check
**Input:** Cumulative blind violation count.  
**Output:** Continue or terminate episode.  
**How:** If cumulative `blind_violations > K` or `t = T`, terminate. Apply bonus or terminal penalty accordingly.

---

## 6. Worked Example

**Graph (excerpt from Hotel Reservation topology):**
```
frontend → search → geo → mongodb-geo
```

**Active SLOs (Hotel Reservation SLA):**

| SLO   | Metric                      | Threshold | Operator | Weight `w_k` |
|-------|-----------------------------|-----------|----------|--------------|
| SLO_1 | search_latency_p99          | 200ms     | <        | 0.5          |
| SLO_2 | geo_processing_time_p99     | 50ms      | <        | 0.3          |

**SLI mapping:**

```
SLO_1 needs: search_latency_p99
  → candidate_nodes(SLO_1) = { search }
     ↑ the search service reports its own end-to-end response time,
       which includes the downstream call to geo

SLO_2 needs: geo_processing_time_p99
  → candidate_nodes(SLO_2) = { geo }
```

Note: SLO_1 and SLO_2 are causally linked — a degraded `geo` is the root cause of a rising `search_latency_p99`. A probe on `search` detects the symptom; a probe on `geo` detects the cause.

**Scenario — Proactive placement:**

At `t = 5`, the agent observes (via its probe on `search`) that `search_latency_p99` is trending upward (+8ms/step, `margin(SLO_1, 5) = 0.12` and falling). The GNN propagates this stress signal to `geo` (a direct dependency of `search`). The agent infers `geo` is at risk and takes `add_probe(geo)` at `t = 6`.

At `t = 9`, `geo_processing_time_p99` spikes to 65ms — a violation of SLO_2 (threshold: 50ms). Because the probe was placed at `t = 6`, the violation is **detected**:

```
blind_violation(SLO_2, 9) = 0   ← covered and violation observed
```

**Reward at optimal placement `{ search, geo }`:**

```
R_t = w₁ · covered(SLO_1, t) + w₂ · covered(SLO_2, t)
      − λ · |P_t|  −  μ · 0  −  ρ · 0
    = 0.5 · 1  +  0.3 · 1  −  λ · 2
    = 0.8 − 2λ
```

The agent earns a positive reward as long as `0.8 > 2λ`, i.e., the combined SLO coverage value outweighs the cost of two probes.

**Counterexample — Blind violation:**

Suppose at `t = 6` the agent takes `no_op` instead. At `t = 9`, `geo_processing_time_p99` spikes to 65ms and `covered(SLO_2, 9) = 0` (no probe on `geo`). Then:

```
R_9 = w₁ · covered(SLO_1, 9)  +  w₂ · 0  −  λ · |P_9|  −  μ · 1  −  0
    = 0.5 · 1  +  0  −  λ · 1  −  μ · 1
    = 0.5 − λ − μ     (large negative reward for being blind to the root cause)
```

---

## 7. Design Decisions and Scope

### 7.1 Observability vs. SLO Health
Coverage is defined purely as **observability** — a probe in place means the SLA can be verified, regardless of whether the metric is currently within threshold. The threshold check is a separate concept (`violation`) that only matters in the context of `blind_violation`. This clean separation avoids conflating the agent's instrumentation decisions with the outcomes of those decisions.

### 7.2 Proactive vs. Reactive Placement
The formulation is designed for **proactive** placement: the agent must place probes before violations occur. The reward term `ρ · removal_risk` and the trend features in `ô_t` (margin, `Δmargin`) encourage the agent to act early. The `μ · blind_violation` term enforces the ultimate consequence of failing to do so.

### 7.3 POMDP vs. MDP
The problem is formally a POMDP because unprobed nodes emit no signals. A plain MDP assumes full state observability, which would require probing all nodes at all times — defeating the purpose of the agent. The POMDP framing captures the fundamental trade-off: probe more nodes to reduce uncertainty, but incur higher overhead cost.

### 7.4 SLI Selection
This work assumes **SLIs are predefined per SLA**. The agent selects where to place probes, not which metrics to collect. Future work could extend the agent to jointly select SLIs and probe placement, which would require a larger action space and a richer reward signal.

### 7.5 Graph Dynamism

The graph `G^(e)` is **static within each episode** but varies across episodes, modeling infrequent operational events grounded in the **Hotel Reservation** topology from DeathStarBench. Nodes are partitioned into two sets:

- **Core (10 nodes, never removed):** The 8 backbone business-logic services (`frontend`, `search`, `geo`, `rate`, `profile`, `recommendation`, `reservation`, `user`) plus `consul` and `jaeger`. These form the stable, always-present backbone.
- **Non-core (14 nodes, independently perturbable):** The 2 extended services (`review`, `attractions`) and all 12 data-store instances (8 MongoDB + 4 Memcached). Each may independently appear or be absent at each episode boundary.

This gives a maximum of **`2^14 = 16,384` distinct episode graphs**, providing sufficient structural diversity for GNN generalization. Cross-episode variation models realistic operational events: data-store replica eviction (MongoDB/Memcached pod failure), extended-service rolling restart, or circuit breaker activation. This design choice:

- Keeps the per-episode POMDP formulation clean and tractable
- Grounds the agent's training environment in the exact topology of a real, documented microservice application
- Forces the agent to handle data-store availability variability via the GNN encoder, without retraining
- Reflects the dominant real-world failure class: storage-layer failures rather than application-layer failures

Intra-episode topology changes (e.g., pod crashes mid-episode) are considered out of scope for the current formulation and remain an open extension.

### 7.6 Training Environment
The agent is trained in a **simulated environment** that:
- Models realistic traffic distributions (diurnal patterns, bursty loads)
- Injects failure scenarios (latency spikes, error bursts, cascading failures)
- Generates SLI values consistent with the service graph topology and upstream-downstream dependencies

---

## 8. Summary Table

| Step | What | How | Tool |
|------|------|-----|------|
| 1 — SLA definition | Formal constraint + weights | Manual input | Given |
| 2 — SLI mapping | Node → metrics table | Static lookup | OpenTelemetry / manual |
| 3 — Observation | Masked SLI window + margin + trend | GNN + feature vector | GNN + masking |
| 4 — Agent action | Add / Remove / No-op | POMDP policy | PPO |
| 5 — Transition | Workload evolution | Simulator step | Workload simulator |
| 6 — SLI collection | Metric values for probed nodes | Probe emission | Workload simulator |
| 7 — Reward | Coverage − overhead − blind violations − removal risk | Weighted formula | Reward function |
| 8 — Termination | Blind violation budget check | `Σ blind_violations > K` | Episode logic |

---

## 9. Open Research Directions

- **Joint SLI and probe placement selection** — extend the agent to also decide which metrics to collect, not just where to probe
- **Dynamic graph adaptation** — agent reacts to topology changes (pod scaling, service failures) in real time
- **Hierarchical SLAs** — some SLAs depend on the satisfaction of sub-SLAs across multiple nodes
- **Multi-agent placement** — cooperative agents for very large graphs exceeding GNN scalability limits
- **Transfer learning** — pre-trained placement policies that generalize across diverse graph topologies without retraining
- **Constrained RL formulation** — explicitly model the violation budget `K` as a Lagrangian constraint rather than a terminal condition, enabling tighter safety guarantees during training
