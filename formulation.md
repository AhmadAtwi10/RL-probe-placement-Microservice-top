# RL-Based Probe Placement for SLA Verification in Microservice Architectures

## Formal Problem Formulation

---

## 1. Background and Motivation

In microservice architectures, ensuring service quality requires monitoring distributed components against defined contracts. This work formalizes the problem of **optimal probe placement** — determining which nodes to instrument in a service graph so that all Service Level Agreement (SLA) violations can be **detected before or as they occur**.

The three key concepts that underpin this formulation are:

- **SLI (Service Level Indicator):** A quantitative metric that measures the performance or reliability of a service, such as `payment_success_rate` or `p99_latency`.
- **SLO (Service Level Objective):** A target value or threshold that an SLI is expected to satisfy, usually over a defined time window, e.g., `payment_success_rate > 99.95%`.
- **SLA (Service Level Agreement):** A formal agreement or contract that defines the service guarantees, typically composed of one or more SLOs and possibly associated penalties or obligations.

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

> **Important — Detection vs. Prevention**
>
> The agent does **not** control whether violations occur. Violations are determined by the workload: traffic patterns, failure injection, diurnal load, and cascading degradation. The agent controls only *where probes are placed*. Its goal is to ensure that when a violation occurs, a probe is already watching that node — so the violation is **detected** rather than **missed**.
>
> A violation that is *detected* (probe in place) is a success of the monitoring system. A violation that goes *undetected* (no probe) is the primary failure mode. The agent is **never rewarded for preventing violations**, only for observing them.

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
  search, geo, rate, profile, recommendation,
  reservation, user, review, attractions

Layer 2 — Data stores (12 nodes):
  mongodb-geo, mongodb-profile, mongodb-rate, mongodb-recommendation,
  mongodb-reservation, mongodb-user, mongodb-review, mongodb-attractions
  memcached-rate, memcached-profile, memcached-reserve, memcached-review

Layer 3 — Infrastructure (2 nodes):
  consul (service registry), jaeger (distributed tracing)
```

The **core call graph** (edges between business-logic services) is:

```
frontend → search, profile, recommendation, reservation, user, review, attractions
search   → geo, rate
```

The **infrastructure edges** (from `depends_on` in docker-compose) are:

```
geo           → mongodb-geo
rate          → mongodb-rate, memcached-rate
profile       → mongodb-profile, memcached-profile
recommendation → mongodb-recommendation
reservation   → mongodb-reservation, memcached-reserve
user          → mongodb-user
review        → mongodb-review, memcached-review
attractions   → mongodb-attractions
all services  → consul
```

Note that `review` and `attractions` are classified as **non-core** in the operational sense: they are subject to perturbation across episodes (pod failure, rolling restart), unlike the eight stable backbone services.

At the start of a new episode, a graph is derived from this base topology via a lightweight **operational perturbation**:

```
G^(e+1) ~ G  — independent Bernoulli perturbation over 14 non-core nodes:

  Non-core business-logic (2):  review, attractions
  Non-core data stores    (12): all mongodb-*, all memcached-*

  For each non-core node v:
    with prob p_fail: v is absent from G^(e+1)  (pod failure, eviction)
    with prob p_rec:  a previously absent v returns  (pod restart, recovery)
```

The **8 backbone services** and infrastructure nodes (`consul`, `jaeger`) are **never removed**. This gives a maximum of `2^14 = 16,384` distinct episode graph configurations, providing sufficient structural diversity for GNN generalization training. Within a single episode the agent operates on a fully known, fixed graph. Across episodes, the GNN encoder recomputes node embeddings from the updated graph, enabling the agent to **generalize across operational configurations without retraining**.

Each node `v ∈ V^(e)` exposes a set of SLIs when probed:

```
SLI(v) = { m1, m2, ..., mn }   — metrics node v can emit if instrumented
```

SLIs fall into two semantic categories:

- **Node-intrinsic SLIs** — properties of the service itself: `cpu_utilization`, `memory_utilization`, `request_queue_depth`, `disk_io_utilization`, `connection_pool_utilization`.
- **Call-derived SLIs** — reported by the *receiving* service: `p99_latency`, `error_rate`, `failure_rate`. This attribution follows standard APM tooling (Jaeger spans, Prometheus `http_request_duration_seconds`).

> **Note on APM tooling:** APM (Application Performance Monitoring) refers to tools such as Prometheus, Jaeger, Datadog, and OpenTelemetry that collect, store, and visualise performance metrics and traces from running services. The SLI values our simulator generates synthetically are what these tools would emit in a real deployment. The attribution convention (metrics reported by the receiving service, not the caller) follows how Jaeger spans and Prometheus histograms are instrumented in practice.

Two additional SLIs are derived from **runtime call graph traces** and incorporated as node-level features:

```
incoming_call_rate(v, t) — inbound gRPC/HTTP calls per second at node v
outgoing_call_rate(v, t) — outbound calls per second initiated by node v
```

These features allow the GNN to distinguish structurally important nodes (high `degree_in`) from currently hot nodes (high `incoming_call_rate`). Two nodes may share the same structural in-degree but receive very different call volumes at runtime. In our implementation these rates are generated synthetically per-node (diurnal pattern + noise) rather than summed from upstream callers, approximating what a real APM tool like Jaeger would report.

**Concrete SLI mapping for the Hotel Reservation application (DeathStarBench):**

| Node | Type | Core | SLI(v) |
|---|---|---|---|
| `frontend` | gateway | yes | request_rate, p99_latency, p50_latency, error_rate, active_connections, incoming_call_rate, outgoing_call_rate |
| `search` | search | yes | search_latency_p99, search_latency_p50, result_count, incoming_call_rate, outgoing_call_rate |
| `geo` | business_logic | yes | processing_time_p99, failure_rate, cpu_utilization, incoming_call_rate, outgoing_call_rate |
| `rate` | business_logic | yes | processing_time_p99, failure_rate, request_queue_depth, incoming_call_rate, outgoing_call_rate |
| `profile` | business_logic | yes | processing_time_p99, failure_rate, cpu_utilization, incoming_call_rate, outgoing_call_rate |
| `recommendation` | business_logic | yes | recommendation_latency_p99, failure_rate, incoming_call_rate, outgoing_call_rate |
| `reservation` | business_logic | yes | processing_time_p99, failure_rate, request_queue_depth, incoming_call_rate, outgoing_call_rate |
| `user` | business_logic | yes | processing_time_p99, failure_rate, cpu_utilization, incoming_call_rate, outgoing_call_rate |
| `review` | business_logic | no | processing_time_p99, failure_rate, cpu_utilization, incoming_call_rate, outgoing_call_rate |
| `attractions` | business_logic | no | processing_time_p99, failure_rate, cpu_utilization, incoming_call_rate, outgoing_call_rate |
| `mongodb-*` | data_store | no | query_latency_p99, write_latency_p99, connection_pool_utilization, disk_io_utilization |
| `memcached-*` | cache | no | hit_rate, miss_rate, eviction_rate, memory_utilization |
| `consul` | registry | yes | request_rate, p99_latency, error_rate |
| `jaeger` | tracing | yes | trace_ingestion_rate, trace_drop_rate |

> **Core vs. non-core:** The 8 backbone business-logic services (`frontend` through `user`) and infrastructure nodes (`consul`, `jaeger`) are core — never removed across episodes. All 12 data-store instances (`mongodb-*`, `memcached-*`) and the 2 extended services (`review`, `attractions`) are non-core and subject to independent Bernoulli perturbation at each episode boundary, giving up to `2^14 = 16,384` distinct episode graph configurations.

> **Infrastructure nodes:** `consul` and `jaeger` remain in `G^(e)` as graph nodes — their structural position informs GNN message passing and captures real hub-and-spoke patterns in the topology. However, both are marked `is_probeable(v) = 0`: the agent cannot place or remove probes on them, and no SLOs are defined over their internal metrics. Monitoring infrastructure-level tools at the application level would create a circular dependency. (Note: Jaeger is a distributed tracing system released as open source by Uber Technologies, used for monitoring and troubleshooting microservices-based distributed systems.)

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
W = { w_k in R+ | k in K }   — importance weights assigned by the system operator
```

**SLA:** The application is governed by a **single** SLA. The SLA is satisfied at time `t` if and only if **all** constituent SLOs are simultaneously satisfied:

```
satisfied(t) = 1   iff   for all k in K,  covered(k, t) = 1  AND  not violation(k, t)
```

**Example — Hotel Reservation SLA (DeathStarBench):**

```
SLA = "Hotel Reservation Service Agreement"
  ├── SLO_0 = (search_latency_p99,               200ms, ≤)   w_0 = 1.0
  ├── SLO_1 = (geo_processing_time_p99,            50ms, ≤)   w_1 = 1.0
  ├── SLO_2 = (reservation_processing_time_p99,   300ms, ≤)   w_2 = 1.0
  ├── SLO_3 = (reservation_failure_rate,           0.5%, ≤)   w_3 = 1.0
  ├── SLO_4 = (mongodb_rate_query_latency_p99,     10ms, ≤)   w_4 = 1.0
  └── SLO_5 = (memcached_rate_hit_rate,              90%, ≥)   w_5 = 1.0
```

Note that `SLO_0` and `SLO_1` are causally linked: a degraded `geo` service causes `search_latency_p99` to spike via the synchronous `search → geo` gRPC dependency. The agent must monitor **both** nodes to distinguish root cause from symptom.

> **Note:** The probe placement problem operates at the **SLO level** — each SLO maps to one metric and requires observing a specific node. The SLA is the terminal criterion: if the agent survived T timesteps while detecting every violation that occurred (zero blind violations exceeding budget K), a terminal bonus is applied. Throughout the formulation, index `k` refers to an individual SLO.
>
> **Implementation note on metric names:** In the implementation (`slo_config.py`), SLO metric names match exactly the SLI keys emitted by nodes — e.g., `processing_time_p99` (not `geo_processing_time_p99`). The `SLO.nodes` field carries node identity; the metric name is used as a lookup key into the SLI snapshot at each timestep.

### 2.3 SLI-to-Node Mapping

For each SLO `k`, the set of candidate nodes — those capable of providing the required SLI — is:

```
candidate_nodes(k) = { v in V^(e) | metric_k in SLI(v) }
```

This mapping is a **static input** to the system, defined by service metadata (e.g., OpenTelemetry instrumentation declarations). The agent does not learn or select which SLIs to collect — it only decides where to place probes.

### 2.4 Coverage and Violation (Separated Concepts)

Two distinct concepts must be kept separate:

**Coverage (observability):** Can the agent observe SLO `k`'s metric at time `t`?

```
covered(k, t) = 1   iff   ∃ v ∈ V^(e) : P_t[v] = 1  AND  metric_k ∈ SLI(v)
```

Coverage is purely about whether a probe is in place — it says nothing about the metric's current value.

**Violation (threshold breach):** Is SLO `k`'s threshold breached at time `t`?

```
violation(k, t) = 1   iff   SLI_k(t)  violates  (threshold_k, operator_k)
```

A violation can occur whether or not a probe is in place — the difference is whether the agent can see it. Violations are caused by the workload and are **outside the agent's control**.

**Blind violation (the dangerous case):** A violation that the agent cannot detect:

```
blind_violation(k, t) = 1   iff   violation(k, t) = 1  AND  covered(k, t) = 0
```

The agent's primary goal is to eliminate blind violations by ensuring probes are placed proactively. A *covered violation* (probe in place, violation detected) is a success. A *blind violation* (no probe, violation missed) is the primary failure mode.

### 2.5 SLO Margin (Risk Signal)

For each SLO `k` with a currently probed node, the **normalized margin** from the threshold measures how close the system is to a violation:

```
margin(k, t) = (threshold_k − SLI_k(t)) / |threshold_k|    [for operator_k = ≤]
margin(k, t) = (SLI_k(t) − threshold_k) / |threshold_k|    [for operator_k = ≥]
```

| `margin(k, t)` | Interpretation |
|---|---|
| ≈ 1 | Safe — far from threshold |
| ≈ 0 | Danger zone — approaching threshold |
| < 0 | SLO violated |

The margin is **only observable for probed nodes**. For unprobed nodes, `margin(k, t)` is unknown — this is the core source of partial observability in the problem.

---

## 3. Formal Optimization Problem

### 3.1 Static Idealized Formulation (Theoretical Benchmark)

When the graph is static and the state is fully observable, the probe placement problem reduces to an instance of **weighted set cover**:

**Decision variable:** Let `S ⊆ V^(e)` be the set of probed nodes.

**Objective:** Find the minimum probe set `S*` that covers all SLOs:

```
S* = argmin |S|
     subject to:  ∀ k ∈ K,  covered(k, S) = 1
```

This is NP-hard in general and serves as a **lower bound** on probe count.

### 3.2 Multi-SLO Extension (Weighted Coverage)

When SLOs have different importances:

```
S* = argmin |S|
     subject to:  Σ_k  w_k · covered(k, S)  ≥  C_min
```

> **Note:** This static formulation assumes a fixed workload and full observability. The POMDP in Section 4 addresses the realistic problem; the static formulation provides a useful baseline for evaluation.

---

## 4. POMDP Formulation

The probe placement problem under dynamic workloads and partial observability is modeled as a **Partially Observable Markov Decision Process (POMDP)**:

```
POMDP = (S, A, O, T, Ω, R, γ)
```

The distinction from a plain MDP is critical: the agent cannot observe the full system state. Any node without a probe emits no signals — its SLI values, trends, and current margin are completely hidden from the agent.

### 4.1 True State Space `S`

```
s_t = (G^(e), X_t, P_t)
```

- `G^(e)` — service graph, fixed throughout the episode
- `X_t` — true SLI values for **all** nodes (including unprobed — completely hidden from agent)
- `P_t` — probe placement vector: `P_t[v] = 1` if node `v` is probed

### 4.2 Observation Space `O`

```
o_t = (G^(e), P_t, ô_t, H_t)
```

- `G^(e)` — episode graph, fully known; fixed within the episode
- `P_t` — current probe placement vector
- `ô_t` — **masked observation vector**:

```
ô_t(v, k) = [ SLI_k(t-N+1), ..., SLI_k(t), margin(k,t), Δmargin(k,t) ]
                                                        if P_t[v] = 1
           = [ masked (zeros) ]                         if P_t[v] = 0
```

The rolling window of `N` timesteps plus margin and Δmargin enables the agent to detect deteriorating trends before a threshold is breached.

- `H_t` — SLO health vector: `H_t[k] = 1` if SLO `k` is currently covered **and** not violated, `0` otherwise.

#### 4.2.1 Node Feature Vector

Each node `v ∈ V^(e)` is described by:

```
x_v(t) = [ x_v^static  |  x_v^dynamic(t) ]
```

**Static features** (fixed for the episode):

```
x_v^static = [
  node_type(v)         : one-hot over service role
  is_probeable(v)      : 1 if agent can probe v; 0 for consul, jaeger
  degree_in(v)         : incoming dependency edges in G^(e)
  degree_out(v)        : outgoing dependency edges in G^(e)
  slo_coverage_mask(v) : binary vector over K — which SLOs node v can cover
  slo_importance(v)    : Σ_{k: metric_k ∈ SLI(v)} w_k
]
```

**Dynamic features** (updated every timestep):

```
x_v^dynamic(t) = [
  P_t[v]                   : 1 if probed, 0 otherwise
  ô_t(v)                   : SLI window + margin + Δmargin per SLO k
                             (all zeros if P_t[v] = 0)
  staleness(v, k)          : timesteps since last observation of SLO k
                             (0 when probed; increments when unprobed;
                              resets to 0 on re-probe) — one counter per SLO k
  incoming_call_rate(v, t) : inbound calls per second
  outgoing_call_rate(v, t) : outbound calls per second
]
```

> **On staleness:** When a node is unprobed, its SLI history buffer is frozen internally but its `ô_t` slots are masked to zero in the observation. The staleness counter tells the agent how long since the last observation of each SLO on that node, without leaking stale SLI values. This allows the policy to reason about data freshness separately from the masked signal.

> **On observation size:** `ô_t(v)` is indexed over the **fixed SLO set K**, not over raw SLI names:
>
> ```
> ô_t(v) = [ ô_t(v, SLO_0) | ô_t(v, SLO_1) | ... | ô_t(v, SLO_{|K|-1}) ]
>            size: |K| × (N + 2)   regardless of |SLI(v)|
> ```
>
> If `v` does not cover SLO `k`, `ô_t(v, k) = 0` always. This keeps the GNN input **fixed-size per node** across all node types.

Call-rate features are **always observable** regardless of probe placement — they are derived from infrastructure-level traces (Jaeger, Envoy), not application probes.

**Concrete dimensions (implementation, N=4, |K|=6):**

```
x_v^static   : 17 values
  (7 type_one_hot + 1 probeable + 1 deg_in + 1 deg_out
   + 6 slo_mask + 1 slo_importance)
x_v^dynamic  : 45 values
  (1 probe_flag + 36 slo_window (6×(N+2) = 6×6 with N=4) + 6 staleness + 2 call_rates)
x_v(t) total : 62 values
```

**Encoding pipeline:**

```
Step 1 — Build node feature vectors (each timestep t):
  x_v(t) = [ x_v^static | x_v^dynamic(t) ]   ∀ v ∈ V^(e)

Step 2 — GNN message passing (L layers):
  h_v^(0) = W_input · x_v(t)                  ← input projection
  h_v^(l) = UPDATE( h_v^(l-1),
             AGGREGATE({ h_u^(l-1) | u ∈ N_in(v) }) )
  h_v     = h_v^(L)

Step 3 — Policy input:
  graph_embedding = READOUT({ h_v | v ∈ V^(e) })   ← masked mean pooling
  policy_input    = concat(graph_embedding, H_t)
```

A **GNN** is used because: (i) it handles variable-size graphs across episodes; (ii) message passing propagates stress signals from probed nodes to unprobed neighbors; (iii) structural features are naturally encoded per-node.

### 4.3 Action Space `A`

```
A = { add_probe(v)    | v ∈ V^(e), is_probeable(v)=1, P_t[v]=0 }
  ∪ { remove_probe(v) | v ∈ V^(e), is_probeable(v)=1, P_t[v]=1 }
  ∪ { no_op }
```

Let `V_p^(e) = { v ∈ V^(e) | is_probeable(v) = 1 }` denote the probeable nodes. This is a **discrete action space** of size `2|V_p^(e)| + 1`.

- `add_probe(v)`: sets `P_{t+1}[v] = 1`; node `v`'s SLIs become observable from `t+1` onward.
- `remove_probe(v)`: sets `P_{t+1}[v] = 0`; node `v`'s SLIs become hidden from `t+1` onward.
- `no_op`: probe placement unchanged.

### 4.4 Transition Function `T`

```
s_{t+1} ~ T(s_t, a_t)
```

Probe placement updates deterministically. SLI values `X_{t+1}` are drawn from a workload distribution (traffic spikes, failure injections, diurnal patterns) modeled by a **workload simulator** during training.

### 4.5 Observation Function `Ω`

```
o_t ~ Ω(s_t, P_t)
```

For each node `v`, if `P_t[v] = 1`, the true SLI value `X_t(v)` is revealed; otherwise it is masked. The observation function is deterministic given the true state.

### 4.6 Reward Function `R`

```
R_t =   Σ_k  w_k · covered(k, t)               ← (1) observability reward
      − λ  · |P_t|                               ← (2) probe overhead penalty
      − μ  · Σ_k  w_k · blind_violation(k, t)    ← (3) blind violation penalty
      − ρ  · removal_risk(a_t, t)                ← (4) risk-weighted removal penalty
```

**Term (1) — Observability reward:** Rewards coverage of SLOs weighted by importance `w_k`.

**Term (2) — Probe overhead penalty:** Encourages minimizing instrumentation cost.

**Term (3) — Blind violation penalty:** The heaviest penalty. Triggered when an SLO threshold is breached on an unprobed node. The penalty is weighted by `w_k` so that missing a high-priority SLO incurs a proportionally larger penalty, consistent with term (1) which also uses `w_k` weights.

**Term (4) — Risk-weighted removal penalty:** Applied when `a_t = remove_probe(v)`:

```
removal_risk(v, t) = Σ_{k: metric_k ∈ SLI(v)}  w_k · max(0, 1 − margin(k, t))
```

**Hyperparameter trade-offs:**

| Parameter | Role | Effect if increased |
|---|---|---|
| `w_k` | SLO importance weight | Agent prioritizes covering SLO `k` |
| `λ` | Overhead sensitivity | Agent uses fewer probes overall |
| `μ` | Blind violation cost | Agent avoids being unprobed during violations |
| `ρ` | Removal risk sensitivity | Agent hesitates to remove near-threshold probes |

**Default values:** `λ=0.05`, `μ=2.0`, `ρ=0.5`, terminal_bonus=`5.0`, terminal_penalty=`-10.0`. In the current implementation all SLO importance weights are uniform, `w_k = 1.0` for every `k` (see `slo_config.py`); the weighting machinery is retained so an operator can assign non-uniform weights without code changes.

**Temporal interplay between ρ and μ:**
- `ρ` fires **immediately at the moment of removal** — provides early shaping signal.
- `μ` fires **when a violation occurs** without a probe — may be delayed many timesteps after the bad removal decision.

Together they close the temporal credit assignment gap.

### 4.7 Episode Structure and Termination

```
t* = min( T,  min{ t : Σ_{τ ≤ t} Σ_k blind_violation(k, τ) > K } )
```

- **Normal termination** at `t = T`: the agent survived the full workload window. This means every SLO violation that occurred during the episode was detected by a probe in place. The terminal bonus rewards **complete observability**, not the absence of violations — violations are caused by the workload and are outside the agent's control. A terminal bonus is awarded.
- **Failure termination** when cumulative blind violations exceed `K`: the agent failed to observe more than `K` violation events. A large terminal penalty is applied.

`K` is a **safety budget** and a curriculum parameter: training begins with a generous `K` and reduces it as the agent improves (`K` stages: `[50, 20, 10, 5, 2]`).

### 4.8 Optimization Objective

```
π* = argmax E[ Σ_{t=0}^{t*} γ^t · R_t ]
     subject to:  E[ Σ_t Σ_k blind_violation(k, t) ] ≤ K
```

where `γ ∈ (0, 1]` is the discount factor. The static formulation in Section 3 serves as a **lower bound** on probe count and a benchmark for evaluating the agent's efficiency.

### 4.9 Algorithm

**Proximal Policy Optimization (PPO)** is the recommended algorithm because:
- It handles discrete, combinatorial action spaces robustly
- It is more stable than DQN for large graphs
- It directly learns a stochastic policy `π(a | o_t)`
- It supports the continuous online operation structure of this formulation

**DQN** is a simpler alternative for small graphs where `|V|` is small and the action space is manageable.

---

## 5. End-to-End Workflow

**Step 1 — SLA Definition:** Manual input of SLO triples and importance weights `w_k`.

**Step 2 — SLI Mapping:** Static lookup of `SLI(v)` and `candidate_nodes(k)` from OpenTelemetry declarations.

**Step 3 — Observation Construction:** GNN encodes `G^(e)`; windowed SLI values, margins, and staleness counters concatenated per node; `H_t` appended at graph level.

**Step 4 — Agent Action:** Policy network `π(a | o_t)` selects action via PPO.

**Step 5 — Environment Transition:** Probe placement updated deterministically; SLI values evolved by workload simulator.

**Step 6 — SLI Collection:** Probed nodes emit metrics; workload simulator generates values for training.

**Step 7 — Reward Computation:** Four-term weighted reward formula applied.

**Step 8 — Termination Check:** If cumulative `blind_violations > K` or `t = T`, terminate and apply bonus or penalty.

---

## 6. Worked Example

**Graph excerpt:**

```
frontend → search → geo → mongodb-geo
```

**Active SLOs:**

| SLO | Metric | Threshold | Operator | Weight `w_k` |
|---|---|---|---|---|
| `SLO_1` | `search_latency_p99` | 200ms | ≤ | 1.0 |
| `SLO_2` | `geo_processing_time_p99` | 50ms | ≤ | 1.0 |

At `t=5`: agent observes rising `search_latency_p99` (+8ms/step, margin = 0.12 falling). GNN propagates stress to `geo`. Agent takes `add_probe(geo)` at `t=6`.

At `t=9`: `geo_processing_time_p99` = 65ms > 50ms. Probe in place since `t=6`:

```
blind_violation(SLO_2, 9) = 0   ← covered and violation observed
```

**Reward at optimal placement `{search, geo}`:**

```
R_t = 1·1 + 1·1 − λ·2 = 2 − 2λ   (positive reward)
```

**Counterexample — Blind violation (no_op at t=6):**

```
R_9 = w_1·covered(SLO_1,9) + 0 − λ·|P_9| − μ·w_2·blind_violation(SLO_2,9)
    = 1·1 + 0 − λ·1 − 2.0·1·1
    = 1 − 0.05 − 2.0 = −1.05   (large negative reward)
```

---

## 7. Design Decisions and Scope

### 7.1 Observability vs. SLO Health

Coverage is defined purely as **observability** — a probe in place means the SLA can be verified, regardless of the metric's current value. The threshold check is a separate concept (`violation`) that only matters in the context of `blind_violation`.

### 7.2 Proactive vs. Reactive Placement

The formulation is designed for **proactive** placement. The reward term `ρ · removal_risk` and trend features (margin, Δmargin) encourage the agent to act early. The `μ · blind_violation` term enforces the ultimate consequence of failing to do so.

### 7.3 POMDP vs. MDP

Unprobed nodes emit no signals, making this formally a POMDP. The POMDP framing captures the fundamental trade-off: probe more nodes to reduce uncertainty, but incur higher overhead cost.

### 7.4 SLI Selection

This work assumes **SLIs are predefined per SLA**. The agent selects where to place probes, not which metrics to collect.

### 7.5 Graph Dynamism

`G^(e)` is static within each episode but varies across episodes. Core nodes (10) are never removed. Non-core nodes (14) are independently perturbable, giving up to `2^14` distinct configurations. Intra-episode topology changes are out of scope.

### 7.6 Training Environment

The agent is trained in a simulated environment with diurnal traffic patterns, failure injection (latency spikes, error bursts, cascading failures), and SLI generation consistent with the service graph topology.

---

## 8. Summary Table

| Step | What | How | Tool |
|---|---|---|---|
| 1 — SLA definition | Formal constraint + weights | Manual input | Given |
| 2 — SLI mapping | Node → metrics table | Static lookup | OpenTelemetry |
| 3 — Observation | Masked SLI window + margin + trend | GNN + feature vector | GNN + masking |
| 4 — Agent action | Add / Remove / No-op | POMDP policy | PPO |
| 5 — Transition | Workload evolution | Simulator step | Workload simulator |
| 6 — SLI collection | Metric values for probed nodes | Probe emission | Workload simulator |
| 7 — Reward | Coverage − overhead − blind − risk | Weighted formula | Reward function |
| 8 — Termination | Blind violation budget check | `Σ blind > K` | Episode logic |

---

## 9. Neural Network Architecture

### 9.1 GNN Encoder — How It Works

**Why a GNN?**

The service graph `G^(e)` has a variable number of nodes across episodes (`|V^(e)|` between 10 and 24). A standard MLP or CNN requires a fixed-size input. A GNN solves both problems:

- Its weights `W_msg` and `W_upd` are **shared across all nodes** — the same weight matrices process every node regardless of graph size.
- **Stress signal propagation:** if `search` is probed and shows rising latency, after one message-passing step `geo` (its direct dependency) will have received that signal in its aggregated messages — even though `geo` is unprobed and its SLI is hidden.

**Input Preparation**

Before the GNN runs, the environment produces padded fixed-size tensors:

```
node_features (MAX_NODES=24, 62)  — padded rows are zeros
edge_index    (2, MAX_EDGES=200)  — padded columns are zeros
node_mask     (24,) bool          — True for real nodes
edge_mask     (200,) bool         — True for real edges
```

**Input Projection (Layer 0)**

```
h_v^(0) = ReLU( W_input · x_v )      W_input: (62, hidden_dim=128)
```

**Message Passing (Layers 1..L, L=3)**

Each GNN layer performs three operations:

*Step 1 — Message:*

```
msg_{u→v}^(l) = W_msg^(l) · h_u^(l-1)      W_msg: (128, 128)
```

Messages from padded edges are zeroed via `edge_mask`.

*Step 2 — Aggregate (mean):*

```
m_v^(l) = (1/deg_in(v)) · Σ_{u ∈ N_in(v)} msg_{u→v}^(l)
```

Mean (not sum) keeps magnitude stable across variable-degree nodes.

*Step 3 — Update:*

```
combined_v = concat( h_v^(l-1), m_v^(l) )       shape: (256,)
h_v^(l)   = ReLU( LayerNorm( W_upd · combined_v ) )
h_v^(l)   = Dropout( h_v^(l) ) * node_mask[v]
```

**Why L=3?** The deepest path in the Hotel Reservation call graph is `frontend → search → geo → mongodb-geo` (3 hops). L=3 ensures every node's embedding is informed by the entire reachable subgraph.

**Stress signal propagation example:**

```
t=5: search is probed, SLI window shows rising latency (Δmargin=−0.08)

GNN Layer 1:
  msg_{search→geo} = W_msg · h_search^(0)   ← stress encoded in message
  m_geo^(1) = msg_{search→geo}              ← geo receives signal
  h_geo^(1) = f(concat(h_geo^(0), m_geo^(1)))

After Layer 1: h_geo reflects search's stress,
even though geo is unprobed and its SLI window = 0 (masked).
```

**Global Readout**

After L layers, a global graph embedding is computed via **masked mean pooling**:

```
z = (1/|V^(e)|) · Σ_{v ∈ V^(e)} h_v^(L)      shape: (128,)
```

Mean pooling is appropriate for the **critic** (overall state quality — a graph-level property). It is **not** used for action scoring because it destroys node identity — see Section 9.2.

**Variable Graph Size — No Retraining Needed**

Because `W_msg`, `W_upd`, and `W_input` are shared across all nodes, the same trained GNN works on any subgraph of `G^(e)`. Absent nodes have their row zeroed and their mask bit set to False, so they contribute nothing to computations.

### 9.2 Policy Network — How It Works

**Overview**

The policy network is an **actor-critic** architecture on top of the GNN encoder. It produces action logits `π(a | o_t)` and state value `V(s)`.

**Full forward pass:**

```
Step 1 — GNN encoder:
  node_emb (24, 128),  graph_emb (128,)  =  GNNEncoder(...)

Step 2 — Global context:
  global_ctx (134,)  =  concat(graph_emb, slo_health)

Step 3 — Gather probeable node embeddings:
  h_prob (|V_p|, 128)  =  node_emb[probeable_indices]

Step 4 — Actor heads:
  logits_add    (|V_p|,)  =  MLP_add(h_prob)        ← per-node, uses h_v
  logits_remove (|V_p|,)  =  MLP_remove(h_prob)     ← per-node, uses h_v
  logit_noop    (1,)      =  MLP_noop(global_ctx)   ← global context

Step 5 — Assemble and mask:
  action_logits (45,)  =  [logit_noop|logits_add|logits_remove|−inf padding]
  action_logits        =  masked_fill(~action_mask, −inf)

Step 6 — Probabilities:
  action_probs  =  softmax(action_logits)
  entropy       =  −Σ p · log(p)   over valid actions

Step 7 — Critic:
  state_value   =  MLP_value(global_ctx)   ← scalar V(s)
```

**Why Three Separate Actor Heads?**

- `actor_add` and `actor_remove` have **separate MLP weights**. Adding a probe is opportunity-seeking; removing one is risk-aware. These require different reasoning. The `ρ · removal_risk` term reinforces this asymmetry.
- `actor_noop` uses **global context** `concat(z, H_t)` because no-op is a graph-level decision ("is the current configuration already good enough?").
- The **critic** also uses `concat(z, H_t)`. Including `H_t` directly gives the critic an explicit signal about current SLO compliance, rather than requiring it to infer health from raw node embeddings.

**The Probeable Index Mapping**

Not every node in `node_emb` (24, 128) is probeable. The function `build_probeable_indices(episode_graph)` builds a mapping once per episode:

```
probeable_ordered = sorted(episode_graph.probeable_nodes)
indices = [ep.present_nodes.index(node) for node in probeable_ordered]
# e.g. [9, 0, 2, ...]   ← row positions in node_emb

h_prob = node_emb[probeable_indices]   # (22, 128) — probeable rows only
```

This mapping is recomputed at every `env.reset()` since present_nodes ordering changes across episodes.

**Action Mask Application**

```
action_mask[0]            = True   (no_op always valid)
action_mask[1..n_p]       = True   iff node not currently probed
action_mask[n_p+1..2n_p]  = True   iff node currently probed
action_mask[2n_p+1..44]   = False  (out-of-range for this episode's |V_p|)
```

After `masked_fill`, `softmax(−∞) = 0.0` exactly. The agent can never sample an invalid action.

---

## 10. Training Pipeline

### 10.1 Rollout Buffer

Pre-allocated tensors for `T_rollout` transitions:

| Field | Shape | Notes |
|---|---|---|
| `node_features` | (T, 24, 62) | Observation at time `t` |
| `edge_index` | (T, 2, 200) | COO, padded |
| `node_mask / edge_mask` | (T, 24) / (T, 200) | Boolean masks |
| `slo_health` | (T, 6) | `H_t` |
| `action_mask` | (T, 45) | Valid actions |
| `probeable_idx` | (T, 22) | Padded; valid entries = `n_p` |
| `actions / log_probs / values` | (T,) | Policy quantities |
| `rewards / dones` | (T,) | Environment feedback |
| `advantages / returns` | (T,) | Computed by GAE |

**GAE (Generalised Advantage Estimation):**

```
For t = T-1 down to 0:
  δ_t = r_t + γ·V(s_{t+1})·(1 − done_t) − V(s_t)
  A_t = δ_t + γ·λ_GAE·(1 − done_t)·A_{t+1}
  R_t = A_t + V(s_t)

Default: γ=0.99, λ_GAE=0.95
```

Episode boundaries (`done=True`) correctly zero the bootstrap term, preventing advantage propagation across episode boundaries.

### 10.2 PPO Update

```
ratio   = exp(log_prob_new − log_prob_old)
L_clip  = −mean(min(ratio·A, clip(ratio, 1±ε)·A))
L_value = 0.5·mean(max((V_new−R)², (V_clip−R)²))   [clipped]
L_total = L_clip + c1·L_value − c2·L_entropy
```

Default (`PPOConfig`): `ε=0.2`, `c1=0.05`, `c2=0.01`, `max_grad_norm=0.5`, `lr=1e-4`, `n_epochs=4`, `batch_size=64`. During training the curriculum scheduler drives the learning rate on a schedule from `3e-4` down to `1e-4` (see §10.3), so the effective starting lr is `3e-4`; the `PPOConfig` default of `1e-4` is the un-curricularized fallback.

Diagnostics per update: `approx_kl`, `clip_fraction`, `explained_variance`.

### 10.3 Curriculum Scheduler

| Quantity | Start | End | Schedule |
|---|---|---|---|
| `K` (blind violation budget) | 50 | 2 | Stage-based: `[50,20,10,5,2]` |
| `entropy_coef` | 0.05 | 0.005 | Linear across stages |
| `learning_rate` | 3e-4 | 1e-4 | Linear across total iterations |

Promotion: `promotion_window` consecutive iterations with `mean_reward ≥ promotion_threshold`. The base `CurriculumConfig` default is `promotion_threshold=0.5`, `promotion_window=20`, but the training presets override these — the `fast` and `full` presets use `promotion_threshold=30.0` with `promotion_window=20`, while `debug` effectively disables promotion (`promotion_threshold=1e9`, `promotion_window=1`, single stage `K=[100]`). On promotion, `K` tightens and the environment is reset.

### 10.4 Trainer Loop

```
for iteration in range(total_iterations):
    last_value = collect_rollout()    # fills buffer, tracks episode stats
    buffer.compute_gae(last_value)
    stats = ppo.update(buffer)
    buffer.reset()
    curriculum.step(mean_reward, iteration)
    logger.log(...)
    checkpointer.save(...)            # every save_interval
```

`EpisodeTracker` accumulates per-step reward, coverage flag, and blind flag, and aggregates over complete episodes in each rollout.

---

## 11. Evaluation

### 11.1 Per-Episode Metrics

| Metric | Definition |
|---|---|
| `total_reward` | Σ r_t |
| `survived` | True if truncated (reached T) |
| `cum_blind` | Total blind violation events |
| `coverage_rate` | Fraction of steps with ≥ 1 SLO covered |
| `blind_rate` | Fraction of steps with ≥ 1 blind violation |
| `mean_probe_count` | Mean `|P_t|` per step |
| `probe_efficiency` | `coverage_rate / max(mean_probe_count, 1)` |
| `weighted_coverage` | Mean `Σ_{k covered} w_k / Σ_k w_k` per step |
| `slo_coverage[k]` | Per-SLO fraction of steps covered |
| `slo_blind[k]` | Per-SLO fraction of steps with blind violation |

`probe_efficiency` is the key metric: it rewards covering SLOs cheaply. An agent covering 80% of SLOs with 2 probes scores higher than one needing 8.

`weighted_coverage` reflects SLO importance through the weights `w_k`. In the current configuration all `w_k = 1.0`, so every SLO contributes equally and `weighted_coverage` reduces to plain coverage. If the operator assigns non-uniform weights, missing a higher-weight SLO reduces the metric more than missing a lower-weight one.

### 11.2 Configuration Presets

```python
from src.config.train_config import get_config
from src.training.trainer import Trainer

cfg = get_config("full")        # or "debug" / "fast"
cfg.run_name = "my_experiment"
Trainer(cfg).train()
```

| Preset | Iterations | Rollout | Total Steps | Time (CPU, approx.) |
|---|---|---|---|---|
| `"debug"` | 5 | 32 | 160 | < 30s |
| `"fast"` | 100 | 512 | 51,200 | ~15–20 min |
| `"full"` | 10,000 | 512 | 5,120,000 | ~20–40h |

(Total Steps = Iterations × Rollout. CPU time estimates are approximate and scale roughly with total steps.)

---

## 12. Implementation Reference

### 12.1 File Structure

```
src/
  simulator/
    config/
      node_config.py          NodeMeta catalog, type one-hot, PROBEABLE_NODES
      slo_config.py           SLO catalog, margin/violation, NODE_TO_SLOS
    graph/
      topology.py             CALL_EDGES (9), INFRA_EDGES (32), ALL_NODES
      episode_graph.py        EpisodeGraph, EpisodeGraphBuilder
    workload/
      sli_generator.py        Diurnal SLI traces, PERCENT_METRICS
      failure_injector.py     Trapezoid failure pulses, BFS propagation
    env/
      reward.py               RewardConfig, RewardInput/Output, compute_reward()
      observation_builder.py  ObservationBuilder (62-dim), flatten_sli()
      probe_env.py            ProbeEnv (Gymnasium), ProbeEnvConfig
  models/
    gnn_encoder.py            GNNLayer, GNNEncoder, build_masks()
    policy_network.py         PolicyNetwork, PolicyOutput, build_probeable_indices()
  training/
    rollout_buffer.py         RolloutBuffer, GAE
    ppo.py                    PPO, PPOConfig, PPOStats
    trainer.py                Trainer, TrainerConfig, EpisodeTracker
    curriculum.py             CurriculumScheduler, CurriculumConfig
  evaluation/
    metrics.py                EpisodeResult, AggregateMetrics, aggregate()
    evaluate.py               evaluate(), evaluate_and_aggregate(), EvalConfig
  utils/
    logger.py                 Logger (JSON Lines + optional TensorBoard)
    checkpointing.py          Checkpointer (save/load)
  config/
    train_config.py           get_config(), describe(), presets
```

### 12.2 Key Implementation Decisions

**Weighted blind violation penalty:** Implementation uses `−μ · Σ_k w_k · blind_violation(k,t)` rather than the flat `−μ · count` in the formulation. Higher-priority SLOs penalise proportionally more when missed, consistent with term (1).

**Staleness counter:** 6 per-node counters added to the dynamic feature block. Masks SLO observation slots to zero when unprobed; counter tells agent data age without leaking stale values. Extends the feature vector from 56 to 62 dimensions.

**Per-node actor heads:** `actor_add` and `actor_remove` use `h_v` directly; `actor_noop` and critic use `concat(z, H_t)`. Preserves node identity for action scoring.

**SLI flattening boundary:** `SLIGenerator` produces `{node: {metric: array}}`. `ObservationBuilder` expects `{(node, metric): float}`. Conversion via `flatten_sli()` is the responsibility of `probe_env.py`.

**Fixed-size observation space:** Padded to (24, 62) node features, (2, 200) edge index, (45,) action mask for gym compatibility. Real nodes/edges identified by boolean masks.

### 12.3 Component Summary

| Component | File | Key API |
|---|---|---|
| Graph topology | `topology.py` | `CALL_EDGES`, `INFRA_EDGES` |
| Episode graph | `episode_graph.py` | `EpisodeGraph`, `EpisodeGraphBuilder` |
| Node metadata | `node_config.py` | `NodeMeta`, `NODE_CATALOG` |
| SLO definitions | `slo_config.py` | `SLO`, `SLO_CATALOG`, `NODE_TO_SLOS` |
| SLI simulation | `sli_generator.py` | `SLIGenerator`, `PERCENT_METRICS` |
| Failure injection | `failure_injector.py` | `FailureInjector`, `FailureEvent` |
| Reward | `reward.py` | `compute_reward()`, `terminal_reward()` |
| Observation | `observation_builder.py` | `ObservationBuilder`, `flatten_sli()` |
| Environment | `probe_env.py` | `ProbeEnv`, `ProbeEnvConfig` |
| GNN encoder | `gnn_encoder.py` | `GNNEncoder`, `build_masks()` |
| Policy | `policy_network.py` | `PolicyNetwork`, `build_probeable_indices()` |
| Buffer | `rollout_buffer.py` | `RolloutBuffer` (GAE included) |
| PPO | `ppo.py` | `PPO`, `PPOConfig`, `PPOStats` |
| Training loop | `trainer.py` | `Trainer`, `TrainerConfig` |
| Curriculum | `curriculum.py` | `CurriculumScheduler` |
| Logging | `logger.py` | `Logger` |
| Checkpointing | `checkpointing.py` | `Checkpointer` |
| Evaluation | `evaluate.py` | `evaluate()`, `evaluate_and_aggregate()` |
| Metrics | `metrics.py` | `EpisodeResult`, `AggregateMetrics` |
| Config entry point | `train_config.py` | `get_config()`, `describe()` |

---

## 13. Open Research Directions

- **Joint SLI and probe placement selection** — extend the agent to also decide which metrics to collect, not just where to probe
- **Dynamic graph adaptation** — agent reacts to topology changes (pod scaling, service failures) in real time
- **Hierarchical SLAs** — some SLAs depend on the satisfaction of sub-SLAs across multiple nodes
- **Multi-agent placement** — cooperative agents for very large graphs exceeding GNN scalability limits
- **Transfer learning** — pre-trained placement policies that generalize across diverse graph topologies without retraining
- **Constrained RL formulation** — explicitly model the violation budget `K` as a Lagrangian constraint rather than a terminal condition
- **Time-of-day encoding** — use the reserved `t` parameter in `ObservationBuilder.build()` for diurnal position features
- **Intra-episode topology changes** — extend to handle pod crashes mid-episode
